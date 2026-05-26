# Data architecture

## Data storage and serving

All data except some geographic data (e.g., satellite images) lives in
relational form in SQLite on the server.

However, to reduce latency, it is always served as compressed pre-computed JSON.
The mapping of relational tables to JSON files is specified in the detailed
domain descriptions.

For tabular data, the format is a JSON object with two fields:

    { "columns": ["row_id", "Data", "Compresa", ...],
      "rows": [[1, "2024-03-15", "Alpe", ...], ...] }

`columns` lists column names once; `rows` is an array of arrays containing pure
data. Every row's first element is `row_id`, used for updates (see below). The
client maps columns by name, so adding a column does not break cached data.

### JSON digest regeneration

Digests are regenerated lazily on read, not on write. This avoids wasted work
during batch data entry (where the inserting user's view is already kept current
via cache sync from the POST response).

A small table tracks staleness:

    digest_status(name TEXT PRIMARY KEY, stale BOOLEAN DEFAULT FALSE)

Write path: after a successful save, the view marks affected digests as stale:
`UPDATE digest_status SET stale = TRUE WHERE name IN (...)`. This is the only
write-path cost.

**The "affected digests" set is whatever the digest's generator reads,
transitively.** A write to table T must invalidate every digest D whose
`generate_D()` reads T — directly or via a join.  Forgetting one is a silent
bug: the on-disk file stays, the staleness flag stays False, and the next
read serves a stale digest *even after a page reload* (the cache miss still
hits the un-regenerated file).

Concrete examples from `apps/campionamenti`:

| Write                                  | Must invalidate                               |
|----------------------------------------|-----------------------------------------------|
| `Survey` create/delete                 | `surveys`, `grids` (N. rilevamenti)           |
| `SampleArea` create/delete             | `sample_areas`, `grids`, `surveys` (N. aree totali) |
| `Sample` create/delete + date edit     | `samples`, `surveys` (Data primo/ultimo, N. aree visitate), `sampled_trees_<survey>` |
| `TreeSample` create/delete             | `sampled_trees_<survey>`, `samples` (N. alberi) |
| `SampleGrid` rename                    | `grids`                                       |
| `Survey` rename                        | `surveys`                                     |

**Cross-domain invalidation** also occurs: a prelievi harvest write
linked to a `harvest_plan_item` marks `harvest_plan_items` stale (piano
di taglio calendar) and re-materializes `volume_actual_m3` on the item.
Each page's doc (`docs/page-*.md`) carries the authoritative
write→digest invalidation table for its domain.

When you add a new write endpoint, audit every `generate_*()` in
`apps/base/digests.py` and add `mark_stale()` for each digest that touches
the written table.  When you change a `generate_*()` to read a new table,
audit every write endpoint that touches that table.  Tests under
`TestDigestInvalidation` (`test/test_campionamenti_views.py`,
`test/test_prelievi_views.py`, `test/test_piano_di_taglio_views.py`)
lock the contract: each test performs a write and re-reads the affected
digest via the public endpoint, asserting that materialized columns
reflect the change.  Add equivalent tests for any new write/digest pair.

Read path (conditional GET for a digest):
1. Check stale flag for the requested digest (one PK lookup).
2. If not stale: normal If-Modified-Since check against file mtime, return 304
   or 200.
3. If stale: regenerate the digest, then serve the new file. Regeneration
   proceeds as follows:
   a. Generate the digest and write it to a temp file (gzip-compressed).
   b. `os.rename()` the temp file to the digest path (POSIX-atomic).
   c. Clear the stale flag: `UPDATE digest_status SET stale = FALSE WHERE
      name = ? AND stale = TRUE`. The `AND stale = TRUE` acts as a
      compare-and-swap so that concurrent readers don't regenerate twice.

   This ordering ensures that on any crash, the stale flag may over-report but
   never under-report: a digest file on disk is always complete and valid.

All digest generation logic lives in `apps/base/digests.py`, since digests
often span multiple domain tables (e.g., a prelievi write also updates
`parcel_year_production.json` used by bosco).

Digest files are gzip-compressed (stored as `.json.gz`) and served with
`Content-Encoding: gzip`.

## Caching

Fetched data is cached client-side in memory. The cache is keyed by a "data_id"
that identifies a particular dataset corresponding to a server-side JSON file,
e.g., "daily harvest operations" or "monthly sawmill production", and it stores
the last refresh time.

The maximum size of the cache is the sum of all tabular and pre-processed data,
on the order of 10-20 MB uncompressed, plus 10-100 MB of satellite imagery in
the worst case.

Page reloads clear all cache state.

When changing domain pages:

1. The app renders from cache immediately if data is available.

1. The app also sends background conditional GETs to the server with
   If-Modified-Since for every dataset displayed in the domain page. If the
   server returns 304 Not Modified, no action is required. If it returns 200 OK,
   the app updates the cache with the newly received data and re-renders. In
   both cases it updates the dataset's last refresh time.

Conditional GETs also run once every 5 minutes (on a timer) for every dataset
visible in a particular domain page (i.e., the page will refresh even if the
user performs no navigation).

Domain switching feels instant for previously-viewed data.

When a domain page is loaded for the first time (empty cache), a modal displays
"Caricamento..." until the initial data fetch completes. Subsequent cache
refreshes happen silently in the background.

## Data entry and cache updates

Data entry forms are Django-rendered HTML fetched as fragments and displayed
in overlay modals (see "Modals" in `docs/ui-design-patterns.md`).

Each form has custom HTML and validation JS as needed, but common patterns (form
interception, error display) are extracted into shared libraries.

Fields that are enum-like (correspond to finite sets of values defined within
the app itself) are implemented as pull-downs. These include worker names, crew
names, tractor names, and tree species names (see below for details).

The process of data entry runs as follows:

1. The user initiates a data addition or edit by clicking on a UI button (the
   visual details of this are below).
1. JS fetches the form HTML from Django (including the CSRF token and an
   idempotency nonce as a hidden field) via `fetchModalForm`.
1. The form is rendered in an overlay modal. The URL *does not change*.
   This is the one exception to the "canonical representation of view state"
   rule, since we never need to share the input form.
1. Client-side JS validation provides immediate feedback: the submit button is
   inactive until JS validation passes.
1. On submit, JS intercepts and forwards the POST request (including the CSRF
   token and idempotency nonce), and waits for a response.
1. The server checks the nonce: if it has already been used (i.e., a previous
   request with the same nonce succeeded), it returns the original success
   response without writing again. This prevents duplicate records when the
   network drops between server-commit and client-receive and the client retries.
   Used nonces are stored in the database with a timestamp. The nightly backup
   cron job also prunes nonces older than 24 hours.
1. The server provides authoritative validation.

The server response has one of three values. The payload is always JSON.

1. Success: Code = 200 OK, payload = { data_id: X, row_id: Y, "record": [
   Y, ... ] }

   The client updates row Y of entry X in the case with the new record and
   refreshes the tabular display.

   Note that a future background conditional GET might refresh other cache
   entries, such as those corresponding to digested data for graphs. Concrete
   example: user enters data corresponding to a new harvest operation. The
   tabular display of harvest operations updates immediately. Digested data for bar chart of
   monthly harvests might be loaded after the next conditional GET.

1. Validation error: Code = 400 Bad request, payload = { status:
   "validation_error", message: "...", html: "..." }

   The page displays the error message in a modal and again displays the blank
   HTML form given by the "html" field.

   This rarely happens, since most server-side validation is consistent with
   client-side validation.

1. Conflict: Code = 400 Bad request, payload = { status: "conflict", message:
   "...", data_id: X, row_id: Y, "record": [ Y, ... ], html: "..."}

   This error happens on attempted edit or delete. Another user edited or
   deleted the entry between the time when the current page's last cache refresh
   and the time of submit. The HTML contains the form populated with the current
   server state. If the record has been deleted on the server, the message
   provides this information and the user can click "submit" to re-add the
   record. If the user escapes out of the edit form, the cached data is updated
   with the returned record, as for a successful update.

## Data deletion and cache updates

If the user deletes a record (more on the UI details of this below), the UI
displays an alert warning that the action cannot be undone. 

If the user confirms, a POST is sent to the server as for data insertion / edit
above.

1. Successful responses contain a row_id but no record field. The client removes
   the given id from the cache.

1. No validation errors are possible.

1. Conflict means that a row was edited since last cache refresh. The response
   contains no HTML but a valid record. The cache is updated as for a successful
   update. The error message is displayed in a modal and the user has the chance
   to try deletion again.

## Optimistic table updates — the contract that keeps writes snappy

Every write that mutates a row in a user-visible table MUST return the
full row in the success response — `record` for single-row writes,
`records` for bulk writes — shaped identically to the corresponding
JSON digest.  The client patches the cache via `cache.updateRow`
(`apps/base/static/base/js/cache.js:66`) or `cache.updateRows`, then
re-renders the table from the cached data via
`table.setData(cache.get(dataId))`.  No network round trip on the hot
path, no server-side digest regeneration on the hot path.  The
server-side `mark_stale()` flag still runs so the next cold reader (or
the next background refresh) regenerates the on-disk digest.

When a write also bumps a *materialised* value in another digest
(e.g., a tree-save changes `samples.N_alberi` and
`surveys.N_aree_visitate`), the response carries those rows too:
`sample_record`, `survey_record` (or `_records`), `grid_record`,
`area_records`.  The Campionamenti client funnels these through a
single `applySideEffects(data)` helper in
`apps/campionamenti/static/campionamenti/js/campionamenti.js` that
patches every affected cache and re-renders the touched view (map,
summary, table).

**The contract that prevents drift.**  Extract a `build_<digest>_record`
helper in `apps/base/digests.py` and call it from BOTH the digest
generator and every write view.  Lock the column-shape contract with
a test that builds a fresh row through the digest generator and a
record through the write view, then asserts they are equal.  See
`build_harvest_record` (`apps/base/digests.py:184`) for the original
pattern and `build_tree_sample_record` etc. for the Campionamenti
versions.

**Why this matters.**  Tree-save on Campionamenti was a 5+ second
round trip during M3d before this pattern landed — the digest
generator scanned thousands of TreeSample rows on every save.
Returning the freshly-built row inline cuts that to a single INSERT +
one helper call, well under 200 ms.  Bulk paths (CSV imports) that
would carry megabytes of records may stay on the slow path —
`tree_csv_import_view` is the documented exception.
