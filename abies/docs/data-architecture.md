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

Each page's doc (`docs/page-*.md`) carries the authoritative
write→digest invalidation table for its domain, including cross-domain
cases (e.g., a prelievi harvest write that marks `harvest_plan_items`
stale).

When you add a new write endpoint, audit every `generate_*()` in
`apps/base/digests.py` and add `mark_stale()` for each digest that touches
the written table.  When you change a `generate_*()` to read a new table,
audit every write endpoint that touches that table.  Tests under
`TestDigestInvalidation` (`test/test_campionamenti_views.py`,
`test/test_prelievi_views.py`, `test/test_piano_di_taglio_views.py`,
`test/test_hypso_views.py`) lock the contract: each test performs a write and
re-reads the affected digest via the public endpoint, asserting that
materialized columns reflect the change.  Add equivalent tests for any new
write/digest pair.

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

Data is cached client-side in memory, keyed by `data_id` (one per
server-side JSON file). Page reloads clear all cache state.

On every domain switch the app renders from cache immediately, then
fires background conditional GETs (If-Modified-Since) for every visible
dataset, re-rendering on 200. A 5-minute timer repeats this even
without navigation. First load (empty cache) shows a "Caricamento..."
modal until data arrives.

## Data entry and cache updates

Forms are Django-rendered HTML fragments displayed in overlay modals
(see `docs/ui-design-patterns.md` §Modals). The data entry flow:

1. JS fetches form HTML (including CSRF token + idempotency nonce) via
   `fetchModalForm` and renders it in a modal. URL does not change.
1. Client-side JS validation gates the submit button.
1. On submit, JS intercepts the POST (with CSRF token + nonce).
1. Server checks nonce for dedup (prevents double-writes on network
   retry). Used nonces are pruned after 24 hours.
1. Server validates and responds.

### Response contract

Payload is always JSON.

**Success** (200): `{ data_id, row_id, record: [row_id, ...] }`.
Client patches the cache row and re-renders. Other digests (charts,
etc.) update on the next background conditional GET.

**Validation error** (400): `{ status: "validation_error", message, html }`.
Modal re-displays the form with the error. Rare — client-side
validation catches most problems first.

**Conflict** (400): `{ status: "conflict", message, data_id, row_id, record, html }`.
Another user modified the row between cache refresh and submit. Modal
shows the server's current state; user can re-submit or dismiss (which
updates the cache with the returned record).

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
(e.g., a tree-save changes `samples.N_alberi`), the response carries
those side-effect rows too. The client patches every affected cache
and re-renders the touched views.

**The contract that prevents drift.**  Extract a `build_<digest>_record`
helper in `apps/base/digests.py` and call it from BOTH the digest
generator and every write view.  Lock the column-shape contract with
a test that builds a fresh row through the digest generator and a
record through the write view, then asserts they are equal.  See
`build_harvest_record` for the original pattern.

Bulk paths (CSV imports) that would return megabytes of records may
skip the optimistic path — `tree_csv_import_view` is the documented
exception.
