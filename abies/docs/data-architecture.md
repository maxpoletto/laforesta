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

    digest_status(name TEXT PRIMARY KEY, stale BOOLEAN DEFAULT FALSE,
                  dirty_seq INTEGER DEFAULT 0)

Write path: after a successful save, the view marks affected digests as stale:
`UPDATE digest_status SET stale = TRUE, dirty_seq = dirty_seq + 1 WHERE
name IN (...)`. This is the only write-path cost.  `dirty_seq` is a
monotonic token; the read path uses it to detect a write that landed mid
regeneration (see below).

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
   a. Snapshot `dirty_seq` for the digest.
   b. Generate the digest and write it to a temp file (gzip-compressed).
   c. `os.rename()` the temp file to the digest path (POSIX-atomic).
   d. Clear the stale flag *only if no write intervened*: `UPDATE
      digest_status SET stale = FALSE WHERE name = ? AND dirty_seq = ?`,
      passing the value snapshotted in (a).

   The `dirty_seq` compare-and-swap closes a lost-update race.  A boolean
   `stale` flag cannot tell "the TRUE I observed before generating" from "a
   TRUE a concurrent writer set while I was generating": clearing on
   `stale = TRUE` would erase that newer write's mark, leaving an on-disk
   digest that predates the write yet reads as fresh forever — only a manual
   file delete recovers it (the extreme case is an empty digest produced when
   the generator's snapshot fell inside a destructive re-import window).
   Because every mark bumps `dirty_seq`, an intervening write makes the
   snapshotted value stale, the `UPDATE` matches no row, and the digest
   stays flagged so the next read regenerates against the newer data.

   The file is renamed into place (c) *before* the flag is cleared (d), so on
   any crash the stale flag may over-report but never under-report: a digest
   file on disk is always complete and valid, and a digest reading as fresh
   always reflects the latest committed write.

All digest generation logic lives in `apps/base/digests.py`, since digests
often span multiple domain tables.

Digest files are gzip-compressed (stored as `.json.gz`) and served with
`Content-Encoding: gzip`.

Digest responses are served `Cache-Control: no-store`.  The app is its own
cache — the in-memory client store (below) plus the conditional GET above —
so the browser's HTTP cache must stay out of the loop.  Without `no-store` a
`Last-Modified` response with no `Cache-Control` is eligible for the
browser's *heuristic* cache: after a write the in-memory store is patched,
but a reload would serve the browser's stale copy without ever revalidating
(a stale table even though the server and DB are correct).  `no-store` still
lets the conditional GET answer 304 for unchanged digests, so it costs no
bandwidth.

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

### Request contract

Every mutating endpoint receives a JSON POST. Form controls are serialized into
a JSON object; CSV import file inputs are read as bytes in the browser and sent
as base64 strings in their normal file field (`file`, `fustaia_file`, or
`ceduo_file`). The server decodes that field back to bytes before running the
shared CSV reader, so UTF-8 and CSV validation remain server-owned. These
JSON upload bodies are capped by `DATA_UPLOAD_MAX_MEMORY_SIZE`, defaulting to
16 MiB and configurable with `DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE`.

Every retryable write carries a client-generated `nonce`. Versioned row edits
and deletes also carry the cached `version`; a missing version is treated as
stale (`0`) and returns a conflict instead of weakening optimistic locking.

### Response contract

Payload is always JSON.

**Success** (200): `{ data_id, row_id, patches: [{ data_id, row_id, record }], deletes: [{ data_id, row_id }] }`.
The top-level `data_id`/`row_id` identify the primary entity; row payloads
travel only through `patches` and `deletes`. The client applies every patch
and re-renders touched views. Other digests update on the next background
conditional GET unless the response carries their rows too.

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

1. Successful responses contain the deleted row in `deletes`. The client removes
   the given id from the cache.

1. Domain validation can still refuse protected deletes, such as an area
   referenced by samples or a harvest plan with active items.

1. Conflict means that a row was edited since last cache refresh. The response
   contains no HTML but a valid record. The cache is updated as for a successful
   update. The error message is displayed in a modal and the user has the chance
   to try deletion again.

## Optimistic table updates — the contract that keeps writes snappy

Every write that mutates a row in a user-visible table MUST return the
full row as a `patches` entry shaped identically to the corresponding
JSON digest: `{data_id, row_id, record}`. Bulk writes return one patch
per row. The client applies the generic envelope with
`cache.applyResponseChanges`, then re-renders touched views from cached
data. No network round trip is needed on the hot path, and no
server-side digest regeneration is needed on the hot path. The
server-side `mark_stale()` flag still runs so the next cold reader (or
the next background refresh) regenerates the on-disk digest.

When a write also bumps a *materialised* value in another digest
(e.g., a tree-save changes `samples.N_alberi`), the response carries
those side-effect rows as additional patches. The client patches every
affected cache and re-renders the touched views.

**The contract that prevents drift.**  Extract a `build_<digest>_record`
helper in `apps/base/digests.py` and call it from BOTH the digest
generator and every write view.  Lock the column-shape contract with
a test that builds a fresh row through the digest generator and a
record through the write view, then asserts they are equal.  See
`build_harvest_record` for the original pattern.

Bulk paths (CSV imports) that would return megabytes of records may
skip the optimistic path — `tree_csv_import_view` is the documented
exception.
