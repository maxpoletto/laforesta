# M3: duplication and Italian/English key audit

30 October 2026. Reviewed revision: `8d0aaa98e30ec1d00de93f66f49262e73282a37d`.
Original audit and proposed resolutions are retained below. Implementation
progress after M1/M2 is recorded at the end and in the inventory status columns.
Stored/offline/wire naming changes remain deferred pending review.

The row-by-row inventory is in [audit-m3-inventory.csv](audit-m3-inventory.csv).
IDs below correspond to that file. Priorities describe implementation order;
they are not a new claim that every duplication is a correctness defect.

## Findings and scope

The most useful work is consolidating policy owners and moving name translation
to explicit adapters. A blanket replacement of Italian words would touch
persisted data, offline queues, CSV interfaces and URLs without fixing those
policy differences.

The review covered application Python and desktop/mobile JS, the shared
constants and localization catalogs, import/export adapters, models, digests,
offline storage/upload paths and relevant tests/documentation. Historical
migrations and vendored libraries were excluded from clone analysis. They are
not appropriate cosmetic-refactoring targets. No production database was read.

Discovery checks:

- Python AST comparison examined 698 functions with at least 18 body AST nodes,
  after removing a leading docstring. One exact multi-function body match was
  found: `_fsync_dir()` in observation storage and Ipso staging. Function names,
  decorators and argument declarations were not part of that body comparison.
  This does not measure semantic duplication or justify a “duplication rate”.
- An eight-nonblank-line lexical scan covered 81 owned `.js` files, excluding
  vendor/catalog files. It found shared collapsible-section wiring and repeated
  classic-script test bootstrapping. Most meaningful near-duplicates differ in
  names, defaults or policy and require manual review.
- Manual call-path inspection found the policy/data mirrors below. Repeated
  names such as `validate_rows()` in independent importers were not counted as
  duplication merely because their names match.
- The existing tests were inspected. No full runtime test suite was run for
  this documentation-only audit. The small name-based Tabacchi check described
  in M1 ran without Django; the cache and concurrency observations are source
  analysis and proposed regression cases, not newly verified runtime failures.

## Policy and calculation duplication

### D01–D03: consolidate these first

**D01 — Actual harvest-volume materialization has two owners.**
[`_rematerialize_volume_actual()`](../apps/prelievi/views.py) and the final
parent loop in [`csv_harvests.apply()`](../apps/prelievi/csv_harvests.py) repeat
the SUM, empty-to-zero rule, parent lookup, comparison and partial save. Both
currently omit the parent's modification timestamp. Marked-volume maintenance
in [`mark_import.py`](../apps/piano_di_taglio/mark_import.py) is related, but
has a different null/empty rule and first-mark state transition.

Create a small plan-item totals owner used by manual and CSV operations, with
explicit marked/actual semantics. Include parent-lock ordering, field ownership,
history and returned parent state; do not just extract the SUM expression.
Implement under the [M2 contract](proposal-m2-versioning.md).

**D02 — Version/timestamp policy is scattered.**
Manual `version += 1`, full saves, partial saves, User's automatic counter and
staging's automatic timestamp are different mechanisms. Some differences are
intentional. Use M2's narrow edit and materialization operations; do not put a
universal increment in `TimestampedModel.save()` or convert User to caller-side
increments. C14 must keep its protections and login exception.

**D03 — Tree volume/mass calculation has parallel wrappers.**
[`mark_volume_and_mass()`](../apps/piano_di_taglio/mark_import.py) and
[`tree_volume_and_mass()`](../apps/campionamenti/csv_trees.py) both choose the
Tabacchi equation, handle unavailable values and multiply by species density.
The latter also handles coppice and an import-time name override. The manual,
CSV and Ipso paths already reuse these wrappers within their domains.

Move the common estimate operation to a neutral base module, parameterized by
an explicit equation identity and density. Domain callers decide applicability
(for example, coppice) and retain their existing unknown/zero behavior. Keep
CSV species-alias resolution in the CSV adapter. Introduce this with M1's
equation identity change, not as an extra rename-only change beforehand.

### D04–D10: data mirrors and cross-layer policy

| ID | Evidence and distinction | Proposed resolution |
| --- | --- | --- |
| D04 | [Python Tabacchi table](../apps/base/tabacchi.py) and [JS table](../apps/base/static/base/js/volume.js) duplicate coefficients/proxy mappings; previews and server calculations legitimately run in different languages | Keep both runtimes and parity tests. If adopting M1, generate coefficient data for both from one checked-in, locale-neutral source. Do not generate the numerical algorithms or require an online preview request |
| D05 | `config/constants.py`, shared `base/js/constants.js`, and Ipso `pwa/constants.js` repeat wire literals; the mobile file explicitly requests alignment | Define a small neutral manifest of cross-runtime wire values and generate the two JS projections, or first add a parity test for the deliberately shared subset. Do not generate UI strings, Python-only settings or unrelated client constants |
| D06 | Upload validation, CSV adapters and form parsers have different integer/boolean rules | Preserve the edge contracts: native Ipso JSON requires real JSON integers/booleans; forms accept their documented strings; CSV tolerates integral spreadsheet values. Share canonical domain validation after decoding, not one permissive parser. `numparse`, `csv_io`, `tree_validation` already supply much of this |
| D07 | Mutation-specific `mark_stale()` lists and response invalidations repeat dependencies across manual/import paths | Give an operation family one dependency description or affected-identity result. Use it for server dirtiness and client changes inside the transaction. Keep row builders shared. A row patched into cache must not also be invalidated in the same response |
| D08 | Python/JS hypsometry, diameter classes, basal area and volume code coexist | Retain server authority and offline/live calculations. Test shared fixtures, rounding, unavailable values and class boundaries. Share coefficient/config data where practical; do not replace working algorithms with a generic cross-language execution framework |
| D09 | [Desktop `geo.distanceMeters()`](../apps/base/static/base/js/geo.js) uses radius `6378137.0`; [Ipso upload](../apps/ipso/static/ipso/pwa/upload.js) has its own Haversine implementation with `6371000` | Choose and document the metric used for sample-area membership before sharing it. Their ratio differs by about 0.112%, which can change a boundary decision. Add near-radius fixtures first. Reuse the existing [Ipso geometry bridge](../apps/ipso/static/ipso/pwa/geo.js); no wholesale PWA module conversion is needed |
| D10 | [Desktop `parcelNames()`](../apps/base/static/base/js/geo.js) reads `layer`/`name`; [backend `parcel_key()`](../apps/base/parcel_geometry.py) additionally accepts `compresa`/`Compresa` and `particella`/`Particella` | Specify supported GeoJSON property aliases and test identical fixture inputs in both languages. Translate to one internal region/parcel representation at the geometry boundary. Do not silently drop currently accepted source formats |

### D11–D15: lower-cost or deliberately retained duplication

**D11 — Formatting and categorical colors.**
[`IpsoFormat`](../apps/ipso/static/ipso/pwa/format.js) and
[`IpsoPalette`](../apps/ipso/static/ipso/pwa/palette.js) mirror desktop helpers.
They are not exactly equivalent: mobile formatting converts numeric strings,
while the desktop decimal formatter returns non-number inputs unchanged.
Extract only a pure common subset or generate the palette constants; test the
input contract before replacing either wrapper. Do not import the whole chart
controller just to obtain a color list. Existing classic-script globals and
offline precaching are part of the deployment work.

**D12 — Structured/free-survey import loops.**
[`sample_import_rows()` and `free_survey_import_rows()`](../apps/ipso/importers.py)
repeat reference loading, session operator extraction and some row handling.
Their sample identity, number allocation, area and coppice rules differ. Retain
two explicit orchestration functions. Extract a shared validated measurement or
reference-resolution step only where identical policy is demonstrated. The
shared `record_measurements()` and sample normalizers are useful starting points.

**D13 — Filesystem primitives.**
The identical `_fsync_dir()` implementations and staging's text/byte atomic
writes are candidates for a small common file utility. Keep the higher-level
observation and upload storage operations separate: their rollback hooks,
checksum identities and cleanup responsibilities differ. R1/S2 recovery tests
must remain in place. This is low priority compared with domain policy.

**D14 — UI section wiring and classic-script test shims.**
Campionamenti and Prelievi repeat the setup around the already shared
`wireCollapsibleToggle()`. A small section-binding helper could remove that
boilerplate if another caller needs it. It is not a reason to build a page
framework. CommonJS/global bootstrapping in mobile modules is also deliberate
compatibility scaffolding; avoid turning it into a new runtime dependency.

**D15 — Fingerprints look similar but encode different identities.**
CSV mark v2 fingerprints, legacy aliases, Ipso record fingerprints, staged
payload checksums and authenticated nonce hashes all serialize data and hash
it. Their field sets, encodings and version prefixes are contracts, not
interchangeable implementations. Keep them explicit and add frozen test
vectors before any serializer extraction. Reformatting decimals or changing
`ensure_ascii` can alter retry identity even if JSON meaning is unchanged.

## Existing sharing to retain

- `save_model_response()`, `conflict_response()`, `submitted_version()` and
  `check_delete_version()` already centralize important write policy.
- C1/C2/C11 already tie nonce receipts, accepted writes and digest dirtiness
  together. Refactoring must not move mutations outside that boundary.
- `mark_parcel_matches_item()` is already shared by manual/import paths;
  do not reintroduce independent parcel-scope checks.
- `normalize_sample_tree_values()` / `normalize_sample_shoot_values()`,
  `csv_io` and `numparse` already own shared validation/decoding rules.
- `_int_ids()` is **not** duplicated: Ipso views import it from importers.
  Importing a private helper is a small ownership/naming issue, not grounds
  for claiming two implementations or introducing another copy.
- Full digests and optimistic patches reuse row builders. Preserve this
  stronger guarantee even if moving services out of large view modules.
- `selectable_species()` already expresses the legacy-Altro exception. Keep
  it narrow; this audit does not propose a new persisted selectability flag.

## Naming audit: four different boundaries

### 1. Internal application code

Use English identifiers and IDs for relationships. Local names such as
`compresa`, `particella`, `genere`, `n_aree`, `n_rilev`, and `ceduoEl` can become
`region_name`, `parcel_name`, `species_name`, `area_count`, `survey_count`, and
`coppiceInput` when their containing code is refactored. These local renames
need no data migration. Keep them out of a behavior-changing patch unless
they directly clarify the edited function.

Do not conflate a parcel's human identifier with a database key:
`parcel_name` is a string such as `1b`; `parcel_id` is an integer FK. Natural
sorting changes ordering, not identity. Never turn `1`, `01`, or `1b` into an
integer as part of a naming cleanup.

### 2. Human input/output

Italian labels and CSV headers belong here. `compresa`, `particella`, `genere`,
`funzione`, `fustaia`, and user-visible names remain valid localized interfaces.
An import adapter translates them to English domain names; export reverses
that mapping. Header aliases and species-name aliases are different concepts.

Likewise, localized digest column labels are an existing table/cache contract.
Converting all desktop digests to English machine keys plus labels might be a
reasonable later architecture change, but it affects every column lookup,
CSV/search configuration and cached digest. It is not required for this audit's
first refactors. Add hidden ID fields for machine joins where needed (M1).

### 3. Ipso persistence and reference compatibility

The current storage is intentionally mixed. [`store.js`](../apps/ipso/static/ipso/pwa/store.js)
uses IndexedDB schema 9; tree rows contain `specie`, `numero`, `particella`,
`gruppo` alongside English measurement fields and canonical Abies IDs. Session
rows contain `data`, `compresa`, `operatore`, `catastrofata`. This is live stored
data, not merely variable spelling.

The reference bundle also publishes `common`, `compresa`, `particella`, and
`ipsometrica`, while uploads use ID-based English keys. The existing
[`canonicalRecord()`](../apps/ipso/static/ipso/pwa/upload.js) already translates
much of this at upload time. It prefers captured IDs and falls back to name
resolution for older records. Preserve that precedence: an old label snapshot
must not override an existing ID after a reference label changes.

The proposed internal mapping is:

| ID | Existing field/context | Canonical in-memory meaning | Persisted/wire action |
| --- | --- | --- | --- |
| N01 | Tree `specie` | `species_name` snapshot, plus `species_id` | Adapt at the store boundary; preserve old records |
| N02 | Tree `numero` | `number` | Preserve integer/null and mode-specific numbering rules |
| N03 | Tree/reference `particella` | `parcel_name`, plus `parcel_id` | Name remains a string; do not confuse with ID |
| N04 | Tree `gruppo` | `group` display/export value | Do not equate with `sample_area_id`; structured mode has an explicit ID |
| N05 | Session `data` | `date` | Keep separate from `started_at`/creation time |
| N06 | Session/reference `compresa` | `region_name`, plus `region_id` | Prefer captured ID; retain label snapshot |
| N07 | Session `operatore` | `operator` | Keep the original text and per-record override semantics |
| N08 | Session `catastrofata` | `damaged` session context | Do not change selection/state rules while renaming |
| N09 | Reference species `common` | `common_name` or display `species_name` | Old bundle remains readable; no localized key names internally |
| N10 | Reference `ipsometrica` | `hypsometry`, preferably indexed by region/species IDs | Add a normalized reference adapter; version a breaking format change |
| N11 | `ipso.operatore`, `ipso.specie` localStorage keys | Preference names, not domain data | Keep old storage keys initially; optional read-old/write-new migration later |
| N12 | `martellate` mode, `terreni.geojson`, Italian route/DOM names | Frozen external identifiers | Keep literals stable; use English code symbols around them |

`group` is not necessarily a biological or database identity, and `catastrofata`
is not a synonym for every intervention flag. The adapter should preserve the
actual meaning, not mechanically translate words to similarly named fields.

### 4. Other machine boundaries

- **N13 — Desktop legacy name filters:** `FIELD_COMPRESA` / `FIELD_PARTICELLA`
  are still used for the sampling import-form query. Normalize inside that
  endpoint first; retain old query keys while callers transition.
- **N14 — CSV parser intermediate dictionaries:** plan/mark importers use
  keys such as `compresa`, `particella`, `numero`, `catastrofata` after header
  mapping. Change these internal dictionaries to explicit English fields,
  while leaving CSV headers/accepted aliases unchanged.
- **N15 — Geometry properties and helper results:** `parcelNames()` returns
  `{compresa, particella}`, unlike English backend/model concepts. Introduce
  a canonical geometry adapter; retain the current export as a compatibility
  wrapper until consumers migrate. Preserve external GeoJSON properties.
- **N16 — Species labels used as scientific keys:** covered by M1. Translate
  at CSV/display boundaries and use Species IDs/equation keys internally.
  Do not translate persisted species names when switching the UI catalog.
- **N17 — Local variables and digest labels:** rename local variables when
  useful; retain localized table/CSV labels through the established catalogs.
  These are different changes with very different compatibility costs.

## Proposed adapter design

Start with pure adapters rather than a database migration:

1. `decodeReference(bundle)` produces one canonical reference view, retaining
   `reference_version` and captured IDs. It accepts currently deployed bundles
   and builds ID indexes for species, regions, parcels and height regressions.
2. `decodeStoredSession()` / `decodeStoredTree()` produce canonical in-memory
   objects from legacy IndexedDB rows. Matching encode functions write the
   existing shape initially. Keep measurement precision, sequence and IDs.
3. Recording/UI/export/upload code consumes canonical objects. A CSV adapter
   still emits the same readable headers/names. An upload adapter still emits
   the current English, ID-based upload schema 1.
4. Name fallback is explicitly legacy behavior. Resolve only a unique match;
   reject ambiguity or a missing mapping with a recoverable error. Never guess
   a taxon or parcel, or silently use a different ID based on a newer label.

Do not maintain two independently mutable versions of every object. Decode
once on read, mutate the canonical object, encode once on persistence. During
incremental rollout, give each migrated function a clearly documented input
shape and keep the bridging adapters at its boundary.

An IndexedDB rewrite can be deferred indefinitely if boundary adapters are
small and reliable. If later justified, it is a separate versioned migration:
test actual older database fixtures, migrate atomically, preserve session IDs,
client record IDs, sequence numbers, photo references and upload statuses.
Do not rewrite an already prepared pending-upload envelope or its checksum just
because internal spelling changed. The IndexedDB schema version, upload schema
version, reference content hash and service-worker APP_VERSION are four
different mechanisms; update only those whose contracts change.

## Implementation order and proof required

1. Adopt M2's policy, then consolidate parent total/version writes (D01/D02).
   Preserve partial/unknown aggregate semantics and nonce rollback tests.
2. Address M1's equation mapping and height-lookup dependencies (D03/D04/N16).
   Keep readable CSV imports and existing estimate values unchanged.
3. Add constant parity coverage, then optionally generate the shared literals
   (D05). Test actual values in all three runtimes; the current
   `test_js_constants.py` checks imported export names, not full value parity.
4. Introduce the reference/storage adapters and migrate mobile controllers in
   small groups (N01–N10). Start with read-only views and export, then recording
   and upload. Keep resumable and pending-upload fixtures throughout.
5. Resolve the distance metric with boundary tests before sharing it (D09).
   Standardize geometry aliases through fixtures (D10/N15).
6. Clean up formatting, colors, filesystem primitives and section boilerplate
   only when their maintenance benefit warrants the extra code movement.

Cross-path tests should compare final stored values and side effects, not only
helper return values. Cover manual/CSV/Ipso entry, inactive/legacy species,
structured and free surveys, coppice shoots, PAI shared identity, zero versus
unknown volume, rejected writes, and dirty/response patch consistency.

Offline tests need old IndexedDB rows and cached references, interrupted
upgrades, pending-upload retries, CSV exports and photographs. Verify that the
same persisted session still yields the same upload identity and that an old
bundle remains useful without a network refresh. A renaming change is complete
only when those compatibility cases survive, not when a text search finds no
Italian field names.

## Implementation progress

One commit per D finding; already-resolved or intentional duplication is
recorded without introducing a redundant refactor.

| Finding | Status | Result / verification |
| --- | --- | --- |
| D01 | Covered by M2 | One intervention totals owner now serves manual/CSV writes; parent locks precede mutations/SUM; distinct marked NULL and actual zero rules retained. M2 totals, rollback and CSV regressions cover it. |
| D02 | Covered by M2 | Row clocks now cover partial writes; accepted edits own tokens; total-only writes preserve metadata tokens; User, SiteSettings and staging intentionally retain separate contracts. Optional edit-helper cleanup is deferred. |
| D03 | Completed | M1 supplies one estimate_volume_and_mass owner for both domains. Removed ignored species-name override parameters; CSV name resolution stays at its boundary. Configured-equation and CSV tests: 36 passed before/after cleanup. |
| D04 | Completed after M1 | Live coefficients already came from DB. Removed the unused browser default/proxy catalogue and name-based API; one-source numeric parity plus DB-to-preview/custom/proxy tests pass (14 Python, preview JS). |
