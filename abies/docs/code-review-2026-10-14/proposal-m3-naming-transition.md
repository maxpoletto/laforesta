# M3: English internal schemas and a bounded compatibility transition

Specification ID: **M3-NAMING-1**. Written 1 November 2026.
Status: **proposal; no migration or rollout has been implemented**.
Source baseline: `472d2e36f6c9708be0364d88f82a01328446665d`.

This document records the design agreed after the initial M3 cleanup. It is
intended to survive multiple intervening commits and application releases.
Implementation commits must cite a section or work-package ID below and update
the [rollout ledger](m3-naming-rollout.md). A later change to a decision must
update this specification explicitly, with its reason and compatibility effect.
Application version numbers alone do not establish completion of this migration.

This proposal supersedes the indefinite read-old/write-old adapter strategy in
the [original M3 audit](audit-m3-duplication-and-naming.md#proposed-adapter-design).
D01–D15 and the three completed internal naming commits remain completed work.
The [original inventory](audit-m3-inventory.csv) supplies the N01–N17 references.
This document authorizes no production data change by itself.

## 1. Decisions and scope

The target policy is:

1. Owned application identifiers are English. Python and stored/wire fields use
   snake_case; JavaScript locals and private view objects use camelCase.
2. Machine keys, enum values, dataset IDs and private API paths are English and
   independent of the UI language. Relationships use IDs, with names retained
   separately as display snapshots where appropriate.
3. User-visible headings, import/export headers, export filenames, messages and
   search vocabulary are localized. Machine keys are never translated to select
   a column or perform a join.
4. A transition release accepts the supported old formats, migrates mutable
   storage, and writes the new formats. Later releases retain that bridge until
   explicit cleanup gates pass. The cleanup release removes its runtime adapters.
5. Immutable operation evidence keeps its accepted representation. This includes
   frozen upload envelopes, retry hashes/receipts, historic audit values and
   existing backups. It is not a second live domain schema.

Species/region/parcel names, notes, operator text and other user data are values,
not identifiers to translate. `compresa` becomes `region_name`; `Capistrano`
remains `Capistrano`. Parcel names `1`, `01`, and `1b` remain distinct strings.
M1's species-ID equation relationships and M2's write/token contracts stay intact.
No measurement, equation, density, volume, mass, total or classification is
recomputed by this migration.

The scope includes IndexedDB rows and metadata, localStorage preferences,
protected Ipso reference/geometry/history responses, upload mode values, staged
upload working data, every desktop tabular dataset, row patches/deletes and
cache invalidations, and the corresponding controllers and import/export
boundaries. It includes legacy values such as `fustaia_file` outside N01–N17.

User-facing routes such as `/prelievi`, `/bosco`, and `/impostazioni` remain
localized and stable, including saved landing-page URLs. Standard GeoJSON keys,
browser database names, third-party APIs and brand names such as Ipso remain
unchanged. Existing downloaded CSVs, canonical source files, backup archives and
historical Django migrations are not rewritten.

Framework identities need a separate distinction. Django app labels, migration
dependency labels, historical audit row IDs and existing physical table names
are persisted infrastructure identities. Preserve them; changing a Python
package must not create a new Django application or new tables. Remaining owned
package/class/import names are a separate review unit (P07), with this mapping:

| Existing application package | English application package |
| --- | --- |
| `apps.bosco` | `apps.forest` |
| `apps.campionamenti` | `apps.sampling` |
| `apps.piano_di_taglio` | `apps.harvest_plans` |
| `apps.prelievi` | `apps.harvests` |
| `apps.mannesi` | `apps.crews` |
| `apps.controllo` | `apps.audit` |
| `apps.impostazioni` | `apps.settings` |

`apps.base`, `apps.ipso` and the configuration package are already English or
brand identifiers. AppConfig class/catalog symbols follow their English domain
names; `verbose_name` remains localized. Set the original `AppConfig.label`
explicitly, preserve `db_table` and migration history, and test content types/
permissions/history. Historical import
shims, if actually required, belong to migration support, not live format
adapters. These moves are isolated from storage conversion commits.

## 2. Releases, versions and compatibility

Use these release roles rather than assuming consecutive numeric versions:

| Role | Meaning |
| --- | --- |
| L | Last deployed legacy server/client; record its exact tag, image and Ipso version in the ledger |
| T | First deployed release containing the complete transition bridge and supported storage migration |
| T+n | Any subsequent release before cleanup; it retains T's compatibility obligations |
| B | Last tested bridge release, retained permanently as a recovery artifact; may be newer than T |
| C | Cleanup release, deployed only after the gates in section 12 pass |

At the source baseline, Ipso is 1.5.6, IndexedDB is version 9, reference and
upload schemas are version 1, and desktop tabular data has no explicit schema
version. These are source observations, not a statement about deployed devices.

Reserve the following independent contracts. If another change uses a reserved
number before implementation, amend this table and the tests before proceeding.

| Contract | Legacy | Transition target | Meaning |
| --- | --- | --- | --- |
| Naming epoch | implicit 1 | 2 | Capability used by diagnostics and rollout gates |
| IndexedDB version | 1–9 legacy upgrade inputs | 10 | Atomic local field/meta migration; new installs create this shape |
| Local session `schema_version` | existing storage version | 10 | Stored session shape; independent of its frozen upload's version |
| Reference `schema_version` | 1 | 2 | English reference keys and ID-based height map |
| Upload `session.schema_version` | 1 | 2 | Version-specific validation and accepted checksum preimage |
| Canonical staged working document | absent | 2 | English working projection; separate from accepted source evidence |
| Desktop table/response `schema_version` | absent, interpreted as 1 only on legacy endpoints | 2 | English column/data IDs, metadata and patch contract |
| Generated parcel geometry | unversioned legacy response | 2 | Canonical English properties; same geometry and coordinates |
| PWA `APP_VERSION` | source baseline 1.5.6 | actual T/B/C version per release | Shell/cache identity; bump for every shipped asset change |
| Reference content hash / digest ETag | existing content identities | recomputed for new representations | Content identities, never substitutes for a schema version |

T and every T+n server support both protocol versions. New clients emit only
version 2, except when replaying an already frozen version-1 upload. Version 1
responses are projections at the legacy boundary, not a domain data model used
by new controllers. Reject unsupported versions explicitly; never silently
interpret a version-2 body as version 1 or retry a rejected mutation as a
different version.

| Server / client | Legacy client | Transition/current client |
| --- | --- | --- |
| L server | Existing behavior | Unsupported; do not activate T clients before T server deployment |
| T or T+n server | Legacy endpoints/reference/upload supported during the bridge | Version-2 endpoints and schemas; frozen version-1 uploads still retry |
| C server | Legacy endpoints return an upgrade-required response; no writes occur | Version 2 supported; pre-T storage requires the retained bridge before use |

T+n must not change schema-2 meanings incidentally. Additive fields require
compatible defaults and tests; breaking changes receive a new contract version
and an amendment to this plan. A deployment does not restart or finish the
transition automatically.

## 3. Canonical names and migration rules

These mappings apply to the named object only. Do not recursively replace keys
inside arbitrary JSON, user notes, photo blobs, CSV text, or frozen envelopes.
Already-English fields retain their current type and meaning unless listed.

| Boundary / object | Legacy | Canonical target |
| --- | --- | --- |
| Stored tree | `specie` | `species_name` |
| Stored tree | `numero` | `number` |
| Stored tree | `particella` | `parcel_name` |
| Stored tree | `gruppo` | `group` |
| Stored session | `data` | `date` |
| Stored session | `compresa` | `region_name` |
| Stored session | `operatore` | `operator` |
| Stored session | `catastrofata` | `damaged` |
| Mode enum in live rows/new uploads | `martellate` | `marks` |
| Reference species | `common`, `latin`, `density` | `common_name`, `latin_name`, `density_q_m3` |
| Reference parcel, sample area, preserved-tree row | `compresa`, `particella` | `region_name`, `parcel_name` |
| Reference parcel | existing `region_name` plus `compresa` | one `region_name`; require agreement before removing duplicate |
| Reference height equations | `hypsometry_by_id` | `hypsometry`, still region-ID → species-ID → coefficients |
| Reference height equations | name-keyed `ipsometrica` | legacy input only; resolve to IDs under the rules below |
| Reference preserved trees | `pai.preserved_trees` | top-level `preserved_trees` array |
| History-detail tree display value | `species` | `species_name`; preserve `species_id` |
| Cached parcel features | `meta['boot:terreni']` | `meta['boot:parcel_features']` |
| Cached history detail | existing history value fields | canonical history fields; retain `history:<kind>:<id>` keys |
| Next-number metadata | `next_number:martellate:<operator>` | `next_number:marks:<operator>` |
| localStorage preference | `ipso.operatore` | `ipso.operator` |
| localStorage preference | `ipso.specie` | `ipso.species_name` |
| Desktop name query fields | `compresa`, `particella` | `region_name`, `parcel_name` |
| Internal high-forest flag | `fustaia` | `highforest` |
| JSON file-upload fields | `fustaia_file`, `ceduo_file` | `highforest_file`, `coppice_file` |
| Dataset/cache/dirty ID | `prelievi` | `harvests` |
| Dataset/cache/dirty ID | `hypso_params` | `height_equations` |
| Reference/geometry URL | `/ipso/terreni.geojson` | `/api/v2/ipso/parcels.geojson` |
| English code symbols | `IPSO_MODE_MARTELLATE`, Italian function/class/catalog symbols | `IPSO_MODE_MARKS` and English equivalents; displayed text remains in catalogs |

Keep `species_id`, `region_id`, `parcel_id`, `sample_area_id`, `survey_id`, tree
and photo IDs, sequence numbers, timestamps and flags unchanged. `group` is not
a sample-area ID; session `damaged` is not an alias for every intervention flag.
Names are not numeric IDs and are never normalized through a natural sorter.

For dual-key legacy fields within one row/object, use property presence, not
truthiness. Preferences and counters have the explicit merge policies in section
4. For an ordinary field, copy the old
value when only the old key exists; keep the new value when only the new key
exists; remove the old key when both contain the same typed value. A genuine
disagreement aborts durable-data migration or rejects the request. Do not pick
one silently. Missing values retain the existing legacy default; preserve the
distinctions among missing, null, empty string, zero and false where meaningful.
These aliases are accepted only by the legacy endpoints/migration adapters.
Version-2 inputs reject legacy field spellings rather than silently dropping
their values or carrying duplicate names forward. Keep the endpoint's normal
validation-response shape/status and preserve form input when reporting this.

Captured IDs always win over name snapshots. Renaming storage does not
re-resolve an existing ID against today's species or parcel names. A legacy row
without required IDs remains explicitly unresolved. Bridge-only resolution may
use a unique match in its captured reference; ambiguity, missing data or a
disagreement with a captured ID requires review. Do not invent IDs or borrow a
different species' height equation. Cleanup is blocked while recoverable rows
still require that fallback. No new general-purpose production repair UI is
included; unexpected unresolved cases require a reviewed recovery operation.

## 4. IndexedDB and preferences

### 4.1 Atomic IndexedDB migration

Use a real version-10 `onupgradeneeded` transaction over the existing `sessions`,
`trees` and `meta` stores. Create the stores/indexes normally on a fresh database.
For an existing database, update rows through cursors without clearing/rebuilding
stores or allocating new keys. Keep the database name `ipso`, auto-increment
generator, indexes and all unknown non-conflicting fields.

The transaction must:

1. Convert all local sessions/trees, including exported and abandoned records;
   set each session's storage `schema_version` to 10.
2. Translate live modes and next-number metadata. If old and new counter keys
   both exist, preserve the maximum valid counter, retaining existing operator
   normalization and the rule that deletion never lowers the counter.
3. Convert last-good cached reference, parcel-feature and history data without
   requiring a network request. Invalid disposable cache rows may be removed
   with a reported diagnostic; durable observations and photos are never removed.
4. Preserve the exact `upload_payload` of every existing session, its record
   count and pending/export/upload status. Do not recursively convert it.
5. Write `meta['naming_migration']` with naming epoch, source/target database
   versions, completion time and converted/unresolved counts in the same
   transaction. This marker establishes the IDB step only.

Any durable-row failure aborts the complete upgrade. The old database remains
recoverable, and recording/upload is disabled until migration succeeds. Show a
localized recovery message; never respond by deleting IndexedDB, registering a
different database name, or creating an apparently empty replacement database.

Do not await network, timers, image decoding or unrelated asynchronous work
inside the version-change transaction. Close older connections on version
change; handle blocked upgrades explicitly. A closed old connection must not
continue accepting saves. Keep existing transaction-completion acknowledgements.
Tests must exercise actual version-change transactions in addition to mocks.

### 4.2 Preferences outside IndexedDB

localStorage is not atomic with IndexedDB. After IDB conversion succeeds, copy
each legacy preference to its new key only when the new key is absent; an
existing new preference wins. Verify the write, then remove that old key.
Repeat this step safely on every T/T+n boot until complete. A quota/storage
failure leaves the old preference available and reports incomplete migration;
it does not roll back or erase successfully migrated observations.

Keep bearer and basemap keys, which are already English. Never put bearer
tokens, operator names or observations in migration diagnostics. Report the
combined IDB/preferences readiness separately from the IDB transaction marker.

### 4.3 Activation, skipped releases and older databases

Ship all new assets and schema migration in the same versioned PWA shell. Keep
operator-controlled service-worker activation; do not force an update during a
field save. Before opening an upgraded database, ensure the complete new shell
is available. Other tabs/PWA windows must close their old database connections.

T/T+n support the older database versions currently accepted by the application,
with fixtures covering versions 1–9 and missing pre-v7 captured IDs. If an old
record cannot be fully resolved, retain it for display/export/recovery and
report the missing IDs; it blocks cleanup rather than disappearing.

C supports a new install (`oldVersion == 0`) and migrated storage versions
10 through its own current database version. A database newer than the running
client is rejected safely. C rejects a pre-10 database by aborting the upgrade
transaction and directing the operator to bridge B. Do not delete old records. The retained
bridge provides the old→10 migration even after it leaves the normal app bundle.
Never assume a device has run an intermediate release merely because it can
download C. A reset browser/profile/origin is a separate installation.

## 5. Reference, geometry and offline history

Version-2 reference responses emit only canonical keys. Use one `hypsometry`
map indexed by region ID then species ID. Values retain `a`, `b`, and
`hypso_param_set_id`. A legacy bundle's ID map is authoritative if present;
a missing pair remains missing. Only when that map is absent may the temporary
adapter resolve the legacy name map through unique IDs in the same bundle.
Unresolvable pairs are reported and do not silently map to another equation.

Keep the existing reference hash algorithm for version 1. Version 2 hashes an
object containing exactly `schema_version`, `species`, `parcels`, `hypsometry`,
`sampling`, `preserved_trees`, `work_packages`, `observation_categories`, and
`upload`. Use UTF-8 JSON with sorted keys, `ensure_ascii=False`, compact
separators, and the first 20 hexadecimal characters of SHA-256, matching the
existing identifier length. Generation time and history remain excluded. Freeze
the new vector independently from v1. Never rewrite
an existing session's captured `reference_version`. A locally converted old
cache retains the source reference hash and records `source_schema_version: 1`;
that source identity is not claimed to be a hash of the converted representation.
Fresh version-2 server bundles carry the new hash. New sessions may use a
successfully converted last-good cache offline.

Generated parcel GeoJSON uses English `region_id`, `region_name`, `parcel_id`,
`parcel_name` properties plus existing English attributes such as `coppice`.
The FeatureCollection carries `schema_version: 2`. Coordinates, ring order,
feature IDs and spatial-selection/area/distance behavior do not change.
Do not reconstruct geometry from names or change CRS; WGS84 mean radius remains
6371009 m for the distance calculation.

Canonical source GeoJSON is human-maintained input. Its supported `layer`/`name`
and localized property aliases remain accepted at the import boundary, like
CSV header aliases. Generated internal geometry and cached live properties use
the new shape. This input support is deliberately not a removable runtime
compatibility adapter.

History index/detail responses use English keys and preserve English kind
values `mark`/`survey`. Labels remain localized. Existing one-hour detail-cache
expiry remains measured from its original successful fetch; migration does not
refresh expiry or turn an expired entry into a usable offline response.
Cached detail URLs are converted by explicit path mapping, not string replacement
in arbitrary user data. Protected responses retain bearer authorization and
no-store behavior; the service worker must not begin caching them.

## 6. Upload identity and staged working data

### 6.1 Accepted source versus canonical working representation

An upload has two distinct purposes: its accepted normalized source establishes
retry identity; its working representation supports preview/import and explicit
administrator corrections. Separate these purposes before removing old decoders.

For a version-1 request, run the baseline version-1 normalization/checksum path
unchanged **before** translating its mode or any keys. The existing server
checksum is over normalized session/records, not raw HTTP whitespace; `csv_text`
is saved separately. Preserve the existing photo normalization/checksum rules.
Version-2 validation has its own frozen preimage and accepts only canonical
names and enum values. Do not change decimal strings, null handling, omitted
fields or source checksums as a side effect of canonicalization.

Existing `IpsoUpload.schema_version` and `checksum` continue to describe accepted
source evidence. Keep `upload.json`, `upload.sha256`, `upload.ready`, exported
CSV and photo bytes as that evidence. Migrate the live model `mode` from
`martellate` to `marks`; historical rows retain their old recorded values.

Add nullable `canonical_schema_version` and `canonical_checksum` fields to the
model/history as a separate migration. Once a projection is published, they
identify a canonical working document at
`<inbox_path>/canonical/<canonical_checksum>.json`. Its shape is:

```json
{
  "canonical_schema_version": 2,
  "source_schema_version": 1,
  "source_checksum": "<accepted source checksum>",
  "payload": {
    "session": {"schema_version": 2, "mode": "marks"},
    "records": []
  }
}
```

The example omits unchanged session/record fields, which remain required under
their mode's contract. `canonical_checksum` hashes the complete working document
with a frozen UTF-8 JSON serializer (`ensure_ascii=False`, sorted keys, compact
separators). It is not used as a mobile retry receipt or an import fingerprint.
Validate source evidence/checksums before deriving this document. Include both
checksums in import-context concurrency checks and diagnostics.

Publish a new content-addressed file before committing the database pointer;
never overwrite the previously referenced working file in place. Use existing
runtime snapshot locking, durability and rollback conventions. A crash before
pointer publication leaves harmless unreferenced output; a committed pointer
must identify a complete validated file. Backups include working files and
source evidence. A bridge command can rebuild a missing projection from verified
source; a checksum mismatch blocks import rather than silently repairing evidence.

The conversion command processes every retained inbox item, including imported,
rejected and conflicting items. It is resumable and skips already verified
projections. Record conversion as migration metadata/history without fabricating
a new receipt/import or changing domain versions/totals. Existing history and
source evidence are never erased to achieve a zero-legacy-live-data count.

### 6.2 Mode corrections and retries

After T, an explicit administrator mode correction updates the canonical working
document and live mode under the existing audit/nonce boundary. It does not
replace accepted source identity. A duplicate replay of that source acknowledges
the original receipt and preserves the corrected working view. Changing upload
mode remains an explicit user action; migration itself never reclassifies work.

The baseline mode-correction implementation changes staged source/checksum.
For items already corrected before T, the source/checksum present at migration
is the baseline evidence; do not claim to recover a previous lost preimage.
Add tests for this deliberate separation of receipt identity and working edits.

Existing pending phone envelopes replay unchanged, using schema 1 and the
existing `/api/ipso/uploads/` endpoint. New envelopes use schema 2 and
`/api/v2/ipso/uploads/`. Keep session UUID, client record IDs, photo IDs/files,
captured timestamps, reference identity and order. Do not rebuild an uncertain
pending envelope from newly migrated rows or current reference data, issue a
new UUID, or drop it after a network error.

A session UUID remains unique across both protocols. The same accepted
schema/checksum replays successfully; a different source for the same UUID
still conflicts, even if a human considers the two sources equivalent. Import
fingerprints and their frozen D15 vectors stay unchanged. Preview/import after
T consumes the canonical working view; source download remains available as
read-only evidence. C requires a valid working projection for every retained
upload, so old source decoders can leave the normal application.

An existing completed legacy envelope may remain as opaque local evidence, with
no legacy retry/import action after C. C is blocked while any envelope can still
be retried under the supported recovery flows. If local-only/abandoned sessions
retain uncertain source envelopes, resolve them explicitly during the bridge;
changing their status must not discard observations, photos or evidence.

## 7. Desktop tables, digests and exports

### 7.1 Schema-2 table contract

Keep compact rows as arrays and retain `row_id` at position zero, including
the audit's existing string event IDs and null IDs in unsaved candidate rows.
Use stable English column IDs, with units where they clarify numeric meaning.
Provide localized display metadata separately:

```json
{
  "schema_version": 2,
  "data_id": "harvests",
  "language": "it",
  "schema_signature": "<sha256>",
  "columns": ["row_id", "version", "region_name", "parcel_name", "mass_q"],
  "column_meta": {
    "row_id": {"label": "ID", "hidden": true, "export": false},
    "version": {"label": "Versione", "hidden": true, "export": false},
    "region_name": {"label": "Compresa", "type": "string", "export": true},
    "parcel_name": {"label": "Particella", "type": "natural", "export": true,
                    "search_match": "exact", "search_aliases": ["part", "particella"]},
    "mass_q": {"label": "Q.li", "type": "number", "unit": "q", "export": true}
  },
  "rows": [[17, 3, "Capistrano", "01b", 120.5]]
}
```

Every column has metadata. `label` and optional `export_label` come from the
established localization catalogs or an entity's display name. Default export
heading is `label`; `export` controls inclusion. Optional `value_labels` maps
machine enum values to localized display/export values. New code branches on
raw enum values/IDs, never their rendered labels. Units use stable machine
tokens such as `q`, `m3`, `ha`, and `percent`; visible units remain localized.
Rendering widths/components stay client policy, not executable server metadata.

`schema_signature` is SHA-256 over compact, sorted-key, UTF-8 JSON containing
the ordered `columns` and complete `column_meta`, including labels. It changes
when the layout/meaning/display metadata changes, not for ordinary row updates.
The full response's byte ETag remains the authoritative HTTP content validator.
Validate unique column IDs, metadata coverage and row length before publishing
or applying data. New controllers use keys such as `parcel_name` for lookups;
`S.COL_PARCEL` supplies presentation only.

Before consumer conversion, commit an executable per-dataset column manifest
and frozen legacy fixtures. Each entry records old position/key, canonical key,
type/unit, label source, export heading/inclusion, and any enum transformation.
Derive mappings from row builders/model meaning, not by translating header text.
For example, the old `Compresa` column can denote an ID or a display name in
different tables. Do not globally map every identical heading to one field.
The manifest must cover the complete dataset inventory in section 7.4 before T.

### 7.2 Dynamic columns and human file interfaces

Dynamic Prelievi columns use these IDs:

| Meaning | Column ID |
| --- | --- |
| Species quintals | `species:<species_id>:mass_q` |
| Species percentage | `species:<species_id>:percent` |
| Tractor quintals | `tractor:<tractor_id>:mass_q` |
| Tractor percentage | `tractor:<tractor_id>:percent` |

Their metadata supplies `entity_type`, `entity_id`, `measure`, and the localized
display/export label. Preserve current species grouping, major/minor rules,
column order and the distinction between mass and percentages. Entity renaming
changes labels/signature/ETag, never the dynamic key or stored allocation.

CSV headers and downloaded filenames stay localized. Existing separator,
decimal/date formatting, BOM, quoting/formula protection, precision, column
ordering and round-trip identity behavior remain unchanged unless independently
approved. Export renders metadata labels/value labels, not machine keys. Keep
human header aliases and user-visible search terms, including `part:1` exact
matching and natural parcel ordering. No machine-schema rename changes species
or parcel text in existing CSVs. Localized table-sort/search values embedded in
user bookmarks are interpreted at that user-interface boundary; they are not
an excuse to retain localized internal column IDs.

### 7.3 Patch, cache and HTTP behavior

Version-2 data-entry envelopes carry `schema_version: 2`. Every row patch and
row-bearing conflict includes its dataset ID and `schema_signature`; a record
must use the same canonical row builder and column order as its full digest.
Deletes and invalidation IDs/prefixes use canonical English dataset names.

Apply a patch only to a complete cached dataset with a matching signature.
Otherwise invalidate and fetch the full dataset; never guess positions or
construct a partial digest from a patch. Preserve generation fencing, cold
fetch behavior and the exclusion of patched/deleted datasets from invalidations
when their layouts match. A missing target cache is not an error or permission
to lose the original invalidation. Validation/conflict responses keep forms
open and preserve user-entered values; version changes do not trigger automatic
resubmission. CSV and staged-import controllers still apply response changes.

Serve v1 and v2 through distinct URLs/cache identities. Version-2 table URLs
identify the configured language (`language=it` initially); validate it against
enabled languages. Cache keys include protocol, data ID and language. Disk
output is separated by protocol/language; never replace a legacy digest file
with schema-2 bytes under the same identity. Each representation computes its
own ETag over its published bytes; stale flags and dirty sequences preserve
the existing transactional/C6/C7/C11 contracts.

Expose canonical dirtiness to domain writers and fan it out in one owner to
versioned representations during T/T+n. Physical DigestStatus identities may
use `v1:<legacy_id>` and `v2:<canonical_id>`; no caller manually duplicates those
lists. At cutover, map existing status rows, merge collisions conservatively
(`stale=True`, `dirty_seq=max(old,new)+1`) and regenerate outputs. Existing
filesystem digests are disposable caches: rebuild rather than rewriting their
compressed contents. Apply the same treatment to dynamic digest families.
C can remove v1 cache/status entries after stopping legacy generators.

### 7.4 Dataset inventory

The baseline registry contains these 16 families: `prelievi`, `parcels`,
`species`, `audit`, `grids`, `surveys`, `sample_areas`, `samples`, `harvest_plans`,
`harvest_plan_items`, `future_production`, `parcel_dendrometry`,
`parcel_dendrometry_points`, `preserved_trees`, `observations`, `hypso_params`.
Rename only the two dataset IDs listed in section 3; the other English IDs stay.
Also cover `sampled_trees_<survey_id>` and `mark_trees_<item_id>` with their
existing English prefixes.

The same schema-2 contract applies to tabular responses built outside the
filesystem registry: Settings tractors/species/categories/users, crews/hours/
credits, the Ipso inbox, hypso compute candidates and tabular report endpoints.
Include side metadata such as observation categories and diameter-class mode.
Non-tabular metadata/detail/report JSON is included in the field/enum audit;
its existing English keys need no cosmetic rewrite. Tests enumerate routes and
generators so a new dataset cannot accidentally bypass the manifest/contract.

## 8. API, forms and internal identifier boundaries

New desktop API roots are `/api/v2/forest/`, `/api/v2/crews/`,
`/api/v2/harvests/`, `/api/v2/sampling/`, `/api/v2/harvest-plans/`,
`/api/v2/audit/`, `/api/v2/settings/`, `/api/v2/species/`, `/api/v2/geo/`,
and `/api/v2/ipso/`. Existing English suffixes retain their meaning; replace
private `pai/` with `preserved-trees/` and `import-martellate/` with
`import-marks/`. New protected mobile reference is
`/api/v2/ipso/reference.json`; parcel geometry is as specified in section 3.
History detail URLs use the versioned Ipso root and unchanged `mark`/`survey`
kind values. Keep the PWA entrypoint `/ipso/` and user-facing page routes.

T retains old API routes as version-1 adapters. They validate/decode before
calling the same domain services and project responses into the old contract.
Do not redirect POST requests, replay them at a new URL, or translate an old
body before computing its nonce request identity. UsedNonce rows and saved
responses remain in their accepted representation until normal expiry. New
and old request paths are different retry identities; callers never switch a
pending request between them. Keep token/permission/rate/byte/photo limits.

New HTML fragment endpoints and owned DOM/CSS/template symbols use English
internal names; visible labels remain localized. Keep legacy fragments/assets
coherent for old pages during the bridge. Package/asset moves must preserve the
complete old versioned shell during active sessions and must not mix versions
through service-worker fallback. A request for an old asset must not receive an
incompatible new module under the same cache identity.

Use an immutable asset namespace `/static/releases/<build_id>/...` for T and
later desktop/PWA assets. `build_id` identifies the complete asset build and is
recorded with the release. All imports, CSS/image references and lazy-loaded
controllers resolve within that build's manifest. Keep L's old asset paths
serving L's content until C; version the new HTML and SW entrypoints normally.
Each service worker precaches its complete manifest and reads its own cache.
Do not let a shared unversioned import fetch a different release's schema/helper
halfway through an open session. Rehearse activation with both old desktop tabs
and an old PWA running. C may remove L's compatibility assets after G05/G06;
retain B's complete manifest/assets in the recovery artifact.

C retains safe tombstone responses at retired API entrypoints: HTTP 410 with
an English `upgrade_required` code, supported schema `[2]`, and localized
explanation. These responses perform no writes and do not accept/translate old
formats. Unknown future schemas are rejected, not guessed. The small tombstone
is not a retained compatibility implementation. Use existing static routing
where possible; do not create a second API framework for this change.

## 9. Adapter register and removal boundaries

Give temporary adapters explicit IDs in code comments/tests and the ledger.
Keep them at named boundaries; domain services/controllers consume canonical
objects and do not test for old field spellings throughout their logic.

| ID | Adapter / support | Required through T+n | Removal condition at C |
| --- | --- | --- | --- |
| A01 | Old IndexedDB rows/meta → version 10 | In PWA upgrade path | Every installation migrated; pre-10 recovery tested through B |
| A02 | Legacy localStorage preference copy | Idempotent boot step | Every installation reports both preference migrations complete |
| A03 | Legacy reference/geometry/history → canonical cache | Offline upgrade and v1 server projections | All active/cached runtime bundles canonical; old-input recovery through B |
| A04 | Upload schema-1 decoder and original checksum path | Frozen/old-client upload retries | No remaining legacy retry source; observation window passes |
| A05 | Staged source → canonical working projection | New v1 receipt and bulk conversion | Every retained item has verified working data; old-source reconstruction moves to B |
| A06 | Legacy API/form/table/response projections and dataset aliases | Old desktop/PWA operation | All active clients new; nonce/traffic and cache gates pass |
| A07 | Name-only legacy identity fallback | Explicit unresolved legacy rows | No recoverable rows still require it |
| A08 | Legacy mode/counter/field aliases | Boundary normalization only | No mutable live rows/meta contain old names/values |
| A09 | Legacy internal geometry helper/result shapes | Until generated/cached data and consumers move | All current consumers use canonical properties/results |

Permanent human file-input parsers, locale label maps, opaque evidence readers,
framework migration identities and the offline recovery artifact are outside
this removal list. Distinguish them in searches/tests so a cleanup does not
break supported CSVs or historical evidence merely to eliminate an Italian word.

## 10. Implementation work packages and review units

No deployment should expose a half-converted release. The packages below can
take several commits, each starting with failing regressions where behavior
changes. Keep moves, contracts and data conversions separately reviewable.
Preparatory commits may ship while legacy defaults remain active; mark T only
when its complete compatibility matrix is tested and deployed.

| ID | Work package | Required evidence before completion |
| --- | --- | --- |
| P01 | Freeze inventories, per-dataset column manifest, legacy fixtures and adapter register | All table families/routes covered; serializer and format goldens frozen |
| P02 | Separate source receipt identity from staged canonical working documents | Same-checksum replay, mode correction, rollback, file publication and backup tests |
| P03 | Add version-2 server contracts/endpoints and legacy projections | Cross-version API/authorization, canonical row/patch and digest dirtiness tests |
| P04 | Convert desktop table metadata, controllers, dynamic keys and exports | All page/report/import paths and localized CSV/search/sort goldens pass |
| P05 | Convert Ipso reference/UI/storage and preferences with real upgrade transactions | Offline/restart/blocked/abort/multi-tab tests; pending payload and photo preservation |
| P06 | Add resumable server state/projection conversion and diagnostics | Check-only, repeat, interruption, collision, backup and restore tests |
| P07 | Finish remaining English symbols/package/asset moves in isolated commits | Migration graph, permissions/content types, historical imports and offline shell checks |
| P08 | Rehearse and deploy T; inventory every installation | Ledger records exact releases, counts, device checks and smoke-test results |
| P09 | Maintain bridge across T+n and gather exit evidence | Every intermediate release preserves the tested matrix and diagnostics |
| P10 | Retain/test B, remove A01–A09 from C's normal runtime, document recovery | Every gate below passes; C and skipped-upgrade/old-backup recovery rehearsed |

The concrete column manifest, migration filenames and actual T/B/C versions are
implementation artifacts to be recorded in the ledger; they are not guessed
here. No implementation may silently choose different canonical field names,
retry identities, loss handling or cleanup criteria from this specification.

## 11. Verification matrix

At minimum, cover these cases with fixtures derived from supported old versions
and the frozen source baseline. Use synthetic data in committed fixtures; do
not commit a production database or credential-bearing browser export.

| Area | Required cases / invariants |
| --- | --- |
| Local rows | All four modes; open/pending/exported/abandoned; null/zero/false; Unicode; parcel `1`/`01`/`1b`; unknown fields; old/new key collision; unchanged IDs/seq/timestamps/flags |
| IndexedDB | New install; versions 1–9; large session/photo set; actual cursor update and abort; interruption leaves all-old or all-new; repeat opens; blocked upgrade/old connection; auto-increment unchanged |
| Preferences/counters | Old only/new only/both; new preference wins; max valid counter; quota/write failure between copy/delete; repeat restart; bearer untouched |
| References | Fully offline conversion; absent ID map; missing pair in authoritative ID map; renamed names with captured IDs; ambiguity/missing IDs reported; invalid cache handled without observation loss |
| Geometry/history | Identical coordinates and parcel selection; source aliases; preserved radius/area behavior; original history expiry; no protected-response SW caching |
| Uploads | Never sent; accepted with lost response; pending retry after migration and reference rename; same UUID/schema/checksum; cross-schema UUID conflict; JSON and multipart photos; canonical working correction survives replay |
| Staging | All retained states; source checksums unchanged; projection checksum/signature; repeated migration; crash before/after DB pointer; corrupt/missing source/working file; admin correction and concurrent import |
| Import identity | D15 frozen vectors; unchanged client record IDs and fingerprint decimal/Unicode bytes; no duplicate TreeMark/TreeSample/Harvest/Observation; rejected writes roll back |
| Desktop data | Every registry family plus dynamic/on-demand/candidate/report data; full digest equals patched row shape; cold cache, stale in-flight fetch and signature mismatch; all-zero/unknown scientific values |
| Dynamic schema | Duplicate/renamed species/tractor display labels never merge columns; percentages distinct from mass; minor grouping unchanged; label rename updates metadata/signature |
| User files/UI | Current CSV goldens, headers/order/units/dates/BOM/formula protection, old aliases, round trips, all export buttons, `part:1`, natural ordering, bookmarks, stale/400 forms retain input |
| Version matrix | L client with T server; T/T+n client with T/T+n server; no T client activation on L; C rejects legacy safely; intermediate versions retain T obligations |
| Operations | Database/media/inbox consistency; restore pre-T and post-T backups; skipped-device upgrade through B; multiple installations/origins; no reset/uninstall required |

Run appropriate targeted tests per commit, then the full Python and JS suites,
Django system/migration checks and the relevant dependency/image checks before
T/C deployment. Real browser IndexedDB/service-worker tests and smoke tests on
the actual mobile browser/PWA platforms are required; mocked Node IDB tests alone
cannot prove upgrade atomicity. Exercise the current SQLite deployment; preserve
backend-independent migration behavior for the planned PostgreSQL move.

## 12. Rollout, diagnostics and cleanup gates

### 12.1 Required evidence and diagnostics

Provide a check-only server command and an explicit resumable apply mode,
provisionally `manage.py naming_transition --check|--apply|--verify`. These names
are proposed interfaces, not commands available at this baseline. Before apply,
check source evidence and report counts, conflicts, missing projections and
which mutable values need conversion. The same verifier is the cleanup gate.

Provide local Ipso migration diagnostics with APP_VERSION, naming epoch,
database version, preference status, remaining legacy-key counts, missing-ID
counts, unresolved frozen-v1-envelope count and cache schemas. A new marker
alone is not proof: inspect actual rows and metadata. Diagnostics never contain
tokens, photo bytes, operator names, observations or raw uploaded payloads.
Keep the operational report separate from the versioned schema/hash preimages.

During T/T+n, tag legacy adapter use in server logs by A-ID, protocol, endpoint
family, timestamp and outcome. Do not log bodies, bearer tokens or URL secrets.
Any application-version header is diagnostic only and confers no authority.
The manual installation ledger is authoritative for this small deployment;
an offline device absent from server logs has not thereby migrated.

### 12.2 Deploying the transition

1. Identify every server instance and every browser/profile/origin/PWA with
   production data, including temporarily disconnected devices. Record an
   explicitly retired installation only after its data disposition is verified.
2. Retain L's immutable release/artifact, verify a coordinated runtime backup
   and rehearse recovery on a separate instance. CSV alone is not a photo backup.
3. Deploy the fully tested T server before activating T clients. Run check-only
   conversion; apply under maintenance/write exclusion using existing backup
   locks; verify the database and canonical files; regenerate versioned digests.
4. Upgrade each Ipso installation through normal shell activation, close other
   tabs, verify IDB/preferences/reference/geometry/history, and perform an offline
   reload plus a recording/export/upload/retry smoke test. Retain data on failure.
5. Reload desktop clients after saving work. Verify maps, Settings, marks,
   samples, harvests, imports, audit and representative CSV exports.
6. Record real deployed versions, test results, counts and unresolved cases.
   T+n releases continue the bridge until the following gates pass.

### 12.3 Gates for cleanup C

Every gate requires dated evidence in the ledger; none is satisfied merely by
a new commit, release number or elapsed time:

- **G01:** Every known installation is verified migrated or explicitly retired
  with its data accounted for. No unexamined offline installation remains.
- **G02:** Verifiers find zero legacy mutable keys/modes/counter/preference
  namespaces and zero recoverable rows requiring name-only identity fallback.
- **G03:** No device envelope can still require a v1 retry. Accepted/lost-response,
  local-only, exported and abandoned recovery cases have been resolved explicitly.
- **G04:** Every retained staged upload has valid canonical working data and
  verified source evidence. Preview/import/download after cleanup has no need
  for a v1 source decoder. The working payload's mode matches the live model,
  its source checksum matches the accepted receipt, and both files pass their
  respective checksum checks. Old source mode spelling is intentional evidence.
- **G05:** No old desktop/PWA client remains active. The observation window has
  lasted at least the nonce TTL (currently 24 hours) after the last legacy web
  mutation and includes a normal offline/record/upload cycle for each active
  installation. It contains no unexplained legacy adapter traffic. A new legacy
  write restarts the relevant window and is investigated.
- **G06:** All current-client table/reference/geometry/history caches use schema
  2 and patches/invalidation identities are canonical. Legacy projection files
  may remain in the separate compatibility namespace until C removes them;
  they are not canonical live data. The cleanup plan stops all old generators
  and jobs before purging that namespace, so none can republish v1 output.
- **G07:** The exact bridge B, its dependency locks/runtime image/assets, this
  specification and migration instructions are retained outside the ephemeral
  running container. Restore of an old backup and upgrade of an old device
  through B have both been rehearsed.
- **G08:** All A01–A09 adapters have a recorded removal or recovery-only
  disposition. Tests prove C accepts canonical data and safely rejects legacy
  requests/storage; supported user-file aliases remain.
- **G09:** Owner/operator explicitly acknowledges the ledger and authorizes C.

No fixed number of intermediate releases or calendar deadline replaces these
gates. Log-only observations are insufficient for disconnected devices.

## 13. Cleanup contents, recovery and rollback

C removes old active request/reference/table projections, old-key fallbacks in
controllers and storage, v1 upload acceptance, legacy dataset aliases, obsolete
geometry helper shapes and boot preference copying. It prunes disposable v1
digest caches/statuses and no longer generates legacy outputs. Keep schema-2
validation, normal future migrations, human file-input adapters and historical
evidence. Add negative tests preventing accidental reintroduction of legacy
machine keys outside the explicit exceptions/recovery fixtures.

Retain B as an immutable Git tag plus executable deployment artifact with locked
dependencies and the associated migration fixtures/instructions. Record its
digests in the ledger. A source tag alone is not proof it can later be rebuilt
or run. C's operator documentation must point to that exact artifact; replace B
only after rehearsing its successor.

For a device that missed T, preserve its browser data, use the same origin and
database/profile, close C, and serve B in a controlled maintenance window on
that instance. A new PWA shell release may be needed to supersede C's cache;
simply copying old `store.js` into C is unsupported. Run B's migration/verification
and then reinstall the current complete shell. Other operators must be kept out
of that recovery window. Never ask the operator to uninstall/reset the app or
clear site data to make an upgrade work.

Serving B's frontend alone does not make C accept its pending v1 uploads. The
rehearsed recovery window needs B's compatible backend/legacy upload boundary
as well as its shell, preserving the server's existing session receipts. Do not
start an old server image against a newer database without proving compatibility
on an isolated copy. Before a later database/runtime/storage change exceeds B's
tested range, retain and rehearse a successor bridge compatible with that target.
The normal current application still contains no old-format runtime adapters;
the legacy capability is a separately retained recovery artifact.

For a pre-T server backup, restore database, media and inbox together onto an
isolated bridge-compatible instance. Run B's schema/state/projection migrations,
verify source checksums, then upgrade to current and rebuild digests. Do not feed
old SQLite state directly into C or start C before its verifier passes. Existing
historical backups stay immutable; the supported restoration route uses B.

After a phone reaches database version 10, L's version-9 client cannot be used
as a rollback reader. Prefer a fixed T+n shell and forward recovery. A server
code rollback after schema changes must use a compatible bridge; before any
snapshot rollback, account for writes and phone uploads accepted since the
snapshot. Restoring a predeployment snapshot over later work is not an automatic
rollback strategy. Do not implement a lossy reverse data migration for naming.

## 14. Documentation and context preservation

Update the ledger after every related commit/deployment with spec revision,
work package, exact SHA/tag/image, contract versions, tests, adapter changes and
outstanding gates. Update [data architecture](../data-architecture.md),
[Ipso](../ipso-abies.md), [database](../database.md), [backups](../backups.md), affected
page docs and the code review as implementation lands. Until then, their current
schema descriptions remain authoritative for the running code.

Keep the original M3 audit as historical evidence, with a link here identifying
the superseded strategy. Cleanup commits must cite the completed ledger gates
and enumerate the removed A-IDs. This specification and ledger remain in Git
after cleanup; removing adapters does not remove the explanation or recovery
route that makes the transition reproducible.
