# Ipso-Abies integration design

Status: draft (2026-06-16)

## Goal

Integrate Ipso more tightly with Abies while keeping the field workflow
offline-first.

Abies owns canonical data, reference datasets, validation, and final import
decisions. Ipso is the mobile field client: it records observations in the
woods, keeps them safe locally, syncs reference data from Abies, and uploads
completed sessions into an Abies-managed staging inbox.

## Design principles

1. **Abies owns the contract.** New Ipso-Abies JSON APIs use Abies internal
   field names and IDs directly. There is no alias layer for new Ipso data:
   `lon`, not `lng`; `species_id`, not `Specie` or `Genere`; `parcel_id`, not
   a free-text parcel name.
2. **Display labels are not data keys.** Italian labels remain UI and CSV
   labels. Machine payloads use locale-independent keys from Abies constants.
3. **Ipso stays offline-first.** The app must remain usable without network
   after reference data has been synced.
4. **Uploads are staged, not immediately imported.** Abies receives and indexes
   new Ipso data, shows a notification, and lets a user map the upload to the
   correct destination before import.
5. **Reference data is versioned.** Every Ipso session records the Abies
   reference version it used, plus any hypsometric parameter set used for
   auto-computed heights.
6. **One mobile shell, multiple modes.** Martellate, Campionamenti, and PAI
   share session management, GPS, map, IndexedDB, sync, and upload code.

## Non-goals

- Building a compatibility layer for mixed Ipso key names in the new API.
- Importing uploaded sessions directly into final Abies tables with no review.
- Making Abies depend on Ipso CSV as the primary internal format.
- Replacing Abies' existing manual CSV import flows in the first step.
- Solving per-operator Abies login on field devices. A device/token model is
  enough for the first integrated version.

## Canonical field contract

New Ipso records should store and upload Abies-native machine fields.

Common session fields:

```json
{
  "session_id": "uuid",
  "mode": "martellate",
  "schema_version": 1,
  "reference_version": "2026-06-16T08:14:22Z:42",
  "operator": "Mario Rossi",
  "created_at": "2026-06-16T07:35:00Z",
  "completed_at": "2026-06-16T11:02:00Z"
}
```

Common record fields:

```json
{
  "client_record_id": "uuid-or-local-sequence",
  "date": "2026-06-16",
  "region_id": 3,
  "parcel_id": 41,
  "species_id": 16,
  "d_cm": 42,
  "h_m": "22.4",
  "h_measured": false,
  "lat": 38.51234,
  "lon": 16.12345,
  "acc_m": 5
}
```

Notes:

- `lat`/`lon` are the canonical coordinate keys. Leaflet's `lng` appears only
  at the map library boundary and is normalized immediately.
- Species are referenced by `species_id`. Ipso may store `species_common_name`
  as a display snapshot for human recovery, but imports resolve by ID.
- Parcels and regions are referenced by ID. Names may be included as display
  snapshots, but they are not lookup keys.
- CSV exports may still use Abies CSV/display column labels, but staged upload
  should be JSON. CSV becomes an attachment/export, not the server contract.

This removes the current `Specie`/`Genere`, `Lon`/`Lng`, and
`Pino`/`Pino Nero`/`Pino Laricio` ambiguity from the integrated path. Because
there are no production deployments yet, the manual CSV import/export paths can
also be tightened as they are touched; no compatibility layer is needed for
old Ipso column names or species aliases.

## Abies reference sync

Abies exposes an Ipso reference endpoint. Ipso downloads it on first launch,
when the user taps sync, and periodically while online.

Preferred shape:

```text
GET /api/ipso/reference/manifest/
Authorization: Bearer <device-token>
```

Response:

```json
{
  "schema_version": 1,
  "reference_version": "2026-06-16T08:14:22Z:42",
  "generated_at": "2026-06-16T08:14:22Z",
  "datasets": {
    "species": {
      "version": "42",
      "url": "/api/ipso/reference/species/"
    },
    "parcels": {
      "version": "17",
      "url": "/api/ipso/reference/parcels/"
    },
    "parcel_geometry": {
      "version": "17",
      "url": "/api/ipso/reference/parcel-geometry/"
    },
    "hypso_params": {
      "version": "9",
      "url": "/api/ipso/reference/hypso-params/"
    },
    "work_packages": {
      "version": "31",
      "url": "/api/ipso/reference/work-packages/"
    }
  }
}
```

Version semantics:

- `schema_version` is the shape/meaning version of the payload for that
  endpoint. It changes only when the JSON contract changes in a way that the
  other side must understand explicitly. The reference manifest and the upload
  payload can both have `schema_version = 1`; the value is scoped to that
  payload type.
- `reference_version` is an opaque Abies content snapshot token. It changes
  whenever any reference dataset in the manifest changes, even if the JSON
  schema is unchanged. Ipso stores this on each session so Abies can later see
  which reference snapshot the operator used in the field.
- Each dataset's `version` is a smaller content token for incremental fetches.
  Ipso compares these to its local versions and downloads only changed
  datasets. The tokens may be digest `dirty_seq` values, mtimes, hashes, or any
  other Abies-owned opaque value.

Each dataset is fetched only when its version changes. Ipso writes downloads to
temporary IndexedDB stores and flips the active reference pointer only after
all required datasets for the new manifest are present.

### Species

Fields:

- `id`
- `common_name`
- `latin_name`
- `density`
- `minor`
- `sort_order`
- `active`

Ipso stores `species_id` in records. It displays `common_name`.

### Parcels and geometry

Parcel dataset fields:

- `region_id`
- `region_name`
- `parcel_id`
- `parcel_name`
- `eclass_id`
- `coppice`
- optional summary fields useful for orientation

Geometry should remain a separate dataset so the app can skip re-downloading a
large GeoJSON file when only species or hypsometry changed. Ipso uses the same
geometry for automatic parcel detection and map orientation.

### Hypsometric parameters

Ipso should pick up the active parameters from Abies Impostazioni
automatically.

Fields:

- `hypso_param_set_id`
- `activated_at`
- `region_id`
- `species_id`
- `func`
- `a`
- `b`
- `n`
- `r2`

Behavior:

- New records use the latest downloaded active parameter set.
- Existing records keep their stored `h_m`, `h_measured`, and
  `hypso_param_set_id`; they are not silently recomputed.
- If parameters change during an active session, Ipso should notify the user
  and use the new set for subsequent records after the sync completes.
- PAI mode does not auto-compute heights.

This keeps field behavior current while preserving an audit trail for how each
auto height was produced.

### Work packages

Abies should expose fieldwork packages that Ipso can select before starting a
session. They reduce later mapping ambiguity but do not eliminate server-side
review.

Initial package/context types:

- `harvest_marking`: open harvest plan items eligible for Martellate.
- `sampling_survey`: active surveys and their sample grids.
- `pai_context`: existing preserved trees and parcel/species context for PAI.

Harvest package fields:

- `work_package_id`
- `type = "harvest_marking"`
- `harvest_plan_id`
- `harvest_plan_item_id`
- `region_id`
- optional `parcel_id`
- display summary: plan name, item label, state, planned year

Sampling package fields:

- `work_package_id`
- `type = "sampling_survey"`
- `survey_id`
- `grid_id`
- sample areas: `sample_area_id`, `number`, `region_id`, `parcel_id`,
  `lat`, `lon`, `r_m`, `visited`

`visited` is true when at least one tree has been sampled in that sample area
for the survey. It is informational only. It must not block returning to the
same area later, because operators can move back and forth between sample
areas.

PAI package/reference fields:

- existing preserved trees from the `preserved_trees` digest, keyed by
  `tree_preserved.id`
- `tree_id`
- `parcel_id`
- `species_id`
- `number`
- `date`
- `estimated_birth_year`
- `d_cm`
- `h_m`
- `h_measured`
- `lat`
- `lon`
- `acc_m`
- `operator`
- `note`

PAI now has an explicit `TreePreserved` observation row. Ipso should treat that
as the PAI destination model: each imported PAI record creates a backing
`Tree(preserved=True, coppice=false)` and one `TreePreserved` row, unless a
future edit/import flow explicitly targets an existing preserved tree.

## Ipso modes

Ipso should become a shared shell with mode-specific schemas and screens.

Shared services:

- reference sync
- session creation/resume/end
- IndexedDB persistence
- GPS and accuracy handling
- map screen
- upload queue
- local CSV/JSON export
- operator/device settings

### Mode: Martellate

Purpose: field tree marking for harvest plans.

Required destination context:

- Prefer selecting a `harvest_marking` work package at session start.
- If no package is selected, Abies can still stage the upload, but import will
  require manual mapping to a harvest plan item.

Record fields:

- common fields above
- `number`
- `catastrofata` session flag or record flag, matching the final Abies model
  decision
- `h_measured`
- `hypso_param_set_id` when `h_m` was auto-computed

Height behavior:

- Auto-compute `h_m` from active hypsometric parameters when
  `(region_id, species_id)` has a parameter row and `d_cm` is present.
- User override sets `h_measured = true`.

Import target:

- `Tree` plus `TreeMark`, attached to a chosen `HarvestPlanItem`.

### Mode: Campionamenti

Purpose: record sampled trees for an Abies survey/grid.

Required destination context:

- Select a `sampling_survey` work package.
- Select or infer `sample_area_id` from map/GPS.

Record fields:

- common tree fields
- `survey_id`
- `sample_area_id`
- `number`
- `coppice`
- `shoot`
- `standard`
- `l10_mm`
- optional `pai`

Height behavior:

- Sample mode requires both `d_cm` and `h_m` to be entered. Ipso must not infer
  sample heights.

Import target:

- `Sample`, `Tree`, and `TreeSample`, with server-side validation that the
  sample area belongs to the selected survey's grid.

### Mode: PAI

Purpose: record PAI trees in the field with a workflow similar to Martellate,
but without automatic height computation.

Required destination context:

- PAI can run without a selected work package.
- If Abies later introduces region- or parcel-scoped PAI work packages, Ipso can
  use them as optional context in the same way as Martellate.

Record fields:

- common tree fields
- `number`
- `estimated_birth_year`
- `operator`
- `note`

Height behavior:

- No D->h auto-fill.
- `d_cm` and `h_m` are optional, matching the current Abies PAI form.
- If `h_m` is entered, `h_measured = true`.

Import target:

- `Tree(preserved=True, coppice=false)` plus `TreePreserved`.
- `TreePreserved.number` is unique within the parcel. If Ipso leaves it blank,
  Abies assigns the next free number for that parcel during import.
- Server-side import computes `volume_m3` and `mass_q` when enough data is
  present.

## Map screen

Every mode should have a one-click map toggle.

Behavior:

- The active form state is preserved exactly when switching to the map and
  back.
- The map is a shared screen over the current session, not a separate workflow.
- A single back button returns to the previous mode screen.

Suggested mode-screen action layout:

- Primary data-entry controls stay above the action area.
- `Visualizza dati raccolti` and `Termina e esporta` become two side-by-side
  half-width buttons.
- `Mappa` is a full-width button below them.
- Campionamenti and PAI should use the same pattern with mode-appropriate
  labels.

Shared layers:

- current GPS position and accuracy circle
- parcel/region boundaries
- current session records
- selected work package context

Mode-specific layers:

- Martellate: open harvest item geometry, already-recorded marks.
- Campionamenti: sample grid, sample areas, visited/unvisited status.
- PAI: existing PAI/preserved trees and parcel context.

Map interactions:

- Tapping a parcel can set `region_id`/`parcel_id` when the mode allows it.
- Tapping a sample area in Campionamenti sets `sample_area_id`.
- Tapping an existing local point opens a compact read-only popover with an
  edit button.

## Upload and staging inbox

Ipso should auto-upload completed sessions to an Abies-managed canonical inbox.
Upload success means "safely staged in Abies", not "imported into final domain
tables".

Endpoint:

```text
POST /api/ipso/uploads/
Authorization: Bearer <device-token>
Content-Type: application/json
X-Ipso-Session-Id: <uuid>
```

Payload:

```json
{
  "session": {
    "session_id": "uuid",
    "mode": "martellate",
    "schema_version": 1,
    "reference_version": "2026-06-16T08:14:22Z:42",
    "work_package_id": "harvest:123",
    "operator": "Mario Rossi",
    "created_at": "2026-06-16T07:35:00Z",
    "completed_at": "2026-06-16T11:02:00Z"
  },
  "records": [
    {
      "client_record_id": "1",
      "date": "2026-06-16",
      "region_id": 3,
      "parcel_id": 41,
      "species_id": 16,
      "number": 1,
      "d_cm": 42,
      "h_m": "22.4",
      "h_measured": false,
      "hypso_param_set_id": 9,
      "lat": 38.51234,
      "lon": 16.12345,
      "acc_m": 5
    }
  ]
}
```

Server behavior:

1. Authenticate the device token.
2. Validate JSON shape and canonical keys.
3. Validate that IDs exist in the current or historical Abies reference scope.
4. Compute a checksum over the normalized payload.
5. Store the payload under a canonical inbox directory.
6. Create or update an `IpsoUpload` database row for notifications/review.
7. Return idempotent success for duplicate `(session_id, checksum)`.
8. Return conflict for duplicate `session_id` with different content.

Suggested filesystem layout:

```text
/var/lib/abies/ipso-inbox/
  2026/
    06/
      <session_id>/
        upload.json
        upload.sha256
        export.csv
```

`upload.json` is the authoritative staged payload. `export.csv` is optional
and exists for operator-friendly backup/recovery.

Suggested `IpsoUpload` fields, replacing the current standalone upload metadata
sidecar:

- `session_id`
- `mode`
- `schema_version`
- `reference_version`
- `work_package_id`
- `operator`
- `record_count`
- `checksum`
- `inbox_path`
- `state`: `received`, `mapped`, `imported`, `rejected`, `conflict`
- `received_at`
- `imported_at`
- `imported_by`
- `target_type`
- `target_id`
- `error_summary`

## Abies review/import UI

Abies should add a top-level `Importazione` / `Import` tab between Mannesi and
Controllo. The nav item shows a small red dot when unreviewed `received` Ipso
uploads exist. The tab opens the Ipso inbox review page.

Inbox list:

- mode
- operator
- date/time
- record count
- suggested work package or target
- state
- import/reject actions

Review page:

- session metadata
- map preview
- table preview
- validation warnings
- target selector
- import button

Mapping rules:

- Martellate: user selects a `HarvestPlanItem`. If Ipso supplied a
  `harvest_plan_item_id` through a work package, preselect it but still let the
  user change it before import.
- Campionamenti: user selects/confirms `Survey`; records already carry
  `sample_area_id`.
- PAI: no work package is required. User reviews the staged records; Abies
  imports them as `TreePreserved` rows after parcel/species validation.

Import should call shared server-side cores, not duplicate view logic:

- `csv_marks.py` or better `ipso_marks.py` for Martellate.
- existing Campionamenti parse/apply logic extended to accept staged JSON.
- a new PAI import core that writes `Tree(preserved=True, coppice=false)` plus
  `TreePreserved`.

After successful import:

- mark the upload `imported`
- store target type/id
- mark affected Abies digests stale
- keep raw inbox files for 30 days after import, then purge the files while
  retaining the `IpsoUpload` database row
- make repeated import impossible without an explicit admin reset

## Session and local storage model

Ipso IndexedDB should move from CSV-shaped records to Abies-shaped records.

Stores:

- `sessions`
- `records`
- `reference_manifests`
- `reference_datasets`
- `upload_attempts`

Session fields:

- `session_id`
- `mode`
- `work_package_id` (nullable)
- `reference_version`
- `operator`
- `status`: `active`, `completed`, `uploaded`, `upload_failed`
- timestamps

Record fields:

- `session_id`
- `client_record_id`
- canonical Abies fields
- mode-specific fields
- `created_at`
- `updated_at`

Migration note:

- A one-time local IndexedDB migration may be needed to preserve any active
  sessions on operators' phones. That is a local data-safety migration, not a
  server-side compatibility layer. The new upload API should remain strict.

## Authentication and deployment

Initial version:

- device/shared bearer token for `/api/ipso/reference/*` and
  `/api/ipso/uploads/`
- HTTPS only
- request size limit on uploads
- idempotent upload by `session_id` and checksum

Deployment decision:

- transition to same-origin immediately
- serve Ipso from the Abies origin under an `/ipso/` path, so service worker,
  uploads, and reference sync are same-origin
- use a separate origin only for local development or a very short deployment
  bridge, not as the target architecture

Source-code location:

- Recommended target: move Ipso source under Abies once Phase 1 starts, likely
  as an `apps/ipso/` Django app owning the PWA static files, service worker,
  reference-sync client code, and Ipso API endpoints.
- Keeping a separate Ipso repo while Abies owns the API contracts would preserve
  the current cross-repo build friction. It is acceptable only as a temporary
  transition while the first same-origin build is being wired.

Later:

- per-device tokens
- device names in the inbox UI
- token revocation from Abies Impostazioni
- optional Abies user binding for operators

## Implementation phases

### Phase 1: Contract cleanup

- Switch Ipso's internal records to Abies canonical field names and IDs.
- Store `species_id`, `region_id`, `parcel_id`, `lat`, `lon`.
- Keep CSV export as a secondary output.
- Extract the current Abies mark CSV importer into a reusable import core.
- Fix the current `Specie`/`Genere` mismatch while doing the extraction, but
  keep the new JSON upload path strict.

### Phase 2: Reference sync

- Add Abies Ipso reference manifest and dataset endpoints.
- Include active hypsometric parameters from Impostazioni.
- Add Ipso reference downloader and atomic IndexedDB replacement.
- Record `reference_version` and `hypso_param_set_id` in sessions/records.

### Phase 3: Staged upload for Martellate

- Add `/api/ipso/uploads/`.
- Store JSON payloads in the canonical inbox directory.
- Add `IpsoUpload` model and notification.
- Add inbox review/import UI for Martellate.
- Import into selected `HarvestPlanItem` through the shared import core.

### Phase 4: Shared map and mode shell

- Split Ipso into shared shell plus mode modules.
- Add the one-click map toggle.
- Move parcel detection/map context into the shared layer.

### Phase 5: Campionamenti and PAI

- Add sampling work packages to the reference bundle.
- Add Campionamenti mode and staged import.
- Add PAI reference context, mode, and staged import using `TreePreserved`.

## Resolved review decisions

- Work-package selection is optional. Ipso should allow unassigned sessions;
  Abies maps them during staged import review.
- A hypsometric parameter update downloaded during an active session takes
  effect only after that session is closed. The active session keeps using the
  reference snapshot it started with.
- PAI imports target `TreePreserved` plus its backing `Tree`.
- Operators are free-text values stored on the device/session. Operators do not
  need Abies user accounts.
- Raw inbox files are retained for 30 days after import. The database upload
  record remains after file purge.
- Ipso does not need to poll Abies for post-import status. From the phone's
  point of view, `uploaded` means safely staged in Abies.
- Ipso moves to the Abies origin immediately.
- Ipso source moves under Abies as `apps/ipso/`.
- The existing Ipso repository stays around for now as a transition/history
  repo.

## Immediate next implementation decisions

The remaining high-leverage implementation decisions are:

1. Define the first strict JSON upload schema for Martellate, including required
   vs optional fields and the normalized checksum input.
2. Define the first reference manifest generator and dataset version tokens.
