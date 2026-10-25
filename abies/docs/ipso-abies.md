# Ipso-Abies integration

Ipso is the mobile field PWA served by Abies under `/ipso/`. It is designed for
trusted crews using operator-managed devices, with a shared secret to prevent
trivial public abuse of the unauthenticated device endpoints.

This is not a per-device or per-user authentication scheme. A device that holds
the shared secret can read the Ipso reference data and stage uploads. Final import
into Abies still requires an authenticated Abies user with writer permission.

## Provisioning and secret

Production deployments set one secret:

- `ABIES_IPSO_SECRET`: shared bearer used by the Ipso PWA for reference downloads
  and staged uploads.

A new device is provisioned by opening:

```text
https://abies.laforesta.it/ipso/#secret=<ABIES_IPSO_SECRET>
```

The fragment is not sent to the server as part of the initial page request. The
PWA reads it in the browser, stores it in `localStorage` under
`ipso.bearer_token`, clears the fragment from the address bar, and then sends it
as:

```http
Authorization: Bearer <secret>
```

Existing installed clients keep using the stored `ipso.bearer_token`. To migrate
from the previous two-token deployment, set `ABIES_IPSO_SECRET` to the old
`ABIES_IPSO_UPLOAD_TOKEN` value if those devices should continue working.

Rotating `ABIES_IPSO_SECRET` revokes all Ipso devices at once. There is currently
no individual device enrollment, revocation list, audit identity, or token
expiry. If individual revocation becomes important, the next model should be a
server-side device credential table, not another global bootstrap secret.

## Abies -> Ipso data

The shell assets are public:

- `/ipso/`
- `/ipso/index.html`
- `/ipso/*.js`, CSS, manifest, and image assets listed by `apps.ipso.views`

Ipso's Italian UI catalog is `apps/ipso/static/ipso/pwa/strings_it.js`, served
as `/ipso/strings_it.js` and loaded before the classic scripts that use the
global `S` object. Node-test imports use the same catalog. A future language
uses a parallel `strings_<lang>.js` file and retargets the shell, asset allowlist,
precache and test-side imports. This is separate from the shared Abies ES-module
catalog: `/static/base/js/strings.js` remains its locale-neutral re-export,
and both that file and its `strings_it.js` dependency are precached for maps.

The service worker caches only shell/static assets. Protected data responses use
`Cache-Control: no-store`, and the service worker bypasses `no-store` requests.
The PWA instead keeps explicit, application-owned last-good snapshots of the
validated reference bundle and parcel features in its IndexedDB `meta` store.
This avoids putting bearer-protected responses in a shared HTTP cache while
still supporting an offline cold start.

Shell updates are versioned by `APP_VERSION` in `version.js`. The page registers
the service worker with `updateViaCache: 'none'` and asks it to check for
updates on boot and whenever the installed app returns to the foreground. A
newly installed worker remains in `waiting` until either the app is fully closed
or the operator presses the footer `Aggiorna app` button, which is shown only
when a waiting worker exists. Pressing the button sends the worker an explicit
activation message and reloads the page once the new worker controls it, keeping
version switches visible and operator-driven.

Ipso 1.5.1 renames the mobile catalog from `strings.js` to `strings_it.js` and
updates its loading/precache references. The version bump gives the new shell
its own cache; existing installed clients keep their cached old shell until
normal operator-driven activation. No IndexedDB schema or session data changes.

Ipso 1.5.2 hardens recording against overlapping saves and closing-session
races. The shared save handler acquires an in-progress guard before any await;
input events cannot re-enable Save while the operation runs. The short cooldown
starts after the durable insert. A successful insert consumes the form before
preferences, session refresh, map refresh and periodic backup. Those follow-ups
run independently: a failure is reported explicitly as **data saved**, not a
failed write inviting another insert. A failed insert retains the form for retry.
The same rules apply to trees and observations.

Protected data endpoints require the shared bearer:

- `/ipso/reference.json`
- `/ipso/terreni.geojson`
- `/api/ipso/history/<mark|survey>/<id>/`

`reference.json` contains the current Abies reference bundle used by Ipso:

- active species, including canonical Abies species IDs and a `selectable`
  flag. Historical `Altro` stays in the reference metadata even if deactivated,
  but is not selectable;
- parcels, regions, parcel IDs, and coppice-management flags;
- active hypsometric parameters;
- sampling surveys, sample grids, sample areas, and existing max tree numbers;
- PAI preserved-tree context;
- active observation categories;
- work-package options used by Ipso modes;
- lightweight `history` metadata (kind, ID, year, label, tree count, and
  detail URL) for marks and surveys containing trees;
- a derived `reference_version` hash.

The history metadata is deliberately excluded from `reference_version` because
it is read-only orientation data, not field-session recording configuration.
Tree coordinates and measurements are never included in `reference.json`.

`terreni.geojson` contains parcel geometry for GPS-driven orientation and parcel
selection in the mobile app.

On boot, Ipso opens IndexedDB before making either protected-data request. It
restores the last-good reference and parcel-feature snapshots and lists locally
resumable sessions immediately. Completed, exported, or abandoned local sessions
remain stored on the device but do not open a blocking startup prompt. When a
bearer is available, Ipso refreshes both resources opportunistically in the
background; only successfully parsed and shape-validated responses replace the
IndexedDB snapshots. A failed refresh leaves the snapshots in use and displays a
persistent warning that recent Abies changes may be unavailable.

A newly provisioned device still needs one successful online reference download
before it can start its first field session. After that, a reload, browser-process
eviction, phone restart, or loss of connectivity does not hide open or
pending-upload sessions. If no valid reference snapshot is available, existing
open sessions remain exportable and pending uploads remain retryable or
local-only, but recording cannot resume until reference data is available. Field
sessions and trees remain in IndexedDB throughout.

### Species catch-all transition

Ipso 1.5.0 and Abies migration 0026 ship together. New-tree species pulldowns
use `Altra conifera` and `Altra latifoglia` instead of Altro in every recording
mode. The client honors `selectable=false` and also excludes Altro by name
when using an older cached reference bundle or a legacy sticky default. It
does not alter saved sessions or observations. Previously recorded Altro trees
remain displayable, exportable, and uploadable by their original species ID;
server staging/import continues accepting historical IDs.

On import, Abies estimates volume for the new choices using the Abete bianco
and Faggio Tabacchi equations, respectively, and mass using the catch-all's
current density. These aliases do not change the species ID, stored field
measurements, or height auto-population. Historical Altro remains unsupported
by Tabacchi. Migration 0027 fills missing volume/mass pairs for earlier
high-forest observations of the two new choices with positive D/h, without
overwriting existing estimates.

After deployment, connected devices must fetch the refreshed references and
activate the new shell with `Aggiorna app` (or fully close/reopen after the
update downloads). A disconnected device cannot receive new species IDs until
it reconnects; failed refreshes retain the last-good snapshot and warning.

## Storico

`Storico` is a top-level Ipso screen for locating previously marked or measured
trees. It lists only aggregates that contain at least one tree; empty harvest
plan items and empty surveys are not historical field work yet.

A mark is the existing Abies aggregate of all `TreeMark` rows linked to one
`HarvestPlanItem`. Its year is `HarvestPlanItem.year_planned`, and its label is:

```text
Martellata: <harvest plan> <region>/<parcel> <year> (<count> alberi)
```

For a region-wide plan item, where no single parcel exists, the scope is the
region name alone. A survey includes all `TreeSample` rows below all of its
`Sample` rows. Its sort year is the year of its latest sample date, and its label
is:

```text
Rilevamento: <survey name> (<count> alberi)
```

The combined list is sorted by descending year. Marks precede surveys when the
year is equal; labels provide the stable order within the same kind and year.
Both structured and unstructured surveys are eligible, including inactive
surveys, as long as they contain measurements.

Selecting a row requests its `detail_url` with the shared bearer. The endpoint
returns only the fields needed by the field map: common species name and
canonical species ID, diameter, height, latitude, and longitude. It also returns
both the total tree count and `mapped_tree_count`. The total includes legacy
records without coordinates; those records cannot produce map dots.

The history map reuses the standard Ipso orientation map, including parcel
geometry, basemap chooser, `Indietro`, `Centra`, live GPS position, and heading
arrow. It initially fits the selected tree coordinates. Historical trees use
standard green dots; tapping one opens species, diameter in centimetres, and
height in metres. `Centra` retains its normal meaning and returns to the current
GPS position when a fix is available.

History detail responses use `Cache-Control: no-store`, so the service worker
does not retain bearer-protected data. After shape and identity validation, the
PWA writes the response to an application-owned IndexedDB `meta` row keyed by
history kind and ID. A row is usable for one hour from its successful fetch.
At or after the one-hour boundary it is treated as expired and a network request
is required; an expired response is not used as an offline fallback. Failed or
invalid responses never replace a cached row.

## Local IndexedDB schema

IndexedDB schema v9 captures canonical identity at the moment data is recorded:
the session stores its region and selected survey IDs, and each observation
stores region, parcel, species, and (for sampling) sample-area IDs. Display names
remain alongside them solely for the operator UI and CSV export. Upload payloads
use the captured IDs directly, so later renames, deactivation, or name reuse do
not remap an observation. Pre-v7 sessions remain uploadable through a legacy
name-resolution fallback against their saved/current reference bundle. Schema
v9 also formalizes the timestamped history-detail rows in the existing `meta`
store; it does not alter saved session or tree records.

## Ipso -> Abies data

When an operator ends a non-empty Ipso session, the PWA first downloads a local
CSV backup to the phone. It then posts a canonical JSON payload to:

```text
POST /api/ipso/uploads/
```

The request must include:

```http
Authorization: Bearer <secret>
X-Ipso-Session-Id: <uuid>
Content-Type: application/json or multipart/form-data
```

The upload body contains:

- `session`: UUID, mode, schema version, reference version, work package,
  operator, timestamps, region, and damage flag;
- `records`: canonical Abies IDs and measurements for the mode;
- optional `csv_text`: the local CSV text for operator/audit recovery;
- for observation uploads with photos, multipart file parts named
  `photo:<client_photo_id>`. Observation location comes from the device GPS
  snapshot captured when the observation form opens, with save-time GPS used
  only if no initial fix was available. Camera-captured photos also include
  the device GPS snapshot captured just
  before the camera opens; gallery photos stay location-free, and photo EXIF
  GPS is ignored because mobile providers commonly strip it.

Every tree save in `martellate`, `samples`, and `free_survey` requires a fresh
device GPS fix. Ipso stores and uploads that fix's latitude, longitude, and
browser-reported `acc_m`, rounded to whole metres. The Abies accuracy fields
remain nullable so historical and non-Ipso imports without accuracy continue
to load.

Ipso currently records every tree as high forest (`coppice=false`). Parcel
economic management never determines individual-tree morphology:
coppice-managed parcels may contain high-forest trees. The parcel and sample-
area coppice flags remain in `reference.json` as management metadata, but the
recording path does not use them to classify a tree. Coppice sampling must be
introduced later as an explicit recording workflow rather than inferred from
the parcel.

Supported modes are:

- `martellate` — marked trees for a harvest-plan item. Height is optional in
  this mode: when no active (region, species) equation is available and the
  operator has no measurement, Ipso records `h_m=null` and
  `h_measured=false`. Abies preserves that state and leaves volume and mass
  null;
- `samples` — predefined/grid-based tree surveys;
- `free_survey` — free/unstructured tree surveys;
- `observations` — point observations with text, categories, GPS, and
  optional photos. The recording form has separate controls for selecting
  existing gallery photos and opening the camera. Ipso ignores photo EXIF GPS;
  the observation position is the initial device GPS fix captured when the
  observation form opens, and camera photos carry the pre-camera device GPS
  snapshot as photo metadata. The photo list displays filename,
  original/upload size, and `ok`/`errore` conversion status.

The unauthenticated upload endpoint validates size, schema, session UUID,
record count, field types, known species/parcels/sample areas/hypsometric sets,
and mode-specific invariants. Observation uploads also validate category IDs
and photo checksums. It stages accepted uploads in `IpsoUpload` and writes
`upload.json`, `upload.sha256`, optional `export.csv`, and optional staged
photo files under `ABIES_IPSO_INBOX_DIR`.

Uploading does not directly mutate forestry records. It creates a staged inbox
item. A logged-in Abies user can view upload metadata and previews. Import or
rejection requires writer permission, and the import endpoints perform the final
mode-specific validation against the selected target. Martellate rows must fit
the selected harvest-plan item scope: exact parcel for parcel items, or same
region for region-wide items. A Martellate row may omit height only when
`h_measured=false`; other tree-survey modes continue to require a positive
height. Predefined sample rows must fit the selected
survey grid; if the session records the survey chosen in Ipso, selecting
another survey is only accepted when it uses the same grid. Free-survey rows
must be imported into an unstructured survey, where the import creates one
`Sample(sample_area=NULL)` for the session. Observation imports do not use a
target selector; they create `Observation` rows, category assignments, and
photo metadata/files directly from the staged session.

For `samples` and `free_survey` imports, validation errors block immediately.
After validation and before writing, Abies computes non-blocking warnings for
rows whose GPS coordinates fall in a different parcel from the submitted
parcel, and for rows with `h_measured=false`. The import page shows these
warnings in a standard proceed/abort modal. Proceeding resubmits with explicit
confirmation; the server recomputes warnings and then writes. The submitted
parcel remains authoritative because GPS can be wrong under tree cover.

Duplicate uploads with the same session ID and checksum are idempotent. A second
upload with the same session ID but different content marks the staged upload as
conflicted.

Authenticated inbox actions use the web API's nonce protocol (see
[data architecture](data-architecture.md)). Forestry changes, import state,
digest invalidations and the completed response receipt commit together.
Staged files and import rows are parsed before claiming the nonce/acquiring
the write lock; the transaction rechecks the upload's state/mode/checksum and
the selected target's version before applying those rows. A changed context
requires reloading and trying again. Rejected imports retain their diagnostic
message separately after rollback, but do not consume the nonce or keep partial
forestry writes. A diagnostic cannot overwrite an upload concurrently imported,
rejected, deleted or edited.

Inbox deletion removes staged files only after the database commit. Mode edits
restore the previous staged payload/checksum if their nonce-backed transaction
rolls back normally. This is not a filesystem crash-recovery protocol: a process
or host interruption during a file rewrite can still require reconciliation,
and interrupted creation of observation photos can leave unreferenced files.

## Tree-number contract

Tree numbers are required for predefined sampled trees and free-survey rows
for preserved trees. They are optional for marks and ordinary free-survey rows.
Abies may propose a next number in the UI. Import code preserves supplied
numbers except for ordinary free-survey rows without one, where Abies assigns
the next sample-local number.

- Martellate marks may have `number = null`. This represents a tree that was
  recorded but not physically numbered. Ipso permits the operator to clear the
  proposed number, and Abies stores/imports that as `TreeMark.number = NULL`.
- Predefined sampled trees must have a positive integer. Ipso requires it
  before save/upload, rejects duplicates in the same sample area for the
  current upload, and checks the current Abies max for that survey/sample
  area. The Abies upload/import paths recheck that the value is present and
  that the sample area belongs to the selected survey/parcel. They do not
  impose a plain `(sample area, number)` uniqueness rule because coppice shoots
  are represented in Abies as separate rows by `shoot`; the current Ipso
  sample UI records only high-forest trees and does not record shoots.
- Free-survey ordinary rows may omit `number`. On import, Abies assigns the
  next number after the explicit ordinary-tree numbers reserved by that upload.
  If a free row supplies a number, it must be positive and unused within that
  uploaded session/sample; another sample in the same survey may reuse it.
  Free-survey rows for preserved trees require a submitted positive number;
  Abies stores that value as the parcel-scoped `preserved_number` and assigns
  the row's sample-local `number` separately.

If `reference.json` is stale, Ipso may propose an outdated next number. Import
code must still preserve the submitted number exactly, and mode-specific server
validation must reject invalid or conflicting rows rather than silently
auto-renumbering them.

The UI preflight and IndexedDB append transaction share one number-scope rule.
The transaction rechecks uniqueness while holding the sessions/trees write
scope, so another tab cannot bypass the preflight. Blank numbers remain valid
where the mode allows them. Appends and deletes require an open session;
pending-upload, exported and abandoned records cannot be changed by a stale
recording tab. Freezing an upload checks its record identities and count against
the current transaction snapshot. If another tab changed the rows while CSV or
payload preparation ran, closing fails and leaves the session open for a fresh
export; it never freezes a payload silently missing those changes. No IndexedDB
schema migration or historical-row rewrite is required. Local-only CSV export
checks the same snapshot before marking an open session exported.

## Authorization boundaries

Device endpoints using the shared bearer:

- `GET /ipso/reference.json`
- `GET /ipso/terreni.geojson`
- `GET /api/ipso/history/<mark|survey>/<id>/`
- `POST /api/ipso/uploads/`

Abies session login required:

- `GET /api/ipso/inbox/`
- `GET /api/ipso/uploads/<id>/`

Abies writer permission required:

- `POST /api/ipso/uploads/<id>/reject/`
- `POST /api/ipso/uploads/<id>/import-martellate/`
- `POST /api/ipso/uploads/<id>/import-samples/`
- `POST /api/ipso/uploads/<id>/import-free-survey/`
- `POST /api/ipso/uploads/<id>/import-observations/`

Abies admin permission required:

- `GET /api/ipso/uploads/<id>/download/`
- `POST /api/ipso/uploads/<id>/delete/`
- `POST /api/ipso/uploads/<id>/mode/`

The shared secret prevents casual unauthenticated reads/uploads. It does not
prove which device or operator made a request, and compromise of the secret gives
access to all bearer-protected Ipso device endpoints until the secret is rotated.

## Rate limiting and abuse controls

The upload endpoint has application-level controls:

- bearer check;
- request size cap: `ABIES_IPSO_UPLOAD_MAX_BYTES`, default 30 MiB;
- record count cap: `ABIES_IPSO_UPLOAD_MAX_RECORDS`;
- in-memory rate limit: `ABIES_IPSO_UPLOAD_RATE_LIMIT` per
  `ABIES_IPSO_UPLOAD_RATE_WINDOW_S`;
- strict payload validation before staging.

The application rate-limit key uses Django `REMOTE_ADDR` by default. When the
request comes from a configured trusted proxy, Abies instead uses the first
address in `X-Forwarded-For`. Configure trusted proxy networks with
`ABIES_IPSO_UPLOAD_TRUSTED_PROXIES`; the deployment default covers loopback and
common Docker bridge networks.

Apache is the public edge and must sanitize forwarding headers before proxying
to Django. Do not let a public client choose its own `X-Forwarded-For`;
clients can spoof that header unless Apache overwrites or removes it.

Because `_UPLOAD_ATTEMPTS` is in-process memory, the app-level limit is per
Django process and resets on restart. Treat it as a modest backstop. Apache-side
controls can still be added for stronger DoS protection, but Apache
`mod_ratelimit` is bandwidth throttling, not request-count rate limiting.
