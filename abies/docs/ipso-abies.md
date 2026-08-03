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

Protected reference endpoints require the shared bearer:

- `/ipso/reference.json`
- `/ipso/terreni.geojson`

`reference.json` contains the current Abies reference bundle used by Ipso:

- active species, including canonical Abies species IDs;
- parcels, regions, parcel IDs, and coppice flags;
- active hypsometric parameters;
- sampling surveys, sample grids, sample areas, and existing max tree numbers;
- PAI preserved-tree context;
- active observation categories;
- work-package options used by Ipso modes;
- a derived `reference_version` hash.

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

IndexedDB schema v7 captures canonical identity at the moment data is recorded:
the session stores its region and selected survey IDs, and each observation
stores region, parcel, species, and (for sampling) sample-area IDs. Display names
remain alongside them solely for the operator UI and CSV export. Upload payloads
use the captured IDs directly, so later renames, deactivation, or name reuse do
not remap an observation. Pre-v7 sessions remain uploadable through a legacy
name-resolution fallback against their saved/current reference bundle.

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
  `photo:<client_photo_id>`. Observation location comes from Ipso's device GPS.
  Camera-captured photos also include the device GPS snapshot captured just
  before the camera opens; gallery photos stay location-free, and photo EXIF
  GPS is ignored because mobile providers commonly strip it.

Supported modes are:

- `martellate` — marked trees for a harvest-plan item;
- `samples` — predefined/grid-based tree surveys;
- `free_survey` — free/unstructured tree surveys;
- `observations` — point observations with text, categories, GPS, and
  optional photos. The recording form has separate controls for selecting
  existing gallery photos and opening the camera. Ipso ignores photo EXIF GPS;
  the observation position is the device GPS fix captured while the
  observation is recorded, and camera photos carry the pre-camera device GPS
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
region for region-wide items. Predefined sample rows must fit the selected
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
  sample UI does not record multiple shoots for one number.
- Free-survey ordinary rows may omit `number`. On import, Abies assigns the
  next sample-local number within the selected unstructured survey. If a free
  row supplies a number, it must be positive and unused in that survey.
  Free-survey rows for preserved trees require a submitted positive number;
  Abies stores that value as the parcel-scoped `preserved_number` and assigns
  the row's sample-local `number` separately.

If `reference.json` is stale, Ipso may propose an outdated next number. Import
code must still preserve the submitted number exactly, and mode-specific server
validation must reject invalid or conflicting rows rather than silently
auto-renumbering them.

## Authorization boundaries

Device endpoints using the shared bearer:

- `GET /ipso/reference.json`
- `GET /ipso/terreni.geojson`
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
