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

Protected reference endpoints require the shared bearer:

- `/ipso/reference.json`
- `/ipso/terreni.geojson`

`reference.json` contains the current Abies reference bundle used by Ipso:

- active species, including canonical Abies species IDs;
- parcels, regions, parcel IDs, and coppice flags;
- active hypsometric parameters;
- sampling surveys, sample grids, sample areas, and existing max tree numbers;
- PAI preserved-tree context;
- work-package options used by Ipso modes;
- a derived `reference_version` hash.

`terreni.geojson` contains parcel geometry for GPS-driven orientation and parcel
selection in the mobile app.

Ipso downloads these on boot after it has a bearer. It also stores field sessions
and trees in IndexedDB for offline operation.

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
Content-Type: application/json
```

The upload body contains:

- `session`: UUID, mode, schema version, reference version, work package,
  operator, timestamps, region, and damage flag;
- `records`: canonical Abies IDs and measurements for the mode;
- optional `csv_text`: the local CSV text for operator/audit recovery.

Supported modes are:

- `martellate`
- `samples`
- `pai`

The unauthenticated upload endpoint validates size, schema, session UUID,
record count, field types, known species/parcels/sample areas/hypsometric sets,
and mode-specific invariants. It stages accepted uploads in `IpsoUpload` and
writes `upload.json`, `upload.sha256`, and optional `export.csv` under
`ABIES_IPSO_INBOX_DIR`.

Uploading does not directly mutate forestry records. It creates a staged inbox
item. A logged-in Abies user can view upload metadata and previews. Import or
rejection requires writer permission, and the import endpoints perform the final
mode-specific validation against the selected target.

Duplicate uploads with the same session ID and checksum are idempotent. A second
upload with the same session ID but different content marks the staged upload as
conflicted.

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
- `POST /api/ipso/uploads/<id>/import-pai/`

The shared secret prevents casual unauthenticated reads/uploads. It does not
prove which device or operator made a request, and compromise of the secret gives
access to all bearer-protected Ipso device endpoints until the secret is rotated.

## Rate limiting and abuse controls

The upload endpoint has application-level controls:

- bearer check;
- request size cap: `ABIES_IPSO_UPLOAD_MAX_BYTES`;
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
