# Abies

Abies is an application for forest management and operations. It supports forest
surveys, harvest planning, tracking of harvesting operations, and management
of lumber crews. It has an information-rich UI based on interactive maps and
spreadsheet-like tables, and provides various JSON API endpoints. The UI is
currently available only in Italian, but makes localization easy thanks to
string constants and language-specific templates. It is written using Django and
pure Javascript.

Operators work from the browser. A new instance can start with an empty map,
an imported parcel map, or a canonical forestry data bundle.

The UI has several sections:
- `Bosco` (`Forest`) is a map-based view that describes forest state using both
  field measurements and satellite imagery;
- `Piano di taglio` (`Harvest plans`) manages harvest plans, the lifecycle of
  harvesting interventions, and the tracking of trees marked for harvesting;
- `Campionamenti` (`Samples`) supports field measurements, including creation of
  sample grids and recording of sampled trees;
- `Prelievi` (`Harvests`) records completed harvests at the granularity of a
  single crew-day;
- `Squadre` (`Teams`) tracks lumber teams and related accounting;
- `Controllo` (`Audit`) is a public audit log of major state changes;
- `Impostazioni` (`Settings`) contains reference data and app-wide controls.

Ipso is a mobile web app that pairs with Abies to support fieldwork, including
harvest preparation and forest inventory. It is a progressive web app served at
`/ipso/` relative to the Abies root. Field devices use a shared bearer secret to
download reference data and stage uploads. Ipso uploads do not directly mutate
forestry records in Abies: they land in the Abies Ipso inbox and must be
imported or rejected by an authenticated writer.

Authentication supports local username/password accounts and Microsoft Entra
OAuth through django-allauth. Users have one of three roles: `reader`, `writer`,
or `admin`. Domain writes are tracked with django-simple-history except for a
small set of high-volume observation tables documented in the audit contract.

Production deployments run Abies as a Docker container. Apache terminates TLS,
serves collected static files directly, and reverse-proxies application traffic
to gunicorn bound on localhost. Runtime state lives outside the image under a
bind-mounted `data/` directory: SQLite database, digests, geodata, Ipso inbox,
and optional canonical bootstrap data. Digest files are stored as `.json.gz`
and served with `Content-Encoding: gzip`. Static files are host-mounted too; a
separate backup mount is used for compressed SQLite and Ipso-inbox backup
archives, including pre-deploy snapshots and scheduled backups.

For a fresh instance with an optional parcel map, follow [INSTALL.md](INSTALL.md).

## More Documentation

- [docs/deployment.md](docs/deployment.md) covers hosted provisioning and
  the first deployment.
- [docs/laforesta.md](docs/laforesta.md) covers the La Foresta installation.
- [docs/bootstrap.md](docs/bootstrap.md): canonical CSV bundle and bootstrap
  contract.
- [docs/backups.md](docs/backups.md): SQLite/Ipso backup, retention, scheduling,
  OneDrive mirroring, and restore notes.
- [docs/database.md](docs/database.md): relational model and invariants.
- [docs/security.md](docs/security.md): roles, authentication, OAuth, CSP,
  audit, and Ipso security model.
- [docs/ipso-abies.md](docs/ipso-abies.md): mobile PWA data flows, shared secret,
  upload staging, authorization, and rate limiting.
- [docs/data-architecture.md](docs/data-architecture.md): digest files and data
  loading strategy.
- [docs/ui-architecture.md](docs/ui-architecture.md),
  [docs/ui-design-patterns.md](docs/ui-design-patterns.md), and
  [docs/ui-maps.md](docs/ui-maps.md): frontend structure and UI conventions.
- `docs/page-*.md`: page-specific behavior for Bosco, Campionamenti, Controllo,
  Impostazioni, Importazione, Login, Piano di taglio, Prelievi, and Squadre.

## Local Development

For first-time setup, follow the [local installation guide](INSTALL.md#fresh-local-instance)
and use the development lock, `requirements-dev.txt`. On an existing checkout,
run `make local` after pulling changes that add migrations or templates.

To deliberately rebuild from the canonical bundle, run `make stop-local` and
invoke `make reset-local` directly. It replaces the local SQLite database and
Ipso inbox, loads the catalog and optional map from `data/canonical`, stages
converted mark uploads, regenerates digests, and creates the admin. Run
`make help` for the main targets.

Local files live under the checkout's gitignored `data/` directory.
`data/db.sqlite3` is the database, `data/canonical/` contains the bootstrap
bundle, `data/ipso-inbox/` holds field uploads, and `data/observation-media/`
holds observation photos. Generated files are in `data/digests/`,
`data/observation-thumbnails/`, `data/satellite/` and `data/map-context/`.
Local environment settings come from `compose/.env.local` and collected
static files go to `staticfiles/`.

To test with current production records, run `make stop-local`, then
`make mirror-prod-to-local`. It uses the configured
`ABIES_DOCKER_CONTEXT` to take a live snapshot of the production SQLite
database, observation photos and Ipso inbox. The command asks before replacing
local data (`MIRROR_ARGS=--yes` confirms in a script). It applies local migrations
and regenerates digests before replacing local state. On older deployments,
it also fetches the parcel map so migrations can import it. The sync keeps local
Copernicus credentials and writes the previous runtime state to
`data/.pre-sync-*`. Copied sessions are cleared. The snapshot contains real
production records and photos, so keep it on an access-controlled machine.
The terrain worker prepares generated terrain and print-map data locally.
Start the app again with `make run-local` and sign in with an account from
the snapshot.

Follow the [satellite setup instructions](INSTALL.md#satellite-imagery) to
enable live downloads for Bosco. The terrain worker can reuse compatible
rasters under `data/satellite/<map_key>/` during the first refresh.
[Terrain refresh and publication](docs/satellite-refresh.md)

The terrain worker also builds the roads, water and contours for Bosco's
printable maps. It refreshes them after published parcel boundaries change.
**Aggiorna dati dei terreni** in Cartografia can queue a refresh immediately.
Print-map context uses public OpenStreetMap and Copernicus GLO-30 data and
requires no Copernicus credentials. Bosco offers the parcel-only map while
matching context is unavailable. `ABIES_MAP_CONTEXT_DIR` configures the
worker's output and Bosco's serving path.

Run tests with the same interpreter used to install the development lock.

```sh
make test PYTHON=.venv/bin/python
```

The four default skips are two Cartografia browser tests, each run at desktop
and phone widths. They require Playwright and Chromium and are enabled by
setting `ABIES_PLAYWRIGHT_MODULE`. [Browser test instructions](docs/page-impostazioni.md#testing-the-road-workflow)

To test Microsoft Entra OAuth, also set these in `compose/.env.local`:
```sh
MS_OAUTH_TENANT
MS_OAUTH_CLIENT_ID
MS_OAUTH_SECRET
```

To test Ipso data uploads, also set `ABIES_IPSO_SECRET` there.

Start the development server and terrain worker together with `make run-local`.

`make deploy-local` syncs branding, runs `make local`, and then starts both
processes with `make run-local`. With `lsof` available, `make run-local` replaces
an Abies development server listening on the chosen port from the same checkout.
If the previous launcher has a terrain worker, both processes stop together.
If another checkout or application owns port 8000, use
`make run-local LOCAL_PORT=8001` and open `http://127.0.0.1:8001/`.
`make stop-local` uses `ps` and `lsof` to stop local Django servers and terrain
workers owned by you in this checkout, including standalone processes and
servers started on custom ports.

### Manual updates to map data

The terrain worker handles routine refreshes. These commands are for
standalone builds, diagnostics and custom data sources.

`make satellite SATELLITE_DIR=data/satellite-build` builds a standalone
copy using the credentials from `.env.satellite`. The legacy
`sync-satellite-dev` and `sync-satellite-prod` targets copy standalone rasters
from `SATELLITE_DIR` to a hosted instance and preserve worker-published
satellite revisions. The legacy `sync-data` target also copies canonical inputs.

`make map-context` builds print-map context separately. To rebuild just one
compresa, pass its name in `MAP_CONTEXT_ARGS`.

```sh
make map-context
make map-context MAP_CONTEXT_ARGS='--region Valverde --force'
```

Each compresa uses its parcel bounding box plus a 500 m buffer. The builder
extracts contours every 20 m, clips and simplifies vectors, then atomically
publishes `data/map-context/<map_key>/context.json.gz`. It reuses matching
builds and keeps the last complete snapshot on failure. Use `--force` to
refresh a matching build, `--dem path/to/wgs84-elevation.tif` for a local
elevation source, and `--osm-file response.json` for a saved Overpass response.
The default Copernicus elevation source includes vegetation.

`MAP_CONTEXT_DIR`, `MAP_CONTEXT_GEOJSON` and `MAP_CONTEXT_ARGS` configure the
manual targets. To copy a standalone build to the standard dev or prod data
directory, use the target for that instance.

```sh
make sync-map-context-dev
make sync-map-context-prod
```

The worker reuses a compatible snapshot for the current parcel geometry.

## Remote Dev And Production

For a first hosted install, follow the
[deployment guide](docs/deployment.md) for provisioning, first deploy,
initial data, and backups.

For routine dev deploys, `make deploy-dev` builds the current checkout,
including uncommitted changes. Use `make deploy-dev REF=<branch-or-sha>` to
build a specific revision. Production deploys use
`make deploy-prod REF=<tag-or-sha>` from a clean checkout.
