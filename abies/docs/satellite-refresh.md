# Satellite data refresh

The running Abies instance refreshes Sentinel-2 imagery and parcel
statistics for Bosco. Writers can trigger an update in Cartografia with
**Aggiorna dati terreni**. The name also accommodates future elevation
and LiDAR sources.

## Credentials

Admins manage the Copernicus client ID and secret under **Impostazioni →
Generali → Credenziali API satellitare**. Each installation has its own
credential pair in SQLite. Abies shows the saved client ID and whether a
secret is configured. A blank secret field keeps the current secret. Removing
the pair is an explicit action. Replacing the client ID requires a new secret.
An admin can check the credentials by requesting a token from Copernicus
without downloading imagery.

The Copernicus secret is encrypted with a Fernet key derived from the existing
`DJANGO_SECRET_KEY` using HKDF with a purpose string specific to satellite
credentials. The protection relies on a unique, random Django secret for each
installation. A same-instance restore can reuse the credentials if it uses the
same Django secret. After rotating that secret, an admin re-enters the
Copernicus pair. When mirroring a production instance to dev ([Backup and
mirroring](backups.md#mirroring-runtime-state-between-instances)), the
destination needs its own pair and the mirroring process preserves it.
Previously downloaded imagery stays available during credential replacement.
[Django secret
key](https://docs.djangoproject.com/en/6.0/ref/settings/#secret-key) ·
[HKDF](https://cryptography.io/en/stable/hazmat/primitives/key-derivation-functions/#hkdf)

Encryption protects the Copernicus secret if someone obtains a copy of SQLite
from a backup, a production-to-dev mirror, or the data volume alone. Someone
who can execute code in the web app or access both SQLite and the runtime
environment can recover the secret.

The saved secret and access tokens are excluded from responses, digests,
request logs, and history. An audit event can record who changed or removed
the pair and when. Both reads and writes check the admin role on the server.

Admins enter the credentials once when enabling live downloads. Compatible
imagery already present on disk is reused, including its historical dates.
Bosco can use those data, and the worker can rebuild their compatible local
derivatives while credentials are unconfigured. The running instance
publishes its imagery.
[Obtaining Copernicus credentials](../INSTALL.md#satellite-imagery)

## Refreshing a compresa

| Trigger | Work |
| --- | --- |
| Hourly check | The worker checks SQLite for new maps and published changes that need a refresh. A check with no pending changes makes no Copernicus requests. |
| **Aggiorna dati terreni** | The button at the bottom of the Cartografia sidebar immediately queues a refresh of the selected compresa. It also retries work already due for that compresa. |
| March and August | The worker checks for new seasonal imagery overnight on or after 1 March and 1 August. |

Writers and admins can request a refresh. The button is disabled when no
compresa is selected or the selected compresa lacks published, cataloged
parcel geometry. The sidebar shows the queued or running state and the last
outcome, including when credentials are needed or when the data were already
current. Published vectors remain available while a refresh runs. The
refresh reads the map as published when the job begins.

### Worker process

Each hosted instance runs one worker as a separate Compose service. It uses
the same Abies image, data volume, and runtime settings as the web service.
Its entry point is a long-running Django management command. It uses the
same `restart: unless-stopped` policy as the web service, so Docker starts
it again if the process exits. On startup the worker requeues any refresh
interrupted by its previous exit. Locally, `make terrain-worker` runs the
command alongside `make serve` with the same settings.
[Process recovery](reliability.md)

The web request writes a queue entry to SQLite immediately and returns the
queued status. The worker polls SQLite every 30 seconds while idle. It also
runs the hourly map-change check and the overnight seasonal check from the
same loop. It handles one refresh at a time. During a long backfill it checks
for due work between acquisition dates. A request made while it is busy
stays queued until it can run.

SQLite persists pending work, selected acquisition dates, and seasonal
outcomes across restarts. Repeated clicks and overlapping schedule triggers
combine into one refresh per compresa, with one follow-up if its geometry
changes during the run. Network access and raster processing happen outside
SQLite write transactions. A missed seasonal run is picked up on the next
nightly check. Temporary failures retry with a delay. Jobs needing credentials
resume when an admin supplies them.

The hourly check reads the IDs and versions of published map features and
compares them with the last refresh for each compresa. This covers edits
made in Cartografia and maps imported through the command line. The worker
checks the input geometry and available files before downloading or
recomputing rasters.

## Imagery coverage and historical backfill

The Copernicus request covers a rectangle and the downloaded GeoTIFFs
retain that rectangle on disk. Bosco uses a separate mask rasterized from
the published parcel polygons to hide pixels outside them and to calculate
parcel statistics. The mask follows the polygons at the raster's pixel
resolution.

The request rectangle is the bounding box of the published parcels and roads
plus a margin. An access road can widen that rectangle while Bosco still
opens on the parcels. When the rectangle expands, the extension keeps the
existing pixel alignment and fetches only the missing area for each
historical date that needs it. There are up to four non-overlapping strips
before any request size limits are applied. A compatible legacy raster grid
becomes the starting grid for that compresa.
[Map coverage design](cartography.md#map-coverage-and-background-work)

The worker limits a compresa's raster to 16 million pixels at the nominal
10 m resolution. If the requested extent exceeds that limit, the refresh
fails and keeps the previous revision.

The first refresh checks every completed winter and summer window from 2015
onward and fills in available dates. A published parcel boundary or assignment
change checks the same history, extends retained imagery where needed, and
rebuilds the mask and averages for every date. If the stored rasters already
cover the new outlines, Abies reuses them. A new footprint can make a
previously empty season eligible for another search. Renames, road-type
changes, and edits to drawings leave imagery and statistics current.

## Publishing terrain data

Each compresa has a directory named for its immutable UUID (`map_key`).
Under that directory, each published revision contains its own complete
set of files. For example:

```text
data/satellite/<region_uuid>/
    revisions/
        <revision_id>/
            manifest.json
            timeseries.json
            parcel-mask.tif
            YYYY-MM-DD/
                ndvi.tif
                ...
```

Abies stages the rasters, date list, mask, and statistics on the same
satellite volume. After validation it uses `os.replace()` to move the
staging directory to its final revision path.
Abies syncs the files and directory entries before exposing the revision.
A short SQLite transaction confirms that the current map inputs still match
the job and sets the compresa's active revision ID. That commit is the
publication point. A crash before it leaves the previous revision active.
Abies can discard the unreferenced directory after recovery.
[Python rename semantics](https://docs.python.org/3.14/library/os.html#os.replace)

Bosco gets the revision ID from the manifest and uses it for subsequent
raster, mask, and statistics requests. This keeps a refresh from mixing
products across requests. During its hourly check, the worker purges
revisions that were superseded at least 24 hours earlier. The timeout starts
when a newer revision takes its place, so an active revision remains
available regardless of age. A client requesting a purged revision reloads
the manifest.
If a refresh fails, compatible previously published data stay usable.
The parcel mask and statistics still have to match the current published
parcel fingerprint.
[Current Bosco data contract](page-bosco.md#map-data)

## Adding seasonal images

The overnight run in March searches January and February of the same year.
The overnight run in August searches June and July. Automatic checks
for a newly due season run between 02:00 and 06:00. Abies adds at most one
suitable date per compresa per season to **Bosco → Evoluzione** after its
imagery and statistics are ready. A later run checks the recorded outcome for
that compresa and season before searching or downloading again.

The catalog limits are 20% cloud cover in winter and 10% in
summer. Abies reads all result pages and ranks qualifying dates by cloud
cover. Copernicus reports the percentage for each tile. If several tiles
are needed to cover the parcels on a date, Abies uses the highest reported
cloud cover of the contributing tiles as that date's score. Abies tries the
lowest-scoring date first and checks that it provides usable parcel
coverage. If none qualifies, that season stays empty.
[Copernicus catalog query and pagination](https://documentation.dataspace.copernicus.eu/APIs/OData.html)
· [Sentinel-2 tile cloud cover](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html)
