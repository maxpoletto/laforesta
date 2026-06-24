# Abies

Abies is an application for forest management and operations. It supports forest
surveys, harvest planning, tracking of harvesting operations, and management
of lumber crews. It has an information-rich UI based on interactive maps and
spreadsheet-like tables, and provides various JSON API endpoints. The UI is
currently available only in Italian, but makes localization easy thanks to
string constants and language-specific templates. It is written using Django and
pure Javascript.

The app is initialized via a canonical data bundle loaded into an empty SQLite
database. Once the initial data is loaded, operators work from the browser. The
UI has several sections:
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
separate backup mount is used for deploy-time SQLite snapshots, but scheduled or
off-host backups are still a separate operational task.

## More Documentation

- [docs/bootstrap.md](docs/bootstrap.md): canonical CSV bundle and bootstrap
  contract.
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
  Impostazioni, Login, Piano di taglio, Prelievi, and Squadre.

## Local Development

Use Python 3.13 or newer. The production image currently uses Python 3.14.

```sh
python3 -m venv ~/venv/abies
. ~/venv/abies/bin/activate
pip install -r requirements.txt
```

Create a canonical data bundle in `data/canonical`. Legacy-data conversion is
owned by the external initialization tooling; Abies only consumes canonical CSVs.
Then run:

```sh
make dev
```

`make dev` resets the local SQLite database, runs the shared canonical-data
load sequence, creates or asks for an admin user, and materializes
language-template symlinks. The shared sequence runs migrations, bootstraps
from `data/canonical`, stages converted mark uploads if present, builds
geodata, and regenerates JSON digests. For smaller steps, use `make migrate`,
`make bootstrap`, `make geo`, `make stage-marks-uploads`, `make digest`, and
`make admin`. Run `make help` for the full local and remote target list.

To make the process completely non-interactive (e.g., during edit-debug cycles),
set the following environment variables:
```sh
DJANGO_SUPERUSER_USERNAME
DJANGO_SUPERUSER_EMAIL
DJANGO_SUPERUSER_PASSWORD
```

Run tests with:

```sh
make test
```

To test Microsoft Entra OAuth, also set:
```sh
MS_OAUTH_TENANT
MS_OAUTH_CLIENT_ID
MS_OAUTH_SECRET
```

To test Ipso data uploads, also set `ABIES_IPSO_SECRET`.

Finally, start the development server with:

```sh
make serve
```

## Remote Dev And Production

Abies has three normal operating modes:

- local development, described above, runs Django directly from the checkout;
- remote dev runs the containerized app at `abies-dev.laforesta.it`, using
  `compose/dev.yml`, `.env.dev`, and host port `127.0.0.1:8001`;
- production runs the containerized app at `abies.laforesta.it`, using
  `compose/prod.yml`, `.env.prod`, and host port `127.0.0.1:8000`.

Remote dev is intended for pre-production checks on the same shape of
infrastructure as production. It should still use `DJANGO_DEBUG=0` and its own
secrets, OAuth redirect URI, data directory, Ipso secret, and database.

The following steps describe a fresh host setup for both remote instances.
Adjust names and ports as needed.

1. Install system packages.

   Install Apache, Python venv support, Docker Engine, Docker Buildx, and a
   Docker Compose CLI usable from the deployment machine. Apache needs these
   modules enabled: `ssl`, `socache_shmcb`, `headers`, `rewrite`, `http2`,
   `reqtimeout`, `proxy`, `proxy_http`, and `deflate`.

2. Configure TLS.

   Issue certificates for all served names, including the Abies prod and dev
   hostnames. A single SAN certificate is fine. Configure renewal so Apache is
   reloaded after successful renewal.

3. Create host directories.

   The container runs as UID/GID `10001`. Create a matching non-login host
   account so bind-mounted files show readable ownership on the VM:

   ```sh
   sudo groupadd --system --gid 10001 abies
   sudo useradd --system --uid 10001 --gid abies \
     --home-dir /nonexistent --shell /usr/sbin/nologin --no-create-home abies
   ```

   Then create these directories on the host:

   ```sh
   sudo mkdir -p /var/lib/abies-prod/data/digests
   sudo mkdir -p /var/lib/abies-prod/data/geo
   sudo mkdir -p /var/lib/abies-prod/data/ipso-inbox
   sudo mkdir -p /var/lib/abies-prod/staticfiles
   sudo mkdir -p /var/backups/abies-prod

   sudo mkdir -p /var/lib/abies-dev/data/digests
   sudo mkdir -p /var/lib/abies-dev/data/geo
   sudo mkdir -p /var/lib/abies-dev/data/ipso-inbox
   sudo mkdir -p /var/lib/abies-dev/staticfiles
   sudo mkdir -p /var/backups/abies-dev

   sudo chown -R abies:abies /var/lib/abies-prod/data /var/lib/abies-prod/staticfiles /var/backups/abies-prod
   sudo chown -R abies:abies /var/lib/abies-dev/data /var/lib/abies-dev/staticfiles /var/backups/abies-dev
   sudo chmod 0750 /var/lib/abies-prod/data /var/backups/abies-prod
   sudo chmod 0750 /var/lib/abies-dev/data /var/backups/abies-dev
   sudo chmod 0755 /var/lib/abies-prod/staticfiles /var/lib/abies-dev/staticfiles
   ```

   `data/ipso-inbox` is used by the Abies-served Ipso PWA upload endpoint. No
   separate mount is needed with the default `ABIES_IPSO_INBOX_DIR`; it is
   already inside the `/app/data` bind mount. If the inbox is moved elsewhere,
   expose that path to the container explicitly.

4. Configure Apache reverse proxies.

   Each vhost should terminate TLS, serve `/static/` directly from the matching
   `staticfiles` directory, and proxy everything else to the loopback port used
   by the compose file. Current defaults are `127.0.0.1:8000` for prod and
   `127.0.0.1:8001` for dev. Set `X-Forwarded-Proto: https`; Django trusts that
   header for secure-cookie behavior. Use a request-body cap at least as large
   as the Django upload limits; 10 MB is the current deployment default.

   Ipso upload rate limiting needs a real client address, not the Docker bridge
   or loopback proxy address. Apache should discard any client-supplied
   forwarding headers before proxying, then add its own `X-Forwarded-For`.
   Configure `ABIES_IPSO_UPLOAD_TRUSTED_PROXIES` so Abies trusts that header
   only from the loopback/Docker proxy peers. In the default Docker bridge
   deployment this can usually be left unset; the app default trusts loopback
   and common Docker bridge networks.

5. Create compose env files.

   Create `compose/.env.prod` and `compose/.env.dev`. Required values include:

   ```env
   DJANGO_DEBUG=0
   DJANGO_SECRET_KEY=<unique-long-secret>
   DJANGO_ALLOWED_HOSTS=<instance-hostname>
   DJANGO_CSRF_TRUSTED_ORIGINS=https://<instance-hostname>
   ABIES_IPSO_SECRET=<shared-ipso-secret>
   MS_OAUTH_TENANT=<tenant-guid-or-name>
   MS_OAUTH_CLIENT_ID=<client-id>
   MS_OAUTH_SECRET=<client-secret>
   ```

   Use separate values for prod and dev. In particular, do not reuse
   `DJANGO_SECRET_KEY`, `ABIES_IPSO_SECRET`, or OAuth redirect URIs between the
   two instances. Optional knobs include `ABIES_IPSO_UPLOAD_MAX_BYTES`,
   `ABIES_IPSO_UPLOAD_MAX_RECORDS`, `ABIES_IPSO_UPLOAD_RATE_LIMIT`,
   `ABIES_IPSO_UPLOAD_RATE_WINDOW_S`,
   `ABIES_IPSO_UPLOAD_TRUSTED_PROXIES`, `ABIES_IPSO_INBOX_DIR`,
   `DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE`, `DJANGO_SECURE_HSTS_SECONDS`, and
   `ABIES_SATELLITE_DIR`.

6. Prepare Docker access.

   The deployment scripts use a Docker context named `vm-abies` by default:

   ```sh
   docker context create vm-abies --docker host=ssh://<user>@<host>
   ```

   Set `ABIES_DOCKER_CONTEXT` if you use a different context name.

7. Deploy the application.

   From this repository, deploy dev with either the current working tree or an
   explicit git ref:

   ```sh
   make deploy-dev
   make deploy-dev REF=<branch-or-sha>
   ```

   Without a ref, the dev deploy builds whatever is currently in the checkout,
   including uncommitted changes. With a ref, it fetches tags and builds a
   temporary archived copy of that ref, leaving the checkout unchanged.

   Production deploys require an explicit ref and refuse to run from a dirty
   tree:

   ```sh
   make deploy-prod REF=<tag-or-sha>
   ```

   The Makefile targets call `bin/deploy`, which builds the image, writes a
   pre-deploy SQLite snapshot to the backup mount when an image already exists,
   runs migrations, runs `manage.py check --deploy --fail-level WARNING` for
   prod, creates the first superuser when none exists and
   `DJANGO_SUPERUSER_USERNAME`, `DJANGO_SUPERUSER_EMAIL`, and
   `DJANGO_SUPERUSER_PASSWORD` are set, collects static files, minifies
   collected JavaScript for prod, and starts the compose stack.

   Deploys are intentionally non-destructive: they do not reset the database,
   and they do not change an existing superuser's password. To replace the
   current superuser credentials explicitly, add `DEPLOY_ARGS=--reset-user`.

8. Load initial data.

   Keep canonical CSVs locally under `data/canonical`, or override the source
   with `CANONICAL_DIR=<path>`. Then run:

   ```sh
   make bootstrap-dev
   make bootstrap-prod
   ```

   These targets rsync the local canonical directory to the matching host path
   under `/var/lib/abies-<instance>/data/canonical`, fix ownership for the
   container UID, then run `bin/bootstrap-data <instance>`, which applies
   migrations before loading the empty database. To sync without bootstrapping,
   use `make stage-canonical-dev` or
   `make stage-canonical-prod`. The bootstrap script runs the same shared
   canonical-data load sequence as local development: migrations, `manage.py
   bootstrap`, staging `data/canonical/marks/*.csv` into the Ipso inbox,
   geodata into `/app/data/geo`, and all digests. It refuses to load into a
   non-empty database.

   Bootstrap loads forestry data only; it does not create users. On a fresh
   instance, run deploy first with `DJANGO_SUPERUSER_*` present in the compose
   env file, then bootstrap. If you manually delete `data/db.sqlite3` after
   deploy, the superuser is gone too; rerun deploy or create a superuser inside
   the container before trying to log in.

   The rsync targets default to `REMOTE=maxp@abies.laforesta.it`; override
   `REMOTE_USER`, `REMOTE_HOST`, or `REMOTE` if needed.

9. Enroll Ipso devices.

   Open the appropriate URL on each trusted field device:

   ```text
   https://abies.laforesta.it/ipso/#secret=<ABIES_IPSO_SECRET>
   https://abies-dev.laforesta.it/ipso/#secret=<DEV_ABIES_IPSO_SECRET>
   ```

   The fragment is stored locally by the PWA and removed from the address bar.
   Rotate the instance's `ABIES_IPSO_SECRET` to revoke all devices enrolled
   against that instance.
