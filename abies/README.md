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
and optional canonical bootstrap data. Static files are host-mounted too; a
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

`make dev` resets the local SQLite database, runs migrations, bootstraps from
`data/canonical`, builds geodata, regenerates JSON digests, creates or asks for
an admin user, and materializes language-template symlinks. For smaller steps,
use `make migrate`, `make bootstrap`, `make geo`, `make digest`, and
`make admin`.

Run tests with:

```sh
make test
```

Start the development server with:

```sh
DJANGO_DEBUG=1 DJANGO_SECRET_KEY=django-insecure-local-dev python3 manage.py runserver
```

## Production Installation

The following steps describe a fresh host setup for two deployments, production
and dev, served as `abies.laforesta.it` and `abies-dev.laforesta.it`.
Adjust names and ports as needed.

1. Install system packages.

   Install Apache, Python venv support, Docker Engine, Docker Buildx, and a
   Docker Compose CLI usable from the deployment machine. Apache needs these
   modules enabled: `ssl`, `socache_shmcb`, `headers`, `rewrite`, `http2`,
   `reqtimeout`, `proxy`, `proxy_http`, `deflate`, `auth_basic`, and
   `authn_file`.

2. Configure TLS.

   Issue certificates for all served names, including the Abies prod and dev
   hostnames. A single SAN certificate is fine. Configure renewal so Apache is
   reloaded after successful renewal.

3. Create host directories.

   The container runs as UID/GID `10001`. Create these directories on the host:

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

   sudo chown -R 10001:10001 /var/lib/abies-prod/data /var/lib/abies-prod/staticfiles /var/backups/abies-prod
   sudo chown -R 10001:10001 /var/lib/abies-dev/data /var/lib/abies-dev/staticfiles /var/backups/abies-dev
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
   as the Django upload limits; 10 MB is the current deployment default. If the
   dev host is protected by HTTP basic auth, exempt
   `/accounts/microsoft/login/callback/` so OAuth can complete.

   Ipso upload rate limiting needs a real client address, not the Docker bridge
   or loopback proxy address. Apache should discard any client-supplied
   forwarding headers before proxying, then add its own `X-Forwarded-For`.
   Configure `ABIES_IPSO_UPLOAD_TRUSTED_PROXIES` so Abies trusts that header
   only from the loopback/Docker proxy peers. In the default Docker bridge
   deployment this can usually be left unset; the app default trusts loopback
   and common Docker bridge networks.

5. Create compose env files.

   Create `compose/.env.prod` and `compose/.env.dev`. Required production
   values include:

   ```env
   DJANGO_DEBUG=0
   DJANGO_SECRET_KEY=<unique-long-secret>
   DJANGO_ALLOWED_HOSTS=abies.laforesta.it
   DJANGO_CSRF_TRUSTED_ORIGINS=https://abies.laforesta.it
   ABIES_IPSO_SECRET=<shared-ipso-secret>
   MS_OAUTH_TENANT=<tenant-guid-or-name>
   MS_OAUTH_CLIENT_ID=<client-id>
   MS_OAUTH_SECRET=<client-secret>
   ```

   For dev, use the dev hostname in `DJANGO_ALLOWED_HOSTS` and
   `DJANGO_CSRF_TRUSTED_ORIGINS`, and a different `DJANGO_SECRET_KEY`. Optional
   knobs include `ABIES_IPSO_UPLOAD_MAX_BYTES`,
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

   From this repository:

   ```sh
   ./bin/deploy dev
   ./bin/deploy prod <tag-or-sha>
   ```

   The deploy script builds the image, writes a pre-deploy SQLite snapshot to
   the backup mount when an image already exists, runs migrations, optionally
   creates a superuser from
   `DJANGO_SUPERUSER_USERNAME`, `DJANGO_SUPERUSER_EMAIL`, and
   `DJANGO_SUPERUSER_PASSWORD`, collects static files, and starts the compose
   stack.

8. Load initial data.

   Stage canonical CSVs on the host under the instance data directory:

   ```text
   /var/lib/abies-prod/data/canonical
   /var/lib/abies-dev/data/canonical
   ```

   Then run:

   ```sh
   ./bin/bootstrap-data prod
   ./bin/bootstrap-data dev
   ```

   This runs `manage.py bootstrap`, builds geodata into `/app/data/geo`, and
   regenerates all digests. The bootstrap command refuses to load into a
   non-empty database.

9. Enroll Ipso devices.

   Open this URL on each trusted field device, replacing the secret with the
   production `ABIES_IPSO_SECRET`:

   ```text
   https://abies.laforesta.it/ipso/#secret=<ABIES_IPSO_SECRET>
   ```

   The fragment is stored locally by the PWA and removed from the address bar.
   Rotate `ABIES_IPSO_SECRET` to revoke all currently enrolled devices.
