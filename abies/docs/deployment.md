# Hosted deployment

The hosted stacks use separate Compose files, environment files, and loopback
ports. The examples below use hostnames for Soc. Agr. La Foresta S.r.l, an
Italian lumber firm.

- Remote dev runs at `abies-dev.laforesta.it` with `compose/dev.yml`,
  `compose/.env.dev`, and host port `127.0.0.1:8001`.
- Production runs at `abies.laforesta.it` with `compose/prod.yml`,
  `compose/.env.prod`, and host port `127.0.0.1:8000`.

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
   sudo mkdir -p /var/lib/abies-prod/data/ipso-inbox
   sudo mkdir -p /var/lib/abies-prod/data/satellite
   sudo mkdir -p /var/lib/abies-prod/staticfiles
   sudo mkdir -p /var/backups/abies-prod

   sudo mkdir -p /var/lib/abies-dev/data/digests
   sudo mkdir -p /var/lib/abies-dev/data/ipso-inbox
   sudo mkdir -p /var/lib/abies-dev/data/satellite
   sudo mkdir -p /var/lib/abies-dev/staticfiles
   sudo mkdir -p /var/backups/abies-dev

   sudo chown -R abies:abies /var/lib/abies-prod/data /var/lib/abies-prod/staticfiles /var/backups/abies-prod
   sudo chown -R abies:abies /var/lib/abies-dev/data /var/lib/abies-dev/staticfiles /var/backups/abies-dev
   sudo chmod 0750 /var/lib/abies-prod/data /var/backups/abies-prod
   sudo chmod 0750 /var/lib/abies-dev/data /var/backups/abies-dev
   sudo chmod 0755 /var/lib/abies-prod/staticfiles /var/lib/abies-dev/staticfiles
   ```

   Place deployment branding assets locally under `data/branding` before
   building or deploying:

   ```text
   data/branding/logo.png
   data/branding/favicon.gif
   data/branding/ipso-logo.gif
   ```

   `bin/deploy` and `make collectstatic` copy these files into fixed static
   paths before static collection. The files are intentionally local deployment
   inputs, not checked into Abies. External initialization tooling can populate
   `data/branding` the same way it populates `data/canonical`.

   `data/ipso-inbox` is used by the Abies-served Ipso PWA upload endpoint. No
   separate mount is needed with the default `ABIES_IPSO_INBOX_DIR`; it is
   already inside the `/app/data` bind mount. If the inbox is moved elsewhere,
   expose that path to the container explicitly.

4. Configure Apache reverse proxies.

   Each vhost should terminate TLS, serve `/static/` directly from the matching
   `staticfiles` directory, and proxy everything else to the loopback port used
   by the compose file. Current defaults are `127.0.0.1:8000` for prod and
   `127.0.0.1:8001` for dev. Set `X-Forwarded-Proto: https`; Django trusts that
   header for secure-cookie behavior. Use a request-body cap aligned with the
   Django upload limits; 30 MiB is the current Abies/Ipso deployment default.

   Serve `/static/` with `Cache-Control: no-cache` so browsers revalidate the
   fixed URLs after deployment. For La Foresta this header is set by the
   Abies vhost template in `system/ansible/templates/apache2-sites/`.
   When switching from the old `immutable` policy, apply the Apache change
   before asking users to clear cached images and files once and reload Abies.
   Cookies and site data stay in place, including Ipso's locally saved records.

   Ipso upload rate limiting needs a real client address, not the Docker bridge
   or loopback proxy address. Apache should discard any client-supplied
   forwarding headers before proxying, then add its own `X-Forwarded-For`.
   Configure `ABIES_IPSO_UPLOAD_TRUSTED_PROXIES` so Abies trusts that header
   only from the loopback/Docker proxy peers. In the default Docker bridge
   deployment this can usually be left unset; the app default trusts loopback
   and common Docker bridge networks.

5. Create compose env files.

   Copy `compose/.env.prod.example` and `compose/.env.dev.example` to
   `compose/.env.prod` and `compose/.env.dev`. Fill in the settings for each
   hostname and installation.

   ```env
   DJANGO_DEBUG=0
   DJANGO_SECRET_KEY=<unique-long-secret>
   DJANGO_ALLOWED_HOSTS=<instance-hostname>
   DJANGO_CSRF_TRUSTED_ORIGINS=https://<instance-hostname>
   ABIES_IPSO_SECRET=<shared-ipso-secret>
   ABIES_APP_NAME=Abies
   ABIES_BRAND_NAME=<deployment-brand-name>
   ABIES_SITE_TITLE=<browser-title>
   ```

   Use a distinct `DJANGO_SECRET_KEY` and `ABIES_IPSO_SECRET` for each
   instance. To enable Microsoft login, set `MS_OAUTH_TENANT`,
   `MS_OAUTH_CLIENT_ID`, and `MS_OAUTH_SECRET` and configure a redirect URI
   for each hostname. [Microsoft OAuth configuration](security.md#ms-365-oauth-configuration)

   For the first admin account, set `DJANGO_SUPERUSER_USERNAME` and
   `DJANGO_SUPERUSER_PASSWORD` before deploying. `DJANGO_SUPERUSER_EMAIL`
   is optional.

   Optional settings include `ABIES_API_READ_RATE_LIMIT`,
   `ABIES_API_WRITE_RATE_LIMIT`, `ABIES_API_RATE_WINDOW_S`,
   `ABIES_IPSO_UPLOAD_MAX_BYTES`,
   `ABIES_IPSO_UPLOAD_MAX_RECORDS`, `ABIES_IPSO_UPLOAD_RATE_LIMIT`,
   `ABIES_IPSO_UPLOAD_RATE_WINDOW_S`,
   `ABIES_IPSO_UPLOAD_TRUSTED_PROXIES`, `ABIES_IPSO_INBOX_DIR`,
   `DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE`, `DJANGO_SECURE_HSTS_SECONDS`, and
   `ABIES_SATELLITE_DIR`. The Compose files set `ABIES_INSTANCE` to `dev` or
   `prod`.

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

   The deployed ref/current-tree description is passed to the container as
   `ABIES_VERSION`. Impostazioni and Ipso show the same version string. For
   example, the release tag `abies-2.0.0` displays as `2.0.0` in both apps.

   The Makefile targets call `bin/deploy`, which builds the image, writes a
   pre-deploy compressed backup to the backup mount, runs migrations, runs
   `manage.py check --deploy --fail-level WARNING` for prod, creates the first
   admin if configured, rebuilds the collected static directory from the
   deployed release, minifies collected JavaScript
   for prod, and starts the Compose stack. Open the instance URL and sign in
   with the admin account.

   Routine deploys preserve the database and existing superuser passwords.
   `DEPLOY_ARGS=--reset-user` removes the existing superusers before
   creating the configured admin.

8. Choose the initial data.

   - **Empty map.** Use **Impostazioni → Cartografia** to draw or import
     regions after signing in.
   - **Map file only.** Place `map.geojson` in `data/canonical`, transfer
     it with `make sync-canonical-prod` or `make sync-canonical-dev`, and
     import it in the target container. This example imports the map in
     production.

     ```sh
     docker --context vm-abies exec abies-prod \
       python manage.py build_geo /app/data/canonical/map.geojson
     ```

     The [local install guide](../INSTALL.md#fresh-local-instance) describes
     the GeoJSON format.
   - **Existing forestry records.** Place the canonical CSVs and optional
     `map.geojson` in `data/canonical`. Run `make bootstrap-dev` or
     `make bootstrap-prod` for the target instance. These targets require an
     empty forestry catalog. They transfer the canonical bundle, fix
     ownership, apply migrations, import the catalog and map, stage converted
     mark uploads, and regenerate digests. [Canonical bundle format](bootstrap.md)

   `CANONICAL_DIR` overrides the local bundle path. The rsync targets default
   to `REMOTE=maxp@abies.laforesta.it`. Override `REMOTE_USER`,
   `REMOTE_HOST`, or `REMOTE` if needed. Use `make sync-canonical-dev` or
   `make sync-canonical-prod` for a transfer alone.

   The Compose `terrain` service prepares hillshade and print-map context
   from the published parcel map. Set up Copernicus credentials to add
   satellite imagery. [Satellite imagery setup](../INSTALL.md#satellite-imagery)

9. Configure backups.

   `bin/deploy` creates a compressed pre-deploy backup before migrations. For
   nightly backups and OneDrive mirroring, install a host-level systemd timer
   that runs `docker exec abies-prod bin/backup --instance prod` at 23:00
   and optionally follows it with `rclone sync`. See
   [the backup guide](backups.md) for the exact service/timer example,
   retention policy, and restore notes.

10. Enroll Ipso devices if operators will use the app.

    Open the appropriate URL on each trusted field device.

    ```text
    https://abies.laforesta.it/ipso/#secret=<ABIES_IPSO_SECRET>
    https://abies-dev.laforesta.it/ipso/#secret=<DEV_ABIES_IPSO_SECRET>
    ```

    The fragment is stored locally by the PWA and removed from the address
    bar. Keep the same `ABIES_IPSO_SECRET` across normal upgrades so already
    enrolled devices remain authorized. Rotate the instance's
    `ABIES_IPSO_SECRET` to revoke all devices enrolled against it. If
    upgrading an older deployment that used separate Ipso tokens, set
    `ABIES_IPSO_SECRET` to the old upload token value to preserve enrollment.
