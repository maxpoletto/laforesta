# Abies production deployment

Abies runs on a single box as one Docker container managed by systemd.
Apache on the host terminates TLS, serves `/abies/static/` directly, and
proxies everything else under `/abies/` to gunicorn on 127.0.0.1:8000.

```
Internet --HTTPS--> Apache :443
                      |
                      +-- /abies/static/  --> /var/lib/abies/static/ (disk)
                      |
                      +-- /abies/*        --> gunicorn :8000 (container)
                                                   |
                                                   +-- /app/data        -> /var/lib/abies/data   (bind)
                                                   +-- /app/staticfiles -> /var/lib/abies/static (bind)
```

All persistent state (SQLite DB, JSON digests, GeoJSON) lives on the host
under `/var/lib/abies/`.  The container is stateless and disposable.

The host path `/var/lib/abies` is a convention set in two files:
`deploy/abies.service` (docker volume mounts) and
`deploy/apache-abies.conf` (Apache `Alias`).  The container itself only
knows `/app/data` and `/app/staticfiles`.

---

## First-time server bootstrap

Upload `deploy/` to the server once (standard path: `/root/abies-deploy/`),
then run `bootstrap.sh`:

```sh
# From your laptop:
scp -r deploy/ root@laforesta.it:/root/abies-deploy/

# On the server:
sudo /root/abies-deploy/bootstrap.sh
```

`bootstrap.sh` is idempotent and does:

1. Installs Docker if missing.
2. Creates `/var/lib/abies/{data,static}` owned by UID 10001 (matches the
   `abies` user inside the image).
3. Creates `/etc/abies/env` from `env.example` (mode 0600, root-owned) —
   **only if it does not already exist**.  Edit it and fill in real
   values before starting the service.
4. Installs `abies.service` into `/etc/systemd/system/` and enables it.

Override any of these by exporting before the script:

```sh
sudo ABIES_UID=20000 ABIES_DATA=/srv/abies/data /root/abies-deploy/bootstrap.sh
```

Available variables: `ABIES_UID`, `ABIES_GID`, `ABIES_DATA`,
`ABIES_STATIC`, `ABIES_ETC`.

### Seed initial data (one-time)

On your laptop:

```sh
make dev          # migrate + import + geo + digest + createsuperuser
```

Then copy the primed DB and geo files to the server:

```sh
scp    data/db.sqlite3 root@laforesta.it:/var/lib/abies/data/
scp -r data/geo        root@laforesta.it:/var/lib/abies/data/
ssh    root@laforesta.it 'chown -R 10001:10001 /var/lib/abies/data'
```

### Wire up Apache

Add the snippet from `deploy/apache-abies.conf` to the existing
laforesta.it:443 vhost in your Apache config repo.  Enable modules if
needed:

```sh
a2enmod proxy proxy_http headers alias
systemctl reload apache2
```

---

## Releases

### From your laptop

```sh
make deploy                      # build, tag, save tarball, scp to server
```

`make deploy` prints the final command to run on the server.  Override
host/user with `make deploy DEPLOY_HOST=other.host DEPLOY_USER=max`.

### On the server

```sh
sudo /root/abies-deploy/install-release.sh <tag>
```

`<tag>` is the short git sha printed by `make deploy`.  The script:

1. `docker load`s the tarball from `/tmp/abies-<tag>.tar.gz`.
2. Retags `abies:<tag>` → `abies:latest`.
3. Restarts `abies.service`.
4. Shows status and deletes the tarball.

The container's entrypoint runs `migrate` and `collectstatic` on every
start, so schema changes and new static assets are picked up
automatically.

### Rollback

Older images stay cached locally until pruned:

```sh
docker images abies              # list available tags
docker tag abies:<older-sha> abies:latest
systemctl restart abies
```

### Restore DB from backup

```sh
systemctl stop abies
cp /path/to/backup.sqlite3 /var/lib/abies/data/db.sqlite3
chown 10001:10001          /var/lib/abies/data/db.sqlite3
systemctl start abies
```

---

## Day to day

```sh
journalctl -u abies -f                 # live logs (systemd + gunicorn)
docker logs -f abies                   # same stream, container view
docker exec -it abies python manage.py shell
systemctl restart abies
docker images abies                    # see cached release history
docker image prune                     # clean up untagged images
```

---

## Notes

- **Port 8000** is published on `127.0.0.1` only; nothing on the container
  is reachable from outside the host.
- **Image tagging**: `make release` tags with the short git sha.  The
  systemd unit always runs `abies:latest`, so a release = retag + restart
  and a rollback is a one-liner (`docker tag abies:<older-sha> abies:latest
  && systemctl restart abies`).
- **Secrets never leave `/etc/abies/env`**: that file is root-owned 0600
  and read by the Docker daemon (via `--env-file`), not exposed to any
  other process.
