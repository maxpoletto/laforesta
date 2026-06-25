# Backups

Abies runtime backups cover the state that cannot be rebuilt from the repository
or canonical data bundle:

- `data/db.sqlite3`, copied with SQLite's online backup API and verified with
  `PRAGMA integrity_check` before archiving;
- `data/ipso-inbox`, included by default as operational evidence for staged
  mobile uploads.

Canonical CSVs, generated digests, generated geodata, collected static files,
and deployment branding are intentionally not backed up by this mechanism. They
are either source inputs managed outside Abies or generated outputs.

Backups are gzip-compressed tar archives written under the instance backup mount
(`/var/backups/abies-prod` or `/var/backups/abies-dev` on the host, mounted as
`/app/backup` in the container). Archive names are timestamped in UTC:

```text
abies-prod-20260625T210000Z-scheduled.tar.gz
abies-prod-20260625T210000Z-predeploy.tar.gz
```

Each archive contains:

```text
manifest.json
 db.sqlite3
 ipso-inbox/...
```

`db.sqlite3` is the transactional consistency anchor. The inbox is a filesystem
snapshot taken just after the SQLite snapshot; Ipso writes files atomically and
keeps them append-style, so this is good enough for recovering staged uploads,
but it is not a filesystem-level transaction.

## Creating A Backup

Inside a running container:

```sh
docker exec abies-prod bin/backup --instance prod
```

From the deployment checkout, using the configured Docker context:

```sh
make backup-prod
make backup-dev
```

Deploys also run `bin/backup --reason predeploy --allow-missing` after the
new image is built and before migrations run. That keeps deploy-time snapshots
compressed and under the same retention policy as scheduled backups.

## Retention

After every successful backup, `bin/backup` prunes archives for the same
instance using this filename-based policy:

- keep all backups less than 7 days old;
- after 7 days, keep only Saturday backups;
- after 183 days, keep only Saturday backups whose ISO week number is divisible
  by 4.

The rule uses the UTC timestamp embedded in the archive name, not file mtime.
That keeps thinning stable after rsync, restore tests, and OneDrive syncs.

To run pruning without creating a new archive:

```sh
docker exec abies-prod bin/backup --instance prod --prune-only
```

## Scheduling

Schedule backups from the host, not from cron inside the application container.
That keeps the container single-purpose, lets the host scheduler track failures
and missed runs, and keeps off-host mirror credentials outside the Abies runtime
environment.

The scheduler should run the backup command inside the running container:

```sh
docker exec abies-<instance> bin/backup --instance <instance>
```

For production, run it nightly at about 23:00. If the host also mirrors backups
off-site, mirror the already-thinned backup directory after `bin/backup`
returns successfully:

```sh
rclone sync --checksum --create-empty-src-dirs \
  /var/backups/abies-<instance> <remote>:<path>
```

Use the same shape for dev only if dev data matters enough to back up.

## OneDrive Mirror

Use `rclone` on the VM host for OneDrive. Keep Microsoft OAuth credentials in
host-owned backup configuration, not in Abies compose env files and not in the
container.

There are two workable credential models:

- interactive user OAuth: configure an rclone remote once as the host user that
  runs the mirror, then let rclone store and refresh its token;
- non-interactive service credentials: configure the rclone remote entirely
  through `RCLONE_CONFIG_*` environment variables loaded by the host scheduler.

The non-interactive model needs an Entra app registration with Microsoft Graph
application permissions for the target drive. Rclone reads environment-backed
remote configuration as `RCLONE_CONFIG_<REMOTE>_<OPTION>`, where `<REMOTE>` is
the uppercased remote name. For a mirror destination such as:

```text
onedrive:abies-prod/backups
```

the remote name is `onedrive`, so the variables are:

```env
RCLONE_CONFIG_ONEDRIVE_TYPE=onedrive
RCLONE_CONFIG_ONEDRIVE_CLIENT_ID=<entra-app-client-id>
RCLONE_CONFIG_ONEDRIVE_CLIENT_SECRET=<entra-app-client-secret>
RCLONE_CONFIG_ONEDRIVE_CLIENT_CREDENTIALS=true
RCLONE_CONFIG_ONEDRIVE_TENANT=<tenant-id-or-domain>
RCLONE_CONFIG_ONEDRIVE_DRIVE_ID=<target-drive-id>
RCLONE_CONFIG_ONEDRIVE_DRIVE_TYPE=business
```

Expected values:

- `CLIENT_ID`: the Entra application/client ID for the app registration;
- `CLIENT_SECRET`: a client secret for that app registration;
- `CLIENT_CREDENTIALS`: `true`, so rclone uses the service credential flow;
- `TENANT`: the Entra tenant ID or tenant domain;
- `DRIVE_ID`: the OneDrive/SharePoint drive ID to mirror into;
- `DRIVE_TYPE`: usually `business` for Microsoft 365/SharePoint storage.

If the remote name changes, replace `ONEDRIVE` in the variable names with the
uppercased remote name. For reference, rclone documents remote config through
environment variables as `RCLONE_CONFIG_<REMOTE>_<OPTION>`, and the OneDrive
backend documents the `client_credentials`, `tenant`, `drive_id`, and
`drive_type` options:

- https://rclone.org/docs/#config-file
- https://rclone.org/onedrive/#client-credentials

If the mirror fails, the host scheduler should report a failure while leaving
the local backup archive on disk.

## Restore Sketch

To inspect or stage a restore from an archive:

```sh
docker exec abies-prod bin/backup \
  --restore /app/backup/abies-prod-YYYYMMDDTHHMMSSZ-scheduled.tar.gz \
  --dest /app/data/restore-check
```

The restore destination must be empty or absent. The command validates archive
paths, rejects links and unexpected members, extracts `manifest.json`,
`db.sqlite3`, and `ipso-inbox`, then runs `PRAGMA integrity_check` on the
restored SQLite database.

For a full restore, stop the instance, preserve the current data directory, run
`bin/backup --restore` into a temporary directory, copy the restored
`db.sqlite3` into `/var/lib/abies-prod/data/db.sqlite3`, restore `ipso-inbox` if
needed, fix ownership to `abies:abies`, and start the container.
