# Backups

Abies runtime backups cover the state that cannot be rebuilt from the repository
or canonical data bundle:

- `data/db.sqlite3`, copied with SQLite's online backup API and verified with
  `PRAGMA integrity_check` before archiving;
- `data/observation-media`, the filesystem-backed photos attached to
  observations;
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
 observation-media/...
 ipso-inbox/...
```

`db.sqlite3` is the transactional consistency anchor. Backup takes an exclusive
cross-process lock at `data/.runtime-snapshot.lock` while copying the database,
observation-media and optional Ipso inbox into a private staging directory.
Media-mutating web requests hold a shared lock through the nonce transaction,
filesystem rollback and commit hooks. This prevents deletion/replacement of a
photo or staged payload between the database snapshot and filesystem capture.
Ordinary concurrent media requests can share the lock; photo reads continue
during backup. Any future command that mutates these files must participate in
the same protocol, or operate with the instance stopped.

The exclusive lock is released before checksum verification and compression;
those use only the private copies. A live photo deleted after capture remains
in that captured backup. Referenced photos must exist and match the database's
byte sizes and SHA-256 checksums, or the backup fails before publication/pruning.
Format-2 manifests record the validated photo references. Older databases
without the observation-photo table have no photo references to validate.

Completed archives are published atomically from unique temporary files. An
existing final archive name is never overwritten, including same-second runs.
This does not serialize every maintenance/mirroring operation; target replacement
still requires the stopped-instance procedure below.

`data/observation-thumbnails` is a regenerable cache and is omitted. Main stored
photos and staged mobile photo bytes are included; thumbnails regenerate when
requested after a restore.

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
`db.sqlite3`, `observation-media` and `ipso-inbox`, then runs
`PRAGMA integrity_check` on the restored SQLite database. It also verifies every
referenced photo's path, size and checksum, including for older archives whose
database contains photo metadata. Validation happens in a private staging
directory before the destination is exposed. A failure leaves the requested
destination absent/empty.

For a full restore, stop the instance, preserve the current data directory, run
`bin/backup --restore` into a temporary directory, copy the restored
`db.sqlite3` into `/var/lib/abies-prod/data/db.sqlite3`, restore
`observation-media` and `ipso-inbox` from the same archive, fix ownership to
`abies:abies`, and start the container. Remove stale SQLite WAL/SHM files while
the instance is stopped. Check observation details, open the restored photos,
and verify pending inbox operations as part of the recovery drill. The archive
validation establishes photo-reference integrity; an operational restore drill
also checks the rest of the application and off-host recovery availability.

## Mirroring Runtime State Between Instances

Use `bin/mirror-instance-state` when remote dev should become a copy of
production for realistic testing:

```sh
make mirror-prod-to-dev MIRROR_ARGS=--yes
```

The tool copies the same runtime state covered by backups: `db.sqlite3`,
`observation-media`, and, by default, `ipso-inbox`. It intentionally does not
copy generated/static or deployment-owned inputs such as canonical CSVs,
satellite data, geodata, branding, collected static files, or compose
environment secrets.

The default `prod -> dev` flow is:

1. create a fresh source backup in the prod backup mount;
2. create a pre-mirror backup of the target dev state;
3. transfer the source archive into the target backup mount;
4. stop the target compose stack;
5. restore `db.sqlite3`, remove stale SQLite WAL/SHM files, and replace the
   target observation media and Ipso inbox with restored copies, or empty
   directories when the source archive has none;
6. run target migrations and regenerate digests using the target image;
7. start the target compose stack.

Run a plan without changing anything:

```sh
bin/mirror-instance-state prod dev --dry-run
```

Because mirroring overwrites target runtime state, non-interactive runs require
`--yes`. The tool refuses to target production unless `--allow-prod-target` is
also provided.
