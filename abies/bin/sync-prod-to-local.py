"""Restore an online production snapshot into this checkout's stopped local instance."""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.base.runtime_snapshot import runtime_snapshot_lock
from apps.bosco.mirror_credentials import prepare, restore


# Use a private, short-lived archive on the production backup mount. Keep
# backup diagnostics on stderr so stdout carries only the archive bytes.
STREAM_BACKUP = r'''
set -eu
work=$(mktemp -d /app/backup/.local-sync.XXXXXX)
trap 'rm -rf "$work"' EXIT
bin/backup --instance prod --reason sync-local --backup-dir "$work" --no-prune >&2
cat "$work"/abies-prod-*.tar.gz
'''
STREAM_LEGACY_MAP = r'''
set -eu
if [ -f /app/data/geo/terreni.geojson ]; then
  cat /app/data/geo/terreni.geojson
fi
'''


def fetch_snapshot(remote, work):
    archive = work / 'prod.tar.gz'
    snapshot = work / 'snapshot'
    with archive.open('wb') as output:
        subprocess.run([*remote, 'sh', '-c', STREAM_BACKUP], stdout=output, check=True)
    subprocess.run([sys.executable, str(ROOT / 'bin/backup'), '--restore', str(archive),
                    '--dest', str(snapshot)], check=True)

    # Production may still be running a version that serves maps from disk.
    with closing(sqlite3.connect(snapshot / 'db.sqlite3')) as connection:
        map_migrated = connection.execute(
            'SELECT 1 FROM django_migrations WHERE app = ? AND name = ?',
            ('base', '0035_map_features'),
        ).fetchone()
    if not map_migrated:
        geo = snapshot / 'geo'
        geo.mkdir()
        legacy_map = geo / 'terreni.geojson'
        with legacy_map.open('wb') as output:
            subprocess.run([*remote, 'sh', '-c', STREAM_LEGACY_MAP], stdout=output, check=True)
        if not legacy_map.stat().st_size:
            legacy_map.unlink()
    return snapshot


def prepare_database(snapshot, target_db, settings):
    import django
    from django.core.management import call_command
    from django.db import connections

    database = snapshot / 'db.sqlite3'
    credentials = snapshot / '.local-credentials.json'
    prepare(target_db, database, credentials)
    settings.DATABASES['default']['NAME'] = database
    settings.DATA_DIR = snapshot
    settings.DIGEST_DIR = snapshot / 'digests'
    settings.GEO_DIR = snapshot / 'geo'
    settings.OBSERVATION_MEDIA_DIR = snapshot / 'observation-media'
    settings.IPSO_INBOX_DIR = snapshot / 'ipso-inbox'
    settings.MAP_CONTEXT_DIR = settings.SATELLITE_DIR = None
    os.environ['ABIES_LEGACY_GEO_DIR'] = str(settings.GEO_DIR)
    django.setup()
    try:
        call_command('migrate', interactive=False)
        restore(database, credentials)
        from django.contrib.sessions.models import Session
        from apps.base.digests import generate_all

        Session.objects.all().delete()
        generate_all()
    finally:
        connections.close_all()


def install_snapshot(snapshot, targets, previous):
    saved, installed = [], []
    try:
        for name, path in targets.items():
            if path.exists() or path.is_symlink():
                shutil.move(path, previous / name)
                saved.append((previous / name, path))
        for name, path in targets.items():
            source = snapshot / name
            if source.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(source, path)
                installed.append((path, source))
    except BaseException:
        for path, source in reversed(installed):
            shutil.move(path, source)
        for backup, path in reversed(saved):
            shutil.move(backup, path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes', action='store_true', help='confirm replacement of local runtime data')
    args = parser.parse_args()

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    from django.conf import settings

    if settings.ABIES_INSTANCE != 'local':
        parser.error('the destination must be an Abies local instance')
    data = settings.DATA_DIR
    database = Path(settings.DATABASES['default']['NAME'])
    lock = settings.RUNTIME_SNAPSHOT_LOCK_PATH
    targets = {
        'db.sqlite3': database,
        'db.sqlite3-wal': database.with_name(f'{database.name}-wal'),
        'db.sqlite3-shm': database.with_name(f'{database.name}-shm'),
        'ipso-inbox': settings.IPSO_INBOX_DIR,
        'observation-media': settings.OBSERVATION_MEDIA_DIR,
        'observation-thumbnails': settings.OBSERVATION_THUMBNAIL_DIR,
        'digests': settings.DIGEST_DIR,
    }
    if not args.yes:
        if not sys.stdin.isatty():
            parser.error('stop the local server and terrain worker, then pass --yes')
        response = input(f'Stop the local server and terrain worker, then type prod->local '
                         f'to replace runtime data in {data}: ')
        if response != 'prod->local':
            raise SystemExit('Aborted.')

    data.mkdir(parents=True, exist_ok=True)
    context = os.environ.get('ABIES_DOCKER_CONTEXT', 'vm-abies')
    remote = ['docker', '--context', context, 'exec', 'abies-prod']
    with TemporaryDirectory(prefix='.sync-prod-', dir=data) as directory:
        snapshot = fetch_snapshot(remote, Path(directory))
        with runtime_snapshot_lock(lock, exclusive=True):
            prepare_database(snapshot, database, settings)
            previous = Path(mkdtemp(prefix='.pre-sync-', dir=data))
            install_snapshot(snapshot, targets, previous)
    if any(previous.iterdir()):
        print(f'Previous local runtime data saved in {previous}.')
    else:
        previous.rmdir()
    print(f'Production snapshot restored in {data}.')


if __name__ == '__main__':
    main()
