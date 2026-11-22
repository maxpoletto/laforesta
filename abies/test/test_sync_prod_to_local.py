import json
import os
import runpy
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'bin/sync-prod-to-local.py'


@pytest.mark.parametrize('legacy_map', [False, True])
def test_sync_restores_production_runtime_into_local_checkout(transactional_db, tmp_path, legacy_map):
    from django.contrib.sessions.models import Session
    from django.db import connection
    from django.db.migrations.executor import MigrationExecutor
    from django.utils import timezone

    source = tmp_path / 'source'
    source.mkdir()
    executor = MigrationExecutor(connection)
    latest = executor.loader.graph.leaf_nodes()
    target = [('base', '0033_refresh_calendar_digest')] if legacy_map else latest
    try:
        models = executor.migrate(target).apps
        region_fields = {} if legacy_map else {'map_key': uuid4()}
        region = models.get_model('base', 'Region').objects.create(name='Production forest', **region_fields)
        eclass = models.get_model('base', 'Eclass').objects.create(name='Test class')
        parcel = models.get_model('base', 'Parcel').objects.create(
            name='1', region=region, eclass=eclass, area_ha=3,
        )
        Session.objects.create(session_key='prod-session', expire_date=timezone.now())
        with closing(sqlite3.connect(source / 'db.sqlite3')) as production:
            connection.connection.backup(production)
    finally:
        if legacy_map:
            MigrationExecutor(connection).migrate(latest)

    geometry = {'type': 'Polygon', 'coordinates': [[[16, 38], [16.01, 38], [16, 38.01], [16, 38]]]}
    map_file = tmp_path / 'terreni.geojson'
    map_file.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'geometry': geometry, 'properties': {'layer': 'Production forest', 'name': '1'},
    }]}))
    for directory, filename in [('ipso-inbox', 'walk.json'), ('observation-media', 'photo.jpg')]:
        (source / directory).mkdir()
        (source / directory / filename).write_text(filename)
    subprocess.run([
        sys.executable, str(ROOT / 'bin/backup'), '--data-dir', str(source),
        '--backup-dir', str(tmp_path), '--instance', 'prod', '--no-prune',
    ], check=True)
    archive, = tmp_path.glob('abies-prod-*.tar.gz')

    local = tmp_path / 'local'
    local.mkdir()
    (local / 'db.sqlite3').write_bytes(b'')
    (local / 'digests').mkdir()
    (local / 'digests' / 'stale.json.gz').write_bytes(b'stale local digest')
    (local / 'branding').mkdir()
    (local / 'branding' / 'keep').touch()

    (tmp_path / 'sync_settings.py').write_text('''
import os
from pathlib import Path
from config.test_settings import *

DATA_DIR = Path(os.environ['TEST_SYNC_DATA'])
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': DATA_DIR / 'db.sqlite3'}}
DIGEST_DIR = DATA_DIR / 'digests'
IPSO_INBOX_DIR = DATA_DIR / 'ipso-inbox'
OBSERVATION_MEDIA_DIR = DATA_DIR / 'observation-media'
OBSERVATION_THUMBNAIL_DIR = DATA_DIR / 'observation-thumbnails'
RUNTIME_SNAPSHOT_LOCK_PATH = DATA_DIR / '.runtime-snapshot.lock'
ABIES_INSTANCE = 'local'
''')
    docker = tmp_path / 'docker'
    docker.write_text('''#!/bin/sh
set -eu
test "$1 $2 $3 $4" = '--context vm-test exec abies-prod'
case "$7" in
  *'bin/backup --instance prod'*) cat "$TEST_SYNC_ARCHIVE" ;;
  *'/app/data/geo/terreni.geojson'*) test "$TEST_SYNC_LEGACY" = '1'; cat "$TEST_SYNC_MAP" ;;
  *) exit 2 ;;
esac
''')
    docker.chmod(0o700)
    result = subprocess.run([sys.executable, str(SCRIPT), '--yes'], capture_output=True, text=True, env={
        **os.environ,
        'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
        'PYTHONPATH': str(tmp_path),
        'TEST_SYNC_DATA': str(local),
        'TEST_SYNC_ARCHIVE': str(archive),
        'TEST_SYNC_LEGACY': str(int(legacy_map)),
        'TEST_SYNC_MAP': str(map_file),
        'DJANGO_SETTINGS_MODULE': 'sync_settings',
        'ABIES_DOCKER_CONTEXT': 'vm-test',
    })
    assert result.returncode == 0, result.stdout + result.stderr
    with closing(sqlite3.connect(local / 'db.sqlite3')) as database:
        assert database.execute('SELECT name FROM base_region').fetchall() == [('Production forest',)]
        assert database.execute('SELECT COUNT(*) FROM django_session').fetchone() == (0,)
        features = database.execute('SELECT parcel_id, geometry FROM base_mapfeature').fetchall()
        assert [(parcel_id, json.loads(shape)) for parcel_id, shape in features] == (
            [(parcel.pk, geometry)] if legacy_map else []
        )
    assert (local / 'observation-media' / 'photo.jpg').read_text() == 'photo.jpg'
    assert (local / 'ipso-inbox' / 'walk.json').read_text() == 'walk.json'
    assert not (local / 'digests' / 'stale.json.gz').exists()
    assert list((local / 'digests').glob('*.json.gz'))
    assert (local / 'branding' / 'keep').exists()
    previous, = local.glob('.pre-sync-*')
    assert (previous / 'digests' / 'stale.json.gz').exists()
    assert (previous / 'db.sqlite3').exists()
    assert not list(local.glob('.sync-prod-*'))


def test_failed_publication_restores_previous_runtime(tmp_path, monkeypatch):
    install_snapshot = runpy.run_path(str(SCRIPT))['install_snapshot']
    local, snapshot, previous = (tmp_path / name for name in ('local', 'snapshot', 'previous'))
    for path in (local, snapshot, previous):
        path.mkdir()
    targets = {name: local / name for name in ['db.sqlite3', 'ipso-inbox']}
    for name in targets:
        (local / name).write_text('local')
        (snapshot / name).write_text('prod')

    move = shutil.move

    def fail_on_install(source, target):
        if source == snapshot / 'ipso-inbox':
            raise OSError('cannot install inbox')
        return move(source, target)

    monkeypatch.setattr(shutil, 'move', fail_on_install)
    with pytest.raises(OSError, match='cannot install inbox'):
        install_snapshot(snapshot, targets, previous)
    assert [path.read_text() for path in targets.values()] == ['local', 'local']
