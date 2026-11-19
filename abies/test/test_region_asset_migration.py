"""Upgrade the generated files in place without fetching their sources."""
import gzip
import json
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.base.map_keys import initial_map_key
from apps.base.models import Region


def test_migrate_region_assets_can_resume_without_changing_mask_order(tmp_path, monkeypatch):
    migration = import_module('apps.base.migrations.0041_region_asset_keys')
    name = 'Alta-Costa'
    key = str(migration.initial_map_key(name))
    sat = tmp_path / 'satellite'
    context = tmp_path / 'context'
    source = sat / name
    source.mkdir(parents=True)
    raster = source / 'parcel-mask.tif'
    raster.write_bytes(b'original raster')
    inode = raster.stat().st_ino
    labels = ['Alta-Costa-10', 'Alta-Costa-Alta-Costa-1', 'Alta-Costa-2']
    (source / 'timeseries.json').write_text(json.dumps({
        'parcels': labels, 'means': {'parcels': dict(zip(labels, [10, 20, 30])), 'forest': [4]},
    }))
    legacy = context / name
    legacy.mkdir(parents=True)
    snapshot = {'schema_version': 1, 'region': name, 'build': {'geometry_hash': 'hash'}, 'features': []}
    (legacy / 'context.json.gz').write_bytes(gzip.compress(json.dumps(snapshot).encode()))
    # The summary was replaced, but the directory move failed.
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'rename', Mock(side_effect=OSError('Read-only filesystem')))
        with pytest.raises(OSError, match='Read-only'):
            migration.move_assets(sat, {name: key}, satellite=True)
    assert (source / 'timeseries.json').is_file()
    migration.move_assets(sat, {name: key}, satellite=True)
    # Simulate restarting after only one of the two roots was upgraded.
    migration.move_assets(sat, {name: key}, satellite=True)
    migration.move_assets(context, {name: key}, satellite=False)
    migration.move_assets(context, {name: key}, satellite=False)
    assert not source.exists() and not legacy.exists()
    assert (sat / key / 'parcel-mask.tif').stat().st_ino == inode
    converted = json.loads((sat / key / 'timeseries.json').read_bytes())
    assert converted['parcels'] == ['10', 'Alta-Costa-1', '2']
    assert converted['means'] == {'parcels': {'10': 10, 'Alta-Costa-1': 20, '2': 30}, 'forest': [4]}
    migrated = json.loads(gzip.decompress((context / key / 'context.json.gz').read_bytes()))
    assert migrated['region_key'] == key and migrated['build'] == snapshot['build']


def test_migration_refuses_to_overwrite_a_second_map(tmp_path):
    migration = import_module('apps.base.migrations.0041_region_asset_keys')
    key = str(migration.initial_map_key('Costa'))
    for directory in ('Costa', key):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / 'manifest.json').write_text(directory)
    with pytest.raises(ValueError, match='already exists'):
        migration.move_assets(tmp_path, {'Costa': key}, satellite=True)
    assert (tmp_path / 'Costa' / 'manifest.json').read_text() == 'Costa'
    assert (tmp_path / key / 'manifest.json').read_text() == key


@pytest.mark.django_db(transaction=True)
def test_upgrade_links_renamed_region_and_pre_catalog_build(settings):
    executor = MigrationExecutor(connection)
    latest = executor.loader.graph.leaf_nodes()
    previous = [('base', '0040_editable_regions')]
    executor.migrate(previous)
    try:
        old = executor.loader.project_state(previous).apps.get_model('base', 'Region')
        region = old.objects.create(name='New name', map_directory='Original name')
        for name in ('Original name', 'No catalog yet'):
            directory = settings.SATELLITE_DIR / name
            directory.mkdir(parents=True)
            (directory / 'dates.txt').write_text('2026-01-01\n')

        MigrationExecutor(connection).migrate(latest)
        migrated = Region.objects.get(pk=region.pk)
        adopted = Region.objects.create(name='No catalog yet')
        assert migrated.map_key == initial_map_key('Original name')
        for item in (migrated, adopted):
            assert (settings.SATELLITE_DIR / str(item.map_key) / 'dates.txt').read_text() == '2026-01-01\n'
        assert not (settings.SATELLITE_DIR / 'Original name').exists()
    finally:
        MigrationExecutor(connection).migrate(latest)
