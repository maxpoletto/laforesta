"""Initial map import, independent feature history, and legacy upgrades."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import StringIO
from pathlib import Path
from threading import Event

import pytest
from django.core.management import call_command, CommandError
from django.db import connection, connections, IntegrityError
from django.db.migrations.executor import MigrationExecutor

from apps.base.map_data import map_geojson
from apps.base.models import MapCollection, MapFeature
from apps.base.parcel_geometry import ParcelGeometryIndex


def feature(name, x=0):
    return {'type': 'Feature', 'id': 'survey-' + str(x),
            'properties': {'layer': 'Capistrano', 'name': 'Capistrano-' + name, 'id': 123},
            'geometry': {'type': 'Polygon', 'coordinates': [
                [[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]],
            ]}}


def test_initial_import_preserves_pieces_and_history(tmp_path, parcels):
    path = tmp_path / 'map.geojson'
    first = feature('1')
    first['geometry']['coordinates'].append([[.1, .1], [.1, .2], [.2, .2], [.1, .1]])
    second = feature('1', 3)
    ridge = [[[8, 0, 712], [9, 0, 714], [8, 1, 718], [8, 0, 712]]]
    second['geometry'] = {'type': 'MultiPolygon', 'coordinates': [second['geometry']['coordinates'], ridge]}
    unlinked = feature('unknown', 5)
    data = {'type': 'FeatureCollection', 'name': 'Forest', 'features': [first, second, unlinked]}
    path.write_text(json.dumps(data))
    report = StringIO()
    call_command('build_geo', tmp_path, stdout=report)
    assert '1 parcels matched, 2 DB parcels without geometry' in report.getvalue()
    call_command('build_geo', path)
    path.unlink()
    a, b, c = MapFeature.objects.all()
    assert [a.parcel_id, b.parcel_id, c.parcel_id] == [parcels[0].pk, parcels[0].pk, None]
    first['properties']['coppice'] = second['properties']['coppice'] = False
    assert map_geojson() == data
    index = ParcelGeometryIndex()
    assert index.parcel_at(.5, 3.5) == parcels[0]
    assert index.parcel_at(.1, 8.1) == parcels[0]
    assert index.parcel_at(.5, 7) is None
    assert index.parcel_at(.15, .14) is None
    old = a.geometry.copy()
    a.geometry = feature('1', 2)['geometry']
    a.version += 1
    a.save()
    assert list(a.history.values_list('version', flat=True)) == [2, 1]
    assert a.history.last().geometry == old
    assert b.history.count() == c.history.count() == 1
    path.write_text(json.dumps(data))
    with pytest.raises(CommandError, match='different map'):
        call_command('build_geo', path)
    assert map_geojson()['features'][0]['geometry'] == a.geometry


def test_optional_invalid_and_map_only_install(tmp_path, db):
    call_command('build_geo', tmp_path)
    assert not MapCollection.objects.exists()
    path = tmp_path / 'terreni.geojson'
    data = {'type': 'FeatureCollection', 'features': [feature('1')]}
    path.write_text(json.dumps({**data, 'features': [feature('1'), {'type': 'Feature'}]}))
    with pytest.raises(CommandError, match='feature 2'):
        call_command('build_geo', tmp_path)
    assert not MapFeature.objects.exists() and not MapCollection.objects.exists()
    path.write_text(json.dumps(data))
    call_command('build_geo', tmp_path)
    assert map_geojson() == data


@pytest.mark.parametrize('geometry', [
    {'type': 'LineString', 'coordinates': [[0, 0], [1, 1]]},
    {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1]]]},
    {'type': 'Polygon', 'coordinates': [[[0, 0], [181, 0], [1, 1], [0, 0]]]},
    {'type': 'Polygon', 'coordinates': [[[0, 0], [0, -91], [1, 1], [0, 0]]]},
    {'type': 'Polygon', 'coordinates': [[[0, 0], [True, 0], [1, 1], [0, 0]]]},
    {'type': 'MultiPolygon', 'coordinates': [feature('1')['geometry']['coordinates'], []]},
], ids=['line', 'open-ring', 'longitude', 'latitude', 'boolean-ordinate', 'empty-second-polygon'])
def test_invalid_second_geometry_leaves_no_partial_map(geometry, tmp_path, db):
    path = tmp_path / 'map.geojson'
    source = [feature('1'), {**feature('2'), 'geometry': geometry}]
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': source}))
    with pytest.raises(CommandError, match='feature 2'):
        call_command('build_geo', tmp_path)
    assert not MapCollection.objects.exists()
    assert not MapFeature.objects.exists()
    assert not MapFeature.history.exists()


def test_preferred_map_takes_precedence_over_legacy_file(tmp_path, db):
    path = tmp_path / 'map.geojson'
    source = {'type': 'FeatureCollection', 'features': [feature('1')]}
    (tmp_path / 'terreni.geojson').write_text(json.dumps(source))
    path.write_text('invalid')
    with pytest.raises(CommandError, match='map.geojson'):
        call_command('build_geo', tmp_path)
    assert not MapCollection.objects.exists()
    source['features'] = [feature('2')]
    path.write_text(json.dumps(source))
    call_command('build_geo', tmp_path)
    assert map_geojson() == source


@pytest.mark.parametrize('member', [None, 'properties', 'geometry'], ids=['collection', 'properties', 'geometry'])
def test_retry_distinguishes_json_booleans_from_numbers(member, tmp_path, db):
    path = tmp_path / 'map.geojson'
    source = {'type': 'FeatureCollection', 'features': [feature('1')]}
    metadata = source['features'][0][member] if member else source
    metadata['survey'] = {'flags': [True], 'scale': 1}
    path.write_text(json.dumps(source))
    call_command('build_geo', path)
    metadata['survey']['scale'] = 1.0
    path.write_text(json.dumps(source, sort_keys=True))
    call_command('build_geo', path)
    metadata['survey']['flags'] = [1]
    path.write_text(json.dumps(source))
    with pytest.raises(CommandError, match='different map'):
        call_command('build_geo', path)
    stored = map_geojson()
    metadata = stored['features'][0][member] if member else stored
    assert metadata['survey']['flags'][0] is True
    assert MapFeature.history.count() == 1


@pytest.mark.django_db(transaction=True)
def test_upgrade_rolls_back_invalid_map_then_migrates_and_serves_it(settings, monkeypatch):
    previous = [('base', '0034_road_types')]
    target = ('base', '0035_map_features')
    executor = MigrationExecutor(connection)
    latest = executor.loader.graph.leaf_nodes()
    monkeypatch.delenv('ABIES_LEGACY_GEO_DIR', raising=False)
    directory = settings.GEO_DIR
    directory.mkdir()
    path = directory / 'terreni.geojson'
    try:
        state = executor.migrate(previous).apps
        region = state.get_model('base', 'Region').objects.create(name='Capistrano')
        eclass = state.get_model('base', 'Eclass').objects.create(name='Old F', coppice=True)
        parcel = state.get_model('base', 'Parcel').objects.create(
            name='01', region=region, eclass=eclass, area_ha=3,
            intervention_interval=18, standards_per_ha=75)
        linked = feature('01')
        linked['properties']['coppice'] = True
        unknown = feature('X', 4)
        source = {'type': 'FeatureCollection', 'name': 'Installed map', 'features': [linked, unknown]}
        path.write_text(json.dumps({**source, 'features': [linked, {'type': 'Feature'}]}))
        with pytest.raises(ValueError, match='feature 2'):
            MigrationExecutor(connection).migrate([target])
        assert target not in MigrationExecutor(connection).loader.applied_migrations
        assert 'base_mapfeature' not in connection.introspection.table_names()

        path.write_text(json.dumps(source))
        MigrationExecutor(connection).migrate([target])
        assert json.loads(path.read_text()) == source
        path.unlink()
        assert map_geojson() == source
        pieces = list(MapFeature.objects.all())
        assert [p.parcel_id for p in pieces] == [parcel.pk, None]
        assert [(p.pk, p.geometry, 1, '+', 'initial-audit-baseline') for p in pieces] == list(
            MapFeature.history.order_by('id').values_list(
                'id', 'geometry', 'version', 'history_type', 'history_change_reason'))
    finally:
        # A failed assertion must also leave later tests at the current schema.
        path.unlink(missing_ok=True)
        MigrationExecutor(connection).migrate(latest)


@pytest.mark.django_db
def test_initial_import_can_retry_after_history_write_fails(tmp_path):
    path = tmp_path / 'map.geojson'
    source = {'type': 'FeatureCollection', 'features': [feature('1'), feature('2', 2)]}
    path.write_text(json.dumps(source))
    with connection.cursor() as cursor:
        cursor.execute("""CREATE TEMP TRIGGER fail_history_insert BEFORE INSERT ON base_historicalmapfeature
                          BEGIN SELECT RAISE(ABORT, 'history write failed'); END""")
        try:
            with pytest.raises(IntegrityError, match='history write failed'):
                call_command('build_geo', path)
        finally:
            cursor.execute('DROP TRIGGER fail_history_insert')
    assert not MapCollection.objects.exists()
    assert not MapFeature.objects.exists()
    assert not MapFeature.history.exists()
    call_command('build_geo', path)
    assert map_geojson() == source
    assert MapFeature.history.count() == 2


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('same_map', [True, False], ids=['identical', 'conflicting'])
def test_concurrent_initial_imports_publish_one_map(tmp_path, monkeypatch, same_map):
    # Use independent connections to a real WAL file, as on an installed instance.
    database = tmp_path / 'map.sqlite3'
    with closing(sqlite3.connect(database)) as snapshot:
        connection.connection.backup(snapshot)
        assert snapshot.execute('PRAGMA journal_mode=WAL').fetchone() == ('wal',)
    monkeypatch.setitem(connections.databases, 'default', {**connection.settings_dict, 'NAME': str(database)})
    first_write = Event()
    competing_write = Event()

    def run(path, first):
        def overlap(execute, sql, params, many, context):
            if not sql.startswith(('UPDATE', 'INSERT')):
                return execute(sql, params, many, context)
            if not first:
                assert first_write.wait(10)
                competing_write.set()
                return execute(sql, params, many, context)
            result = execute(sql, params, many, context)
            # Hold the first transaction's write lock until the other importer tries to write.
            first_write.set()
            assert competing_write.wait(10)
            return result

        try:
            report = StringIO()
            try:
                with connections['default'].execute_wrapper(overlap):
                    call_command('build_geo', path, stdout=report)
                result = 'duplicate' if 'already imported' in report.getvalue() else 'imported'
            except CommandError as exc:
                assert 'different map' in str(exc)
                result = 'conflict'
            return result, map_geojson(), MapFeature.history.count()
        finally:
            connections.close_all()

    sources = [{'type': 'FeatureCollection', 'features': [feature('1', x)]}
               for x in (0, 0 if same_map else 3)]
    paths = [tmp_path / name for name in ('first.geojson', 'second.geojson')]
    for path, source in zip(paths, sources):
        path.write_text(json.dumps(source))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, paths, [True, False]))
    assert results == [('imported', sources[0], 1),
                       ('duplicate' if same_map else 'conflict', sources[0], 1)]


def test_later_csv_bootstrap_links_map_only_import(tmp_path, db):
    from test.test_bootstrap import _make_dir

    folder = Path(_make_dir(tmp_path))
    path = folder / 'map.geojson'
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [feature('1')]}))
    call_command('build_geo', path)
    part = MapFeature.objects.get()
    assert part.parcel_id is None
    call_command('bootstrap', folder, '--check')
    part.refresh_from_db()
    assert part.parcel_id is None and part.version == 1
    call_command('bootstrap', folder)
    part.refresh_from_db()
    assert part.parcel.name == '1' and part.version == 2
    assert ParcelGeometryIndex().parcel_at(.5, .5) == part.parcel
    export_dir = tmp_path / 'export'
    call_command('build_geo', path, output_dir=export_dir)
    assert json.loads((export_dir / 'terreni.geojson').read_text()) == map_geojson()


def test_builders_default_to_the_database_map(tmp_path, parcels, import_parcel_map, monkeypatch):
    from unittest.mock import Mock

    from apps.bosco import satellite
    from apps.bosco.management.commands import build_map_context

    import_parcel_map(parcels[0])
    data = map_geojson()
    dates = Mock(return_value=[])
    build = Mock(return_value=({'features': []}, True))
    monkeypatch.setattr(satellite, 'find_region_dates', dates)
    monkeypatch.setattr(build_map_context, 'build_region', build)
    call_command('build_satellite', 'dates', '--output-dir', str(tmp_path))
    call_command('build_map_context', output_dir=tmp_path)
    assert dates.call_args.args[:2] == (data, parcels[0].region.name)
    assert build.call_args.args[:2] == (data, parcels[0].region.name)


def test_canonical_bootstrap_commits_map_and_catalog_together(tmp_path, db):
    from apps.base.models import Parcel, Region
    from test.test_bootstrap import _make_dir

    folder = Path(_make_dir(tmp_path))
    path = folder / 'map.geojson'
    path.write_text('{"type":"FeatureCollection","features":[{}]}')
    with pytest.raises(CommandError):
        call_command('bootstrap', folder)
    assert not Region.objects.exists() and not MapFeature.objects.exists()
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [feature('1')]}))
    call_command('bootstrap', folder, '--check')
    assert not Parcel.objects.exists() and not MapCollection.objects.exists()
    call_command('bootstrap', folder)
    assert MapFeature.objects.get().parcel == Parcel.objects.get(name='1')
