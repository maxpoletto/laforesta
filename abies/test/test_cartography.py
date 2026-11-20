"""The road workflow starts with an actual field upload."""
import uuid
from base64 import b64encode

import pytest
from django.urls import reverse

from apps.ipso.models import IpsoUpload


ROOT = '/api/impostazioni/cartography/'


def write(client, region, action, **values):
    return client.post(f'{ROOT}{region.pk}/{action}/',
                       {'nonce': str(uuid.uuid4()), **values}, content_type='application/json')


def workspace(client, region):
    response = client.get(f'{ROOT}{region.pk}/')
    assert response.status_code == 200, response.content
    return response.json()


def import_file(client, region, fmt, text, **values):
    return write(client, region, 'import-file', name='Posizioni', format=fmt,
                 file=b64encode(text.encode()).decode(), **values)


def test_import_geojson_and_gpx_keep_distinct_parts(client, writer_user, regions):
    from apps.base.models import MapDrawing

    client.force_login(writer_user)
    region = regions[0]
    polygon = '[[[16,38],[17,38],[17,39],[16,38]],[[16.2,38.1],[16.8,38.1],[16.8,38.6],[16.2,38.1]]]'
    geojson = ('{"type":"GeometryCollection","geometries":[{"type":"MultiPoint",'
               '"coordinates":[[16,38,123],[17,39]]},{"type":"Polygon","coordinates":' + polygon + '}]}')
    response = import_file(client, region, 'geojson', geojson)
    assert response.status_code == 200, response.content
    parts = MapDrawing.objects.get(pk=response.json()['drawing_id']).parts
    assert [part['type'] for part in parts] == ['Point', 'Point', 'Polygon']
    assert parts[0]['coordinates'] == [16, 38] and len(parts[-1]['coordinates']) == 2

    gpx = '''<gpx xmlns="http://www.topografix.com/GPX/1/1">
        <wpt lat="38" lon="16"/><trk><trkseg>
          <trkpt lat="38.1" lon="16"/><trkpt lat="38.2" lon="16"/>
        </trkseg><trkseg><trkpt lat="38.3" lon="16"/></trkseg></trk></gpx>'''
    response = import_file(client, region, 'gpx', gpx)
    assert response.status_code == 200, response.content
    parts = MapDrawing.objects.get(pk=response.json()['drawing_id']).parts
    assert [part['type'] for part in parts] == ['Point', 'LineString', 'Point']
    assert parts[1]['coordinates'] == [[16, 38.1], [16, 38.2]]


def test_dxf_import_requires_crs_and_projects_coordinates(client, writer_user, regions):
    from apps.base.models import MapDrawing

    client.force_login(writer_user)
    region = regions[0]
    dxf = '0\nSECTION\n2\nENTITIES\n0\nLINE\n10\n0\n20\n0\n11\n111319.490793\n21\n111325.142866\n0\nENDSEC\n0\nEOF\n'
    assert import_file(client, region, 'dxf', dxf).status_code == 400
    assert import_file(client, region, 'dxf', dxf, epsg='999999').status_code == 400
    assert import_file(client, region, 'dxf', dxf, epsg='4978').status_code == 400
    assert import_file(client, region, 'dxf', dxf.replace('111319.490793', '1e100'), epsg='3857').status_code == 400
    assert import_file(client, region, 'dxf', dxf.replace('LINE', 'ARC'), epsg='3857').status_code == 400
    assert not MapDrawing.objects.exists()
    response = import_file(client, region, 'dxf', dxf, epsg='3857')
    assert response.status_code == 200, response.content
    part, = MapDrawing.objects.get(pk=response.json()['drawing_id']).parts
    assert part['type'] == 'LineString'
    assert part['coordinates'][0] == [0, 0]
    assert part['coordinates'][1] == pytest.approx([1, 1])
    ring = ('0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n90\n3\n70\n1\n'
            '10\n16\n20\n38\n10\n17\n20\n38\n10\n17\n20\n39\n0\nENDSEC\n0\nEOF\n')
    response = import_file(client, region, 'dxf', ring, epsg='4326')
    assert response.status_code == 200, response.content
    assert MapDrawing.objects.get(pk=response.json()['drawing_id']).parts == [
        {'type': 'Polygon', 'coordinates': [[[16, 38], [17, 38], [17, 39], [16, 38]]]},
    ]


@pytest.mark.parametrize(('fmt', 'text'), [
    ('geojson', '{"type":"Point","coordinates":[25,390]}'),
    ('geojson', '{"type":"MultiPoint","coordinates":[[20,39],[21,"39"]]}'),
    ('gpx', '<!DOCTYPE gpx><gpx><wpt lat="38" lon="16"/></gpx>'),
    ('gpx', '<gpx><wpt lat="38"/></gpx>'),
])
def test_bad_import_files_do_not_create_drawings(client, writer_user, regions, fmt, text):
    client.force_login(writer_user)
    assert import_file(client, regions[0], fmt, text).status_code == 400
    assert workspace(client, regions[0])['drawings'] == []


def test_import_survey_uses_local_free_trees_or_structured_plot_centers(client, writer_user, parcels):
    from apps.base.models import MapDrawing, Sample, SampleArea, SampleGrid, Species, Survey, Tree, TreeSample

    client.force_login(writer_user)
    parcel, _, distant = parcels
    region = parcel.region
    species = Species.objects.create(common_name='Abete di prova')
    free = Survey.objects.create(name='Libero')
    sample = Sample.objects.create(survey=free, date='2026-11-19')

    def tree_at(target_sample, where, number, **gps):
        tree = Tree.objects.create(species=species)
        return TreeSample.objects.create(sample=target_sample, parcel=where, tree=tree, number=number,
                                         d_cm=20, h_m=12, **gps)

    tree_at(sample, parcel, 1, lon=16, lat=38, acc_m=2, note='Albero')
    tree_at(sample, parcel, 2)
    tree_at(sample, distant, 3, lon=17, lat=39)
    grid = SampleGrid.objects.create(name='Griglia')
    structured = Survey.objects.create(name='Predefinito', sample_grid=grid)
    plot = SampleArea.objects.create(sample_grid=grid, parcel=parcel, number='1', lon=16.2, lat=38.2)
    visited = Sample.objects.create(survey=structured, sample_area=plot, date='2026-11-19')
    tree_at(visited, parcel, 4, lon=16.25, lat=38.25)
    SampleArea.objects.create(sample_grid=grid, parcel=parcel, number='2', lon=16.3, lat=38.3)
    empty = Survey.objects.create(name='Senza coordinate')
    result = client.get(f'{ROOT}{region.pk}/surveys/')
    assert result.status_code == 200, result.content
    assert result.json()['surveys'] == [
        {'id': free.pk, 'name': free.name, 'structured': False, 'count': 1},
        {'id': structured.pk, 'name': structured.name, 'structured': True, 'count': 1},
    ]
    for survey, coords, fix in [(free, [16, 38], [2, 'Albero']), (structured, [16.2, 38.2], [None, ''])]:
        response = write(client, region, 'import-survey', survey_id=survey.pk)
        assert response.status_code == 200, response.content
        drawing = MapDrawing.objects.get(pk=response.json()['drawing_id'])
        assert drawing.name == survey.name
        assert drawing.parts == [{'type': 'Point', 'coordinates': coords, 'fixes': [fix]}]
    assert write(client, distant.region, 'import-survey', survey_id=structured.pk).status_code == 400
    assert write(client, region, 'import-survey', survey_id=empty.pk).status_code == 400
    original = MapDrawing.objects.get(region=region, name=free.name)
    assert write(client, region, 'remove-drawing', id=original.pk, version=original.version).status_code == 200
    assert write(client, region, 'import-survey', survey_id=free.pk).status_code == 200
    assert MapDrawing.objects.get(region=region, name=free.name).parts == original.parts


@pytest.fixture
def walk(client, writer_user, settings, tmp_path, regions):
    client.force_login(writer_user)
    settings.IPSO_SECRET = 'field-token'
    settings.IPSO_INBOX_DIR = tmp_path
    session_id = str(uuid.uuid4())
    tail = str(uuid.uuid4())
    segments = [session_id, session_id, None, tail, tail]
    rows = [{
        'client_record_id': str(i + 1), 'date': '2026-11-18', 'segment': segment,
        'lat': 38 + i / 100, 'lon': 16, 'acc_m': 2.3, 'taken_at': '2026-11-18T09:00:00Z',
        'note': '' if segment else 'Cancello',
    } for i, segment in enumerate(segments)]
    response = client.post(reverse('ipso-upload-page'), {
        'session': {'session_id': session_id, 'mode': 'points', 'schema_version': 1,
                    'damaged': False, 'name': 'Pista nord', 'region_id': regions[0].pk},
        'records': rows, 'offset': 0, 'total': len(rows), 'transfer': 'a' * 64,
    }, content_type='application/json', HTTP_AUTHORIZATION='Bearer field-token')
    assert response.status_code == 200, response.content
    return IpsoUpload.objects.get(session_id=session_id)


def test_offline_walk_requires_a_region_id_and_can_recover_a_deleted_region(
        client, writer_user, regions, settings, tmp_path):
    from apps.base.models import MapPointSet, Region

    client.force_login(writer_user)
    settings.IPSO_SECRET = 'field-token'
    settings.IPSO_INBOX_DIR = tmp_path
    old_region = Region.objects.create(name='Compresa cancellata')
    old_id = old_region.pk
    old_region.delete()
    session_id = str(uuid.uuid4())
    session = {'session_id': session_id, 'mode': 'points', 'schema_version': 1, 'damaged': False, 'name': 'Prima'}
    data = {'session': session, 'records': [
        {'client_record_id': '1', 'date': '2026-11-18', 'lat': 38, 'lon': 16, 'acc_m': 2,
         'taken_at': '2026-11-18T10:00:00Z'},
    ], 'offset': 0, 'total': 1, 'transfer': 'a' * 64}
    url = reverse('ipso-upload-page')
    for invalid in [None, 0, -1, True, 2**53]:
        session['region_id'] = invalid
        assert client.post(url, data, content_type='application/json',
                           HTTP_AUTHORIZATION='Bearer field-token').status_code == 422
    assert not IpsoUpload.objects.exists()
    session['region_id'] = old_id
    response = client.post(url, data, content_type='application/json', HTTP_AUTHORIZATION='Bearer field-token')
    assert response.status_code == 200, response.content
    upload = IpsoUpload.objects.get(session_id=session_id)
    detail_url = f'/api/ipso/uploads/{upload.pk}/'
    detail = client.get(detail_url).json()
    assert detail['session']['region_id'] == old_id
    assert {row['id'] for row in detail['regions']} == {region.pk for region in regions}
    response = write(client, regions[0], 'import', upload_id=upload.pk, name='Dopo')
    assert response.status_code == 200, response.content
    source = MapPointSet.objects.get(pk=response.json()['source_id'])
    assert source.region == regions[0] and source.name == 'Dopo'
    assert source.records[0]['lon'] == 16
    detail = client.get(detail_url).json()
    assert detail['session']['name'] == 'Prima' and 'regions' not in detail


def test_import_restore_after_inbox_cleanup(client, walk, regions, admin_user):
    region, other = regions[:2]
    assert workspace(client, region)['uploads'][0]['id'] == walk.pk
    assert workspace(client, other)['uploads'] == []
    assert write(client, other, 'import', upload_id=walk.pk).status_code == 400
    assert write(client, region, 'import', upload_id=walk.pk).status_code == 200
    state = workspace(client, region)
    assert state['uploads'] == []
    source, = state['sources']
    line, = state['drawings']
    first, gate, last = line['parts']
    assert first['coordinates'] == [[16, 38], [16, 38.01]]
    assert gate['type'] == 'Point' and last['type'] == 'LineString'
    assert gate['fixes'] == [[2.3, 'Cancello']]
    edited = [{**first, 'coordinates': [[16.1, 38], [16, 38.01]]}, gate]
    assert write(client, region, 'adjust', id=line['id'], version=line['version'], parts=edited).status_code == 200
    assert write(client, region, 'adjust', id=line['id'], version=line['version'], parts=edited).status_code == 400
    client.force_login(admin_user)
    assert client.post(reverse('ipso-upload-delete', args=[walk.pk]), {'nonce': str(uuid.uuid4())},
                       content_type='application/json').status_code == 200
    assert write(client, region, 'restore', source_id=source['id']).json()['added'] == 0
    drawings = workspace(client, region)['drawings']
    assert drawings[0]['parts'] == edited
    versions = [[d['id'], d['version']] for d in drawings]
    assert write(client, region, 'clear', versions=versions).status_code == 200
    assert write(client, region, 'restore', source_id=source['id']).json()['added'] == 1
    assert workspace(client, region)['drawings'][0]['parts'] == line['parts']


def test_region_and_edit_guards(client, walk, regions, reader_user):
    region, other = regions[:2]
    assert write(client, region, 'import', upload_id=walk.pk).status_code == 200
    state = workspace(client, region)
    line = state['drawings'][0]
    for geometry in ({}, {'type': 'Point', 'coordinates': [16]},
                     {'type': 'LineString', 'coordinates': [[False, 38], [16, 91]]}):
        assert write(client, region, 'adjust', id=line['id'], version=1, parts=[geometry]).status_code == 400
    assert write(client, region, 'clear', versions=[]).status_code == 400
    assert write(client, other, 'restore', source_id=state['sources'][0]['id']).status_code == 404
    client.force_login(reader_user)
    assert client.get(f'{ROOT}{region.pk}/').status_code == 403
    assert write(client, region, 'clear', versions=[]).status_code == 403


def test_walk_to_published_road(client, walk, regions):
    region = regions[0]
    assert write(client, region, 'import', upload_id=walk.pk).status_code == 200
    state = workspace(client, region)
    drawing, = state['drawings']
    first, gate, last = drawing['parts']
    kind = client.get(ROOT).json()['road_types'][0]['id']
    road_data = {'id': drawing['id'], 'name': 'Via nord', 'road_type_id': kind, 'version': 1}
    assert write(client, region, 'road', **road_data).status_code == 400
    copy = write(client, region, 'copy', id=drawing['id'], version=1, parts=[last], name='Laterale')
    assert copy.status_code == 200
    joined = {'type': 'LineString', 'coordinates': first['coordinates'] + last['coordinates']}
    response = write(client, region, 'road', **road_data, parts=[joined])
    assert response.status_code == 200, response.content
    road, = client.get('/api/geo/roads.geojson').json()['features']
    assert road['geometry'] == joined and road['properties']['road_type_id'] == kind
    copied, = workspace(client, region)['drawings']
    assert copied['name'] == 'Laterale' and copied['parts'] == [last]
    assert write(client, region, 'restore', source_id=state['sources'][0]['id']).json()['added'] == 1
    assert workspace(client, region)['drawings'][-1]['parts'] == drawing['parts']


def test_publish_empty_region_for_field_device(client, writer_user, settings):
    settings.IPSO_SECRET = 'field-token'
    client.force_login(writer_user)
    auth = {'HTTP_AUTHORIZATION': 'Bearer field-token'}
    original = client.get('/ipso/reference.json', **auth).json()
    response = client.post(ROOT + 'regions/', {'name': 'Costa nuova', 'nonce': str(uuid.uuid4())},
                           content_type='application/json')
    assert response.status_code == 200
    region = {'id': response.json()['region_id'], 'name': 'Costa nuova'}
    assert region in client.get(ROOT).json()['regions']
    ref = client.get('/ipso/reference.json', HTTP_AUTHORIZATION='Bearer field-token').json()
    assert ref['reference_version'] != original['reference_version']
    assert region in ref['regions']
    assert ref['parcels'] == []


@pytest.mark.django_db(transaction=True)
def test_point_import_checks_staged_file_before_reserving_writer(client, regions, walk, monkeypatch):
    from django.db import connection
    from apps.impostazioni import cartography

    read = cartography.read_staged_payload

    def read_without_lock(upload):
        assert not connection.in_atomic_block
        return read(upload)

    monkeypatch.setattr(cartography, 'read_staged_payload', read_without_lock)
    body = {'nonce': str(uuid.uuid4()), 'upload_id': walk.pk}
    url = f'{ROOT}{regions[0].pk}/import/'
    first = client.post(url, body, content_type='application/json')
    assert first.status_code == 200
    assert client.post(url, body, content_type='application/json').json() == first.json()
    assert len(workspace(client, regions[0])['drawings']) == 1


def test_remove_working_copy_then_restore_whole_walk(client, regions, walk):
    region = regions[0]
    write(client, region, 'import', upload_id=walk.pk)
    state = workspace(client, region)
    drawing, = state['drawings']
    assert write(client, region, 'remove-drawing', id=drawing['id'], version=2).status_code == 400
    assert write(client, region, 'remove-drawing', id=drawing['id'], version=1).status_code == 200
    assert write(client, region, 'restore', source_id=state['sources'][0]['id']).json()['added'] == 1
    assert workspace(client, region)['drawings'][0]['parts'] == drawing['parts']


def region_details(client, region):
    return client.get(f'{ROOT}regions/{region.pk}/').json()


def test_region_editor_lifecycle(client, writer_user, regions):
    from apps.base.models import Region

    client.force_login(writer_user)
    created = client.post(ROOT + 'regions/', {'name': '  Abetaia  ', 'description': 'Versante nord'},
                          content_type='application/json')
    assert created.status_code == 200
    region = Region.objects.get(pk=created.json()['region_id'])
    details = region_details(client, region)
    assert details['name'] == 'Abetaia' and details['description'] == 'Versante nord'
    assert details['deletion'] is not None
    edit_url = f'{ROOT}regions/{region.pk}/save/'
    body = {'name': regions[0].name, 'description': 'Modificata', 'version': details['version']}
    assert client.post(edit_url, body, content_type='application/json').status_code == 400
    body['name'] = 'abeti nuovi'
    assert client.post(edit_url, body, content_type='application/json').status_code == 200
    assert client.post(edit_url, body, content_type='application/json').json()['status'] == 'conflict'
    names = [r['name'] for r in client.get(ROOT).json()['regions']]
    assert names == sorted(names, key=str.casefold)
    fresh = region_details(client, region)
    assert fresh['description'] == 'Modificata' and fresh['version'] == details['version'] + 1
    url = f'{ROOT}regions/{region.pk}/delete/'
    assert client.post(url, {'version': details['version'], 'review': details['deletion']['review']},
                       content_type='application/json').status_code == 400
    assert client.post(url, {'version': fresh['version'], 'review': fresh['deletion']['review']},
                       content_type='application/json').status_code == 200
    assert not Region.objects.filter(pk=region.pk).exists()


def test_region_delete_reviews_map_content_and_protects_fieldwork(client, writer_user, parcels):
    from apps.base.models import MapFeature, Observation, Parcel, RoadType

    client.force_login(writer_user)
    region = parcels[0].region
    preview = region_details(client, region)
    assert preview['deletion']['parcels'] == 2
    road = MapFeature.objects.create(region=region, road_type=RoadType.objects.first(),
                                     geometry={'type': 'LineString', 'coordinates': [[16, 38], [16, 39]]})
    url = f'{ROOT}regions/{region.pk}/delete/'

    def remove(preview):
        return client.post(url, {'version': preview['version'], 'review': preview['deletion']['review']},
                           content_type='application/json')

    assert remove(preview).status_code == 400
    fresh = region_details(client, region)
    assert fresh['deletion']['roads'] == 1
    obs = Observation.objects.create(region=region, date='2026-11-19', lat=38, lon=16)
    assert region_details(client, region)['deletion'] is None
    assert remove(fresh).status_code == 400
    assert MapFeature.objects.filter(pk=road.pk).exists()
    obs.delete()
    assert remove(fresh).status_code == 200
    assert not Parcel.objects.filter(region_id=region.pk).exists()
    assert not MapFeature.objects.filter(pk=road.pk).exists()
    assert Parcel.objects.filter(pk=parcels[-1].pk).exists()


def test_pending_and_imported_walk_protect_region(client, regions, walk):
    assert region_details(client, regions[0])['deletion'] is None
    assert write(client, regions[0], 'import', upload_id=walk.pk).status_code == 200
    assert region_details(client, regions[0])['deletion'] is None


def test_rename_preserves_parcels_and_built_map_data(client, writer_user, parcels, import_parcel_map, settings):
    from apps.base.map_data import map_geojson
    from apps.bosco.map_context_data import CONTEXT_FILE, SCHEMA_VERSION, publish_context, region_geometry
    from apps.bosco.satellite import parcel_snapshot_key
    from apps.base.models import MapFeature

    client.force_login(writer_user)
    region = parcels[0].region
    import_parcel_map(parcels[0])
    feature = MapFeature.objects.get(parcel=parcels[0])
    MapFeature.objects.create(members={'type': 'Feature', 'properties': {'layer': region.name, 'Particella': '99'}},
                              geometry=feature.geometry)
    key = region.map_key
    sat = settings.SATELLITE_DIR / str(key)
    sat.mkdir(parents=True)
    (sat / 'manifest.json').write_text('{"dates":[]}')
    import json
    key_id = str(parcels[0].pk)
    snapshot = parcel_snapshot_key([(parcels[0].pk, feature.geometry)])
    summary = json.dumps({'schema_version': 3, 'parcel_snapshot': snapshot,
                          'parcels': [key_id], 'means': {'parcels': {key_id: {'ndvi': [0.4]}}}})
    (sat / 'timeseries.json').write_text(summary)
    fingerprint, _ = region_geometry(map_geojson(), region.name)
    publish_context(settings.MAP_CONTEXT_DIR / str(key) / CONTEXT_FILE, {
        'schema_version': SCHEMA_VERSION, 'region_key': str(key), 'build': {'geometry_hash': fingerprint},
    })
    response = client.post(f'{ROOT}regions/{region.pk}/save/', {'version': 1, 'name': 'Nuova costa'},
                           content_type='application/json')
    assert response.status_code == 200
    assert 'parcels' in response.json()['invalidates']['data_ids']
    assert {f['properties']['layer'] for f in map_geojson()['features']} == {'Nuova costa'}
    assert client.get(f'/api/bosco/satellite/{region.pk}/manifest/').status_code == 200
    region.refresh_from_db()
    assert region.map_key == key
    series = client.get(f'/api/bosco/satellite/{region.pk}/timeseries/')
    assert series.content == summary.encode()
    context = client.get(reverse('bosco-map-context', args=[region.pk]))
    assert context.get('Content-Encoding') == 'gzip'


def test_satellite_data_follows_parcel_id_and_rejects_changed_geometry(client, writer_user, parcels, settings):
    import json

    import numpy as np

    from apps.base.models import MapFeature
    from apps.bosco.satellite import parcel_snapshot_key, save_geotiff

    client.force_login(writer_user)
    parcel = parcels[0]
    region = parcel.region
    polygon = {'type': 'Polygon', 'coordinates': [[[16, 38], [17, 38], [17, 39], [16, 38]]]}
    feature = MapFeature.objects.create(parcel=parcel, geometry=polygon)
    snapshot = parcel_snapshot_key([(parcel.pk, polygon)])
    directory = settings.SATELLITE_DIR / str(region.map_key)
    directory.mkdir(parents=True)
    (directory / 'timeseries.json').write_text(json.dumps({
        'schema_version': 3, 'parcel_snapshot': snapshot, 'means': {'parcels': {str(parcel.pk): {'ndvi': [0.4]}}},
    }))
    save_geotiff(directory / 'parcel-mask.tif', np.ones((2, 2), dtype=np.uint8), [16, 38, 17, 39], 2, 2,
                 tags={'parcel_snapshot': snapshot})
    base = f'/api/bosco/satellite/{region.pk}'
    urls = [f'{base}/timeseries/', f'{base}/raw/parcel-mask.json']
    assert write(client, region, 'parcel-properties', id=parcel.pk, version=parcel.version, name='Rinominata',
                 confirm_rename=True, eclass_id=parcel.eclass_id).status_code == 200
    etags = [client.get(url)['ETag'] for url in urls]
    assert client.get(urls[0]).json()['means']['parcels'] == {str(parcel.pk): {'ndvi': [0.4]}}
    review = client.get(f'{ROOT}{region.pk}/parcels/{parcel.pk}/boundaries/').json()['review']
    assert write(client, region, 'parcel-boundaries', id=parcel.pk, review=review,
                 edits=[{'id': feature.pk, 'part': 0, 'ring': 0, 'index': 1, 'operation': 'move',
                         'point': [16.9, 38]}]).status_code == 200
    for url, etag in zip(urls, etags, strict=True):
        assert client.get(url, HTTP_IF_NONE_MATCH=etag).status_code == 404


def test_plan_and_survey_references_prevent_region_deletion(client, writer_user, hypso_samples, parcels):
    from apps.base.models import HarvestPlan, HarvestPlanItem

    client.force_login(writer_user)
    assert region_details(client, hypso_samples['region'])['deletion'] is None
    region = parcels[-1].region
    review = region_details(client, region)
    plan = HarvestPlan.objects.create(name='Piano', year_start=2026, year_end=2030)
    item = HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcels[-1], year_planned=2028)
    assert region_details(client, region)['deletion'] is None
    assert client.post(f'{ROOT}regions/{region.pk}/delete/', {
        'version': review['version'], 'review': review['deletion']['review'],
    }, content_type='application/json').status_code == 400
    assert HarvestPlanItem.objects.filter(pk=item.pk).exists()
    item.delete()
    HarvestPlanItem.objects.create(harvest_plan=plan, region=region, damaged=True, year_planned=2028)
    assert region_details(client, region)['deletion'] is None


def test_partial_ipso_walk_protects_only_its_region(client, writer_user, settings, regions, tmp_path):
    client.force_login(writer_user)
    settings.IPSO_SECRET = 'field-token'
    settings.IPSO_INBOX_DIR = tmp_path
    walk_id = str(uuid.uuid4())
    session = {'session_id': walk_id, 'mode': 'points', 'schema_version': 1, 'damaged': False,
               'name': 'Strada', 'region_id': regions[0].pk}
    record = {'client_record_id': '1', 'date': '2026-11-19', 'segment': walk_id,
              'lat': 38, 'lon': 16, 'acc_m': 2, 'taken_at': '2026-11-19T09:00:00Z', 'note': ''}
    for offset in (0, 1):
        result = client.post(reverse('ipso-upload-page'), {
            'session': session, 'records': [{**record, 'client_record_id': str(offset + 1)}],
            'offset': offset, 'total': 2, 'transfer': 'a' * 64,
        }, content_type='application/json', HTTP_AUTHORIZATION='Bearer field-token')
        assert result.status_code == 200, result.content
        assert region_details(client, regions[0])['deletion'] is None
        assert region_details(client, regions[1])['deletion'] is not None


def test_new_compresa_can_adopt_installed_outlines(client, writer_user, settings):
    from apps.base.models import MapFeature, Region

    client.force_login(writer_user)
    name = 'Bosco alto'
    feature = MapFeature.objects.create(members={'properties': {'Compresa': name, 'Particella': 'A'}},
                                        geometry={'type': 'Polygon', 'coordinates': [[[16, 38], [17, 38],
                                                                                      [16, 39], [16, 38]]]})
    from apps.base.map_keys import initial_map_key
    directory = settings.SATELLITE_DIR / str(initial_map_key(name))
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text('{"dates":[]}')
    result = client.post(ROOT + 'regions/', {'name': name}, content_type='application/json')
    region = Region.objects.get(pk=result.json()['region_id'])
    assert client.get(f'/api/bosco/satellite/{region.pk}/manifest/').status_code == 200
    review = region_details(client, region)
    assert review['deletion']['outlines'] == 1
    assert client.post(f'{ROOT}regions/{region.pk}/delete/', {
        'version': review['version'], 'review': review['deletion']['review'],
    }, content_type='application/json').status_code == 200
    assert not MapFeature.objects.filter(pk=feature.pk).exists()
    result = client.post(ROOT + 'regions/', {'name': name}, content_type='application/json')
    replacement = Region.objects.get(pk=result.json()['region_id'])
    assert replacement.map_key != region.map_key
    assert client.get(f'/api/bosco/satellite/{replacement.pk}/manifest/').status_code == 404


def test_pending_legacy_page_with_known_region_leaves_other_regions_editable(client, writer_user, regions):
    from apps.ipso.models import IpsoUploadPage
    from apps.ipso.upload_pages import _page_body

    client.force_login(writer_user)
    IpsoUploadPage.objects.create(session_id='legacy', offset=0, end=1, transfer='a' * 64, checksum='b' * 64,
                                  body=_page_body([{'region_id': regions[0].id}], '', {}))
    assert region_details(client, regions[0])['deletion'] is None
    assert region_details(client, regions[1])['deletion'] is not None


def test_parcel_promotion_lists_destinations_in_natural_order(client, writer_user, parcels):
    import re

    from apps.base.models import Parcel

    client.force_login(writer_user)
    parcel = parcels[0]
    for name in ['10', '2b', '2a']:
        Parcel.objects.create(region=parcel.region, eclass=parcel.eclass, name=name)
    response = client.get(f'{ROOT}{parcel.region_id}/parcel-form/')
    assert response.status_code == 200
    options = re.search(r'<optgroup[^>]*>(.*?)</optgroup>', response.json()['html'], re.S).group(1)
    assert re.findall(r'<option[^>]*>([^<]+)</option>', options) == ['1', '2', '2a', '2b', '10']


def test_parcel_form_uses_canonical_machine_values(client, writer_user, regions, settings):
    from django.utils.formats import number_format
    from django.utils.translation import override

    from apps.base.models import Eclass, Parcel

    client.force_login(writer_user)
    settings.LANGUAGE_CODE = 'it'
    settings.USE_THOUSAND_SEPARATOR = True
    region = regions[0]
    eclass = Eclass.objects.create(pk=3456, name='Z')
    parcel = Parcel.objects.create(pk=1234, region=region, name='1', version=2345,
                                   eclass=eclass, altitude_max_m=1600)
    with override('it'):
        assert number_format(1234) == '1.234'
        html = client.get(f'{ROOT}{region.pk}/parcel-form/').json()['html']
        assert '<option value="1234" data-version="2345">' in html
        assert '<option value="3456"' in html
        html = client.get(f'{ROOT}{region.pk}/parcels/{parcel.pk}/properties/').json()['html']
        assert 'name="row_id" value="1234"' in html
        assert 'name="version" value="2345"' in html
        assert 'value="1600"' in html


def test_promote_first_parcel_then_add_a_piece(client, writer_user, regions, eclasses, settings):
    from apps.base.digests import build_parcel_record
    from apps.base.models import Parcel

    client.force_login(writer_user)
    region = regions[0]
    settings.IPSO_SECRET = 'field-token'
    auth = {'HTTP_AUTHORIZATION': 'Bearer field-token'}
    original_version = client.get('/ipso/reference.json', **auth).json()['reference_version']
    form = client.get(f'{ROOT}{region.pk}/parcel-form/')
    assert form.status_code == 200
    import re
    html = form.json()['html']
    assert re.search(fr'<option value="{eclasses[0].pk}"[^>]+selected', html)
    assert 'required' not in re.search(r'<input[^>]+name="area_ha"[^>]*>', html).group()
    polygon = {'type': 'Polygon', 'coordinates': [[[16, 38], [16.1, 38], [16, 38.1], [16, 38]]]}
    def draw(shape):
        result = write(client, region, 'create', name='Confine', parts=[shape])
        assert result.status_code == 200, result.content
        return result.json()['drawing_id']
    first = draw(polygon)
    data = {'id': first, 'version': 1, 'name': '2a', 'eclass_id': eclasses[0].pk}
    assert write(client, region, 'parcel', **data, area_ha='abc').status_code == 400
    assert write(client, region, 'parcel', **data).status_code == 200
    parcel = Parcel.objects.get(region=region)
    assert parcel.name == '2a' and parcel.area_ha is None
    assert build_parcel_record(parcel)
    feature, = client.get('/api/geo/map.geojson').json()['features']
    assert feature['properties']['name'] == region.name + '-2a'
    reference = client.get('/ipso/reference.json', **auth).json()
    assert reference['reference_version'] != original_version
    assert [row['parcel_id'] for row in reference['parcels']] == [parcel.pk]
    assert workspace(client, region)['drawings'] == []
    # Appending a piece also compares against imported outlines with elevations.
    existing = parcel.map_features.get()
    existing.geometry['coordinates'][0][1].append(100)
    existing.save(update_fields=['geometry'])
    original = existing.geometry
    next_piece = {'type': 'Polygon', 'coordinates': [[[17, 38], [17.1, 38], [17, 38.1], [17, 38]]]}
    other = draw(next_piece)
    append = {'id': other, 'version': 1, 'parcel_id': parcel.pk}
    assert write(client, region, 'parcel', **append, parcel_version=0).status_code == 400
    assert write(client, region, 'parcel', **append, parcel_version=parcel.version, parts=[polygon]).status_code == 400
    assert write(client, region, 'parcel', id=other, version=1, name='2a', eclass_id=eclasses[0].pk).status_code == 400
    assert write(client, region, 'parcel', **append, parcel_version=parcel.version).status_code == 200
    assert parcel.map_features.count() == 2
    existing.refresh_from_db()
    assert existing.geometry == original
    parcel.refresh_from_db()
    assert parcel.version == 2


def test_promotion_protects_other_parcels_and_legacy_outlines(client, writer_user, parcels):
    from apps.base.models import MapFeature

    west, east = parcels[:2]
    region = west.region
    client.force_login(writer_user)
    def square(x):
        return {'type': 'Polygon', 'coordinates': [[[x, 38], [x + 1, 38], [x + 1, 39], [x, 39], [x, 38]]]}

    MapFeature.objects.create(parcel=west, geometry=square(16))
    MapFeature.objects.create(geometry=square(18),
        members={'type': 'Feature', 'properties': {'layer': region.name, 'name': '99'}})
    drawing = write(client, region, 'create', name='Confine', parts=[square(16)]).json()['drawing_id']
    base = {'id': drawing, 'version': 1}
    for destination in [{'parcel_id': east.pk, 'parcel_version': east.version},
                        {'name': 'Nuova', 'eclass_id': west.eclass_id}]:
        for x in [16, 18]:
            rejected = write(client, region, 'parcel', **base, parts=[square(x)], **destination)
            assert rejected.status_code == 400, rejected.content
            assert 'Il nuovo confine si sovrappone a un altro poligono' in rejected.json()['message']
    assert west.map_features.count() == 1 and not east.map_features.exists()
    response = write(client, region, 'parcel', **base, parts=[square(17)],
                     parcel_id=east.pk, parcel_version=east.version)
    assert response.status_code == 200, response.content
    assert east.map_features.get().geometry == square(17)


def test_edit_road_properties_points_and_delete(client, writer_user, regions):
    from apps.base.models import MapFeature, RoadType

    region = regions[0]
    client.force_login(writer_user)
    kind = RoadType.objects.create(name='Pista di prova', sort_order=1)
    road = MapFeature.objects.create(region=region, name='Prima', road_type=kind,
                                     geometry={'type': 'LineString', 'coordinates': [[16, 38], [17, 38]]})
    assert write(client, region, 'road-properties', id=road.pk, version=1,
                 name='Nuova', road_type_id=kind.pk).status_code == 200
    assert write(client, region, 'road-points', id=road.pk, version=1, parts=[]).status_code == 400
    coords = [[16, 38], [16.5, 38.1], [17, 38]]
    assert write(client, region, 'road-points', id=road.pk, version=2,
                 parts=[{'type': 'LineString', 'coordinates': coords}]).status_code == 200
    road.refresh_from_db()
    assert road.version == 3 and road.name == 'Nuova' and road.geometry['coordinates'] == coords
    assert write(client, region, 'remove-road', id=road.pk, version=3).status_code == 200
    assert not MapFeature.objects.filter(pk=road.pk).exists()


def test_rename_parcel_keeps_records_and_requires_confirmation(client, writer_user, parcels):
    from apps.base.models import HarvestPlan, HarvestPlanItem

    parcel = parcels[0]
    region = parcel.region
    client.force_login(writer_user)
    row = next(row for row in workspace(client, region)['parcels'] if row['id'] == parcel.pk)
    assert row['in_use'] is False and row['version'] == parcel.version
    properties = client.get(f'{ROOT}{region.pk}/parcels/{parcel.pk}/properties/')
    assert 'data-action="delete"' not in properties.json()['html']
    plan = HarvestPlan.objects.create(name='Prova', year_start=2027, year_end=2030)
    item = HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcel, year_planned=2028)
    row = next(row for row in workspace(client, region)['parcels'] if row['id'] == parcel.pk)
    assert row['in_use'] is True
    body = {'id': parcel.pk, 'version': parcel.version, 'name': 'Nuova', 'eclass_id': parcel.eclass_id}
    assert write(client, region, 'parcel-properties', **body).status_code == 400
    assert write(client, region, 'parcel-properties', **body, confirm_rename=True).status_code == 200
    parcel.refresh_from_db()
    assert parcel.name == 'Nuova' and item.parcel_id == parcel.pk
    assert write(client, region, 'remove-parcel', id=parcel.pk, version=parcel.version).status_code == 400


def test_shared_boundary_moves_both_parcels_and_preserves_pieces(client, writer_user, parcels):
    from apps.base.models import MapFeature

    west, east = parcels[:2]
    client.force_login(writer_user)
    left = MapFeature.objects.create(parcel=west, geometry={'type': 'Polygon',
        'coordinates': [[[16, 38], [17, 38], [17, 39], [16, 39], [16, 38]]]})
    # The importer accepts elevations on individual positions. A neighboring
    # outline with a mix of 2D and 3D positions must still join the workspace.
    right = MapFeature.objects.create(parcel=east, geometry={'type': 'Polygon',
        'coordinates': [[[17, 38, 100], [18, 38], [18, 39], [17, 39], [17, 38.5, 110], [17, 38, 100]]]})
    path = f'{ROOT}{west.region_id}/parcels/{west.pk}/boundaries/'
    snapshot = client.get(path).json()
    outline = next(row for row in snapshot['features'] if row['id'] == left.pk)
    ring = outline['parts'][0]['coordinates'][0]
    edits = [{'id': left.pk, 'part': 0, 'ring': 0, 'index': ring.index([17, 38.5]),
              'point': [17.1, 38.5], 'operation': 'move'}]
    body = {'id': west.pk, 'review': snapshot['review'], 'edits': edits}
    invalid = {**body, 'edits': [{**edits[0], 'point': [18.2, 38.5]}]}
    assert write(client, west.region, 'parcel-boundaries', **invalid).status_code == 400
    left.refresh_from_db()
    right.refresh_from_db()
    assert left.version == right.version == 1
    assert write(client, west.region, 'parcel-boundaries', **body).status_code == 200
    left.refresh_from_db()
    right.refresh_from_db()
    for outline in (left, right):
        assert outline.version == 2
        assert [17.1, 38.5] in outline.geometry['coordinates'][0]
    assert write(client, west.region, 'parcel-boundaries', **body).status_code == 400

    snapshot = client.get(path).json()
    ring = next(row for row in snapshot['features'] if row['id'] == left.pk)['parts'][0]['coordinates'][0]
    remove = {'id': left.pk, 'part': 0, 'ring': 0, 'operation': 'remove'}
    body = {'id': west.pk, 'review': snapshot['review']}
    junction = {**remove, 'index': ring.index([17, 38])}
    assert write(client, west.region, 'parcel-boundaries', **body, edits=[junction]).status_code == 400
    left.refresh_from_db()
    right.refresh_from_db()
    assert left.version == right.version == 2

    interior = {**remove, 'index': ring.index([17.1, 38.5])}
    assert write(client, west.region, 'parcel-boundaries', **body, edits=[interior]).status_code == 200
    left.refresh_from_db()
    right.refresh_from_db()
    for outline in (left, right):
        assert outline.version == 3
        assert [17.1, 38.5] not in outline.geometry['coordinates'][0]
        assert len(outline.geometry['coordinates'][0]) == 5


def test_boundary_edit_can_add_piece_and_reject_overlap(client, writer_user, parcels):
    from apps.base.models import MapFeature

    parcel, neighbor = parcels[:2]
    client.force_login(writer_user)
    first = {'type': 'Polygon', 'coordinates': [[[16, 38], [17, 38], [17, 39], [16, 38]]]}
    second = {'type': 'Polygon', 'coordinates': [[[18, 38], [19, 38], [19, 39], [18, 38]]]}
    existing = MapFeature.objects.create(parcel=parcel, geometry=first)
    distant = {'type': 'MultiPolygon',
               'coordinates': [[[[20, 38, 100], [21, 38], [21, 39], [20, 38, 100]]]]}
    imported = MapFeature.objects.create(parcel=neighbor, geometry=distant)
    snapshot = client.get(f'{ROOT}{parcel.region_id}/parcels/{parcel.pk}/boundaries/').json()
    body = {'id': parcel.pk, 'review': snapshot['review']}
    assert write(client, parcel.region, 'parcel-boundaries', **body, added=[first]).status_code == 400
    assert write(client, parcel.region, 'parcel-boundaries', **body, added=[second]).status_code == 200
    existing.refresh_from_db()
    assert existing.geometry == first and existing.version == 1
    imported.refresh_from_db()
    assert imported.geometry == distant and imported.version == 1
    parcel.refresh_from_db()
    assert parcel.version == 2 and parcel.map_features.count() == 2
    assert write(client, parcel.region, 'remove-parcel', id=parcel.pk, version=2).status_code == 200
    assert not MapFeature.objects.filter(parcel_id=parcel.pk).exists()


def test_opening_boundary_pieces_preserves_parcel_and_protects_its_last_outline(client, writer_user, parcels):
    from apps.base.models import HarvestPlan, HarvestPlanItem, MapFeature

    parcel, neighbor = parcels[:2]
    client.force_login(writer_user)
    first = [[[16, 38], [17, 38], [17, 39], [16, 38]]]
    second = [[[18, 38], [19, 38], [19, 39], [18, 38]]]
    outline = MapFeature.objects.create(parcel=parcel,
        geometry={'type': 'MultiPolygon', 'coordinates': [first, second]})
    other = MapFeature.objects.create(parcel=neighbor, geometry={'type': 'Polygon',
        'coordinates': [[[17, 38], [18, 38], [17, 39], [17, 38]]]})
    plan = HarvestPlan.objects.create(name='Prova', year_start=2027, year_end=2030)
    item = HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcel, year_planned=2028)
    path = f'{ROOT}{parcel.region_id}/parcels/{parcel.pk}/boundaries/'

    def detach(fid, part=0):
        return {'id': fid, 'part': part, 'operation': 'detach'}

    def save(*edits, added=()):
        return write(client, parcel.region, 'parcel-boundaries', id=parcel.pk,
                     review=client.get(path).json()['review'], edits=list(edits), added=list(added))

    assert save(detach(other.pk)).status_code == 400
    assert save(detach(outline.pk, 1)).status_code == 200
    outline.refresh_from_db()
    other.refresh_from_db()
    assert outline.geometry == {'type': 'Polygon', 'coordinates': first}
    assert other.version == 1
    response = save(detach(outline.pk))
    assert response.status_code == 400
    assert outline.pk == parcel.map_features.get().pk
    assert save(detach(outline.pk), added=[{'type': 'Polygon', 'coordinates': second}]).status_code == 200
    assert not parcel.map_features.filter(pk=outline.pk).exists()
    item.delete()
    assert save(detach(parcel.map_features.get().pk)).status_code == 200
    assert not parcel.map_features.exists()
    parcel.refresh_from_db()
