"""The road workflow starts with an actual field upload."""
import uuid

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


@pytest.fixture
def walk(client, writer_user, settings, tmp_path, regions):
    client.force_login(writer_user)
    settings.IPSO_SECRET = 'field-token'
    settings.IPSO_INBOX_DIR = tmp_path
    session_id = str(uuid.uuid4())
    segments = [session_id, session_id, None, str(uuid.uuid4())]
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


def test_import_restore_after_inbox_cleanup(client, walk, regions, admin_user):
    region, other = regions[:2]
    assert workspace(client, region)['uploads'][0]['id'] == walk.pk
    assert workspace(client, other)['uploads'] == []
    assert write(client, other, 'import', upload_id=walk.pk).status_code == 400
    assert write(client, region, 'import', upload_id=walk.pk).status_code == 200
    state = workspace(client, region)
    assert state['uploads'] == []
    source, = state['sources']
    line, gate, last = state['drawings']
    assert line['geometry'] == {'type': 'LineString', 'coordinates': [[16, 38], [16, 38.01]]}
    assert gate['geometry']['type'] == last['geometry']['type'] == 'Point'
    assert gate['fixes'] == [[2.3, 'Cancello']]
    edited = {'type': 'LineString', 'coordinates': [[16.1, 38], [16, 38.01]]}
    assert write(client, region, 'adjust', id=line['id'], version=line['version'], geometry=edited).status_code == 200
    assert write(client, region, 'adjust', id=line['id'], version=line['version'], geometry=edited).status_code == 400
    client.force_login(admin_user)
    assert client.post(reverse('ipso-upload-delete', args=[walk.pk]), {'nonce': str(uuid.uuid4())},
                       content_type='application/json').status_code == 200
    assert write(client, region, 'restore', source_id=source['id']).json()['added'] == 0
    drawings = workspace(client, region)['drawings']
    assert drawings[0]['geometry'] == edited
    versions = [[d['id'], d['version']] for d in drawings]
    assert write(client, region, 'clear', versions=versions).status_code == 200
    assert write(client, region, 'restore', source_id=source['id']).json()['added'] == 3
    assert workspace(client, region)['drawings'][0]['geometry'] == line['geometry']


def test_region_and_edit_guards(client, walk, regions, reader_user):
    region, other = regions[:2]
    assert write(client, region, 'import', upload_id=walk.pk).status_code == 200
    state = workspace(client, region)
    line = state['drawings'][0]
    for geometry in ({}, {'type': 'Point', 'coordinates': [16, 38]},
                     {'type': 'LineString', 'coordinates': [[False, 38], [16, 91]]}):
        assert write(client, region, 'adjust', id=line['id'], version=1, geometry=geometry).status_code == 400
    assert write(client, region, 'clear', versions=[]).status_code == 400
    assert write(client, other, 'restore', source_id=state['sources'][0]['id']).status_code == 404
    client.force_login(reader_user)
    assert client.get(f'{ROOT}{region.pk}/').status_code == 403
    assert write(client, region, 'clear', versions=[]).status_code == 403


def test_walk_to_published_road(client, walk, regions):
    region = regions[0]
    assert write(client, region, 'import', upload_id=walk.pk).status_code == 200
    state = workspace(client, region)
    line, gate, _ = state['drawings']
    kind = client.get(ROOT).json()['road_types'][0]['id']
    road_data = {'name': 'Via nord', 'road_type_id': kind, 'version': 1}
    assert write(client, region, 'road', id=gate['id'], **road_data).status_code == 400
    coords = [[16, 38], [16.012, 38.02]]
    geometry = {'type': 'LineString', 'coordinates': coords}
    assert write(client, region, 'adjust', id=line['id'], version=1, geometry=geometry).status_code == 200
    assert write(client, region, 'road', id=line['id'], **road_data).status_code == 400
    response = write(client, region, 'road', id=line['id'], **(road_data | {'version': 2}))
    assert response.status_code == 200, response.content
    draft, = workspace(client, region)['roads']
    assert not draft['published']
    assert not client.get('/api/geo/roads.geojson').json()['features']
    assert write(client, region, 'publish', versions=[]).status_code == 400
    assert write(client, region, 'publish', versions=[[draft['id'], draft['version']]]).status_code == 200
    road, = client.get('/api/geo/roads.geojson').json()['features']
    assert road['geometry'] == geometry and road['properties']['road_type_id'] == kind
    state = workspace(client, region)
    assert write(client, region, 'clear', versions=[[d['id'], d['version']] for d in state['drawings']]).status_code == 200
    assert write(client, region, 'restore', source_id=state['sources'][0]['id']).json()['added'] == 3
    restored = workspace(client, region)
    assert restored['drawings'][0]['geometry'] == line['geometry']
    assert [r['id'] for r in restored['roads']] == [draft['id']]


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
    assert len(workspace(client, regions[0])['drawings']) == 3


def test_discard_one_working_copy_or_road_draft(client, regions, walk):
    from apps.base.models import MapFeature, RoadType

    region = regions[0]
    kind = RoadType.objects.create(name='Test road', sort_order=10)
    write(client, region, 'import', upload_id=walk.pk)
    state = workspace(client, region)
    line, point, _ = state['drawings']
    assert write(client, region, 'remove-drawing', id=point['id'], version=2).status_code == 400
    assert write(client, region, 'remove-drawing', id=point['id'], version=1).status_code == 200
    assert write(client, region, 'restore', source_id=state['sources'][0]['id']).json()['added'] == 1
    assert [d['sequence'] for d in workspace(client, region)['drawings']] == [1, 2, 3]
    road_id = write(client, region, 'road', id=line['id'], version=1,
                    name='Prova', road_type_id=kind.id).json()['road_id']
    assert write(client, region, 'discard-road', id=road_id, version=1).status_code == 200
    assert not MapFeature.objects.filter(pk=road_id).exists()
    public = MapFeature.objects.create(region=region, road_type=kind, published=True,
                                       geometry=line['geometry'], name='Strada')
    assert write(client, region, 'discard-road', id=public.pk, version=1).status_code == 400
    assert MapFeature.objects.filter(pk=public.pk, published=True).exists()


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
    summary = json.dumps({'schema_version': 2, 'parcels': ['1'], 'means': {'parcels': {'1': {'ndvi': [0.4]}}}})
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
    assert b''.join(series.streaming_content) == summary.encode()
    context = client.get(reverse('bosco-map-context', args=[region.pk]))
    assert context.get('Content-Encoding') == 'gzip'


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
