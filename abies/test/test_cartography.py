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
