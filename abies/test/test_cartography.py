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
