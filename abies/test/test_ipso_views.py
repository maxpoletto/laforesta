"""Tests for the Abies-served Ipso PWA."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from django.db import IntegrityError
from django.test import Client, override_settings
from django.urls import reverse

from apps.base.models import (
    HYPSO_FUNC_LN, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    HypsoParam, HypsoParamSet, HypsoParamSource, SampleArea, SampleGrid,
    Survey, Tree, TreeMark, TreePreserved,
)
from apps.ipso import views as ipso_views
from apps.ipso.models import IpsoUpload, IpsoUploadState


def _body(resp) -> bytes:
    return b''.join(resp.streaming_content)


@pytest.fixture
def writer_client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


@pytest.fixture
def reader_client(reader_user):
    client = Client()
    client.force_login(reader_user)
    return client


def test_index_is_public_and_uses_relative_assets(db):
    resp = Client().get('/ipso/')

    assert resp.status_code == 200
    body = _body(resp).decode()
    assert '<title>Ipso' in body
    assert 'href="style.css"' in body
    assert 'href="/static/vendor/leaflet/leaflet.css"' in body
    assert 'href="/static/base/css/map-basemaps.css"' in body
    assert 'id="screen-mode"' in body
    assert 'id="btn-mode-samples" class="btn-secondary btn-big" type="button"' in body
    assert 'id="btn-mode-pai" class="btn-secondary btn-big" type="button"' in body
    assert 'id="screen-map"' in body
    assert 'src="modes.js"' in body
    assert 'src="/static/vendor/leaflet/leaflet.js"' in body
    assert 'src="map.js"' in body
    assert 'src="app.js"' in body


def test_service_worker_served_from_ipso_scope(db):
    resp = Client().get('/ipso/sw.js')

    assert resp.status_code == 200
    assert resp['Content-Type'].startswith('text/javascript')
    body = _body(resp)
    assert b'ipso service worker' in body
    assert b'/static/base/js/map-common.js' in body
    assert b'/static/base/css/map-basemaps.css' in body


@pytest.mark.parametrize(('path', 'needle'), [
    ('/ipso/map.js', b'createOrientationMap'),
    ('/ipso/modes.js', b'IpsoModes'),
])
def test_referenced_static_helpers_are_served(db, path, needle):
    resp = Client().get(path)

    assert resp.status_code == 200
    assert resp['Content-Type'].startswith('text/javascript')
    assert needle in _body(resp)


def test_manifest_is_served(db):
    resp = Client().get('/ipso/manifest.webmanifest')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/manifest+json'
    assert json.loads(_body(resp))['short_name'] == 'Ipso'


def test_reference_json_comes_from_abies_data(db, regions, parcels, species):
    grid = SampleGrid.objects.create(name='Ipso grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcels[0], number='3',
        lat=38.51234, lon=16.12345, r_m=15,
    )
    survey = Survey.objects.create(
        name='Ipso survey', sample_grid=grid, active=True,
    )
    preserved_tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        coppice=False, estimated_birth_year=1920, lat=38.45678, lon=16.12345,
    )
    preserved = TreePreserved.objects.create(
        tree=preserved_tree, parcel=parcels[0], number=7,
        date=date(2024, 9, 15), d_cm=42, h_m=Decimal('18.50'),
        h_measured=True, lat=38.45678, lon=16.12345, acc_m=6,
        operator='Mario', note='nota',
    )
    active = HypsoParamSet.objects.create(source=HypsoParamSource.IMPORTED)
    HypsoParam.objects.create(
        param_set=active, region=regions[0], species=species[0],
        func=HYPSO_FUNC_LN, a=Decimal('7.0000'), b=Decimal('-4.0000'),
        r2=Decimal('0.5000'), n=12,
    )

    resp = Client().get('/ipso/reference.json')

    assert resp.status_code == 200
    data = resp.json()
    assert data['schema_version'] == 1
    assert data['reference_version']
    assert data['species'][0]['common'] == 'Abete'
    parcel = next(p for p in data['parcels']
                  if p['compresa'] == 'Capistrano' and p['particella'] == '1')
    assert parcel['region_id'] == regions[0].id
    assert parcel['parcel_id'] == parcels[0].id
    assert data['ipsometrica']['Capistrano']['Abete'] == {
        'a': 7.0, 'b': -4.0, 'hypso_param_set_id': active.id,
    }
    assert data['work_packages'] == [{
        'id': f'sampling_survey:{survey.id}',
        'kind': 'sampling_survey',
        'label': 'Ipso survey',
        'survey_id': survey.id,
        'sample_grid_id': grid.id,
    }]
    assert data['sampling']['surveys'] == [{
        'survey_id': survey.id,
        'name': 'Ipso survey',
        'sample_grid_id': grid.id,
        'sample_grid_name': 'Ipso grid',
    }]
    assert data['sampling']['sample_areas'] == [{
        'sample_area_id': area.id,
        'sample_grid_id': grid.id,
        'region_id': parcels[0].region_id,
        'parcel_id': parcels[0].id,
        'compresa': 'Capistrano',
        'particella': '1',
        'number': '3',
        'lat': 38.51234,
        'lon': 16.12345,
        'r_m': 15,
    }]
    assert data['pai']['preserved_trees'] == [{
        'tree_preserved_id': preserved.id,
        'tree_id': preserved_tree.id,
        'region_id': parcels[0].region_id,
        'parcel_id': parcels[0].id,
        'compresa': 'Capistrano',
        'particella': '1',
        'species_id': species[0].id,
        'number': 7,
        'estimated_birth_year': 1920,
        'date': '2024-09-15',
        'd_cm': 42,
        'h_m': '18.50',
        'h_measured': True,
        'lat': 38.45678,
        'lon': 16.12345,
        'acc_m': 6,
        'operator': 'Mario',
        'note': 'nota',
    }]


def test_terreni_geojson_has_empty_fallback(db):
    resp = Client().get('/ipso/terreni.geojson')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/geo+json'
    assert resp.json() == {'type': 'FeatureCollection', 'features': []}


def _upload_payload(
        parcels, species, *, mode='martellate',
        session_id='11111111-1111-4111-8111-111111111111'):
    parcel = parcels[0]
    sp = species[0]
    return {
        'session': {
            'session_id': session_id,
            'mode': mode,
            'schema_version': 1,
            'reference_version': '',
            'work_package_id': '',
            'operator': 'Mario Rossi',
            'created_at': '2026-06-17T08:00:00Z',
            'completed_at': '2026-06-17T09:00:00Z',
            'damaged': False,
            'region_id': parcel.region_id,
        },
        'records': [{
            'client_record_id': '1',
            'date': '2026-06-17',
            'region_id': parcel.region_id,
            'parcel_id': parcel.id,
            'species_id': sp.id,
            'number': 1,
            'd_cm': 42,
            'h_m': '22',
            'h_measured': False,
            'hypso_param_set_id': None,
            'lat': 38.51234,
            'lon': 16.12345,
            'acc_m': 5,
        }],
        'csv_text': 'csv backup',
    }



def _harvest_item(parcels):
    plan = HarvestPlan.objects.create(
        name='Piano Ipso', year_start=2026, year_end=2026,
    )
    return HarvestPlanItem.objects.create(
        harvest_plan=plan,
        parcel=parcels[0],
        year_planned=2026,
        state=HarvestPlanItemState.PLANNED,
    )


def _post_upload(client, payload, token='test-token'):
    return client.post(
        reverse('ipso-upload-session'),
        data=json.dumps(payload),
        content_type='application/json',
        HTTP_AUTHORIZATION=f'Bearer {token}',
        HTTP_X_IPSO_SESSION_ID=payload['session']['session_id'],
    )


@override_settings(IPSO_UPLOAD_TOKEN='configured-token')
def test_upload_rejects_bad_token(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    resp = _post_upload(Client(), _upload_payload(parcels, species), token='wrong')

    assert resp.status_code == 401
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_stages_json_and_metadata(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body['ok'] is True
    assert body['duplicate'] is False
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.mode == 'martellate'
    assert upload.record_count == 1
    assert Path(upload.inbox_path).is_dir()
    staged = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    assert staged['records'][0]['hypso_param_set_id'] is None
    assert staged['records'][0]['lon'] == 16.12345
    assert (Path(upload.inbox_path) / 'upload.sha256').is_file()
    assert (Path(upload.inbox_path) / 'export.csv').read_text() == 'csv backup'


@pytest.mark.parametrize(('mode', 'session_id'), [
    ('samples', '22222222-2222-4222-8222-222222222222'),
    ('pai', '33333333-3333-4333-8333-333333333333'),
])
@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_stages_non_martellate_modes(
        db, parcels, species, settings, tmp_path, mode, session_id):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, mode=mode, session_id=session_id)
    if mode == 'pai':
        payload['records'][0]['d_cm'] = None
        payload['records'][0]['h_m'] = None

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 200
    upload = IpsoUpload.objects.get(session_id=session_id)
    assert upload.mode == mode
    assert upload.state == IpsoUploadState.RECEIVED
    staged = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    assert staged['session']['mode'] == mode
    assert staged['records'][0]['hypso_param_set_id'] is None


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_rejects_unknown_mode(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, mode='invalid')

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_duplicate_is_idempotent(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    first = _post_upload(Client(), payload)
    second = _post_upload(Client(), payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()['duplicate'] is True
    assert IpsoUpload.objects.count() == 1


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_conflicts_on_same_session_different_payload(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    changed = _upload_payload(parcels, species)
    changed['records'][0]['d_cm'] = 43

    resp = _post_upload(Client(), changed)

    assert resp.status_code == 409
    assert resp.json()['error'] == 'conflict'
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    assert upload.state == IpsoUploadState.CONFLICT


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_integrity_error_does_not_write_files(
        db, parcels, species, settings, tmp_path, monkeypatch):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)

    def raise_integrity_error(**kwargs):
        raise IntegrityError

    def fail_write(*args, **kwargs):
        pytest.fail('upload files were written before the session row was claimed')

    monkeypatch.setattr(ipso_views.IpsoUpload.objects, 'create', raise_integrity_error)
    monkeypatch.setattr(ipso_views, '_write_upload_files', fail_write)

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 409
    assert resp.json()['error'] == 'conflict'
    assert not settings.IPSO_INBOX_DIR.exists()


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_rejects_unknown_species(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload['records'][0]['species_id'] = 999999

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_UPLOAD_TOKEN='visible-token')
def test_upload_config_serves_configured_token(db):
    resp = Client().get('/ipso/upload-config.js')

    assert resp.status_code == 200
    assert b'visible-token' in resp.content


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_shell_shows_importazione_dot_for_received_upload(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    assert _post_upload(Client(), _upload_payload(parcels, species)).status_code == 200

    resp = writer_client.get('/importazione')

    assert resp.status_code == 200
    body = resp.content.decode()
    assert 'data-tab="importazione"' in body
    assert 'data-ipso-pending-dot hidden' not in body


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_inbox_data_lists_received_upload(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200

    resp = writer_client.get(reverse('ipso-inbox-data'))

    assert resp.status_code == 200
    data = resp.json()
    assert data['pending_count'] == 1
    assert data['columns'][0] == 'row_id'
    assert data['rows'][0][2] == 'martellate'
    assert data['rows'][0][3] == 'Mario Rossi'
    assert data['rows'][0][5] == 'Da importare'


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_detail_previews_staged_records(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data['upload']['state'] == IpsoUploadState.RECEIVED
    assert data['record_count'] == 1
    assert data['records'][0]['parcel'] == 'Capistrano 1'
    assert data['records'][0]['species'] == 'Abete'
    assert data['records'][0]['lon'] == 16.12345


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_writer_can_reject_upload(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-reject', args=[upload.id]),
        data=json.dumps({'reason': 'Duplicato'}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.REJECTED
    assert upload.error_summary == 'Duplicato'
    assert writer_client.get(reverse('ipso-inbox-data')).json()['pending_count'] == 0


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_reader_cannot_reject_upload(reader_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = reader_client.post(
        reverse('ipso-upload-reject', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 403
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_upload_detail_lists_martellate_targets(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    payload['session']['work_package_id'] = f'harvest:{item.id}'
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data['suggested_target_id'] == item.id
    assert data['targets'][0]['id'] == item.id
    assert 'Piano Ipso' in data['targets'][0]['label']
    assert 'Capistrano 1' in data['targets'][0]['label']


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_writer_imports_upload_into_harvest_plan_item(
        writer_client, writer_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()['imported'] == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.imported_by == writer_user
    assert upload.target_type == 'harvest_plan_item'
    assert upload.target_id == item.id
    tm = TreeMark.objects.select_related('tree', 'tree__species').get()
    assert tm.harvest_plan_item == item
    assert tm.tree.parcel == parcels[0]
    assert tm.tree.species == species[0]
    assert tm.number == 1
    assert tm.date == date(2026, 6, 17)
    assert tm.d_cm == 42
    assert tm.h_m == Decimal('22.00')
    assert tm.lat == 38.51234
    assert tm.lon == 16.12345
    assert tm.operator == 'Mario Rossi'
    item.refresh_from_db()
    assert item.state == HarvestPlanItemState.MARKED
    assert item.date_actual == date(2026, 6, 17)


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_import_rejects_second_submit_without_duplicate_marks(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    body = json.dumps({'harvest_plan_item_id': item.id})
    url = reverse('ipso-upload-import-martellate', args=[upload.id])

    first = writer_client.post(url, data=body, content_type='application/json')
    second = writer_client.post(url, data=body, content_type='application/json')

    assert first.status_code == 200
    assert first.json()['imported'] == 1
    assert second.status_code == 400
    assert second.json()['message'] == 'Solo i caricamenti da importare possono essere importati.'
    assert TreeMark.objects.count() == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED


@override_settings(IPSO_UPLOAD_TOKEN='test-token')
def test_import_rejects_reader(reader_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = reader_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 403
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
