"""Tests for the Abies-served Ipso PWA."""

import io
import json
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from django.db import IntegrityError
from django.test import Client, override_settings
from django.urls import reverse

from apps.base.models import (
    HYPSO_FUNC_LN, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    HypsoParam, HypsoParamSet, HypsoParamSource, Parcel, Sample,
    SampleArea, SampleGrid, Survey, Tree, TreeMark, TreePreserved,
    TreeSample,
)
from apps.ipso import staging as ipso_staging
from apps.ipso import views as ipso_views
from apps.ipso.models import IpsoUpload, IpsoUploadState
from config import strings as S


def _body(resp) -> bytes:
    return b''.join(resp.streaming_content)


@pytest.fixture
def admin_client(admin_user):
    client = Client()
    client.force_login(admin_user)
    return client


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
    assert 'name="apple-mobile-web-app-title" content="Ipso"' in body
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
    assert 'upload-config.js' not in body


def test_service_worker_served_from_ipso_scope(db):
    resp = Client().get('/ipso/sw.js')

    assert resp.status_code == 200
    assert resp['Content-Type'].startswith('text/javascript')
    body = _body(resp)
    assert b'ipso service worker' in body
    assert b'/static/base/js/map-common.js' in body
    assert b'/static/base/css/map-basemaps.css' in body
    assert b'upload-config.js' not in body
    assert b'./reference.json' not in body
    assert b'./terreni.geojson' not in body
    assert b"req.cache === 'no-store'" in body
    assert b"cacheControl.includes('no-store')" in body


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
    manifest = json.loads(_body(resp))
    assert manifest['name'] == 'Ipso'
    assert manifest['short_name'] == 'Ipso'


@override_settings(IPSO_SECRET='test-token')
def test_ipso_data_downloads_require_bearer(db):
    client = Client()

    assert client.get('/ipso/reference.json').status_code == 401
    assert client.get('/ipso/terreni.geojson').status_code == 401
    assert client.get(
        '/ipso/reference.json', HTTP_AUTHORIZATION='Bearer wrong',
    ).status_code == 401
    assert client.get(
        '/ipso/terreni.geojson', HTTP_AUTHORIZATION='Bearer wrong',
    ).status_code == 401


@override_settings(IPSO_SECRET='test-token')
def test_reference_json_comes_from_abies_data(db, regions, parcels, species):
    grid = SampleGrid.objects.create(name='Ipso grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcels[0], number='3',
        lat=38.51234, lon=16.12345, r_m=15,
    )
    survey = Survey.objects.create(
        name='Ipso survey', sample_grid=grid, active=True,
    )
    inactive_survey = Survey.objects.create(
        name='Ipso survey 2', sample_grid=grid, active=False,
    )
    sample = Sample.objects.create(
        sample_area=area, survey=survey, date=date(2024, 9, 16),
    )
    sampled_tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], coppice=False,
    )
    TreeSample.objects.create(
        sample=sample, tree=sampled_tree, number=8, d_cm=30, h_m=Decimal('18.00'),
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

    resp = Client().get(
        '/ipso/reference.json', HTTP_AUTHORIZATION='Bearer test-token',
    )

    assert resp.status_code == 200
    assert resp['Vary'] == 'Authorization'
    assert 'no-store' in resp['Cache-Control']
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
    assert data['work_packages'] == [
        {
            'id': f'sampling_survey:{survey.id}',
            'kind': 'sampling_survey',
            'label': 'Ipso survey',
            'survey_id': survey.id,
            'sample_grid_id': grid.id,
        },
        {
            'id': f'sampling_survey:{inactive_survey.id}',
            'kind': 'sampling_survey',
            'label': 'Ipso survey 2',
            'survey_id': inactive_survey.id,
            'sample_grid_id': grid.id,
        },
    ]
    assert data['sampling']['surveys'] == [
        {
            'survey_id': survey.id,
            'name': 'Ipso survey',
            'sample_grid_id': grid.id,
            'sample_grid_name': 'Ipso grid',
            'sample_area_max_numbers': {str(area.id): 8},
        },
        {
            'survey_id': inactive_survey.id,
            'name': 'Ipso survey 2',
            'sample_grid_id': grid.id,
            'sample_grid_name': 'Ipso grid',
            'sample_area_max_numbers': {},
        },
    ]
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
        'coppice': False,
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
        'coppice': False,
    }]


@override_settings(IPSO_SECRET='test-token')
def test_terreni_geojson_has_empty_fallback(db):
    resp = Client().get(
        '/ipso/terreni.geojson', HTTP_AUTHORIZATION='Bearer test-token',
    )

    assert resp.status_code == 200
    assert resp['Vary'] == 'Authorization'
    assert 'no-store' in resp['Cache-Control']
    assert resp['Content-Type'] == 'application/geo+json'
    assert resp.json() == {'type': 'FeatureCollection', 'features': []}


def _upload_payload(
        parcels, species, *, mode='martellate',
        session_id='11111111-1111-4111-8111-111111111111',
        session_overrides=None, record_overrides=None):
    parcel = parcels[0]
    sp = species[0]
    session = {
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
    }
    record = {
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
    }
    if session_overrides:
        session.update(session_overrides)
    if record_overrides:
        record.update(record_overrides)
    return {'session': session, 'records': [record], 'csv_text': 'csv backup'}


def _sample_survey(parcel, *, name='Ipso survey'):
    grid = SampleGrid.objects.create(name=f'{name} grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='3',
        lat=38.51234, lon=16.12345, r_m=15,
    )
    survey = Survey.objects.create(name=name, sample_grid=grid, active=True)
    return survey, area


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


def _stage_upload_direct(settings, tmp_path, payload, csv_text='csv backup'):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    checksum = ipso_staging.payload_checksum(payload)
    inbox_path = ipso_staging.upload_inbox_path(payload['session']['session_id'])
    ipso_staging.write_upload_files(inbox_path, payload, checksum, csv_text)
    return IpsoUpload.objects.create(
        **ipso_staging.upload_model_fields(payload, checksum, inbox_path),
    )


@override_settings(IPSO_SECRET='configured-token')
def test_upload_rejects_bad_token(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    resp = _post_upload(Client(), _upload_payload(parcels, species), token='wrong')

    assert resp.status_code == 401
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='')
def test_upload_rejects_when_token_unconfigured(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'

    resp = _post_upload(Client(), _upload_payload(parcels, species))

    assert resp.status_code == 401
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_stages_json_and_metadata(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species,
        record_overrides={'lat': 38.512345, 'lon': 16.123455},
    )

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
    assert staged['records'][0]['lat'] == 38.51235
    assert staged['records'][0]['lon'] == 16.12346
    assert (Path(upload.inbox_path) / 'upload.sha256').is_file()
    assert (Path(upload.inbox_path) / 'export.csv').read_text() == 'csv backup'


@pytest.mark.parametrize(('mode', 'session_id'), [
    ('samples', '22222222-2222-4222-8222-222222222222'),
    ('pai', '33333333-3333-4333-8333-333333333333'),
])
@override_settings(IPSO_SECRET='test-token')
def test_upload_stages_non_martellate_modes(
        db, parcels, species, settings, tmp_path, mode, session_id):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, mode=mode, session_id=session_id)
    if mode == 'samples':
        survey, area = _sample_survey(parcels[0])
        payload['session']['work_package_id'] = f'sampling_survey:{survey.id}'
        payload['records'][0]['sample_area_id'] = area.id
    resp = _post_upload(Client(), payload)

    assert resp.status_code == 200
    upload = IpsoUpload.objects.get(session_id=session_id)
    assert upload.mode == mode
    assert upload.state == IpsoUploadState.RECEIVED
    staged = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    assert staged['session']['mode'] == mode
    assert staged['records'][0]['hypso_param_set_id'] is None
    if mode == 'samples':
        assert staged['records'][0]['sample_area_id'] == area.id


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_empty_records(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload['records'] = []

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert 'records' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token', IPSO_UPLOAD_MAX_BYTES=10)
def test_upload_rejects_body_over_cap(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'

    resp = _post_upload(Client(), _upload_payload(parcels, species))

    assert resp.status_code == 413
    assert resp.json()['error'] == 'too_large'
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token', IPSO_UPLOAD_MAX_RECORDS=1)
def test_upload_rejects_record_count_over_cap(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    second = dict(payload['records'][0])
    second['client_record_id'] = '2'
    second['number'] = 2
    payload['records'].append(second)

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert 'records' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(
    IPSO_SECRET='test-token',
    IPSO_UPLOAD_RATE_LIMIT=1,
    IPSO_UPLOAD_RATE_WINDOW_S=60,
)
def test_upload_endpoint_is_rate_limited(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    ipso_views._UPLOAD_ATTEMPTS.clear()
    try:
        first = _post_upload(Client(), _upload_payload(parcels, species))
        payload = _upload_payload(
            parcels, species,
            session_id='22222222-2222-4222-8222-222222222222',
        )
        second = _post_upload(Client(), payload)
    finally:
        ipso_views._UPLOAD_ATTEMPTS.clear()

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()['error'] == 'rate_limited'


@override_settings(IPSO_UPLOAD_TRUSTED_PROXIES=('172.16.0.0/12',))
def test_upload_rate_key_uses_forwarded_client_from_trusted_proxy(db):
    request = Client().post(
        '/api/ipso/uploads/',
        REMOTE_ADDR='172.18.0.1',
        HTTP_X_FORWARDED_FOR='203.0.113.10',
    ).wsgi_request

    assert ipso_views._upload_rate_key(request) == '203.0.113.10'


@override_settings(IPSO_UPLOAD_TRUSTED_PROXIES=('172.16.0.0/12',))
def test_upload_rate_key_ignores_forwarded_client_from_untrusted_peer(db):
    request = Client().post(
        '/api/ipso/uploads/',
        REMOTE_ADDR='198.51.100.20',
        HTTP_X_FORWARDED_FOR='203.0.113.10',
    ).wsgi_request

    assert ipso_views._upload_rate_key(request) == '198.51.100.20'


@override_settings(IPSO_UPLOAD_TRUSTED_PROXIES=('172.16.0.0/12',))
def test_upload_rate_key_ignores_invalid_forwarded_client(db):
    request = Client().post(
        '/api/ipso/uploads/',
        REMOTE_ADDR='172.18.0.1',
        HTTP_X_FORWARDED_FOR='not-an-ip',
    ).wsgi_request

    assert ipso_views._upload_rate_key(request) == '172.18.0.1'


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_pai_without_height(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333333',
    )
    payload['records'][0]['h_m'] = None

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_pai_without_number(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333339',
        record_overrides={'number': None},
    )

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert 'numero obbligatorio' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_samples_without_number(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    _survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222229',
        record_overrides={'sample_area_id': area.id, 'number': None},
    )

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert 'numero obbligatorio' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_unknown_mode(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, mode='invalid')

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_duplicate_is_idempotent(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    first = _post_upload(Client(), payload)
    second = _post_upload(Client(), payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()['duplicate'] is True
    assert IpsoUpload.objects.count() == 1


@override_settings(IPSO_SECRET='test-token')
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


@override_settings(IPSO_SECRET='test-token')
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


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_unknown_species(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload['records'][0]['species_id'] = 999999

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_null_parcel_id(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload['records'][0]['parcel_id'] = None

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()['error'] == 'invalid_payload'
    assert 'parcel_id' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='visible-token')
def test_upload_config_does_not_serve_configured_token(db):
    resp = Client().get('/ipso/upload-config.js')

    assert resp.status_code == 404
    assert b'visible-token' not in resp.content


@override_settings(IPSO_SECRET='test-token')
def test_shell_shows_importazione_dot_for_received_upload(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    assert _post_upload(Client(), _upload_payload(parcels, species)).status_code == 200

    resp = writer_client.get('/importazione')

    assert resp.status_code == 200
    body = resp.content.decode()
    assert 'data-tab="importazione"' in body
    assert 'data-ipso-pending-dot hidden' not in body


@override_settings(IPSO_SECRET='test-token')
def test_inbox_data_lists_received_upload(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200

    resp = writer_client.get(reverse('ipso-inbox-data'))

    assert resp.status_code == 200
    data = resp.json()
    assert data['pending_count'] == 1
    row = dict(zip(data['columns'], data['rows'][0]))
    assert data['columns'][0] == 'row_id'
    assert row['_ipso_state'] == IpsoUploadState.RECEIVED
    assert row[S.COL_DATE] == '2026-06-17'
    assert row[S.IPSO_COL_MODE] == S.IPSO_MODE_MARTELLATE_LABEL
    assert row[S.IPSO_COL_OPERATOR] == 'Mario Rossi'
    assert row[S.IPSO_COL_STATE] == S.IPSO_STATE_RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_upload_detail_previews_staged_records(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, record_overrides={'h_m': '22.5'})
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data['upload']['state'] == IpsoUploadState.RECEIVED
    assert data['upload']['mode'] == 'martellate'
    assert data['upload']['mode_label'] == S.IPSO_MODE_MARTELLATE_LABEL
    assert data['upload']['record_date'] == '2026-06-17'
    assert data['record_count'] == 1
    assert data['records'][0]['seq'] == 1
    assert data['records'][0]['parcel'] == 'Capistrano 1'
    assert data['records'][0]['species'] == 'Abete'
    assert data['records'][0]['h_m'] == 22.5
    assert data['records'][0]['lon'] == 16.12345


@override_settings(IPSO_SECRET='test-token')
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


@override_settings(IPSO_SECRET='test-token')
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


@override_settings(IPSO_SECRET='test-token')
def test_admin_downloads_staged_upload_zip(
        admin_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = admin_client.get(reverse('ipso-upload-download', args=[upload.id]))

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/zip'
    assert 'no-store' in resp['Cache-Control']
    assert f'ipso-upload-{upload.session_id}.zip' in resp['Content-Disposition']
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert sorted(zf.namelist()) == ['export.csv', 'upload.json', 'upload.sha256']
    assert json.loads(zf.read('upload.json'))['session']['mode'] == 'martellate'
    assert zf.read('export.csv').decode() == 'csv backup'


@override_settings(IPSO_SECRET='test-token')
def test_writer_cannot_download_delete_or_edit_upload_mode(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    assert writer_client.get(
        reverse('ipso-upload-download', args=[upload.id]),
    ).status_code == 403
    assert writer_client.post(
        reverse('ipso-upload-delete', args=[upload.id]),
        data=json.dumps({}), content_type='application/json',
    ).status_code == 403
    assert writer_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({'mode': 'pai'}), content_type='application/json',
    ).status_code == 403
    upload.refresh_from_db()
    assert upload.mode == 'martellate'
    assert Path(upload.inbox_path).is_dir()


@override_settings(IPSO_SECRET='test-token')
def test_admin_updates_upload_mode_and_staged_payload(
        admin_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333338',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    inbox_path = Path(upload.inbox_path)
    original_csv = (inbox_path / 'export.csv').read_text()

    resp = admin_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({'mode': 'martellate'}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    upload.refresh_from_db()
    assert upload.mode == 'martellate'
    staged = json.loads((inbox_path / 'upload.json').read_text())
    assert staged['session']['mode'] == 'martellate'
    assert (inbox_path / 'upload.sha256').read_text().strip() == upload.checksum
    assert (inbox_path / 'export.csv').read_text() == original_csv
    assert resp.json()['upload']['mode'] == 'martellate'
    assert resp.json()['upload']['mode_label'] == S.IPSO_MODE_MARTELLATE_LABEL


@override_settings(IPSO_SECRET='test-token')
def test_admin_cannot_update_mode_after_domain_import(
        admin_client, writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    assert writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    ).status_code == 200

    resp = admin_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({'mode': 'pai'}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORTED_CANNOT_EDIT_MODE in resp.json()['message']
    upload.refresh_from_db()
    assert upload.mode == 'martellate'


@override_settings(IPSO_SECRET='test-token')
def test_admin_deletes_staged_upload_record_and_files(
        admin_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    inbox_path = Path(upload.inbox_path)

    resp = admin_client.post(
        reverse('ipso-upload-delete', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert not IpsoUpload.objects.filter(id=upload.id).exists()
    assert not inbox_path.exists()


@override_settings(IPSO_SECRET='test-token')
def test_upload_detail_lists_martellate_targets(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    item.note = 'Nota molto lunga per selettore destinazione'
    item.save(update_fields=['note'])
    payload = _upload_payload(parcels, species)
    payload['session']['work_package_id'] = f'harvest:{item.id}'
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data['suggested_target_id'] == item.id
    assert data['targets'][0]['id'] == item.id
    label = data['targets'][0]['label']
    assert 'Piano Ipso' in label
    assert 'Capistrano 1' in label
    assert 'Nota molto lunga per…' in label
    assert 'selettore destinazione' not in label


@override_settings(IPSO_SECRET='test-token')
def test_upload_detail_lists_sample_targets(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0])
    target = Survey.objects.create(
        name='Ipso inactive target', sample_grid=survey.sample_grid, active=False,
    )
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222222',
        session_overrides={'work_package_id': f'sampling_survey:{target.id}'},
        record_overrides={'sample_area_id': area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data['suggested_target_id'] == target.id
    assert any(t['id'] == target.id for t in data['targets'])
    assert any('Ipso inactive target' in t['label'] for t in data['targets'])
    assert data['upload']['work_package_label'] == 'Ipso inactive target - Ipso survey grid'
    assert data['records'][0]['sample_area_id'] == area.number
    inbox = writer_client.get(reverse('ipso-inbox-data')).json()
    row = dict(zip(inbox['columns'], inbox['rows'][0]))
    assert row[S.IPSO_COL_WORK_PACKAGE] == 'Ipso inactive target - Ipso survey grid'
    assert 'sampling_survey:' not in json.dumps(inbox)


@override_settings(IPSO_SECRET='test-token')
def test_import_rejects_upload_mode_mismatch(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    _, area = _sample_survey(parcels[0])
    item = _harvest_item(parcels)
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222227',
        record_overrides={'sample_area_id': area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_MODE_UNSUPPORTED in resp.json()['message']
    assert TreeMark.objects.count() == 0
    assert TreeSample.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.target_type == ''

@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_rejects_coppice_target(
        writer_client, regions, eclasses, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='C-target', region=regions[0],
        eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
    )
    item = _harvest_item([coppice_parcel])
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_INVALID_MARTELLATE_TARGET in resp.json()['message']
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_region_wide_target_accepts_coppice_parcel(
        writer_client, regions, eclasses, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='C-damaged', region=regions[0],
        eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
    )
    plan = HarvestPlan.objects.create(
        name='Piano Ipso danni', year_start=2026, year_end=2026,
    )
    item = HarvestPlanItem.objects.create(
        harvest_plan=plan, region=regions[0], year_planned=2026,
        damaged=True, state=HarvestPlanItemState.PLANNED,
    )
    payload = _upload_payload([coppice_parcel], species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200, resp.json()
    mark = TreeMark.objects.get()
    assert mark.harvest_plan_item == item
    assert mark.tree.parcel == coppice_parcel
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED


@pytest.mark.parametrize(('file_state', 'expected_error'), [
    ('missing', S.IPSO_ERR_UPLOAD_JSON_MISSING),
    ('invalid', S.IPSO_ERR_UPLOAD_JSON_INVALID),
])
@override_settings(IPSO_SECRET='test-token')
def test_import_reports_staged_payload_file_errors(
        writer_client, parcels, species, settings, tmp_path, file_state,
        expected_error):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    upload_json = Path(upload.inbox_path) / 'upload.json'
    if file_state == 'missing':
        upload_json.unlink()
    else:
        upload_json.write_text('{not-json', encoding='utf-8')

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert expected_error in resp.json()['message']
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.error_summary == expected_error


@pytest.mark.parametrize('state', [
    IpsoUploadState.REJECTED,
    IpsoUploadState.CONFLICT,
])
@override_settings(IPSO_SECRET='test-token')
def test_import_rejects_non_received_upload_states(
        writer_client, parcels, species, settings, tmp_path, state):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    upload.state = state
    upload.save(update_fields=['state'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_UPLOAD_NOT_RECEIVED in resp.json()['message']
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == state


@override_settings(IPSO_SECRET='test-token')
def test_writer_imports_upload_into_harvest_plan_item(
        writer_client, writer_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(
        parcels, species,
        record_overrides={'h_m': '22.346', 'lat': 38.512345, 'lon': 16.123455},
    )
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
    assert tm.h_m == Decimal('22.35')
    assert tm.lat == 38.51235
    assert tm.lon == 16.12346
    assert tm.operator == 'Mario Rossi'
    item.refresh_from_db()
    assert item.state == HarvestPlanItemState.MARKED
    assert item.date_actual == date(2026, 6, 17)



@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_rejects_duplicate_mark_number(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    tree = Tree.objects.create(species=species[0], parcel=parcels[0])
    TreeMark.objects.create(
        harvest_plan_item=item, tree=tree, number=7,
        date=date(2026, 6, 16), d_cm=30, h_m=Decimal('18.00'),
        operator='Mario Rossi',
    )
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111121',
        record_overrides={'number': 7},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.ERR_MARK_NUMBER_DUPLICATE.format(7) in resp.json()['message']
    assert TreeMark.objects.count() == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED

@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_rejects_edited_invalid_number(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111119',
    )
    payload['records'][0]['number'] = 'abc'
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_INVALID.format(1) in resp.json()['message']
    assert TreeMark.objects.count() == 0

@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_rejects_edited_non_positive_number(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111120',
    )
    payload['records'][0]['number'] = 0
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(1) in resp.json()['message']
    assert TreeMark.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_preserves_blank_numbers_without_auto_numbering(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    for n in [1, 2, 3]:
        tree = Tree.objects.create(species=species[0], parcel=parcels[0])
        TreeMark.objects.create(
            harvest_plan_item=item, tree=tree, number=n,
            date=date(2026, 6, 16), d_cm=30, h_m=Decimal('18.00'),
            operator='Mario Rossi',
        )
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111118',
        record_overrides={'number': None, 'client_record_id': 'd'},
    )
    base = payload['records'][0]
    payload['records'] = []
    for idx, number in enumerate([None, None, 4, 5, 6], start=1):
        row = dict(base)
        row['client_record_id'] = f'new-{idx}'
        row['number'] = number
        row['d_cm'] = 40 + idx
        payload['records'].append(row)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()['imported'] == 5
    assert list(
        TreeMark.objects.filter(harvest_plan_item=item)
        .order_by('id')
        .values_list('number', flat=True)
    ) == [1, 2, 3, None, None, 4, 5, 6]


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_area_outside_selected_survey_grid(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    source_survey, area = _sample_survey(parcels[0], name='Source survey')
    target_survey, _ = _sample_survey(parcels[0], name='Target survey')
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222228',
        record_overrides={'sample_area_id': area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': target_survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert (S.IPSO_ERR_IMPORT_RECORD_AREA_OUT_OF_SURVEY.format(1)
            in resp.json()['message'])
    assert TreeSample.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.error_summary == S.IPSO_ERR_IMPORT_RECORD_AREA_OUT_OF_SURVEY.format(1)
    assert source_survey.sample_grid_id == area.sample_grid_id
    assert target_survey.sample_grid_id != area.sample_grid_id


@override_settings(IPSO_SECRET='test-token')
def test_writer_imports_samples_upload_into_survey(
        writer_client, writer_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222222',
        record_overrides={'sample_area_id': area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()['imported'] == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.imported_by == writer_user
    assert upload.target_type == 'survey'
    assert upload.target_id == survey.id
    sample = Sample.objects.get(survey=survey, sample_area=area)
    assert sample.date == date(2026, 6, 17)
    ts = TreeSample.objects.select_related('tree', 'tree__species').get(sample=sample)
    assert ts.tree.parcel == parcels[0]
    assert ts.tree.species == species[0]
    assert ts.tree.lat == 38.51234
    assert ts.tree.lon == 16.12345
    assert ts.number == 1
    assert ts.d_cm == 42
    assert ts.h_m == Decimal('22.00')
    assert ts.volume_m3 is not None


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_staged_missing_number(
        writer_client, parcels, species, settings, tmp_path):
    survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222230',
        record_overrides={'sample_area_id': area.id, 'number': None},
    )
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(1) in resp.json()['message']
    assert TreeSample.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_supports_coppice_parcels(
        writer_client, regions, eclasses, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='C1', region=regions[0], eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
    )
    survey, area = _sample_survey(coppice_parcel, name='Ceduo survey')
    payload = _upload_payload(
        [coppice_parcel], species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222223',
        record_overrides={'sample_area_id': area.id, 'species_id': species[1].id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    ts = TreeSample.objects.select_related('tree').get()
    assert ts.tree.coppice is True
    assert ts.volume_m3 is None
    assert ts.mass_q is None


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_supports_coppice_shoots_with_same_number(
        writer_client, regions, eclasses, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='C2', region=regions[0], eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
    )
    survey, area = _sample_survey(coppice_parcel, name='Ceduo shoots survey')
    payload = _upload_payload(
        [coppice_parcel], species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222225',
        record_overrides={
            'sample_area_id': area.id,
            'coppice': True,
            'shoot': 1,
        },
    )
    second = dict(payload['records'][0])
    second['client_record_id'] = '2'
    second['shoot'] = 2
    payload['records'].append(second)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()['imported'] == 2
    rows = list(TreeSample.objects.order_by('shoot'))
    assert [row.number for row in rows] == [1, 1]
    assert [row.shoot for row in rows] == [1, 2]
    assert all(row.tree.coppice for row in rows)
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_existing_number_shoot(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0])
    sample = Sample.objects.create(survey=survey, sample_area=area, date=date(2026, 6, 17))
    tree = Tree.objects.create(species=species[0], parcel=parcels[0])
    TreeSample.objects.create(
        sample=sample, tree=tree, number=1, d_cm=30, h_m=Decimal('18.00'),
    )
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222226',
        record_overrides={'sample_area_id': area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(1) in resp.json()['message']
    assert TreeSample.objects.count() == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED



@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_duplicate_number_shoot_in_upload(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222227',
        record_overrides={'sample_area_id': area.id, 'number': 9, 'shoot': 0},
    )
    second = dict(payload['records'][0])
    second['client_record_id'] = '2'
    second['d_cm'] = 43
    payload['records'].append(second)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(2) in resp.json()['message']
    assert TreeSample.objects.count() == 0

@override_settings(IPSO_SECRET='test-token')
def test_pai_import_rejects_duplicate_tree_number_in_parcel(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    tree = Tree.objects.create(species=species[0], parcel=parcels[0], preserved=True)
    TreePreserved.objects.create(
        tree=tree, parcel=parcels[0], number=1, date=date(2026, 6, 17),
        d_cm=30, h_m=Decimal('18.00'), lat=38.51234, lon=16.12345,
    )
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333335',
        record_overrides={'number': 1},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert 'numero PAI già presente' in resp.json()['message']
    assert TreePreserved.objects.count() == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_pai_import_integrity_error_returns_validation(
        writer_client, parcels, species, settings, tmp_path, monkeypatch):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333336',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    def raise_integrity_error(rows):
        raise IntegrityError

    monkeypatch.setattr(ipso_views, 'apply_pai_rows', raise_integrity_error)

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert 'Numero PAI già presente' in resp.json()['message']
    assert TreePreserved.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert 'Numero PAI già presente' in upload.error_summary


@override_settings(IPSO_SECRET='test-token')
def test_pai_import_rejects_staged_missing_number(
        writer_client, parcels, species, settings, tmp_path):
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333337',
        record_overrides={'number': None},
    )
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(1) in resp.json()['message']
    assert TreePreserved.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_writer_imports_pai_upload(writer_client, writer_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333333',
        record_overrides={
            'number': 1,
            'estimated_birth_year': 1920,
            'note': 'nota PAI',
        },
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()['imported'] == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.imported_by == writer_user
    assert upload.target_type == 'pai'
    pai = TreePreserved.objects.select_related('tree', 'tree__species').get()
    assert pai.tree.parcel == parcels[0]
    assert pai.tree.species == species[0]
    assert pai.number == 1
    assert pai.date == date(2026, 6, 17)
    assert pai.d_cm == 42
    assert pai.h_m == Decimal('22.00')
    assert pai.h_measured is True
    assert pai.operator == 'Mario Rossi'
    assert pai.note == 'nota PAI'
    assert pai.tree.coppice is False
    assert pai.volume_m3 is None
    assert pai.mass_q is None


@override_settings(IPSO_SECRET='test-token')
def test_pai_import_supports_coppice_parcels(
        writer_client, regions, eclasses, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='P1', region=regions[0], eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
    )
    payload = _upload_payload(
        [coppice_parcel], species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333334',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    pai = TreePreserved.objects.select_related('tree').get()
    assert pai.tree.coppice is False
    assert pai.volume_m3 is None
    assert pai.mass_q is None


@override_settings(IPSO_SECRET='test-token')
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


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_second_submit_without_duplicate_samples(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222224',
        record_overrides={'sample_area_id': area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload['session']['session_id'])
    body = json.dumps({'survey_id': survey.id})
    url = reverse('ipso-upload-import-samples', args=[upload.id])

    first = writer_client.post(url, data=body, content_type='application/json')
    second = writer_client.post(url, data=body, content_type='application/json')

    assert first.status_code == 200
    assert second.status_code == 400
    assert TreeSample.objects.count() == 1


@override_settings(IPSO_SECRET='test-token')
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
