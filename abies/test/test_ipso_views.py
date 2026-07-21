"""Tests for the Abies-served Ipso PWA."""

import gzip
import io
import json
import zipfile
from datetime import date, datetime, timezone as dt_timezone
from decimal import Decimal
from pathlib import Path

import pytest
from django.db import IntegrityError
from django.test import Client, RequestFactory, override_settings
from django.urls import reverse

from apps.base.models import (
    HYPSO_FUNC_LN, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    HypsoParam, HypsoParamSet, HypsoParamSource, Parcel, Sample,
    SampleArea, SampleGrid, Survey, Tree, TreeMark,
    TreeSample, UsedNonce,
)
from apps.ipso import staging as ipso_staging
from apps.ipso import views as ipso_views
from apps.ipso.models import IpsoUpload, IpsoUploadState
from config import strings as S
from config.constants import (
    COLUMNS, DETAIL, DUPLICATE, ERROR, FIELD_ACC_M, FIELD_CLIENT_RECORD_ID,
    FIELD_COMPLETED_AT, FIELD_COPPICE, FIELD_CREATED_AT,
    FIELD_CSV_TEXT, FIELD_D_CM, FIELD_DAMAGED, FIELD_DATE,
    FIELD_ESTIMATED_BIRTH_YEAR, FIELD_HARVEST_PLAN_ITEM_ID, FIELD_H_M,
    FIELD_H_MEASURED,
    FIELD_HYPSO_PARAM_SET_ID, FIELD_LAT, FIELD_LON, FIELD_MODE,
    FIELD_L10_MM, FIELD_MODE_LABEL, FIELD_NONCE, FIELD_NOTE, FIELD_NUMBER,
    FIELD_OPERATOR, FIELD_PARCEL, FIELD_PARCEL_ID, FIELD_PRESERVED,
    FIELD_PRESSLER_COEFF, FIELD_REASON, FIELD_RECORD_DATE,
    FIELD_REFERENCE_VERSION, FIELD_REGION_ID, FIELD_SAMPLE_AREA_ID,
    FIELD_SCHEMA_VERSION, FIELD_STANDARD, FIELD_TREE_PRESERVED_ID,
    FIELD_SEQ, FIELD_SESSION_ID, FIELD_SHOOT, FIELD_SPECIES, FIELD_SPECIES_ID,
    FIELD_SURVEY_ID, FIELD_WORK_PACKAGE_ID, FIELD_WORK_PACKAGE_LABEL,
    IMPORTED, IPSO_ERROR_CONFLICT, IPSO_ERROR_INVALID_PAYLOAD,
    IPSO_ERROR_RATE_LIMITED, IPSO_ERROR_TOO_LARGE, IPSO_MODE_MARTELLATE,
    IPSO_MODE_PAI, IPSO_MODE_SAMPLES, IPSO_UPLOAD_FILE_READY, MESSAGE,
    PENDING_COUNT, RECORDS, ROWS, ROW_ID, SESSION, UPLOAD,
)


def _body(resp) -> bytes:
    return b''.join(resp.streaming_content)


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


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


@pytest.fixture(autouse=True)
def clear_ipso_upload_rate_limit():
    ipso_views._UPLOAD_ATTEMPTS.clear()
    yield
    ipso_views._UPLOAD_ATTEMPTS.clear()


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
    assert b'url.origin !== self.location.origin' in body
    assert b"new Request(url, { cache: 'no-cache' })" in body
    assert b"req.cache === 'no-store'" in body
    assert b"cacheControl.includes('no-store')" in body
    assert b'n.startsWith(CACHE_PREFIX) && n !== CACHE' in body


def test_ipso_registers_service_worker_without_http_cache(db):
    resp = Client().get('/ipso/app.js')

    assert resp.status_code == 200
    assert resp['Content-Type'].startswith('text/javascript')
    body = _body(resp)
    assert b"updateViaCache: 'none'" in body
    assert b'registration.update()' in body


def test_ipso_formats_timestamps_in_local_timezone(db):
    value = datetime(2026, 1, 1, 12, 0, tzinfo=dt_timezone.utc)

    assert ipso_views._format_dt(value) == '2026-01-01 13:00'


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


@override_settings(IPSO_SECRET='old-shared-secret')
def test_ipso_data_download_accepts_preserved_shared_secret_after_upgrade(db):
    resp = Client().get(
        '/ipso/reference.json', HTTP_AUTHORIZATION='Bearer old-shared-secret',
    )

    assert resp.status_code == 200
    assert resp['Vary'] == 'Authorization'


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
    unstructured_survey = Survey.objects.create(
        name='Ipso unstructured survey',
    )
    sample = Sample.objects.create(
        sample_area=area, survey=survey, date=date(2024, 9, 16),
    )
    sampled_tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], coppice=False,
    )
    TreeSample.objects.create(
        sample=sample, tree=sampled_tree, parcel=parcels[0],
        number=8, d_cm=30, h_m=Decimal('18.00'),
    )
    preserved_tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        coppice=False, estimated_birth_year=1920, lat=38.45678, lon=16.12345,
    )
    preserved = _preserved_sample(
        preserved_tree, parcels[0], number=7, sample_date=date(2024, 9, 15),
        d_cm=42, h_m=Decimal('18.50'), h_measured=True,
        lat=38.45678, lon=16.12345, acc_m=6, operator='Mario', note='nota',
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
    assert all(
        s['survey_id'] != unstructured_survey.id
        for s in data['sampling']['surveys']
    )
    assert all(
        wp['survey_id'] != unstructured_survey.id
        for wp in data['work_packages']
        if wp['kind'] == 'sampling_survey'
    )
    assert data['sampling']['sample_areas'] == [{
        FIELD_SAMPLE_AREA_ID: area.id,
        'sample_grid_id': grid.id,
        'region_id': parcels[0].region_id,
        'parcel_id': parcels[0].id,
        'compresa': 'Capistrano',
        'particella': '1',
        FIELD_NUMBER: '3',
        'lat': 38.51234,
        'lon': 16.12345,
        'r_m': 15,
        'coppice': False,
    }]
    assert data['pai']['preserved_trees'] == [{
        FIELD_TREE_PRESERVED_ID: preserved.id,
        'tree_id': preserved_tree.id,
        'region_id': parcels[0].region_id,
        'parcel_id': parcels[0].id,
        'compresa': 'Capistrano',
        'particella': '1',
        FIELD_SPECIES_ID: species[0].id,
        FIELD_NUMBER: 7,
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
        parcels, species, *, mode=IPSO_MODE_MARTELLATE,
        session_id='11111111-1111-4111-8111-111111111111',
        session_overrides=None, record_overrides=None):
    parcel = parcels[0]
    sp = species[0]
    session = {
        FIELD_SESSION_ID: session_id,
        FIELD_MODE: mode,
        FIELD_SCHEMA_VERSION: 1,
        'reference_version': '',
        'work_package_id': '',
        FIELD_OPERATOR: 'Mario Rossi',
        'created_at': '2026-06-17T08:00:00Z',
        'completed_at': '2026-06-17T09:00:00Z',
        FIELD_DAMAGED: False,
        FIELD_REGION_ID: parcel.region_id,
    }
    record = {
        FIELD_CLIENT_RECORD_ID: '1',
        FIELD_DATE: '2026-06-17',
        FIELD_REGION_ID: parcel.region_id,
        FIELD_PARCEL_ID: parcel.id,
        FIELD_SPECIES_ID: sp.id,
        FIELD_NUMBER: 1,
        FIELD_D_CM: 42,
        FIELD_H_M: '22',
        FIELD_H_MEASURED: False,
        FIELD_HYPSO_PARAM_SET_ID: None,
        FIELD_LAT: 38.51234,
        FIELD_LON: 16.12345,
        FIELD_ACC_M: 5,
    }
    if session_overrides:
        session.update(session_overrides)
    if record_overrides:
        record.update(record_overrides)
    return {SESSION: session, RECORDS: [record], FIELD_CSV_TEXT: 'csv backup'}


def _legacy_ipso_1_1_4_payload(
        parcels, species, *, mode=IPSO_MODE_MARTELLATE,
        session_id='44444444-1444-4444-8444-444444444401',
        work_package_id='', sample_area_id=None, number=1,
        record_overrides=None,
):
    """Payload shape emitted by abies-1.1.4 Ipso buildUploadPayload()."""
    parcel = parcels[0]
    sp = species[0]
    session = {
        FIELD_SESSION_ID: session_id,
        FIELD_MODE: mode,
        FIELD_SCHEMA_VERSION: 1,
        FIELD_REFERENCE_VERSION: 'abies-1.1.4-reference',
        FIELD_WORK_PACKAGE_ID: work_package_id,
        FIELD_OPERATOR: 'Mario Rossi',
        FIELD_CREATED_AT: '2026-06-17T08:00:00Z',
        FIELD_COMPLETED_AT: '2026-06-17T09:00:00Z',
        FIELD_DAMAGED: False,
        FIELD_REGION_ID: parcel.region_id,
    }
    record = {
        FIELD_CLIENT_RECORD_ID: '1',
        FIELD_DATE: '2026-06-17',
        FIELD_REGION_ID: parcel.region_id,
        FIELD_PARCEL_ID: parcel.id,
        FIELD_SPECIES_ID: sp.id,
        FIELD_NUMBER: number,
        FIELD_D_CM: 42,
        FIELD_H_M: '22',
        FIELD_H_MEASURED: False,
        FIELD_HYPSO_PARAM_SET_ID: None,
        FIELD_LAT: 38.51234,
        FIELD_LON: 16.12345,
        FIELD_ACC_M: 5,
    }
    if mode == IPSO_MODE_SAMPLES:
        record.update({
            FIELD_SAMPLE_AREA_ID: sample_area_id,
            FIELD_COPPICE: False,
            FIELD_SHOOT: 0,
            FIELD_STANDARD: False,
            FIELD_L10_MM: 0,
            FIELD_PRESSLER_COEFF: '2',
            FIELD_PRESERVED: False,
            # abies-1.1.4 did not include per-record operator/note for samples.
        })
    elif mode == IPSO_MODE_PAI:
        record.update({
            FIELD_ESTIMATED_BIRTH_YEAR: 1920,
            FIELD_OPERATOR: 'Mario Rossi',
            FIELD_NOTE: 'nota PAI da 1.1.4',
        })
    # Martellate records deliberately carry no per-record operator in the
    # 1.1.4 client; the session operator is the fallback.
    if record_overrides:
        record.update(record_overrides)
    return {
        SESSION: session, RECORDS: [record],
        FIELD_CSV_TEXT: 'csv backup 1.1.4',
    }


def _preserved_sample(
        tree, parcel, *, number=7, sample_date=date(2024, 9, 15),
        d_cm=42, h_m=Decimal('18.50'), h_measured=True, lat=38.45678,
        lon=16.12345, acc_m=6, operator='Mario', note='nota',
):
    survey = Survey.objects.create(name=f'Ipso PAI survey {tree.id}-{number}')
    sample = Sample.objects.create(
        sample_area=None, survey=survey, date=sample_date,
    )
    return TreeSample.objects.create(
        sample=sample, tree=tree, parcel=parcel, number=number,
        preserved_number=number, d_cm=d_cm, h_m=h_m,
        h_measured=h_measured, lat=lat, lon=lon, acc_m=acc_m,
        operator=operator, note=note,
    )


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
    session = payload.get(SESSION, {}) if isinstance(payload, dict) else {}
    session_id = session.get(FIELD_SESSION_ID, '') if isinstance(session, dict) else ''
    return client.post(
        reverse('ipso-upload-session'),
        data=json.dumps(payload),
        content_type='application/json',
        HTTP_AUTHORIZATION=f'Bearer {token}',
        HTTP_X_IPSO_SESSION_ID=session_id,
    )


def _stage_upload_direct(settings, tmp_path, payload, csv_text='csv backup'):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    checksum = ipso_staging.payload_checksum(payload)
    inbox_path = ipso_staging.upload_inbox_path(payload[SESSION][FIELD_SESSION_ID])
    ipso_staging.write_upload_files(inbox_path, payload, checksum, csv_text)
    return IpsoUpload.objects.create(
        **ipso_staging.upload_model_fields(payload, checksum, inbox_path),
    )


def _assert_nonce_saved(user, nonce):
    used = UsedNonce.objects.get(user=user, nonce=nonce)
    assert used.response_json


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
    assert body[DUPLICATE] is False
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.mode == IPSO_MODE_MARTELLATE
    assert upload.record_count == 1
    assert upload.record_date == '2026-06-17'
    assert Path(upload.inbox_path).is_dir()
    staged = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    assert staged[RECORDS][0][FIELD_HYPSO_PARAM_SET_ID] is None
    assert staged[RECORDS][0][FIELD_LAT] == 38.51235
    assert staged[RECORDS][0][FIELD_LON] == 16.12346
    assert (Path(upload.inbox_path) / 'upload.sha256').is_file()
    assert (Path(upload.inbox_path) / IPSO_UPLOAD_FILE_READY).is_file()
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
        payload[SESSION]['work_package_id'] = f'sampling_survey:{survey.id}'
        payload[RECORDS][0][FIELD_SAMPLE_AREA_ID] = area.id
    resp = _post_upload(Client(), payload)

    assert resp.status_code == 200
    upload = IpsoUpload.objects.get(session_id=session_id)
    assert upload.mode == mode
    assert upload.state == IpsoUploadState.RECEIVED
    staged = json.loads((Path(upload.inbox_path) / 'upload.json').read_text())
    assert staged[SESSION][FIELD_MODE] == mode
    assert staged[RECORDS][0][FIELD_HYPSO_PARAM_SET_ID] is None
    if mode == 'samples':
        assert staged[RECORDS][0][FIELD_SAMPLE_AREA_ID] == area.id


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_empty_records(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload[RECORDS] = []

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert RECORDS in resp.json()[DETAIL]
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token', IPSO_UPLOAD_MAX_BYTES=10)
def test_upload_rejects_body_over_cap(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'

    resp = _post_upload(Client(), _upload_payload(parcels, species))

    assert resp.status_code == 413
    assert resp.json()[ERROR] == IPSO_ERROR_TOO_LARGE
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token', IPSO_UPLOAD_MAX_RECORDS=1)
def test_upload_rejects_record_count_over_cap(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    second = dict(payload[RECORDS][0])
    second[FIELD_CLIENT_RECORD_ID] = '2'
    second[FIELD_NUMBER] = 2
    payload[RECORDS].append(second)

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert RECORDS in resp.json()[DETAIL]
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
    assert second.json()[ERROR] == IPSO_ERROR_RATE_LIMITED


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
    payload[RECORDS][0][FIELD_H_M] = None

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_pai_without_number(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333339',
        record_overrides={FIELD_NUMBER: None},
    )

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert 'numero obbligatorio' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_samples_without_number(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    _survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222229',
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id, FIELD_NUMBER: None},
    )

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert 'numero obbligatorio' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_unknown_mode(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, mode='invalid')

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_duplicate_is_idempotent(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    first = _post_upload(Client(), payload)
    second = _post_upload(Client(), payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()[DUPLICATE] is True
    assert IpsoUpload.objects.count() == 1


@override_settings(IPSO_SECRET='test-token', IPSO_UPLOAD_RATE_LIMIT=0)
def test_upload_duplicate_repairs_staged_files(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    inbox_path = Path(upload.inbox_path)
    (inbox_path / 'upload.json').write_text('{"corrupted": true}\n', encoding='utf-8')
    (inbox_path / 'upload.sha256').unlink()
    (inbox_path / 'export.csv').unlink()
    (inbox_path / IPSO_UPLOAD_FILE_READY).unlink()

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 200
    assert resp.json()[DUPLICATE] is True
    staged = json.loads((inbox_path / 'upload.json').read_text(encoding='utf-8'))
    assert staged[SESSION][FIELD_SESSION_ID] == payload[SESSION][FIELD_SESSION_ID]
    assert staged[RECORDS][0][FIELD_NUMBER] == 1
    staged_checksum = (inbox_path / 'upload.sha256').read_text(encoding='utf-8').strip()
    assert staged_checksum == upload.checksum
    assert (inbox_path / 'export.csv').read_text(encoding='utf-8') == 'csv backup'
    ready_checksum = (
        inbox_path / IPSO_UPLOAD_FILE_READY
    ).read_text(encoding='utf-8').strip()
    assert ready_checksum == upload.checksum


@override_settings(IPSO_SECRET='test-token')
def test_upload_conflicts_on_same_session_different_payload(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    changed = _upload_payload(parcels, species)
    changed[RECORDS][0][FIELD_D_CM] = 43

    resp = _post_upload(Client(), changed)

    assert resp.status_code == 409
    assert resp.json()[ERROR] == IPSO_ERROR_CONFLICT
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    assert upload.state == IpsoUploadState.CONFLICT


@pytest.mark.parametrize('state', [
    IpsoUploadState.IMPORTED,
    IpsoUploadState.REJECTED,
])
@override_settings(IPSO_SECRET='test-token')
def test_upload_conflict_retry_preserves_terminal_state(
        db, parcels, species, settings, tmp_path, state):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    upload.state = state
    upload.error_summary = 'terminal state detail'
    upload.save(update_fields=['state', 'error_summary'])
    changed = _upload_payload(parcels, species)
    changed[RECORDS][0][FIELD_D_CM] = 43

    resp = _post_upload(Client(), changed)

    assert resp.status_code == 409
    assert resp.json()[ERROR] == IPSO_ERROR_CONFLICT
    upload.refresh_from_db()
    assert upload.state == state
    assert upload.error_summary == 'terminal state detail'


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
    assert resp.json()[ERROR] == IPSO_ERROR_CONFLICT
    assert not settings.IPSO_INBOX_DIR.exists()


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_unknown_species(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload[RECORDS][0]['species_id'] = 999999

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert IpsoUpload.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_null_parcel_id(db, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    payload[RECORDS][0]['parcel_id'] = None

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert 'parcel_id' in resp.json()['detail']
    assert IpsoUpload.objects.count() == 0


@pytest.mark.parametrize(('mutate', 'expected_detail'), [
    (lambda p: p.__setitem__(SESSION, []), S.IPSO_ERR_FIELD_OBJECT.format(SESSION)),
    (lambda p: p.__setitem__(RECORDS, {}), S.IPSO_ERR_FIELD_ARRAY.format(RECORDS)),
    (lambda p: p.__setitem__(FIELD_CSV_TEXT, 123), S.IPSO_ERR_CSV_TEXT_STRING),
    (
        lambda p: p[SESSION].__setitem__(FIELD_SCHEMA_VERSION, '1'),
        S.IPSO_ERR_FIELD_INTEGER.format(FIELD_SCHEMA_VERSION),
    ),
    (
        lambda p: p[SESSION].__setitem__(FIELD_DAMAGED, 'false'),
        S.IPSO_ERR_FIELD_BOOLEAN.format(FIELD_DAMAGED),
    ),
    (
        lambda p: p[RECORDS][0].__setitem__(FIELD_H_MEASURED, 'false'),
        S.IPSO_ERR_FIELD_BOOLEAN.format(FIELD_H_MEASURED),
    ),
    (
        lambda p: p[RECORDS][0].__setitem__(FIELD_LAT, '38.5'),
        S.IPSO_ERR_FIELD_NUMBER_NULL.format(FIELD_LAT),
    ),
])
@override_settings(IPSO_SECRET='test-token')
def test_upload_rejects_typed_payload_errors(
        db, parcels, species, settings, tmp_path, mutate, expected_detail):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    mutate(payload)

    resp = _post_upload(Client(), payload)

    assert resp.status_code == 422
    assert resp.json()[ERROR] == IPSO_ERROR_INVALID_PAYLOAD
    assert resp.json()[DETAIL] == expected_detail
    assert IpsoUpload.objects.count() == 0


def _record_with(base, **updates):
    row = dict(base)
    row.update(updates)
    return [row]


@pytest.mark.parametrize(('records_factory', 'expected_error'), [
    (lambda base, parcels, species: {}, S.IPSO_ERR_IMPORT_RECORDS_ARRAY),
    (
        lambda base, parcels, species: ['bad'],
        S.IPSO_ERR_IMPORT_RECORD_INVALID.format(1),
    ),
    (
        lambda base, parcels, species: _record_with(base, **{FIELD_PARCEL_ID: 999999}),
        S.IPSO_ERR_IMPORT_RECORD_PARCEL_NOT_FOUND.format(1),
    ),
    (
        lambda base, parcels, species: _record_with(base, **{FIELD_PARCEL_ID: parcels[1].id}),
        S.IPSO_ERR_IMPORT_RECORD_MARK_TARGET_MISMATCH.format(1),
    ),
    (
        lambda base, parcels, species: _record_with(base, **{FIELD_SPECIES_ID: 999999}),
        S.IPSO_ERR_IMPORT_RECORD_SPECIES_NOT_FOUND.format(1),
    ),
    (
        lambda base, parcels, species: _record_with(base, **{FIELD_D_CM: 0}),
        S.IPSO_ERR_IMPORT_RECORD_DH_DATE_INVALID.format(1),
    ),
    (
        lambda base, parcels, species: _record_with(base, **{FIELD_NUMBER: 'abc'}),
        S.IPSO_ERR_RECORD_NUMBER_INVALID.format(1),
    ),
    (
        lambda base, parcels, species: _record_with(base, **{FIELD_NUMBER: 0}),
        S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(1),
    ),
])
def test_martellate_import_rows_reports_row_errors(
        db, parcels, species, records_factory, expected_error):
    item = _harvest_item(parcels)
    upload = IpsoUpload(session_id='row-error-session', mode=IPSO_MODE_MARTELLATE)
    payload = _upload_payload(parcels, species)
    base = payload[RECORDS][0]
    payload[RECORDS] = records_factory(base, parcels, species)

    rows, errors = ipso_views._martellate_import_rows(upload, payload, item)

    assert rows == []
    assert expected_error in errors


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


def test_shell_renders_ipso_upload_modes_from_constants(writer_client, db):
    resp = writer_client.get('/importazione')

    assert resp.status_code == 200
    body = resp.content.decode()
    assert f'name="{FIELD_MODE}"' in body
    assert f'value="{IPSO_MODE_MARTELLATE}"' in body
    assert f'value="{IPSO_MODE_SAMPLES}"' in body
    assert f'value="{IPSO_MODE_PAI}"' in body


@override_settings(IPSO_SECRET='test-token')
def test_inbox_data_lists_received_upload(
        writer_client, parcels, species, settings, tmp_path, monkeypatch):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200

    def fail_staged_payload_read(_upload):
        raise AssertionError('inbox_data must not read staged upload files')
    monkeypatch.setattr(ipso_views, '_read_staged_payload', fail_staged_payload_read)

    resp = writer_client.get(reverse('ipso-inbox-data'))

    assert resp.status_code == 200
    data = resp.json()
    assert data[PENDING_COUNT] == 1
    row = dict(zip(data['columns'], data['rows'][0]))
    assert data['columns'][0] == 'row_id'
    assert row['_ipso_state'] == IpsoUploadState.RECEIVED
    assert row[S.COL_DATE] == '2026-06-17'
    assert row[S.IPSO_COL_MODE] == S.IPSO_MODE_MARTELLATE_LABEL
    assert row[S.IPSO_COL_OPERATOR] == 'Mario Rossi'
    assert row[S.IPSO_COL_STATE] == S.IPSO_STATE_RECEIVED


def test_inbox_data_batches_work_package_and_target_labels(
        db, writer_user, parcels, django_assert_num_queries):
    item = _harvest_item(parcels)
    survey, _area = _sample_survey(parcels[0])
    IpsoUpload.objects.create(
        session_id='batch-harvest', mode=IPSO_MODE_MARTELLATE, schema_version=1,
        reference_version='', work_package_id=f'harvest:{item.id}',
        operator='Mario', record_count=1, record_date='2026-06-17',
        checksum='0' * 64, inbox_path='/tmp/ipso/batch-harvest',
        target_type='harvest_plan_item', target_id=item.id,
    )
    IpsoUpload.objects.create(
        session_id='batch-survey', mode='samples', schema_version=1,
        reference_version='', work_package_id=f'sampling_survey:{survey.id}',
        operator='Luisa', record_count=1, record_date='2026-06-18',
        checksum='1' * 64, inbox_path='/tmp/ipso/batch-survey',
        target_type='survey', target_id=survey.id,
    )
    request = RequestFactory().get(reverse('ipso-inbox-data'))
    request.user = writer_user

    with django_assert_num_queries(3):
        resp = ipso_views.inbox_data(request)

    assert resp.status_code == 200
    data = json.loads(resp.content)
    rows = [dict(zip(data['columns'], row)) for row in data['rows']]
    assert any(
        row[S.IPSO_COL_WORK_PACKAGE].startswith('Piano Ipso') for row in rows
    )
    assert any(
        row[S.IPSO_COL_WORK_PACKAGE] == 'Ipso survey - Ipso survey grid'
        for row in rows
    )
    assert all('harvest:' not in json.dumps(row) for row in rows)
    assert all('sampling_survey:' not in json.dumps(row) for row in rows)


@override_settings(IPSO_SECRET='test-token')
def test_upload_detail_previews_staged_records(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species, record_overrides={'h_m': '22.5'})
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data[UPLOAD]['state'] == IpsoUploadState.RECEIVED
    assert data[UPLOAD][FIELD_MODE] == IPSO_MODE_MARTELLATE
    assert data[UPLOAD][FIELD_MODE_LABEL] == S.IPSO_MODE_MARTELLATE_LABEL
    assert data[UPLOAD][FIELD_RECORD_DATE] == '2026-06-17'
    assert data['record_count'] == 1
    assert data[RECORDS][0]['seq'] == 1
    assert data[RECORDS][0]['parcel'] == 'Capistrano 1'
    assert data[RECORDS][0]['species'] == 'Abete'
    assert data[RECORDS][0][FIELD_H_M] == 22.5
    assert data[RECORDS][0][FIELD_LON] == 16.12345


@override_settings(IPSO_SECRET='test-token')
def test_upload_detail_reports_staged_checksum_mismatch(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    upload = _stage_upload_direct(settings, tmp_path, payload)
    upload_json = Path(upload.inbox_path) / 'upload.json'
    staged = json.loads(upload_json.read_text(encoding='utf-8'))
    staged[RECORDS][0][FIELD_NUMBER] = 99
    upload_json.write_text(json.dumps(staged), encoding='utf-8')

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    assert resp.json()['file_error'] == S.IPSO_ERR_UPLOAD_CHECKSUM_MISMATCH
    assert resp.json()[RECORDS] == []


@pytest.mark.parametrize(('method', 'url_name', 'body'), [
    ('get', 'ipso-upload-detail', None),
    ('post', 'ipso-upload-reject', {}),
    ('get', 'ipso-upload-download', None),
    ('post', 'ipso-upload-delete', {}),
    ('post', 'ipso-upload-mode', {FIELD_MODE: IPSO_MODE_PAI}),
    ('post', 'ipso-upload-import-martellate', {FIELD_HARVEST_PLAN_ITEM_ID: 1}),
    ('post', 'ipso-upload-import-samples', {FIELD_SURVEY_ID: 1}),
    ('post', 'ipso-upload-import-pai', {}),
])
def test_upload_id_endpoints_return_404_for_unknown_upload(
        admin_client, method, url_name, body):
    url = reverse(url_name, args=[999999])
    if method == 'get':
        resp = admin_client.get(url)
    else:
        resp = admin_client.post(
            url, data=json.dumps(body), content_type='application/json',
        )

    assert resp.status_code == 404


@override_settings(IPSO_SECRET='test-token')
def test_writer_can_reject_upload(
        writer_client, writer_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    nonce = 'ipso-reject-test-nonce'
    resp = writer_client.post(
        reverse('ipso-upload-reject', args=[upload.id]),
        data=json.dumps({FIELD_REASON: 'Duplicato', FIELD_NONCE: nonce}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.REJECTED
    assert upload.error_summary == 'Duplicato'
    _assert_nonce_saved(writer_user, nonce)
    assert writer_client.get(reverse('ipso-inbox-data')).json()[PENDING_COUNT] == 0


@pytest.mark.parametrize('state', [
    IpsoUploadState.IMPORTED,
    IpsoUploadState.REJECTED,
    IpsoUploadState.CONFLICT,
])
@override_settings(IPSO_SECRET='test-token')
def test_reject_rejects_non_received_upload_states(
        writer_client, parcels, species, settings, tmp_path, state):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    upload.state = state
    upload.error_summary = 'existing state detail'
    upload.save(update_fields=['state', 'error_summary'])

    resp = writer_client.post(
        reverse('ipso-upload-reject', args=[upload.id]),
        data=json.dumps({FIELD_REASON: 'new reject reason'}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    upload.refresh_from_db()
    assert upload.state == state
    assert upload.error_summary == 'existing state detail'


@override_settings(IPSO_SECRET='test-token')
def test_reader_cannot_reject_upload(reader_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = admin_client.get(reverse('ipso-upload-download', args=[upload.id]))

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/zip'
    assert 'no-store' in resp['Cache-Control']
    assert f'ipso-upload-{upload.session_id}.zip' in resp['Content-Disposition']
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert sorted(zf.namelist()) == ['export.csv', 'upload.json', 'upload.sha256']
    assert json.loads(zf.read('upload.json'))[SESSION][FIELD_MODE] == IPSO_MODE_MARTELLATE
    assert zf.read('export.csv').decode() == 'csv backup'


@override_settings(IPSO_SECRET='test-token')
def test_writer_cannot_download_delete_or_edit_upload_mode(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

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
    assert upload.mode == IPSO_MODE_MARTELLATE
    assert Path(upload.inbox_path).is_dir()


@override_settings(IPSO_SECRET='test-token')
def test_admin_updates_upload_mode_and_staged_payload(
        admin_client, admin_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333338',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    inbox_path = Path(upload.inbox_path)
    original_csv = (inbox_path / 'export.csv').read_text()

    nonce = 'ipso-mode-test-nonce'
    resp = admin_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({FIELD_MODE: IPSO_MODE_MARTELLATE, FIELD_NONCE: nonce}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    upload.refresh_from_db()
    assert upload.mode == IPSO_MODE_MARTELLATE
    staged = json.loads((inbox_path / 'upload.json').read_text())
    assert staged[SESSION][FIELD_MODE] == IPSO_MODE_MARTELLATE
    assert (inbox_path / 'upload.sha256').read_text().strip() == upload.checksum
    assert (inbox_path / 'export.csv').read_text() == original_csv
    assert resp.json()[UPLOAD][FIELD_MODE] == IPSO_MODE_MARTELLATE
    assert resp.json()[UPLOAD][FIELD_MODE_LABEL] == S.IPSO_MODE_MARTELLATE_LABEL
    _assert_nonce_saved(admin_user, nonce)


@override_settings(IPSO_SECRET='test-token')
def test_admin_updates_rejected_upload_mode_preserves_rejection_detail(
        admin_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(
        parcels, species, mode=IPSO_MODE_PAI,
        session_id='33333333-3333-4333-8333-333333333339',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    upload.state = IpsoUploadState.REJECTED
    upload.error_summary = 'operator rejected this upload'
    upload.save(update_fields=['state', 'error_summary'])
    original_updated_at = upload.updated_at

    resp = admin_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({FIELD_MODE: IPSO_MODE_MARTELLATE}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    upload.refresh_from_db()
    assert upload.mode == IPSO_MODE_MARTELLATE
    assert upload.error_summary == 'operator rejected this upload'
    assert upload.updated_at >= original_updated_at
    assert upload.history.filter(
        mode=IPSO_MODE_MARTELLATE,
        error_summary='operator rejected this upload',
    ).exists()


@override_settings(IPSO_SECRET='test-token')
def test_admin_rejects_unsupported_upload_mode(
        admin_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    original_mode = upload.mode
    original_checksum = upload.checksum

    resp = admin_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({FIELD_MODE: 'unsupported'}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert resp.json()[MESSAGE] == S.IPSO_ERR_MODE_UNSUPPORTED
    upload.refresh_from_db()
    assert upload.mode == original_mode
    assert upload.checksum == original_checksum


@override_settings(IPSO_SECRET='test-token')
def test_admin_cannot_update_mode_with_corrupted_staged_payload(
        admin_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    original_mode = upload.mode
    original_checksum = upload.checksum
    upload_json = Path(upload.inbox_path) / 'upload.json'
    upload_json.write_text('{not-json', encoding='utf-8')

    resp = admin_client.post(
        reverse('ipso-upload-mode', args=[upload.id]),
        data=json.dumps({FIELD_MODE: IPSO_MODE_PAI}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert resp.json()[MESSAGE] == S.IPSO_ERR_UPLOAD_JSON_INVALID
    upload.refresh_from_db()
    assert upload.mode == original_mode
    assert upload.checksum == original_checksum
    assert upload_json.read_text(encoding='utf-8') == '{not-json'


@override_settings(IPSO_SECRET='test-token')
def test_admin_cannot_update_mode_after_domain_import(
        admin_client, writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
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
    assert S.IPSO_ERR_IMPORTED_CANNOT_EDIT_MODE in resp.json()[MESSAGE]
    upload.refresh_from_db()
    assert upload.mode == IPSO_MODE_MARTELLATE


@override_settings(IPSO_SECRET='test-token')
def test_admin_deletes_staged_upload_record_and_files(
        admin_client, admin_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    inbox_path = Path(upload.inbox_path)

    nonce = 'ipso-delete-test-nonce'
    resp = admin_client.post(
        reverse('ipso-upload-delete', args=[upload.id]),
        data=json.dumps({FIELD_NONCE: nonce}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert not IpsoUpload.objects.filter(id=upload.id).exists()
    assert IpsoUpload.history.filter(id=upload.id, history_type='-').exists()
    assert not inbox_path.exists()
    _assert_nonce_saved(admin_user, nonce)


@override_settings(IPSO_SECRET='test-token')
def test_upload_detail_lists_martellate_targets(writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    item.note = 'Nota molto lunga per selettore destinazione'
    item.save(update_fields=['note'])
    payload = _upload_payload(parcels, species)
    payload[SESSION]['work_package_id'] = f'harvest:{item.id}'
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

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
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.get(reverse('ipso-upload-detail', args=[upload.id]))

    assert resp.status_code == 200
    data = resp.json()
    assert data['suggested_target_id'] == target.id
    assert any(t['id'] == target.id for t in data['targets'])
    assert any('Ipso inactive target' in t['label'] for t in data['targets'])
    assert data[UPLOAD][FIELD_WORK_PACKAGE_LABEL] == 'Ipso inactive target - Ipso survey grid'
    assert data[RECORDS][0][FIELD_SAMPLE_AREA_ID] == area.number
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
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_MODE_UNSUPPORTED in resp.json()[MESSAGE]
    assert TreeMark.objects.count() == 0
    assert TreeSample.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.target_type == ''


@override_settings(IPSO_SECRET='test-token')
def test_abies_1_1_4_martellate_payload_uploads_and_imports(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _legacy_ipso_1_1_4_payload(
        parcels, species,
        session_id='41414141-1414-4141-8141-414141414101',
    )
    assert FIELD_OPERATOR not in payload[RECORDS][0]

    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({FIELD_HARVEST_PLAN_ITEM_ID: item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200, resp.content
    mark = TreeMark.objects.select_related('tree', 'parcel').get()
    assert mark.parcel == parcels[0]
    assert mark.tree.parcel == parcels[0]
    assert mark.operator == 'Mario Rossi'
    assert mark.d_cm == 42
    assert mark.h_m == Decimal('22.00')
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED


@override_settings(IPSO_SECRET='test-token')
def test_abies_1_1_4_samples_payload_uploads_and_imports(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0], name='Legacy Ipso samples')
    payload = _legacy_ipso_1_1_4_payload(
        parcels, species, mode=IPSO_MODE_SAMPLES,
        session_id='42424242-2424-4242-8242-424242424202',
        work_package_id=f'sampling_survey:{survey.id}',
        sample_area_id=area.id,
        record_overrides={FIELD_H_MEASURED: True},
    )
    assert FIELD_OPERATOR not in payload[RECORDS][0]
    assert FIELD_NOTE not in payload[RECORDS][0]

    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({FIELD_SURVEY_ID: survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200, resp.content
    row = (TreeSample.objects
           .select_related('sample', 'tree', 'parcel')
           .get(sample__survey=survey))
    assert row.sample.sample_area == area
    assert row.parcel == parcels[0]
    assert row.tree.parcel == parcels[0]
    assert row.h_measured is True
    assert row.operator == 'Mario Rossi'
    assert row.note == ''
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED


@override_settings(IPSO_SECRET='test-token')
def test_old_normalized_samples_upload_imports_after_release_1(
        writer_client, parcels, species, settings, tmp_path):
    """Inbox JSON written by abies-1.1.4 lacks sample operator/note fields."""
    survey, area = _sample_survey(parcels[0], name='Old staged samples')
    payload = _legacy_ipso_1_1_4_payload(
        parcels, species, mode=IPSO_MODE_SAMPLES,
        session_id='42424242-2424-4242-8242-424242424203',
        work_package_id=f'sampling_survey:{survey.id}',
        sample_area_id=area.id,
    )
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({FIELD_SURVEY_ID: survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200, resp.content
    row = TreeSample.objects.get(sample__survey=survey)
    assert row.operator == 'Mario Rossi'
    assert row.note == ''


@override_settings(IPSO_SECRET='test-token')
def test_abies_1_1_4_pai_payload_uploads_and_imports(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    payload = _legacy_ipso_1_1_4_payload(
        parcels, species, mode=IPSO_MODE_PAI,
        session_id='43434343-3434-4343-8343-434343434303',
        number=12,
    )

    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 200, resp.content
    row = (TreeSample.objects
           .select_related('sample', 'tree', 'parcel')
           .get(preserved_number__isnull=False))
    assert row.parcel == parcels[0]
    assert row.tree.parcel == parcels[0]
    assert row.preserved_number == 12
    assert row.sample.date == date(2026, 6, 17)
    assert row.operator == 'Mario Rossi'
    assert row.note == 'nota PAI da 1.1.4'
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_rejects_coppice_target(
        writer_client, regions, eclasses, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='C-target', region=regions[0],
        eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
        intervention_interval=18, standards_per_ha=75,
    )
    item = _harvest_item([coppice_parcel])
    payload = _upload_payload(parcels, species)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_INVALID_MARTELLATE_TARGET in resp.json()[MESSAGE]
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_rejects_rows_outside_parcel_target(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    target = parcels[0]
    other = Parcel.objects.create(
        name='other', region=target.region, eclass=target.eclass,
        area_ha=Decimal('1.0'),
    )
    item = _harvest_item([target])
    payload = _upload_payload(
        [other], species,
        session_id='11111111-1111-4111-8111-111111111131',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert (S.IPSO_ERR_IMPORT_RECORD_MARK_TARGET_MISMATCH.format(1)
            in resp.json()[MESSAGE])
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
        intervention_interval=18, standards_per_ha=75,
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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

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
    ('missing_json', S.IPSO_ERR_UPLOAD_JSON_MISSING),
    ('invalid_json', S.IPSO_ERR_UPLOAD_JSON_INVALID),
    ('missing_sha', S.IPSO_ERR_UPLOAD_SHA_MISSING),
    ('invalid_sha', S.IPSO_ERR_UPLOAD_SHA_INVALID),
    ('json_checksum_mismatch', S.IPSO_ERR_UPLOAD_CHECKSUM_MISMATCH),
    ('db_checksum_mismatch', S.IPSO_ERR_UPLOAD_CHECKSUM_MISMATCH),
])
@override_settings(IPSO_SECRET='test-token')
def test_import_reports_staged_payload_file_errors(
        writer_client, parcels, species, settings, tmp_path, file_state,
        expected_error):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(parcels, species)
    upload = _stage_upload_direct(settings, tmp_path, payload)
    inbox_path = Path(upload.inbox_path)
    upload_json = inbox_path / 'upload.json'
    if file_state == 'missing_json':
        upload_json.unlink()
    elif file_state == 'invalid_json':
        upload_json.write_text('{not-json', encoding='utf-8')
    elif file_state == 'missing_sha':
        (inbox_path / 'upload.sha256').unlink()
    elif file_state == 'invalid_sha':
        (inbox_path / 'upload.sha256').write_text('not-a-sha\n', encoding='utf-8')
    elif file_state == 'json_checksum_mismatch':
        staged = json.loads(upload_json.read_text(encoding='utf-8'))
        staged[RECORDS][0][FIELD_NUMBER] = 99
        upload_json.write_text(json.dumps(staged), encoding='utf-8')
    elif file_state == 'db_checksum_mismatch':
        upload.checksum = '0' * 64
        upload.save(update_fields=['checksum'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert expected_error in resp.json()[MESSAGE]
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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    upload.state = state
    upload.save(update_fields=['state'])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_UPLOAD_NOT_RECEIVED in resp.json()[MESSAGE]
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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    nonce = 'ipso-import-mark-test-nonce'
    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id, FIELD_NONCE: nonce}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()[IMPORTED] == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.imported_by == writer_user
    assert upload.target_type == 'harvest_plan_item'
    assert upload.target_id == item.id
    _assert_nonce_saved(writer_user, nonce)
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
        harvest_plan_item=item, tree=tree, parcel=parcels[0], number=7,
        date=date(2026, 6, 16), d_cm=30, h_m=Decimal('18.00'),
        operator='Mario Rossi',
    )
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111121',
        record_overrides={FIELD_NUMBER: 7},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.ERR_MARK_NUMBER_DUPLICATE.format(7) in resp.json()[MESSAGE]
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
    payload[RECORDS][0][FIELD_NUMBER] = 'abc'
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_INVALID.format(1) in resp.json()[MESSAGE]
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
    payload[RECORDS][0][FIELD_NUMBER] = 0
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(1) in resp.json()[MESSAGE]
    assert TreeMark.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_preserves_blank_numbers_without_auto_numbering(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    for n in [1, 2, 3]:
        tree = Tree.objects.create(species=species[0], parcel=parcels[0])
        TreeMark.objects.create(
            harvest_plan_item=item, tree=tree, parcel=parcels[0], number=n,
            date=date(2026, 6, 16), d_cm=30, h_m=Decimal('18.00'),
            operator='Mario Rossi',
        )
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111118',
        record_overrides={FIELD_NUMBER: None, FIELD_CLIENT_RECORD_ID: 'd'},
    )
    base = payload[RECORDS][0]
    payload[RECORDS] = []
    for idx, number in enumerate([None, None, 4, 5, 6], start=1):
        row = dict(base)
        row[FIELD_CLIENT_RECORD_ID] = f'new-{idx}'
        row[FIELD_NUMBER] = number
        row[FIELD_D_CM] = 40 + idx
        payload[RECORDS].append(row)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()[IMPORTED] == 5
    assert list(
        TreeMark.objects.filter(harvest_plan_item=item)
        .order_by('id')
        .values_list('number', flat=True)
    ) == [1, 2, 3, None, None, 4, 5, 6]


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_allows_recovery_to_survey_with_same_grid(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    source_survey, area = _sample_survey(parcels[0], name='Source survey')
    target_survey = Survey.objects.create(
        name='Target survey', sample_grid=source_survey.sample_grid, active=True,
    )
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222231',
        session_overrides={
            'work_package_id': f'sampling_survey:{source_survey.id}',
        },
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': target_survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200, resp.json()
    assert Sample.objects.get().survey == target_survey
    assert TreeSample.objects.count() == 1


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_write_errors_do_not_claim_upload(
        writer_client, parcels, species, settings, tmp_path, monkeypatch):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111122',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    class FailedMarkImport:
        imported = 0
        skipped_duplicates = 0
        errors = ['late mark validation error']

    monkeypatch.setattr(
        ipso_views, 'import_mark_rows',
        lambda _item, _rows: FailedMarkImport(),
    )

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert 'late mark validation error' in resp.json()[MESSAGE]
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.error_summary == 'late mark validation error'
    assert TreeMark.objects.count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_martellate_import_integrity_error_returns_validation(
        writer_client, parcels, species, settings, tmp_path, monkeypatch):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    item = _harvest_item(parcels)
    payload = _upload_payload(
        parcels, species,
        session_id='11111111-1111-4111-8111-111111111123',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    def raise_integrity_error(_item, _rows):
        raise IntegrityError

    monkeypatch.setattr(ipso_views, 'import_mark_rows', raise_integrity_error)

    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORT_MARK_CONFLICT in resp.json()[MESSAGE]
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert S.IPSO_ERR_IMPORT_MARK_CONFLICT in upload.error_summary


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_integrity_error_returns_validation(
        writer_client, parcels, species, settings, tmp_path, monkeypatch):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222225',
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    def raise_integrity_error(_survey, _rows):
        raise IntegrityError

    monkeypatch.setattr(ipso_views.csv_trees, 'apply', raise_integrity_error)

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORT_SAMPLE_CONFLICT in resp.json()[MESSAGE]
    assert TreeSample.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert S.IPSO_ERR_IMPORT_SAMPLE_CONFLICT in upload.error_summary


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_target_with_different_source_grid(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    source_survey, area = _sample_survey(parcels[0], name='Source survey')
    target_survey, _ = _sample_survey(parcels[0], name='Target survey')
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222232',
        session_overrides={
            'work_package_id': f'sampling_survey:{source_survey.id}',
        },
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': target_survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_SAMPLES_TARGET_GRID_MISMATCH in resp.json()[MESSAGE]
    assert TreeSample.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_area_outside_selected_survey_grid(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    source_survey, area = _sample_survey(parcels[0], name='Source survey')
    target_survey, _ = _sample_survey(parcels[0], name='Target survey')
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222228',
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': target_survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert (S.IPSO_ERR_IMPORT_RECORD_AREA_OUT_OF_SURVEY.format(1)
            in resp.json()[MESSAGE])
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
    settings.DIGEST_DIR = tmp_path / 'digests'
    survey, area = _sample_survey(parcels[0])
    trees_url = f'/api/campionamenti/trees/{survey.id}/'
    assert _read_gzip_json(writer_client.get(trees_url))[ROWS] == []
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222222',
        record_overrides={
            FIELD_SAMPLE_AREA_ID: area.id,
            FIELD_H_MEASURED: True,
            FIELD_NOTE: 'nota rilevamento',
        },
    )
    payload[RECORDS].append({
        **payload[RECORDS][0],
        FIELD_CLIENT_RECORD_ID: '2',
        FIELD_NUMBER: 2,
        FIELD_H_MEASURED: False,
    })
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    nonce = 'ipso-import-samples-test-nonce'
    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id, FIELD_NONCE: nonce}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()[IMPORTED] == 2
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.imported_by == writer_user
    assert upload.target_type == 'survey'
    assert upload.target_id == survey.id
    _assert_nonce_saved(writer_user, nonce)
    sample = Sample.objects.get(survey=survey, sample_area=area)
    assert sample.date == date(2026, 6, 17)
    samples = list(
        TreeSample.objects
        .select_related('tree', 'tree__species', 'parcel')
        .filter(sample=sample)
        .order_by(FIELD_NUMBER)
    )
    assert [ts.h_measured for ts in samples] == [True, False]
    ts = samples[0]
    assert ts.tree.parcel == parcels[0]
    assert ts.tree.species == species[0]
    assert ts.tree.lat == 38.51234
    assert ts.tree.lon == 16.12345
    assert ts.parcel == parcels[0]
    assert ts.lat == 38.51234
    assert ts.lon == 16.12345
    assert ts.acc_m == 5
    assert ts.operator == 'Mario Rossi'
    assert ts.note == 'nota rilevamento'
    assert ts.number == 1
    assert ts.d_cm == 42
    assert ts.h_m == Decimal('22.00')
    assert ts.volume_m3 is not None
    trees = _read_gzip_json(writer_client.get(trees_url))
    assert len(trees[ROWS]) == 2
    assert trees[ROWS][0][trees[COLUMNS].index(ROW_ID)] == ts.id
    assert trees[ROWS][0][trees[COLUMNS].index(S.COL_TREE_NUM)] == 1


@override_settings(IPSO_SECRET='test-token')
def test_writer_rejects_samples_import_into_unstructured_survey(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    unstructured = Survey.objects.create(name='Ipso unstructured target')
    structured, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='23232323-2323-4323-8323-232323232323',
        session_overrides={'work_package_id': f'sampling_survey:{structured.id}'},
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': unstructured.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.ERR_SURVEY_STRUCTURED_REQUIRED in resp.json()[MESSAGE]
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert Sample.objects.filter(survey=unstructured).count() == 0


@override_settings(IPSO_SECRET='test-token')
def test_samples_import_rejects_staged_missing_number(
        writer_client, parcels, species, settings, tmp_path):
    survey, area = _sample_survey(parcels[0])
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222230',
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id, FIELD_NUMBER: None},
    )
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(1) in resp.json()[MESSAGE]
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
        intervention_interval=18, standards_per_ha=75,
    )
    survey, area = _sample_survey(coppice_parcel, name='Ceduo survey')
    payload = _upload_payload(
        [coppice_parcel], species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222223',
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id, FIELD_SPECIES_ID: species[1].id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

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
        intervention_interval=18, standards_per_ha=75,
    )
    survey, area = _sample_survey(coppice_parcel, name='Ceduo shoots survey')
    payload = _upload_payload(
        [coppice_parcel], species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222225',
        record_overrides={
            FIELD_SAMPLE_AREA_ID: area.id,
            'coppice': True,
            FIELD_SHOOT: 1,
        },
    )
    second = dict(payload[RECORDS][0])
    second[FIELD_CLIENT_RECORD_ID] = '2'
    second[FIELD_SHOOT] = 2
    payload[RECORDS].append(second)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()[IMPORTED] == 2
    rows = list(TreeSample.objects.order_by('shoot'))
    assert [row.number for row in rows] == [1, 1]
    assert [row.shoot for row in rows] == [1, 2]
    assert {row.tree_id for row in rows} == {rows[0].tree_id}
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
        sample=sample, tree=tree, parcel=parcels[0],
        number=1, d_cm=30, h_m=Decimal('18.00'),
    )
    payload = _upload_payload(
        parcels, species, mode='samples',
        session_id='22222222-2222-4222-8222-222222222226',
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(1) in resp.json()[MESSAGE]
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
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id, FIELD_NUMBER: 9, FIELD_SHOOT: 0},
    )
    second = dict(payload[RECORDS][0])
    second[FIELD_CLIENT_RECORD_ID] = '2'
    second[FIELD_D_CM] = 43
    payload[RECORDS].append(second)
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-samples', args=[upload.id]),
        data=json.dumps({'survey_id': survey.id}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(2) in resp.json()[MESSAGE]
    assert TreeSample.objects.count() == 0

@override_settings(IPSO_SECRET='test-token')
def test_pai_import_rejects_duplicate_tree_number_in_parcel(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    tree = Tree.objects.create(species=species[0], parcel=parcels[0], preserved=True)
    _preserved_sample(
        tree, parcels[0], number=1, sample_date=date(2026, 6, 17),
        d_cm=30, h_m=Decimal('18.00'), lat=38.51234, lon=16.12345,
    )
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333335',
        record_overrides={FIELD_NUMBER: 1},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert resp.json()[MESSAGE] == (
        S.IPSO_ERR_IMPORT_RECORD_PAI_NUMBER_DUPLICATE.format(1)
    )
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 1
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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    def raise_integrity_error(rows):
        raise IntegrityError

    monkeypatch.setattr(ipso_views, 'apply_pai_rows', raise_integrity_error)

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert resp.json()[MESSAGE] == S.IPSO_ERR_IMPORT_PAI_NUMBER_CONFLICT
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.error_summary == S.IPSO_ERR_IMPORT_PAI_NUMBER_CONFLICT


@override_settings(IPSO_SECRET='test-token')
def test_pai_import_rejects_staged_missing_number(
        writer_client, parcels, species, settings, tmp_path):
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333337',
        record_overrides={FIELD_NUMBER: None},
    )
    upload = _stage_upload_direct(settings, tmp_path, payload)

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 400
    assert S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(1) in resp.json()[MESSAGE]
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED


@override_settings(IPSO_SECRET='test-token')
def test_writer_imports_pai_upload(writer_client, writer_user, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    settings.DIGEST_DIR = tmp_path / 'digests'
    preserved_url = '/api/bosco/preserved-trees/data/'
    assert _read_gzip_json(writer_client.get(preserved_url))[ROWS] == []
    payload = _upload_payload(
        parcels, species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333333',
        record_overrides={
            FIELD_NUMBER: 1,
            'estimated_birth_year': 1920,
            'note': 'nota PAI',
        },
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    nonce = 'ipso-import-pai-test-nonce'
    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({FIELD_NONCE: nonce}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()[IMPORTED] == 1
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.IMPORTED
    assert upload.imported_by == writer_user
    assert upload.target_type == 'pai'
    _assert_nonce_saved(writer_user, nonce)
    pai = (TreeSample.objects
           .select_related('sample', 'tree', 'tree__species', 'parcel')
           .get(preserved_number__isnull=False))
    assert pai.tree.parcel == parcels[0]
    assert pai.parcel == parcels[0]
    assert pai.tree.species == species[0]
    assert pai.preserved_number == 1
    assert pai.sample.date == date(2026, 6, 17)
    assert pai.d_cm == 42
    assert pai.h_m == Decimal('22.00')
    assert pai.h_measured is True
    assert pai.operator == 'Mario Rossi'
    assert pai.note == 'nota PAI'
    assert pai.tree.coppice is False
    assert pai.volume_m3 is None
    assert pai.mass_q is None
    preserved = _read_gzip_json(writer_client.get(preserved_url))
    assert len(preserved[ROWS]) == 1
    assert preserved[ROWS][0][preserved[COLUMNS].index(ROW_ID)] == pai.id
    assert preserved[ROWS][0][preserved[COLUMNS].index(S.COL_NUMBER)] == 1


@override_settings(IPSO_SECRET='test-token')
def test_pai_import_supports_coppice_parcels(
        writer_client, regions, eclasses, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    coppice_parcel = Parcel.objects.create(
        name='P1', region=regions[0], eclass=next(e for e in eclasses if e.coppice),
        area_ha=Decimal('1.0'),
        intervention_interval=18, standards_per_ha=75,
    )
    payload = _upload_payload(
        [coppice_parcel], species, mode='pai',
        session_id='33333333-3333-4333-8333-333333333334',
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = writer_client.post(
        reverse('ipso-upload-import-pai', args=[upload.id]),
        data=json.dumps({}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    pai = TreeSample.objects.select_related('tree').get(preserved_number__isnull=False)
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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
    body = json.dumps({'harvest_plan_item_id': item.id})
    url = reverse('ipso-upload-import-martellate', args=[upload.id])

    first = writer_client.post(url, data=body, content_type='application/json')
    second = writer_client.post(url, data=body, content_type='application/json')

    assert first.status_code == 200
    assert first.json()[IMPORTED] == 1
    assert second.status_code == 400
    assert second.json()[MESSAGE] == S.IPSO_ERR_UPLOAD_NOT_RECEIVED
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
        record_overrides={FIELD_SAMPLE_AREA_ID: area.id},
    )
    assert _post_upload(Client(), payload).status_code == 200
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])
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
    upload = IpsoUpload.objects.get(session_id=payload[SESSION][FIELD_SESSION_ID])

    resp = reader_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 403
    assert TreeMark.objects.count() == 0
    upload.refresh_from_db()
    assert upload.state == IpsoUploadState.RECEIVED
