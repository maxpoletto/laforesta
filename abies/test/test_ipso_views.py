"""Tests for the Abies-served Ipso PWA."""

import json
from decimal import Decimal
from pathlib import Path

from django.test import Client, override_settings
from django.urls import reverse

from apps.base.models import (
    HYPSO_FUNC_LN, HypsoParam, HypsoParamSet, HypsoParamSource,
)
from apps.ipso.models import IpsoUpload, IpsoUploadState


def _body(resp) -> bytes:
    return b''.join(resp.streaming_content)


def test_index_is_public_and_uses_relative_assets(db):
    resp = Client().get('/ipso/')

    assert resp.status_code == 200
    body = _body(resp).decode()
    assert '<title>Ipso' in body
    assert 'href="style.css"' in body
    assert 'src="app.js"' in body


def test_service_worker_served_from_ipso_scope(db):
    resp = Client().get('/ipso/sw.js')

    assert resp.status_code == 200
    assert resp['Content-Type'].startswith('text/javascript')
    assert b'ipso service worker' in _body(resp)


def test_manifest_is_served(db):
    resp = Client().get('/ipso/manifest.webmanifest')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/manifest+json'
    assert json.loads(_body(resp))['short_name'] == 'Ipso'


def test_reference_json_comes_from_abies_data(db, regions, parcels, species):
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
    assert data['species'][0]['common'] == 'Abete'
    parcel = next(p for p in data['parcels']
                  if p['compresa'] == 'Capistrano' and p['particella'] == '1')
    assert parcel['region_id'] == regions[0].id
    assert parcel['parcel_id'] == parcels[0].id
    assert data['ipsometrica']['Capistrano']['Abete'] == {'a': 7.0, 'b': -4.0}


def test_terreni_geojson_has_empty_fallback(db):
    resp = Client().get('/ipso/terreni.geojson')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/geo+json'
    assert resp.json() == {'type': 'FeatureCollection', 'features': []}


def _upload_payload(parcels, species, *, session_id='11111111-1111-4111-8111-111111111111'):
    parcel = parcels[0]
    sp = species[0]
    return {
        'session': {
            'session_id': session_id,
            'mode': 'martellate',
            'schema_version': 1,
            'reference_version': '',
            'work_package_id': '',
            'operator': 'Mario Rossi',
            'created_at': '2026-06-17T08:00:00Z',
            'completed_at': '2026-06-17T09:00:00Z',
            'catastrofata': False,
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
            'lat': 38.51234,
            'lon': 16.12345,
            'acc_m': 5,
        }],
        'csv_text': 'csv backup',
    }


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
    assert staged['records'][0]['lon'] == 16.12345
    assert (Path(upload.inbox_path) / 'upload.sha256').is_file()
    assert (Path(upload.inbox_path) / 'export.csv').read_text() == 'csv backup'


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
