"""Tests for Bosco API views."""

import pytest
from django.test import Client

from apps.base.models import Region, Tree
from config.constants import (
    DELETES, DIGEST_PRESERVED_TREES, FIELD_LAT, FIELD_LON, FIELD_NONCE,
    FIELD_PARCEL_ID, FIELD_SPECIES_ID, FIELD_YEAR, PATCHES, ROW_ID, STATUS,
    STATUS_CONFLICT, VERSION,
)


@pytest.fixture
def reader_client(reader_user):
    c = Client()
    c.force_login(reader_user)
    return c


@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


@pytest.mark.parametrize('path', [
    '/api/bosco/parcels/data/',
    '/api/bosco/species/data/',
    '/api/bosco/preserved-trees/data/',
    '/api/bosco/future-production/data/',
    '/api/bosco/parcel-dendrometry/data/',
    '/api/bosco/parcel-dendrometry-points/data/',
])
def test_bosco_digest_endpoints_reader_access(
        reader_client, path, parcels, species, tmp_path, settings,
):
    settings.DIGEST_DIR = tmp_path

    resp = reader_client.get(path)

    assert resp.status_code == 200
    assert resp['Content-Encoding'] == 'gzip'
    assert resp['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('path', [
    '/api/bosco/parcels/data/',
    '/api/bosco/species/data/',
    '/api/bosco/preserved-trees/data/',
    '/api/bosco/future-production/data/',
    '/api/bosco/parcel-dendrometry/data/',
    '/api/bosco/parcel-dendrometry-points/data/',
])
def test_bosco_digest_endpoints_require_login(client, path):
    resp = client.get(path)
    assert resp.status_code == 302
    assert '/login/' in resp.url


def test_pai_form_requires_writer(reader_client, regions):
    resp = reader_client.get(f'/api/bosco/pai/form/?region_id={regions[0].id}')
    assert resp.status_code == 403


def test_pai_form_writer_access(writer_client, regions, parcels, species):
    resp = writer_client.get(f'/api/bosco/pai/form/?region_id={regions[0].id}')

    assert resp.status_code == 200
    html = resp.json()['html']
    assert 'id="bosco-pai-form"' in html
    assert 'Capistrano 1' in html


def test_pai_save_creates_preserved_tree(writer_client, parcels, species):
    body = {
        FIELD_SPECIES_ID: str(species[0].id),
        FIELD_PARCEL_ID: str(parcels[0].id),
        FIELD_YEAR: '2026',
        FIELD_LAT: '38,123456',
        FIELD_LON: '16.123456',
        FIELD_NONCE: 'pai-create',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree = Tree.objects.get(species=species[0], parcel=parcels[0])
    assert tree.preserved is True
    assert tree.year == 2026
    assert tree.lat == 38.12346
    data = resp.json()
    assert data[PATCHES][0]['data_id'] == DIGEST_PRESERVED_TREES
    assert data[PATCHES][0]['row_id'] == tree.id
    assert data[PATCHES][0]['record'][0] == tree.id


def test_pai_save_stale_edit_conflicts(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        year=2025, lat=38.1, lon=16.1, version=2,
    )
    body = {
        ROW_ID: str(tree.id), VERSION: '1',
        FIELD_SPECIES_ID: str(species[1].id),
        FIELD_PARCEL_ID: str(parcels[1].id),
        FIELD_YEAR: '2026', FIELD_LAT: '38.2', FIELD_LON: '16.2',
        FIELD_NONCE: 'pai-conflict',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    data = resp.json()
    assert data[STATUS] == STATUS_CONFLICT
    assert data[PATCHES][0]['record'][0] == tree.id
    assert 'bosco-pai-form' in data['html']


def test_pai_delete_clears_preserved_flag(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        year=2025, lat=38.1, lon=16.1, version=3,
    )
    body = {ROW_ID: str(tree.id), VERSION: '3', FIELD_NONCE: 'pai-delete'}

    resp = writer_client.post('/api/bosco/pai/delete/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree.refresh_from_db()
    assert tree.preserved is False
    assert tree.version == 4
    assert resp.json()[DELETES] == [{
        'data_id': DIGEST_PRESERVED_TREES,
        'row_id': tree.id,
    }]


def _stream_text(resp):
    return b''.join(resp.streaming_content).decode('utf-8')


def test_satellite_manifest_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    region_dir.mkdir()
    (region_dir / 'manifest.json').write_text(
        '{"dates":["2026-01-01"],"bbox":[[38,16],[39,17]]}',
    )
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/manifest/')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/json'
    assert resp['Cache-Control'] == 'no-cache'
    assert '"dates"' in _stream_text(resp)


def test_satellite_timeseries_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    region_dir.mkdir()
    (region_dir / 'timeseries.json').write_text(
        '{"dates":["2026-01-01"],"means":{"parcels":{}}}',
    )
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/timeseries/')

    assert resp.status_code == 200
    assert '"means"' in _stream_text(resp)


def test_satellite_manifest_conditional_get(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    region_dir.mkdir()
    (region_dir / 'manifest.json').write_text('{"dates":["2026-01-01"]}')
    settings.SATELLITE_DIR = tmp_path

    r1 = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/manifest/')
    r2 = reader_client.get(
        f'/api/bosco/satellite/{regions[0].id}/manifest/',
        HTTP_IF_MODIFIED_SINCE=r1['Last-Modified'],
    )

    assert r2.status_code == 304
    assert r2['Cache-Control'] == 'no-cache'


def test_satellite_unknown_region_404(reader_client, tmp_path, settings):
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get('/api/bosco/satellite/999999/manifest/')

    assert resp.status_code == 404


@pytest.mark.parametrize('path_suffix', ['manifest/', 'timeseries/'])
def test_satellite_endpoints_require_login(client, regions, path_suffix):
    resp = client.get(f'/api/bosco/satellite/{regions[0].id}/{path_suffix}')
    assert resp.status_code == 302
    assert '/login/' in resp.url


def test_satellite_endpoint_404s_for_missing_file(reader_client, regions, tmp_path, settings):
    (tmp_path / regions[0].name).mkdir()
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/manifest/')

    assert resp.status_code == 404


def test_satellite_region_name_cannot_escape_base(reader_client, tmp_path, settings):
    region = Region.objects.create(name='../outside')
    outside = tmp_path.parent / 'outside'
    outside.mkdir(exist_ok=True)
    (outside / 'manifest.json').write_text('{"dates":[]}')
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{region.id}/manifest/')

    assert resp.status_code == 404
