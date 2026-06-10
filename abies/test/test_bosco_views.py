"""Tests for Bosco API views."""

import pytest
from django.test import Client

from apps.base.models import Region


@pytest.fixture
def reader_client(reader_user):
    c = Client()
    c.force_login(reader_user)
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
