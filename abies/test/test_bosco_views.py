"""Tests for Bosco API views."""

import pytest
from django.test import Client


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
