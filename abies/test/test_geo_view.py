"""Tests for the geo file serving view."""

import pytest
from django.test import Client


@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


class TestGeoView:
    def test_unknown_file_404s(self, writer_client, tmp_path, settings):
        settings.GEO_DIR = tmp_path
        resp = writer_client.get('/api/geo/secret.txt')
        assert resp.status_code == 404

    def test_whitelisted_missing_file_404s(self, writer_client, tmp_path,
                                           settings):
        """A whitelisted name with no file on disk still 404s."""
        settings.GEO_DIR = tmp_path
        resp = writer_client.get('/api/geo/terreni.geojson')
        assert resp.status_code == 404

    def test_serves_whitelisted_file(self, writer_client, tmp_path, settings):
        settings.GEO_DIR = tmp_path
        (tmp_path / 'terreni.geojson').write_text(
            '{"type": "FeatureCollection", "features": []}'
        )
        resp = writer_client.get('/api/geo/terreni.geojson')
        assert resp.status_code == 200
        assert resp['Content-Type'] == 'application/geo+json'

    def test_path_traversal_blocked(self, writer_client, settings):
        """Whitelist rejects path-traversal attempts."""
        resp = writer_client.get('/api/geo/..%2F..%2Fetc%2Fpasswd')
        assert resp.status_code == 404

    def test_requires_auth(self, db):
        resp = Client().get('/api/geo/terreni.geojson')
        assert resp.status_code == 302

    def test_sets_no_cache(self, writer_client, tmp_path, settings):
        """Geo files are browser-cacheable but must revalidate: a reload
        picks up a changed file, while an unchanged one 304s (below)."""
        settings.GEO_DIR = tmp_path
        (tmp_path / 'terreni.geojson').write_text(
            '{"type": "FeatureCollection", "features": []}')
        resp = writer_client.get('/api/geo/terreni.geojson')
        assert resp.status_code == 200
        assert 'no-cache' in resp['Cache-Control']

    def test_unchanged_file_revalidates_to_304(self, writer_client, tmp_path,
                                               settings):
        """An unchanged geo file revalidates to 304 — no re-transfer."""
        settings.GEO_DIR = tmp_path
        (tmp_path / 'terreni.geojson').write_text(
            '{"type": "FeatureCollection", "features": []}')
        r1 = writer_client.get('/api/geo/terreni.geojson')
        r2 = writer_client.get('/api/geo/terreni.geojson',
                               HTTP_IF_MODIFIED_SINCE=r1['Last-Modified'])
        assert r2.status_code == 304
