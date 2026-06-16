"""Tests for auth views: login, logout, shell access."""

import pytest
from django.test import Client

from apps.base.models import LoginMethod, User
from config import strings as S
from config.constants import (
    FIELD_PASSWORD, FIELD_USERNAME,
)


@pytest.fixture
def client(db):
    """All view tests need DB access (axes checks lockout on every request)."""
    return Client()


@pytest.fixture
def logged_in_client(admin_user):
    c = Client()
    c.force_login(admin_user)
    return c


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

class TestLoginPage:
    def test_get_renders(self, client):
        resp = client.get('/login/')
        assert resp.status_code == 200
        assert b'Nome utente' in resp.content

    def test_post_success_redirects(self, client, admin_user):
        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'testpass123!',
        })
        assert resp.status_code == 302
        assert resp.url == '/prelievi'

    def test_post_failure_returns_400(self, client, admin_user):
        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'wrong',
        })
        assert resp.status_code == 400
        assert 'non validi' in resp.content.decode()

    def test_oauth_method_user_cannot_use_password_login(self, client):
        User.objects.create_user(
            username='oauthuser@example.com',
            email='oauthuser@example.com',
            password='oauthpass123!',
            login_method=LoginMethod.OAUTH,
        )

        resp = client.post('/login/', {
            FIELD_USERNAME: 'oauthuser@example.com',
            FIELD_PASSWORD: 'oauthpass123!',
        })

        assert resp.status_code == 400
        assert 'non validi' in resp.content.decode()

    def test_post_with_next(self, client, admin_user):
        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'testpass123!',
            'next': '/impostazioni',
        })
        assert resp.status_code == 302
        assert resp.url == '/impostazioni'

    @pytest.mark.parametrize('next_url', [
        'https://evil.example/phish',
        '//evil.example/phish',
    ])
    def test_post_rejects_external_next(self, client, admin_user, next_url):
        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'testpass123!',
            'next': next_url,
        })
        assert resp.status_code == 302
        assert resp.url == '/prelievi'

    def test_authenticated_user_redirected_to_shell(self, logged_in_client):
        resp = logged_in_client.get('/login/')
        assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Shell access control
# ---------------------------------------------------------------------------

class TestShellAccess:
    def test_unauthenticated_redirects_to_login(self, client):
        resp = client.get('/prelievi')
        assert resp.status_code == 302
        assert '/login/' in resp.url

    def test_authenticated_gets_shell(self, logged_in_client):
        resp = logged_in_client.get('/prelievi')
        assert resp.status_code == 200
        assert b'data-role=' in resp.content
        assert b'app.js' in resp.content

    def test_shell_serves_all_domain_paths(self, logged_in_client):
        for path in ('/bosco', '/prelievi',
                     '/controllo', '/impostazioni'):
            resp = logged_in_client.get(path)
            assert resp.status_code == 200, f'{path} returned {resp.status_code}'

    def test_shell_contains_csrf(self, logged_in_client):
        resp = logged_in_client.get('/prelievi')
        assert b'data-csrf=' in resp.content

    def test_shell_contains_user_role(self, logged_in_client):
        resp = logged_in_client.get('/prelievi')
        assert 'data-role="admin"' in resp.content.decode()


# ---------------------------------------------------------------------------
# Shared digest endpoints
# ---------------------------------------------------------------------------

class TestSharedDigests:
    def test_species_data_requires_login(self, client):
        resp = client.get('/api/species/data/')
        assert resp.status_code == 302
        assert '/login/' in resp.url

    def test_species_data_authenticated(self, logged_in_client, species):
        resp = logged_in_client.get('/api/species/data/')
        assert resp.status_code == 200
        assert resp['Content-Encoding'] == 'gzip'
        assert resp['Cache-Control'] == 'no-store'


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

class TestLogout:
    def test_logout_redirects_to_login(self, logged_in_client):
        resp = logged_in_client.get('/logout/')
        assert resp.status_code == 302

    def test_logout_clears_session(self, logged_in_client):
        logged_in_client.get('/logout/')
        resp = logged_in_client.get('/prelievi')
        assert resp.status_code == 302  # redirected to login
