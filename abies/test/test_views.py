"""Tests for auth views: login, logout, shell access."""

import pytest
from django.test import Client


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
        resp = client.get('/abies/login/')
        assert resp.status_code == 200
        assert b'Nome utente' in resp.content

    def test_post_success_redirects(self, client, admin_user):
        resp = client.post('/abies/login/', {
            'username': 'testadmin', 'password': 'testpass123!',
        })
        assert resp.status_code == 302
        assert resp.url == '/abies/prelievi'

    def test_post_failure_returns_400(self, client, admin_user):
        resp = client.post('/abies/login/', {
            'username': 'testadmin', 'password': 'wrong',
        })
        assert resp.status_code == 400
        assert 'non validi' in resp.content.decode()

    def test_post_with_next(self, client, admin_user):
        resp = client.post('/abies/login/', {
            'username': 'testadmin', 'password': 'testpass123!',
            'next': '/abies/impostazioni',
        })
        assert resp.status_code == 302
        assert resp.url == '/abies/impostazioni'

    def test_authenticated_user_redirected_to_shell(self, logged_in_client):
        resp = logged_in_client.get('/abies/login/')
        assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Shell access control
# ---------------------------------------------------------------------------

class TestShellAccess:
    def test_unauthenticated_redirects_to_login(self, client):
        resp = client.get('/abies/prelievi')
        assert resp.status_code == 302
        assert '/abies/login/' in resp.url

    def test_authenticated_gets_shell(self, logged_in_client):
        resp = logged_in_client.get('/abies/prelievi')
        assert resp.status_code == 200
        assert b'data-role=' in resp.content
        assert b'app.js' in resp.content

    def test_shell_serves_all_domain_paths(self, logged_in_client):
        for path in ('/abies/bosco', '/abies/prelievi',
                     '/abies/controllo', '/abies/impostazioni'):
            resp = logged_in_client.get(path)
            assert resp.status_code == 200, f'{path} returned {resp.status_code}'

    def test_shell_contains_csrf(self, logged_in_client):
        resp = logged_in_client.get('/abies/prelievi')
        assert b'data-csrf=' in resp.content

    def test_shell_contains_user_role(self, logged_in_client):
        resp = logged_in_client.get('/abies/prelievi')
        assert 'data-role="admin"' in resp.content.decode()


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

class TestLogout:
    def test_logout_redirects_to_login(self, logged_in_client):
        resp = logged_in_client.get('/abies/logout/')
        assert resp.status_code == 302

    def test_logout_clears_session(self, logged_in_client):
        logged_in_client.get('/abies/logout/')
        resp = logged_in_client.get('/abies/prelievi')
        assert resp.status_code == 302  # redirected to login
