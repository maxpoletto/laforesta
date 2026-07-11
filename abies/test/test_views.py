"""Tests for auth views: login, logout, shell access."""

import pytest
from allauth.socialaccount.models import SocialApp
from django.contrib.sites.models import Site
from django.test import Client, override_settings

from apps.base.models import LoginMethod, SiteSettings, User
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

    @override_settings(SOCIALACCOUNT_PROVIDERS={
        'microsoft': {'APP': {'client_id': '', 'secret': ''}, 'TENANT': 'common'},
    })
    def test_hides_microsoft_login_when_oauth_unconfigured(self, client):
        resp = client.get('/login/')
        assert resp.status_code == 200
        assert b'Accedi con Microsoft' not in resp.content

    @override_settings(SOCIALACCOUNT_PROVIDERS={
        'microsoft': {
            'APP': {'client_id': 'client-id', 'secret': 'client-secret'},
            'TENANT': 'contoso.onmicrosoft.com',
        },
    })
    def test_shows_microsoft_login_when_oauth_configured(self, client):
        resp = client.get('/login/')
        assert resp.status_code == 200
        assert b'Accedi con Microsoft' in resp.content

    @override_settings(SOCIALACCOUNT_PROVIDERS={
        'microsoft': {'TENANT': 'common'},
    })
    def test_hides_microsoft_login_when_only_common_tenant_is_configured(
            self, client):
        app = SocialApp.objects.create(
            provider='microsoft',
            name='Microsoft',
            client_id='client-id',
            secret='client-secret',
        )
        app.sites.add(Site.objects.get_current())

        resp = client.get('/login/')

        assert resp.status_code == 200
        assert b'Accedi con Microsoft' not in resp.content

    @override_settings(SOCIALACCOUNT_PROVIDERS={
        'microsoft': {'TENANT': 'common'},
    })
    def test_shows_microsoft_login_when_social_app_configured(self, client):
        app = SocialApp.objects.create(
            provider='microsoft',
            name='Microsoft',
            client_id='client-id',
            secret='client-secret',
            settings={'tenant': 'contoso.onmicrosoft.com'},
        )
        app.sites.add(Site.objects.get_current())

        resp = client.get('/login/')

        assert resp.status_code == 200
        assert b'Accedi con Microsoft' in resp.content

    def test_post_success_redirects(self, client, admin_user):
        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'testpass123!',
        })
        assert resp.status_code == 302
        assert resp.url == '/prelievi'

    def test_post_success_redirects_to_user_landing_page(self, client, admin_user):
        admin_user.landing_page = '/bosco?mode=evoluzione'
        admin_user.save(update_fields=['landing_page'])

        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'testpass123!',
        })

        assert resp.status_code == 302
        assert resp.url == '/bosco?mode=evoluzione'

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

    def test_post_rejects_external_next_to_user_landing_page(
            self, client, admin_user):
        admin_user.landing_page = '/campionamenti'
        admin_user.save(update_fields=['landing_page'])

        resp = client.post('/login/', {
            FIELD_USERNAME: 'testadmin', FIELD_PASSWORD: 'testpass123!',
            'next': 'https://evil.example/phish',
        })

        assert resp.status_code == 302
        assert resp.url == '/campionamenti'

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

    def test_root_redirects_to_effective_landing_page(self, logged_in_client):
        settings = SiteSettings.load()
        settings.default_landing_page = '/bosco'
        settings.save()

        resp = logged_in_client.get('/')

        assert resp.status_code == 302
        assert resp.url == '/bosco'

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

    def test_shell_shows_settings_tab_to_oauth_reader(self, reader_user):
        reader_user.login_method = LoginMethod.OAUTH
        reader_user.email = 'reader@example.com'
        reader_user.save(update_fields=['login_method', 'email'])
        client = Client()
        client.force_login(reader_user)

        resp = client.get('/prelievi')

        assert resp.status_code == 200
        assert 'href="/impostazioni"' in resp.content.decode()


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
    def test_logout_requires_post(self, logged_in_client):
        resp = logged_in_client.get('/logout/')
        assert resp.status_code == 405

    def test_logout_redirects_to_login(self, logged_in_client):
        resp = logged_in_client.post('/logout/')
        assert resp.status_code == 302

    def test_logout_clears_session(self, logged_in_client):
        logged_in_client.post('/logout/')
        resp = logged_in_client.get('/prelievi')
        assert resp.status_code == 302  # redirected to login
