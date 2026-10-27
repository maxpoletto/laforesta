"""Microsoft authentication uses only the tenant and credentials from settings."""

from urllib.parse import parse_qs, urlparse

import pytest
from allauth.socialaccount.adapter import get_adapter
from allauth.socialaccount.models import SocialApp
from django.contrib.auth import SESSION_KEY
from django.contrib.sites.models import Site
from django.core.checks import Tags, run_checks
from django.test import RequestFactory
from django.urls import reverse


pytestmark = pytest.mark.django_db


@pytest.fixture
def microsoft_settings(settings):
    settings.SOCIALACCOUNT_PROVIDERS = {'microsoft': {
        'APP': {'client_id': 'env-client', 'secret': 'env-secret'},
        'TENANT': 'company.onmicrosoft.com',
    }}
    return settings


def legacy_database_app():
    app = SocialApp.objects.create(
        provider='microsoft', name='Old database configuration',
        client_id='database-client', secret='database-secret',
        settings={'tenant': 'common'},
    )
    app.sites.add(Site.objects.get_current())
    return app


def assert_denied(client, response):
    assert response.status_code == 302
    assert response.url == reverse('login')
    assert SESSION_KEY not in client.session
    page = client.get(response.url)
    assert page.status_code == 200
    assert 'Accesso Microsoft non consentito' in page.content.decode()


def test_app_discovery_never_queries_database(microsoft_settings, django_assert_num_queries):
    request = RequestFactory().get(reverse('microsoft_login'))
    with django_assert_num_queries(0):
        apps = get_adapter(request).list_apps(request, provider='microsoft')
    assert len(apps) == 1
    assert apps[0].pk is None
    assert apps[0].client_id == 'env-client'
    assert apps[0].settings['tenant'] == 'company.onmicrosoft.com'


def test_database_app_cannot_compete_with_environment_app(client, microsoft_settings):
    legacy_database_app()
    response = client.post(reverse('microsoft_login'))
    assert response.status_code == 302
    target = urlparse(response.url)
    assert target.netloc == 'login.microsoftonline.com'
    assert target.path == '/company.onmicrosoft.com/oauth2/v2.0/authorize'
    assert parse_qs(target.query)['client_id'] == ['env-client']


def test_database_app_cannot_enable_disabled_oauth(client, settings):
    legacy_database_app()
    settings.SOCIALACCOUNT_PROVIDERS = {'microsoft': {}}
    assert_denied(client, client.post(reverse('microsoft_login')))


def test_unconfigured_oauth_route_does_not_raise_server_error(client, settings):
    settings.SOCIALACCOUNT_PROVIDERS = {'microsoft': {}}
    assert_denied(client, client.post(reverse('microsoft_login')))


@pytest.mark.parametrize('tenant', ['', ' ', 'common', ' ORGANIZATIONS ', 'consumers'])
def test_direct_login_rejects_invalid_tenant(client, microsoft_settings, tenant):
    microsoft_settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT'] = tenant
    assert_denied(client, client.post(reverse('microsoft_login')))


def test_callback_revalidates_configuration_before_token_exchange(
        client, microsoft_settings, monkeypatch):
    start = client.post(reverse('microsoft_login'))
    state = parse_qs(urlparse(start.url).query)['state'][0]
    microsoft_settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT'] = 'common'

    def unexpected_http(*args, **kwargs):
        pytest.fail('Invalid OAuth configuration reached an external HTTP endpoint')

    monkeypatch.setattr('requests.Session.request', unexpected_http)
    response = client.get(reverse('microsoft_callback'), {'state': state, 'code': 'code'})
    assert_denied(client, response)


@pytest.mark.parametrize('tenant', ['', 'common', 'organizations', 'consumers'])
def test_system_check_reports_invalid_configured_tenant(microsoft_settings, tenant):
    microsoft_settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT'] = tenant
    errors = run_checks(tags=[Tags.security])
    assert any(error.id == 'base.E001' for error in errors)


def test_system_check_allows_password_only_install(settings):
    settings.SOCIALACCOUNT_PROVIDERS = {'microsoft': {}}
    assert not [error for error in run_checks(tags=[Tags.security])
                if error.id.startswith('base.')]
