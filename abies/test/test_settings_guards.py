"""Tests for production-safety guards in config.settings."""

import importlib.util
from pathlib import Path

import pytest

SETTINGS_PATH = Path(__file__).resolve().parents[1] / 'config' / 'settings.py'
DEFAULT_SECRET_KEY = 'django-insecure-change-me-before-deployment'
ENV_KEYS = (
    'DJANGO_DEBUG',
    'DJANGO_SECRET_KEY',
    'DJANGO_ALLOWED_HOSTS',
    'DJANGO_CSRF_TRUSTED_ORIGINS',
    'MS_OAUTH_CLIENT_ID',
    'MS_OAUTH_SECRET',
    'MS_OAUTH_TENANT',
    'DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE',
    'ABIES_IPSO_SECRET',
    'ABIES_INSTANCE',
    'ABIES_APP_NAME',
    'ABIES_BRAND_NAME',
    'ABIES_SITE_TITLE',
    'AXES_IPWARE_PROXY_COUNT',
    'AXES_IPWARE_PROXY_ORDER',
    'AXES_IPWARE_PROXY_TRUSTED_IPS',
    'AXES_IPWARE_META_PRECEDENCE_ORDER',
)
BRAND_ENV = {
    'ABIES_APP_NAME': 'Abies',
    'ABIES_BRAND_NAME': 'Test',
    'ABIES_SITE_TITLE': 'Abies test',
}


def load_settings(monkeypatch, *, brand=True, **env):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    if brand:
        for key, value in BRAND_ENV.items():
            monkeypatch.setenv(key, value)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    spec = importlib.util.spec_from_file_location(
        'config_settings_under_test', SETTINGS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('debug_value', ['1', 'true', 'True', 'yes', 'on'])
def test_debug_true_allows_local_defaults(monkeypatch, debug_value):
    settings = load_settings(monkeypatch, DJANGO_DEBUG=debug_value)

    assert settings.DEBUG is True
    assert settings.SECRET_KEY == DEFAULT_SECRET_KEY
    assert settings.ALLOWED_HOSTS == ['*']
    assert settings.SOCIALACCOUNT_PROVIDERS['microsoft'] == {}


def test_debug_defaults_false(monkeypatch):
    with pytest.raises(RuntimeError, match='DJANGO_ALLOWED_HOSTS'):
        load_settings(monkeypatch, DJANGO_SECRET_KEY='prod-secret')


@pytest.mark.parametrize('missing', [
    'ABIES_APP_NAME', 'ABIES_BRAND_NAME', 'ABIES_SITE_TITLE',
])
def test_branding_text_env_is_required(monkeypatch, missing):
    env = BRAND_ENV.copy()
    del env[missing]
    with pytest.raises(RuntimeError, match=missing):
        load_settings(monkeypatch, brand=False, DJANGO_DEBUG='1', **env)


def test_branding_text_and_asset_paths(monkeypatch):
    settings = load_settings(
        monkeypatch,
        DJANGO_DEBUG='1',
        ABIES_INSTANCE=' Dev ',
        ABIES_APP_NAME='Trees',
        ABIES_BRAND_NAME='Crew',
        ABIES_SITE_TITLE='Trees for Crew',
    )
    assert settings.ABIES_INSTANCE == 'dev'
    assert settings.ABIES_APP_NAME == 'Trees'
    assert settings.ABIES_BRAND_NAME == 'Crew'
    assert settings.ABIES_SITE_TITLE == 'Trees for Crew'
    assert settings.ABIES_BRAND_LOGO_STATIC == 'base/img/brand-logo.png'
    assert settings.ABIES_BRAND_FAVICON_STATIC == 'base/img/brand-favicon.gif'


def test_data_upload_memory_size_default_and_override(monkeypatch):
    settings = load_settings(monkeypatch, DJANGO_DEBUG='1')
    assert settings.DATA_UPLOAD_MAX_MEMORY_SIZE == 16 * 1024 * 1024

    settings = load_settings(
        monkeypatch,
        DJANGO_DEBUG='1',
        DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE='33554432',
    )
    assert settings.DATA_UPLOAD_MAX_MEMORY_SIZE == 32 * 1024 * 1024


@pytest.mark.parametrize('env', [{}, {'DJANGO_DEBUG': '0'}, {'DJANGO_DEBUG': 'False'}])
def test_production_requires_secret_key(monkeypatch, env):
    with pytest.raises(RuntimeError, match='DJANGO_SECRET_KEY must be set'):
        load_settings(monkeypatch, **env)


@pytest.mark.parametrize('secret', [DEFAULT_SECRET_KEY, f' {DEFAULT_SECRET_KEY} '])
def test_production_rejects_development_secret(monkeypatch, secret):
    with pytest.raises(RuntimeError, match='development default'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='0',
            DJANGO_SECRET_KEY=secret,
            DJANGO_ALLOWED_HOSTS='abies.example.test',
        )


def test_production_requires_allowed_hosts_after_secret_key(monkeypatch):
    with pytest.raises(RuntimeError, match='DJANGO_ALLOWED_HOSTS'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='False',
            DJANGO_SECRET_KEY='prod-secret',
        )


def test_production_allows_socialapp_only_oauth_config(monkeypatch):
    settings = load_settings(
        monkeypatch,
        DJANGO_DEBUG='False',
        DJANGO_SECRET_KEY='prod-secret',
        DJANGO_ALLOWED_HOSTS='abies.example.test',
        ABIES_IPSO_SECRET='prod-token',
    )

    assert settings.SOCIALACCOUNT_PROVIDERS['microsoft'] == {}


@pytest.mark.parametrize('tenant', ['common', ' organizations ', 'Consumers'])
def test_production_rejects_broad_oauth_tenants(monkeypatch, tenant):
    with pytest.raises(RuntimeError, match='single Entra tenant'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='False',
            DJANGO_SECRET_KEY='prod-secret',
            DJANGO_ALLOWED_HOSTS='abies.example.test',
            MS_OAUTH_TENANT=tenant,
        )


@pytest.mark.parametrize('env', [
    {'MS_OAUTH_CLIENT_ID': 'client-id'},
    {'MS_OAUTH_SECRET': 'client-secret'},
])
def test_rejects_partial_oauth_env_config(monkeypatch, env):
    with pytest.raises(RuntimeError, match='set together'):
        load_settings(monkeypatch, DJANGO_DEBUG='1', **env)


def test_debug_omits_empty_oauth_app(monkeypatch):
    settings = load_settings(monkeypatch, DJANGO_DEBUG='1')
    assert 'APP' not in settings.SOCIALACCOUNT_PROVIDERS['microsoft']


def test_debug_requires_tenant_when_oauth_env_credentials_are_set(monkeypatch):
    with pytest.raises(RuntimeError, match='MS_OAUTH_TENANT'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='1',
            MS_OAUTH_CLIENT_ID='client-id',
            MS_OAUTH_SECRET='client-secret',
        )


def test_production_requires_tenant_when_oauth_env_credentials_are_set(monkeypatch):
    with pytest.raises(RuntimeError, match='MS_OAUTH_TENANT'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='False',
            DJANGO_SECRET_KEY='prod-secret',
            DJANGO_ALLOWED_HOSTS='abies.example.test',
            MS_OAUTH_CLIENT_ID='client-id',
            MS_OAUTH_SECRET='client-secret',
            ABIES_IPSO_SECRET='prod-token',
        )


def test_debug_rejects_common_tenant_when_oauth_env_credentials_are_set(monkeypatch):
    with pytest.raises(RuntimeError, match='single Entra tenant'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='1',
            MS_OAUTH_CLIENT_ID='client-id',
            MS_OAUTH_SECRET='client-secret',
            MS_OAUTH_TENANT='common',
        )


def test_debug_accepts_tenant_specific_oauth_env_credentials(monkeypatch):
    settings = load_settings(
        monkeypatch,
        DJANGO_DEBUG='1',
        MS_OAUTH_CLIENT_ID=' client-id ',
        MS_OAUTH_SECRET=' client-secret ',
        MS_OAUTH_TENANT=' contoso.onmicrosoft.com ',
    )

    assert (
        settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT']
        == 'contoso.onmicrosoft.com'
    )
    assert (
        settings.SOCIALACCOUNT_PROVIDERS['microsoft']['APP']['client_id']
        == 'client-id'
    )


def test_production_requires_ipso_secret_after_oauth_guards(monkeypatch):
    with pytest.raises(RuntimeError, match='ABIES_IPSO_SECRET'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='False',
            DJANGO_SECRET_KEY='prod-secret',
            DJANGO_ALLOWED_HOSTS='abies.example.test',
            MS_OAUTH_TENANT='contoso.onmicrosoft.com',
        )


def test_production_loads_with_secret_key_allowed_hosts_and_oauth_tenant(monkeypatch):
    settings = load_settings(
        monkeypatch,
        DJANGO_DEBUG='False',
        DJANGO_SECRET_KEY='prod-secret',
        DJANGO_ALLOWED_HOSTS='abies.example.test, www.example.test',
        MS_OAUTH_CLIENT_ID=' client-id ',
        MS_OAUTH_SECRET=' client-secret ',
        MS_OAUTH_TENANT=' contoso.onmicrosoft.com ',
        ABIES_IPSO_SECRET='prod-token',
    )

    assert settings.DEBUG is False
    assert settings.SECRET_KEY == 'prod-secret'
    assert settings.IPSO_SECRET == 'prod-token'
    assert settings.ALLOWED_HOSTS == ['abies.example.test', 'www.example.test']
    assert settings.SESSION_COOKIE_SECURE is True
    assert settings.CSRF_COOKIE_SECURE is True
    assert settings.AXES_ENABLED is True
    assert (
        settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT']
        == 'contoso.onmicrosoft.com'
    )
    assert (
        settings.SOCIALACCOUNT_PROVIDERS['microsoft']['APP']['client_id']
        == 'client-id'
    )
    assert (
        settings.SOCIALACCOUNT_PROVIDERS['microsoft']['APP']['secret']
        == 'client-secret'
    )


def test_axes_proxy_settings_are_env_configurable(monkeypatch):
    settings = load_settings(
        monkeypatch,
        DJANGO_DEBUG='False',
        DJANGO_SECRET_KEY='prod-secret',
        DJANGO_ALLOWED_HOSTS='abies.example.test',
        ABIES_IPSO_SECRET='prod-token',
        AXES_IPWARE_PROXY_COUNT='1',
        AXES_IPWARE_PROXY_ORDER='right-most',
        AXES_IPWARE_PROXY_TRUSTED_IPS='127.0.0.1, 10.0.0.0/8',
        AXES_IPWARE_META_PRECEDENCE_ORDER='HTTP_X_FORWARDED_FOR, REMOTE_ADDR',
    )

    assert settings.AXES_IPWARE_PROXY_COUNT == 1
    assert settings.AXES_IPWARE_PROXY_ORDER == 'right-most'
    assert settings.AXES_IPWARE_PROXY_TRUSTED_IPS == ('127.0.0.1', '10.0.0.0/8')
    assert settings.AXES_IPWARE_META_PRECEDENCE_ORDER == (
        'HTTP_X_FORWARDED_FOR', 'REMOTE_ADDR',
    )
