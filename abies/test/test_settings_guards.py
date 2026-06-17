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
    'ABIES_IPSO_UPLOAD_TOKEN',
)


def load_settings(monkeypatch, **env):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
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
    assert settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT'] == 'common'


def test_debug_defaults_false(monkeypatch):
    with pytest.raises(RuntimeError, match='DJANGO_ALLOWED_HOSTS'):
        load_settings(monkeypatch, DJANGO_SECRET_KEY='prod-secret')


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


def test_production_requires_oauth_tenant_after_base_guards(monkeypatch):
    with pytest.raises(RuntimeError, match='MS_OAUTH_TENANT must be set'):
        load_settings(
            monkeypatch,
            DJANGO_DEBUG='False',
            DJANGO_SECRET_KEY='prod-secret',
            DJANGO_ALLOWED_HOSTS='abies.example.test',
        )


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


def test_production_requires_ipso_upload_token_after_base_guards(monkeypatch):
    with pytest.raises(RuntimeError, match='ABIES_IPSO_UPLOAD_TOKEN'):
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
        MS_OAUTH_TENANT=' contoso.onmicrosoft.com ',
        ABIES_IPSO_UPLOAD_TOKEN='prod-token',
    )

    assert settings.DEBUG is False
    assert settings.SECRET_KEY == 'prod-secret'
    assert settings.IPSO_UPLOAD_TOKEN == 'prod-token'
    assert settings.ALLOWED_HOSTS == ['abies.example.test', 'www.example.test']
    assert settings.SESSION_COOKIE_SECURE is True
    assert settings.CSRF_COOKIE_SECURE is True
    assert settings.AXES_ENABLED is True
    assert (
        settings.SOCIALACCOUNT_PROVIDERS['microsoft']['TENANT']
        == 'contoso.onmicrosoft.com'
    )
