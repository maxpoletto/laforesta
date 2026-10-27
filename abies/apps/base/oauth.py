"""The single supported Microsoft OAuth configuration, supplied by environment."""

from allauth.socialaccount.models import SocialApp
from django.conf import settings
from django.core.checks import Error, Tags, register


def microsoft_oauth_app() -> SocialApp | None:
    """Return allauth's unsaved app object, never consulting database apps."""
    provider = settings.SOCIALACCOUNT_PROVIDERS.get('microsoft', {})
    if 'APPS' in provider:
        return None
    app = provider.get('APP') or {}
    tenant = (provider.get('TENANT') or '').strip()
    if (not app.get('client_id') or not app.get('secret') or not tenant
            or tenant.lower() in settings.MS_OAUTH_BROAD_TENANTS):
        return None
    return SocialApp(
        provider='microsoft', client_id=app['client_id'], secret=app['secret'],
        settings={'tenant': tenant},
    )


@register(Tags.security)
def check_microsoft_oauth(app_configs, **kwargs):
    provider = settings.SOCIALACCOUNT_PROVIDERS.get('microsoft', {})
    if (provider.get('APP') and microsoft_oauth_app() is None) or 'APPS' in provider:
        return [Error(
            'Microsoft OAuth requires environment credentials and a single Entra tenant.',
            hint='Set MS_OAUTH_CLIENT_ID, MS_OAUTH_SECRET and MS_OAUTH_TENANT together.',
            id='base.E001',
        )]
    return []
