"""Landing-page validation and resolution."""

from urllib.parse import urlsplit, urlunsplit

from django.conf import settings

from apps.base.models import SiteSettings

APP_ROUTE_DOMAINS = frozenset({
    'bosco',
    'piano-di-taglio',
    'campionamenti',
    'squadre',
    'prelievi',
    'importazione',
    'controllo',
    'impostazioni',
})
MAX_LANDING_PAGE_LENGTH = 255


def clean_landing_page(value: str | None) -> str:
    """Return a normalized same-site SPA URL, or raise ValueError."""
    value = (value or '').strip()
    if not value:
        return ''
    if len(value) > MAX_LANDING_PAGE_LENGTH:
        raise ValueError
    if any(ord(ch) < 32 for ch in value) or '\\' in value:
        raise ValueError

    parts = urlsplit(value)
    if parts.scheme or parts.netloc or not parts.path.startswith('/'):
        raise ValueError

    domain = parts.path.strip('/').split('/', 1)[0]
    if domain not in APP_ROUTE_DOMAINS:
        raise ValueError

    cleaned = urlunsplit(('', '', parts.path, parts.query, parts.fragment))
    if len(cleaned) > MAX_LANDING_PAGE_LENGTH:
        raise ValueError
    return cleaned


def user_landing_page(user) -> str:
    """Resolve a user's effective landing page."""
    for candidate in (
        getattr(user, 'landing_page', ''),
        _site_default_landing_page(),
        settings.LOGIN_REDIRECT_URL,
    ):
        try:
            page = clean_landing_page(candidate)
        except ValueError:
            continue
        if page:
            return page
    return '/prelievi'


def _site_default_landing_page() -> str:
    obj = SiteSettings.objects.only('default_landing_page').filter(
        singleton_id=1,
    ).first()
    return obj.default_landing_page if obj else ''
