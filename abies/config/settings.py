"""Django settings for Abies."""

import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data lives outside the Django project tree in production (Docker mount).
# In dev, it sits at BASE_DIR / 'data'.
DATA_DIR = BASE_DIR / 'data'
DIGEST_DIR = DATA_DIR / 'digests'
GEO_DIR = DATA_DIR / 'geo'
_DEFAULT_SATELLITE_DIR = BASE_DIR.parent / 'bosco' / 'data' / 'satellite'
if not _DEFAULT_SATELLITE_DIR.exists():
    _DEFAULT_SATELLITE_DIR = DATA_DIR / 'satellite'
SATELLITE_DIR = Path(os.environ.get(
    'ABIES_SATELLITE_DIR', str(_DEFAULT_SATELLITE_DIR),
))
IPSO_INBOX_DIR = Path(os.environ.get(
    'ABIES_IPSO_INBOX_DIR', str(DATA_DIR / 'ipso-inbox'),
))
IPSO_UPLOAD_TOKEN = os.environ.get('ABIES_IPSO_UPLOAD_TOKEN', '').strip()
IPSO_UPLOAD_MAX_BYTES = int(os.environ.get(
    'ABIES_IPSO_UPLOAD_MAX_BYTES', str(2 * 1024 * 1024),
))
IPSO_UPLOAD_MAX_RECORDS = int(os.environ.get(
    'ABIES_IPSO_UPLOAD_MAX_RECORDS', '500',
))
IPSO_UPLOAD_RATE_LIMIT = int(os.environ.get(
    'ABIES_IPSO_UPLOAD_RATE_LIMIT', '60',
))
IPSO_UPLOAD_RATE_WINDOW_S = int(os.environ.get(
    'ABIES_IPSO_UPLOAD_RATE_WINDOW_S', '60',
))

_DEFAULT_SECRET_KEY = 'django-insecure-change-me-before-deployment'
_DEFAULT_MS_OAUTH_TENANT = 'common'
_BROAD_MS_OAUTH_TENANTS = frozenset({'common', 'organizations', 'consumers'})


def _env_bool(name, *, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


DEBUG = _env_bool('DJANGO_DEBUG', default=False)
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', '').strip()

if DEBUG and not SECRET_KEY:
    SECRET_KEY = _DEFAULT_SECRET_KEY
elif not SECRET_KEY:
    raise RuntimeError('DJANGO_SECRET_KEY must be set when DEBUG=0')
elif SECRET_KEY == _DEFAULT_SECRET_KEY:
    raise RuntimeError(
        'DJANGO_SECRET_KEY must not use the development default when DEBUG=0',
    )

# Comma-separated host list via env; '*' is only acceptable in dev.
_hosts = os.environ.get('DJANGO_ALLOWED_HOSTS', '').strip()
if _hosts:
    ALLOWED_HOSTS = [h.strip() for h in _hosts.split(',') if h.strip()]
elif DEBUG:
    ALLOWED_HOSTS = ['*']
else:
    raise RuntimeError('DJANGO_ALLOWED_HOSTS must be set when DEBUG=0')

# Comma-separated origins for CSRF (e.g. 'https://laforesta.it').
_csrf = os.environ.get('DJANGO_CSRF_TRUSTED_ORIGINS', '').strip()
CSRF_TRUSTED_ORIGINS = [o.strip() for o in _csrf.split(',') if o.strip()]

_ms_oauth_tenant = os.environ.get('MS_OAUTH_TENANT', '').strip()
if DEBUG and not _ms_oauth_tenant:
    _ms_oauth_tenant = _DEFAULT_MS_OAUTH_TENANT
elif not _ms_oauth_tenant:
    raise RuntimeError('MS_OAUTH_TENANT must be set when DEBUG=0')
elif not DEBUG and _ms_oauth_tenant.lower() in _BROAD_MS_OAUTH_TENANTS:
    raise RuntimeError(
        'MS_OAUTH_TENANT must pin a single Entra tenant when DEBUG=0',
    )

if not DEBUG and not IPSO_UPLOAD_TOKEN:
    raise RuntimeError('ABIES_IPSO_UPLOAD_TOKEN must be set when DEBUG=0')

# Apache terminates TLS and forwards plain HTTP; trust the X-Forwarded-Proto
# header so Django recognises the request as secure for cookie flags, etc.
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True

# Django's default Referrer-Policy is 'same-origin', which strips Referer
# entirely on cross-origin requests.  OpenStreetMap's tile usage policy
# requires a Referer header on tile requests (see osm.wiki/Blocked) — under
# 'same-origin' the browser sends nothing and OSM returns 403.  The modern
# default 'strict-origin-when-cross-origin' sends only the origin
# (e.g. 'https://abies.laforesta.it/') on cross-origin requests, which is
# enough to satisfy OSM without leaking the full URL or query string.
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# Production is HTTPS-only behind Apache. Keep these aligned with
# `manage.py check --deploy --fail-level WARNING` in bin/deploy.
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = True
    SECURE_HSTS_SECONDS = int(os.environ.get(
        'DJANGO_SECURE_HSTS_SECONDS', '31536000',
    ))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

# --- Apps -------------------------------------------------------------------

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    # Third-party
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.microsoft',
    'axes',
    'simple_history',
    # Project
    'apps.base',
    'apps.ipso',
    'apps.prelievi',
    'apps.bosco',
    'apps.campionamenti',
    'apps.controllo',
    'apps.impostazioni',
    'apps.piano_di_taglio',
    'apps.mannesi',
]

SITE_ID = 1

# --- Auth --------------------------------------------------------------------

AUTH_USER_MODEL = 'base.User'

AUTHENTICATION_BACKENDS = [
    'axes.backends.AxesStandaloneBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

SESSION_COOKIE_AGE = 43200  # 12 hours

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/prelievi'

# JSON CSV imports carry base64 file bytes in request bodies. Django's 2.5 MiB
# default is too tight once base64 overhead is included, so keep an explicit
# ceiling with a deployment override.
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get(
    'DJANGO_DATA_UPLOAD_MAX_MEMORY_SIZE', str(16 * 1024 * 1024),
))

# --- allauth -----------------------------------------------------------------

ACCOUNT_LOGIN_METHODS = {'username'}
ACCOUNT_SIGNUP_FIELDS = ['username*', 'password1*', 'password2*']
ACCOUNT_EMAIL_VERIFICATION = 'none'
ACCOUNT_LOGIN_ON_GET = True
# Disable allauth's own signup — users are created by admins.
ACCOUNT_ADAPTER = 'apps.base.auth.NoSignupAdapter'
# Match OAuth logins to admin-whitelisted users by email (Microsoft's Graph
# API does not flag emails as verified, so allauth's default match fails).
SOCIALACCOUNT_ADAPTER = 'apps.base.auth.WhitelistSocialAdapter'

SOCIALACCOUNT_AUTO_SIGNUP = False
SOCIALACCOUNT_EMAIL_AUTHENTICATION = True
SOCIALACCOUNT_EMAIL_AUTHENTICATION_AUTO_CONNECT = True
# MS 365 OAuth: configure client_id/secret via Django admin or env vars.
SOCIALACCOUNT_PROVIDERS = {
    'microsoft': {
        'APP': {
            'client_id': os.environ.get('MS_OAUTH_CLIENT_ID', ''),
            'secret': os.environ.get('MS_OAUTH_SECRET', ''),
        },
        'TENANT': _ms_oauth_tenant,
    },
}

# --- django-axes -------------------------------------------------------------

AXES_ENABLED = not DEBUG
AXES_FAILURE_LIMIT = 5
AXES_COOLOFF_TIME = timedelta(minutes=30)
AXES_LOCKOUT_PARAMETERS = ['username', 'ip_address']
AXES_RESET_ON_SUCCESS = True

# --- Middleware --------------------------------------------------------------

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'allauth.account.middleware.AccountMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'axes.middleware.AxesMiddleware',
    'simple_history.middleware.HistoryRequestMiddleware',
    'apps.base.middleware.CSPMiddleware',
    'apps.base.middleware.NonceMiddleware',
    'apps.base.middleware.RateLimitMiddleware',
]

# --- URLs / WSGI -------------------------------------------------------------

ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'

# --- Templates ---------------------------------------------------------------

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'apps.base.context_processors.strings',
            ],
        },
    },
]

# --- Database ----------------------------------------------------------------

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': DATA_DIR / 'db.sqlite3',
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- I18n / L10n -------------------------------------------------------------

LANGUAGE_CODE = 'it'
TIME_ZONE = 'Europe/Rome'
USE_I18N = True
USE_TZ = True

# --- Static files ------------------------------------------------------------

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
