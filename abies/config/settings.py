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

SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY', 'django-insecure-change-me-before-deployment',
)

DEBUG = os.environ.get('DJANGO_DEBUG', '1') == '1'

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

# Apache terminates TLS and forwards plain HTTP; trust the X-Forwarded-Proto
# header so Django recognises the request as secure for cookie flags, etc.
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True

if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

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
    'apps.prelievi',
    'apps.bosco',
    'apps.controllo',
    'apps.impostazioni',
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

LOGIN_URL = '/abies/login/'
LOGIN_REDIRECT_URL = '/abies/prelievi'

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
        'TENANT': os.environ.get('MS_OAUTH_TENANT', 'common'),
    },
}

# --- django-axes -------------------------------------------------------------

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

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
