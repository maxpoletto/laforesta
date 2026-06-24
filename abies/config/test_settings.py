"""Test settings shim.

Production settings require deployment branding env. Pytest loads Django
settings before ``test/conftest.py``, so provide neutral test-only values
before importing the real settings module.
"""

import os

os.environ.setdefault('DJANGO_DEBUG', '1')
os.environ.setdefault('ABIES_APP_NAME', 'Abies')
os.environ.setdefault('ABIES_BRAND_NAME', 'Test')
os.environ.setdefault('ABIES_SITE_TITLE', 'Abies test')

from config.settings import *  # noqa: F401,F403
