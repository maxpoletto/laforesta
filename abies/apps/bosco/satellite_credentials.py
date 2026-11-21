"""Encrypt the Copernicus secret with the installation's Django key."""
from base64 import urlsafe_b64encode

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from django.conf import settings
from django.utils.encoding import force_bytes
from django.views.decorators.debug import sensitive_variables

from apps.base.models import SatelliteCredentials


def _cipher():
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
               info=b'abies/copernicus-credentials/v1').derive(force_bytes(settings.SECRET_KEY))
    return Fernet(urlsafe_b64encode(key))


@sensitive_variables()
def encrypt_secret(secret):
    return _cipher().encrypt(secret.encode()).decode('ascii')


@sensitive_variables()
def decrypt_secret(row):
    if row and row.encrypted_secret:
        try:
            return _cipher().decrypt(row.encrypted_secret.encode('ascii')).decode()
        except (InvalidToken, UnicodeError):
            return None
    return None


@sensitive_variables()
def load_credentials():
    row = SatelliteCredentials.objects.filter(pk=1).first()
    secret = decrypt_secret(row)
    return (row.client_id, secret) if secret and row.client_id else None
