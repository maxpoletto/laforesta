"""Copernicus credentials belong to the instance and are never returned."""
import pytest

from apps.base.models import SatelliteCredentials
from apps.bosco.satellite_credentials import load_credentials


URL = '/api/impostazioni/satellite-credentials/'


def post(client, **body):
    return client.post(URL, body, content_type='application/json')


@pytest.mark.parametrize('role', ['reader_user', 'writer_user'])
def test_credentials_require_admin(client, request, role):
    client.force_login(request.getfixturevalue(role))
    assert client.get(URL).status_code == 403
    assert post(client, action='save', version=0, client_id='id', client_secret='secret').status_code == 403


def test_save_rotate_and_test_credentials(client, admin_user, settings, monkeypatch):
    client.force_login(admin_user)
    assert client.get(URL).json() == {'client_id': '', 'configured': False, 'version': 0}
    saved = post(client, action='save', version=0, client_id='local-id', client_secret='original-secret')
    assert saved.status_code == 200 and b'original-secret' not in saved.content
    row = SatelliteCredentials.objects.get()
    assert row.encrypted_secret != 'original-secret'
    assert 'encrypted_secret' not in {field.name for field in row.history.model._meta.fields}
    assert load_credentials() == ('local-id', 'original-secret')
    assert 'no-store' in client.get(URL)['Cache-Control']
    assert post(client, action='save', version=row.version, client_id='changed', client_secret='').status_code == 400
    assert post(client, action='remove', version=0).status_code == 409

    def token(credentials):
        assert credentials == ('local-id', 'original-secret')
        return 'access-token'

    monkeypatch.setattr('apps.bosco.satellite.get_access_token', token)
    result = post(client, action='test', version=row.version, client_id='local-id', client_secret='')
    assert result.status_code == 200 and b'access-token' not in result.content
    settings.SECRET_KEY = 'different-installation'
    assert load_credentials() is None
    client.force_login(admin_user)
    assert client.get(URL).json()['configured'] is False
    assert post(client, action='remove', version=row.version).status_code == 200
    assert client.get(URL).json()['client_id'] == ''
