"""The open-tab version check shares the release identifier with the shell."""

from django.test import override_settings
from django.urls import reverse


@override_settings(ABIES_VERSION='2.0.2')
def test_signed_in_reader_can_check_the_current_version(client, reader_user):
    client.force_login(reader_user)
    url = reverse('app-version')

    shell = client.get('/impostazioni')
    assert b'data-app-version="2.0.2"' in shell.content
    assert f'data-app-version-url="{url}"'.encode() in shell.content
    assert 'no-store' in shell['Cache-Control']

    with override_settings(ABIES_VERSION='2.0.3'):
        response = client.get(url)
    assert response.status_code == 200
    assert response.json() == {'version': '2.0.3'}
    assert 'no-store' in response['Cache-Control']


def test_version_check_requires_login(client):
    response = client.get(reverse('app-version'))
    assert response.status_code == 302
    assert response.url.startswith('/login/')
