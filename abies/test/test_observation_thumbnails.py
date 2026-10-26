"""Protected small previews for current and already-stored observation photos."""

import hashlib
from io import BytesIO

import pytest
from django.test import Client
from PIL import Image

from apps.base.models import Observation, ObservationPhoto
from test.image_helpers import image_bytes


@pytest.fixture
def photo(reader_user, regions, settings, tmp_path):
    settings.OBSERVATION_MEDIA_DIR = tmp_path / 'photos'
    settings.OBSERVATION_THUMBNAIL_DIR = tmp_path / 'thumbnails'
    observation = Observation.objects.create(
        date='2026-10-26', text='Existing photo', region=regions[0], lat=38.5, lon=16.2,
    )
    content = image_bytes((2400, 1600))
    path = settings.OBSERVATION_MEDIA_DIR / '1' / 'legacy.jpg'
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    photo = ObservationPhoto.objects.create(
        observation=observation, file_path='1/legacy.jpg', content_type='image/jpeg',
        checksum=hashlib.sha256(content).hexdigest(), size_bytes=len(content),
        width_px=2400, height_px=1600, original_filename='legacy.jpg',
    )
    client = Client()
    client.force_login(reader_user)
    return photo, client


def thumbnail_url(photo):
    return f'/api/bosco/observations/photos/{photo.pk}/thumbnail/'


def test_detail_advertises_separate_thumbnail_and_full_image(photo):
    photo, client = photo
    response = client.get(f'/api/bosco/observations/{photo.observation_id}/detail/')
    metadata = response.json()['photos'][0]
    assert metadata['thumbnail_url'] == thumbnail_url(photo)
    assert metadata['url'] != metadata['thumbnail_url']


def test_existing_photo_gets_a_small_cached_preview_without_rewriting_it(photo, settings, monkeypatch):
    photo, client = photo
    original = photo.absolute_path().read_bytes()
    settings.OBSERVATION_PHOTO_MAX_BYTES = 1  # Upload limits do not rewrite/block stored media.
    response = client.get(thumbnail_url(photo))
    assert response.status_code == 200
    assert response['Content-Type'] == 'image/jpeg'
    assert response['Cache-Control'] == 'no-store'
    content = b''.join(response.streaming_content)
    with Image.open(BytesIO(content)) as preview:
        assert preview.size == (320, 213)
    assert len(content) < len(original) / 4
    assert photo.absolute_path().read_bytes() == original
    assert list(settings.OBSERVATION_THUMBNAIL_DIR.rglob('*.jpg'))

    def unexpected_decode(*_args, **_kwargs):
        pytest.fail('cached thumbnail was decoded again')

    monkeypatch.setattr('apps.base.observation_thumbnails.jpeg_rendition', unexpected_decode)
    again = client.get(thumbnail_url(photo), HTTP_IF_MODIFIED_SINCE=response['Last-Modified'])
    assert again.status_code == 304


def test_thumbnail_requires_login_before_generation(photo, settings):
    photo, _ = photo
    response = Client().get(thumbnail_url(photo))
    assert response.status_code == 302
    assert '/login/' in response.url
    assert not settings.OBSERVATION_THUMBNAIL_DIR.exists()


@pytest.mark.parametrize('failure', ['missing', 'invalid', 'pixels'])
def test_bad_legacy_sources_do_not_raise_500(photo, settings, failure):
    photo, client = photo
    if failure == 'missing':
        photo.absolute_path().unlink()
    elif failure == 'invalid':
        photo.absolute_path().write_bytes(b'not an image')
    else:
        settings.OBSERVATION_PHOTO_MAX_PIXELS = 1
    assert client.get(thumbnail_url(photo)).status_code == 404


def test_edit_form_uses_thumbnail_but_keeps_full_photo_link(photo, writer_user):
    photo, client = photo
    client.force_login(writer_user)
    html = client.get(f'/api/bosco/observations/form/{photo.observation_id}/').json()['html']
    assert f'src="{thumbnail_url(photo)}"' in html
    assert f'href="/api/bosco/observations/photos/{photo.pk}/"' in html
