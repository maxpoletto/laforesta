"""Image limits must reject before persisting any observation or retry receipt."""

from io import BytesIO

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from PIL import Image

from apps.base.models import Observation, ObservationCategory, ObservationPhoto, UsedNonce
from test.image_helpers import image_bytes


@pytest.fixture
def observation_upload(writer_user, regions, settings, tmp_path):
    settings.OBSERVATION_MEDIA_DIR = tmp_path / 'photos'
    category = ObservationCategory.objects.create(name='Photo limit test')
    client = Client()
    client.force_login(writer_user)

    def save(*contents):
        return client.post('/api/bosco/observations/save/', {
            'region_id': regions[0].pk, 'date': '2026-10-26', 'text': 'Keep this note',
            'category_ids': [category.pk], 'lat': '38.5', 'lon': '16.2',
            'nonce': 'photo-test', 'photos': [
                SimpleUploadedFile(f'photo-{i}.jpg', content, 'image/jpeg')
                for i, content in enumerate(contents)
            ],
        })

    return save


@pytest.mark.parametrize('limit', ['file', 'request', 'pixels', 'count'])
def test_upload_limits_leave_no_rows_files_or_receipt(observation_upload, settings, limit):
    content = image_bytes()
    contents = [content]
    if limit == 'file':
        settings.OBSERVATION_PHOTO_MAX_BYTES = len(content) - 1
    elif limit == 'request':
        settings.OBSERVATION_UPLOAD_MAX_BYTES = len(content) * 2 - 1
        contents *= 2
    elif limit == 'pixels':
        settings.OBSERVATION_PHOTO_MAX_PIXELS = 799
    else:
        settings.OBSERVATION_MAX_PHOTOS = 1
        contents.append(image_bytes(color='blue'))

    response = observation_upload(*contents)

    assert response.status_code == 413, response.content[:200]
    assert response.json()['status'] == 'validation_error'
    assert response.json()['message']
    assert not response.json().get('html')  # Keep selected files/form state for retry.
    assert not Observation.objects.exists()
    assert not ObservationPhoto.objects.exists()
    assert not UsedNonce.objects.exists()
    assert not list(settings.OBSERVATION_MEDIA_DIR.rglob('*'))


def test_invalid_image_is_a_validation_error(observation_upload):
    response = observation_upload(b'\xff\xd8\xffnot-a-jpeg')

    assert response.status_code == 400
    assert response.json()['message']
    assert not Observation.objects.exists()


def test_web_upload_normalizes_real_bytes_and_metadata(observation_upload, settings):
    response = observation_upload(image_bytes((2048, 1024), fmt='PNG'))

    assert response.status_code == 200, response.content
    photo = ObservationPhoto.objects.get()
    content = photo.absolute_path().read_bytes()
    assert photo.content_type == 'image/jpeg'
    assert photo.size_bytes == len(content)
    assert (photo.width_px, photo.height_px) == (1024, 512)
    with Image.open(BytesIO(content)) as image:
        assert image.format == 'JPEG'
        assert image.size == (1024, 512)


def test_multipart_parser_stops_before_a_buffering_handler(settings, rf):
    from django.core.files.uploadhandler import FileUploadHandler
    from apps.base.uploads import ObservationUploadHandler
    settings.OBSERVATION_PHOTO_MAX_BYTES = 4
    request = rf.post('/api/bosco/observations/save/', {
        'photos': SimpleUploadedFile('large.jpg', b'123456'),
    })

    class BufferSpy(FileUploadHandler):
        def receive_data_chunk(self, raw_data, start):
            pytest.fail('oversized file reached a buffering handler')

        def file_complete(self, file_size):
            return None

    request.upload_handlers = [ObservationUploadHandler(request), BufferSpy(request)]
    assert not request.FILES
    assert request.photo_upload_error.status_code == 413
