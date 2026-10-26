"""Real image decoding, bounded input, and server-owned observation renditions."""

from io import BytesIO

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from test.image_helpers import image_bytes


def test_resize_and_reencode_without_losing_aspect_ratio():
    from apps.base.observation_images import prepare_photo
    prepared = prepare_photo(image_bytes((2400, 1200), fmt='PNG'))
    with Image.open(BytesIO(prepared.content)) as result:
        assert result.format == 'JPEG'
        assert result.size == (1024, 512)
    assert (prepared.width, prepared.height) == (1024, 512)
    assert len(prepared.content) < len(image_bytes((2400, 1200), fmt='PNG'))


def test_small_images_are_not_upscaled_and_orientation_is_applied():
    from apps.base.observation_images import prepare_photo
    exif = Image.Exif()
    exif[274] = 6
    prepared = prepare_photo(image_bytes((100, 50), exif=exif))
    with Image.open(BytesIO(prepared.content)) as result:
        assert result.size == (50, 100)
        assert result.getexif().get(274, 1) == 1


def test_metadata_survives_resizing_in_separate_fields(jpeg_with_exif_gps):
    from apps.base.observation_images import prepare_photo
    prepared = prepare_photo(jpeg_with_exif_gps(38.5, 16.25), extract_metadata=True)
    assert (prepared.lat, prepared.lon) == pytest.approx((38.5, 16.25))


def test_capture_time_retains_its_offset():
    from apps.base.observation_images import prepare_photo
    exif = Image.Exif()
    exif[36867] = '2026:10:26 12:34:56'
    exif[36881] = '+02:00'
    prepared = prepare_photo(image_bytes(exif=exif), extract_metadata=True)
    assert prepared.taken_at.isoformat() == '2026-10-26T12:34:56+02:00'


def test_heic_input_remains_supported():
    from apps.base.observation_images import prepare_photo
    prepared = prepare_photo(image_bytes((1200, 600), fmt='HEIF'))
    with Image.open(BytesIO(prepared.content)) as result:
        assert result.format == 'JPEG'
        assert result.size == (1024, 512)


@pytest.mark.parametrize('content', [b'not an image', b'\xff\xd8\xfffake', b'GIF89afake'])
def test_signatures_alone_are_not_accepted(content):
    from apps.base.observation_images import PhotoError, prepare_photo
    with pytest.raises(PhotoError):
        prepare_photo(content)


def test_pixel_limit_is_checked_before_decoding(settings, monkeypatch):
    from apps.base.observation_images import PhotoError, prepare_photo
    content = image_bytes((100, 100))
    settings.OBSERVATION_PHOTO_MAX_PIXELS = 9999
    monkeypatch.setattr(Image.Image, 'load', lambda *_args: pytest.fail('decoded oversized image'))
    with pytest.raises(PhotoError) as error:
        prepare_photo(content)
    assert error.value.status_code == 413


def test_chunked_read_stops_at_limit_even_if_reported_size_is_wrong(settings):
    from apps.base.observation_images import PhotoError, UploadBudget, read_photo
    settings.OBSERVATION_PHOTO_MAX_BYTES = 4

    class Upload:
        size = 1

        def chunks(self):
            yield b'12'
            yield b'345'
            pytest.fail('continued consuming oversized file')

    with pytest.raises(PhotoError) as error:
        read_photo(Upload(), UploadBudget())
    assert error.value.status_code == 413


def test_request_budget_counts_bytes_across_files(settings):
    from apps.base.observation_images import PhotoError, UploadBudget, read_photo
    settings.OBSERVATION_UPLOAD_MAX_BYTES = 5
    budget = UploadBudget()
    assert read_photo(SimpleUploadedFile('a', b'123'), budget) == b'123'
    with pytest.raises(PhotoError) as error:
        read_photo(SimpleUploadedFile('b', b'456'), budget)
    assert error.value.status_code == 413


def test_photo_count_is_bounded_before_reading_another_file(settings):
    from apps.base.observation_images import PhotoError, UploadBudget, read_photo
    settings.OBSERVATION_UPLOAD_MAX_PHOTOS = 1
    budget = UploadBudget()
    read_photo(SimpleUploadedFile('a', b'1'), budget)
    with pytest.raises(PhotoError):
        read_photo(SimpleUploadedFile('b', b'2'), budget)
