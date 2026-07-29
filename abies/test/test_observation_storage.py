from pathlib import Path

import pytest

from apps.base.observation_storage import (
    normalize_observation_photo_content_type, observation_photo_absolute_path,
    observation_photo_relative_path, sniff_observation_photo_content_type,
)


def test_observation_photo_relative_path_ignores_user_filename_path():
    checksum = 'a' * 64

    path = observation_photo_relative_path(12, checksum, original_filename='../../evil.jpg')

    assert path == f'12/{checksum}.jpg'
    assert '..' not in Path(path).parts


def test_observation_photo_relative_path_prefers_content_type_extension():
    checksum = 'b' * 64

    path = observation_photo_relative_path(12, checksum, original_filename='photo.png', content_type='image/jpeg')

    assert path == f'12/{checksum}.jpg'


def test_observation_photo_relative_path_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        observation_photo_relative_path(0, 'a' * 64)
    with pytest.raises(ValueError):
        observation_photo_relative_path(1, 'not-a-checksum')


def test_observation_photo_absolute_path_resolves_under_media_root(settings, tmp_path):
    settings.OBSERVATION_MEDIA_DIR = tmp_path / 'observation-media'

    path = observation_photo_absolute_path('12/photo.jpg')

    assert path == tmp_path / 'observation-media' / '12' / 'photo.jpg'


def test_observation_photo_absolute_path_rejects_unsafe_relative_paths():
    for path in ('/tmp/photo.jpg', '../photo.jpg', '12/../../photo.jpg'):
        with pytest.raises(ValueError):
            observation_photo_absolute_path(path)


def test_normalize_observation_photo_content_type_accepts_matching_rasters():
    assert normalize_observation_photo_content_type(
        'image/jpeg', b'\xff\xd8\xffjpeg',
    ) == 'image/jpeg'
    assert normalize_observation_photo_content_type(
        'image/png; charset=binary', b'\x89PNG\r\n\x1a\nrest',
    ) == 'image/png'
    assert normalize_observation_photo_content_type(
        'image/webp', b'RIFF1234WEBPrest',
    ) == 'image/webp'
    assert normalize_observation_photo_content_type(
        'image/gif', b'GIF89arest',
    ) == 'image/gif'
    assert normalize_observation_photo_content_type(
        'image/heic', b'\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heic',
    ) == 'image/heic'


def test_normalize_observation_photo_content_type_rejects_active_content():
    assert normalize_observation_photo_content_type(
        'text/html', b'<script>alert(1)</script>',
    ) is None
    assert normalize_observation_photo_content_type(
        'image/svg+xml', b'<svg></svg>',
    ) is None
    assert normalize_observation_photo_content_type(
        'image/jpeg', b'<html></html>',
    ) is None


def test_sniff_observation_photo_content_type_infers_supported_raster():
    assert sniff_observation_photo_content_type(b'\xff\xd8\xffjpeg') == 'image/jpeg'
    assert sniff_observation_photo_content_type(b'<html></html>') is None
