from pathlib import Path

import pytest

from apps.base.observation_storage import (
    observation_photo_absolute_path, observation_photo_relative_path,
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
