"""Regenerable observation previews; primary photo bytes remain immutable."""

import hashlib
from pathlib import Path

from django.conf import settings

from apps.base.observation_images import decoded_photo, jpeg_rendition
from apps.base.observation_storage import atomic_write_observation_photo

THUMBNAIL_MAX_DIMENSION = 320  # Up to twice the 160px preview height.


def observation_thumbnail(photo):
    key = hashlib.sha256(f'{photo.file_path}\0{photo.checksum}'.encode()).hexdigest()
    path = Path(settings.OBSERVATION_THUMBNAIL_DIR) / 'v1-320' / f'{key}.jpg'
    if not path.is_file():
        with decoded_photo(photo.absolute_path()) as image:
            content, _, _ = jpeg_rendition(image, THUMBNAIL_MAX_DIMENSION)
        atomic_write_observation_photo(path, content)
    return path
