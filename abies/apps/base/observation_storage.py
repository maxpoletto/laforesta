"""Filesystem helpers for observation photos."""

from __future__ import annotations

import re
from pathlib import Path

from django.conf import settings

_CHECKSUM_RE = re.compile(r'^[0-9a-f]{64}$')
_CONTENT_TYPE_EXTENSIONS = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'image/heic': '.heic',
    'image/heif': '.heif',
}
_FILENAME_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif',
}


def observation_photo_relative_path(
        observation_id: int, checksum: str, *, original_filename: str = '',
        content_type: str = '',
) -> str:
    """Return a safe relative storage path for an observation photo.

    The original filename only contributes a whitelisted extension fallback;
    directory components and basename are ignored. The stable filename is the
    server-computed checksum, so user-provided paths can never escape the
    observation media root.
    """
    if observation_id <= 0:
        raise ValueError('observation_id must be positive')
    normalized_checksum = checksum.strip().lower()
    if not _CHECKSUM_RE.match(normalized_checksum):
        raise ValueError('checksum must be a 64-character hex digest')
    extension = _photo_extension(original_filename, content_type)
    return f'{observation_id}/{normalized_checksum}{extension}'


def observation_photo_absolute_path(relative_path: str) -> Path:
    """Resolve a stored relative photo path under OBSERVATION_MEDIA_DIR."""
    rel = Path(relative_path)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('unsafe observation photo path')
    return Path(settings.OBSERVATION_MEDIA_DIR) / rel


def _photo_extension(original_filename: str, content_type: str) -> str:
    by_type = _CONTENT_TYPE_EXTENSIONS.get(content_type.lower().strip())
    if by_type:
        return by_type
    suffix = Path(original_filename).suffix.lower()
    if suffix in _FILENAME_EXTENSIONS:
        return suffix
    return '.bin'
