"""Filesystem helpers for observation photos."""

from __future__ import annotations

import os
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
_CONTENT_TYPE_ALIASES = {
    'image/jpg': 'image/jpeg',
}
_HEIF_BRANDS = {
    b'heic', b'heix', b'hevc', b'hevx', b'heim', b'heis',
    b'hevm', b'hevs', b'mif1', b'msf1',
}
_FILENAME_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic', '.heif',
}


def normalize_observation_photo_content_type(
        content_type: str, content: bytes | None = None,
) -> str | None:
    """Return a safe raster content type when the upload is acceptable."""
    normalized = _normalize_content_type_label(content_type)
    if normalized not in _CONTENT_TYPE_EXTENSIONS:
        return None
    if content is not None and not _content_matches_type(normalized, content):
        return None
    return normalized


def sniff_observation_photo_content_type(content: bytes) -> str | None:
    """Infer a supported raster content type from file signatures."""
    for content_type in _CONTENT_TYPE_EXTENSIONS:
        if _content_matches_type(content_type, content):
            return content_type
    return None


def _normalize_content_type_label(content_type: str) -> str:
    normalized = str(content_type or '').split(';', 1)[0].strip().lower()
    return _CONTENT_TYPE_ALIASES.get(normalized, normalized)


def _content_matches_type(content_type: str, content: bytes) -> bool:
    if content_type == 'image/jpeg':
        return content.startswith(b'\xff\xd8\xff')
    if content_type == 'image/png':
        return content.startswith(b'\x89PNG\r\n\x1a\n')
    if content_type == 'image/webp':
        return (
            len(content) >= 12
            and content.startswith(b'RIFF')
            and content[8:12] == b'WEBP'
        )
    if content_type == 'image/gif':
        return content.startswith((b'GIF87a', b'GIF89a'))
    if content_type in {'image/heic', 'image/heif'}:
        return _content_is_heif(content)
    return False


def _content_is_heif(content: bytes) -> bool:
    if len(content) < 12 or content[4:8] != b'ftyp':
        return False
    brands = {content[8:12]}
    compatible = content[16:40]
    brands.update(
        compatible[i:i + 4]
        for i in range(0, len(compatible) - 3, 4)
    )
    return bool(brands & _HEIF_BRANDS)


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


def atomic_write_observation_photo(path: Path, content: bytes) -> None:
    """Durably write an observation photo under its final storage path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def _photo_extension(original_filename: str, content_type: str) -> str:
    by_type = _CONTENT_TYPE_EXTENSIONS.get(content_type.lower().strip())
    if by_type:
        return by_type
    suffix = Path(original_filename).suffix.lower()
    if suffix in _FILENAME_EXTENSIONS:
        return suffix
    return '.bin'


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
