"""Filesystem helpers for observation photos."""

from __future__ import annotations

import os
import re
import struct
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
_EXIF_PREFIX = b'Exif\0\0'
_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1}
_TIFF_GPS_IFD_TAG = 0x8825
_GPS_LAT_REF_TAG = 1
_GPS_LAT_TAG = 2
_GPS_LON_REF_TAG = 3
_GPS_LON_TAG = 4


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


def extract_observation_photo_gps(content: bytes) -> tuple[float, float] | None:
    """Return JPEG EXIF GPS coordinates as ``(lat, lon)``, or ``None``.

    EXIF is user-supplied metadata, so parsing is deliberately conservative:
    malformed data, unsupported TIFF shapes, missing tags, and out-of-range
    coordinates are treated as absent metadata rather than upload errors.
    """
    payload = _jpeg_exif_payload(content)
    if payload is None:
        return None
    try:
        coords = _exif_gps_coordinates(payload)
    except (struct.error, ValueError, UnicodeDecodeError):
        return None
    if coords is None:
        return None
    lat, lon = coords
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _jpeg_exif_payload(content: bytes) -> bytes | None:
    if not content.startswith(b'\xff\xd8'):
        return None
    pos = 2
    end = len(content)
    while pos + 1 < end:
        if content[pos] != 0xff:
            return None
        while pos < end and content[pos] == 0xff:
            pos += 1
        if pos >= end:
            return None
        marker = content[pos]
        pos += 1
        if marker in {0xd9, 0xda}:
            return None
        if marker == 0x01 or 0xd0 <= marker <= 0xd7:
            continue
        if pos + 2 > end:
            return None
        segment_len = int.from_bytes(content[pos:pos + 2], 'big')
        if segment_len < 2:
            return None
        payload_start = pos + 2
        payload_end = pos + segment_len
        if payload_end > end:
            return None
        payload = content[payload_start:payload_end]
        if marker == 0xe1 and payload.startswith(_EXIF_PREFIX):
            return payload[len(_EXIF_PREFIX):]
        pos = payload_end
    return None


def _exif_gps_coordinates(tiff: bytes) -> tuple[float, float] | None:
    if len(tiff) < 8:
        return None
    byte_order = tiff[:2]
    if byte_order == b'II':
        endian = '<'
    elif byte_order == b'MM':
        endian = '>'
    else:
        return None
    if _tiff_u16(tiff, 2, endian) != 42:
        return None
    ifd0_offset = _tiff_u32(tiff, 4, endian)
    gps_offset = _tiff_long(tiff, ifd0_offset, _TIFF_GPS_IFD_TAG, endian)
    if gps_offset is None:
        return None

    lat_ref = _tiff_ascii(tiff, gps_offset, _GPS_LAT_REF_TAG, endian)
    lon_ref = _tiff_ascii(tiff, gps_offset, _GPS_LON_REF_TAG, endian)
    lat_dms = _tiff_rationals(tiff, gps_offset, _GPS_LAT_TAG, endian, 3)
    lon_dms = _tiff_rationals(tiff, gps_offset, _GPS_LON_TAG, endian, 3)
    if lat_ref not in {'N', 'S'} or lon_ref not in {'E', 'W'}:
        return None
    if lat_dms is None or lon_dms is None:
        return None

    lat = _dms_to_decimal(lat_dms)
    lon = _dms_to_decimal(lon_dms)
    if lat_ref == 'S':
        lat = -lat
    if lon_ref == 'W':
        lon = -lon
    return lat, lon


def _tiff_u16(data: bytes, offset: int, endian: str) -> int:
    return struct.unpack_from(endian + 'H', data, offset)[0]


def _tiff_u32(data: bytes, offset: int, endian: str) -> int:
    return struct.unpack_from(endian + 'I', data, offset)[0]


def _tiff_entry(data: bytes, ifd_offset: int, tag: int, endian: str):
    if ifd_offset < 0 or ifd_offset + 2 > len(data):
        raise ValueError('invalid IFD offset')
    count = _tiff_u16(data, ifd_offset, endian)
    entries_offset = ifd_offset + 2
    entries_end = entries_offset + count * 12
    if entries_end + 4 > len(data):
        raise ValueError('truncated IFD')
    for index in range(count):
        offset = entries_offset + index * 12
        if _tiff_u16(data, offset, endian) == tag:
            value_type = _tiff_u16(data, offset + 2, endian)
            value_count = _tiff_u32(data, offset + 4, endian)
            value = _tiff_value_bytes(
                data, offset + 8, value_type, value_count, endian,
            )
            return value_type, value_count, value
    return None


def _tiff_value_bytes(
        data: bytes, value_offset: int, value_type: int, count: int, endian: str,
) -> bytes:
    type_size = _TIFF_TYPE_SIZES.get(value_type)
    if type_size is None:
        raise ValueError('unsupported TIFF type')
    byte_count = type_size * count
    if byte_count <= 4:
        return data[value_offset:value_offset + byte_count]
    offset = _tiff_u32(data, value_offset, endian)
    if offset < 0 or offset + byte_count > len(data):
        raise ValueError('invalid TIFF value offset')
    return data[offset:offset + byte_count]


def _tiff_long(data: bytes, ifd_offset: int, tag: int, endian: str) -> int | None:
    entry = _tiff_entry(data, ifd_offset, tag, endian)
    if entry is None:
        return None
    value_type, count, value = entry
    if value_type != 4 or count != 1 or len(value) < 4:
        return None
    return struct.unpack_from(endian + 'I', value, 0)[0]


def _tiff_ascii(data: bytes, ifd_offset: int, tag: int, endian: str) -> str | None:
    entry = _tiff_entry(data, ifd_offset, tag, endian)
    if entry is None:
        return None
    value_type, _count, value = entry
    if value_type != 2:
        return None
    return value.split(b'\0', 1)[0].decode('ascii')


def _tiff_rationals(
        data: bytes, ifd_offset: int, tag: int, endian: str, expected: int,
) -> list[float] | None:
    entry = _tiff_entry(data, ifd_offset, tag, endian)
    if entry is None:
        return None
    value_type, count, value = entry
    if value_type != 5 or count != expected or len(value) < expected * 8:
        return None
    result = []
    for index in range(expected):
        numerator, denominator = struct.unpack_from(
            endian + 'II', value, index * 8,
        )
        if denominator == 0:
            raise ValueError('zero EXIF rational denominator')
        result.append(numerator / denominator)
    return result


def _dms_to_decimal(values: list[float]) -> float:
    degrees, minutes, seconds = values
    return degrees + minutes / 60 + seconds / 3600


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
