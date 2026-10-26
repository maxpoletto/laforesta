"""Shared bounds and image decoding for desktop and mobile observations."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import warnings

from django.conf import settings
from django.utils import timezone
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from apps.base.observation_storage import extract_observation_photo_gps
from config import strings as S

register_heif_opener()

PHOTO_MAX_DIMENSION = 1024
PHOTO_JPEG_QUALITY = 85


class PhotoError(Exception):
    def __init__(self, message, *, too_large=False):
        super().__init__(message)
        self.status_code = 413 if too_large else 400


def check_photo_count(count):
    if count > settings.OBSERVATION_MAX_PHOTOS:
        raise PhotoError(S.ERR_OBSERVATION_PHOTO_COUNT.format(settings.OBSERVATION_MAX_PHOTOS),
                         too_large=True)


class UploadBudget:
    """The same byte/count accounting in the HTTP handler and bounded reader."""

    def __init__(self):
        self.total = self.current = self.count = 0

    def start_file(self):
        self.current = 0
        self.count += 1
        if self.count > settings.OBSERVATION_UPLOAD_MAX_PHOTOS:
            raise PhotoError(S.ERR_OBSERVATION_PHOTO_COUNT.format(settings.OBSERVATION_UPLOAD_MAX_PHOTOS),
                             too_large=True)

    def add(self, length):
        self.current += length
        self.total += length
        if self.current > settings.OBSERVATION_PHOTO_MAX_BYTES:
            raise PhotoError(S.ERR_OBSERVATION_PHOTO_BYTES, too_large=True)
        if self.total > settings.OBSERVATION_UPLOAD_MAX_BYTES:
            raise PhotoError(S.ERR_OBSERVATION_UPLOAD_BYTES, too_large=True)


def read_photo(uploaded, budget):
    budget.start_file()
    content = BytesIO()
    for chunk in uploaded.chunks():
        budget.add(len(chunk))
        content.write(chunk)
    return content.getvalue()


@contextmanager
def decoded_photo(content):
    """Check dimensions before decoding; accept only fully decodable rasters."""
    if len(content) > settings.OBSERVATION_PHOTO_MAX_BYTES:
        raise PhotoError(S.ERR_OBSERVATION_PHOTO_BYTES, too_large=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in {'JPEG', 'PNG', 'WEBP', 'GIF', 'HEIF'}:
                    raise PhotoError(S.ERR_OBSERVATION_PHOTO_INVALID)
                if image.width * image.height > settings.OBSERVATION_PHOTO_MAX_PIXELS:
                    raise PhotoError(S.ERR_OBSERVATION_PHOTO_PIXELS, too_large=True)
                image.load()
                yield image
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise PhotoError(S.ERR_OBSERVATION_PHOTO_PIXELS, too_large=True) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise PhotoError(S.ERR_OBSERVATION_PHOTO_INVALID) from exc


def inspect_photo(content):
    with decoded_photo(content) as image:
        return image.size


@dataclass(frozen=True)
class PreparedPhoto:
    content: bytes
    width: int
    height: int
    lat: float | None = None
    lon: float | None = None
    taken_at: datetime | None = None


def _taken_at(image):
    try:
        exif = image.getexif()
        camera = exif.get_ifd(34665) if 34665 in exif else exif
        raw = camera.get(36867) or exif.get(306)
        if not raw:
            return None
        value = datetime.strptime(str(raw), '%Y:%m:%d %H:%M:%S')
        offset = camera.get(36881)
        if offset:
            return datetime.fromisoformat(value.isoformat() + str(offset))
        return timezone.make_aware(value)
    except (ValueError, TypeError, KeyError, OSError):
        return None


def jpeg_rendition(image, max_dimension):
    image = ImageOps.exif_transpose(image)
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    if image.mode in {'RGBA', 'LA'} or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
    else:
        image = image.convert('RGB')
    out = BytesIO()
    image.save(out, 'JPEG', quality=PHOTO_JPEG_QUALITY, optimize=True)
    return out.getvalue(), image.width, image.height


def prepare_photo(content, *, extract_metadata=False):
    with decoded_photo(content) as image:
        gps = extract_observation_photo_gps(content) if extract_metadata else None
        taken_at = _taken_at(image) if extract_metadata else None
        encoded, width, height = jpeg_rendition(image, PHOTO_MAX_DIMENSION)
    return PreparedPhoto(encoded, width, height,
                         gps[0] if gps else None, gps[1] if gps else None, taken_at)
