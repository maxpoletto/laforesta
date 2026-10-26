"""Small genuine images for upload and decoder tests."""

from io import BytesIO

from PIL import Image


def image_bytes(size=(40, 20), *, fmt='JPEG', color='green', exif=None):
    out = BytesIO()
    Image.new('RGB', size, color).save(out, fmt, **({'exif': exif} if exif else {}))
    return out.getvalue()
