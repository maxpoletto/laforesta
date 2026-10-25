import hashlib
import os
from collections import OrderedDict
from pathlib import Path

from django.http import FileResponse, HttpResponse
from django.utils.cache import patch_cache_control
from django.utils.http import http_date, parse_etags, parse_http_date_safe

CACHE_NO_CACHE = 'no_cache'
CACHE_NO_STORE = 'no_store'
_FILE_ETAGS = OrderedDict()
_MAX_FILE_ETAGS = 128


def _content_etag(file, stat):
    """Hash each published file once, caching by its complete stat identity.

    Files are published by atomic rename. Include ctime as well as mtime so
    replacements/backdated files cannot reuse a hash just because their size
    and Last-Modified date match. Hash and stream the same open descriptor.
    """
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity in _FILE_ETAGS:
        _FILE_ETAGS.move_to_end(identity)
        return _FILE_ETAGS[identity]
    etag = f'"{hashlib.file_digest(file, "sha256").hexdigest()}"'
    file.seek(0)
    _FILE_ETAGS[identity] = etag
    if len(_FILE_ETAGS) > _MAX_FILE_ETAGS:
        _FILE_ETAGS.popitem(last=False)
    return etag


def conditional_file_response(
        request,
        path: str | Path,
        *,
        content_type: str,
        cache_control: str,
        content_encoding: str | None = None,
        use_etag: bool = False,
) -> FileResponse | HttpResponse:
    file = open(path, 'rb')
    try:
        stat = os.fstat(file.fileno())
        etag = _content_etag(file, stat) if use_etag else None
        response = not_modified_response(
            request, stat.st_mtime, cache_control=cache_control,
            etag=etag,
        )
        if response is not None:
            file.close()
            return response

        response = FileResponse(file, content_type=content_type)
        if content_encoding:
            response['Content-Encoding'] = content_encoding
        response['Last-Modified'] = http_date(stat.st_mtime)
        if etag is not None:
            response['ETag'] = etag
        apply_cache_control(response, cache_control)
        return response
    except BaseException:
        file.close()
        raise


def not_modified_response(
        request, mtime: float, *, cache_control: str,
        etag: str | None = None,
) -> HttpResponse | None:
    inm = request.META.get('HTTP_IF_NONE_MATCH')
    if etag is not None:
        # GET/HEAD If-None-Match uses weak comparison and takes precedence
        # over dates. Digests never trust dates alone: second-resolution
        # timestamps cannot identify a published generation.
        tags = parse_etags(inm or '')
        if '*' not in tags and etag not in [tag.removeprefix('W/') for tag in tags]:
            return None
    else:
        if inm is not None:
            return None
        ims = request.META.get('HTTP_IF_MODIFIED_SINCE')
        if not ims:
            return None
        ims_ts = parse_http_date_safe(ims)
        if ims_ts is None or ims_ts < int(mtime):
            return None
    response = HttpResponse(status=304)
    response['Last-Modified'] = http_date(mtime)
    if etag is not None:
        response['ETag'] = etag
    apply_cache_control(response, cache_control)
    return response


def apply_cache_control(response: HttpResponse, cache_control: str) -> None:
    if cache_control == CACHE_NO_CACHE:
        patch_cache_control(response, no_cache=True)
    elif cache_control == CACHE_NO_STORE:
        patch_cache_control(response, no_store=True)
    else:
        raise ValueError(f'Unknown cache_control: {cache_control}')
