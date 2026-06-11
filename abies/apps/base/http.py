import os
from pathlib import Path

from django.http import FileResponse, HttpResponse
from django.utils.cache import patch_cache_control
from django.utils.http import http_date, parse_http_date_safe

CACHE_NO_CACHE = 'no_cache'
CACHE_NO_STORE = 'no_store'


def conditional_file_response(
        request,
        path: str | Path,
        *,
        content_type: str,
        cache_control: str,
        content_encoding: str | None = None,
) -> FileResponse | HttpResponse:
    mtime = os.path.getmtime(path)
    response = not_modified_response(request, mtime, cache_control=cache_control)
    if response is not None:
        return response

    response = FileResponse(open(path, 'rb'), content_type=content_type)
    if content_encoding:
        response['Content-Encoding'] = content_encoding
    response['Last-Modified'] = http_date(mtime)
    apply_cache_control(response, cache_control)
    return response


def not_modified_response(request, mtime: float, *, cache_control: str) -> HttpResponse | None:
    ims = request.META.get('HTTP_IF_MODIFIED_SINCE')
    if not ims:
        return None
    ims_ts = parse_http_date_safe(ims)
    if ims_ts is None or ims_ts < int(mtime):
        return None
    response = HttpResponse(status=304)
    apply_cache_control(response, cache_control)
    return response


def apply_cache_control(response: HttpResponse, cache_control: str) -> None:
    if cache_control == CACHE_NO_CACHE:
        patch_cache_control(response, no_cache=True)
    elif cache_control == CACHE_NO_STORE:
        patch_cache_control(response, no_store=True)
    else:
        raise ValueError(f'Unknown cache_control: {cache_control}')
