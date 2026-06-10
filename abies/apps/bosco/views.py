"""Bosco API views."""

import os
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404, HttpResponse
from django.utils.cache import patch_cache_control
from django.utils.http import http_date, parse_http_date_safe

from apps.base.digests import serve_digest
from apps.base.models import Region
from config.constants import (
    DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES, FIELD_SPECIES,
)

ALLOWED_SATELLITE_JSON = {'manifest.json', 'timeseries.json'}


@login_required
def parcels_data(request):
    return serve_digest(request, 'parcels')


@login_required
def species_data(request):
    return serve_digest(request, FIELD_SPECIES)


@login_required
def preserved_trees_data(request):
    return serve_digest(request, DIGEST_PRESERVED_TREES)


@login_required
def future_production_data(request):
    return serve_digest(request, DIGEST_FUTURE_PRODUCTION)


@login_required
def parcel_dendrometry_data(request):
    return serve_digest(request, DIGEST_PARCEL_DENDROMETRY)


@login_required
def parcel_dendrometry_points_data(request):
    return serve_digest(request, DIGEST_PARCEL_DENDROMETRY_POINTS)


@login_required
def satellite_manifest(request, region_id):
    return _satellite_json_response(request, region_id, 'manifest.json')


@login_required
def satellite_timeseries(request, region_id):
    return _satellite_json_response(request, region_id, 'timeseries.json')


def _satellite_json_response(request, region_id, filename):
    path = _satellite_json_path(region_id, filename)
    mtime = os.path.getmtime(path)
    ims = request.META.get('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims_ts = parse_http_date_safe(ims)
        if ims_ts is not None and ims_ts >= int(mtime):
            response = HttpResponse(status=304)
            patch_cache_control(response, no_cache=True)
            return response
    response = FileResponse(open(path, 'rb'), content_type='application/json')
    response['Last-Modified'] = http_date(mtime)
    patch_cache_control(response, no_cache=True)
    return response


def _satellite_json_path(region_id, filename):
    if filename not in ALLOWED_SATELLITE_JSON:
        raise Http404
    try:
        region = Region.objects.get(pk=region_id)
    except Region.DoesNotExist as exc:
        raise Http404 from exc

    base = Path(settings.SATELLITE_DIR).resolve()
    path = (base / region.name / filename).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise Http404 from exc
    if not path.is_file():
        raise Http404
    return path
