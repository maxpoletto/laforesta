"""Bosco API views."""

import os
from datetime import date as date_type
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.utils.cache import patch_cache_control
from django.utils.http import http_date, parse_http_date_safe
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import (
    build_preserved_tree_record, mark_stale, serve_digest,
)
from apps.base.models import Parcel, Region, Species, Tree, parcel_sort_key
from apps.base.numparse import coord_float, int_or_none, parse_decimal
from apps.base.responses import (
    conflict_response, parse_json_body, row_delete, row_patch, submitted_version,
    success_response, validation_error,
)
from config.constants import (
    DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES, FIELD_LAT,
    FIELD_LON, FIELD_PARCEL_ID, FIELD_REGION_ID, FIELD_SPECIES,
    FIELD_SPECIES_ID, FIELD_YEAR, HTML, ROW_ID, VERSION,
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
@require_writer
def pai_form_view(request, tree_id: int | None = None):
    return JsonResponse({HTML: _render_pai_form(request, tree_id)})


@login_required
@require_writer
@require_POST
def pai_save_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id, values, errors = _parse_pai_body(body)
    if errors:
        return validation_error(errors, html=_render_pai_form(request, row_id, body))

    with transaction.atomic():
        if row_id is None:
            tree = Tree.objects.create(preserved=True, coppice=False, **values)
        else:
            tree = _preserved_tree_qs(for_update=True).filter(id=row_id).first()
            if tree is None:
                raise Http404
            if tree.version != submitted_version(body):
                return conflict_response(
                    data_id=DIGEST_PRESERVED_TREES, row_id=tree.id,
                    record=build_preserved_tree_record(tree),
                    html=_render_pai_form(request, tree.id),
                )
            for field, value in values.items():
                setattr(tree, field, value)
            tree.version += 1
            tree.save()
        mark_stale(DIGEST_PRESERVED_TREES)

    tree = _preserved_tree_qs().get(id=tree.id)
    return success_response(
        request, body, data_id=DIGEST_PRESERVED_TREES, row_id=tree.id,
        patches=[row_patch(
            DIGEST_PRESERVED_TREES, tree.id, build_preserved_tree_record(tree),
        )],
    )


@login_required
@require_writer
@require_POST
def pai_delete_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id = int_or_none(body.get(ROW_ID))
    if row_id is None:
        raise Http404

    with transaction.atomic():
        tree = _preserved_tree_qs(for_update=True).filter(id=row_id).first()
        if tree is None:
            raise Http404
        if tree.version != submitted_version(body):
            return conflict_response(
                data_id=DIGEST_PRESERVED_TREES, row_id=tree.id,
                record=build_preserved_tree_record(tree),
            )
        tree.preserved = False
        tree.version += 1
        tree.save(update_fields=['preserved', VERSION])
        mark_stale(DIGEST_PRESERVED_TREES)

    return success_response(
        request, body, data_id=DIGEST_PRESERVED_TREES, row_id=row_id,
        deletes=[row_delete(DIGEST_PRESERVED_TREES, row_id)],
    )


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


def _render_pai_form(request, tree_id: int | None = None, values: dict | None = None):
    tree = None
    if tree_id is not None:
        tree = _preserved_tree_qs().filter(id=tree_id).first()
        if tree is None:
            raise Http404

    region_id = int_or_none(request.GET.get(FIELD_REGION_ID))
    selected_parcel_id = int_or_none(values.get(FIELD_PARCEL_ID)) if values else None
    if selected_parcel_id is None:
        selected_parcel_id = tree.parcel_id if tree else int_or_none(request.GET.get(FIELD_PARCEL_ID))
    selected_species_id = int_or_none(values.get(FIELD_SPECIES_ID)) if values else None
    if selected_species_id is None:
        selected_species_id = tree.species_id if tree else None

    parcels = list(Parcel.objects.select_related('region', 'eclass'))
    if region_id:
        parcels.sort(key=lambda p: (p.region_id != region_id, parcel_sort_key(p)))
    else:
        parcels.sort(key=parcel_sort_key)

    return render_to_string('bosco/_pai_form.html', {
        'tree': tree,
        'version': tree.version if tree else 0,
        'species': Species.objects.order_by('sort_order', 'common_name'),
        'parcels': parcels,
        'selected_species_id': selected_species_id,
        'selected_parcel_id': selected_parcel_id,
        'year': _form_value(values, FIELD_YEAR, tree.year if tree else date_type.today().year),
        'lat': _form_value(values, FIELD_LAT, tree.lat if tree else ''),
        'lon': _form_value(values, FIELD_LON, tree.lon if tree else ''),
    }, request=request)


def _form_value(values: dict | None, key: str, default):
    if values and key in values:
        return values.get(key)
    return default if default is not None else ''


def _parse_pai_body(body: dict):
    row_id = int_or_none(body.get(ROW_ID))
    species_id = int_or_none(body.get(FIELD_SPECIES_ID))
    parcel_id = int_or_none(body.get(FIELD_PARCEL_ID))
    year = int_or_none(body.get(FIELD_YEAR))
    lat = coord_float(parse_decimal(body.get(FIELD_LAT)))
    lon = coord_float(parse_decimal(body.get(FIELD_LON)))

    errors = []
    if species_id is None or not Species.objects.filter(id=species_id).exists():
        errors.append('Specie obbligatoria.')
    if parcel_id is None or not Parcel.objects.filter(id=parcel_id).exists():
        errors.append('Particella obbligatoria.')
    if year is None:
        errors.append('Anno obbligatorio.')
    if lat is None or lon is None:
        errors.append('Lat e Lon obbligatorie.')

    return row_id, {
        'species_id': species_id,
        'parcel_id': parcel_id,
        'year': year,
        'lat': lat,
        'lon': lon,
    }, errors


def _preserved_tree_qs(*, for_update=False):
    qs = Tree.objects.filter(preserved=True).select_related('parcel__region', 'species')
    return qs.select_for_update() if for_update else qs


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
