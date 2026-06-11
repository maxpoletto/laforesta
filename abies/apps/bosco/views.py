"""Bosco API views."""

import io
import os
import re
from datetime import date as date_type
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio

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
    build_parcel_record, build_preserved_tree_record, mark_stale, serve_digest,
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
ALLOWED_SATELLITE_LAYERS = {'ndvi', 'ndmi', 'evi'}
SATELLITE_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
SATELLITE_DIFF_RAMP = (
    (0, np.array([180, 30, 30], dtype=np.float32)),
    (128, np.array([255, 255, 255], dtype=np.float32)),
    (255, np.array([30, 130, 30], dtype=np.float32)),
)
SATELLITE_INSIDE_ALPHA = 210
SATELLITE_OUTSIDE_ALPHA = 60

PARCELS_DIGEST = 'parcels'
FIELD_AREA_HA = 'area_ha'
FIELD_AVE_AGE = 'ave_age'
FIELD_LOCATION_NAME = 'location_name'
FIELD_ALTITUDE_MIN_M = 'altitude_min_m'
FIELD_ALTITUDE_MAX_M = 'altitude_max_m'
FIELD_ASPECT = 'aspect'
FIELD_GRADE_PCT = 'grade_pct'
FIELD_DESC_VEG = 'desc_veg'
FIELD_DESC_GEO = 'desc_geo'

PARCEL_METADATA_TEXT_FIELDS = {
    FIELD_LOCATION_NAME: ('Località', 200),
    FIELD_ASPECT: ('Esposizione', 20),
    FIELD_DESC_VEG: ('Descrizione vegetazione', None),
    FIELD_DESC_GEO: ('Descrizione geologia', None),
}


@login_required
def parcels_data(request):
    return serve_digest(request, 'parcels')


@login_required
def species_data(request):
    return serve_digest(request, FIELD_SPECIES)


@login_required
@require_writer
def parcel_metadata_form_view(request, parcel_id: int):
    return JsonResponse({HTML: _render_parcel_metadata_form(request, parcel_id)})


@login_required
@require_writer
@require_POST
def parcel_metadata_save_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id = int_or_none(body.get(ROW_ID))
    if row_id is None:
        raise Http404
    values, errors = _parse_parcel_metadata_body(body)
    if errors:
        return validation_error(errors, html=_render_parcel_metadata_form(request, row_id, body))

    with transaction.atomic():
        parcel = (Parcel.objects.select_for_update()
                  .select_related('region', 'eclass')
                  .filter(id=row_id).first())
        if parcel is None:
            raise Http404
        if parcel.version != submitted_version(body):
            return conflict_response(
                data_id=PARCELS_DIGEST, row_id=parcel.id,
                record=build_parcel_record(parcel),
                html=_render_parcel_metadata_form(request, parcel.id),
            )
        for field, value in values.items():
            setattr(parcel, field, value)
        parcel.version += 1
        parcel.save(update_fields=[*values.keys(), VERSION])
        mark_stale(PARCELS_DIGEST)

    return success_response(
        request, body, data_id=PARCELS_DIGEST, row_id=parcel.id,
        patches=[row_patch(PARCELS_DIGEST, parcel.id, build_parcel_record(parcel))],
    )


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


@login_required
def satellite_diff_png(request, region_id, layer, date1, date2):
    if layer not in ALLOWED_SATELLITE_LAYERS:
        raise Http404
    if not SATELLITE_DATE_RE.fullmatch(date1) or not SATELLITE_DATE_RE.fullmatch(date2):
        raise Http404

    path1 = _satellite_file_path(region_id, date1, f'{layer}.tif')
    path2 = _satellite_file_path(region_id, date2, f'{layer}.tif')
    mask_path = _satellite_file_path(region_id, 'parcel-mask.tif', required=False)
    paths = [path1, path2, *([mask_path] if mask_path else [])]
    mtime = max(os.path.getmtime(path) for path in paths)
    not_modified = _satellite_not_modified_response(request, mtime)
    if not_modified is not None:
        return not_modified

    png, max_abs = _render_satellite_diff_png(path1, path2, mask_path)
    response = HttpResponse(png, content_type='image/png')
    response['Last-Modified'] = http_date(mtime)
    response['X-Bosco-Max-Abs'] = f'{max_abs / 127.5:.6g}'
    patch_cache_control(response, no_cache=True)
    return response


def _render_parcel_metadata_form(request, parcel_id: int, values: dict | None = None):
    parcel = Parcel.objects.select_related('region', 'eclass').filter(id=parcel_id).first()
    if parcel is None:
        raise Http404
    form_values = _parcel_metadata_form_values(parcel, values)
    return render_to_string('bosco/_parcel_metadata_form.html', {
        'parcel': parcel,
        'version': parcel.version,
        'values': form_values,
    }, request=request)


def _parcel_metadata_form_values(parcel, values: dict | None = None):
    def value(key, default):
        if values and key in values:
            return values.get(key) or ''
        return default if default is not None else ''

    return {
        FIELD_AREA_HA: value(FIELD_AREA_HA, parcel.area_ha),
        FIELD_AVE_AGE: value(FIELD_AVE_AGE, parcel.ave_age),
        FIELD_LOCATION_NAME: value(FIELD_LOCATION_NAME, parcel.location_name),
        FIELD_ALTITUDE_MIN_M: value(FIELD_ALTITUDE_MIN_M, parcel.altitude_min_m),
        FIELD_ALTITUDE_MAX_M: value(FIELD_ALTITUDE_MAX_M, parcel.altitude_max_m),
        FIELD_ASPECT: value(FIELD_ASPECT, parcel.aspect),
        FIELD_GRADE_PCT: value(FIELD_GRADE_PCT, parcel.grade_pct),
        FIELD_DESC_VEG: value(FIELD_DESC_VEG, parcel.desc_veg),
        FIELD_DESC_GEO: value(FIELD_DESC_GEO, parcel.desc_geo),
    }


def _parse_parcel_metadata_body(body: dict):
    errors = []
    area_ha = parse_decimal(body.get(FIELD_AREA_HA))
    if area_ha is None or area_ha <= 0:
        errors.append('Superficie obbligatoria.')

    values = {
        FIELD_AREA_HA: area_ha,
        FIELD_AVE_AGE: _optional_int(body, FIELD_AVE_AGE, 'Età media', errors),
        FIELD_LOCATION_NAME: _text_value(body, FIELD_LOCATION_NAME, errors),
        FIELD_ALTITUDE_MIN_M: _optional_int(body, FIELD_ALTITUDE_MIN_M, 'Altitudine minima', errors),
        FIELD_ALTITUDE_MAX_M: _optional_int(body, FIELD_ALTITUDE_MAX_M, 'Altitudine massima', errors),
        FIELD_ASPECT: _text_value(body, FIELD_ASPECT, errors),
        FIELD_GRADE_PCT: _optional_int(body, FIELD_GRADE_PCT, 'Pendenza', errors),
        FIELD_DESC_VEG: _text_value(body, FIELD_DESC_VEG, errors),
        FIELD_DESC_GEO: _text_value(body, FIELD_DESC_GEO, errors),
    }
    alt_min = values[FIELD_ALTITUDE_MIN_M]
    alt_max = values[FIELD_ALTITUDE_MAX_M]
    if alt_min is not None and alt_max is not None and alt_min > alt_max:
        errors.append('Altitudine minima maggiore della massima.')
    return values, errors


def _optional_int(body: dict, key: str, label: str, errors: list[str]):
    raw = body.get(key)
    if raw in (None, ''):
        return None
    value = int_or_none(raw)
    if value is None:
        errors.append(f'{label} deve essere un numero intero.')
    return value


def _text_value(body: dict, key: str, errors: list[str]):
    label, max_len = PARCEL_METADATA_TEXT_FIELDS[key]
    value = str(body.get(key) or '').strip()
    if max_len is not None and len(value) > max_len:
        errors.append(f'{label} troppo lunga.')
    return value


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
    not_modified = _satellite_not_modified_response(request, mtime)
    if not_modified is not None:
        return not_modified
    response = FileResponse(open(path, 'rb'), content_type='application/json')
    response['Last-Modified'] = http_date(mtime)
    patch_cache_control(response, no_cache=True)
    return response


def _satellite_not_modified_response(request, mtime):
    ims = request.META.get('HTTP_IF_MODIFIED_SINCE')
    if not ims:
        return None
    ims_ts = parse_http_date_safe(ims)
    if ims_ts is None or ims_ts < int(mtime):
        return None
    response = HttpResponse(status=304)
    patch_cache_control(response, no_cache=True)
    return response


def _satellite_json_path(region_id, filename):
    if filename not in ALLOWED_SATELLITE_JSON:
        raise Http404
    return _satellite_file_path(region_id, filename)


def _satellite_file_path(region_id, *parts, required=True):
    region_dir = _satellite_region_dir(region_id)
    path = region_dir.joinpath(*parts).resolve()
    try:
        path.relative_to(region_dir)
    except ValueError as exc:
        raise Http404 from exc
    if not path.is_file():
        if required:
            raise Http404
        return None
    return path


def _satellite_region_dir(region_id):
    try:
        region = Region.objects.get(pk=region_id)
    except Region.DoesNotExist as exc:
        raise Http404 from exc

    base = Path(settings.SATELLITE_DIR).resolve()
    path = (base / region.name).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise Http404 from exc
    return path


def _render_satellite_diff_png(path1, path2, mask_path):
    raster1 = _read_raster(path1).astype(np.float32)
    raster2 = _read_raster(path2).astype(np.float32)
    if raster1.shape != raster2.shape:
        raise Http404

    mask = None
    if mask_path is not None:
        mask = _read_raster(mask_path) > 0
        if mask.shape != raster1.shape:
            raise Http404

    diff = raster2 - raster1
    finite = np.isfinite(diff)
    range_mask = finite & mask if mask is not None else finite
    if np.any(range_mask):
        selected = diff[range_mask]
        max_abs = float(max(abs(np.min(selected)), abs(np.max(selected)))) or 1.0
    else:
        max_abs = 1.0

    positions = np.clip(np.rint(((np.nan_to_num(diff) / max_abs) + 1) * 127.5), 0, 255)
    rgb = _diff_ramp_rgb(positions)
    alpha = np.full(diff.shape, SATELLITE_INSIDE_ALPHA, dtype=np.uint8)
    if mask is not None:
        alpha[:] = SATELLITE_OUTSIDE_ALPHA
        alpha[mask] = SATELLITE_INSIDE_ALPHA
    alpha[~finite] = 0
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(rgba).save(out, format='PNG', optimize=True)
    return out.getvalue(), max_abs


def _read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)


def _diff_ramp_rgb(positions):
    values = positions.astype(np.float32)
    rgb = np.empty((*positions.shape, 3), dtype=np.float32)
    low_pos, low_rgb = SATELLITE_DIFF_RAMP[0]
    mid_pos, mid_rgb = SATELLITE_DIFF_RAMP[1]
    high_pos, high_rgb = SATELLITE_DIFF_RAMP[2]

    lower = values <= mid_pos
    lower_t = ((values[lower] - low_pos) / (mid_pos - low_pos))[:, None]
    upper_t = ((values[~lower] - mid_pos) / (high_pos - mid_pos))[:, None]
    rgb[lower] = low_rgb + (mid_rgb - low_rgb) * lower_t
    rgb[~lower] = mid_rgb + (high_rgb - mid_rgb) * upper_t
    return np.rint(rgb).astype(np.uint8)
