"""Bosco API views."""

import base64
import os
import re
from datetime import date as date_type
from pathlib import Path

import numpy as np
import rasterio

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Max
from django.http import Http404, JsonResponse
from django.template.loader import render_to_string
from django.utils.http import http_date
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import (
    build_parcel_record, build_preserved_tree_record, mark_stale, serve_digest,
)
from apps.base.http import (
    CACHE_NO_CACHE, apply_cache_control, conditional_file_response, not_modified_response,
)
from apps.base.models import Parcel, Region, Species, Tree, TreePreserved, parcel_sort_key
from apps.base.numparse import coord_float, int_or_none, parse_decimal
from apps.base.responses import (
    conflict_response, parse_json_body, row_delete, row_patch, submitted_version,
    success_response, validation_error,
)
from config import strings as S
from config.constants import (
    DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PARCELS, DIGEST_PRESERVED_TREES,
    FIELD_ACC_M, FIELD_DATE, FIELD_D_CM, FIELD_ESTIMATED_BIRTH_YEAR, FIELD_H_M,
    FIELD_H_MEASURED, FIELD_LAT, FIELD_LON, FIELD_NOTE, FIELD_NUMBER,
    FIELD_OPERATOR, FIELD_PARCEL_ID, FIELD_REGION_ID, FIELD_SPECIES_ID, HTML,
    ROW_ID, VERSION,
)

ALLOWED_SATELLITE_JSON = {'manifest.json', 'timeseries.json'}
ALLOWED_SATELLITE_LAYERS = {'ndvi', 'ndmi', 'evi'}
SATELLITE_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
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
    FIELD_LOCATION_NAME: (S.COL_LOCATION, 200),
    FIELD_ASPECT: (S.COL_ASPECT, 20),
    FIELD_DESC_VEG: (S.LABEL_BOSCO_VEG_DESC, None),
    FIELD_DESC_GEO: (S.LABEL_BOSCO_GEO_DESC, None),
}


@login_required
def parcels_data(request):
    return serve_digest(request, DIGEST_PARCELS)


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
                data_id=DIGEST_PARCELS, row_id=parcel.id,
                record=build_parcel_record(parcel),
                html=_render_parcel_metadata_form(request, parcel.id),
            )
        for field, value in values.items():
            setattr(parcel, field, value)
        parcel.version += 1
        parcel.save(update_fields=[*values.keys(), VERSION])
        mark_stale(DIGEST_PARCELS, 'audit')

    return success_response(
        request, body, data_id=DIGEST_PARCELS, row_id=parcel.id,
        patches=[row_patch(DIGEST_PARCELS, parcel.id, build_parcel_record(parcel))],
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
            tree = Tree.objects.create(
                species_id=values[FIELD_SPECIES_ID],
                parcel_id=values[FIELD_PARCEL_ID],
                estimated_birth_year=values[FIELD_ESTIMATED_BIRTH_YEAR],
                lat=values[FIELD_LAT],
                lon=values[FIELD_LON],
                acc_m=values[FIELD_ACC_M],
                preserved=True,
                coppice=False,
            )
            pai = TreePreserved.objects.create(
                tree=tree,
                parcel_id=values[FIELD_PARCEL_ID],
                number=values[FIELD_NUMBER],
                date=values[FIELD_DATE],
                d_cm=values[FIELD_D_CM],
                h_m=values[FIELD_H_M],
                h_measured=values[FIELD_H_MEASURED],
                lat=values[FIELD_LAT],
                lon=values[FIELD_LON],
                acc_m=values[FIELD_ACC_M],
                operator=values[FIELD_OPERATOR],
                note=values[FIELD_NOTE],
            )
        else:
            pai = _preserved_tree_qs(for_update=True).filter(id=row_id).first()
            if pai is None:
                raise Http404
            if pai.version != submitted_version(body):
                fresh_pai = _preserved_tree_qs().get(id=pai.id)
                return conflict_response(
                    data_id=DIGEST_PRESERVED_TREES, row_id=fresh_pai.id,
                    record=build_preserved_tree_record(fresh_pai),
                    html=_render_pai_form(request, fresh_pai.id),
                )
            tree = pai.tree
            tree.species_id = values[FIELD_SPECIES_ID]
            tree.parcel_id = values[FIELD_PARCEL_ID]
            tree.estimated_birth_year = values[FIELD_ESTIMATED_BIRTH_YEAR]
            tree.lat = values[FIELD_LAT]
            tree.lon = values[FIELD_LON]
            tree.acc_m = values[FIELD_ACC_M]
            tree.preserved = True
            tree.coppice = False
            tree.version += 1
            tree.save()
            pai.parcel_id = values[FIELD_PARCEL_ID]
            pai.number = values[FIELD_NUMBER]
            pai.date = values[FIELD_DATE]
            pai.d_cm = values[FIELD_D_CM]
            pai.h_m = values[FIELD_H_M]
            pai.h_measured = values[FIELD_H_MEASURED]
            pai.lat = values[FIELD_LAT]
            pai.lon = values[FIELD_LON]
            pai.acc_m = values[FIELD_ACC_M]
            pai.operator = values[FIELD_OPERATOR]
            pai.note = values[FIELD_NOTE]
            pai.version += 1
            pai.save()
        mark_stale(DIGEST_PRESERVED_TREES)

    pai = _preserved_tree_qs().get(id=pai.id)
    return success_response(
        request, body, data_id=DIGEST_PRESERVED_TREES, row_id=pai.id,
        patches=[row_patch(
            DIGEST_PRESERVED_TREES, pai.id, build_preserved_tree_record(pai),
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
        pai = _preserved_tree_qs(for_update=True).filter(id=row_id).first()
        if pai is None:
            raise Http404
        if pai.version != submitted_version(body):
            fresh_pai = _preserved_tree_qs().get(id=pai.id)
            return conflict_response(
                data_id=DIGEST_PRESERVED_TREES, row_id=fresh_pai.id,
                record=build_preserved_tree_record(fresh_pai),
            )
        tree = pai.tree
        pai.delete()
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
def satellite_raw(request, region_id, layer, date):
    if layer not in ALLOWED_SATELLITE_LAYERS:
        raise Http404
    if not SATELLITE_DATE_RE.fullmatch(date):
        raise Http404

    path = _satellite_file_path(region_id, date, f'{layer}.tif')
    return _satellite_raw_response(request, path)


@login_required
def satellite_mask_raw(request, region_id):
    path = _satellite_file_path(region_id, 'parcel-mask.tif')
    return _satellite_raw_response(request, path)


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
    fields = (
        (FIELD_AREA_HA, parcel.area_ha),
        (FIELD_AVE_AGE, parcel.ave_age),
        (FIELD_LOCATION_NAME, parcel.location_name),
        (FIELD_ALTITUDE_MIN_M, parcel.altitude_min_m),
        (FIELD_ALTITUDE_MAX_M, parcel.altitude_max_m),
        (FIELD_ASPECT, parcel.aspect),
        (FIELD_GRADE_PCT, parcel.grade_pct),
        (FIELD_DESC_VEG, parcel.desc_veg),
        (FIELD_DESC_GEO, parcel.desc_geo),
    )
    return {key: _form_value(values, key, default, blank=True) for key, default in fields}


def _parse_parcel_metadata_body(body: dict):
    errors = []
    area_ha = parse_decimal(body.get(FIELD_AREA_HA))
    if area_ha is None or area_ha <= 0:
        errors.append(S.ERR_BOSCO_AREA_REQUIRED)

    values = {
        FIELD_AREA_HA: area_ha,
        FIELD_AVE_AGE: _optional_int(body, FIELD_AVE_AGE, S.LABEL_BOSCO_AVE_AGE, errors),
        FIELD_LOCATION_NAME: _text_value(body, FIELD_LOCATION_NAME, errors),
        FIELD_ALTITUDE_MIN_M: _optional_int(
            body, FIELD_ALTITUDE_MIN_M, S.LABEL_BOSCO_ALTITUDE_MIN, errors,
        ),
        FIELD_ALTITUDE_MAX_M: _optional_int(
            body, FIELD_ALTITUDE_MAX_M, S.LABEL_BOSCO_ALTITUDE_MAX, errors,
        ),
        FIELD_ASPECT: _text_value(body, FIELD_ASPECT, errors),
        FIELD_GRADE_PCT: _optional_int(body, FIELD_GRADE_PCT, S.LABEL_BOSCO_GRADE, errors),
        FIELD_DESC_VEG: _text_value(body, FIELD_DESC_VEG, errors),
        FIELD_DESC_GEO: _text_value(body, FIELD_DESC_GEO, errors),
    }
    alt_min = values[FIELD_ALTITUDE_MIN_M]
    alt_max = values[FIELD_ALTITUDE_MAX_M]
    if alt_min is not None and alt_max is not None and alt_min > alt_max:
        errors.append(S.ERR_BOSCO_ALTITUDE_RANGE)
    return values, errors


def _optional_int(body: dict, key: str, label: str, errors: list[str]):
    raw = body.get(key)
    if raw in (None, ''):
        return None
    value = int_or_none(raw)
    if value is None:
        errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(label))
    return value


def _text_value(body: dict, key: str, errors: list[str]):
    label, max_len = PARCEL_METADATA_TEXT_FIELDS[key]
    value = str(body.get(key) or '').strip()
    if max_len is not None and len(value) > max_len:
        errors.append(S.ERR_BOSCO_TEXT_TOO_LONG.format(label))
    return value


def _render_pai_form(request, pai_id: int | None = None, values: dict | None = None):
    pai = None
    if pai_id is not None:
        pai = _preserved_tree_qs().filter(id=pai_id).first()
        if pai is None:
            raise Http404
    tree = pai.tree if pai else None

    region_id = int_or_none(request.GET.get(FIELD_REGION_ID))
    selected_parcel_id = int_or_none(values.get(FIELD_PARCEL_ID)) if values else None
    if selected_parcel_id is None:
        selected_parcel_id = pai.parcel_id if pai else int_or_none(request.GET.get(FIELD_PARCEL_ID))
    selected_species_id = int_or_none(values.get(FIELD_SPECIES_ID)) if values else None
    if selected_species_id is None:
        selected_species_id = tree.species_id if tree else None

    parcels = list(Parcel.objects.select_related('region', 'eclass'))
    if region_id:
        parcels.sort(key=lambda p: (p.region_id != region_id, parcel_sort_key(p)))
    else:
        parcels.sort(key=parcel_sort_key)

    return render_to_string('bosco/_pai_form.html', {
        'pai': pai,
        'tree': tree,
        'version': pai.version if pai else 0,
        'species': Species.objects.order_by('sort_order', 'common_name'),
        'parcels': parcels,
        'selected_species_id': selected_species_id,
        'selected_parcel_id': selected_parcel_id,
        'number': _form_value(values, FIELD_NUMBER, pai.number if pai else ''),
        'date': _form_value(values, FIELD_DATE, pai.date.isoformat() if pai and pai.date else ''),
        'estimated_birth_year': _form_value(
            values, FIELD_ESTIMATED_BIRTH_YEAR,
            tree.estimated_birth_year if tree else '',
        ),
        'd_cm': _form_value(values, FIELD_D_CM, pai.d_cm if pai else '', blank=True),
        'h_m': _form_value(values, FIELD_H_M, pai.h_m if pai else '', blank=True),
        'lat': _coord_form_value(values, FIELD_LAT, pai.lat if pai else request.GET.get(FIELD_LAT, '')),
        'lon': _coord_form_value(values, FIELD_LON, pai.lon if pai else request.GET.get(FIELD_LON, '')),
        'note': _form_value(values, FIELD_NOTE, pai.note if pai else '', blank=True),
    }, request=request)


def _next_pai_number(parcel_id: int, row_id: int | None = None):
    qs = TreePreserved.objects.filter(parcel_id=parcel_id)
    if row_id is not None:
        qs = qs.exclude(id=row_id)
    return (qs.aggregate(max_number=Max(FIELD_NUMBER))['max_number'] or 0) + 1


def _form_value(values: dict | None, key: str, default, *, blank=False):
    if values and key in values:
        value = values.get(key)
        if blank:
            return value or ''
        return value
    return default if default is not None else ''


def _coord_form_value(values: dict | None, key: str, default):
    value = _form_value(values, key, default)
    if value in (None, ''):
        return ''
    coord = coord_float(parse_decimal(value))
    if coord is None:
        return value
    return f'{coord:.5f}'


def _parse_pai_body(body: dict):
    row_id = int_or_none(body.get(ROW_ID))
    species_id = int_or_none(body.get(FIELD_SPECIES_ID))
    parcel_id = int_or_none(body.get(FIELD_PARCEL_ID))
    raw_number = body.get(FIELD_NUMBER)
    auto_number = raw_number in (None, '')
    number = None if auto_number else int_or_none(raw_number)
    estimated_birth_year, birth_year_ok = _optional_form_int(
        body, FIELD_ESTIMATED_BIRTH_YEAR,
    )
    d_cm, d_ok = _optional_form_int(body, FIELD_D_CM)
    h_m = parse_decimal(body.get(FIELD_H_M))
    lat = coord_float(parse_decimal(body.get(FIELD_LAT)))
    lon = coord_float(parse_decimal(body.get(FIELD_LON)))
    acc_m, acc_ok = _optional_form_int(body, FIELD_ACC_M)
    date, date_ok = _optional_form_date(body.get(FIELD_DATE))
    note = (body.get(FIELD_NOTE) or '').strip()
    operator = (body.get(FIELD_OPERATOR) or '').strip()

    errors = []
    if species_id is None or not Species.objects.filter(id=species_id).exists():
        errors.append(S.ERR_BOSCO_SPECIES_REQUIRED)
    parcel_exists = parcel_id is not None and Parcel.objects.filter(id=parcel_id).exists()
    if not parcel_exists:
        errors.append(S.ERR_BOSCO_PARCEL_REQUIRED)
    if auto_number and parcel_exists:
        number = _next_pai_number(parcel_id, row_id)
    elif number is None or number <= 0:
        errors.append(S.ERR_BOSCO_POSITIVE_INTEGER_REQUIRED.format(S.COL_NUMBER))
    if number is not None and parcel_exists:
        duplicates = TreePreserved.objects.filter(parcel_id=parcel_id, number=number)
        if row_id is not None:
            duplicates = duplicates.exclude(id=row_id)
        if duplicates.exists():
            errors.append(S.ERR_BOSCO_PAI_NUMBER_DUPLICATE)
    if not birth_year_ok:
        errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.COL_ESTIMATED_BIRTH_YEAR))
    if not d_ok or (d_cm is not None and d_cm <= 0):
        errors.append(S.ERR_BOSCO_POSITIVE_INTEGER_REQUIRED.format(S.COL_D_CM))
    if body.get(FIELD_H_M) not in (None, '') and (h_m is None or h_m <= 0):
        errors.append(S.ERR_MARK_H_REQUIRED)
    if not acc_ok:
        errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.CSV_COL_ACC_M))
    if not date_ok:
        errors.append(S.ERR_DATE_INVALID)
    if lat is None or lon is None:
        errors.append(S.ERR_BOSCO_LAT_LON_REQUIRED)

    return row_id, {
        FIELD_SPECIES_ID: species_id,
        FIELD_PARCEL_ID: parcel_id,
        FIELD_NUMBER: number,
        FIELD_DATE: date,
        FIELD_ESTIMATED_BIRTH_YEAR: estimated_birth_year,
        FIELD_D_CM: d_cm,
        FIELD_H_M: h_m,
        FIELD_H_MEASURED: h_m is not None,
        FIELD_LAT: lat,
        FIELD_LON: lon,
        FIELD_ACC_M: acc_m,
        FIELD_OPERATOR: operator,
        FIELD_NOTE: note,
    }, errors


def _optional_form_int(body: dict, key: str):
    raw = body.get(key)
    if raw is None or str(raw).strip() == '':
        return None, True
    value = int_or_none(str(raw).strip())
    return value, value is not None


def _optional_form_date(raw):
    raw = (raw or '').strip()
    if not raw:
        return None, True
    try:
        return date_type.fromisoformat(raw), True
    except ValueError:
        return None, False


def _preserved_tree_qs(*, for_update=False):
    qs = (TreePreserved.objects
          .filter(tree__preserved=True)
          .select_related('parcel__region', 'tree__species'))
    return qs.select_for_update() if for_update else qs


def _satellite_json_response(request, region_id, filename):
    return conditional_file_response(
        request,
        _satellite_json_path(region_id, filename),
        content_type='application/json',
        cache_control=CACHE_NO_CACHE,
    )


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


def _satellite_raw_response(request, path):
    mtime = os.path.getmtime(path)
    not_modified = not_modified_response(request, mtime, cache_control=CACHE_NO_CACHE)
    if not_modified is not None:
        return not_modified

    payload = _satellite_raw_payload(path)
    response = JsonResponse(payload)
    response['Last-Modified'] = http_date(mtime)
    apply_cache_control(response, CACHE_NO_CACHE)
    return response


def _satellite_raw_payload(path):
    with rasterio.open(path) as src:
        data = np.ascontiguousarray(src.read(1).astype(np.uint8, copy=False))
        bounds = src.bounds
        payload = {
            'width': src.width,
            'height': src.height,
            'bbox': [[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            'data': base64.b64encode(data.tobytes()).decode('ascii'),
        }
    return payload
