"""Serve the Ipso offline PWA from the Abies origin."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import shutil
import time
from collections import defaultdict, deque
from datetime import timezone
from decimal import Decimal, InvalidOperation
from ipaddress import ip_address, ip_network
from pathlib import Path

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.staticfiles import finders
from django.core.exceptions import RequestDataTooBig
from django.db import IntegrityError, transaction
from django.db.models import Max
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.utils import timezone as django_timezone
from django.utils.cache import patch_vary_headers
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from apps.base import csv_io
from apps.base.auth import require_admin, require_writer
from apps.base.http import (
    CACHE_NO_CACHE, CACHE_NO_STORE, apply_cache_control,
    conditional_file_response,
)
from apps.base.models import (
    HYPSO_FUNC_LN, HarvestPlanItem, HarvestPlanItemState, HypsoParam,
    HypsoParamSet, Parcel, SampleArea, Species, Survey, TreePreserved,
    TreeSample, natural_sort_key, parcel_sort_key,
)
from apps.base.numparse import coord_float, to_decimal
from apps.base.responses import row_delete, success_response, validation_error
from apps.campionamenti import csv_trees
from apps.ipso import staging as ipso_staging
from apps.ipso.importers import (
    apply_pai_rows, pai_import_rows, record_measurements, sample_import_rows,
)
from apps.ipso.models import IpsoUpload, IpsoUploadState
from apps.piano_di_taglio.mark_import import (
    MarkImportRow, import_mark_rows, ipso_mark_fingerprint,
    mark_number_duplicate_errors,
)
from config import strings as S
from config.constants import (
    COLUMNS, DETAIL, DUPLICATE, ERROR, FIELD_ACC_M, FIELD_COPPICE,
    FIELD_DAMAGED, FIELD_CLIENT_RECORD_ID,
    FIELD_COMPLETED_AT, FIELD_CREATED_AT, FIELD_CSV_TEXT, FIELD_DATE,
    FIELD_D_CM, FIELD_ESTIMATED_BIRTH_YEAR, FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_H_M, FIELD_H_MEASURED, FIELD_HYPSO_PARAM_SET_ID, FIELD_LAT,
    FIELD_LON, FIELD_L10_MM, FIELD_MODE, FIELD_NOTE,
    FIELD_NUMBER, FIELD_OPERATOR, FIELD_PARCEL_ID,
    FIELD_PRESERVED, FIELD_PRESSLER_COEFF, FIELD_REFERENCE_VERSION,
    FIELD_REGION_ID, FIELD_SAMPLE_AREA_ID, FIELD_SAMPLE_GRID_ID,
    FIELD_SCHEMA_VERSION,
    FIELD_SESSION_ID, FIELD_SHOOT, FIELD_SPECIES_ID, FIELD_STANDARD,
    FIELD_TREE_ID, FIELD_TREE_PRESERVED_ID,
    FIELD_STATE, FIELD_SURVEY_ID,
    FIELD_WORK_PACKAGE_ID, FIELD_WORK_PACKAGE_LABEL, FILE_ERROR, IMPORTED,
    IPSO_ERROR_AUTH, IPSO_ERROR_CONFLICT, IPSO_ERROR_INVALID_PAYLOAD,
    IPSO_ERROR_RATE_LIMITED, IPSO_ERROR_TOO_LARGE, IPSO_MODE_MARTELLATE,
    IPSO_MODE_PAI, IPSO_MODE_SAMPLES, IPSO_REF_GENERATED_AT,
    IPSO_REF_HYPSOMETRY, IPSO_REF_PAI, IPSO_REF_PARCELS,
    IPSO_REF_PRESERVED_TREES, IPSO_REF_SAMPLE_AREA_MAX_NUMBERS,
    DATA_ID_IPSO_UPLOADS,
    IPSO_REF_SAMPLE_AREAS, IPSO_REF_SAMPLING, IPSO_REF_SPECIES,
    IPSO_REF_SURVEYS, IPSO_REF_WORK_PACKAGES,
    IPSO_REFERENCE_JSON, IPSO_REFERENCE_VERSION_KEYS,
    IPSO_TARGET_HARVEST_PLAN_ITEM, IPSO_TARGET_PAI, IPSO_TARGET_SURVEY,
    IPSO_TERRENI_GEOJSON, IPSO_WORK_PACKAGE_HARVEST_PREFIX,
    IPSO_WORK_PACKAGE_SAMPLING_SURVEY, IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX,
    IPSO_UPLOAD_CONFIG_JS, IPSO_UPLOAD_FILE_CSV, IPSO_UPLOAD_FILE_JSON,
    IPSO_UPLOAD_FILE_SHA256, IPSO_UPLOAD_MODES, MESSAGE,
    OK,
    PENDING_COUNT, PRESSLER_DEFAULT, RECORD_COUNT, RECORDS, ROW_ID, ROWS,
    SESSION, SKIPPED_DUPLICATES, STORED_AS, SUGGESTED_TARGET_ID, TARGETS,
    TREE_H_QUANTUM, UPLOAD,
)

SCHEMA_VERSION = 1
UPLOAD_SCHEMA_VERSION = 1
ALLOWED_UPLOAD_MODES = set(IPSO_UPLOAD_MODES)

ASSET_CONTENT_TYPES = {
    '.css': 'text/css; charset=utf-8',
    '.gif': 'image/gif',
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.png': 'image/png',
    '.webmanifest': 'application/manifest+json',
}

ASSET_FILES = {
    'app.js',
    'constants.js',
    'csv.js',
    'download.js',
    'format.js',
    'geo.js',
    'gps.js',
    'index.html',
    'ipso.js',
    'manifest.webmanifest',
    'map.js',
    'modes.js',
    'numpad.js',
    'palette.js',
    'parcel-locator.js',
    'session.js',
    'store.js',
    'strings.js',
    'style.css',
    'sw.js',
    'upload-flow.js',
    'upload.js',
    'version.js',
    'img/brand-logo.gif',
    'img/icon-192.png',
    'img/icon-512.png',
    'img/icon-512-maskable.png',
}

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_SESSION_ID_RE = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)
_BEARER_PREFIX = 'Bearer '
_UPLOAD_ATTEMPTS = defaultdict(deque)


class UploadValidationError(ValueError):
    pass


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    return _asset_response(request, 'index.html')


@require_GET
def asset(request: HttpRequest, asset_path: str) -> HttpResponse:
    if asset_path in {IPSO_REFERENCE_JSON, IPSO_TERRENI_GEOJSON, IPSO_UPLOAD_CONFIG_JS}:
        raise Http404
    return _asset_response(request, asset_path)


def _asset_response(request: HttpRequest, asset_path: str) -> HttpResponse:
    if asset_path not in ASSET_FILES:
        raise Http404
    path = _static_path(asset_path)
    content_type = ASSET_CONTENT_TYPES.get(Path(asset_path).suffix)
    if content_type is None:
        raise Http404
    return conditional_file_response(
        request, path, content_type=content_type, cache_control=CACHE_NO_CACHE,
    )


def _static_path(asset_path: str) -> Path:
    found = finders.find(f'ipso/pwa/{asset_path}')
    if not found:
        raise Http404
    return Path(found)


@require_GET
def reference_json(request: HttpRequest) -> HttpResponse:
    """Current-client reference bundle, generated from Abies data.

    The existing Ipso UI still consumes the legacy display fields
    (`common`, `compresa`, `particella`) while the integrated upload path uses
    the added canonical Abies IDs (`id`, `region_id`, `parcel_id`).
    """
    if not _upload_authorized(request):
        return _auth_error()
    sampling = _sampling_context()
    pai = _pai_context()
    payload = {
        FIELD_SCHEMA_VERSION: SCHEMA_VERSION,
        IPSO_REF_GENERATED_AT: django_timezone.now().astimezone(timezone.utc)
                       .isoformat(timespec='seconds').replace('+00:00', 'Z'),
        IPSO_REF_SPECIES: _species_rows(),
        IPSO_REF_PARCELS: _parcel_rows(),
        IPSO_REF_HYPSOMETRY: _ipsometrica(),
        IPSO_REF_SAMPLING: sampling,
        IPSO_REF_PAI: pai,
        IPSO_REF_WORK_PACKAGES: _work_packages(sampling),
    }
    payload[FIELD_REFERENCE_VERSION] = _reference_version(payload)
    response = _json_response(
        payload, content_type='application/json', cache_control=CACHE_NO_STORE,
    )
    return _authorized_data_response(response)


def _species_rows() -> list[dict]:
    return [
        {
            'id': sp.id,
            'common': sp.common_name,
            'latin': sp.latin_name,
            'sort_order': sp.sort_order,
            'density': float(sp.density),
        }
        for sp in Species.objects.filter(active=True).order_by('sort_order', 'common_name')
    ]


def _parcel_rows() -> list[dict]:
    parcels = sorted(
        Parcel.objects.select_related('region', 'eclass'),
        key=parcel_sort_key,
    )
    return [
        {
            FIELD_REGION_ID: p.region_id,
            'region_name': p.region.name,
            FIELD_PARCEL_ID: p.id,
            'compresa': p.region.name,
            'particella': p.name,
            FIELD_COPPICE: p.eclass.coppice,
        }
        for p in parcels
    ]


def _sampling_context() -> dict:
    surveys = list(
        Survey.objects
        .select_related('sample_grid')
        .order_by('name', 'id')
    )
    grid_ids = {s.sample_grid_id for s in surveys}
    areas = []
    if grid_ids:
        areas = list(
            SampleArea.objects
            .filter(sample_grid_id__in=grid_ids)
            .select_related('sample_grid', 'parcel__region', 'parcel__eclass')
        )
        areas.sort(key=lambda a: (
            a.sample_grid.name, parcel_sort_key(a.parcel),
            natural_sort_key(a.number), a.id,
        ))
    max_numbers = _sample_area_max_numbers_by_survey(
        [s.id for s in surveys], [a.id for a in areas],
    )
    return {
        IPSO_REF_SURVEYS: [
            {
                FIELD_SURVEY_ID: s.id,
                'name': s.name,
                FIELD_SAMPLE_GRID_ID: s.sample_grid_id,
                'sample_grid_name': s.sample_grid.name,
                IPSO_REF_SAMPLE_AREA_MAX_NUMBERS: max_numbers.get(s.id, {}),
            }
            for s in surveys
        ],
        IPSO_REF_SAMPLE_AREAS: [
            {
                FIELD_SAMPLE_AREA_ID: a.id,
                FIELD_SAMPLE_GRID_ID: a.sample_grid_id,
                FIELD_REGION_ID: a.parcel.region_id,
                FIELD_PARCEL_ID: a.parcel_id,
                'compresa': a.parcel.region.name,
                'particella': a.parcel.name,
                FIELD_NUMBER: a.number,
                FIELD_LAT: a.lat,
                FIELD_LON: a.lon,
                'r_m': a.r_m,
                FIELD_COPPICE: a.parcel.eclass.coppice,
            }
            for a in areas
        ],
    }


def _sample_area_max_numbers_by_survey(
        survey_ids: list[int], area_ids: list[int],
) -> dict[int, dict[str, int]]:
    if not survey_ids or not area_ids:
        return {}
    rows = (
        TreeSample.objects
        .filter(sample__survey_id__in=survey_ids, sample__sample_area_id__in=area_ids)
        .values('sample__survey_id', 'sample__sample_area_id')
        .annotate(max_number=Max(FIELD_NUMBER))
    )
    out: dict[int, dict[str, int]] = defaultdict(dict)
    for row in rows:
        max_number = row['max_number']
        if max_number is not None:
            out[row['sample__survey_id']][str(row['sample__sample_area_id'])] = max_number
    return dict(out)


def _pai_context() -> dict:
    rows = (
        TreePreserved.objects
        .select_related('tree__species', 'parcel__region')
        .order_by('parcel__region__name', 'parcel__name', FIELD_NUMBER, 'id')
    )
    return {
        IPSO_REF_PRESERVED_TREES: [
            {
                FIELD_TREE_PRESERVED_ID: p.id,
                FIELD_TREE_ID: p.tree_id,
                FIELD_REGION_ID: p.parcel.region_id,
                FIELD_PARCEL_ID: p.parcel_id,
                'compresa': p.parcel.region.name,
                'particella': p.parcel.name,
                FIELD_SPECIES_ID: p.tree.species_id,
                FIELD_NUMBER: p.number,
                FIELD_ESTIMATED_BIRTH_YEAR: p.tree.estimated_birth_year,
                FIELD_DATE: p.date.isoformat() if p.date else '',
                FIELD_D_CM: p.d_cm,
                FIELD_H_M: str(p.h_m) if p.h_m is not None else None,
                FIELD_H_MEASURED: p.h_measured,
                FIELD_LAT: p.lat,
                FIELD_LON: p.lon,
                FIELD_ACC_M: p.acc_m,
                FIELD_OPERATOR: p.operator,
                FIELD_NOTE: p.note,
                FIELD_COPPICE: p.tree.coppice,
            }
            for p in rows
        ],
    }


def _work_packages(sampling: dict) -> list[dict]:
    return [
        {
            'id': f"{IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX}{s[FIELD_SURVEY_ID]}",
            'kind': IPSO_WORK_PACKAGE_SAMPLING_SURVEY,
            'label': s['name'],
            FIELD_SURVEY_ID: s[FIELD_SURVEY_ID],
            'sample_grid_id': s['sample_grid_id'],
        }
        for s in sampling[IPSO_REF_SURVEYS]
    ]


def _ipsometrica() -> dict:
    active = HypsoParamSet.objects.active().first()
    if active is None:
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    params = (HypsoParam.objects
              .filter(param_set=active, func=HYPSO_FUNC_LN)
              .select_related('region', 'species')
              .order_by('region__name', 'species__common_name'))
    for p in params:
        out.setdefault(p.region.name, {})[p.species.common_name] = {
            'a': float(p.a),
            'b': float(p.b),
            FIELD_HYPSO_PARAM_SET_ID: active.id,
        }
    return out


def _reference_version(payload: dict) -> str:
    raw = json.dumps({
        key: payload[key]
        for key in IPSO_REFERENCE_VERSION_KEYS
    }, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]


@require_GET
def terreni_geojson(request: HttpRequest) -> HttpResponse:
    if not _upload_authorized(request):
        return _auth_error()
    path = Path(settings.GEO_DIR) / IPSO_TERRENI_GEOJSON
    if path.is_file():
        response = conditional_file_response(
            request, path, content_type='application/geo+json',
            cache_control=CACHE_NO_STORE,
        )
        return _authorized_data_response(response)
    response = _json_response(
        {'type': 'FeatureCollection', 'features': []},
        content_type='application/geo+json', cache_control=CACHE_NO_STORE,
    )
    return _authorized_data_response(response)


IPSO_INBOX_STATE_COL = '_ipso_state'

INBOX_COLUMNS = [
    ROW_ID, IPSO_INBOX_STATE_COL, S.IPSO_COL_RECEIVED, S.COL_DATE,
    S.IPSO_COL_MODE, S.IPSO_COL_OPERATOR, S.IPSO_COL_RECORDS,
    S.IPSO_COL_STATE, S.IPSO_COL_WORK_PACKAGE, S.IPSO_COL_TARGET,
    S.IPSO_COL_ERROR,
]
STATE_LABELS = {
    IpsoUploadState.RECEIVED: S.IPSO_STATE_RECEIVED,
    IpsoUploadState.IMPORTED: S.IPSO_STATE_IMPORTED,
    IpsoUploadState.REJECTED: S.IPSO_STATE_REJECTED,
    IpsoUploadState.CONFLICT: S.IPSO_STATE_CONFLICT,
}
MODE_LABELS = {
    IPSO_MODE_MARTELLATE: S.IPSO_MODE_MARTELLATE_LABEL,
    IPSO_MODE_SAMPLES: S.IPSO_MODE_SAMPLES_LABEL,
    IPSO_MODE_PAI: S.IPSO_MODE_PAI_LABEL,
}
REFERENCE_LABELS = {
    'legacy-converted': S.IPSO_REFERENCE_LEGACY_CONVERTED,
}
STAGED_UPLOAD_ARCHIVE_FILES = (
    IPSO_UPLOAD_FILE_JSON,
    IPSO_UPLOAD_FILE_SHA256,
    IPSO_UPLOAD_FILE_CSV,
)


@login_required
@require_GET
def inbox_data(request: HttpRequest) -> JsonResponse:
    uploads = IpsoUpload.objects.order_by('-received_at')
    payload = {
        COLUMNS: INBOX_COLUMNS,
        ROWS: [_inbox_row(u) for u in uploads],
        PENDING_COUNT: uploads.filter(state=IpsoUploadState.RECEIVED).count(),
    }
    return _api_json(payload)


@login_required
@require_GET
def upload_detail(request: HttpRequest, upload_id: int) -> JsonResponse:
    upload = _get_upload(upload_id)
    payload, file_error = _read_staged_payload(upload)
    session = payload.get(SESSION, {}) if payload else {}
    records = payload.get(RECORDS, []) if payload else []
    targets, suggested_target_id = _upload_targets(upload)
    return _api_json({
        UPLOAD: _upload_metadata(upload),
        SESSION: session,
        RECORDS: _preview_records(records),
        RECORD_COUNT: len(records),
        FILE_ERROR: file_error,
        TARGETS: targets,
        SUGGESTED_TARGET_ID: suggested_target_id,
    })


@login_required
@require_writer
@require_POST
def reject_upload(request: HttpRequest, upload_id: int) -> JsonResponse:
    with transaction.atomic():
        upload = _get_upload(upload_id, for_update=True)
        if upload.state == IpsoUploadState.IMPORTED:
            return validation_error([S.IPSO_ERR_IMPORTED_CANNOT_REJECT])
        reason = _reject_reason(request)
        updated = (IpsoUpload.objects
                   .filter(id=upload.id)
                   .exclude(state=IpsoUploadState.IMPORTED)
                   .update(state=IpsoUploadState.REJECTED, error_summary=reason))
        if not updated:
            return validation_error([S.IPSO_ERR_IMPORTED_CANNOT_REJECT])
        upload.state = IpsoUploadState.REJECTED
        upload.error_summary = reason
        data = _upload_metadata(upload)
    return success_response(request, None, extra={UPLOAD: data})


@login_required
@require_admin
@require_GET
def download_upload(request: HttpRequest, upload_id: int) -> HttpResponse:
    upload = _get_upload(upload_id)
    files = _staged_upload_files(upload)
    if not files:
        raise Http404
    return csv_io.zip_response(files, _upload_archive_filename(upload))


@login_required
@require_admin
@require_POST
def delete_upload(request: HttpRequest, upload_id: int) -> JsonResponse:
    with transaction.atomic():
        upload = _get_upload(upload_id, for_update=True)
        inbox_path = Path(upload.inbox_path)
        upload.delete()
    _remove_staged_upload_files(inbox_path)
    return success_response(
        request, None,
        data_id=DATA_ID_IPSO_UPLOADS, row_id=upload_id,
        deletes=[row_delete(DATA_ID_IPSO_UPLOADS, upload_id)],
    )


@login_required
@require_admin
@require_POST
def update_upload_mode(request: HttpRequest, upload_id: int) -> JsonResponse:
    try:
        body = _request_json(request)
        mode = _str(body, FIELD_MODE)
    except UploadValidationError as e:
        return validation_error([str(e)])
    if mode not in ALLOWED_UPLOAD_MODES:
        return validation_error([S.IPSO_ERR_MODE_UNSUPPORTED])

    with transaction.atomic():
        upload = _get_upload(upload_id, for_update=True)
        if upload.state == IpsoUploadState.IMPORTED:
            return validation_error([S.IPSO_ERR_IMPORTED_CANNOT_EDIT_MODE])
        payload, file_error = _read_staged_payload(upload)
        if file_error:
            return validation_error([file_error])
        session = payload.get(SESSION) if isinstance(payload, dict) else None
        if not isinstance(session, dict):
            return validation_error([S.IPSO_ERR_FIELD_OBJECT.format(SESSION)])
        session[FIELD_MODE] = mode
        checksum = _payload_checksum(payload)
        ipso_staging.write_payload_files(Path(upload.inbox_path), payload, checksum)
        upload.mode = mode
        upload.checksum = checksum
        upload.error_summary = ''
        upload.save(update_fields=['mode', 'checksum', 'error_summary'])
        data = _upload_metadata(upload)
    return success_response(request, body, extra={UPLOAD: data})


@login_required
@require_writer
@require_POST
def import_martellate_upload(request: HttpRequest, upload_id: int) -> JsonResponse:
    try:
        body = _request_json(request)
        item_id = _int(body, FIELD_HARVEST_PLAN_ITEM_ID)
    except UploadValidationError as e:
        return validation_error([str(e)])

    with transaction.atomic():
        upload = _get_upload(upload_id, for_update=True)
        if upload.mode != IPSO_MODE_MARTELLATE:
            return validation_error([S.IPSO_ERR_MODE_UNSUPPORTED])
        if upload.state != IpsoUploadState.RECEIVED:
            return validation_error([S.IPSO_ERR_UPLOAD_NOT_RECEIVED])
        item = (HarvestPlanItem.objects
                .select_for_update()
                .filter(id=item_id).first())
        if item is None:
            return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
        if item.state == HarvestPlanItemState.CLOSED:
            return validation_error([S.ERR_MARK_ITEM_CLOSED])
        if not _is_valid_martellate_target(item):
            return validation_error([S.IPSO_ERR_INVALID_MARTELLATE_TARGET])

        payload, file_error = _read_staged_payload(upload)
        if file_error:
            return _upload_validation_error(upload, [file_error])
        rows, errors = _martellate_import_rows(upload, payload, item)
        if errors:
            return _upload_validation_error(upload, errors)
        duplicate_errors = mark_number_duplicate_errors(item.id, rows)
        if duplicate_errors:
            return _upload_validation_error(upload, duplicate_errors)

        if not _claim_upload_import(
                upload, request.user, IPSO_TARGET_HARVEST_PLAN_ITEM, item.id):
            return validation_error([S.IPSO_ERR_UPLOAD_NOT_RECEIVED])
        result = import_mark_rows(item, rows)
        if result.errors:
            return _upload_validation_error(upload, result.errors)
        data = _upload_metadata(upload)
    return success_response(request, body, extra={
        IMPORTED: result.imported,
        SKIPPED_DUPLICATES: result.skipped_duplicates,
        UPLOAD: data,
    })


@login_required
@require_writer
@require_POST
def import_samples_upload(request: HttpRequest, upload_id: int) -> JsonResponse:
    try:
        body = _request_json(request)
        survey_id = _int(body, FIELD_SURVEY_ID)
    except UploadValidationError as e:
        return validation_error([str(e)])

    with transaction.atomic():
        upload = _get_upload(upload_id, for_update=True)
        if upload.mode != IPSO_MODE_SAMPLES:
            return validation_error([S.IPSO_ERR_MODE_UNSUPPORTED])
        if upload.state != IpsoUploadState.RECEIVED:
            return validation_error([S.IPSO_ERR_UPLOAD_NOT_RECEIVED])
        survey = (Survey.objects
                  .select_for_update()
                  .filter(id=survey_id).first())
        if survey is None:
            return validation_error([S.IPSO_ERR_INVALID_SAMPLES_TARGET])
        source_grid_id = _source_survey_grid_id(upload.work_package_id)
        if source_grid_id is not None and survey.sample_grid_id != source_grid_id:
            return validation_error([S.IPSO_ERR_SAMPLES_TARGET_GRID_MISMATCH])

        payload, file_error = _read_staged_payload(upload)
        if file_error:
            return _upload_validation_error(upload, [file_error])
        rows, errors = sample_import_rows(payload, survey)
        if errors:
            return _upload_validation_error(upload, errors)

        if not _claim_upload_import(upload, request.user, IPSO_TARGET_SURVEY, survey.id):
            return validation_error([S.IPSO_ERR_UPLOAD_NOT_RECEIVED])
        result = csv_trees.apply(survey, rows)
        data = _upload_metadata(upload)
    return success_response(request, body, extra={
        IMPORTED: result['n_trees'],
        UPLOAD: data,
    })


@login_required
@require_writer
@require_POST
def import_pai_upload(request: HttpRequest, upload_id: int) -> JsonResponse:
    try:
        body = _request_json(request)
    except UploadValidationError as e:
        return validation_error([str(e)])

    try:
        with transaction.atomic():
            upload = _get_upload(upload_id, for_update=True)
            if upload.mode != IPSO_MODE_PAI:
                return validation_error([S.IPSO_ERR_MODE_UNSUPPORTED])
            if upload.state != IpsoUploadState.RECEIVED:
                return validation_error([S.IPSO_ERR_UPLOAD_NOT_RECEIVED])

            payload, file_error = _read_staged_payload(upload)
            if file_error:
                return _upload_validation_error(upload, [file_error])
            rows, errors = pai_import_rows(payload)
            if errors:
                return _upload_validation_error(upload, errors)

            if not _claim_upload_import(upload, request.user, IPSO_TARGET_PAI, None):
                return validation_error([S.IPSO_ERR_UPLOAD_NOT_RECEIVED])
            imported = apply_pai_rows(rows)
            data = _upload_metadata(upload)
    except IntegrityError:
        upload = _get_upload(upload_id)
        return _upload_validation_error(upload, [S.IPSO_ERR_IMPORT_PAI_NUMBER_CONFLICT])
    return success_response(request, body, extra={
        IMPORTED: imported,
        UPLOAD: data,
    })


@csrf_exempt
@require_POST
def upload_session(request: HttpRequest) -> JsonResponse:
    if _upload_rate_limited(request):
        return _api_json({OK: False, ERROR: IPSO_ERROR_RATE_LIMITED}, status=429)
    if _upload_too_large(request):
        return _api_json({
            OK: False,
            ERROR: IPSO_ERROR_TOO_LARGE,
            DETAIL: S.IPSO_ERR_UPLOAD_TOO_LARGE,
        }, status=413)
    if not _upload_authorized(request):
        return _auth_error()
    try:
        payload = _request_json(request)
        normalized, csv_text = _validate_upload_payload(payload, request)
    except UploadValidationError as e:
        return _api_json({OK: False, ERROR: IPSO_ERROR_INVALID_PAYLOAD, DETAIL: str(e)}, status=422)

    checksum = _payload_checksum(normalized)
    session_id = normalized[SESSION][FIELD_SESSION_ID]
    try:
        with transaction.atomic():
            inbox_path = _upload_inbox_path(session_id)
            upload = IpsoUpload.objects.create(
                **ipso_staging.upload_model_fields(normalized, checksum, inbox_path)
            )
            _write_upload_files(inbox_path, normalized, checksum, csv_text)
    except IntegrityError:
        return _duplicate_upload_response(session_id, checksum)

    return _api_json({
        OK: True,
        STORED_AS: upload.inbox_path,
        DUPLICATE: False,
    })


def _duplicate_upload_response(session_id: str, checksum: str) -> JsonResponse:
    existing = IpsoUpload.objects.filter(session_id=session_id).first()
    if existing is None:
        return _api_json({OK: False, ERROR: IPSO_ERROR_CONFLICT}, status=409)
    if hmac.compare_digest(existing.checksum, checksum):
        return _api_json({
            OK: True,
            STORED_AS: existing.inbox_path,
            DUPLICATE: True,
        })
    existing.state = IpsoUploadState.CONFLICT
    existing.error_summary = S.IPSO_ERR_DUPLICATE_SESSION_CONTENT
    existing.save(update_fields=['state', 'error_summary'])
    return _api_json({OK: False, ERROR: IPSO_ERROR_CONFLICT}, status=409)


def _get_upload(upload_id: int, *, for_update: bool = False) -> IpsoUpload:
    qs = IpsoUpload.objects
    if for_update:
        qs = qs.select_for_update()
    try:
        return qs.get(id=upload_id)
    except IpsoUpload.DoesNotExist as e:
        raise Http404 from e


def _inbox_row(upload: IpsoUpload) -> list:
    return [
        upload.id,
        upload.state,
        _format_dt(upload.received_at),
        _upload_record_date(upload),
        _mode_label(upload.mode),
        upload.operator,
        upload.record_count,
        STATE_LABELS.get(upload.state, upload.state),
        _work_package_label(upload),
        _target_label(upload),
        upload.error_summary,
    ]


def _upload_metadata(upload: IpsoUpload) -> dict:
    return {
        'id': upload.id,
        FIELD_SESSION_ID: upload.session_id,
        FIELD_MODE: upload.mode,
        'mode_label': _mode_label(upload.mode),
        FIELD_SCHEMA_VERSION: upload.schema_version,
        FIELD_REFERENCE_VERSION: upload.reference_version,
        'reference_version_label': _reference_label(upload.reference_version),
        FIELD_WORK_PACKAGE_ID: upload.work_package_id,
        FIELD_WORK_PACKAGE_LABEL: _work_package_label(upload),
        FIELD_OPERATOR: upload.operator,
        'record_date': _upload_record_date(upload),
        RECORD_COUNT: upload.record_count,
        'checksum': upload.checksum,
        FIELD_STATE: upload.state,
        'state_label': STATE_LABELS.get(upload.state, upload.state),
        'received_at': _format_dt(upload.received_at),
        'imported_at': _format_dt(upload.imported_at),
        'target_type': upload.target_type,
        'target_id': upload.target_id,
        'target_label': _target_label(upload),
        'error_summary': upload.error_summary,
    }


def _staged_upload_files(upload: IpsoUpload) -> list[tuple[str, bytes]]:
    root = Path(upload.inbox_path)
    if not root.is_dir():
        return []
    files = []
    for name in STAGED_UPLOAD_ARCHIVE_FILES:
        path = root / name
        if path.is_file():
            files.append((name, path.read_bytes()))
    return files


def _upload_archive_filename(upload: IpsoUpload) -> str:
    return f'ipso-upload-{upload.session_id}.zip'


def _remove_staged_upload_files(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _work_package_label(upload: IpsoUpload) -> str:
    raw = (upload.work_package_id or '').strip()
    if raw.startswith(IPSO_WORK_PACKAGE_HARVEST_PREFIX):
        item_id = raw.split(':', 1)[1]
        if item_id.isdigit():
            item = (HarvestPlanItem.objects
                    .select_related('harvest_plan', 'parcel__region', 'parcel__eclass', 'region')
                    .filter(id=int(item_id)).first())
            if item is not None:
                return _harvest_item_label(item)
    if raw.startswith(IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX):
        survey_id = raw.split(':', 1)[1]
        if survey_id.isdigit():
            survey = (Survey.objects
                      .select_related('sample_grid')
                      .filter(id=int(survey_id)).first())
            if survey is not None:
                return _survey_label(survey)
    return ''


def _target_label(upload: IpsoUpload) -> str:
    if upload.target_type == IPSO_TARGET_HARVEST_PLAN_ITEM and upload.target_id:
        item = (HarvestPlanItem.objects
                .select_related('harvest_plan', 'parcel__region', 'parcel__eclass', 'region')
                .filter(id=upload.target_id).first())
        if item is not None:
            return _harvest_item_label(item)
    if upload.target_type == IPSO_TARGET_SURVEY and upload.target_id:
        survey = Survey.objects.filter(id=upload.target_id).first()
        if survey is not None:
            return _survey_label(survey)
    if upload.target_type == IPSO_TARGET_PAI:
        return S.IPSO_TARGET_PAI_LABEL
    if upload.target_type and upload.target_id:
        return f'{upload.target_type}:{upload.target_id}'
    return ''


def _format_dt(value) -> str:
    if not value:
        return ''
    return value.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def _reference_label(reference_version: str) -> str:
    raw = (reference_version or '').strip()
    return REFERENCE_LABELS.get(raw, raw)


def _upload_record_date(upload: IpsoUpload) -> str:
    return upload.record_date or ''


def _read_staged_payload(upload: IpsoUpload) -> tuple[dict, str]:
    path = Path(upload.inbox_path) / IPSO_UPLOAD_FILE_JSON
    try:
        return json.loads(path.read_text(encoding='utf-8')), ''
    except FileNotFoundError:
        return {}, S.IPSO_ERR_UPLOAD_JSON_MISSING
    except json.JSONDecodeError:
        return {}, S.IPSO_ERR_UPLOAD_JSON_INVALID


def _preview_sequence(value, fallback: int):
    if type(value) is int:
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        if text:
            return text
    return fallback


def _preview_decimal(value):
    if value is None or value == '':
        return value
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return value
    return float(number) if number.is_finite() else value


def _preview_records(records: list) -> list[dict]:
    if not isinstance(records, list):
        return []
    species_ids = {
        r.get(FIELD_SPECIES_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SPECIES_ID)) is int
    }
    parcel_ids = {
        r.get(FIELD_PARCEL_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_PARCEL_ID)) is int
    }
    sample_area_ids = {
        r.get(FIELD_SAMPLE_AREA_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SAMPLE_AREA_ID)) is int
    }
    species = {
        sp.id: sp.common_name
        for sp in Species.objects.filter(id__in=species_ids)
    }
    parcels = {
        p.id: f'{p.region.name} {p.name}'
        for p in Parcel.objects.filter(id__in=parcel_ids).select_related('region')
    }
    sample_areas = {
        a.id: a.number
        for a in SampleArea.objects.filter(id__in=sample_area_ids)
    }
    out = []
    for i, row in enumerate(records[:500], start=1):
        if not isinstance(row, dict):
            continue
        out.append({
            'seq': _preview_sequence(row.get(FIELD_CLIENT_RECORD_ID), i),
            FIELD_DATE: row.get(FIELD_DATE, ''),
            'parcel': parcels.get(row.get(FIELD_PARCEL_ID), str(row.get(FIELD_PARCEL_ID, ''))),
            FIELD_SAMPLE_AREA_ID: sample_areas.get(row.get(FIELD_SAMPLE_AREA_ID), ''),
            'species': species.get(row.get(FIELD_SPECIES_ID), str(row.get(FIELD_SPECIES_ID, ''))),
            FIELD_NUMBER: row.get(FIELD_NUMBER),
            FIELD_D_CM: row.get(FIELD_D_CM),
            FIELD_H_M: _preview_decimal(row.get(FIELD_H_M)),
            FIELD_H_MEASURED: bool(row.get(FIELD_H_MEASURED)),
            FIELD_LAT: row.get(FIELD_LAT),
            FIELD_LON: row.get(FIELD_LON),
            FIELD_ACC_M: row.get(FIELD_ACC_M),
        })
    return out


def _reject_reason(request: HttpRequest) -> str:
    try:
        body = json.loads(request.body.decode('utf-8') or '{}')
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = {}
    reason = body.get('reason', '') if isinstance(body, dict) else ''
    return reason.strip() or S.IPSO_REJECT_DEFAULT_REASON


def _upload_targets(upload: IpsoUpload) -> tuple[list[dict], int | None]:
    if upload.mode == IPSO_MODE_MARTELLATE:
        return _martellate_targets(), _suggested_harvest_item_id(upload.work_package_id)
    if upload.mode == IPSO_MODE_SAMPLES:
        return _survey_targets(), _suggested_survey_id(upload.work_package_id)
    return [], None


def _martellate_targets() -> list[dict]:
    items = (HarvestPlanItem.objects
             .select_related('harvest_plan', 'parcel__region', 'parcel__eclass', 'region')
             .exclude(state=HarvestPlanItemState.CLOSED)
             .order_by('harvest_plan__name', 'year_planned', 'id'))
    out = []
    for item in items:
        if _is_valid_martellate_target(item):
            out.append({'id': item.id, 'label': _harvest_item_label(item)})
    return out


def _is_valid_martellate_target(item: HarvestPlanItem) -> bool:
    if item.state == HarvestPlanItemState.CLOSED:
        return False
    return not (item.parcel_id and item.parcel.eclass.coppice)


def _harvest_item_label(item: HarvestPlanItem) -> str:
    if item.parcel_id:
        area = f'{item.parcel.region.name} {item.parcel.name}'
    elif item.region_id:
        area = item.region.name
    else:
        area = '-'
    state = HarvestPlanItemState(item.state).label
    base = f'{item.harvest_plan.name} - {area} - {item.year_planned} ({state})'
    note = _short_target_note(item.note)
    return f'{base} - {note}' if note else base


def _short_target_note(note: str | None, limit: int = 20) -> str:
    text = ' '.join((note or '').split())
    if len(text) <= limit:
        return text
    return f'{text[:limit]}…'


def _suggested_harvest_item_id(work_package_id: str) -> int | None:
    raw = (work_package_id or '').strip()
    if raw.startswith(IPSO_WORK_PACKAGE_HARVEST_PREFIX):
        raw = raw.split(':', 1)[1]
    if not raw.isdigit():
        return None
    item_id = int(raw)
    item = (HarvestPlanItem.objects
            .select_related('parcel__eclass')
            .filter(id=item_id).first())
    if item is not None and _is_valid_martellate_target(item):
        return item_id
    return None


def _survey_targets() -> list[dict]:
    surveys = (Survey.objects
               .select_related('sample_grid')
               .order_by('name', 'id'))
    return [{'id': survey.id, 'label': _survey_label(survey)} for survey in surveys]


def _survey_label(survey: Survey) -> str:
    return f'{survey.name} - {survey.sample_grid.name}'


def _suggested_survey_id(work_package_id: str) -> int | None:
    raw = (work_package_id or '').strip()
    if raw.startswith(IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX):
        raw = raw.split(':', 1)[1]
    if not raw.isdigit():
        return None
    survey_id = int(raw)
    if Survey.objects.filter(id=survey_id).exists():
        return survey_id
    return None


def _source_survey_grid_id(work_package_id: str) -> int | None:
    survey_id = _suggested_survey_id(work_package_id)
    if survey_id is None:
        return None
    return (Survey.objects
            .filter(id=survey_id)
            .values_list('sample_grid_id', flat=True)
            .first())


def _upload_validation_error(upload: IpsoUpload, errors: list[str]) -> JsonResponse:
    upload.error_summary = errors[0] if errors else S.ERROR_GENERIC
    upload.save(update_fields=['error_summary'])
    return validation_error(errors)


def _claim_upload_import(
        upload: IpsoUpload, user, target_type: str, target_id: int | None,
) -> bool:
    imported_at = django_timezone.now()
    claimed = (IpsoUpload.objects
               .filter(id=upload.id, state=IpsoUploadState.RECEIVED)
               .update(
                   state=IpsoUploadState.IMPORTED,
                   imported_at=imported_at,
                   imported_by_id=user.id,
                   target_type=target_type,
                   target_id=target_id,
                   error_summary='',
               ))
    if not claimed:
        return False
    upload.state = IpsoUploadState.IMPORTED
    upload.imported_at = imported_at
    upload.imported_by = user
    upload.target_type = target_type
    upload.target_id = target_id
    upload.error_summary = ''
    return True


def _staged_mark_number(record: dict, index: int) -> tuple[int | None, str | None]:
    raw = record.get(FIELD_NUMBER)
    if raw is None:
        return None, None
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return None, S.IPSO_ERR_RECORD_NUMBER_INVALID.format(index)
    if number <= 0:
        return None, S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(index)
    return number, None


def _martellate_import_rows(
        upload: IpsoUpload, payload: dict, item: HarvestPlanItem,
) -> tuple[list[MarkImportRow], list[str]]:
    session = payload.get(SESSION, {}) if isinstance(payload, dict) else {}
    records = payload.get(RECORDS, []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return [], [S.IPSO_ERR_IMPORT_RECORDS_ARRAY]

    species_ids = {
        r.get(FIELD_SPECIES_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SPECIES_ID)) is int
    }
    parcel_ids = {
        r.get(FIELD_PARCEL_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_PARCEL_ID)) is int
    }
    sample_area_ids = {
        r.get(FIELD_SAMPLE_AREA_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SAMPLE_AREA_ID)) is int
    }
    species = {
        sp.id: sp
        for sp in Species.objects.filter(id__in=species_ids)
    }
    parcels = {
        p.id: p
        for p in Parcel.objects.filter(id__in=parcel_ids).select_related('region', 'eclass')
    }
    item_region = item.region or (item.parcel.region if item.parcel else None)
    session_operator = (
        (session.get(FIELD_OPERATOR) or '').strip()
        if isinstance(session, dict) else ''
    )

    rows = []
    errors = []
    for i, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(S.IPSO_ERR_IMPORT_RECORD_INVALID.format(i))
            continue
        parcel = parcels.get(record.get(FIELD_PARCEL_ID))
        if parcel is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_PARCEL_NOT_FOUND.format(i))
            continue
        if parcel.eclass.coppice and item.parcel_id is not None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_COPPICE_MARTELLATE.format(i))
            continue
        if item_region and parcel.region_id != item_region.id:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_MARK_TARGET_MISMATCH.format(i))
            continue
        if item.parcel_id is not None and parcel.id != item.parcel_id:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_MARK_TARGET_MISMATCH.format(i))
            continue
        sp = species.get(record.get(FIELD_SPECIES_ID))
        if sp is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SPECIES_NOT_FOUND.format(i))
            continue
        measurements = record_measurements(record)
        if measurements is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_DH_DATE_INVALID.format(i))
            continue
        number, number_error = _staged_mark_number(record, i)
        if number_error:
            errors.append(number_error)
            continue
        operator = (record.get(FIELD_OPERATOR) or session_operator).strip()
        rows.append(MarkImportRow(
            date=measurements.date,
            parcel=parcel,
            species=sp,
            number=number,
            d_cm=measurements.d_cm,
            h_m=measurements.h_m,
            h_measured=bool(record.get(FIELD_H_MEASURED)),
            lat=record.get(FIELD_LAT),
            lon=record.get(FIELD_LON),
            acc_m=record.get(FIELD_ACC_M),
            operator=operator,
            fingerprint=ipso_mark_fingerprint(upload.session_id, record),
        ))
    return rows, errors


def _configured_ipso_secret() -> str:
    return str(getattr(settings, 'IPSO_SECRET', '') or '').strip()


def _upload_authorized(request: HttpRequest) -> bool:
    expected = _configured_ipso_secret()
    if not expected:
        return False
    header = request.headers.get('Authorization', '')
    if not header.startswith(_BEARER_PREFIX):
        return False
    return hmac.compare_digest(header[len(_BEARER_PREFIX):], expected)


def _auth_error() -> JsonResponse:
    response = _api_json({OK: False, ERROR: IPSO_ERROR_AUTH}, status=401)
    response['WWW-Authenticate'] = 'Bearer'
    return response


def _authorized_data_response(response: HttpResponse) -> HttpResponse:
    patch_vary_headers(response, ['Authorization'])
    return response


def _upload_rate_limited(request: HttpRequest) -> bool:
    limit = _setting_int('IPSO_UPLOAD_RATE_LIMIT', 60)
    window_s = _setting_int('IPSO_UPLOAD_RATE_WINDOW_S', 60)
    if limit <= 0 or window_s <= 0:
        return False
    now = time.monotonic()
    cutoff = now - window_s
    attempts = _UPLOAD_ATTEMPTS[_upload_rate_key(request)]
    while attempts and attempts[0] <= cutoff:
        attempts.popleft()
    if len(attempts) >= limit:
        return True
    attempts.append(now)
    return False


def _upload_rate_key(request: HttpRequest) -> str:
    remote_addr = request.META.get('REMOTE_ADDR') or ''
    forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR') or ''
    if forwarded_for and _request_from_trusted_proxy(remote_addr):
        client_ip = forwarded_for.split(',', 1)[0].strip()
        try:
            return str(ip_address(client_ip))
        except ValueError:
            pass
    return remote_addr or 'unknown'


def _request_from_trusted_proxy(remote_addr: str) -> bool:
    try:
        addr = ip_address(remote_addr)
    except ValueError:
        return False
    for proxy in getattr(settings, 'IPSO_UPLOAD_TRUSTED_PROXIES', ()):
        try:
            if addr in ip_network(str(proxy), strict=False):
                return True
        except ValueError:
            continue
    return False


def _upload_too_large(request: HttpRequest) -> bool:
    max_bytes = _setting_int('IPSO_UPLOAD_MAX_BYTES', 2 * 1024 * 1024)
    if max_bytes <= 0:
        return False
    content_length = request.META.get('CONTENT_LENGTH')
    if content_length:
        try:
            if int(content_length) > max_bytes:
                return True
        except ValueError:
            pass
    try:
        return len(request.body) > max_bytes
    except RequestDataTooBig:
        return True


def _setting_int(name: str, default: int) -> int:
    try:
        return int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _request_json(request: HttpRequest) -> dict:
    try:
        payload = json.loads(request.body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise UploadValidationError(S.IPSO_ERR_JSON_MALFORMED) from e
    if not isinstance(payload, dict):
        raise UploadValidationError(S.IPSO_ERR_PAYLOAD_OBJECT)
    return payload


def _validate_upload_payload(payload: dict, request: HttpRequest) -> tuple[dict, str | None]:
    session = _dict(payload, SESSION)
    records = _list(payload, RECORDS)
    if not records:
        raise UploadValidationError(S.IPSO_ERR_RECORDS_EMPTY)
    max_records = _setting_int('IPSO_UPLOAD_MAX_RECORDS', 500)
    if max_records > 0 and len(records) > max_records:
        raise UploadValidationError(S.IPSO_ERR_RECORDS_TOO_MANY.format(max_records))
    csv_text = payload.get(FIELD_CSV_TEXT)
    if csv_text is not None and not isinstance(csv_text, str):
        raise UploadValidationError(S.IPSO_ERR_CSV_TEXT_STRING)

    normalized_session = _normalize_session(session)
    header_id = request.headers.get('X-Ipso-Session-Id', '')
    if header_id and header_id != normalized_session[FIELD_SESSION_ID]:
        raise UploadValidationError(S.IPSO_ERR_HEADER_SESSION_MISMATCH)

    normalized_records = [
        _normalize_record(normalized_session[FIELD_MODE], i, row)
        for i, row in enumerate(records, start=1)
    ]
    _validate_record_ids(normalized_session[FIELD_MODE], normalized_records)
    return {SESSION: normalized_session, RECORDS: normalized_records}, csv_text


def _normalize_session(session: dict) -> dict:
    session_id = _str(session, FIELD_SESSION_ID)
    if not _SESSION_ID_RE.match(session_id):
        raise UploadValidationError(S.IPSO_ERR_SESSION_ID_UUID)
    mode = _str(session, FIELD_MODE)
    if mode not in ALLOWED_UPLOAD_MODES:
        raise UploadValidationError(S.IPSO_ERR_MODE_UNSUPPORTED)
    schema_version = _int(session, FIELD_SCHEMA_VERSION)
    if schema_version != UPLOAD_SCHEMA_VERSION:
        raise UploadValidationError(S.IPSO_ERR_SCHEMA_VERSION_UNSUPPORTED)
    normalized = {
        FIELD_SESSION_ID: session_id,
        FIELD_MODE: mode,
        FIELD_SCHEMA_VERSION: schema_version,
        FIELD_REFERENCE_VERSION: _opt_str(session, FIELD_REFERENCE_VERSION),
        FIELD_WORK_PACKAGE_ID: _opt_str(session, FIELD_WORK_PACKAGE_ID),
        FIELD_OPERATOR: _opt_str(session, FIELD_OPERATOR),
        FIELD_CREATED_AT: _opt_str(session, FIELD_CREATED_AT),
        FIELD_COMPLETED_AT: _opt_str(session, FIELD_COMPLETED_AT),
        FIELD_DAMAGED: _bool(session, FIELD_DAMAGED),
    }
    region_id = session.get(FIELD_REGION_ID)
    if region_id is not None:
        normalized[FIELD_REGION_ID] = _int(session, FIELD_REGION_ID)
    return normalized


def _normalize_record(mode: str, index: int, row: object) -> dict:
    if not isinstance(row, dict):
        raise UploadValidationError(S.IPSO_ERR_RECORD_OBJECT.format(index))
    date = _str(row, FIELD_DATE)
    if not _DATE_RE.match(date):
        raise UploadValidationError(S.IPSO_ERR_RECORD_DATE_INVALID.format(index))

    d_cm = _int(row, FIELD_D_CM)
    if d_cm is not None and d_cm <= 0:
        raise UploadValidationError(S.IPSO_ERR_RECORD_D_CM_POSITIVE.format(index))

    h_m = _decimal(row, FIELD_H_M)
    if h_m is not None and h_m <= 0:
        raise UploadValidationError(S.IPSO_ERR_RECORD_H_M_POSITIVE.format(index))

    number = _opt_int(row, FIELD_NUMBER)
    if number is not None and number <= 0:
        raise UploadValidationError(S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(index))
    if mode in {IPSO_MODE_SAMPLES, IPSO_MODE_PAI} and number is None:
        raise UploadValidationError(S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(index))

    sample_area_id = _opt_int(row, FIELD_SAMPLE_AREA_ID)
    if mode == IPSO_MODE_SAMPLES and sample_area_id is None:
        raise UploadValidationError(S.IPSO_ERR_RECORD_SAMPLE_AREA_REQUIRED.format(index))

    hypso_param_set_id = _opt_int(row, FIELD_HYPSO_PARAM_SET_ID)
    if mode != IPSO_MODE_MARTELLATE and hypso_param_set_id is not None:
        raise UploadValidationError(
            S.IPSO_ERR_RECORD_HYPSO_ONLY_MARTELLATE.format(index)
        )

    normalized = {
        FIELD_CLIENT_RECORD_ID: _str(row, FIELD_CLIENT_RECORD_ID),
        FIELD_DATE: date,
        FIELD_REGION_ID: _int(row, FIELD_REGION_ID),
        FIELD_PARCEL_ID: _int(row, FIELD_PARCEL_ID),
        FIELD_SPECIES_ID: _int(row, FIELD_SPECIES_ID),
        FIELD_NUMBER: number,
        FIELD_D_CM: d_cm,
        FIELD_H_M: format(h_m, 'f') if h_m is not None else None,
        FIELD_H_MEASURED: _bool(row, FIELD_H_MEASURED),
        FIELD_HYPSO_PARAM_SET_ID: hypso_param_set_id,
        FIELD_LAT: _opt_coord_float(row, FIELD_LAT),
        FIELD_LON: _opt_coord_float(row, FIELD_LON),
        FIELD_ACC_M: _opt_int(row, FIELD_ACC_M),
    }
    if mode == IPSO_MODE_MARTELLATE:
        normalized[FIELD_OPERATOR] = _opt_str(row, FIELD_OPERATOR)
    if mode == IPSO_MODE_SAMPLES:
        pressler_coeff = _opt_decimal(row, FIELD_PRESSLER_COEFF) or PRESSLER_DEFAULT
        shoot = _opt_int(row, FIELD_SHOOT) or 0
        l10_mm = _opt_int(row, FIELD_L10_MM) or 0
        if pressler_coeff <= 0 or shoot < 0 or l10_mm < 0:
            raise UploadValidationError(S.IPSO_ERR_RECORD_SAMPLE_FIELDS_INVALID.format(index))
        normalized.update({
            FIELD_SAMPLE_AREA_ID: sample_area_id,
            FIELD_COPPICE: _opt_bool(row, FIELD_COPPICE),
            FIELD_SHOOT: shoot,
            FIELD_STANDARD: _opt_bool(row, FIELD_STANDARD) or False,
            FIELD_L10_MM: l10_mm,
            FIELD_PRESSLER_COEFF: format(pressler_coeff, 'f'),
            FIELD_PRESERVED: _opt_bool(row, FIELD_PRESERVED) or False,
        })
    elif mode == IPSO_MODE_PAI:
        normalized.update({
            FIELD_ESTIMATED_BIRTH_YEAR: _opt_int(row, FIELD_ESTIMATED_BIRTH_YEAR),
            FIELD_OPERATOR: _opt_str(row, FIELD_OPERATOR),
            FIELD_NOTE: _opt_str(row, FIELD_NOTE),
        })
    return normalized


def _validate_record_ids(mode: str, records: list[dict]) -> None:
    species_ids = {r[FIELD_SPECIES_ID] for r in records}
    parcel_ids = {r[FIELD_PARCEL_ID] for r in records}
    sample_area_ids = {
        r[FIELD_SAMPLE_AREA_ID] for r in records
        if mode == IPSO_MODE_SAMPLES and r.get(FIELD_SAMPLE_AREA_ID) is not None
    }
    hypso_ids = {
        r[FIELD_HYPSO_PARAM_SET_ID] for r in records
        if r[FIELD_HYPSO_PARAM_SET_ID] is not None
    }
    valid_species = set(Species.objects.filter(id__in=species_ids).values_list('id', flat=True))
    parcels = {
        p.id: p.region_id
        for p in Parcel.objects.filter(id__in=parcel_ids).select_related('region')
    }
    sample_areas = {
        area.id: area.parcel_id
        for area in SampleArea.objects.filter(id__in=sample_area_ids)
    }
    missing_species = species_ids - valid_species
    if missing_species:
        raise UploadValidationError(S.IPSO_ERR_UNKNOWN_SPECIES_ID.format(min(missing_species)))
    missing_parcels = parcel_ids - set(parcels)
    if missing_parcels:
        raise UploadValidationError(S.IPSO_ERR_UNKNOWN_PARCEL_ID.format(min(missing_parcels)))
    missing_sample_areas = sample_area_ids - set(sample_areas)
    if missing_sample_areas:
        raise UploadValidationError(S.IPSO_ERR_UNKNOWN_SAMPLE_AREA_ID.format(min(missing_sample_areas)))
    valid_hypso = set(HypsoParamSet.objects.filter(id__in=hypso_ids).values_list('id', flat=True))
    missing_hypso = hypso_ids - valid_hypso
    if missing_hypso:
        raise UploadValidationError(S.IPSO_ERR_UNKNOWN_HYPSO_PARAM_SET_ID.format(min(missing_hypso)))
    for i, row in enumerate(records, start=1):
        if parcels[row[FIELD_PARCEL_ID]] != row[FIELD_REGION_ID]:
            raise UploadValidationError(S.IPSO_ERR_RECORD_PARCEL_REGION.format(i))
        if mode == IPSO_MODE_SAMPLES:
            if sample_areas[row[FIELD_SAMPLE_AREA_ID]] != row[FIELD_PARCEL_ID]:
                raise UploadValidationError(S.IPSO_ERR_RECORD_SAMPLE_AREA_PARCEL.format(i))


def _payload_checksum(payload: dict) -> str:
    return ipso_staging.payload_checksum(payload)


def _upload_inbox_path(session_id: str) -> Path:
    return ipso_staging.upload_inbox_path(session_id)


def _write_upload_files(
        session_dir: Path, payload: dict, checksum: str, csv_text: str | None,
) -> Path:
    return ipso_staging.write_upload_files(session_dir, payload, checksum, csv_text)


def _dict(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise UploadValidationError(S.IPSO_ERR_FIELD_OBJECT.format(key))
    return value


def _list(payload: dict, key: str) -> list:
    value = payload.get(key)
    if not isinstance(value, list):
        raise UploadValidationError(S.IPSO_ERR_FIELD_ARRAY.format(key))
    return value


def _str(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UploadValidationError(S.IPSO_ERR_FIELD_REQUIRED.format(key))
    return value.strip()


def _opt_str(payload: dict, key: str) -> str:
    value = payload.get(key, '')
    if value is None:
        return ''
    if not isinstance(value, str):
        raise UploadValidationError(S.IPSO_ERR_FIELD_STRING.format(key))
    return value.strip()


def _int(payload: dict, key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise UploadValidationError(S.IPSO_ERR_FIELD_INTEGER.format(key))
    return value


def _opt_int(payload: dict, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise UploadValidationError(S.IPSO_ERR_FIELD_INTEGER_NULL.format(key))
    return value


def _bool(payload: dict, key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise UploadValidationError(S.IPSO_ERR_FIELD_BOOLEAN.format(key))
    return value


def _opt_bool(payload: dict, key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is not bool:
        raise UploadValidationError(S.IPSO_ERR_FIELD_BOOLEAN.format(key))
    return value


def _opt_coord_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise UploadValidationError(S.IPSO_ERR_FIELD_NUMBER_NULL.format(key))
    out = coord_float(to_decimal(value, '.'))
    if out is None:
        raise UploadValidationError(S.IPSO_ERR_FIELD_FINITE.format(key))
    return out


def _opt_decimal(payload: dict, key: str) -> Decimal | None:
    value = payload.get(key)
    if value is None:
        return None
    return _decimal(payload, key)


def _decimal(payload: dict, key: str) -> Decimal:
    value = payload.get(key)
    if type(value) not in {int, float, str}:
        raise UploadValidationError(S.IPSO_ERR_FIELD_DECIMAL.format(key))
    dec = to_decimal(value, '.')
    if dec is None:
        raise UploadValidationError(S.IPSO_ERR_FIELD_DECIMAL_FINITE.format(key))
    return dec


def _json_response(
        payload: dict, *, content_type: str,
        cache_control: str = CACHE_NO_STORE,
) -> HttpResponse:
    response = HttpResponse(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
        content_type=content_type,
    )
    apply_cache_control(response, cache_control)
    return response


def _api_json(payload: dict, *, status: int = 200) -> JsonResponse:
    response = JsonResponse(payload, status=status)
    apply_cache_control(response, CACHE_NO_STORE)
    return response
