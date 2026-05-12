"""Campionamenti API views.

M3a: read endpoints (4 eager digests + per-survey lazy digest).
M3d-write: manual tree+sample entry form + save endpoint.

Form endpoints follow the standard Abies idiom (see
`apps.prelievi.views`): a GET returns an HTML fragment, a POST
processes the submission and returns either {row_id, record} or a
validation_error / conflict payload.
"""

import json
from datetime import date as date_type
from decimal import Decimal, InvalidOperation

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import Http404, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import mark_stale, serve_digest
from apps.base.middleware import save_nonce
from apps.base.models import (
    Sample, SampleArea, SampleGrid, Species, Survey, Tree, TreeSample,
)
from config import strings as S


# ---------------------------------------------------------------------------
# Data endpoints (M3a)
# ---------------------------------------------------------------------------

@login_required
def grids_data(request):
    return serve_digest(request, 'grids')


@login_required
def surveys_data(request):
    return serve_digest(request, 'surveys')


@login_required
def sample_areas_data(request):
    return serve_digest(request, 'sample_areas')


@login_required
def samples_data(request):
    return serve_digest(request, 'samples')


@login_required
def sampled_trees_data(request, survey_id: int):
    """Per-survey sampled-tree digest.  Lazily generated on first hit."""
    if survey_id <= 0:
        raise Http404
    return serve_digest(request, f'sampled_trees_{survey_id}')


# ---------------------------------------------------------------------------
# Manual tree + sample entry form (M3d-write)
# ---------------------------------------------------------------------------

@login_required
def tree_form_view(request, ts_id: int | None = None):
    """Return the HTML fragment for adding or editing a TreeSample.

    Query params (add path only):
      ?survey=<id>&area=<id>  — required: the survey + sample area
                                under which the new tree will be created.
    """
    survey_id = int(request.GET.get('survey', 0)) or None
    area_id = int(request.GET.get('area', 0)) or None
    return JsonResponse({'html': _render_tree_form(
        request, ts_id, survey_id, area_id,
    )})


@login_required
@require_writer
@require_POST
def tree_save_view(request):
    """Create or update a TreeSample (and its parent Tree row).

    For M3d-write we ship the fustaia path.  Coppice (per-shoot) entry
    will follow when the form needs it; the schema and Section 3
    digest are already coppice-aware.
    """
    body = json.loads(request.body)
    ts_id, parsed, errors = _parse_tree_body(body)
    if errors:
        return _validation_error(errors, ts_id, request, body)

    sample = _find_or_create_sample(parsed)
    if sample is None:
        return _validation_error(
            [S.ERR_AREA_OUT_OF_SURVEY], ts_id, request, body,
        )

    # Reject duplicate tree-number within the same Sample (same area
    # + same survey).  Same number across different surveys / sample
    # areas is fine (cross-sample tree identity convention).
    dup = TreeSample.objects.filter(
        sample=sample, number=parsed['number'], shoot=0,
    )
    if ts_id:
        dup = dup.exclude(id=ts_id)
    if dup.exists():
        return _validation_error(
            [S.ERR_TREE_NUMBER_DUPLICATE.format(parsed['number'])],
            ts_id, request, body,
        )

    with transaction.atomic():
        if ts_id:
            ts = _update_tree_sample(ts_id, sample, parsed)
        else:
            tree = Tree.objects.create(
                species_id=parsed['species_id'],
                parcel_id=sample.sample_area.parcel_id,
                lat=parsed['lat'], lng=parsed['lng'],
                preserved=parsed['preserved'],
                coppice=parsed['coppice'],
            )
            ts = TreeSample.objects.create(
                sample=sample, tree=tree, shoot=0, standard=False,
                number=parsed['number'],
                d_cm=parsed['d_cm'], h_m=parsed['h_m'],
                l10_mm=parsed['l10_mm'],
                volume_m3=parsed['volume_m3'],
                mass_q=parsed['mass_q'],
            )

        mark_stale(f'sampled_trees_{sample.survey_id}', 'samples', 'audit')

    response_data = {
        'data_id': f'sampled_trees_{sample.survey_id}',
        'row_id': ts.id,
    }
    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_tree_form(request, ts_id, survey_id, area_id):
    ts = sample = area = survey = None
    if ts_id:
        ts = (TreeSample.objects
                .select_related('sample__survey',
                                'sample__sample_area__parcel__region',
                                'tree__species')
                .get(id=ts_id))
        sample = ts.sample
        area = sample.sample_area
        survey = sample.survey
    else:
        if survey_id is None or area_id is None:
            raise Http404('survey and area required')
        survey = Survey.objects.get(id=survey_id)
        area = SampleArea.objects.select_related('parcel__region').get(id=area_id)
        if area.sample_grid_id != survey.sample_grid_id:
            raise Http404('sample_area not in survey grid')
        sample = Sample.objects.filter(
            sample_area=area, survey=survey,
        ).first()

    species = list(Species.objects.filter(active=True).order_by('sort_order'))
    return render_to_string('campionamenti/_tree_form.html', {
        'ts': ts,
        'tree': ts.tree if ts else None,
        'sample': sample,
        'area': area,
        'survey': survey,
        'species': species,
        'sample_date': sample.date if sample else date_type.today(),
    }, request=request)


def _parse_tree_body(body):
    """Extract and validate fields for a fustaia tree+sample save."""
    errors = []
    ts_id = body.get('row_id')
    ts_id = int(ts_id) if ts_id else None

    try:
        d_cm = int(body['d_cm'])
        if d_cm <= 0:
            errors.append(S.ERR_D_POSITIVE)
    except (KeyError, ValueError, TypeError):
        d_cm = 0
        errors.append(S.ERR_D_POSITIVE)

    try:
        h_m = Decimal(str(body.get('h_m', '0') or '0'))
        if h_m <= 0:
            errors.append(S.ERR_H_POSITIVE)
    except InvalidOperation:
        h_m = Decimal('0')
        errors.append(S.ERR_H_POSITIVE)

    try:
        l10_mm = int(body.get('l10_mm', 0) or 0)
    except (ValueError, TypeError):
        l10_mm = 0

    coppice = str(body.get('fustaia', 'true')).lower() in ('false', '0', 'no')
    if coppice:
        # Per spec, coppice rows have NULL V and m.  Form for coppice
        # entry isn't shipped yet (per-shoot block); reject for now.
        errors.append(S.ERR_COPPICE_NOT_YET_SUPPORTED)

    parsed = {
        'sample_area_id': int(body['sample_area_id']),
        'survey_id': int(body['survey_id']),
        'species_id': int(body['species_id']),
        'number': int(body.get('number', 0) or 0),
        'd_cm': d_cm,
        'h_m': h_m.quantize(Decimal('0.01')),
        'l10_mm': l10_mm,
        'volume_m3': _decimal_or_none(body.get('volume_m3')),
        'mass_q': _decimal_or_none(body.get('mass_q')),
        'lat': _float_or_none(body.get('lat')),
        'lng': _float_or_none(body.get('lng')),
        'preserved': bool(body.get('preserved')),
        'coppice': coppice,
    }
    if not parsed['number']:
        errors.append(S.ERR_TREE_NUMBER_REQUIRED)
    return ts_id, parsed, errors


def _decimal_or_none(v):
    if v in (None, '', 'null'): return None
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return None


def _float_or_none(v):
    if v in (None, '', 'null'): return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _find_or_create_sample(parsed):
    """Return the Sample row for (survey, sample_area), creating it if
    needed.  Returns None when the area doesn't belong to the survey's
    grid (caller surfaces a validation error)."""
    area = SampleArea.objects.select_related('parcel').get(
        id=parsed['sample_area_id'],
    )
    survey = Survey.objects.get(id=parsed['survey_id'])
    if area.sample_grid_id != survey.sample_grid_id:
        return None
    sample, _ = Sample.objects.get_or_create(
        sample_area=area, survey=survey,
        defaults={'date': date_type.today()},
    )
    return sample


def _update_tree_sample(ts_id, sample, parsed):
    ts = TreeSample.objects.select_for_update().get(id=ts_id)
    ts.sample = sample
    ts.number = parsed['number']
    ts.d_cm = parsed['d_cm']
    ts.h_m = parsed['h_m']
    ts.l10_mm = parsed['l10_mm']
    ts.volume_m3 = parsed['volume_m3']
    ts.mass_q = parsed['mass_q']
    ts.version += 1
    ts.save()
    # Tree fields that can change on edit.
    tree = ts.tree
    tree.species_id = parsed['species_id']
    tree.preserved = parsed['preserved']
    tree.lat = parsed['lat']
    tree.lng = parsed['lng']
    tree.version += 1
    tree.save()
    return ts


def _validation_error(errors, ts_id, request, body):
    survey_id = int(body.get('survey_id', 0)) or None
    area_id = int(body.get('sample_area_id', 0)) or None
    # Skip the form re-render when the survey/area combo is itself
    # invalid (Http404 from _render_tree_form would mask the real
    # error).  Client just shows the error message in that case.
    try:
        html = _render_tree_form(request, ts_id, survey_id, area_id)
    except Http404:
        html = ''
    return JsonResponse({
        'status': 'validation_error',
        'message': ' '.join(errors),
        'html': html,
    }, status=400)


# ---------------------------------------------------------------------------
# Grid + Survey CRUD (M3d-write; "Crea vuota/o" path)
# ---------------------------------------------------------------------------

@login_required
def grid_form_view(request):
    """Return the HTML fragment for creating a new (empty) grid."""
    return JsonResponse({'html': render_to_string(
        'campionamenti/_grid_form.html', {}, request=request,
    )})


@login_required
@require_writer
@require_POST
def grid_save_view(request):
    body = json.loads(request.body)
    name = (body.get('name') or '').strip()
    description = (body.get('description') or '').strip()
    if not name:
        return _simple_validation_error(S.ERR_GRID_NAME_REQUIRED)
    if SampleGrid.objects.filter(name=name).exists():
        return _simple_validation_error(S.ERR_GRID_NAME_DUPLICATE)

    with transaction.atomic():
        grid = SampleGrid.objects.create(name=name, description=description)
        mark_stale('grids', 'audit')

    response_data = {'data_id': 'grids', 'row_id': grid.id}
    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


@login_required
def survey_form_view(request):
    """Return the HTML fragment for creating a new (empty) survey."""
    grids = SampleGrid.objects.order_by('-modified_at')
    return JsonResponse({'html': render_to_string(
        'campionamenti/_survey_form.html', {'grids': grids}, request=request,
    )})


@login_required
@require_writer
@require_POST
def survey_save_view(request):
    body = json.loads(request.body)
    name = (body.get('name') or '').strip()
    description = (body.get('description') or '').strip()
    grid_id = body.get('sample_grid_id')

    if not name:
        return _simple_validation_error(S.ERR_SURVEY_NAME_REQUIRED)
    if not grid_id:
        return _simple_validation_error(S.ERR_SURVEY_GRID_REQUIRED)
    if Survey.objects.filter(name=name).exists():
        return _simple_validation_error(S.ERR_SURVEY_NAME_DUPLICATE)
    try:
        grid = SampleGrid.objects.get(id=int(grid_id))
    except (SampleGrid.DoesNotExist, ValueError):
        return _simple_validation_error(S.ERR_SURVEY_GRID_REQUIRED)

    with transaction.atomic():
        survey = Survey.objects.create(
            name=name, sample_grid=grid, description=description,
        )
        mark_stale('surveys', 'audit')

    response_data = {'data_id': 'surveys', 'row_id': survey.id}
    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


def _simple_validation_error(message):
    """Shorter validation-error helper for grid/survey saves (no form
    re-render — the client just shows the message)."""
    return JsonResponse({
        'status': 'validation_error',
        'message': message,
        'html': '',
    }, status=400)
