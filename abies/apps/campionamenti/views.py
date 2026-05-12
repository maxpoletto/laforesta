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
    Parcel, Region, Sample, SampleArea, SampleGrid, Species, Survey, Tree,
    TreeSample,
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

    # If the user picked an existing tree from the pulldown, that tree is
    # authoritative for species / coppice / lat / lng / number (cross-sample
    # tree identity per campionamenti.md §"Cross-sample tree identity").
    # Validate it lives in this sample area before we trust it.
    existing_tree = None
    if parsed['tree_pick_existing_id'] is not None:
        existing_tree = _resolve_existing_tree(
            parsed['tree_pick_existing_id'], sample.sample_area_id,
        )
        if existing_tree is None:
            return _validation_error(
                [S.ERR_TREE_NUMBER_REQUIRED], ts_id, request, body,
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
        elif existing_tree is not None:
            # Reuse the existing Tree row.  Do not create a new Tree.
            ts = TreeSample.objects.create(
                sample=sample, tree=existing_tree, shoot=0, standard=False,
                number=parsed['number'],
                d_cm=parsed['d_cm'], h_m=parsed['h_m'],
                l10_mm=parsed['l10_mm'],
                volume_m3=parsed['volume_m3'],
                mass_q=parsed['mass_q'],
            )
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
                                'sample__sample_area__parcel__eclass',
                                'tree__species')
                .get(id=ts_id))
        sample = ts.sample
        area = sample.sample_area
        survey = sample.survey
    else:
        if survey_id is None or area_id is None:
            raise Http404('survey and area required')
        survey = Survey.objects.get(id=survey_id)
        area = SampleArea.objects.select_related(
            'parcel__region', 'parcel__eclass',
        ).get(id=area_id)
        if area.sample_grid_id != survey.sample_grid_id:
            raise Http404('sample_area not in survey grid')
        sample = Sample.objects.filter(
            sample_area=area, survey=survey,
        ).first()

    species = list(Species.objects.filter(active=True).order_by('sort_order'))
    prior_trees, next_number = _prior_trees_for_area(area, exclude_ts_id=ts_id)

    return render_to_string('campionamenti/_tree_form.html', {
        'ts': ts,
        'tree': ts.tree if ts else None,
        'sample': sample,
        'area': area,
        'survey': survey,
        'species': species,
        'sample_date': sample.date if sample else date_type.today(),
        'prior_trees': prior_trees,
        'next_number': next_number,
        # Fustaia default: ceduo when the parcel's eclass is a coppice class.
        'fustaia_default': not area.parcel.eclass.coppice,
    }, request=request)


def _prior_trees_for_area(area, exclude_ts_id=None):
    """Build the list backing the "Numero albero" pulldown.

    Returns (prior_trees, next_number) where:
      - prior_trees is a list of dicts (one per distinct Tree in this area,
        most-recent measurement first within each tree) sorted by tree number;
      - next_number is `max(tree_sample.number)+1` across all samples in this
        area, or 1 if empty.

    Same `number` is reused across surveys for the same physical tree
    (cross-sample identity, per campionamenti.md §"Cross-sample tree
    identity").
    """
    qs = (TreeSample.objects
            .filter(sample__sample_area_id=area.id, shoot=0)
            .select_related('tree__species', 'sample')
            .order_by('tree_id', '-sample__date'))
    if exclude_ts_id:
        qs = qs.exclude(id=exclude_ts_id)

    by_tree = {}
    max_number = 0
    for ts in qs:
        max_number = max(max_number, ts.number or 0)
        if ts.tree_id in by_tree:
            continue       # already have the most-recent measurement for this tree
        by_tree[ts.tree_id] = {
            'tree_id': ts.tree_id,
            'number': ts.number,
            'species_id': ts.tree.species_id,
            'species_common_name': ts.tree.species.common_name,
            'coppice': ts.tree.coppice,
            'lat': ts.tree.lat,
            'lng': ts.tree.lng,
            'last_d_cm': ts.d_cm,
            'last_h_m': ts.h_m,
        }
    prior = sorted(by_tree.values(), key=lambda r: r['number'])
    return prior, max_number + 1


def _parse_tree_body(body):
    """Extract and validate fields for a fustaia tree+sample save.

    `tree_pick` is `'new'` for a brand-new tree (a Tree row will be created)
    or the integer id of an existing Tree in this sample area.  In the latter
    case the existing tree's species / coppice / lat / lng are reused on
    save, regardless of what the body's species_id / fustaia / lat / lng
    fields contain (they may be present but the client should have locked
    them in the UI — server treats the existing Tree as authoritative).
    """
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

    # tree_pick: 'new' or an integer Tree id.  Older payloads without
    # tree_pick default to 'new' so the existing call sites keep working.
    tree_pick_raw = body.get('tree_pick', 'new')
    tree_pick_existing_id = None
    if tree_pick_raw not in (None, '', 'new'):
        try:
            tree_pick_existing_id = int(tree_pick_raw)
        except (ValueError, TypeError):
            errors.append(S.ERR_TREE_NUMBER_REQUIRED)

    parsed = {
        'sample_area_id': int(body['sample_area_id']),
        'survey_id': int(body['survey_id']),
        'species_id': int(body['species_id']) if body.get('species_id') else None,
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
        'tree_pick_existing_id': tree_pick_existing_id,
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


def _resolve_existing_tree(tree_id, sample_area_id):
    """Return the Tree with the given id iff it has at least one TreeSample
    in the requested sample_area.  Otherwise return None — keeps the
    cross-sample identity convention scoped to a single area (a tree in a
    different area is not the same physical tree).
    """
    in_area = TreeSample.objects.filter(
        tree_id=tree_id, sample__sample_area_id=sample_area_id,
    ).exists()
    if not in_area:
        return None
    return Tree.objects.filter(id=tree_id).first()


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
    """Return the HTML fragment for the "Nuova griglia" modal — three
    creation paths (empty / auto / csv) per `campionamenti.md` §1.
    """
    return JsonResponse({'html': render_to_string(
        'campionamenti/_grid_modal.html', {}, request=request,
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
    """Return the HTML fragment for the "Nuovo rilevamento" modal — two
    creation paths (empty / csv) per `campionamenti.md` §2.
    """
    grids = SampleGrid.objects.order_by('-modified_at')
    return JsonResponse({'html': render_to_string(
        'campionamenti/_survey_modal.html', {'grids': grids}, request=request,
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


@login_required
@require_writer
@require_POST
def grid_save_auto_view(request):
    """Persist an auto-generated grid (Section 1, "Genera automaticamente").

    Body shape:
      {
        "name": "...",
        "description": "...",
        "r_m": 12,
        "points": [
           {"compresa": "Capistrano", "particella": "1",
            "lat": 38.51, "lng": 16.11},
           ...
        ],
        "nonce": "..."
      }

    Per-point compresa+particella resolves to a Parcel.  Resolution
    failure aborts the entire commit (no partial state).
    """
    body = json.loads(request.body)
    name = (body.get('name') or '').strip()
    description = (body.get('description') or '').strip()
    points = body.get('points') or []
    try:
        r_m = int(body.get('r_m') or 12)
    except (ValueError, TypeError):
        r_m = 12

    if not name:
        return _simple_validation_error(S.ERR_GRID_NAME_REQUIRED)
    if SampleGrid.objects.filter(name=name).exists():
        return _simple_validation_error(S.ERR_GRID_NAME_DUPLICATE)
    if not points:
        return _simple_validation_error(S.ERR_GRID_AUTO_NO_POINTS)

    resolved, err = _resolve_grid_points(points)
    if err:
        return _simple_validation_error(err)

    with transaction.atomic():
        grid = SampleGrid.objects.create(name=name, description=description)
        SampleArea.objects.bulk_create([
            SampleArea(
                sample_grid=grid,
                parcel=parcel,
                number=str(i + 1),
                lat=pt['lat'], lng=pt['lng'],
                r_m=r_m,
                note='',
            )
            for i, (pt, parcel) in enumerate(resolved)
        ])
        mark_stale('grids', 'sample_areas', 'audit')

    response_data = {'data_id': 'grids', 'row_id': grid.id}
    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


def _resolve_grid_points(points):
    """Return ([(point, parcel), ...], error_msg).  error_msg is non-None
    when the first unresolved point is hit — the caller aborts the whole
    commit at that point (per spec §"Grid CSV import": no partial state)."""
    region_cache = {}
    parcel_cache = {}
    resolved = []
    for pt in points:
        compresa = (pt.get('compresa') or '').strip()
        particella = str(pt.get('particella') or '').strip()
        if not compresa or not particella:
            return None, S.ERR_GRID_AUTO_PARCEL_UNRESOLVED.format(
                compresa or '?', particella or '?',
            )
        region = region_cache.get(compresa.lower())
        if region is None:
            region = Region.objects.filter(name__iexact=compresa).first()
            if region is None:
                return None, S.ERR_GRID_AUTO_PARCEL_UNRESOLVED.format(
                    compresa, particella,
                )
            region_cache[compresa.lower()] = region
        key = (region.id, particella)
        parcel = parcel_cache.get(key)
        if parcel is None:
            parcel = Parcel.objects.filter(
                region=region, name=particella,
            ).first()
            if parcel is None:
                return None, S.ERR_GRID_AUTO_PARCEL_UNRESOLVED.format(
                    compresa, particella,
                )
            parcel_cache[key] = parcel
        resolved.append((pt, parcel))
    return resolved, None


def _simple_validation_error(message):
    """Shorter validation-error helper for grid/survey saves (no form
    re-render — the client just shows the message)."""
    return JsonResponse({
        'status': 'validation_error',
        'message': message,
        'html': '',
    }, status=400)
