"""Campionamenti API views.

M3a: read endpoints (4 eager digests + per-survey lazy digest).
M3d-write: manual tree+sample entry form + save endpoint.

Form endpoints follow the standard Abies idiom (see
`apps.prelievi.views`): a GET returns an HTML fragment, a POST
processes the submission and returns either {row_id, record} or a
validation_error / conflict payload.
"""

import csv
import io
import json
from datetime import date as date_type
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import Http404, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import (
    build_grid_record, build_sample_area_record, build_sample_record,
    build_survey_record, build_tree_sample_record,
    mark_stale, serve_digest,
)
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

    Fustaia path: one TreeSample row with shoot=0.
    Coppice path: N TreeSample rows sharing one Tree, one per shoot
    (parsed from the `shoots` JSON field).  Edit of a coppice row
    updates a single TreeSample (no new shoots).
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

    # Reject (sample, number) referring to a different tree — within a
    # sample area, a `number` identifies one physical tree across all
    # samples in which it appears.  Excluding our target tree (when
    # known) lets coppice multi-shoot creates and same-tree edits pass.
    dup = TreeSample.objects.filter(sample=sample, number=parsed['number'])
    if ts_id:
        dup = dup.exclude(id=ts_id)
    if existing_tree is not None:
        dup = dup.exclude(tree_id=existing_tree.id)
    if dup.exists():
        return _validation_error(
            [S.ERR_TREE_NUMBER_DUPLICATE.format(parsed['number'])],
            ts_id, request, body,
        )

    # Coppice create with existing tree: detect shoot-number collisions
    # against existing TreeSamples on this (sample, tree) before commit
    # so we surface a friendly error instead of an IntegrityError.
    if parsed['coppice'] and not ts_id and existing_tree is not None:
        existing_shoots = set(TreeSample.objects.filter(
            sample=sample, tree=existing_tree,
        ).values_list('shoot', flat=True))
        new_shoots = {s['shoot'] for s in parsed['shoots']}
        collision = sorted(existing_shoots & new_shoots)
        if collision:
            return _validation_error(
                [S.ERR_COPPICE_SHOOT_DUPLICATE.format(collision[0])],
                ts_id, request, body,
            )

    with transaction.atomic():
        if ts_id:
            ts = _update_tree_sample(ts_id, sample, parsed)
            created_or_updated_ids = [ts.id]
        elif parsed['coppice']:
            tree = existing_tree or Tree.objects.create(
                species_id=parsed['species_id'],
                parcel_id=sample.sample_area.parcel_id,
                lat=parsed['lat'], lng=parsed['lng'],
                preserved=parsed['preserved'],
                coppice=True,
            )
            created_or_updated_ids = []
            for sh in parsed['shoots']:
                ts = TreeSample.objects.create(
                    sample=sample, tree=tree, shoot=sh['shoot'],
                    standard=sh['standard'],
                    number=parsed['number'],
                    d_cm=sh['d_cm'], h_m=sh['h_m'], l10_mm=sh['l10_mm'],
                    volume_m3=None, mass_q=None,
                )
                created_or_updated_ids.append(ts.id)
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
            created_or_updated_ids = [ts.id]
        else:
            tree = Tree.objects.create(
                species_id=parsed['species_id'],
                parcel_id=sample.sample_area.parcel_id,
                lat=parsed['lat'], lng=parsed['lng'],
                preserved=parsed['preserved'],
                coppice=False,
            )
            ts = TreeSample.objects.create(
                sample=sample, tree=tree, shoot=0, standard=False,
                number=parsed['number'],
                d_cm=parsed['d_cm'], h_m=parsed['h_m'],
                l10_mm=parsed['l10_mm'],
                volume_m3=parsed['volume_m3'],
                mass_q=parsed['mass_q'],
            )
            created_or_updated_ids = [ts.id]

        # The tree form carries an editable Data field — apply the
        # user-chosen date to the parent Sample if it differs from the
        # current value.  Sample is unique per (survey, area), so this
        # bumps the date for every other tree in this sample too; same
        # semantics as the legacy inline date selector this replaces.
        if parsed['date'] is not None and parsed['date'] != sample.date:
            sample.date = parsed['date']
            sample.version += 1
            sample.save()

        # tree_save can create a new Sample (first tree in an area) which
        # affects surveys.N_aree_visitate / Data primo / Data ultimo.
        mark_stale(
            f'sampled_trees_{sample.survey_id}', 'samples', 'surveys', 'audit',
        )

    # Build the cache-update payload — see CLAUDE.md §"Optimistic table
    # updates".  Re-fetch with select_related so build_tree_sample_record
    # doesn't N+1 on attributes the digest expects.
    fresh_ts_qs = TreeSample.objects.filter(
        id__in=created_or_updated_ids,
    ).select_related(
        'sample', 'sample__sample_area__parcel__region',
        'tree__species', 'tree__parcel',
    )
    records = [build_tree_sample_record(ts) for ts in fresh_ts_qs]
    sample.refresh_from_db()
    sample.survey  # touch to avoid lazy load in build_survey_record
    response_data = {
        'data_id': f'sampled_trees_{sample.survey_id}',
        'row_id': created_or_updated_ids[-1],
        'records': records,
        'sample_record': build_sample_record(sample),
        'survey_record': build_survey_record(sample.survey),
    }
    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


@login_required
@require_writer
@require_POST
def tree_delete_view(request, ts_id: int):
    """Delete a single TreeSample row.  The underlying Tree row is
    *not* deleted — only the measurement, per spec §"Editing /
    deletion" ("Deleting a single tree_sample row leaves both the
    sample and the underlying tree row intact").
    """
    ts = TreeSample.objects.select_related(
        'sample__survey',
    ).filter(id=ts_id).first()
    if ts is None:
        return JsonResponse({'status': 'not_found'}, status=404)
    sample = ts.sample
    survey = sample.survey
    survey_id = survey.id
    ts.delete()
    # N. alberi just dropped; surveys.N_aree_visitate may also change
    # if this was the last tree on its area.
    mark_stale(f'sampled_trees_{survey_id}', 'samples', 'surveys', 'audit')
    sample.refresh_from_db()
    return JsonResponse({
        'data_id': f'sampled_trees_{survey_id}',
        'row_id': ts_id,
        'sample_record': build_sample_record(sample),
        'survey_record': build_survey_record(survey),
    })


@login_required
def area_form_view(request, area_id: int | None = None):
    """Form fragment for adding or editing a SampleArea.

    Query params (add path): ?grid=<id>&lat=...&lng=... (lat/lng optional —
    pre-fill from a click on the Griglie map).
    """
    area = None
    grid = None
    if area_id:
        area = SampleArea.objects.select_related(
            'parcel__region', 'sample_grid',
        ).filter(id=area_id).first()
        if area is None:
            raise Http404
        grid = area.sample_grid
    else:
        grid_id = int(request.GET.get('grid', 0)) or None
        if grid_id is None:
            raise Http404('grid required')
        grid = SampleGrid.objects.filter(id=grid_id).first()
        if grid is None:
            raise Http404
    initial_lat = request.GET.get('lat', '')
    initial_lng = request.GET.get('lng', '')

    regions = list(Region.objects.order_by('name'))
    parcels = list(Parcel.objects.select_related('region').order_by(
        'region__name', 'name',
    ))
    return JsonResponse({'html': render_to_string(
        'campionamenti/_area_form.html', {
            'area': area, 'grid': grid,
            'regions': regions, 'parcels': parcels,
            'initial_lat': initial_lat, 'initial_lng': initial_lng,
        }, request=request,
    )})


@login_required
@require_writer
@require_POST
def area_save_view(request):
    """Create or update a SampleArea."""
    body = json.loads(request.body)
    try:
        grid_id = int(body['sample_grid_id'])
        parcel_id = int(body['parcel_id'])
        number = (body.get('number') or '').strip()
        lat = float(body['lat'])
        lng = float(body['lng'])
        r_m = int(body.get('r_m') or 12)
    except (KeyError, ValueError, TypeError):
        return _simple_validation_error(S.ERROR_GENERIC)

    if not number:
        return _simple_validation_error(S.ERR_AREA_NUMBER_REQUIRED)

    altitude_raw = body.get('altitude_m')
    altitude = None
    if altitude_raw not in (None, '', 'null'):
        try:
            altitude = int(altitude_raw)
        except (ValueError, TypeError):
            altitude = None
    note = (body.get('note') or '').strip()

    grid = SampleGrid.objects.filter(id=grid_id).first()
    parcel = Parcel.objects.filter(id=parcel_id).first()
    if grid is None or parcel is None:
        return JsonResponse({'status': 'not_found'}, status=404)

    area_id = body.get('row_id')
    area_id = int(area_id) if area_id else None

    with transaction.atomic():
        if area_id:
            area = SampleArea.objects.select_for_update().filter(
                id=area_id, sample_grid=grid,
            ).first()
            if area is None:
                return JsonResponse({'status': 'not_found'}, status=404)
            area.parcel = parcel
            area.number = number
            area.lat = lat
            area.lng = lng
            area.altitude_m = altitude
            area.r_m = r_m
            area.note = note
            area.version += 1
            area.save()
        else:
            area = SampleArea.objects.create(
                sample_grid=grid, parcel=parcel, number=number,
                lat=lat, lng=lng, altitude_m=altitude, r_m=r_m, note=note,
            )
        # `surveys` digest's N. aree totali depends on the area count
        # per grid → must invalidate surveys on area writes too.
        mark_stale('sample_areas', 'grids', 'surveys', 'audit')

    # Reload with select_related so build_sample_area_record doesn't N+1.
    area = SampleArea.objects.select_related('parcel__region').get(id=area.id)
    response_data = {
        'data_id': 'sample_areas',
        'row_id': area.id,
        'record': build_sample_area_record(area),
        'grid_record': build_grid_record(grid),
        'survey_records': [
            build_survey_record(sv)
            for sv in Survey.objects.filter(sample_grid=grid)
        ],
    }
    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


@login_required
@require_writer
@require_POST
def area_delete_view(request, area_id: int):
    """Delete a SampleArea.  Refused if any Sample references it."""
    area = SampleArea.objects.select_related('sample_grid').filter(
        id=area_id,
    ).first()
    if area is None:
        return JsonResponse({'status': 'not_found'}, status=404)
    if Sample.objects.filter(sample_area=area).exists():
        return _simple_validation_error(S.ERR_AREA_IN_USE)
    grid = area.sample_grid
    area.delete()
    # See area_save_view: surveys digest depends on the per-grid area count.
    mark_stale('sample_areas', 'grids', 'surveys', 'audit')
    return JsonResponse({
        'data_id': 'sample_areas',
        'row_id': area_id,
        'grid_record': build_grid_record(grid),
        'survey_records': [
            build_survey_record(sv)
            for sv in Survey.objects.filter(sample_grid=grid)
        ],
    })


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
        # Default species pick for the New-tree path: Abete on fustaia,
        # Castagno on ceduo.  Tolerate missing species (e.g., in tests
        # with a minimal fixture): falls back to the first species.
        'default_species_id': (
            None if ts else _default_species_id(species, area.parcel.eclass.coppice)
        ),
    }, request=request)


def _default_species_id(species, parcel_is_coppice):
    """Pick the default species for a newly added tree.

    Fustaia parcel → `S.SPECIES_DEFAULT_FUSTAIA`, ceduo parcel →
    `S.SPECIES_DEFAULT_CEDUO`.  Match on Species.common_name with
    `__iexact` first, then a substring fallback for variants like
    "Abete bianco".  Returns `None` when the species list is empty;
    otherwise the first species in the list if neither match hits.
    """
    if not species:
        return None
    target = (
        S.SPECIES_DEFAULT_CEDUO if parcel_is_coppice
        else S.SPECIES_DEFAULT_FUSTAIA
    ).lower()
    exact = next(
        (sp for sp in species if (sp.common_name or '').lower() == target),
        None,
    )
    if exact is not None:
        return exact.id
    partial = next(
        (sp for sp in species if target in (sp.common_name or '').lower()),
        None,
    )
    if partial is not None:
        return partial.id
    return species[0].id


def _prior_trees_for_area(area, exclude_ts_id=None):
    """Build the list backing the "Numero albero" pulldown.

    Returns (prior_trees, next_number) where:
      - prior_trees is a list of dicts (one per distinct Tree in this area,
        most-recent measurement first within each tree) sorted by tree number;
      - next_number is `max(tree_sample.number)+1` across all samples in this
        area, or 1 if empty.

    Same `number` is reused across surveys for the same physical tree
    (cross-sample identity, per campionamenti.md §"Cross-sample tree
    identity").  For coppice trees the entry also carries `next_shoot`
    = max(shoot for this tree across all samples) + 1, which the form
    uses as the starting pollone number when the operator picks this
    tree from the pulldown.
    """
    qs = (TreeSample.objects
            .filter(sample__sample_area_id=area.id)
            .select_related('tree__species', 'sample')
            .order_by('tree_id', '-sample__date', '-shoot'))
    if exclude_ts_id:
        qs = qs.exclude(id=exclude_ts_id)

    by_tree = {}
    max_number = 0
    max_shoot_by_tree = {}
    for ts in qs:
        max_number = max(max_number, ts.number or 0)
        max_shoot_by_tree[ts.tree_id] = max(
            max_shoot_by_tree.get(ts.tree_id, 0), ts.shoot or 0,
        )
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
    for tid, row in by_tree.items():
        row['next_shoot'] = max_shoot_by_tree.get(tid, 0) + 1
    prior = sorted(by_tree.values(), key=lambda r: r['number'])
    return prior, max_number + 1


def _parse_tree_body(body):
    """Extract and validate fields for a tree+sample save.

    Two branches:
      - Fustaia (default): reads d_cm/h_m/l10_mm/volume_m3/mass_q from
        top-level fields.  Creates exactly one TreeSample on save.
      - Coppice: reads a `shoots` JSON array of {shoot, standard, d_cm,
        h_m, l10_mm} entries.  Creates N TreeSamples on save (or
        updates a single one on the edit path — in that case the array
        carries exactly one row).

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

    # Date is editable inline in the tree form (replaces the separate
    # inline selector that used to live above the alberi table).  The
    # field is optional in the wire format: missing → keep existing
    # sample.date (edit) or default to today (new).  Invalid → error.
    date_raw = body.get('date')
    parsed_date = None
    if date_raw not in (None, '', 'null'):
        try:
            parsed_date = date_type.fromisoformat(str(date_raw))
        except (ValueError, TypeError):
            errors.append(S.ERR_DATE_INVALID)

    coppice = str(body.get('fustaia', 'true')).lower() in ('false', '0', 'no')

    # tree_pick: 'new' or an integer Tree id.  Older payloads without
    # tree_pick default to 'new' so the existing call sites keep working.
    tree_pick_raw = body.get('tree_pick', 'new')
    tree_pick_existing_id = None
    if tree_pick_raw not in (None, '', 'new'):
        try:
            tree_pick_existing_id = int(tree_pick_raw)
        except (ValueError, TypeError):
            errors.append(S.ERR_TREE_NUMBER_REQUIRED)

    if coppice:
        shoots, shoot_errors = _parse_shoots(body.get('shoots'))
        errors.extend(shoot_errors)
        # Coppice rows carry no per-tree volume / mass — Tabacchi only
        # applies to fustaia, per the spec's "V/m blank for ceduo".
        d_cm = 0
        h_m = Decimal('0')
        l10_mm = 0
    else:
        shoots = []
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

    parsed = {
        'sample_area_id': int(body['sample_area_id']),
        'survey_id': int(body['survey_id']),
        'species_id': int(body['species_id']) if body.get('species_id') else None,
        'number': int(body.get('number', 0) or 0),
        'd_cm': d_cm,
        'h_m': h_m.quantize(Decimal('0.01')) if not coppice else h_m,
        'l10_mm': l10_mm,
        'volume_m3': _decimal_or_none(body.get('volume_m3')) if not coppice else None,
        'mass_q': _decimal_or_none(body.get('mass_q')) if not coppice else None,
        'lat': _float_or_none(body.get('lat')),
        'lng': _float_or_none(body.get('lng')),
        'preserved': bool(body.get('preserved')),
        'coppice': coppice,
        'shoots': shoots,
        'tree_pick_existing_id': tree_pick_existing_id,
        'date': parsed_date,
    }
    if not parsed['number']:
        errors.append(S.ERR_TREE_NUMBER_REQUIRED)
    return ts_id, parsed, errors


def _parse_shoots(raw):
    """Parse the coppice `shoots` JSON array.

    Returns (shoots, errors).  Each shoot dict carries int `shoot`, bool
    `standard`, int `d_cm` (>0), Decimal `h_m` (>0, 2 d.p.), int `l10_mm`.
    """
    if not raw:
        return [], [S.ERR_COPPICE_NO_SHOOTS]
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return [], [S.ERR_COPPICE_NO_SHOOTS]
    if not isinstance(items, list) or not items:
        return [], [S.ERR_COPPICE_NO_SHOOTS]

    shoots = []
    errors = []
    for item in items:
        try:
            shoot_num = int(item.get('shoot', 0))
            d_cm = int(item.get('d_cm', 0))
            h_m = Decimal(str(item.get('h_m', '0') or '0'))
            l10_mm = int(item.get('l10_mm', 0) or 0)
            standard = bool(item.get('standard'))
        except (ValueError, TypeError, InvalidOperation):
            errors.append(S.ERR_D_POSITIVE)
            continue
        if d_cm <= 0:
            errors.append(S.ERR_D_POSITIVE)
        if h_m <= 0:
            errors.append(S.ERR_H_POSITIVE)
        shoots.append({
            'shoot': shoot_num, 'standard': standard,
            'd_cm': d_cm, 'h_m': h_m.quantize(Decimal('0.01')),
            'l10_mm': l10_mm,
        })
    return shoots, errors


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
        defaults={'date': parsed.get('date') or date_type.today()},
    )
    return sample


def _update_tree_sample(ts_id, sample, parsed):
    ts = TreeSample.objects.select_for_update().get(id=ts_id)
    ts.sample = sample
    ts.number = parsed['number']
    if parsed['coppice']:
        # Coppice edit form sends exactly one shoot row (the one being
        # edited); the multi-row "Aggiungi pollone" path is add-only.
        sh = parsed['shoots'][0]
        ts.shoot = sh['shoot']
        ts.standard = sh['standard']
        ts.d_cm = sh['d_cm']
        ts.h_m = sh['h_m']
        ts.l10_mm = sh['l10_mm']
        ts.volume_m3 = None
        ts.mass_q = None
    else:
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

    The CSV path imports INTO an existing grid (mirrors the survey CSV
    import); the target-grid dropdown needs the current list.
    """
    grids = SampleGrid.objects.order_by('-modified_at')
    return JsonResponse({'html': render_to_string(
        'campionamenti/_grid_modal.html', {'grids': grids}, request=request,
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

    response_data = {
        'data_id': 'grids',
        'row_id': grid.id,
        'record': build_grid_record(grid),
    }
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
    surveys = Survey.objects.order_by('-modified_at')
    return JsonResponse({'html': render_to_string(
        'campionamenti/_survey_modal.html', {
            'grids': grids, 'surveys': surveys,
        }, request=request,
    )})


@login_required
@require_writer
@require_POST
def grid_edit_view(request, grid_id: int):
    """Edit a grid's name / description (no cascade)."""
    grid = SampleGrid.objects.filter(id=grid_id).first()
    if grid is None:
        return JsonResponse({'status': 'not_found'}, status=404)
    body = json.loads(request.body)
    name = (body.get('name') or '').strip()
    description = (body.get('description') or '').strip()
    if not name:
        return _simple_validation_error(S.ERR_GRID_NAME_REQUIRED)
    if SampleGrid.objects.filter(name=name).exclude(id=grid.id).exists():
        return _simple_validation_error(S.ERR_GRID_NAME_DUPLICATE)
    grid.name = name
    grid.description = description
    grid.version += 1
    grid.save()
    mark_stale('grids', 'audit')
    return JsonResponse({
        'data_id': 'grids',
        'row_id': grid.id,
        'record': build_grid_record(grid),
    })


@login_required
@require_writer
@require_POST
def grid_delete_view(request, grid_id: int):
    """Delete a grid.  Refused if any Survey references it (Survey.sample_grid
    is PROTECT — the only way to "force" delete a populated grid is to delete
    its surveys first)."""
    grid = SampleGrid.objects.filter(id=grid_id).first()
    if grid is None:
        return JsonResponse({'status': 'not_found'}, status=404)
    if Survey.objects.filter(sample_grid=grid).exists():
        return _simple_validation_error(S.ERR_GRID_IN_USE)
    with transaction.atomic():
        # SampleArea cascades.
        grid.delete()
        mark_stale('grids', 'sample_areas', 'audit')
    return JsonResponse({'data_id': 'grids', 'row_id': grid_id})


@login_required
@require_writer
@require_POST
def survey_edit_view(request, survey_id: int):
    """Edit a survey's name / description."""
    survey = Survey.objects.filter(id=survey_id).first()
    if survey is None:
        return JsonResponse({'status': 'not_found'}, status=404)
    body = json.loads(request.body)
    name = (body.get('name') or '').strip()
    description = (body.get('description') or '').strip()
    if not name:
        return _simple_validation_error(S.ERR_SURVEY_NAME_REQUIRED)
    if Survey.objects.filter(name=name).exclude(id=survey.id).exists():
        return _simple_validation_error(S.ERR_SURVEY_NAME_DUPLICATE)
    survey.name = name
    survey.description = description
    survey.version += 1
    survey.save()
    mark_stale('surveys', 'audit')
    return JsonResponse({
        'data_id': 'surveys',
        'row_id': survey.id,
        'record': build_survey_record(survey),
    })


# ---------------------------------------------------------------------------
# CSV imports (Bucket 3) — Grid CSV + Tree-and-sample CSV
# ---------------------------------------------------------------------------

GRID_CSV_REQUIRED = ['Compresa', 'Particella', 'Area saggio', 'Lon', 'Lat',
                     'Quota', 'Raggio']
TREE_CSV_REQUIRED = ['Compresa', 'Particella', 'Area saggio', 'Albero',
                     'Pollone', 'Matricina', 'D_cm', 'H_m', 'L10_mm',
                     'Genere', 'Fustaia']
TREE_CSV_OPTIONAL = ['Data', 'PAI']


@login_required
@require_writer
@require_POST
def grid_csv_import_view(request):
    """Upload a CSV → append N SampleAreas to an existing SampleGrid.

    Multipart form fields:
      - sample_grid_id (required)
      - file           (required, .csv)

    Mirrors tree_csv_import_view's "import INTO existing parent" shape
    so a single grid can be populated incrementally from multiple
    files.  Rejects rows that would duplicate an existing
    (parcel, number) within the target grid; rejects rows that
    duplicate each other within the same upload.
    """
    grid_id = request.POST.get('sample_grid_id')
    upload = request.FILES.get('file')

    if not grid_id:
        return _simple_validation_error(S.ERR_CSV_GRID_REQUIRED)
    if upload is None:
        return _simple_validation_error(S.ERR_CSV_FILE_REQUIRED)

    grid = SampleGrid.objects.filter(id=int(grid_id)).first()
    if grid is None:
        return JsonResponse({'status': 'not_found'}, status=404)

    try:
        rows = _parse_csv(upload, GRID_CSV_REQUIRED)
    except _CsvError as e:
        return _simple_validation_error(str(e))

    parcel_cache = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }
    # Natural key per grid: (parcel_id, number).  Pre-load the existing
    # areas so we can reject collisions without round-tripping.
    existing_keys = set(SampleArea.objects.filter(sample_grid=grid)
                                          .values_list('parcel_id', 'number'))

    errors = []
    parsed_rows = []
    seen_in_csv = set()
    for i, row in enumerate(rows, 2):  # row 2 = first data row (after header)
        compresa = row['Compresa'].strip()
        particella = row['Particella'].strip()
        parcel = parcel_cache.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(
                S.ERR_CSV_ROW_PARCEL.format(i, compresa, particella),
            )
            continue
        number = row['Area saggio'].strip()
        key = (parcel.id, number)
        if key in existing_keys or key in seen_in_csv:
            errors.append(S.ERR_CSV_ROW_AREA_DUPLICATE.format(
                i, compresa, particella, number,
            ))
            continue
        seen_in_csv.add(key)
        try:
            parsed_rows.append({
                'parcel': parcel,
                'number': number,
                'lat': float(row['Lat']),
                'lng': float(row['Lon']),
                'altitude': _int_or_none_str(row['Quota']),
                'r_m': int(float(row['Raggio'])) if row['Raggio'].strip()
                       else 12,
                'note': '',
            })
        except (ValueError, KeyError) as exc:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, str(exc)))
    if errors:
        return _csv_error_list(errors)
    if not parsed_rows:
        return _simple_validation_error(S.ERR_CSV_EMPTY)

    with transaction.atomic():
        SampleArea.objects.bulk_create([
            SampleArea(
                sample_grid=grid,
                parcel=r['parcel'], number=r['number'],
                lat=r['lat'], lng=r['lng'],
                altitude_m=r['altitude'], r_m=r['r_m'], note=r['note'],
            )
            for r in parsed_rows
        ])
        # Surveys digest carries N. aree totali per grid → stale when
        # we add areas to a grid that has surveys.
        mark_stale('grids', 'sample_areas', 'surveys', 'audit')

    grid.refresh_from_db()
    area_qs = SampleArea.objects.filter(sample_grid=grid) \
                                 .select_related('parcel__region')
    response_data = {
        'data_id': 'grids', 'row_id': grid.id,
        'n_areas': len(parsed_rows),
        'record': build_grid_record(grid),
        'area_records': [build_sample_area_record(sa) for sa in area_qs],
        'survey_records': [
            build_survey_record(sv)
            for sv in Survey.objects.filter(sample_grid=grid)
        ],
    }
    nonce = request.POST.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


@login_required
@require_writer
@require_POST
def tree_csv_import_view(request):
    """Upload a CSV → create Samples + Trees + TreeSamples on a survey.

    Multipart form fields:
      - survey_id (required)
      - default_date (required if CSV lacks a Data column)
      - file (required, .csv)
    """
    survey_id = request.POST.get('survey_id')
    upload = request.FILES.get('file')
    default_date_str = (request.POST.get('default_date') or '').strip()

    if not survey_id:
        return _simple_validation_error(S.ERR_CSV_SURVEY_REQUIRED)
    if upload is None:
        return _simple_validation_error(S.ERR_CSV_FILE_REQUIRED)

    survey = Survey.objects.filter(id=int(survey_id)).select_related(
        'sample_grid',
    ).first()
    if survey is None:
        return JsonResponse({'status': 'not_found'}, status=404)

    try:
        rows = _parse_csv(upload, TREE_CSV_REQUIRED)
    except _CsvError as e:
        return _simple_validation_error(str(e))

    has_date_column = bool(rows) and 'Data' in rows[0]
    if not has_date_column and not default_date_str:
        return _simple_validation_error(S.ERR_CSV_DATE_REQUIRED)
    default_date = None
    if default_date_str:
        try:
            default_date = date_type.fromisoformat(default_date_str)
        except ValueError:
            return _simple_validation_error(S.ERR_CSV_DATE_REQUIRED)

    parcel_cache = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }
    area_cache = {
        (sa.parcel.region.name.lower(), sa.parcel.name, sa.number): sa
        for sa in SampleArea.objects.filter(sample_grid=survey.sample_grid)
                       .select_related('parcel__region')
    }
    species_cache = {s.common_name.lower(): s for s in Species.objects.all()}

    # Tabacchi inputs use English-side species names; reuse the GENERE_MAP
    # from the management command to handle minor naming drift.
    from apps.base.management.commands.import_sampled_trees import GENERE_MAP
    from apps.base.tabacchi import has_species, tabacchi_volume_m3

    errors = []
    parsed = []
    for i, row in enumerate(rows, 2):
        compresa = row['Compresa'].strip()
        particella = row['Particella'].strip()
        adc = row['Area saggio'].strip()
        area = area_cache.get((compresa.lower(), particella, adc))
        if area is None:
            errors.append(S.ERR_CSV_ROW_AREA.format(i, compresa, particella, adc))
            continue
        try:
            number = int(row['Albero'])
            shoot = int(row['Pollone'] or 0)
            standard = _bool_str(row['Matricina'])
            d_cm = int(float(row['D_cm']))
            h_m = Decimal(row['H_m']).quantize(Decimal('0.01'),
                                              rounding=ROUND_HALF_UP)
            l10_mm = int(float(row['L10_mm'])) if row['L10_mm'].strip() else 0
            fustaia = _bool_str(row['Fustaia'])
            coppice = not fustaia
            preserved = _bool_str(row.get('PAI', '')) if 'PAI' in row else False
        except (ValueError, InvalidOperation) as exc:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, str(exc)))
            continue

        genere = row['Genere'].strip()
        mapped = GENERE_MAP.get(genere, genere)
        species = species_cache.get(mapped.lower())
        if species is None:
            errors.append(S.ERR_CSV_ROW_SPECIES.format(i, genere))
            continue

        # Per-row date (if column present) else default.
        if has_date_column and row.get('Data', '').strip():
            try:
                row_date = date_type.fromisoformat(row['Data'].strip())
            except ValueError:
                errors.append(S.ERR_CSV_ROW_PARSE.format(
                    i, f'Data: {row["Data"]}',
                ))
                continue
        else:
            row_date = default_date

        if coppice or not has_species(mapped):
            volume_m3 = None
            mass_q = None
        else:
            volume_m3 = tabacchi_volume_m3(d_cm, h_m, mapped)
            mass_q = (volume_m3 * species.density).quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP,
            )

        parsed.append({
            'area': area, 'date': row_date, 'parcel': area.parcel,
            'species': species, 'coppice': coppice, 'preserved': preserved,
            'number': number, 'shoot': shoot, 'standard': standard,
            'd_cm': d_cm, 'h_m': h_m, 'l10_mm': l10_mm,
            'volume_m3': volume_m3, 'mass_q': mass_q,
        })
    if errors:
        return _csv_error_list(errors)
    if not parsed:
        return _simple_validation_error(S.ERR_CSV_EMPTY)

    with transaction.atomic():
        # Group rows by (area, date) into Samples (one sample per area+date).
        sample_by_key = {}
        for r in parsed:
            key = (r['area'].id, r['date'])
            if key in sample_by_key:
                continue
            sample, _ = Sample.objects.get_or_create(
                sample_area=r['area'], survey=survey,
                defaults={'date': r['date']},
            )
            sample_by_key[key] = sample

        # Walk parsed rows: create one Tree + one TreeSample per row.
        n_trees = 0
        for r in parsed:
            sample = sample_by_key[(r['area'].id, r['date'])]
            tree = Tree.objects.create(
                species=r['species'], parcel=r['parcel'],
                preserved=r['preserved'], coppice=r['coppice'],
            )
            TreeSample.objects.create(
                sample=sample, tree=tree, shoot=r['shoot'],
                standard=r['standard'], number=r['number'],
                d_cm=r['d_cm'], h_m=r['h_m'], l10_mm=r['l10_mm'],
                volume_m3=r['volume_m3'], mass_q=r['mass_q'],
            )
            n_trees += 1

        mark_stale(
            f'sampled_trees_{survey.id}', 'samples', 'surveys', 'audit',
        )

    return JsonResponse({
        'data_id': 'surveys', 'row_id': survey.id,
        'n_samples': len(sample_by_key),
        'n_trees': n_trees,
    })


class _CsvError(Exception):
    pass


def _parse_csv(upload, required_cols):
    """Decode + parse a Django uploaded CSV file.  Raises _CsvError on
    malformed file or missing required columns."""
    try:
        text = upload.read().decode('utf-8-sig')
    except UnicodeDecodeError:
        raise _CsvError(S.ERR_CSV_NOT_UTF8)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise _CsvError(S.ERR_CSV_EMPTY)
    missing = [c for c in required_cols if c not in reader.fieldnames]
    if missing:
        raise _CsvError(
            S.ERR_CSV_MISSING_COLS.format(', '.join(missing)),
        )
    return list(reader)


def _csv_error_list(errors):
    """Validation-error response carrying a per-row error list."""
    return JsonResponse({
        'status': 'validation_error',
        'message': errors[0] if errors else '',
        'errors': errors,
        'html': '',
    }, status=400)


def _int_or_none_str(s):
    s = (s or '').strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _bool_str(s):
    return str(s).strip().lower() in ('true', '1', 'yes', 'si', 'sì')


@login_required
@require_writer
@require_POST
def survey_delete_view(request, survey_id: int):
    """Delete a survey.  Cascades to its Samples and their TreeSamples
    (per the FK on_delete=CASCADE chain in models.py).  Tree rows
    remain (TreeSample.tree is PROTECT).
    """
    survey = Survey.objects.filter(id=survey_id).first()
    if survey is None:
        return JsonResponse({'status': 'not_found'}, status=404)
    with transaction.atomic():
        survey.delete()
        mark_stale(
            f'sampled_trees_{survey_id}', 'samples', 'surveys', 'grids',
            'audit',
        )
    return JsonResponse({'data_id': 'surveys', 'row_id': survey_id})


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
        # `grids` digest carries N. rilevamenti per grid, so creating a
        # survey must invalidate grids too.
        mark_stale('surveys', 'grids', 'audit')

    response_data = {
        'data_id': 'surveys', 'row_id': survey.id,
        'record': build_survey_record(survey),
        'grid_record': build_grid_record(grid),
    }
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

    area_qs = SampleArea.objects.filter(sample_grid=grid) \
                                 .select_related('parcel__region')
    response_data = {
        'data_id': 'grids', 'row_id': grid.id,
        'record': build_grid_record(grid),
        'area_records': [build_sample_area_record(sa) for sa in area_qs],
    }
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
