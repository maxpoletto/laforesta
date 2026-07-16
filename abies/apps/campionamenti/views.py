"""Campionamenti API views.

Read endpoints (4 eager digests + per-survey lazy digest).
Manual tree+sample entry form + save endpoint.

Form endpoints follow the standard Abies idiom (see
`apps.prelievi.views`): a GET returns an HTML fragment, a POST
processes the submission and returns a generic patches/deletes envelope or a
validation_error / conflict payload.
"""

import json
from datetime import date as date_type
from decimal import Decimal

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import Http404, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base import csv_io
from apps.base.auth import require_writer
from apps.base.digests import (
    build_grid_record, build_sample_area_record, build_sample_record,
    build_survey_record, build_tree_sample_record,
    mark_stale, serve_digest,
)
from apps.base.numparse import coord_float, int_or_none, parse_decimal
from apps.base.responses import (
    conflict_response, csv_error_list, parse_json_body, row_delete,
    row_patch, row_patches, save_model_response, submitted_version,
    success_response, validation_error,
)
from apps.base.models import (
    Parcel, Region, Sample, SampleArea, SampleGrid, Species, Survey, Tree,
    TreeSample, parcel_sort_key,
)
from apps.campionamenti import csv_grid, csv_trees
from apps.campionamenti.tree_validation import (
    normalize_sample_shoot_values, normalize_sample_tree_values,
)
from config import strings as S
from config.constants import (
    BOSCO_DENDROMETRY_DIGESTS, BOSCO_TREE_DIGESTS, DEFAULT_RADIUS_M,
    PRESSLER_DEFAULT,
    FIELD_ALTITUDE, FIELD_ALTITUDE_M, FIELD_AREA,
    FIELD_COMPRESA, FIELD_COPPICE, FIELD_DATE, FIELD_DEFAULT_DATE,
    FIELD_DESCRIPTION, FIELD_D_CM,
    FIELD_FILE, FIELD_HIGHFOREST, FIELD_H_M, FIELD_L10_MM, FIELD_LAT, FIELD_LON,
    FIELD_PRESSLER_COEFF,
    FIELD_MASS_Q,
    FIELD_NAME, FIELD_NEXT_SHOOT, FIELD_NOTE, FIELD_NUMBER,
    FIELD_PARCEL, FIELD_PARCEL_ID, FIELD_PARTICELLA, FIELD_POINTS,
    FIELD_PRESERVED, FIELD_R_M,
    FIELD_SAMPLE_AREA_ID, FIELD_SAMPLE_GRID_ID, FIELD_SHOOT, FIELD_SHOOTS,
    FIELD_SORT_ORDER, FIELD_SPECIES, FIELD_SPECIES_ID, FIELD_STANDARD,
    FIELD_SURVEY_ID, FIELD_TREE_PICK, FIELD_TREE_PICK_EXISTING_ID,
    FIELD_VOLUME_M3, HTML, ROW_ID, SAMPLE_GRID_UNSTRUCTURED, STATUS,
    STATUS_NOT_FOUND, VERSION, is_truthy,
)


# ---------------------------------------------------------------------------
# Data endpoints
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
    if survey_id <= 0 or not Survey.objects.filter(id=survey_id).exists():
        raise Http404
    return serve_digest(request, f'sampled_trees_{survey_id}')


# ---------------------------------------------------------------------------
# Manual tree + sample entry form
# ---------------------------------------------------------------------------

@login_required
@require_writer
def tree_form_view(request, ts_id: int | None = None):
    """Return the HTML fragment for adding or editing a TreeSample.

    Query params (add path only):
      ?survey=<id>&area=<id>  — required: the survey + sample area
                                under which the new tree will be created.
    """
    survey_raw = request.GET.get('survey')
    area_raw = request.GET.get(FIELD_AREA)
    survey_id = int_or_none(survey_raw) if survey_raw else None
    area_id = int_or_none(area_raw) if area_raw else None
    if (survey_raw and survey_id is None) or (area_raw and area_id is None):
        raise Http404
    return JsonResponse({HTML: _render_tree_form(
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
    body, error = parse_json_body(request)
    if error:
        return error
    ts_id, parsed, errors = _parse_tree_body(body)
    if errors:
        return _validation_error(errors, ts_id, request, body)

    sample = _find_or_create_sample(parsed)
    if sample is None:
        return _validation_error(
            [S.ERR_AREA_OUT_OF_SURVEY], ts_id, request, body,
        )
    if parsed[FIELD_DATE] is not None and parsed[FIELD_DATE] != sample.date:
        return _validation_error(
            [_sample_date_conflict_error(sample)], ts_id, request, body,
        )

    # If the user picked an existing tree from the pulldown, that tree is
    # authoritative for species / coppice / lat / lon / number (cross-sample
    # tree identity per campionamenti.md §"Cross-sample tree identity").
    # Validate it lives in this sample area before we trust it.
    existing_tree = None
    if parsed[FIELD_TREE_PICK_EXISTING_ID] is not None:
        existing_tree = _resolve_existing_tree(
            parsed[FIELD_TREE_PICK_EXISTING_ID], sample.sample_area_id,
        )
        if existing_tree is None:
            return _validation_error(
                [S.ERR_TREE_NUMBER_REQUIRED], ts_id, request, body,
            )

    # A tree number can appear at most once in the active sample.  This
    # also catches stale/direct posts for existing trees hidden from the
    # pulldown because the active sample already has that number.
    dup = TreeSample.objects.filter(sample=sample, number=parsed[FIELD_NUMBER])
    if ts_id:
        dup = dup.exclude(id=ts_id)
        if existing_tree is not None and parsed[FIELD_COPPICE]:
            dup = dup.exclude(tree=existing_tree)
    if dup.exists():
        if existing_tree is not None and dup.filter(tree=existing_tree).exists():
            return _validation_error(
                [S.ERR_TREE_ALREADY_IN_SAMPLE.format(parsed[FIELD_NUMBER])],
                ts_id, request, body,
            )
        return _validation_error(
            [S.ERR_TREE_NUMBER_DUPLICATE.format(parsed[FIELD_NUMBER])],
            ts_id, request, body,
        )

    # Coppice create with existing tree: detect shoot-number collisions
    # against existing TreeSamples on this (sample, tree) before commit
    # so we surface a friendly error instead of an IntegrityError.
    if parsed[FIELD_COPPICE] and not ts_id and existing_tree is not None:
        existing_shoots = set(TreeSample.objects.filter(
            sample=sample, tree=existing_tree,
        ).values_list(FIELD_SHOOT, flat=True))
        new_shoots = {s[FIELD_SHOOT] for s in parsed[FIELD_SHOOTS]}
        collision = sorted(existing_shoots & new_shoots)
        if collision:
            return _validation_error(
                [S.ERR_COPPICE_SHOOT_DUPLICATE.format(collision[0])],
                ts_id, request, body,
            )

    with transaction.atomic():
        if ts_id:
            ts = _update_tree_sample(ts_id, sample, parsed, body, request)
            if isinstance(ts, JsonResponse):
                return ts
            created_or_updated_ids = [ts.id]
        elif parsed[FIELD_COPPICE]:
            tree = existing_tree or Tree.objects.create(
                species_id=parsed[FIELD_SPECIES_ID],
                parcel_id=sample.sample_area.parcel_id,
                lat=parsed[FIELD_LAT], lon=parsed[FIELD_LON],
                preserved=parsed[FIELD_PRESERVED],
                coppice=True,
            )
            created_or_updated_ids = []
            for sh in parsed[FIELD_SHOOTS]:
                ts = TreeSample.objects.create(
                    sample=sample, tree=tree, shoot=sh[FIELD_SHOOT],
                    standard=sh[FIELD_STANDARD],
                    number=parsed[FIELD_NUMBER],
                    d_cm=sh[FIELD_D_CM], h_m=sh[FIELD_H_M], l10_mm=sh[FIELD_L10_MM],
                    pressler_coeff=parsed[FIELD_PRESSLER_COEFF],
                    volume_m3=None, mass_q=None,
                )
                created_or_updated_ids.append(ts.id)
        elif existing_tree is not None:
            # Reuse the existing Tree row.  Do not create a new Tree.
            ts = TreeSample.objects.create(
                sample=sample, tree=existing_tree, shoot=0, standard=False,
                number=parsed[FIELD_NUMBER],
                d_cm=parsed[FIELD_D_CM], h_m=parsed[FIELD_H_M],
                l10_mm=parsed[FIELD_L10_MM],
                pressler_coeff=parsed[FIELD_PRESSLER_COEFF],
                volume_m3=parsed[FIELD_VOLUME_M3],
                mass_q=parsed[FIELD_MASS_Q],
            )
            created_or_updated_ids = [ts.id]
        else:
            tree = Tree.objects.create(
                species_id=parsed[FIELD_SPECIES_ID],
                parcel_id=sample.sample_area.parcel_id,
                lat=parsed[FIELD_LAT], lon=parsed[FIELD_LON],
                preserved=parsed[FIELD_PRESERVED],
                coppice=False,
            )
            ts = TreeSample.objects.create(
                sample=sample, tree=tree, shoot=0, standard=False,
                number=parsed[FIELD_NUMBER],
                d_cm=parsed[FIELD_D_CM], h_m=parsed[FIELD_H_M],
                l10_mm=parsed[FIELD_L10_MM],
                pressler_coeff=parsed[FIELD_PRESSLER_COEFF],
                volume_m3=parsed[FIELD_VOLUME_M3],
                mass_q=parsed[FIELD_MASS_Q],
            )
            created_or_updated_ids = [ts.id]

        # tree_save can create a new Sample (first tree in an area) which
        # affects surveys.N_aree_visitate / Data primo / Data ultimo.  It can
        # also change Bosco dendrometry and PAI digests via TreeSample / Tree.
        mark_stale(
            f'sampled_trees_{sample.survey_id}', 'samples', 'surveys',
            *BOSCO_TREE_DIGESTS, 'audit',
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
    sample_record = build_sample_record(sample)
    survey_record = build_survey_record(sample.survey)
    return success_response(
        request, body,
        data_id=f'sampled_trees_{sample.survey_id}',
        row_id=created_or_updated_ids[-1],
        patches=[
            *row_patches(f'sampled_trees_{sample.survey_id}', records),
            row_patch('samples', sample_record[0], sample_record),
            row_patch('surveys', survey_record[0], survey_record),
        ],
    )


@login_required
@require_writer
@require_POST
def tree_delete_view(request, ts_id: int):
    """Delete a single TreeSample row.  The underlying Tree row is
    *not* deleted — only the measurement, per spec §"Editing /
    deletion" ("Deleting a single tree_sample row leaves both the
    sample and the underlying tree row intact").
    """
    body, error = parse_json_body(request)
    if error:
        return error
    ts = TreeSample.objects.select_related(
        'sample__survey',
    ).filter(id=ts_id).first()
    if ts is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if ts.version != submitted_version(body):
        return conflict_response(
            data_id=f'sampled_trees_{ts.sample.survey_id}',
            row_id=ts.id,
            record=build_tree_sample_record(ts),
        )
    sample = ts.sample
    survey = sample.survey
    survey_id = survey.id
    ts.delete()
    # N. alberi just dropped; surveys.N_aree_visitate may also change
    # if this was the last tree on its area.
    mark_stale(
        f'sampled_trees_{survey_id}', 'samples', 'surveys',
        *BOSCO_DENDROMETRY_DIGESTS, 'audit',
    )
    sample.refresh_from_db()
    sample_record = build_sample_record(sample)
    survey_record = build_survey_record(survey)
    return success_response(
        request, body,
        data_id=f'sampled_trees_{survey_id}',
        row_id=ts_id,
        patches=[
            row_patch('samples', sample_record[0], sample_record),
            row_patch('surveys', survey_record[0], survey_record),
        ],
        deletes=[row_delete(f'sampled_trees_{survey_id}', ts_id)],
    )


def _next_area_numbers(grid) -> dict[int, int]:
    """Map region_id → suggested next area number within `grid`.

    Area numbers are unique per (grid, region).  `SampleArea.number` is free
    text (usually numeric like '1', but may be prefixed, e.g. 'C1'): the
    suggestion is max(integer-valued numbers in this grid+region) + 1, ignoring
    non-integer labels; a region with no integer-valued areas starts at 1.
    Suggestion only — the field stays free text and the writer may override it.
    """
    max_by_region: dict[int, int] = {}
    for region_id, number in (SampleArea.objects
                              .filter(sample_grid=grid)
                              .values_list('parcel__region_id', 'number')):
        n = int_or_none(number)
        if n is not None:
            max_by_region[region_id] = max(max_by_region.get(region_id, 0), n)
    return {rid: mx + 1 for rid, mx in max_by_region.items()}


def _area_number_taken(grid, region_id, number, exclude_id=None) -> bool:
    """True if another area in this grid+region already uses `number`.

    Area numbers are unique per (grid, region) — see SampleArea.  Pass the
    edited area's id as `exclude_id` so an unchanged save doesn't collide with
    itself.
    """
    qs = SampleArea.objects.filter(
        sample_grid=grid, parcel__region_id=region_id, number=number,
    )
    if exclude_id is not None:
        qs = qs.exclude(id=exclude_id)
    return qs.exists()


def _parcel_from_names(parcels, compresa, particella):
    """Resolve (compresa, particella) names to a Parcel from `parcels`, or None.

    Matches the CSV import's key: case-insensitive region name + exact parcel
    name.  `parcels` must have `region` pre-loaded.
    """
    compresa = (compresa or '').strip().lower()
    particella = (particella or '').strip()
    if not compresa or not particella:
        return None
    return next(
        (p for p in parcels
         if p.region.name.lower() == compresa and p.name == particella),
        None,
    )


@login_required
@require_writer
def area_form_view(request, area_id: int | None = None):
    """Form fragment for adding or editing a SampleArea.

    Query params (add path): ?grid=<id>&lat=...&lon=... (lat/lon optional —
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
        grid_id = int_or_none(request.GET.get('grid'))
        if grid_id is None:
            raise Http404('grid required')
        grid = SampleGrid.objects.filter(id=grid_id).first()
        if grid is None:
            raise Http404
    initial_lat = request.GET.get(FIELD_LAT, '')
    initial_lon = request.GET.get(FIELD_LON, '')

    regions = list(Region.objects.order_by('name'))
    parcels = sorted(Parcel.objects.select_related('region'),
                     key=parcel_sort_key)
    next_by_region = _next_area_numbers(grid)
    for p in parcels:
        p.next_number = next_by_region.get(p.region_id, 1)

    # Pre-select region+parcel.  Editing: the area's own parcel.  Adding via
    # map-click: the parcel under the click, passed as compresa+particella
    # names.  Adding via the button: nothing pre-selected.
    sel_parcel = area.parcel if area else _parcel_from_names(
        parcels, request.GET.get(FIELD_COMPRESA), request.GET.get(FIELD_PARTICELLA),
    )
    return JsonResponse({HTML: render_to_string(
        'campionamenti/_area_form.html', {
            FIELD_AREA: area, 'grid': grid, 'default_radius_m': DEFAULT_RADIUS_M,
            'regions': regions, 'parcels': parcels,
            'selected_region_id': sel_parcel.region_id if sel_parcel else None,
            'selected_parcel_id': sel_parcel.id if sel_parcel else None,
            'lat': coord_float(parse_decimal(area.lat if area else initial_lat)),
            'lon': coord_float(parse_decimal(area.lon if area else initial_lon)),
        }, request=request,
    )})


@login_required
@require_writer
@require_POST
def area_save_view(request):
    """Create or update a SampleArea."""
    body, error = parse_json_body(request)
    if error:
        return error
    try:
        grid_id = int(body[FIELD_SAMPLE_GRID_ID])
        parcel_id = int(body[FIELD_PARCEL_ID])
        number = (body.get(FIELD_NUMBER) or '').strip()
        lat = coord_float(parse_decimal(body[FIELD_LAT]))
        lon = coord_float(parse_decimal(body[FIELD_LON]))
        r_m = int(body.get(FIELD_R_M) or DEFAULT_RADIUS_M)
        if lat is None or lon is None:
            raise ValueError
    except (KeyError, ValueError, TypeError):
        return validation_error([S.ERROR_GENERIC])

    if not number:
        return validation_error([S.ERR_AREA_NUMBER_REQUIRED])

    altitude_raw = body.get(FIELD_ALTITUDE_M)
    altitude = None
    if altitude_raw not in (None, '', 'null'):
        try:
            altitude = int(altitude_raw)
        except (ValueError, TypeError):
            altitude = None
    note = (body.get(FIELD_NOTE) or '').strip()

    grid = SampleGrid.objects.filter(id=grid_id).first()
    parcel = Parcel.objects.filter(id=parcel_id).first()
    if grid is None or parcel is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    area_id = body.get(ROW_ID)
    area_id = int(area_id) if area_id else None

    if _area_number_taken(grid, parcel.region_id, number, exclude_id=area_id):
        return validation_error([S.ERR_AREA_NUMBER_DUPLICATE])

    with transaction.atomic():
        if area_id:
            area = SampleArea.objects.select_for_update().filter(
                id=area_id, sample_grid=grid,
            ).first()
            if area is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            if area.version != submitted_version(body):
                fresh_area = SampleArea.objects.select_related('parcel__region').get(id=area.id)
                return conflict_response(
                    data_id='sample_areas', row_id=fresh_area.id,
                    record=build_sample_area_record(fresh_area),
                )
            area.parcel = parcel
            area.number = number
            area.lat = lat
            area.lon = lon
            area.altitude_m = altitude
            area.r_m = r_m
            area.note = note
            area.version += 1
            area.save()
        else:
            area = SampleArea.objects.create(
                sample_grid=grid, parcel=parcel, number=number,
                lat=lat, lon=lon, altitude_m=altitude, r_m=r_m, note=note,
            )
        # `surveys` digest's N. aree totali depends on the area count
        # per grid → must invalidate surveys on area writes too.
        mark_stale(
            'sample_areas', 'grids', 'surveys',
            *BOSCO_DENDROMETRY_DIGESTS, 'audit',
        )

    # Reload with select_related so build_sample_area_record doesn't N+1.
    area = SampleArea.objects.select_related('parcel__region').get(id=area.id)
    area_record = build_sample_area_record(area)
    grid_record = build_grid_record(grid)
    survey_records = [
        build_survey_record(sv)
        for sv in Survey.objects.filter(sample_grid=grid)
    ]
    return success_response(
        request, body,
        data_id='sample_areas', row_id=area.id,
        patches=[
            row_patch('sample_areas', area.id, area_record),
            row_patch('grids', grid_record[0], grid_record),
            *row_patches('surveys', survey_records),
        ],
    )


@login_required
@require_writer
@require_POST
def area_delete_view(request, area_id: int):
    """Delete a SampleArea.  Refused if any Sample references it."""
    body, error = parse_json_body(request)
    if error:
        return error
    area = SampleArea.objects.select_related('sample_grid').filter(
        id=area_id,
    ).first()
    if area is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if area.version != submitted_version(body):
        return conflict_response(
            data_id='sample_areas', row_id=area.id,
            record=build_sample_area_record(area),
        )
    if Sample.objects.filter(sample_area=area).exists():
        return validation_error([S.ERR_AREA_IN_USE])
    grid = area.sample_grid
    area.delete()
    # See area_save_view: surveys digest depends on the per-grid area count.
    mark_stale('sample_areas', 'grids', 'surveys', 'audit')
    grid_record = build_grid_record(grid)
    survey_records = [
        build_survey_record(sv)
        for sv in Survey.objects.filter(sample_grid=grid)
    ]
    return success_response(
        request, body,
        data_id='sample_areas', row_id=area_id,
        patches=[
            row_patch('grids', grid_record[0], grid_record),
            *row_patches('surveys', survey_records),
        ],
        deletes=[row_delete('sample_areas', area_id)],
    )


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

    species = list(Species.objects.filter(active=True).order_by(FIELD_SORT_ORDER))
    tree = ts.tree if ts else None
    is_coppice = area.parcel.eclass.coppice
    default_species_id = None if ts else _default_species_id(species, is_coppice)
    default_species = next((sp for sp in species if sp.id == default_species_id), None)
    prior_trees, next_number = _prior_trees_for_area(
        area, exclude_ts_id=ts_id, current_sample=sample,
    )

    return render_to_string('campionamenti/_tree_form.html', {
        'ts': ts,
        'tree': tree,
        'sample': sample,
        FIELD_AREA: area,
        'survey': survey,
        FIELD_SPECIES: species,
        'sample_date': sample.date if sample else date_type.today(),
        'prior_trees': prior_trees,
        'next_number': next_number,
        'fustaia_default': not is_coppice,
        'default_species_id': default_species_id,
        # Shared _tree_fields.html context.
        'selected_species_id': default_species_id,
        'is_edit': bool(ts),
        'show_ceduo': True,
        'show_l10': True,
        'ceduo_checked': tree.coppice if tree else False,
        'pai_checked': tree.preserved if tree else False,
        'edit_species_name': tree.species.common_name if tree else '',
        'edit_species_id': tree.species_id if tree else '',
        'edit_species_density': tree.species.density if tree else '',
        'edit_species_pressler_default': tree.species.pressler_default if tree else '',
        'show_pressler': True,
        'pressler_coeff': (
            ts.pressler_coeff if ts else (
                default_species.pressler_default if default_species
                else PRESSLER_DEFAULT
            )
        ),
        'd_cm': ts.d_cm if ts else '',
        'h_m': ts.h_m if ts else '',
        'l10_mm': ts.l10_mm if ts else 0,
        'lat': round((tree.lat if tree and tree.lat is not None
                      else area.lat) if ts else area.lat, 5),
        'lon': round((tree.lon if tree and tree.lon is not None
                      else area.lon) if ts else area.lon, 5),
    }, request=request)


def _default_species_id(species, parcel_is_coppice):
    """Pick the default species for a newly added tree.

    Fustaia parcel → `S.SPECIES_DEFAULT_HIGHFOREST`, ceduo parcel →
    `S.SPECIES_DEFAULT_COPPICE`.  Match on Species.common_name with
    `__iexact` first, then a substring fallback for variants like
    "Abete bianco".  Returns `None` when the species list is empty;
    otherwise the first species in the list if neither match hits.
    """
    if not species:
        return None
    target = (
        S.SPECIES_DEFAULT_COPPICE if parcel_is_coppice
        else S.SPECIES_DEFAULT_HIGHFOREST
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


def _prior_trees_for_area(area, exclude_ts_id=None, current_sample=None):
    """Build the list backing the "Numero albero" pulldown.

    Returns (prior_trees, next_number) where:
      - prior_trees is a list of dicts (one per distinct Tree in this area,
        most-recent measurement first within each tree) sorted by tree number;
      - next_number is `max(tree_sample.number)+1` across all samples in this
        area, or 1 if empty.

    Tree numbers already present in `current_sample` are omitted from
    prior_trees because the active sample cannot contain the same number
    twice, even when historical data has separate Tree rows for that
    number.

    Same `number` is reused across surveys for the same physical tree
    (cross-sample identity, per campionamenti.md §"Cross-sample tree
    identity").  For coppice trees the entry also carries `next_shoot`
    = max(shoot for this tree across all samples) + 1, which the form
    uses as the starting pollone number when the operator picks this
    tree from the pulldown.
    """
    qs = (TreeSample.objects
            .filter(sample__sample_area_id=area.id)
            .select_related('tree__species', 'sample__survey')
            .order_by('tree_id', '-sample__date', '-shoot'))
    if exclude_ts_id:
        qs = qs.exclude(id=exclude_ts_id)

    by_tree = {}
    max_number = 0
    max_shoot_by_tree = {}
    unavailable_numbers = set()
    current_sample_id = current_sample.id if current_sample else None
    for ts in qs:
        max_number = max(max_number, ts.number or 0)
        max_shoot_by_tree[ts.tree_id] = max(
            max_shoot_by_tree.get(ts.tree_id, 0), ts.shoot or 0,
        )
        if current_sample_id is not None and ts.sample_id == current_sample_id:
            unavailable_numbers.add(ts.number)
        if ts.tree_id in by_tree:
            continue       # already have the most-recent measurement for this tree
        by_tree[ts.tree_id] = {
            'tree_id': ts.tree_id,
            FIELD_NUMBER: ts.number,
            FIELD_SPECIES_ID: ts.tree.species_id,
            'species_common_name': ts.tree.species.common_name,
            FIELD_COPPICE: ts.tree.coppice,
            FIELD_LAT: ts.tree.lat,
            FIELD_LON: ts.tree.lon,
            'last_d_cm': ts.d_cm,
            'last_h_m': ts.h_m,
            'last_survey_name': ts.sample.survey.name,
            'last_sample_date': ts.sample.date,
        }
    by_tree = {
        tid: row for tid, row in by_tree.items()
        if row[FIELD_NUMBER] not in unavailable_numbers
    }
    for tid, row in by_tree.items():
        row[FIELD_NEXT_SHOOT] = max_shoot_by_tree.get(tid, 0) + 1
    prior = sorted(by_tree.values(), key=lambda r: r[FIELD_NUMBER])
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
    case the existing tree's species / coppice / lat / lon are reused on
    save, regardless of what the body's species_id / fustaia / lat / lon
    fields contain (they may be present but the client should have locked
    them in the UI — server treats the existing Tree as authoritative).
    """
    errors = []
    ts_id = body.get(ROW_ID)
    ts_id = int(ts_id) if ts_id else None

    # The field is optional in the wire format: missing keeps the
    # existing sample.date or defaults a new Sample to today.  A submitted
    # date may create a Sample, but may not rewrite an existing one.
    date_raw = body.get(FIELD_DATE)
    parsed_date = None
    if date_raw not in (None, '', 'null'):
        try:
            parsed_date = date_type.fromisoformat(str(date_raw))
        except (ValueError, TypeError):
            errors.append(S.ERR_DATE_INVALID)

    pressler_coeff = parse_decimal(body.get(FIELD_PRESSLER_COEFF))
    if pressler_coeff is None:
        pressler_coeff = PRESSLER_DEFAULT
    if pressler_coeff <= 0:
        errors.append(S.ERR_PRESSLER_POSITIVE)

    coppice = str(body.get(FIELD_HIGHFOREST, 'true')).lower() in ('false', '0', 'no')

    # tree_pick: 'new' or an integer Tree id.  Older payloads without
    # tree_pick default to 'new' so the existing call sites keep working.
    tree_pick_raw = body.get(FIELD_TREE_PICK, 'new')
    tree_pick_existing_id = None
    if tree_pick_raw not in (None, '', 'new'):
        try:
            tree_pick_existing_id = int(tree_pick_raw)
        except (ValueError, TypeError):
            errors.append(S.ERR_TREE_NUMBER_REQUIRED)

    sample_area_id = int_or_none(body.get(FIELD_SAMPLE_AREA_ID))
    survey_id = int_or_none(body.get(FIELD_SURVEY_ID))
    species_id = (int_or_none(body.get(FIELD_SPECIES_ID))
                  if body.get(FIELD_SPECIES_ID) else None)
    number = int_or_none(body.get(FIELD_NUMBER, 0) or 0)
    if sample_area_id is None:
        errors.append(S.IPSO_ERR_FIELD_REQUIRED.format(S.COL_SAMPLE_AREA))
    if survey_id is None:
        errors.append(S.ERR_CSV_SURVEY_REQUIRED)
    if body.get(FIELD_SPECIES_ID) and species_id is None:
        errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.COL_SPECIES))
    if number is None:
        errors.append(S.ERR_TREE_NUMBER_REQUIRED)

    if coppice:
        shoots, shoot_errors = _parse_shoots(body.get(FIELD_SHOOTS))
        errors.extend(shoot_errors)
        # Coppice rows carry no per-tree volume / mass — Tabacchi only
        # applies to fustaia, per the spec's "V/m blank for ceduo".
        d_cm = 0
        h_m = Decimal('0')
        l10_mm = 0
    else:
        shoots = []
        try:
            d_cm = int(body[FIELD_D_CM])
        except (KeyError, ValueError, TypeError):
            d_cm = 0
            errors.append(S.ERR_D_POSITIVE)

        h_m = parse_decimal(body.get(FIELD_H_M)) or Decimal('0')

        l10_mm = int_or_none(body.get(FIELD_L10_MM, 0) or 0)
        if l10_mm is None:
            l10_mm = 0
            errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.COL_L10_MM))
        values = normalize_sample_tree_values(
            number=number,
            d_cm=d_cm,
            h_m=h_m,
            shoot=0,
            l10_mm=l10_mm,
            pressler_coeff=pressler_coeff,
        )
        if values is None:
            if d_cm <= 0:
                errors.append(S.ERR_D_POSITIVE)
            if h_m <= 0:
                errors.append(S.ERR_H_POSITIVE)
            if l10_mm < 0:
                errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.COL_L10_MM))
        else:
            d_cm = values.d_cm
            h_m = values.h_m
            l10_mm = values.l10_mm
            pressler_coeff = values.pressler_coeff
    parsed = {
        FIELD_SAMPLE_AREA_ID: sample_area_id,
        FIELD_SURVEY_ID: survey_id,
        FIELD_SPECIES_ID: species_id,
        FIELD_NUMBER: number or 0,
        FIELD_D_CM: d_cm,
        FIELD_H_M: h_m,
        FIELD_L10_MM: l10_mm,
        FIELD_PRESSLER_COEFF: pressler_coeff,
        FIELD_VOLUME_M3: parse_decimal(body.get(FIELD_VOLUME_M3)) if not coppice else None,
        FIELD_MASS_Q: parse_decimal(body.get(FIELD_MASS_Q)) if not coppice else None,
        FIELD_LAT: coord_float(parse_decimal(body.get(FIELD_LAT))),
        FIELD_LON: coord_float(parse_decimal(body.get(FIELD_LON))),
        FIELD_PRESERVED: is_truthy(body.get(FIELD_PRESERVED)),
        FIELD_COPPICE: coppice,
        FIELD_SHOOTS: shoots,
        FIELD_TREE_PICK_EXISTING_ID: tree_pick_existing_id,
        FIELD_DATE: parsed_date,
    }
    if not parsed[FIELD_NUMBER]:
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
            shoot_num = int(item.get(FIELD_SHOOT, 0))
            d_cm = int(item.get(FIELD_D_CM, 0))
            l10_mm = int(item.get(FIELD_L10_MM, 0) or 0)
            standard = bool(item.get(FIELD_STANDARD))
        except (ValueError, TypeError):
            errors.append(S.ERR_D_POSITIVE)
            continue
        h_m = parse_decimal(item.get(FIELD_H_M)) or Decimal('0')
        values = normalize_sample_shoot_values(
            shoot=shoot_num, d_cm=d_cm, h_m=h_m, l10_mm=l10_mm,
        )
        if values is None:
            if d_cm <= 0:
                errors.append(S.ERR_D_POSITIVE)
            if h_m <= 0:
                errors.append(S.ERR_H_POSITIVE)
            if shoot_num < 0:
                errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.COL_COPPICE_SHOOT))
            if l10_mm < 0:
                errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(S.COL_L10_MM))
            continue
        shoots.append({
            FIELD_SHOOT: values.shoot, FIELD_STANDARD: standard,
            FIELD_D_CM: values.d_cm, FIELD_H_M: values.h_m,
            FIELD_L10_MM: values.l10_mm,
        })
    return shoots, errors


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
    area = SampleArea.objects.select_related(FIELD_PARCEL).get(
        id=parsed[FIELD_SAMPLE_AREA_ID],
    )
    survey = Survey.objects.get(id=parsed[FIELD_SURVEY_ID])
    if area.sample_grid_id != survey.sample_grid_id:
        return None
    sample, _ = Sample.objects.get_or_create(
        sample_area=area, survey=survey,
        defaults={FIELD_DATE: parsed.get(FIELD_DATE) or date_type.today()},
    )
    return sample


def _sample_date_conflict_error(sample):
    area = sample.sample_area
    parcel = area.parcel
    return S.ERR_SAMPLE_DATE_CONFLICT.format(
        parcel.region.name, parcel.name, area.number, sample.date.isoformat(),
    )


def _update_tree_sample(ts_id, sample, parsed, body, request):
    ts = (TreeSample.objects
          .select_for_update()
          .select_related(
              'sample__survey', 'sample__sample_area__parcel__region',
              'tree__species', 'tree__parcel',
          )
          .get(id=ts_id))
    if ts.version != submitted_version(body):
        return conflict_response(
            data_id=f'sampled_trees_{ts.sample.survey_id}', row_id=ts.id,
            record=build_tree_sample_record(ts),
            html=_render_tree_form(request, ts.id, None, None),
        )
    ts.sample = sample
    ts.number = parsed[FIELD_NUMBER]
    if parsed[FIELD_COPPICE]:
        # Coppice edit form sends exactly one shoot row (the one being
        # edited); the multi-row "Aggiungi pollone" path is add-only.
        sh = parsed[FIELD_SHOOTS][0]
        ts.shoot = sh[FIELD_SHOOT]
        ts.standard = sh[FIELD_STANDARD]
        ts.d_cm = sh[FIELD_D_CM]
        ts.h_m = sh[FIELD_H_M]
        ts.l10_mm = sh[FIELD_L10_MM]
        ts.pressler_coeff = parsed[FIELD_PRESSLER_COEFF]
        ts.volume_m3 = None
        ts.mass_q = None
    else:
        ts.d_cm = parsed[FIELD_D_CM]
        ts.h_m = parsed[FIELD_H_M]
        ts.l10_mm = parsed[FIELD_L10_MM]
        ts.pressler_coeff = parsed[FIELD_PRESSLER_COEFF]
        ts.volume_m3 = parsed[FIELD_VOLUME_M3]
        ts.mass_q = parsed[FIELD_MASS_Q]
    ts.version += 1
    ts.save()
    # Tree fields that can change on edit.
    tree = ts.tree
    tree.species_id = parsed[FIELD_SPECIES_ID]
    tree.preserved = parsed[FIELD_PRESERVED]
    tree.lat = parsed[FIELD_LAT]
    tree.lon = parsed[FIELD_LON]
    tree.version += 1
    tree.save()
    return ts


def _validation_error(errors, ts_id, request, body):
    survey_id = int_or_none(body.get(FIELD_SURVEY_ID))
    area_id = int_or_none(body.get(FIELD_SAMPLE_AREA_ID))
    # Skip the form re-render when the survey/area combo is itself
    # invalid (Http404 from _render_tree_form would mask the real
    # error).  Client just shows the error message in that case.
    try:
        html = _render_tree_form(request, ts_id, survey_id, area_id)
    except Http404:
        html = ''
    return validation_error(errors, html=html)


# ---------------------------------------------------------------------------
# Grid + Survey CRUD
# ---------------------------------------------------------------------------

@login_required
@require_writer
def grid_form_view(request):
    """Return the HTML fragment for the "Nuova griglia" modal — three
    creation paths (empty / auto / csv) per `campionamenti.md` §1.

    The CSV path imports INTO an existing grid (mirrors the survey CSV
    import); the target-grid dropdown needs the current list.
    """
    grids = SampleGrid.objects.order_by('-modified_at')
    return JsonResponse({HTML: render_to_string(
        'campionamenti/_grid_modal.html', {'grids': grids}, request=request,
    )})


@login_required
@require_writer
@require_POST
def grid_save_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    name = (body.get(FIELD_NAME) or '').strip()
    description = (body.get(FIELD_DESCRIPTION) or '').strip()
    if not name:
        return validation_error([S.ERR_GRID_NAME_REQUIRED])
    return save_model_response(
        request, body, model=SampleGrid, data_id='grids',
        values={FIELD_NAME: name, FIELD_DESCRIPTION: description},
        row_fn=build_grid_record, stale=('grids', 'audit'),
        unique_field=FIELD_NAME, unique_value=name,
        unique_error=S.ERR_GRID_NAME_DUPLICATE,
    )


@login_required
@require_writer
def survey_form_view(request):
    """Return the HTML fragment for the "Nuovo rilevamento" modal — two
    creation paths (empty / csv) per `campionamenti.md` §2.
    """
    grids = SampleGrid.objects.order_by('-modified_at')
    surveys = Survey.objects.order_by('-modified_at')
    return JsonResponse({HTML: render_to_string(
        'campionamenti/_survey_modal.html', {
            'grids': grids, 'surveys': surveys,
            'sample_grid_unstructured_value': SAMPLE_GRID_UNSTRUCTURED,
        }, request=request,
    )})


@login_required
@require_writer
@require_POST
def grid_edit_view(request, grid_id: int):
    """Edit a grid's name / description (no cascade)."""
    grid = SampleGrid.objects.filter(id=grid_id).first()
    if grid is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    body, error = parse_json_body(request)
    if error:
        return error
    name = (body.get(FIELD_NAME) or '').strip()
    description = (body.get(FIELD_DESCRIPTION) or '').strip()
    if not name:
        return validation_error([S.ERR_GRID_NAME_REQUIRED])
    return save_model_response(
        request, body, model=SampleGrid, data_id='grids', row_id=grid.id,
        values={FIELD_NAME: name, FIELD_DESCRIPTION: description},
        row_fn=build_grid_record, stale=('grids', 'audit'),
        unique_field=FIELD_NAME, unique_value=name,
        unique_error=S.ERR_GRID_NAME_DUPLICATE,
    )


@login_required
@require_writer
@require_POST
def grid_delete_view(request, grid_id: int):
    """Delete a grid.  Refused if any Survey references it (Survey.sample_grid
    is PROTECT — the only way to "force" delete a populated grid is to delete
    its surveys first)."""
    body, error = parse_json_body(request)
    if error:
        return error
    grid = SampleGrid.objects.filter(id=grid_id).first()
    if grid is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if grid.version != submitted_version(body):
        return conflict_response(
            data_id='grids', row_id=grid.id, record=build_grid_record(grid),
        )
    if Survey.objects.filter(sample_grid=grid).exists():
        return validation_error([S.ERR_GRID_IN_USE])
    with transaction.atomic():
        # SampleArea cascades.
        grid.delete()
        mark_stale('grids', 'sample_areas', 'audit')
    return success_response(
        request, body,
        data_id='grids', row_id=grid_id,
        deletes=[row_delete('grids', grid_id)],
    )


@login_required
@require_writer
@require_POST
def survey_edit_view(request, survey_id: int):
    """Edit a survey's name / description."""
    survey = Survey.objects.filter(id=survey_id).first()
    if survey is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    body, error = parse_json_body(request)
    if error:
        return error
    name = (body.get(FIELD_NAME) or '').strip()
    description = (body.get(FIELD_DESCRIPTION) or '').strip()
    if not name:
        return validation_error([S.ERR_SURVEY_NAME_REQUIRED])
    return save_model_response(
        request, body, model=Survey, data_id='surveys', row_id=survey.id,
        values={FIELD_NAME: name, FIELD_DESCRIPTION: description},
        row_fn=build_survey_record,
        stale=('surveys', *BOSCO_DENDROMETRY_DIGESTS, 'audit'),
        unique_field=FIELD_NAME, unique_value=name,
        unique_error=S.ERR_SURVEY_NAME_DUPLICATE,
    )


# ---------------------------------------------------------------------------
# CSV imports (Bucket 3) — Grid CSV + Tree-and-sample CSV
# ---------------------------------------------------------------------------

@login_required
@require_writer
@require_POST
def grid_csv_import_view(request):
    """Upload a CSV → append N SampleAreas to an existing SampleGrid.

    JSON fields:
      - sample_grid_id (required)
      - file           (required, base64 CSV bytes)

    Mirrors tree_csv_import_view's "import INTO existing parent" shape
    so a single grid can be populated incrementally from multiple
    files.  Rejects rows that would duplicate an existing
    (parcel, number) within the target grid; rejects rows that
    duplicate each other within the same upload.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    grid_id = body.get(FIELD_SAMPLE_GRID_ID)
    try:
        upload = csv_io.json_file_bytes(body, FIELD_FILE)
    except csv_io.CsvError as e:
        return validation_error([str(e)])

    if not grid_id:
        return validation_error([S.ERR_CSV_GRID_REQUIRED])
    if upload is None:
        return validation_error([S.ERR_CSV_FILE_REQUIRED])

    grid_pk = int_or_none(grid_id)
    if grid_pk is None:
        return validation_error([S.ERR_CSV_GRID_REQUIRED])
    grid = SampleGrid.objects.filter(id=grid_pk).first()
    if grid is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    try:
        reader = csv_io.read(upload)
    except csv_io.CsvError as e:
        return validation_error([str(e)])
    cols, missing = csv_grid.resolve_columns(reader.fieldnames or [])
    if missing:
        return validation_error([S.ERR_CSV_MISSING_COLS.format(', '.join(missing))])

    parsed_rows, errors = csv_grid.validate_rows(
        reader, cols, csv_grid.db_indexes(grid),
    )
    if errors:
        return csv_error_list(errors)
    if not parsed_rows:
        return validation_error([S.ERR_CSV_EMPTY])

    csv_grid.apply(grid, parsed_rows)

    grid.refresh_from_db()
    area_qs = SampleArea.objects.filter(sample_grid=grid) \
                                 .select_related('parcel__region')
    grid_record = build_grid_record(grid)
    area_records = [build_sample_area_record(sa) for sa in area_qs]
    survey_records = [
        build_survey_record(sv)
        for sv in Survey.objects.filter(sample_grid=grid)
    ]
    return success_response(
        request, body,
        data_id='grids', row_id=grid.id,
        patches=[
            row_patch('grids', grid_record[0], grid_record),
            *row_patches('sample_areas', area_records),
            *row_patches('surveys', survey_records),
        ],
        extra={'n_areas': len(parsed_rows)},
    )


@login_required
@require_writer
@require_POST
def tree_csv_import_view(request):
    """Upload a CSV → create Samples + Trees + TreeSamples on a survey.

    JSON fields:
      - survey_id (required)
      - default_date (required if CSV lacks a Data column)
      - file (required, base64 CSV bytes)
    """
    body, error = parse_json_body(request)
    if error:
        return error
    survey_id = body.get(FIELD_SURVEY_ID)
    try:
        upload = csv_io.json_file_bytes(body, FIELD_FILE)
    except csv_io.CsvError as e:
        return validation_error([str(e)])
    default_date_str = (body.get(FIELD_DEFAULT_DATE) or '').strip()

    if not survey_id:
        return validation_error([S.ERR_CSV_SURVEY_REQUIRED])
    if upload is None:
        return validation_error([S.ERR_CSV_FILE_REQUIRED])

    survey_pk = int_or_none(survey_id)
    if survey_pk is None:
        return validation_error([S.ERR_CSV_SURVEY_REQUIRED])
    survey = Survey.objects.filter(id=survey_pk).select_related(
        'sample_grid',
    ).first()
    if survey is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if survey.sample_grid_id is None:
        return validation_error([S.ERR_SURVEY_STRUCTURED_REQUIRED])

    try:
        reader = csv_io.read(upload, csv_trees.TREE_CSV_REQUIRED)
    except csv_io.CsvError as e:
        return validation_error([str(e)])

    has_date_column = bool(reader) and S.CSV_COL_DATA in reader[0]
    if not has_date_column and not default_date_str:
        return validation_error([S.ERR_CSV_DATE_REQUIRED])
    default_date = None
    if default_date_str:
        try:
            default_date = date_type.fromisoformat(default_date_str)
        except ValueError:
            return validation_error([S.ERR_CSV_DATE_REQUIRED])

    parsed, errors = csv_trees.validate_rows(
        reader, csv_trees.db_indexes(survey),
        has_date_column=has_date_column, default_date=default_date,
    )
    if errors:
        return csv_error_list(errors)
    if not parsed:
        return validation_error([S.ERR_CSV_EMPTY])

    counts = csv_trees.apply(survey, parsed)

    return success_response(
        request, body,
        data_id='surveys', row_id=survey.id,
        extra=counts,
    )


@login_required
@require_writer
@require_POST
def survey_delete_view(request, survey_id: int):
    """Delete a survey.  Cascades to its Samples and their TreeSamples
    (per the FK on_delete=CASCADE chain in models.py).  Tree rows
    remain (TreeSample.tree is PROTECT).
    """
    body, error = parse_json_body(request)
    if error:
        return error
    survey = Survey.objects.filter(id=survey_id).first()
    if survey is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if survey.version != submitted_version(body):
        return conflict_response(
            data_id='surveys', row_id=survey.id,
            record=build_survey_record(survey),
        )
    with transaction.atomic():
        survey.delete()
        mark_stale(
            f'sampled_trees_{survey_id}', 'samples', 'surveys', 'grids',
            *BOSCO_DENDROMETRY_DIGESTS, 'audit',
        )
    return success_response(
        request, body,
        data_id='surveys', row_id=survey_id,
        deletes=[row_delete('surveys', survey_id)],
    )


@login_required
@require_writer
@require_POST
def survey_save_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    name = (body.get(FIELD_NAME) or '').strip()
    description = (body.get(FIELD_DESCRIPTION) or '').strip()
    grid_id = body.get(FIELD_SAMPLE_GRID_ID)

    if not name:
        return validation_error([S.ERR_SURVEY_NAME_REQUIRED])
    if not grid_id:
        return validation_error([S.ERR_SURVEY_GRID_REQUIRED])
    if grid_id == SAMPLE_GRID_UNSTRUCTURED:
        grid = None
    else:
        try:
            grid = SampleGrid.objects.get(id=int(grid_id))
        except (SampleGrid.DoesNotExist, ValueError):
            return validation_error([S.ERR_SURVEY_GRID_REQUIRED])

    return save_model_response(
        request, body, model=Survey, data_id='surveys',
        values={FIELD_NAME: name, 'sample_grid': grid, FIELD_DESCRIPTION: description},
        row_fn=build_survey_record,
        stale=('surveys', 'grids', *BOSCO_DENDROMETRY_DIGESTS, 'audit'),
        unique_field=FIELD_NAME, unique_value=name,
        unique_error=S.ERR_SURVEY_NAME_DUPLICATE,
        extra_patches=lambda _survey: (
            [row_patch('grids', grid.id, build_grid_record(grid))]
            if grid is not None else []
        ),
    )


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
            "lat": 38.51, "lon": 16.11},
           ...
        ],
        "nonce": "..."
      }

    Per-point compresa+particella resolves to a Parcel.  Resolution
    failure aborts the entire commit (no partial state).
    """
    body, error = parse_json_body(request)
    if error:
        return error
    name = (body.get(FIELD_NAME) or '').strip()
    description = (body.get(FIELD_DESCRIPTION) or '').strip()
    points = body.get(FIELD_POINTS) or []
    try:
        r_m = int(body.get(FIELD_R_M) or DEFAULT_RADIUS_M)
    except (ValueError, TypeError):
        r_m = DEFAULT_RADIUS_M

    if not name:
        return validation_error([S.ERR_GRID_NAME_REQUIRED])
    if SampleGrid.objects.filter(name=name).exists():
        return validation_error([S.ERR_GRID_NAME_DUPLICATE])
    if not points:
        return validation_error([S.ERR_GRID_AUTO_NO_POINTS])

    resolved, err = _resolve_grid_points(points)
    if err:
        return validation_error([err])

    with transaction.atomic():
        grid = SampleGrid.objects.create(name=name, description=description)
        # Number areas per compresa (region), restarting at 1 in each, so
        # auto-generated numbers are unique per compresa like manual ones.
        per_region_count: dict[int, int] = {}
        for pt, parcel in resolved:
            n = per_region_count.get(parcel.region_id, 0) + 1
            per_region_count[parcel.region_id] = n
            SampleArea.objects.create(
                sample_grid=grid,
                parcel=parcel,
                number=str(n),
                lat=pt[FIELD_LAT], lon=pt[FIELD_LON],
                r_m=r_m,
                note='',
            )
        mark_stale('grids', 'sample_areas', 'audit')

    area_qs = SampleArea.objects.filter(sample_grid=grid) \
                                 .select_related('parcel__region')
    grid_record = build_grid_record(grid)
    area_records = [build_sample_area_record(sa) for sa in area_qs]
    return success_response(
        request, body,
        data_id='grids', row_id=grid.id,
        patches=[
            row_patch('grids', grid_record[0], grid_record),
            *row_patches('sample_areas', area_records),
        ],
    )


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
