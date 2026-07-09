"""Piano di taglio API views.

Read endpoints (digest passthrough), plan + plan-item CRUD, plan-level
and per-item CSV import/export, and the cantiere transition save view.

Form endpoints follow the standard Abies idiom (see
`apps.prelievi.views`): a GET returns an HTML fragment, a POST processes
the submission and returns a generic patches/deletes envelope or a
``validation_error`` / ``conflict`` payload.
"""

import re
from dataclasses import dataclass
from datetime import date as date_type

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Q
from django.http import Http404, HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base import csv_io
from apps.base.numparse import coord_float, int_or_none, parse_decimal
from apps.base.responses import (
    conflict_response, csv_error_list, parse_json_body, row_delete,
    row_patch, save_model_response, submitted_version, success_response,
    validation_error,
)
from apps.base.digests import (
    build_harvest_plan_item_record,
    build_harvest_plan_record,
    build_tree_mark_record,
    mark_stale,
    serve_digest,
)
from apps.base.models import (
    HarvestPlan,
    HarvestPlanItem,
    HarvestPlanItemState,
    HarvestTransition,
    Parcel,
    Region,
    Species,
    Tree,
    TreeMark,
    parcel_sort_key,
    render_flag_note,
)
from apps.piano_di_taglio import csv_plan
from apps.piano_di_taglio.mark_import import (
    MarkImportRow, auto_advance_to_marked as _auto_advance_to_marked,
    csv_mark_fingerprint, import_mark_rows,
    next_mark_number as _next_mark_number,
    rematerialize_volume_marked as _rematerialize_volume_marked,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S
from config.constants import (
    DATA_ID,
    DIGEST_FUTURE_PRODUCTION,
    FIELD_ACC_M,
    FIELD_COPPICE_FILE,
    FIELD_D_CM,
    FIELD_DAMAGED,
    FIELD_DATE,
    FIELD_DESCRIPTION,
    FIELD_FILE,
    FIELD_HIGHFOREST_FILE,
    FIELD_H_M,
    FIELD_H_MEASURED,
    FIELD_HARVEST_PLAN_ID,
    FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_INTERVENTION_AREA_HA,
    FIELD_LAT,
    FIELD_LON,
    FIELD_MASS_Q,
    FIELD_NAME,
    FIELD_NOTE,
    FIELD_NUMBER,
    FIELD_OPEN,
    FIELD_OPERATOR,
    FIELD_PARCEL_ID,
    FIELD_PSR,
    FIELD_REGION_ID,
    FIELD_SPECIES_ID,
    FIELD_UNHEALTHY,
    FIELD_VOLUME_M3,
    FIELD_VOLUME_PLANNED_M3,
    FIELD_YEAR_END,
    FIELD_YEAR_PLANNED,
    FIELD_YEAR_START,
    HTML,
    RECORD,
    ROW_ID,
    STATUS,
    STATUS_NOT_FOUND,
    TRANSITION_RECORDS,
    VERSION,
    is_truthy,
)


_MARK_NUMBER_MISSING = object()


# ---------------------------------------------------------------------------
# Digest passthrough endpoints
# ---------------------------------------------------------------------------

@login_required
def plans_data_view(request):
    return serve_digest(request, 'harvest_plans')


@login_required
def items_data_view(request):
    return serve_digest(request, 'harvest_plan_items')


@login_required
def mark_trees_data_view(request, item_id: int):
    """Per-item tree-mark digest, lazily generated on first hit."""
    if item_id <= 0:
        raise Http404
    return serve_digest(request, f'mark_trees_{item_id}')


# ---------------------------------------------------------------------------
# Plan CRUD
# ---------------------------------------------------------------------------

@login_required
@require_writer
def plan_form_view(request, plan_id: int | None = None):
    """Form fragment for the "Crea vuoto" / edit-plan modal."""
    plan = None
    if plan_id:
        plan = HarvestPlan.objects.filter(id=plan_id).first()
        if plan is None:
            raise Http404
    return JsonResponse({HTML: render_to_string(
        'piano_di_taglio/_plan_form.html', {'plan': plan}, request=request,
    )})


@login_required
@require_writer
@require_POST
def plan_save_view(request):
    """Create or update a HarvestPlan (name + description + year range)."""
    body, error = parse_json_body(request)
    if error:
        return error
    plan_id = int_or_none(body.get(ROW_ID))

    name, description, year_start, year_end, errors = _parse_plan_body(body)
    if errors:
        return validation_error(errors)

    return save_model_response(
        request, body, model=HarvestPlan, data_id='harvest_plans', row_id=plan_id,
        values={
            FIELD_NAME: name,
            FIELD_DESCRIPTION: description,
            FIELD_YEAR_START: year_start,
            FIELD_YEAR_END: year_end,
        },
        row_fn=build_harvest_plan_record,
        stale=('harvest_plans', DIGEST_FUTURE_PRODUCTION, 'audit'),
        unique_field=FIELD_NAME, unique_value=name,
        unique_error=S.ERR_PLAN_NAME_DUPLICATE, unique_case_insensitive=True,
    )


@login_required
@require_writer
@require_POST
def plan_delete_view(request, plan_id: int):
    """Delete a HarvestPlan.  Gated: every item must be in state `planned`.

    The CSV-download forced-step is handled client-side: the JS download
    of `plan_export_view` precedes this call.  We still verify the gate
    server-side so a hand-crafted POST cannot bypass it.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    plan = HarvestPlan.objects.filter(id=plan_id).first()
    if plan is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if plan.version != submitted_version(body):
        return conflict_response(
            data_id='harvest_plans', row_id=plan.id,
            record=build_harvest_plan_record(plan),
        )

    bad = HarvestPlanItem.objects.filter(harvest_plan=plan).exclude(
        state=HarvestPlanItemState.PLANNED,
    ).exists()
    if bad:
        return validation_error([S.ERR_PLAN_HAS_ACTIVE_ITEMS])

    with transaction.atomic():
        # HarvestPlanItem and ParcelPlanDetail both cascade to HarvestPlan.
        plan.delete()
        mark_stale('harvest_plans', 'harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')
    return success_response(
        request, body,
        data_id='harvest_plans', row_id=plan_id,
        deletes=[row_delete('harvest_plans', plan_id)],
    )


# ---------------------------------------------------------------------------
# Plan CSV import — single endpoint dispatching on which base64 CSV field(s) are present
# ---------------------------------------------------------------------------


@login_required
@require_writer
@require_POST
def plan_csv_import_view(request):
    """Create a HarvestPlan from CSVs, or upsert CSV rows into an existing one.

    JSON fields:
      - ``harvest_plan_id`` — optional: target an existing plan for upsert.
        When set, ``name``/``description`` are ignored and the existing
        plan's are kept; rows from the CSVs are upserted (see below).
      - ``name``            — required when ``harvest_plan_id`` is absent.
      - ``description``     — optional plan description (create path only).
      - ``fustaia_file``    — optional base64 CSV bytes for piano.csv.
      - ``ceduo_file``      — optional base64 CSV bytes for ceduo.csv.

    At least one CSV must be attached.  Abies-exported rows carry an optional
    ``ID`` column so duplicate-looking rows round-trip exactly.  Hand-authored
    rows without ``ID`` keep the legacy upsert key
    ``(harvest_plan, parcel, year_planned)``.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    plan_id = int_or_none(body.get(FIELD_HARVEST_PLAN_ID))
    if plan_id is not None:
        target_plan = HarvestPlan.objects.filter(id=plan_id).first()
        if target_plan is None:
            return validation_error([S.ERR_PLAN_NOT_FOUND])
        name = target_plan.name
        description = target_plan.description
    else:
        target_plan = None
        name = (body.get(FIELD_NAME) or '').strip()
        description = (body.get(FIELD_DESCRIPTION) or '').strip()
        if not name:
            return validation_error([S.ERR_PLAN_NAME_REQUIRED])
        if HarvestPlan.objects.filter(name__iexact=name).exists():
            return validation_error([S.ERR_PLAN_NAME_DUPLICATE])

    try:
        fustaia_upload = csv_io.json_file_bytes(body, FIELD_HIGHFOREST_FILE)
        ceduo_upload = csv_io.json_file_bytes(body, FIELD_COPPICE_FILE)
    except csv_io.CsvError as e:
        return validation_error([str(e)])

    fustaia_rows = csv_plan.read_optional(
        fustaia_upload, csv_plan.HIGHFOREST_REQUIRED, csv_plan.HIGHFOREST_OPTIONAL)
    ceduo_rows = csv_plan.read_optional(
        ceduo_upload, csv_plan.COPPICE_REQUIRED, csv_plan.COPPICE_OPTIONAL)
    if fustaia_rows is None and ceduo_rows is None:
        return validation_error([S.ERR_CSV_NO_FILES])
    for result in (fustaia_rows, ceduo_rows):
        if isinstance(result, csv_plan.CsvError):
            return validation_error([result.message])

    idx = csv_plan.db_indexes()
    errors: list[str] = []
    fustaia_parsed = (
        csv_plan.parse_fustaia_rows(fustaia_rows, idx.parcels, idx.regions, errors)
        if fustaia_rows else []
    )
    ceduo_parsed = (
        csv_plan.parse_ceduo_rows(ceduo_rows, idx.parcels, errors)
        if ceduo_rows else []
    )
    if errors:
        return csv_error_list(errors)

    plan, n_items = csv_plan.apply(
        target_plan=target_plan, name=name, description=description,
        fustaia_parsed=fustaia_parsed, ceduo_parsed=ceduo_parsed,
    )

    return success_response(
        request, body,
        data_id='harvest_plans', row_id=plan.id,
        patches=[row_patch('harvest_plans', plan.id, build_harvest_plan_record(plan))],
        extra={'n_items': n_items},
    )


# ---------------------------------------------------------------------------
# Plan-level Esporta — zip of fustaia.csv + ceduo.csv
# ---------------------------------------------------------------------------

@login_required
def plan_export_view(request, plan_id: int):
    """Return a zip of the fustaia + ceduo CSVs in the active install locale's
    CSV format (``;``+``,`` for Italian, ``,``+``.`` for a dot-decimal locale),
    matching the per-table CSV export.

    Each CSV carries the full column set of the corresponding calendar
    table (display column names, not only the round-trip-required
    subset).  The importer accepts both naming conventions, so a freshly
    exported plan re-imports cleanly.
    """
    plan = HarvestPlan.objects.filter(id=plan_id).first()
    if plan is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    safe_name = _safe_filename(plan.name)
    return csv_io.zip_csv_response(
        csv_plan.render_plan_csvs(plan),
        f'piano_{safe_name}.zip',
    )


@login_required
def plan_section_export_view(request, plan_id: int, section: str):
    """Return one section CSV using the same renderer as the plan zip."""
    plan = HarvestPlan.objects.filter(id=plan_id).first()
    if plan is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    try:
        content = csv_plan.render_plan_section_csv(plan, section)
        filename = csv_plan.PLAN_SECTION_FILENAMES[section]
    except ValueError as exc:
        raise Http404 from exc

    response = HttpResponse(content, content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    response['Cache-Control'] = 'no-store'
    return response


# ---------------------------------------------------------------------------
# Plan-item CRUD
# ---------------------------------------------------------------------------

@login_required
def item_data_view(request, item_id: int):
    """Modal metadata for one HarvestPlanItem.

    Returns the current materialised digest row plus the item's
    HarvestTransition rows (Apri / Chiudi events).  Used by the
    View/Edit modal to render the metadata pane.
    """
    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'parcel__eclass',
                            'region', 'harvest_plan')
            .filter(id=item_id)
            .first())
    if item is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    transitions = list(HarvestTransition.objects
                       .filter(harvest_plan_item=item)
                       .order_by('date', 'id'))
    return JsonResponse({
        DATA_ID: 'harvest_plan_items',
        ROW_ID: item.id,
        RECORD: build_harvest_plan_item_record(item),
        TRANSITION_RECORDS: [_transition_record(t) for t in transitions],
    })


@login_required
@require_writer
def item_form_view(request, item_id: int | None = None):
    """Form fragment for the Nuovo intervento / edit-item modal.

    Query params (add path only):
      ?plan=<id>  — required: the plan the new item will live under.
    """
    item = plan = None
    if item_id:
        item = (HarvestPlanItem.objects
                .select_related('harvest_plan', 'parcel__region',
                                'parcel__eclass', 'region')
                .filter(id=item_id)
                .first())
        if item is None:
            raise Http404
        plan = item.harvest_plan
    else:
        plan_id = int_or_none(request.GET.get('plan'))
        if plan_id is None:
            raise Http404('plan required')
        plan = HarvestPlan.objects.filter(id=plan_id).first()
        if plan is None:
            raise Http404
    regions = list(Region.objects.order_by('name'))
    parcels = sorted(Parcel.objects.select_related('region', 'eclass'),
                     key=parcel_sort_key)
    return JsonResponse({HTML: render_to_string(
        'piano_di_taglio/_item_form.html', {
            'item': item, 'plan': plan,
            'regions': regions, 'parcels': parcels,
            'today_year': date_type.today().year,
        }, request=request,
    )})


@login_required
@require_writer
@require_POST
def item_save_view(request):
    """Create or update a HarvestPlanItem."""
    body, error = parse_json_body(request)
    if error:
        return error
    item_id = int_or_none(body.get(ROW_ID))

    parsed, errors = _parse_item_body(body)
    if errors:
        return validation_error(errors)

    with transaction.atomic():
        if item_id is not None:
            item = HarvestPlanItem.objects.select_for_update().filter(
                id=item_id,
            ).first()
            if item is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            version = submitted_version(body)
            if item.version != version:
                item = (HarvestPlanItem.objects
                        .select_related('parcel__region', 'parcel__eclass',
                                        'region', 'harvest_plan')
                        .get(id=item.id))
                return conflict_response(
                    data_id='harvest_plan_items', row_id=item.id,
                    record=build_harvest_plan_item_record(item),
                )
            for field, value in parsed.items():
                setattr(item, field, value)
            item.version += 1
            try:
                item.clean()
                item.save()
            except Exception as exc:
                return validation_error([str(exc)])
        else:
            plan_id = int_or_none(body.get(FIELD_HARVEST_PLAN_ID))
            if plan_id is None:
                return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
            plan = HarvestPlan.objects.filter(id=plan_id).first()
            if plan is None:
                return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
            item = HarvestPlanItem(
                harvest_plan=plan,
                state=HarvestPlanItemState.PLANNED,
                **parsed,
            )
            try:
                item.clean()
                item.save()
            except Exception as exc:
                return validation_error([str(exc)])
        mark_stale('harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')

    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'parcel__eclass',
                            'region', 'harvest_plan')
            .get(id=item.id))
    return success_response(
        request, body,
        data_id='harvest_plan_items', row_id=item.id,
        patches=[row_patch(
            'harvest_plan_items', item.id, build_harvest_plan_item_record(item),
        )],
    )


@login_required
@require_writer
@require_POST
def item_delete_view(request, item_id: int):
    """Delete a HarvestPlanItem.  Gated: state must be PLANNED.

    A non-PLANNED item also typically has TreeMark / Harvest /
    HarvestTransition rows that the FK PROTECT cascade would refuse;
    the user must clear those first.  The CSV-download forced-step is
    client-side (precedes this call); the server only enforces the
    state gate.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    item = HarvestPlanItem.objects.filter(id=item_id).first()
    if item is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if item.version != submitted_version(body):
        item = (HarvestPlanItem.objects
                .select_related('parcel__region', 'parcel__eclass',
                                'region', 'harvest_plan')
                .get(id=item.id))
        return conflict_response(
            data_id='harvest_plan_items', row_id=item.id,
            record=build_harvest_plan_item_record(item),
        )
    if item.state != HarvestPlanItemState.PLANNED:
        return validation_error([S.ERR_PLAN_ITEM_STATE_NOT_PLANNED])

    with transaction.atomic():
        try:
            item.delete()
        except Exception:
            return validation_error([S.ERR_PLAN_ITEM_HAS_DEPS])
        mark_stale('harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')

    return success_response(
        request, body,
        data_id='harvest_plan_items', row_id=item_id,
        deletes=[row_delete('harvest_plan_items', item_id)],
    )


# ---------------------------------------------------------------------------
# Per-item Esporta — zip of martellate_<id>.csv + prelievi_<id>.csv
# ---------------------------------------------------------------------------

@login_required
def item_export_view(request, item_id: int):
    """Zip with this item's tree_marks and linked harvests.

    Reused as the forced-download step before per-item deletion.
    """
    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'region', 'harvest_plan')
            .filter(id=item_id)
            .first())
    if item is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    # martellate_<id>.csv
    delimiter, decimal_sep = csv_io.export_format()
    marks_buf, marks_w = csv_io.csv_buffer(delimiter)
    marks_w.writerow([
        S.CSV_COL_DATA, S.CSV_COL_REGION, S.CSV_COL_PARCEL,
        S.CSV_COL_DAMAGED, S.CSV_COL_NUMBER, S.CSV_COL_SPECIES,
        S.CSV_COL_D_CM, S.CSV_COL_H_M, S.CSV_COL_H_MEASURED,
        S.CSV_COL_LAT, S.CSV_COL_LON, S.CSV_COL_ACC_M, S.CSV_COL_OPERATOR,
    ])
    marks_qs = (TreeMark.objects
                .filter(harvest_plan_item=item)
                .select_related('tree__species', 'tree__parcel__region')
                .order_by('date', 'id'))
    for tm in marks_qs:
        # Compresa / Particella reflect the linked tree's parcel; for
        # region-wide items, the parcel may differ from the item's
        # missing one (which is by design).
        parcel = tm.tree.parcel
        marks_w.writerow([
            tm.date.isoformat(),
            parcel.region.name,
            parcel.name,
            '1' if item.damaged else '0',
            tm.number,
            tm.tree.species.common_name,
            tm.d_cm,
            csv_io.format_decimal(tm.h_m, decimal_sep),
            '1' if tm.h_measured else '0',
            csv_io.format_decimal(tm.lat, decimal_sep),
            csv_io.format_decimal(tm.lon, decimal_sep),
            tm.acc_m if tm.acc_m is not None else '',
            tm.operator,
        ])

    # prelievi_<id>.csv — one row per linked Harvest with species/tractor
    # percent breakdowns as separate trailing columns.
    species_list = list(Species.objects.order_by('sort_order')
                                       .values_list('id', 'common_name'))
    from apps.base.models import Tractor
    tractor_list = list(Tractor.objects.order_by('name', 'manufacturer', 'model', 'id'))
    species_ids = [sid for sid, _ in species_list]
    tractor_ids = [t.id for t in tractor_list]
    tractor_labels = [t.display_name for t in tractor_list]

    prelievi_buf, prelievi_w = csv_io.csv_buffer(delimiter)
    prelievi_w.writerow(
        [S.CSV_COL_DATA, S.CSV_COL_REGION, S.CSV_COL_PARCEL,
         S.CSV_COL_CREW, S.CSV_COL_VDP, S.CSV_COL_PRODUCT,
         S.CSV_COL_QUINTALS, S.CSV_COL_NOTE, S.CSV_COL_EXTRA_NOTE]
        + [sn for _, sn in species_list]
        + tractor_labels
    )
    harvests_qs = (Harvest.objects
                   .filter(harvest_plan_item=item)
                   .select_related('parcel__region', 'crew', 'product')
                   .order_by('date', 'id'))
    sp_map: dict[int, dict[int, int]] = {}
    for hs in HarvestSpecies.objects.filter(harvest__in=harvests_qs):
        sp_map.setdefault(hs.harvest_id, {})[hs.species_id] = hs.percent
    tr_map: dict[int, dict[int, int]] = {}
    for ht in HarvestTractor.objects.filter(harvest__in=harvests_qs):
        tr_map.setdefault(ht.harvest_id, {})[ht.tractor_id] = ht.percent
    for h in harvests_qs:
        prelievi_w.writerow(
            [h.date.isoformat(), h.parcel.region.name, h.parcel.name,
             h.crew.name, h.record1 or '', h.product.name,
             csv_io.format_decimal(h.mass_q, decimal_sep),
             render_flag_note(h.damaged, h.unhealthy, h.psr),
             h.note or '']
            + [sp_map.get(h.id, {}).get(sid, 0) for sid in species_ids]
            + [tr_map.get(h.id, {}).get(tid, 0) for tid in tractor_ids]
        )

    region_name = (item.region or item.parcel.region).name
    parcel_name = item.parcel.name if item.parcel else ''
    parts = [str(item.year_planned), region_name, parcel_name]
    safe_name = _safe_filename('-'.join(p for p in parts if p))
    return csv_io.zip_csv_response(
        [
            (f'martellate_{item.id}.csv', marks_buf.getvalue()),
            (f'prelievi_{item.id}.csv', prelievi_buf.getvalue()),
        ],
        f'{safe_name}.zip',
    )


# ---------------------------------------------------------------------------
# Apri / Chiudi cantiere
# ---------------------------------------------------------------------------

# Valid (current_state, open_flag) -> new_state transitions.
_ALLOWED_TRANSITIONS = {
    (HarvestPlanItemState.PLANNED, True):     HarvestPlanItemState.OPEN,
    (HarvestPlanItemState.MARKED, True):      HarvestPlanItemState.OPEN,
    (HarvestPlanItemState.OPEN, False):       HarvestPlanItemState.CLOSED,
    (HarvestPlanItemState.HARVESTING, False): HarvestPlanItemState.CLOSED,
}


@login_required
@require_writer
@require_POST
def transition_save_view(request):
    """Apri / Chiudi cantiere — creates a HarvestTransition row and
    advances ``HarvestPlanItem.state`` server-side.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    item_id = int_or_none(body.get(FIELD_HARVEST_PLAN_ITEM_ID))
    open_flag = is_truthy(body.get(FIELD_OPEN))
    date_raw = (body.get(FIELD_DATE) or '').strip()
    note = (body.get(FIELD_NOTE) or '').strip()

    if item_id is None:
        return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
    if not date_raw:
        return validation_error([S.ERR_TRANSITION_DATE_REQUIRED])
    try:
        date = date_type.fromisoformat(date_raw)
    except ValueError:
        return validation_error([S.ERR_DATE_INVALID])

    with transaction.atomic():
        item = HarvestPlanItem.objects.select_for_update().filter(
            id=item_id,
        ).first()
        if item is None:
            return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
        cur = HarvestPlanItemState(item.state)
        new_state = _ALLOWED_TRANSITIONS.get((cur, open_flag))
        if new_state is None:
            return validation_error([S.ERR_TRANSITION_INVALID_STATE])

        transition = HarvestTransition.objects.create(
            harvest_plan_item=item, open=open_flag, date=date, note=note,
        )
        item.state = new_state
        # Set date_actual on Apri-cantiere if it isn't already set
        # (auto-mark sets it from the first TreeMark date).
        if open_flag and item.date_actual is None:
            item.date_actual = date
        item.version += 1
        item.save()
        mark_stale('harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')

    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'parcel__eclass',
                            'region', 'harvest_plan')
            .get(id=item.id))
    item_record = build_harvest_plan_item_record(item)
    return success_response(
        request, body,
        patches=[row_patch('harvest_plan_items', item_record[0], item_record)],
    )


# ---------------------------------------------------------------------------
# Tree-mark form + CRUD
# ---------------------------------------------------------------------------

@login_required
@require_writer
def mark_form_view(request, mark_id: int | None = None):
    """Return the HTML fragment for adding or editing a TreeMark.

    Query params (add path only):
      ?item=<id>  — required: the HarvestPlanItem under which the mark lives.
    """
    from config.constants import FIELD_SORT_ORDER
    tm = tree = None
    if mark_id:
        tm = (TreeMark.objects
              .select_related('tree__species', 'harvest_plan_item__parcel__region',
                              'harvest_plan_item__region')
              .get(id=mark_id))
        item = tm.harvest_plan_item
        tree = tm.tree
    else:
        item_id = int_or_none(request.GET.get('item'))
        if item_id is None:
            raise Http404
        item = (HarvestPlanItem.objects
                .select_related('parcel__region', 'region')
                .filter(id=item_id).first())
        if item is None:
            raise Http404

    species = list(Species.objects.filter(active=True).order_by(FIELD_SORT_ORDER))
    compresa = (item.region or item.parcel.region).name if (item.region or item.parcel) else ''
    particella = item.parcel.name if item.parcel else ''

    # Region-wide items: offer a parcel pulldown.
    parcel_choices = None
    selected_parcel_id = None
    if item.region_id and not item.parcel_id:
        parcel_choices = list(
            Parcel.objects.filter(region=item.region)
            .order_by('name')
        )
        if tm and tree:
            selected_parcel_id = tree.parcel_id

    next_number = tm.number if tm else _next_mark_number(item.id)

    ctx = {
        'tm': tm,
        'tree': tree,
        'item': item,
        'species': species,
        'compresa': compresa,
        'particella': particella,
        'parcel_choices': parcel_choices,
        'selected_parcel_id': selected_parcel_id,
        'next_number': next_number,
        'date': tm.date.isoformat() if tm else date_type.today().isoformat(),
        'operator': tm.operator if tm else '',
        'selected_species_id': tree.species_id if tree else (
            species[0].id if species else None
        ),
        'd_cm': tm.d_cm if tm else '',
        'h_m': tm.h_m if tm else '',
        'lat': round(tm.lat, 5) if tm and tm.lat is not None else '',
        'lon': round(tm.lon, 5) if tm and tm.lon is not None else '',
        # Shared _tree_fields.html context.
        'show_ceduo': False,
        'show_l10': False,
        'is_edit': False,
    }
    return JsonResponse({HTML: render_to_string(
        'piano_di_taglio/_mark_form.html', ctx, request=request,
    )})


def _parse_optional_mark_number(body: dict):
    if FIELD_NUMBER not in body:
        return _MARK_NUMBER_MISSING, None
    raw = body.get(FIELD_NUMBER)
    if raw in (None, ''):
        return None, None
    number = int_or_none(raw)
    if number is None or number <= 0:
        return None, S.ERR_MARK_NUMBER_INVALID
    return number, None


def _parse_csv_mark_number(reader, row: dict):
    raw = row.get('numero')
    if raw is None or str(raw).strip() == '':
        return None, None
    number = reader.integer(raw)
    if number is None or number <= 0:
        return None, raw
    return number, None


@login_required
@require_writer
@require_POST
def mark_save_view(request):
    """Create or update a single TreeMark (manual entry or pencil-edit).

    The client computes volume_m3 and mass_q via volume.js and sends
    them as-is; no server-side recompute on this path.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    row_id = int_or_none(body.get(ROW_ID))
    item_id = int_or_none(body.get(FIELD_HARVEST_PLAN_ITEM_ID))
    species_id = int_or_none(body.get(FIELD_SPECIES_ID))
    d_cm = int_or_none(body.get(FIELD_D_CM))
    h_m = parse_decimal(body.get(FIELD_H_M))
    h_measured = is_truthy(body.get(FIELD_H_MEASURED))
    volume_m3 = parse_decimal(body.get(FIELD_VOLUME_M3))
    mass_q = parse_decimal(body.get(FIELD_MASS_Q))
    lat = coord_float(parse_decimal(body.get(FIELD_LAT)))
    lon = coord_float(parse_decimal(body.get(FIELD_LON)))
    acc_m = int_or_none(body.get(FIELD_ACC_M))
    operator = (body.get(FIELD_OPERATOR) or '').strip()
    date_raw = (body.get(FIELD_DATE) or '').strip()
    number, number_error = _parse_optional_mark_number(body)
    parcel_id = int_or_none(body.get(FIELD_PARCEL_ID))

    errors = []
    if number_error:
        errors.append(number_error)
    if item_id is None:
        errors.append(S.ERR_PLAN_ITEM_NOT_FOUND)
    if species_id is None:
        errors.append(S.ERR_MARK_SPECIES_REQUIRED)
    if d_cm is None or d_cm <= 0:
        errors.append(S.ERR_MARK_D_REQUIRED)
    if h_m is None or h_m <= 0:
        errors.append(S.ERR_MARK_H_REQUIRED)
    if not operator:
        errors.append(S.ERR_MARK_OPERATOR_REQUIRED)
    if not date_raw:
        errors.append(S.ERR_DATE_REQUIRED)
    else:
        try:
            date = date_type.fromisoformat(date_raw)
        except ValueError:
            errors.append(S.ERR_DATE_INVALID)
            date = None
    if errors:
        return validation_error(errors)

    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'region')
            .filter(id=item_id).first())
    if item is None:
        return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
    if item.state == HarvestPlanItemState.CLOSED:
        return validation_error([S.ERR_MARK_ITEM_CLOSED])

    species = Species.objects.filter(id=species_id).first()
    if species is None:
        return validation_error([S.ERR_MARK_SPECIES_REQUIRED])
    if number is not _MARK_NUMBER_MISSING and number is not None:
        duplicate = TreeMark.objects.filter(
            harvest_plan_item_id=item.id, number=number,
        )
        if row_id:
            duplicate = duplicate.exclude(id=row_id)
        if duplicate.exists():
            return validation_error([S.ERR_MARK_NUMBER_DUPLICATE.format(number)])

    parcel = _resolve_mark_parcel(item, parcel_id)
    if isinstance(parcel, JsonResponse):
        return parcel

    with transaction.atomic():
        if row_id:
            # Update existing mark.
            tm = (TreeMark.objects
                  .select_related('tree')
                  .filter(id=row_id).first())
            if tm is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            version = submitted_version(body)
            if tm.version != version:
                fresh_tm = (TreeMark.objects
                            .select_related('tree__species')
                            .get(id=tm.id))
                return conflict_response(
                    data_id=f'mark_trees_{item.id}', row_id=fresh_tm.id,
                    record=build_tree_mark_record(fresh_tm),
                )
            if number is not _MARK_NUMBER_MISSING:
                tm.number = number
            tm.date = date
            tm.d_cm = d_cm
            tm.h_m = h_m
            tm.h_measured = h_measured
            tm.volume_m3 = volume_m3
            tm.mass_q = mass_q
            tm.lat = lat
            tm.lon = lon
            tm.acc_m = acc_m
            tm.operator = operator
            tm.version += 1
            tm.save()
            tree = tm.tree
            tree.species = species
            tree.parcel = parcel
            tree.lat = lat
            tree.lon = lon
            tree.acc_m = acc_m
            tree.save()
        else:
            # Create new mark + tree.
            tree = Tree.objects.create(
                species=species, parcel=parcel,
                lat=lat, lon=lon, acc_m=acc_m,
            )
            tm = TreeMark.objects.create(
                harvest_plan_item=item, tree=tree,
                number=None if number is _MARK_NUMBER_MISSING else number,
                date=date, d_cm=d_cm, h_m=h_m,
                h_measured=h_measured,
                volume_m3=volume_m3, mass_q=mass_q,
                lat=lat, lon=lon, acc_m=acc_m,
                operator=operator,
            )
            _auto_advance_to_marked(item, date)

        _rematerialize_volume_marked(item.id)
        mark_stale(f'mark_trees_{item.id}', 'harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')

    tm = (TreeMark.objects
          .select_related('tree__species')
          .get(id=tm.id))
    item_fresh = (HarvestPlanItem.objects
                  .select_related('parcel__region', 'parcel__eclass',
                                  'region', 'harvest_plan')
                  .get(id=item.id))
    mark_record = build_tree_mark_record(tm)
    item_record = build_harvest_plan_item_record(item_fresh)
    return success_response(
        request, body,
        data_id=f'mark_trees_{item.id}', row_id=tm.id,
        patches=[
            row_patch(f'mark_trees_{item.id}', tm.id, mark_record),
            row_patch('harvest_plan_items', item_record[0], item_record),
        ],
    )


@login_required
@require_writer
@require_POST
def mark_delete_view(request):
    """Delete a single TreeMark (and its orphaned Tree).

    State stays monotonic per B3: count can return to zero but state
    does not revert to ``planned``.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    row_id = int(body[ROW_ID])
    version = submitted_version(body)

    tm = (TreeMark.objects
          .select_related('tree', 'harvest_plan_item')
          .filter(id=row_id).first())
    if tm is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if tm.version != version:
        fresh_tm = (TreeMark.objects
                    .select_related('tree__species')
                    .get(id=tm.id))
        return conflict_response(
            data_id=f'mark_trees_{tm.harvest_plan_item_id}',
            row_id=fresh_tm.id,
            record=build_tree_mark_record(fresh_tm),
        )

    item = tm.harvest_plan_item
    if item.state == HarvestPlanItemState.CLOSED:
        return validation_error([S.ERR_MARK_ITEM_CLOSED])

    tree = tm.tree
    item_id = item.id
    with transaction.atomic():
        tm.delete()
        if not TreeMark.objects.filter(tree=tree).exists():
            tree.delete()
        _rematerialize_volume_marked(item_id)
        mark_stale(f'mark_trees_{item_id}', 'harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')

    item_fresh = (HarvestPlanItem.objects
                  .select_related('parcel__region', 'parcel__eclass',
                                  'region', 'harvest_plan')
                  .get(id=item_id))
    item_record = build_harvest_plan_item_record(item_fresh)
    return success_response(
        request, body,
        data_id=f'mark_trees_{item_id}', row_id=row_id,
        patches=[row_patch('harvest_plan_items', item_record[0], item_record)],
        deletes=[row_delete(f'mark_trees_{item_id}', row_id)],
    )


# ---------------------------------------------------------------------------
# Tree-mark CSV import
# ---------------------------------------------------------------------------

@login_required
@require_writer
@require_POST
def mark_csv_import_view(request):
    """Import tree marks from an ipso CSV file.

    Server-side ``tabacchi_volume_m3`` fills volume and mass for each
    row.  The ``import_fingerprint`` (SHA-256 of the row content) deduplicates
    re-imports: existing fingerprints are silently skipped.
    """
    body, error = parse_json_body(request)
    if error:
        return error
    item_id = int_or_none(body.get(FIELD_HARVEST_PLAN_ITEM_ID))
    try:
        upload = csv_io.json_file_bytes(body, FIELD_FILE)
    except csv_io.CsvError as e:
        return validation_error([str(e)])

    if item_id is None:
        return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
    if upload is None:
        return validation_error([S.ERR_CSV_FILE_REQUIRED])

    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'region')
            .filter(id=item_id).first())
    if item is None:
        return validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
    if item.state == HarvestPlanItemState.CLOSED:
        return validation_error([S.ERR_MARK_ITEM_CLOSED])

    result = csv_plan.read_optional(upload, required={
        'date':      [S.CSV_COL_DATA],
        'compresa':  [S.CSV_COL_REGION],
        'particella': [S.CSV_COL_PARCEL],
        'species':   [S.CSV_COL_SPECIES],
        'd_cm':      [S.CSV_COL_D_CM],
        'h_m':       [S.CSV_COL_H_M],
    }, optional={
        'catastrofata': [S.CSV_COL_DAMAGED],
        'numero':    [S.CSV_COL_NUMBER],
        'h_measured': [S.CSV_COL_H_MEASURED],
        'lat':       [S.CSV_COL_LAT],
        'lon':       [S.CSV_COL_LON],
        'acc_m':     [S.CSV_COL_ACC_M],
        'operator':  [S.CSV_COL_OPERATOR],
    })
    if isinstance(result, csv_plan.CsvError):
        return validation_error([result.message])
    reader, rows = result.reader, result.rows

    # Resolve lookups.
    parcel_map = {}
    for p in Parcel.objects.select_related('region').all():
        parcel_map[(p.region.name.lower(), p.name.lower())] = p
    species_map = {sp.common_name.lower(): sp
                   for sp in Species.objects.all()}

    item_region = item.region or (item.parcel.region if item.parcel else None)
    errors = []
    parsed = []
    for i, row in enumerate(rows, start=1):
        date_str = row['date'].strip()
        compresa = row['compresa'].strip()
        particella = row['particella'].strip()
        species_name = row['species'].strip()
        operator = row.get('operator', '').strip()

        try:
            date = _parse_date_flex(date_str)
        except ValueError:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, date_str))
            continue

        parcel = parcel_map.get((compresa.lower(), particella.lower()))
        if parcel is None:
            errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                i, compresa, particella))
            continue
        if item_region and parcel.region_id != item_region.id:
            errors.append(S.ERR_MARK_PARCEL_NOT_IN_REGION)
            continue

        species = species_map.get(species_name.lower())
        if species is None:
            errors.append(S.ERR_CSV_SPECIES_NOT_FOUND.format(i, species_name))
            continue

        d_cm = reader.integer(row.get('d_cm'))
        h_m = reader.decimal(row.get('h_m'))
        if d_cm is None or h_m is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f"{row.get('d_cm', '')}/{row.get('h_m', '')}"))
            continue

        lat = coord_float(reader.decimal(row.get('lat')))
        lon = coord_float(reader.decimal(row.get('lon')))
        acc_m = reader.integer(row.get('acc_m'))
        h_measured = is_truthy(row.get('h_measured'))

        numero, numero_error = _parse_csv_mark_number(reader, row)
        if numero_error is not None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, numero_error))
            continue
        parsed.append(MarkImportRow(
            date=date, parcel=parcel, species=species,
            number=numero, d_cm=d_cm, h_m=h_m, h_measured=h_measured,
            lat=lat, lon=lon, acc_m=acc_m, operator=operator,
            fingerprint=csv_mark_fingerprint(
                date=date, species_name=species_name, d_cm=d_cm, h_m=h_m,
                lat=lat, lon=lon, operator=operator,
            ),
        ))

    if errors:
        return validation_error(errors)

    import_result = import_mark_rows(item, parsed)
    if import_result.errors:
        return validation_error(import_result.errors)

    item_fresh = (HarvestPlanItem.objects
                  .select_related('parcel__region', 'parcel__eclass',
                                  'region', 'harvest_plan')
                  .get(id=item.id))
    item_record = build_harvest_plan_item_record(item_fresh)
    return success_response(
        request, body,
        patches=[row_patch('harvest_plan_items', item_record[0], item_record)],
        extra={
            'imported': import_result.imported,
            'skipped_duplicates': import_result.skipped_duplicates,
        },
    )


# ---------------------------------------------------------------------------
# Mark helpers
# ---------------------------------------------------------------------------

def _resolve_mark_parcel(item, parcel_id):
    """Determine the parcel for a new tree mark.

    Parcel-scoped items: use the item's parcel.
    Region-wide items: require an explicit parcel_id in the request.
    """
    if item.parcel_id:
        return item.parcel
    if parcel_id is None:
        return validation_error([S.ERR_MARK_PARCEL_REQUIRED])
    parcel = Parcel.objects.filter(id=parcel_id).first()
    if parcel is None:
        return validation_error([S.ERR_MARK_PARCEL_REQUIRED])
    if item.region_id and parcel.region_id != item.region_id:
        return validation_error([S.ERR_MARK_PARCEL_NOT_IN_REGION])
    return parcel


def _parse_date_flex(s: str) -> date_type:
    """Parse ISO (YYYY-MM-DD) or Italian (DD/MM/YYYY) date."""
    if '/' in s:
        parts = s.split('/')
        return date_type(int(parts[2]), int(parts[1]), int(parts[0]))
    return date_type.fromisoformat(s)


def _parse_plan_body(body):
    """Extract + validate plan fields.  Returns (name, description,
    year_start, year_end, errors)."""
    errors = []
    name = (body.get(FIELD_NAME) or '').strip()
    description = (body.get(FIELD_DESCRIPTION) or '').strip()
    if not name:
        errors.append(S.ERR_PLAN_NAME_REQUIRED)
    try:
        year_start = int(body.get(FIELD_YEAR_START))
        year_end = int(body.get(FIELD_YEAR_END))
    except (TypeError, ValueError):
        errors.append(S.ERROR_GENERIC)
        return name, description, 0, 0, errors
    if year_end < year_start:
        errors.append(S.ERR_PLAN_YEAR_RANGE)
    return name, description, year_start, year_end, errors


def _parse_item_body(body):
    """Extract + validate plan-item fields.  Returns (parsed_dict, errors).

    The form cascade always submits both `region_id` and `parcel_id`.
    Storage is `region XOR parcel`, so the server normalises: when
    `parcel_id` is set, the row is parcel-scoped and the submitted
    `region_id` is dropped (it's redundant — derivable via
    `parcel.region`).  A blank `parcel_id` means a region-wide item,
    which is only valid when one of `damaged`/`unhealthy` is checked.
    """
    errors: list[str] = []
    region_id = int_or_none(body.get(FIELD_REGION_ID))
    parcel_id = int_or_none(body.get(FIELD_PARCEL_ID))
    damaged = is_truthy(body.get(FIELD_DAMAGED))
    unhealthy = is_truthy(body.get(FIELD_UNHEALTHY))
    psr = is_truthy(body.get(FIELD_PSR))
    if parcel_id is not None:
        # Parcel-scoped item: region is implicit through parcel.
        region_id = None
    elif region_id is not None:
        # Region-wide item: only valid for catastrofato / fitosanitario.
        if not (damaged or unhealthy):
            errors.append(S.ERR_PLAN_ITEM_REGION_REQUIRES_FLAG)
    else:
        errors.append(S.ERR_PLAN_ITEM_COMPRESA_REQUIRED)
    try:
        year_planned = int(body.get(FIELD_YEAR_PLANNED))
    except (TypeError, ValueError):
        errors.append(S.ERR_DATE_REQUIRED)
        year_planned = 0
    volume = parse_decimal(body.get(FIELD_VOLUME_PLANNED_M3))
    if volume is not None and volume <= 0:
        errors.append(S.ERR_PLAN_ITEM_VOLUME_NEGATIVE)
        volume = None
    area = parse_decimal(body.get(FIELD_INTERVENTION_AREA_HA))
    if area is not None and area <= 0:
        errors.append(S.ERR_PLAN_ITEM_AREA_NEGATIVE)
        area = None
    note = (body.get(FIELD_NOTE) or '').strip()

    parsed = {
        'region_id': region_id,
        'parcel_id': parcel_id,
        'year_planned': year_planned,
        'volume_planned_m3': volume,
        'intervention_area_ha': area,
        'damaged': damaged,
        'unhealthy': unhealthy,
        'psr': psr,
        'note': note,
    }
    return parsed, errors


def _transition_record(t) -> list:
    """One-row payload for HarvestTransition (used in the modal data
    response and on transition save)."""
    return [t.id, t.harvest_plan_item_id, t.open,
            t.date.isoformat(), t.note]


_SAFE_RE = re.compile(r'[^A-Za-z0-9._-]+')


def _safe_filename(s: str) -> str:
    return _SAFE_RE.sub('_', s).strip('_') or 'plan'
