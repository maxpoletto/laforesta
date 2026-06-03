"""Piano di taglio API views.

Read endpoints (digest passthrough), plan + plan-item CRUD, plan-level
and per-item CSV import/export, and the cantiere transition save view.

Form endpoints follow the standard Abies idiom (see
`apps.prelievi.views`): a GET returns an HTML fragment, a POST processes
the submission and returns either ``{row_id, record}`` or a
``validation_error`` / ``conflict`` payload.
"""

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import date as date_type
from typing import Iterable

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Q
from django.http import Http404, HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base import csv_io
from apps.base.numparse import coord_float, int_or_none, parse_decimal
from apps.base.responses import csv_error_list, validation_error
from apps.base.digests import (
    build_harvest_plan_item_record,
    build_harvest_plan_record,
    build_tree_mark_record,
    mark_stale,
    serve_digest,
)
from apps.base.middleware import save_nonce
from apps.base.models import (
    HarvestDetail,
    HarvestPlan,
    HarvestPlanItem,
    HarvestPlanItemState,
    HarvestTransition,
    ParcelPlanDetail,
    Parcel,
    Region,
    Species,
    Tree,
    TreeMark,
    next_sequence_number,
    parcel_sort_key,
    render_flag_note,
    tree_mass_q,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S
from config.constants import (
    COLUMNS,
    DATA_ID,
    FIELD_ACC_M,
    FIELD_CEDUO_FILE,
    FIELD_D_CM,
    FIELD_DAMAGED,
    FIELD_DATE,
    FIELD_DESCRIPTION,
    FIELD_ERRORS,
    FIELD_FILE,
    FIELD_FUSTAIA_FILE,
    FIELD_H_M,
    FIELD_H_MEASURED,
    FIELD_HARVEST_PLAN_ID,
    FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_INTERVENTION_AREA_HA,
    FIELD_LAT,
    FIELD_LON,
    FIELD_MASS_Q,
    FIELD_NAME,
    FIELD_NONCE,
    FIELD_NOTE,
    FIELD_NUMBER,
    FIELD_OPEN,
    FIELD_OPERATOR,
    FIELD_PARCEL_ID,
    FIELD_PSR,
    FIELD_REGION_ID,
    FIELD_SPECIES_ID,
    FIELD_TURNO_A,
    FIELD_UNHEALTHY,
    FIELD_VOLUME_M3,
    FIELD_VOLUME_PLANNED_M3,
    FIELD_YEAR_END,
    FIELD_YEAR_PLANNED,
    FIELD_YEAR_START,
    HTML,
    ITEM_RECORD,
    ITEM_RECORDS,
    MESSAGE,
    PLAN_RECORD,
    RECORD,
    RECORDS,
    ROW_ID,
    ROWS,
    STATUS,
    STATUS_CONFLICT,
    STATUS_NOT_FOUND,
    STATUS_VALIDATION_ERROR,
    TRANSITION_RECORDS,
    VERSION,
    is_truthy,
)


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
    body = json.loads(request.body)
    plan_id = int_or_none(body.get(ROW_ID))

    name, description, year_start, year_end, errors = _parse_plan_body(body)
    if errors:
        return validation_error(errors)

    # Uniqueness check (case-insensitive — `HarvestPlan.name` has a
    # plain unique constraint, but we surface a friendly message early).
    dup = HarvestPlan.objects.filter(name__iexact=name)
    if plan_id:
        dup = dup.exclude(id=plan_id)
    if dup.exists():
        return validation_error([S.ERR_PLAN_NAME_DUPLICATE])

    with transaction.atomic():
        if plan_id is not None:
            plan = HarvestPlan.objects.select_for_update().filter(
                id=plan_id,
            ).first()
            if plan is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            submitted_version = int_or_none(body.get(VERSION))
            if (submitted_version is not None
                    and plan.version != submitted_version):
                return _conflict_response_plan(plan)
            plan.name = name
            plan.description = description
            plan.year_start = year_start
            plan.year_end = year_end
            plan.version += 1
            plan.save()
        else:
            plan = HarvestPlan.objects.create(
                name=name, description=description,
                year_start=year_start, year_end=year_end,
            )
        mark_stale('harvest_plans', 'audit')

    response_data = {
        DATA_ID: 'harvest_plans',
        ROW_ID: plan.id,
        RECORD: build_harvest_plan_record(plan),
    }
    nonce = body.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


@login_required
@require_writer
@require_POST
def plan_delete_view(request, plan_id: int):
    """Delete a HarvestPlan.  Gated: every item must be in state `planned`.

    The CSV-download forced-step is handled client-side: the JS download
    of `plan_export_view` precedes this call.  We still verify the gate
    server-side so a hand-crafted POST cannot bypass it.
    """
    plan = HarvestPlan.objects.filter(id=plan_id).first()
    if plan is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    bad = HarvestPlanItem.objects.filter(harvest_plan=plan).exclude(
        state=HarvestPlanItemState.PLANNED,
    ).exists()
    if bad:
        return validation_error([S.ERR_PLAN_HAS_ACTIVE_ITEMS])

    with transaction.atomic():
        # HarvestPlanItem and ParcelPlanDetail both cascade to HarvestPlan.
        plan.delete()
        mark_stale('harvest_plans', 'harvest_plan_items', 'audit')
    return JsonResponse({DATA_ID: 'harvest_plans', ROW_ID: plan_id})


# ---------------------------------------------------------------------------
# Plan CSV import — single endpoint dispatching on which file(s) attached
# ---------------------------------------------------------------------------

# Column aliases — each logical field accepts the legacy pdg-2026 name
# AND the display-name used by the per-table CSV export, so a freshly
# exported plan zip round-trips through the importer.  `required` raises
# a missing-column error pre-parse; `optional` keys are absent from row
# dicts when the CSV doesn't carry them.
_FUSTAIA_REQUIRED = {
    'compresa':   [S.CSV_COL_COMPRESA],                          # 'Compresa'
    'particella': [S.CSV_COL_PARTICELLA],                        # 'Particella'
    'anno':       [S.COL_YEAR_PLANNED, S.CSV_COL_ANNO],          # 'Anno previsto' | 'Anno'
    'volume':     [S.COL_VOLUME_PLANNED, S.CSV_COL_PRELIEVO_M3], # 'Volume previsto' | 'Prelievo (m³)'
}
_FUSTAIA_OPTIONAL = {
    # Holds the flag string ("Catastrofato" / "Fitosanitario" / "PSR")
    # in Abies exports; required for region-wide rows (Particella = 'X').
    'note':       [S.COL_NOTE],
}
_CEDUO_REQUIRED = {
    'anno':       [S.COL_YEAR_PLANNED, S.CSV_COL_ANNO],
    'compresa':   [S.CSV_COL_COMPRESA],
    'particella': [S.CSV_COL_PARTICELLA],
    'superficie': [S.CSV_COL_SUPERFICIE_HA],                     # 'Superficie intervento (ha)'
    'turno':      [S.CSV_COL_TURNO_A],                           # 'Turno (a)'
}
_CEDUO_OPTIONAL = {
    # In Abies exports: 'Altre note' = free-text, 'Note' = flag string.
    # In legacy pdg-2026 exports: only 'Note' is present and is itself
    # free-text — handled by checking `has_altre_note` at parse time.
    'free_note':  [S.CSV_COL_EXTRA_NOTE],                        # 'Altre note'
    'flag_note':  [S.COL_NOTE],                                  # 'Note' (flag string)
}


@login_required
@require_writer
@require_POST
def plan_csv_import_view(request):
    """Create a HarvestPlan from CSVs, or upsert CSV rows into an existing one.

    Multipart form fields:
      - ``harvest_plan_id`` — optional: target an existing plan for upsert.
        When set, ``name``/``description`` are ignored and the existing
        plan's are kept; rows from the CSVs are upserted (see below).
      - ``name``            — required when ``harvest_plan_id`` is absent.
      - ``description``     — optional plan description (create path only).
      - ``fustaia_file``    — optional: piano.csv (high-forest schedule).
      - ``ceduo_file``      — optional: ceduo.csv (coppice schedule).

    At least one CSV must be attached.  Upsert key: calendar items dedup
    on ``(harvest_plan, parcel, year_planned)``.  Re-importing the same
    file is a no-op; re-importing a revised file overwrites the matching
    rows in place.
    """
    plan_id = int_or_none(request.POST.get(FIELD_HARVEST_PLAN_ID))
    if plan_id is not None:
        target_plan = HarvestPlan.objects.filter(id=plan_id).first()
        if target_plan is None:
            return validation_error([S.ERR_PLAN_NOT_FOUND])
        name = target_plan.name
        description = target_plan.description
    else:
        target_plan = None
        name = (request.POST.get(FIELD_NAME) or '').strip()
        description = (request.POST.get(FIELD_DESCRIPTION) or '').strip()
        if not name:
            return validation_error([S.ERR_PLAN_NAME_REQUIRED])
        if HarvestPlan.objects.filter(name__iexact=name).exists():
            return validation_error([S.ERR_PLAN_NAME_DUPLICATE])

    fustaia_rows = _read_optional(
        request.FILES.get(FIELD_FUSTAIA_FILE),
        _FUSTAIA_REQUIRED, _FUSTAIA_OPTIONAL,
    )
    ceduo_rows = _read_optional(
        request.FILES.get(FIELD_CEDUO_FILE),
        _CEDUO_REQUIRED, _CEDUO_OPTIONAL,
    )
    if fustaia_rows is None and ceduo_rows is None:
        return validation_error([S.ERR_CSV_NO_FILES])
    for result in (fustaia_rows, ceduo_rows):
        if isinstance(result, _CsvError):
            return validation_error([result.message])

    region_cache = {r.name.lower(): r for r in Region.objects.all()}
    parcel_cache = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }

    errors: list[str] = []
    fustaia_parsed = (
        _parse_fustaia_rows(fustaia_rows, parcel_cache, region_cache, errors)
        if fustaia_rows else []
    )
    ceduo_parsed = (
        _parse_ceduo_rows(ceduo_rows, parcel_cache, errors)
        if ceduo_rows else []
    )
    if errors:
        return csv_error_list(errors)

    all_years = (
        [r[FIELD_YEAR_PLANNED] for r in fustaia_parsed]
        + [r[FIELD_YEAR_PLANNED] for r in ceduo_parsed]
    )

    with transaction.atomic():
        if target_plan is None:
            year_start = min(all_years) if all_years else date_type.today().year
            year_end = max(all_years) if all_years else year_start
            plan = HarvestPlan.objects.create(
                name=name, description=description,
                year_start=year_start, year_end=year_end,
            )
        else:
            plan = target_plan
            if all_years:
                new_start = min(plan.year_start, *all_years)
                new_end = max(plan.year_end, *all_years)
                if new_start != plan.year_start or new_end != plan.year_end:
                    plan.year_start = new_start
                    plan.year_end = new_end
                    plan.version += 1
                    plan.save()

        # Coppice rows need a HarvestDetail per (description, interval).
        # Seed the cache from existing details under this plan so re-imports
        # reuse the same rows.
        detail_cache: dict[tuple[str, int], HarvestDetail] = {
            (ppd.harvest_detail.description, ppd.harvest_detail.interval):
                ppd.harvest_detail
            for ppd in ParcelPlanDetail.objects
                .filter(harvest_plan=plan)
                .select_related('harvest_detail')
            if ppd.harvest_detail.interval is not None
        }
        for r in ceduo_parsed:
            interval = r[FIELD_TURNO_A]
            desc = f'{S.HARVEST_DETAIL} {interval}a'
            key = (desc, interval)
            hd = detail_cache.get(key)
            if hd is None:
                hd = HarvestDetail.objects.create(
                    description=desc, interval=interval,
                )
                detail_cache[key] = hd
            ParcelPlanDetail.objects.update_or_create(
                harvest_plan=plan, parcel=r[FIELD_PARCEL_ID],
                defaults={'harvest_detail': hd},
            )

        n_items = 0
        for r in fustaia_parsed:
            flag_defaults = {
                'volume_planned_m3': r[FIELD_VOLUME_PLANNED_M3],
                'damaged':   r[FIELD_DAMAGED],
                'unhealthy': r[FIELD_UNHEALTHY],
                'psr':       r[FIELD_PSR],
            }
            if r[FIELD_PARCEL_ID] is not None:
                # Parcel-scoped: dedup on (plan, parcel, year_planned).
                HarvestPlanItem.objects.update_or_create(
                    harvest_plan=plan,
                    parcel=r[FIELD_PARCEL_ID],
                    year_planned=r[FIELD_YEAR_PLANNED],
                    defaults=flag_defaults,
                )
            else:
                # Region-wide: dedup on (plan, region, parcel=NULL, year).
                HarvestPlanItem.objects.update_or_create(
                    harvest_plan=plan,
                    region=r[FIELD_REGION_ID],
                    parcel=None,
                    year_planned=r[FIELD_YEAR_PLANNED],
                    defaults=flag_defaults,
                )
            n_items += 1
        for r in ceduo_parsed:
            HarvestPlanItem.objects.update_or_create(
                harvest_plan=plan,
                parcel=r[FIELD_PARCEL_ID],
                year_planned=r[FIELD_YEAR_PLANNED],
                defaults={
                    'intervention_area_ha': r[FIELD_INTERVENTION_AREA_HA],
                    'note':      r[FIELD_NOTE],
                    'damaged':   r[FIELD_DAMAGED],
                    'unhealthy': r[FIELD_UNHEALTHY],
                    'psr':       r[FIELD_PSR],
                },
            )
            n_items += 1

        mark_stale('harvest_plans', 'harvest_plan_items', 'audit')

    response_data = {
        DATA_ID: 'harvest_plans',
        ROW_ID: plan.id,
        RECORD: build_harvest_plan_record(plan),
        'n_items': n_items,
    }
    nonce = request.POST.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Plan-level Esporta CSV — zip of fustaia.csv + ceduo.csv
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

    items = (HarvestPlanItem.objects
             .filter(harvest_plan=plan)
             .select_related('parcel__region', 'parcel__eclass', 'region')
             .order_by('year_planned', 'id'))
    parcel_intervals = {
        ppd.parcel_id: ppd.harvest_detail.interval
        for ppd in ParcelPlanDetail.objects
                       .filter(harvest_plan=plan)
                       .select_related('harvest_detail')
        if ppd.harvest_detail.interval is not None
    }

    delimiter, decimal_sep = csv_io.export_format()
    fustaia_buf = io.StringIO()
    fustaia_w = csv.writer(fustaia_buf, delimiter=delimiter)
    fustaia_w.writerow([
        S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL,
        S.COL_COMPRESA, S.COL_PARCEL, S.COL_STATE, S.COL_NOTE,
        S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED, S.COL_VOLUME_ACTUAL,
    ])

    ceduo_buf = io.StringIO()
    ceduo_w = csv.writer(ceduo_buf, delimiter=delimiter)
    ceduo_w.writerow([
        S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL,
        S.COL_COMPRESA, S.COL_PARCEL, S.COL_STATE, S.COL_NOTE,
        S.COL_INTERVENTION_AREA_HA, S.COL_PARCEL_AREA_HA,
        S.COL_TURNO_A, S.COL_VOLUME_ACTUAL, S.COL_EXTRA_NOTE,
    ])

    for it in items:
        is_region_wide = it.parcel_id is None
        if is_region_wide:
            compresa = it.region.name
            particella = S.PARCEL_WHOLE_REGION_MARK
            parcel_area = ''
            is_coppice = False  # region-wide items always export as fustaia
        else:
            compresa = it.parcel.region.name
            particella = it.parcel.name
            parcel_area = csv_io.format_decimal(it.parcel.area_ha, decimal_sep)
            is_coppice = it.parcel.eclass.coppice
        anno_eff = it.date_actual.year if it.date_actual else ''
        flag_note = render_flag_note(it.damaged, it.unhealthy, it.psr)
        state_label = HarvestPlanItemState(it.state).label
        if is_coppice:
            ceduo_w.writerow([
                it.year_planned, anno_eff,
                compresa, particella,
                state_label, flag_note,
                csv_io.format_decimal(it.intervention_area_ha, decimal_sep),
                parcel_area,
                parcel_intervals.get(it.parcel_id, ''),
                csv_io.format_decimal(it.volume_actual_m3, decimal_sep),
                it.note or '',
            ])
        else:
            fustaia_w.writerow([
                it.year_planned, anno_eff,
                compresa, particella,
                state_label, flag_note,
                csv_io.format_decimal(it.volume_planned_m3, decimal_sep),
                csv_io.format_decimal(it.volume_marked_m3, decimal_sep),
                csv_io.format_decimal(it.volume_actual_m3, decimal_sep),
            ])

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(S.CSV_FILE_FUSTAIA, fustaia_buf.getvalue())
        zf.writestr(S.CSV_FILE_CEDUO,   ceduo_buf.getvalue())

    response = HttpResponse(zip_buf.getvalue(), content_type='application/zip')
    safe_name = _safe_filename(plan.name)
    response['Content-Disposition'] = (
        f'attachment; filename="piano_{safe_name}.zip"'
    )
    # Browsers happily cache download URLs by default; without this the
    # next click of "Esporta CSV" returns last time's bytes even after a
    # plan edit.  no-store also keeps stale-after-deploy bugs from
    # masquerading as "server isn't reloading my code".
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
    body = json.loads(request.body)
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
            submitted_version = int_or_none(body.get(VERSION))
            if (submitted_version is not None
                    and item.version != submitted_version):
                return _conflict_response_item(item)
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
        mark_stale('harvest_plan_items', 'audit')

    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'parcel__eclass',
                            'region', 'harvest_plan')
            .get(id=item.id))
    response_data = {
        DATA_ID: 'harvest_plan_items',
        ROW_ID: item.id,
        RECORD: build_harvest_plan_item_record(item),
    }
    nonce = body.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


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
    item = HarvestPlanItem.objects.filter(id=item_id).first()
    if item is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if item.state != HarvestPlanItemState.PLANNED:
        return validation_error([S.ERR_PLAN_ITEM_STATE_NOT_PLANNED])

    with transaction.atomic():
        try:
            item.delete()
        except Exception:
            return validation_error([S.ERR_PLAN_ITEM_HAS_DEPS])
        mark_stale('harvest_plan_items', 'audit')

    return JsonResponse({DATA_ID: 'harvest_plan_items', ROW_ID: item_id})


# ---------------------------------------------------------------------------
# Per-item Esporta CSV — zip of martellate_<id>.csv + prelievi_<id>.csv
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
    marks_buf = io.StringIO()
    marks_w = csv.writer(marks_buf, delimiter=delimiter)
    marks_w.writerow([
        S.CSV_COL_DATA, S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA,
        S.CSV_COL_CATASTROFATA, S.CSV_COL_NUMERO, S.CSV_COL_GENERE,
        S.CSV_COL_D_CM, S.CSV_COL_H_M, S.CSV_COL_H_MEASURED,
        S.CSV_COL_LAT, S.CSV_COL_LON, S.CSV_COL_ACC_M, S.CSV_COL_OPERATORE,
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
    tractor_list = list(Tractor.objects.order_by('manufacturer', 'model'))
    species_ids = [sid for sid, _ in species_list]
    tractor_ids = [t.id for t in tractor_list]
    tractor_labels = [f'{t.manufacturer} {t.model}'.strip()
                      for t in tractor_list]

    prelievi_buf = io.StringIO()
    prelievi_w = csv.writer(prelievi_buf, delimiter=delimiter)
    prelievi_w.writerow(
        [S.CSV_COL_DATA, S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA,
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

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f'martellate_{item.id}.csv', marks_buf.getvalue())
        zf.writestr(f'prelievi_{item.id}.csv', prelievi_buf.getvalue())

    response = HttpResponse(zip_buf.getvalue(), content_type='application/zip')
    region_name = (item.region or item.parcel.region).name
    parcel_name = item.parcel.name if item.parcel else ''
    parts = [str(item.year_planned), region_name, parcel_name]
    safe_name = _safe_filename('-'.join(p for p in parts if p))
    response['Content-Disposition'] = (
        f'attachment; filename="{safe_name}.zip"'
    )
    response['Cache-Control'] = 'no-store'
    return response


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
    body = json.loads(request.body)
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
        mark_stale('harvest_plan_items', 'audit')

    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'parcel__eclass',
                            'region', 'harvest_plan')
            .get(id=item.id))
    response_data = {
        DATA_ID: 'harvest_plan_items',
        ROW_ID: item.id,
        ITEM_RECORD: build_harvest_plan_item_record(item),
        RECORD: _transition_record(transition),
    }
    nonce = body.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Tree-mark form + CRUD
# ---------------------------------------------------------------------------

@login_required
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


@login_required
@require_writer
@require_POST
def mark_save_view(request):
    """Create or update a single TreeMark (manual entry or pencil-edit).

    The client computes volume_m3 and mass_q via volume.js and sends
    them as-is; no server-side recompute on this path.
    """
    body = json.loads(request.body)
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
    number = int_or_none(body.get(FIELD_NUMBER))
    parcel_id = int_or_none(body.get(FIELD_PARCEL_ID))

    errors = []
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
            version = int(body.get(VERSION, 0))
            if tm.version != version:
                return JsonResponse({
                    STATUS: STATUS_CONFLICT, MESSAGE: S.ERROR_CONFLICT,
                }, status=400)
            if number is not None:
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
            if number is None:
                number = _next_mark_number(item_id)
            tm = TreeMark.objects.create(
                harvest_plan_item=item, tree=tree,
                number=number,
                date=date, d_cm=d_cm, h_m=h_m,
                h_measured=h_measured,
                volume_m3=volume_m3, mass_q=mass_q,
                lat=lat, lon=lon, acc_m=acc_m,
                operator=operator,
            )
            _auto_advance_to_marked(item, date)

        _rematerialize_volume_marked(item.id)
        mark_stale(f'mark_trees_{item.id}', 'harvest_plan_items', 'audit')

    tm = (TreeMark.objects
          .select_related('tree__species')
          .get(id=tm.id))
    item_fresh = (HarvestPlanItem.objects
                  .select_related('parcel__region', 'parcel__eclass',
                                  'region', 'harvest_plan')
                  .get(id=item.id))
    response_data = {
        DATA_ID: f'mark_trees_{item.id}',
        ROW_ID: tm.id,
        RECORD: build_tree_mark_record(tm),
        ITEM_RECORD: build_harvest_plan_item_record(item_fresh),
    }
    nonce = body.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


@login_required
@require_writer
@require_POST
def mark_delete_view(request):
    """Delete a single TreeMark (and its orphaned Tree).

    State stays monotonic per B3: count can return to zero but state
    does not revert to ``planned``.
    """
    body = json.loads(request.body)
    row_id = int(body[ROW_ID])
    version = int(body.get(VERSION, 0))

    tm = (TreeMark.objects
          .select_related('tree', 'harvest_plan_item')
          .filter(id=row_id).first())
    if tm is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if tm.version != version:
        return JsonResponse({
            STATUS: STATUS_CONFLICT, MESSAGE: S.ERROR_CONFLICT,
        }, status=400)

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
        mark_stale(f'mark_trees_{item_id}', 'harvest_plan_items', 'audit')

    item_fresh = (HarvestPlanItem.objects
                  .select_related('parcel__region', 'parcel__eclass',
                                  'region', 'harvest_plan')
                  .get(id=item_id))
    response_data = {
        DATA_ID: f'mark_trees_{item_id}',
        ROW_ID: row_id,
        ITEM_RECORD: build_harvest_plan_item_record(item_fresh),
    }
    nonce = body.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


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
    import hashlib
    from apps.base.tabacchi import tabacchi_volume_m3

    item_id = int_or_none(request.POST.get(FIELD_HARVEST_PLAN_ITEM_ID))
    upload = request.FILES.get(FIELD_FILE)

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

    result = _read_optional(upload, required={
        'date':      [S.CSV_COL_DATA],
        'compresa':  [S.CSV_COL_COMPRESA],
        'particella': [S.CSV_COL_PARTICELLA],
        'species':   [S.CSV_COL_GENERE],
        'd_cm':      [S.CSV_COL_D_CM],
        'h_m':       [S.CSV_COL_H_M],
    }, optional={
        'catastrofata': [S.CSV_COL_CATASTROFATA],
        'numero':    [S.CSV_COL_NUMERO],
        'h_measured': [S.CSV_COL_H_MEASURED],
        'lat':       [S.CSV_COL_LAT],
        'lon':       [S.CSV_COL_LON],
        'acc_m':     [S.CSV_COL_ACC_M],
        'operator':  [S.CSV_COL_OPERATORE],
    })
    if isinstance(result, _CsvError):
        return validation_error([result.message])
    reader, rows = result.reader, result.rows

    # Resolve lookups.
    parcel_map = {}
    for p in Parcel.objects.select_related('region').all():
        parcel_map[(p.region.name.lower(), p.name.lower())] = p
    species_map = {sp.common_name.lower(): sp
                   for sp in Species.objects.all()}

    # Existing fingerprints for dedup.
    existing_fps = set(
        TreeMark.objects
        .filter(harvest_plan_item_id=item_id,
                import_fingerprint__isnull=False)
        .values_list('import_fingerprint', flat=True)
    )

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

        fp_src = f'{date}|{species_name}|{d_cm}|{h_m}|{lat}|{lon}|{operator}'
        fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()
        if fingerprint in existing_fps:
            continue

        try:
            volume_m3 = tabacchi_volume_m3(d_cm, h_m, species.common_name)
            mass_q = tree_mass_q(volume_m3, species.density)
        except (ValueError, KeyError):
            volume_m3 = None
            mass_q = None

        numero = reader.integer(row.get('numero'))

        parsed.append({
            'date': date, 'parcel': parcel, 'species': species,
            'number': numero,
            'd_cm': d_cm, 'h_m': h_m, 'h_measured': h_measured,
            'volume_m3': volume_m3, 'mass_q': mass_q,
            'lat': lat, 'lon': lon, 'acc_m': acc_m,
            'operator': operator, 'fingerprint': fingerprint,
        })
        existing_fps.add(fingerprint)

    if errors:
        return validation_error(errors)

    with transaction.atomic():
        next_number = _next_mark_number(item_id)
        for p in parsed:
            number = p['number']
            if number is None:
                number = next_number
                next_number += 1
            tree = Tree.objects.create(
                species=p['species'], parcel=p['parcel'],
                lat=p['lat'], lon=p['lon'], acc_m=p['acc_m'],
            )
            TreeMark.objects.create(
                harvest_plan_item=item, tree=tree,
                number=number,
                date=p['date'], d_cm=p['d_cm'], h_m=p['h_m'],
                h_measured=p['h_measured'],
                volume_m3=p['volume_m3'], mass_q=p['mass_q'],
                lat=p['lat'], lon=p['lon'], acc_m=p['acc_m'],
                operator=p['operator'],
                import_fingerprint=p['fingerprint'],
            )

        if parsed:
            earliest_date = min(p['date'] for p in parsed)
            _auto_advance_to_marked(item, earliest_date)
            _rematerialize_volume_marked(item.id)

        mark_stale(f'mark_trees_{item.id}', 'harvest_plan_items', 'audit')

    item_fresh = (HarvestPlanItem.objects
                  .select_related('parcel__region', 'parcel__eclass',
                                  'region', 'harvest_plan')
                  .get(id=item.id))
    return JsonResponse({
        ITEM_RECORD: build_harvest_plan_item_record(item_fresh),
        'imported': len(parsed),
        'skipped_duplicates': len(rows) - len(parsed) - len(errors),
    })


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


def _auto_advance_to_marked(item, mark_date):
    """Auto-advance state from planned → marked on first TreeMark."""
    if item.state == HarvestPlanItemState.PLANNED:
        item.state = HarvestPlanItemState.MARKED
        item.date_actual = mark_date
        item.version += 1
        item.save()


def _rematerialize_volume_marked(item_id: int) -> None:
    """Recompute volume_marked_m3 on the linked HarvestPlanItem."""
    from django.db.models import Sum
    total = (TreeMark.objects
             .filter(harvest_plan_item_id=item_id)
             .aggregate(s=Sum('volume_m3'))['s'])
    HarvestPlanItem.objects.filter(id=item_id).update(
        volume_marked_m3=total,
    )


def _next_mark_number(item_id: int) -> int:
    """Return max(tree_mark.number)+1 for the item, or 1 if no marks exist."""
    return next_sequence_number(
        TreeMark.objects.filter(harvest_plan_item_id=item_id), FIELD_NUMBER,
    )


def _parse_date_flex(s: str) -> date_type:
    """Parse ISO (YYYY-MM-DD) or Italian (DD/MM/YYYY) date."""
    if '/' in s:
        parts = s.split('/')
        return date_type(int(parts[2]), int(parts[1]), int(parts[0]))
    return date_type.fromisoformat(s)


# ---------------------------------------------------------------------------
# Internal helpers — parsing
# ---------------------------------------------------------------------------

class _CsvError:
    def __init__(self, message: str):
        self.message = message


@dataclass
class _AliasedCsv:
    """A parsed CSV paired with its rows remapped to logical alias names, so
    cell parsing goes through ``reader`` (which carries the detected decimal
    separator) and header presence through ``reader.fieldnames``."""
    reader: csv_io.CsvReader
    rows: list[dict]


def _read_optional(upload, required, optional=None) -> _AliasedCsv | _CsvError | None:
    """Parse an uploaded CSV if present.

    Delimiter is auto-detected (``,`` or ``;``) so files exported in
    either Italian (``;``) or pdg-2026's English (``,``) style both
    round-trip.  ``required`` and ``optional`` map logical alias names
    to candidate column-name lists; the first candidate that appears in
    the header wins.  Returns an `_AliasedCsv` whose rows are alias-keyed
    dicts (optional aliases that did not resolve are absent from each row),
    `None` if no file was uploaded, or `_CsvError` on a malformed file.
    """
    if upload is None:
        return None
    try:
        reader = csv_io.read(upload)
    except csv_io.CsvError as exc:
        return _CsvError(str(exc))
    fieldset = set(reader.fieldnames)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for alias, candidates in required.items():
        hit = next((c for c in candidates if c in fieldset), None)
        if hit is None:
            missing.append(candidates[0])
        else:
            resolved[alias] = hit
    if missing:
        return _CsvError(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
    for alias, candidates in (optional or {}).items():
        hit = next((c for c in candidates if c in fieldset), None)
        if hit is not None:
            resolved[alias] = hit
    rows = [{alias: r.get(col, '') for alias, col in resolved.items()}
            for r in reader]
    return _AliasedCsv(reader, rows)


# Substrings (case-insensitive) recognised in the Note column for
# flag inference on import.  Matches the per-flag labels used by
# `render_flag_note` so a round-trip preserves the boolean.
_FLAG_KEYWORDS = {
    'damaged':   S.FLAG_DAMAGED.lower(),
    'unhealthy': S.FLAG_UNHEALTHY.lower(),
    'psr':       S.FLAG_PSR.lower(),
}


def _parse_flag_keywords(note: str) -> tuple[bool, bool, bool]:
    """Scan a Note cell for flag keywords. Returns (damaged, unhealthy, psr)."""
    s = (note or '').lower()
    return (
        _FLAG_KEYWORDS['damaged']   in s,
        _FLAG_KEYWORDS['unhealthy'] in s,
        _FLAG_KEYWORDS['psr']       in s,
    )


def _parse_fustaia_rows(data, parcel_cache, region_cache, errors):
    """Parse fustaia.csv rows.  A row is either parcel-scoped (the
    common case) or whole-region (Particella = ``X`` — see
    ``S.PARCEL_WHOLE_REGION_MARK``).  The Note column, when present,
    is scanned for ``Catastrofato`` / ``Fitosanitario`` / ``PSR``
    substrings to drive the damaged / unhealthy / psr flags — required
    for whole-region rows, optional for parcel-scoped ones (preserves
    flags on round-trip).
    """
    out = []
    for i, row in enumerate(data.rows, 2):
        compresa = (row.get('compresa') or '').strip()
        particella = (row.get('particella') or '').strip()
        note_raw = (row.get('note') or '').strip()
        damaged, unhealthy, psr = _parse_flag_keywords(note_raw)
        is_whole_region = (
            particella.upper() == S.PARCEL_WHOLE_REGION_MARK.upper()
        )
        if is_whole_region:
            region = region_cache.get(compresa.lower())
            if region is None:
                errors.append(S.ERR_CSV_REGION_NOT_FOUND.format(i, compresa))
                continue
            if not (damaged or unhealthy):
                errors.append(S.ERR_CSV_WHOLE_REGION_REQUIRES_FLAG.format(
                    i, particella,
                ))
                continue
            parcel = None
        else:
            region = None
            parcel = parcel_cache.get((compresa.lower(), particella))
            if parcel is None:
                errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                    i, compresa, particella,
                ))
                continue
        year = data.reader.integer(row.get('anno'))
        if year is None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_ANNO, row.get('anno', ''),
            ))
            continue
        volume = data.reader.decimal(row.get('volume'))
        out.append({
            FIELD_REGION_ID: region,
            FIELD_PARCEL_ID: parcel,
            FIELD_YEAR_PLANNED: year,
            FIELD_VOLUME_PLANNED_M3: volume,
            FIELD_DAMAGED:   damaged,
            FIELD_UNHEALTHY: unhealthy,
            FIELD_PSR:       psr,
        })
    return out


def _parse_ceduo_rows(data, parcel_cache, errors):
    """Parse ceduo.csv rows.  Disambiguates Abies-exported (`Altre note`
    = free-text + `Note` = flag string) from legacy pdg-2026 (`Note` =
    free-text, no flag column) via header presence: if `Altre note` is
    in the header, treat `Note` as the flag string; otherwise treat
    `Note` as free-text and skip flag parsing.
    """
    out = []
    has_altre_note = S.CSV_COL_EXTRA_NOTE in data.reader.fieldnames
    for i, row in enumerate(data.rows, 2):
        compresa = (row.get('compresa') or '').strip()
        particella = (row.get('particella') or '').strip()
        parcel = parcel_cache.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                i, compresa, particella,
            ))
            continue
        year = data.reader.integer(row.get('anno'))
        interval = data.reader.integer(row.get('turno'))
        area = data.reader.decimal(row.get('superficie'))
        if year is None or interval is None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_TURNO_A, row.get('turno', ''),
            ))
            continue
        if has_altre_note:
            free_note = (row.get('free_note') or '').strip()
            damaged, unhealthy, psr = _parse_flag_keywords(
                row.get('flag_note') or '',
            )
        else:
            # Legacy pdg-2026: 'Note' is free-text and there are no flags.
            free_note = (row.get('flag_note') or row.get('free_note') or '').strip()
            damaged = unhealthy = psr = False
        out.append({
            FIELD_PARCEL_ID: parcel,
            FIELD_YEAR_PLANNED: year,
            FIELD_INTERVENTION_AREA_HA: area,
            FIELD_TURNO_A: interval,
            FIELD_NOTE: free_note,
            FIELD_DAMAGED:   damaged,
            FIELD_UNHEALTHY: unhealthy,
            FIELD_PSR:       psr,
        })
    return out


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


def _conflict_response_plan(plan):
    return JsonResponse({
        STATUS: STATUS_CONFLICT, MESSAGE: S.ERROR_CONFLICT,
        DATA_ID: 'harvest_plans', ROW_ID: plan.id,
        RECORD: build_harvest_plan_record(plan),
    }, status=400)


def _conflict_response_item(item):
    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'parcel__eclass',
                            'region', 'harvest_plan')
            .get(id=item.id))
    return JsonResponse({
        STATUS: STATUS_CONFLICT, MESSAGE: S.ERROR_CONFLICT,
        DATA_ID: 'harvest_plan_items', ROW_ID: item.id,
        RECORD: build_harvest_plan_item_record(item),
    }, status=400)


_SAFE_RE = re.compile(r'[^A-Za-z0-9._-]+')


def _safe_filename(s: str) -> str:
    return _SAFE_RE.sub('_', s).strip('_') or 'plan'
