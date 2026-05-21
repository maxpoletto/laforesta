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
from datetime import date as date_type
from decimal import Decimal, InvalidOperation
from typing import Iterable

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Q
from django.http import Http404, HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import (
    build_harvest_plan_item_record,
    build_harvest_plan_record,
    build_harvest_record,
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
    TreeHeightRegression,
    render_flag_note,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S
from config.constants import (
    COLUMNS,
    DATA_ID,
    FIELD_CEDUO_FILE,
    FIELD_DAMAGED,
    FIELD_DATE,
    FIELD_DESCRIPTION,
    FIELD_ERRORS,
    FIELD_FUSTAIA_FILE,
    FIELD_HARVEST_PLAN_ID,
    FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_INTERVENTION_AREA_HA,
    FIELD_NAME,
    FIELD_NONCE,
    FIELD_NOTE,
    FIELD_OPEN,
    FIELD_PARCEL_ID,
    FIELD_PSR,
    FIELD_REGION_ID,
    FIELD_REGRESSION_FILE,
    FIELD_TURNO_A,
    FIELD_UNHEALTHY,
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
    ROW_ID,
    ROWS,
    STATUS,
    STATUS_CONFLICT,
    STATUS_NOT_FOUND,
    STATUS_VALIDATION_ERROR,
    TRANSITION_RECORDS,
    VERSION,
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
def regressions_data_view(request):
    return serve_digest(request, 'tree_height_regressions')


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
    plan_id = _int_or_none(body.get(ROW_ID))

    name, description, year_start, year_end, errors = _parse_plan_body(body)
    if errors:
        return _validation_error(errors)

    # Uniqueness check (case-insensitive — `HarvestPlan.name` has a
    # plain unique constraint, but we surface a friendly message early).
    dup = HarvestPlan.objects.filter(name__iexact=name)
    if plan_id:
        dup = dup.exclude(id=plan_id)
    if dup.exists():
        return _validation_error([S.ERR_PLAN_NAME_DUPLICATE])

    with transaction.atomic():
        if plan_id is not None:
            plan = HarvestPlan.objects.select_for_update().filter(
                id=plan_id,
            ).first()
            if plan is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            submitted_version = _int_or_none(body.get(VERSION))
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
        return _validation_error([S.ERR_PLAN_HAS_ACTIVE_ITEMS])

    with transaction.atomic():
        # HarvestPlanItem and TreeHeightRegression both cascade to HarvestPlan
        # in the schema.  ParcelPlanDetail also cascades on plan delete.
        plan.delete()
        mark_stale(
            'harvest_plans', 'harvest_plan_items',
            'tree_height_regressions', 'audit',
        )
    return JsonResponse({DATA_ID: 'harvest_plans', ROW_ID: plan_id})


# ---------------------------------------------------------------------------
# Plan CSV import — single endpoint dispatching on which file(s) attached
# ---------------------------------------------------------------------------

FUSTAIA_CSV_REQUIRED = [
    S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA,
    S.CSV_COL_ANNO, S.CSV_COL_PRELIEVO_M3,
]
CEDUO_CSV_REQUIRED = [
    S.CSV_COL_ANNO, S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA,
    S.CSV_COL_SUPERFICIE_HA, S.CSV_COL_TURNO_A,
]
REGRESSION_CSV_REQUIRED = [
    S.CSV_COL_COMPRESA, S.CSV_COL_GENERE, S.CSV_COL_FUNZIONE,
    S.CSV_COL_A, S.CSV_COL_B, S.CSV_COL_R2, S.CSV_COL_N_REGRESSION,
]

_DEFAULT_REGRESSION_FN = 'ln'


@login_required
@require_writer
@require_POST
def plan_csv_import_view(request):
    """Create a HarvestPlan from one or more uploaded CSVs.

    Multipart form fields:
      - ``name``            — required: plan name.
      - ``description``     — optional plan description.
      - ``fustaia_file``    — optional: piano.csv (high-forest schedule).
      - ``ceduo_file``      — optional: ceduo.csv (coppice schedule).
      - ``regression_file`` — optional: equazioni_ipsometro.csv.

    At least one of the three CSV files must be present.  ``year_start``
    and ``year_end`` are derived from the union of ``Anno`` columns
    seen across the calendar files (regression CSV has no year column).
    """
    name = (request.POST.get(FIELD_NAME) or '').strip()
    description = (request.POST.get(FIELD_DESCRIPTION) or '').strip()
    if not name:
        return _validation_error([S.ERR_PLAN_NAME_REQUIRED])
    if HarvestPlan.objects.filter(name__iexact=name).exists():
        return _validation_error([S.ERR_PLAN_NAME_DUPLICATE])

    fustaia_rows = _read_optional(
        request.FILES.get(FIELD_FUSTAIA_FILE), FUSTAIA_CSV_REQUIRED,
    )
    ceduo_rows = _read_optional(
        request.FILES.get(FIELD_CEDUO_FILE), CEDUO_CSV_REQUIRED,
    )
    regression_rows = _read_optional(
        request.FILES.get(FIELD_REGRESSION_FILE), REGRESSION_CSV_REQUIRED,
    )
    if (fustaia_rows is None and ceduo_rows is None
            and regression_rows is None):
        return _validation_error([S.ERR_CSV_NO_FILES])
    # _read_optional returns either rows-list or a _CsvError-tagged tuple.
    for rows in (fustaia_rows, ceduo_rows, regression_rows):
        if isinstance(rows, _CsvError):
            return _validation_error([rows.message])

    region_cache = {r.name.lower(): r for r in Region.objects.all()}
    parcel_cache = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }
    species_cache = {sp.common_name.lower(): sp
                     for sp in Species.objects.all()}

    errors: list[str] = []
    fustaia_parsed = _parse_fustaia_rows(
        fustaia_rows or [], parcel_cache, errors,
    )
    ceduo_parsed = _parse_ceduo_rows(
        ceduo_rows or [], parcel_cache, errors,
    )
    regression_parsed = _parse_regression_rows(
        regression_rows or [], region_cache, species_cache, errors,
    )
    if errors:
        return _csv_error_list(errors)

    all_years = (
        [r[FIELD_YEAR_PLANNED] for r in fustaia_parsed]
        + [r[FIELD_YEAR_PLANNED] for r in ceduo_parsed]
    )
    if all_years:
        year_start = min(all_years)
        year_end = max(all_years)
    else:
        # Regression-only import: year range still needs a value.  Default to
        # the current civil year.
        year_start = year_end = date_type.today().year

    with transaction.atomic():
        plan = HarvestPlan.objects.create(
            name=name, description=description,
            year_start=year_start, year_end=year_end,
        )

        # Coppice harvest_detail / parcel_plan_detail rows: deduplicate
        # on (description, interval) within this plan so coppice parcels
        # sharing a rotation share one harvest_detail row.
        detail_cache: dict[tuple[str, int], HarvestDetail] = {}
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
            ParcelPlanDetail.objects.get_or_create(
                harvest_plan=plan, parcel=r[FIELD_PARCEL_ID],
                defaults={'harvest_detail': hd},
            )

        items_created = []
        for r in fustaia_parsed:
            items_created.append(HarvestPlanItem.objects.create(
                harvest_plan=plan,
                parcel=r[FIELD_PARCEL_ID],
                year_planned=r[FIELD_YEAR_PLANNED],
                volume_planned_m3=r[FIELD_VOLUME_PLANNED_M3],
                state=HarvestPlanItemState.PLANNED,
            ))
        for r in ceduo_parsed:
            items_created.append(HarvestPlanItem.objects.create(
                harvest_plan=plan,
                parcel=r[FIELD_PARCEL_ID],
                year_planned=r[FIELD_YEAR_PLANNED],
                intervention_area_ha=r[FIELD_INTERVENTION_AREA_HA],
                note=r[FIELD_NOTE],
                state=HarvestPlanItemState.PLANNED,
            ))

        for r in regression_parsed:
            TreeHeightRegression.objects.create(
                harvest_plan=plan,
                region=r[FIELD_REGION_ID],
                species=r['species'],
                function=r['function'],
                a=r['a'], b=r['b'], r2=r['r2'], n=r['n'],
            )

        mark_stale(
            'harvest_plans', 'harvest_plan_items',
            'tree_height_regressions', 'audit',
        )

    response_data = {
        DATA_ID: 'harvest_plans',
        ROW_ID: plan.id,
        RECORD: build_harvest_plan_record(plan),
        'n_items': len(items_created),
        'n_regressions': len(regression_parsed),
    }
    nonce = request.POST.get(FIELD_NONCE)
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Plan-level Esporta CSV — zip of piano.csv + ceduo.csv + equazioni.csv
# ---------------------------------------------------------------------------

@login_required
def plan_export_view(request, plan_id: int):
    """Return a zip of the three round-trip CSVs for a plan."""
    plan = HarvestPlan.objects.filter(id=plan_id).first()
    if plan is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    items = (HarvestPlanItem.objects
             .filter(harvest_plan=plan)
             .select_related('parcel__region', 'parcel__eclass', 'region')
             .order_by('year_planned', 'id'))
    regressions = (TreeHeightRegression.objects
                   .filter(harvest_plan=plan)
                   .select_related('region', 'species')
                   .order_by('region__name', 'species__common_name'))
    parcel_intervals = {
        ppd.parcel_id: ppd.harvest_detail.interval
        for ppd in ParcelPlanDetail.objects
                       .filter(harvest_plan=plan)
                       .select_related('harvest_detail')
        if ppd.harvest_detail.interval is not None
    }

    fustaia_buf = io.StringIO()
    fustaia_w = csv.writer(fustaia_buf)
    fustaia_w.writerow([
        S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA,
        S.CSV_COL_ANNO, S.CSV_COL_PRELIEVO_M3,
    ])

    ceduo_buf = io.StringIO()
    ceduo_w = csv.writer(ceduo_buf)
    ceduo_w.writerow([
        S.CSV_COL_ANNO, S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA,
        S.CSV_COL_SUPERFICIE_HA, S.CSV_COL_SUPERFICIE_TOT_HA,
        S.CSV_COL_TURNO_A, S.CSV_COL_NOTE,
    ])

    for it in items:
        if it.parcel_id is None:
            # Region-wide items don't round-trip through these CSVs (no
            # parcel column).  Skip them — they exist only as manual
            # entries from the Nuovo intervento modal.
            continue
        is_coppice = it.parcel.eclass.coppice
        if is_coppice:
            ceduo_w.writerow([
                it.year_planned,
                it.parcel.region.name,
                it.parcel.name,
                _fmt_decimal(it.intervention_area_ha),
                _fmt_decimal(it.parcel.area_ha),
                parcel_intervals.get(it.parcel_id, ''),
                it.note or '',
            ])
        else:
            fustaia_w.writerow([
                it.parcel.region.name,
                it.parcel.name,
                it.year_planned,
                _fmt_decimal(it.volume_planned_m3),
            ])

    regression_buf = io.StringIO()
    regression_w = csv.writer(regression_buf)
    regression_w.writerow([
        S.CSV_COL_COMPRESA, S.CSV_COL_GENERE, S.CSV_COL_FUNZIONE,
        S.CSV_COL_A, S.CSV_COL_B, S.CSV_COL_R2, S.CSV_COL_N_REGRESSION,
    ])
    for r in regressions:
        regression_w.writerow([
            r.region.name,
            r.species.common_name,
            r.function,
            _fmt_decimal(r.a),
            _fmt_decimal(r.b),
            _fmt_decimal(r.r2),
            r.n,
        ])

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('piano.csv', fustaia_buf.getvalue())
        zf.writestr('ceduo.csv', ceduo_buf.getvalue())
        zf.writestr('equazioni_ipsometro.csv', regression_buf.getvalue())

    response = HttpResponse(zip_buf.getvalue(), content_type='application/zip')
    safe_name = _safe_filename(plan.name)
    response['Content-Disposition'] = (
        f'attachment; filename="piano_{safe_name}.zip"'
    )
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
        plan_id = _int_or_none(request.GET.get('plan'))
        if plan_id is None:
            raise Http404('plan required')
        plan = HarvestPlan.objects.filter(id=plan_id).first()
        if plan is None:
            raise Http404
    regions = list(Region.objects.order_by('name'))
    parcels = list(Parcel.objects.select_related('region', 'eclass')
                                  .order_by('region__name', 'name'))
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
    item_id = _int_or_none(body.get(ROW_ID))

    parsed, errors = _parse_item_body(body)
    if errors:
        return _validation_error(errors)

    with transaction.atomic():
        if item_id is not None:
            item = HarvestPlanItem.objects.select_for_update().filter(
                id=item_id,
            ).first()
            if item is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            submitted_version = _int_or_none(body.get(VERSION))
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
                return _validation_error([str(exc)])
        else:
            plan_id = _int_or_none(body.get(FIELD_HARVEST_PLAN_ID))
            if plan_id is None:
                return _validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
            plan = HarvestPlan.objects.filter(id=plan_id).first()
            if plan is None:
                return _validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
            item = HarvestPlanItem(
                harvest_plan=plan,
                state=HarvestPlanItemState.PLANNED,
                **parsed,
            )
            try:
                item.clean()
                item.save()
            except Exception as exc:
                return _validation_error([str(exc)])
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
        return _validation_error([S.ERR_PLAN_ITEM_STATE_NOT_PLANNED])

    with transaction.atomic():
        try:
            item.delete()
        except Exception:
            return _validation_error([S.ERR_PLAN_ITEM_HAS_DEPS])
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
    from apps.base.models import TreeMark
    item = (HarvestPlanItem.objects
            .select_related('parcel__region', 'region', 'harvest_plan')
            .filter(id=item_id)
            .first())
    if item is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)

    # martellate_<id>.csv
    marks_buf = io.StringIO()
    marks_w = csv.writer(marks_buf)
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
            tm.id,
            tm.tree.species.common_name,
            tm.d_cm,
            _fmt_decimal(tm.h_m),
            '1' if tm.h_measured else '0',
            tm.lat,
            tm.lon,
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
    prelievi_w = csv.writer(prelievi_buf)
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
             _fmt_decimal(h.mass_q),
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
    response['Content-Disposition'] = (
        f'attachment; filename="intervento_{item.id}.zip"'
    )
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
    item_id = _int_or_none(body.get(FIELD_HARVEST_PLAN_ITEM_ID))
    open_flag = bool(body.get(FIELD_OPEN))
    date_raw = (body.get(FIELD_DATE) or '').strip()
    note = (body.get(FIELD_NOTE) or '').strip()

    if item_id is None:
        return _validation_error([S.ERR_PLAN_ITEM_NOT_FOUND])
    if not date_raw:
        return _validation_error([S.ERR_TRANSITION_DATE_REQUIRED])
    try:
        date = date_type.fromisoformat(date_raw)
    except ValueError:
        return _validation_error([S.ERR_DATE_INVALID])

    with transaction.atomic():
        item = HarvestPlanItem.objects.select_for_update().filter(
            id=item_id,
        ).first()
        if item is None:
            return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
        cur = HarvestPlanItemState(item.state)
        new_state = _ALLOWED_TRANSITIONS.get((cur, open_flag))
        if new_state is None:
            return _validation_error([S.ERR_TRANSITION_INVALID_STATE])

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
# Internal helpers — parsing
# ---------------------------------------------------------------------------

class _CsvError:
    def __init__(self, message: str):
        self.message = message


def _read_optional(upload, required_cols):
    """Parse an uploaded CSV if present.  Returns a list of dicts, None
    if no file was uploaded, or a _CsvError sentinel on parse failure.
    """
    if upload is None:
        return None
    try:
        text = upload.read().decode('utf-8-sig')
    except UnicodeDecodeError:
        return _CsvError(S.ERR_CSV_NOT_UTF8)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return _CsvError(S.ERR_CSV_EMPTY)
    missing = [c for c in required_cols if c not in reader.fieldnames]
    if missing:
        return _CsvError(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
    return list(reader)


def _parse_fustaia_rows(rows, parcel_cache, errors):
    out = []
    for i, row in enumerate(rows, 2):
        compresa = (row.get(S.CSV_COL_COMPRESA) or '').strip()
        particella = (row.get(S.CSV_COL_PARTICELLA) or '').strip()
        parcel = parcel_cache.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(i, compresa, particella))
            continue
        try:
            year = int(float(row[S.CSV_COL_ANNO]))
            volume = _decimal_or_none(row[S.CSV_COL_PRELIEVO_M3])
        except (ValueError, KeyError) as exc:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_ANNO, str(exc),
            ))
            continue
        out.append({
            FIELD_PARCEL_ID: parcel,
            FIELD_YEAR_PLANNED: year,
            FIELD_VOLUME_PLANNED_M3: volume,
        })
    return out


def _parse_ceduo_rows(rows, parcel_cache, errors):
    out = []
    for i, row in enumerate(rows, 2):
        compresa = (row.get(S.CSV_COL_COMPRESA) or '').strip()
        particella = (row.get(S.CSV_COL_PARTICELLA) or '').strip()
        parcel = parcel_cache.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(i, compresa, particella))
            continue
        try:
            year = int(float(row[S.CSV_COL_ANNO]))
            area = _decimal_or_none(row[S.CSV_COL_SUPERFICIE_HA])
            interval = int(float(row[S.CSV_COL_TURNO_A]))
        except (ValueError, KeyError) as exc:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_TURNO_A, str(exc),
            ))
            continue
        note = (row.get(S.CSV_COL_NOTE) or '').strip()
        out.append({
            FIELD_PARCEL_ID: parcel,
            FIELD_YEAR_PLANNED: year,
            FIELD_INTERVENTION_AREA_HA: area,
            FIELD_TURNO_A: interval,
            FIELD_NOTE: note,
        })
    return out


def _parse_regression_rows(rows, region_cache, species_cache, errors):
    out = []
    for i, row in enumerate(rows, 2):
        compresa = (row.get(S.CSV_COL_COMPRESA) or '').strip()
        genere = (row.get(S.CSV_COL_GENERE) or '').strip()
        function = (row.get(S.CSV_COL_FUNZIONE) or '').strip().lower()
        region = region_cache.get(compresa.lower())
        if region is None:
            errors.append(S.ERR_CSV_REGION_NOT_FOUND.format(i, compresa))
            continue
        species = species_cache.get(genere.lower())
        if species is None:
            errors.append(S.ERR_CSV_SPECIES_NOT_FOUND.format(i, genere))
            continue
        if function != _DEFAULT_REGRESSION_FN:
            errors.append(S.ERR_CSV_FUNCTION_INVALID.format(i, function))
            continue
        try:
            a = Decimal(row[S.CSV_COL_A])
            b = Decimal(row[S.CSV_COL_B])
            r2 = Decimal(row[S.CSV_COL_R2])
            n = int(float(row[S.CSV_COL_N_REGRESSION]))
        except (InvalidOperation, ValueError, KeyError) as exc:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_A, str(exc),
            ))
            continue
        out.append({
            FIELD_REGION_ID: region, 'species': species,
            'function': function, 'a': a, 'b': b, 'r2': r2, 'n': n,
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
    """Extract + validate plan-item fields.  Returns (parsed_dict, errors)."""
    errors: list[str] = []
    region_id = _int_or_none(body.get(FIELD_REGION_ID))
    parcel_id = _int_or_none(body.get(FIELD_PARCEL_ID))
    if (region_id is None) == (parcel_id is None):
        errors.append(S.ERR_PLAN_ITEM_REGION_XOR_PARCEL)
    damaged = bool(body.get(FIELD_DAMAGED))
    unhealthy = bool(body.get(FIELD_UNHEALTHY))
    psr = bool(body.get(FIELD_PSR))
    if region_id is not None and not (damaged or unhealthy):
        errors.append(S.ERR_PLAN_ITEM_REGION_REQUIRES_FLAG)
    try:
        year_planned = int(body.get(FIELD_YEAR_PLANNED))
    except (TypeError, ValueError):
        errors.append(S.ERR_DATE_REQUIRED)
        year_planned = 0
    volume = _decimal_or_none(body.get(FIELD_VOLUME_PLANNED_M3))
    if volume is not None and volume <= 0:
        errors.append(S.ERR_PLAN_ITEM_VOLUME_NEGATIVE)
        volume = None
    area = _decimal_or_none(body.get(FIELD_INTERVENTION_AREA_HA))
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


def _validation_error(errors):
    return JsonResponse({
        STATUS: STATUS_VALIDATION_ERROR,
        MESSAGE: ' '.join(errors),
        FIELD_ERRORS: errors,
        HTML: '',
    }, status=400)


def _csv_error_list(errors):
    return JsonResponse({
        STATUS: STATUS_VALIDATION_ERROR,
        MESSAGE: errors[0] if errors else '',
        FIELD_ERRORS: errors,
        HTML: '',
    }, status=400)


def _int_or_none(v):
    if v in (None, '', 'null'):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _decimal_or_none(v):
    if v in (None, '', 'null'):
        return None
    try:
        return Decimal(str(v))
    except (TypeError, InvalidOperation):
        return None


def _fmt_decimal(v) -> str:
    if v is None:
        return ''
    if isinstance(v, Decimal):
        # Strip trailing zeros for round-trip fidelity with the
        # `.0` / `.00` style pdg-2026 emits.
        s = format(v, 'f')
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s or '0'
    return str(v)


_SAFE_RE = re.compile(r'[^A-Za-z0-9._-]+')


def _safe_filename(s: str) -> str:
    return _SAFE_RE.sub('_', s).strip('_') or 'plan'
