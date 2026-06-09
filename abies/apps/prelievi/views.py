"""Prelievi API views: data, form, save, delete.

New harvests carry a ``harvest_plan_item_id`` chosen from a Cantiere
pulldown (items in state ``open`` or ``harvesting``); the parcel and the
damaged/unhealthy/psr flags propagate from the linked plan item, and
the item state auto-advances ``open -> harvesting`` on the first
linked harvest insert.  Legacy CSV-imported rows without a plan item
remain editable but their parcel is treated as authoritative.
"""

import json
from datetime import date as date_type
from decimal import Decimal

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Sum
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import (
    aggregate_sp_pcts,
    prelievi_species_cols,
    build_harvest_plan_item_record,
    build_harvest_record,
    mark_stale,
    serve_digest,
)
from apps.base.numparse import int_or_none, parse_decimal
from apps.base.responses import (
    conflict_response, row_delete, row_patch, success_response,
    validation_error,
)
from apps.base.models import (
    Crew, HarvestPlanItem, HarvestPlanItemState, Parcel, Product, Species,
    Tractor, next_sequence_number, parcel_sort_key,
)
from apps.prelievi.models import (
    Harvest, HarvestSpecies, HarvestTractor, harvest_volume_m3,
)
from config import strings as S
from config.constants import (
    DATA_ID, FIELD_CREW_ID, FIELD_DATE, FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_MASS_Q, FIELD_NOTE, FIELD_PARCEL_ID, FIELD_PRODUCT_ID,
    FIELD_RECORD1, FIELD_RECORD2, FIELD_SORT_ORDER, FIELD_SPECIES_ID,
    FIELD_VOLUME_M3, HTML, MESSAGE, RECORD, ROW_ID, STATUS,
    VERSION,
)


# ---------------------------------------------------------------------------
# Data endpoint
# ---------------------------------------------------------------------------

@login_required
def data_view(request):
    """Serve prelievi.json.gz (conditional GET + lazy regeneration)."""
    return serve_digest(request, 'prelievi')


# ---------------------------------------------------------------------------
# Form endpoint
# ---------------------------------------------------------------------------

@login_required
def form_view(request, op_id=None):
    """Return add/edit form HTML fragment."""
    return JsonResponse({HTML: _render_form(op_id, request)})


# ---------------------------------------------------------------------------
# Save endpoint
# ---------------------------------------------------------------------------

_OPEN_STATES = (HarvestPlanItemState.OPEN, HarvestPlanItemState.HARVESTING)


@login_required
@require_writer
@require_POST
def save_view(request):
    """Create or update a harvest.

    For new harvests, a Cantiere (``harvest_plan_item_id``) is required
    and must be in state ``open`` or ``harvesting``.  The chosen item
    supplies the parcel and the three boolean flags.  Editing an
    existing harvest preserves its current plan-item link (legacy rows
    can have NULL) and re-syncs its flags to the linked item.
    """
    body = json.loads(request.body)

    row_id, parsed, errors = _parse_body(body)
    if errors:
        return _validation_error(errors, row_id, request, body)

    if row_id is not None:
        conflict = _check_update_conflict(row_id, body, request)
        if conflict is not None:
            return conflict

    record1 = parsed[FIELD_RECORD1]
    if record1 is not None:
        dup = Harvest.objects.filter(record1=record1)
        if row_id:
            dup = dup.exclude(id=row_id)
        if dup.exists():
            return _validation_error(
                [S.ERR_VDP_DUPLICATE.format(record1)], row_id, request, body,
            )

    sp_pcts, tr_pcts, pct_errors = _parse_percentages(body)
    if pct_errors:
        return _validation_error(pct_errors, row_id, request, body)

    parsed[FIELD_VOLUME_M3] = _compute_volume_m3(parsed[FIELD_MASS_Q], sp_pcts)

    with transaction.atomic():
        if row_id:
            op = _update_op(row_id, parsed, body, request)
            if isinstance(op, JsonResponse):
                return op
        else:
            op = Harvest.objects.create(**parsed)
            item = op.harvest_plan_item
            if (item is not None
                    and item.state == HarvestPlanItemState.OPEN):
                item.state = HarvestPlanItemState.HARVESTING
                item.version += 1
                item.save()

        _write_junctions(op, sp_pcts, tr_pcts)
        if op.harvest_plan_item_id is not None:
            _rematerialize_volume_actual(op.harvest_plan_item_id)
        stale = ['prelievi', 'parcel_year_production', 'audit']
        if op.harvest_plan_item_id is not None:
            stale.append('harvest_plan_items')
        mark_stale(*stale)

    op = Harvest.objects.select_related(
        'parcel__region', 'crew', 'product', 'harvest_plan_item',
    ).get(id=op.id)
    record = build_harvest_record(op)
    patches = [row_patch('prelievi', op.id, record)]
    if op.harvest_plan_item_id is not None:
        item_fresh = (HarvestPlanItem.objects
                      .select_related('parcel__region', 'parcel__eclass',
                                      'region', 'harvest_plan')
                      .get(id=op.harvest_plan_item_id))
        item_record = build_harvest_plan_item_record(item_fresh)
        patches.append(row_patch(
            'harvest_plan_items', item_fresh.id, item_record,
        ))

    return success_response(
        request, body, data_id='prelievi', row_id=op.id,
        patches=patches,
    )


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------

@login_required
@require_writer
@require_POST
def delete_view(request):
    """Delete a harvest (with version check)."""
    body = json.loads(request.body)
    row_id = int(body[ROW_ID])
    version = int(body.get(VERSION, 0))

    try:
        op = Harvest.objects.select_related(
            'parcel__region', 'crew', 'product', 'harvest_plan_item',
        ).get(id=row_id)
    except Harvest.DoesNotExist:
        return JsonResponse({MESSAGE: S.ERR_NOT_FOUND}, status=404)

    if op.version != version:
        return conflict_response(
            data_id='prelievi', row_id=row_id, record=build_harvest_record(op),
        )

    had_item_id = op.harvest_plan_item_id
    with transaction.atomic():
        op.delete()
        if had_item_id is not None:
            _rematerialize_volume_actual(had_item_id)
        stale = ['prelievi', 'parcel_year_production', 'audit']
        if had_item_id is not None:
            stale.append('harvest_plan_items')
        mark_stale(*stale)

    patches = []
    if had_item_id is not None:
        item_fresh = (HarvestPlanItem.objects
                      .select_related('parcel__region', 'parcel__eclass',
                                      'region', 'harvest_plan')
                      .get(id=had_item_id))
        item_record = build_harvest_plan_item_record(item_fresh)
        patches.append(row_patch(
            'harvest_plan_items', item_fresh.id, item_record,
        ))
    return success_response(
        request, body, data_id='prelievi', row_id=row_id,
        patches=patches, deletes=[row_delete('prelievi', row_id)],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rematerialize_volume_actual(item_id: int) -> None:
    """Recompute volume_actual_m3 on the linked HarvestPlanItem."""
    total = (Harvest.objects
             .filter(harvest_plan_item_id=item_id)
             .aggregate(s=Sum('volume_m3'))['s']) or 0
    HarvestPlanItem.objects.filter(id=item_id).update(
        volume_actual_m3=total,
    )


def _open_cantieri():
    """Items currently in state ``open`` or ``harvesting``.

    Used both to populate the Cantiere pulldown on the add form and to
    validate the submitted ``harvest_plan_item_id`` server-side.
    """
    return (HarvestPlanItem.objects
            .filter(state__in=_OPEN_STATES)
            .select_related('parcel__region', 'region', 'harvest_plan')
            .order_by('harvest_plan__name', 'year_planned', 'id'))


def _cantiere_label(item) -> str:
    """Human-readable Cantiere pulldown label."""
    plan = item.harvest_plan.name
    if item.parcel_id is not None:
        scope = f'{item.parcel.region.name} {item.parcel.name}'
    else:
        scope = f'{item.region.name} {S.LABEL_ALL_PARCELS}'
    return f'{plan} · {item.year_planned} · {scope}'


def _cantiere_region_id(item) -> int:
    """Region id of an open cantiere — used by the form's Particella
    pulldown to filter on the client side when the item is region-wide.
    """
    if item.parcel_id is not None:
        return item.parcel.region_id
    return item.region_id


def _form_context(op_id=None, vals=None):
    """Build template context for the prelievi form."""
    op = None
    sp_pcts = {}
    tr_pcts = {}

    if op_id:
        op = Harvest.objects.select_related(
            'parcel__region', 'crew', 'product', 'harvest_plan_item',
        ).get(id=op_id)
        sp_pcts = dict(
            HarvestSpecies.objects.filter(harvest=op).values_list(FIELD_SPECIES_ID, 'percent'),
        )
        tr_pcts = dict(
            HarvestTractor.objects.filter(harvest=op).values_list('tractor_id', 'percent'),
        )
    elif vals:
        for key, val in vals.items():
            if key.startswith('sp_') and val:
                sp_pcts[int(key[3:])] = int(val)
            elif key.startswith('tr_') and val:
                tr_pcts[int(key[3:])] = int(val)

    # Aggregate minor-species percentages into Altro for display.
    _, _, minor_ids, other_id = prelievi_species_cols()
    sp_pcts = aggregate_sp_pcts(sp_pcts, minor_ids, other_id)

    v = vals or {}

    # Fresh add form: pre-fill VDP with the next free value, max(VDP)+1.  On
    # edits and on re-render after a validation error the stored/submitted
    # value governs (via the template), so no default is injected.
    default_record1 = ''
    if op_id is None and not vals:
        default_record1 = next_sequence_number(Harvest.objects, FIELD_RECORD1)

    cantieri = [
        {
            'id': it.id,
            'label': _cantiere_label(it),
            'damaged': it.damaged,
            'unhealthy': it.unhealthy,
            'psr': it.psr,
            'region_id': _cantiere_region_id(it),
            'parcel_id': it.parcel_id,  # NULL for region-wide cantieri.
        }
        for it in _open_cantieri()
    ]
    return {
        'op': op,
        'vals': v,
        'default_record1': default_record1,
        'cantieri': cantieri,
        'parcels': sorted(Parcel.objects.select_related('region'),
                         key=parcel_sort_key),
        'crews': Crew.objects.filter(active=True).order_by('name'),
        'products': Product.objects.order_by('name'),
        'species_data': [
            (sp.id, sp.common_name, sp_pcts.get(sp.id, 0))
            for sp in Species.objects.filter(active=True, minor=False)
                                     .order_by(FIELD_SORT_ORDER)
        ],
        'tractor_data': [
            (tr.id, f'{tr.manufacturer} {tr.model}'.strip(), tr_pcts.get(tr.id, 0))
            for tr in Tractor.objects.filter(active=True).order_by('manufacturer', 'model')
        ],
    }


def _render_form(op_id, request, vals=None):
    return render_to_string(
        'prelievi/_form.html', _form_context(op_id, vals), request=request,
    )


def _parse_body(body):
    """Extract and validate core fields from the POST body.

    Returns (row_id, parsed_dict, error_list).  For new harvests the
    ``harvest_plan_item_id`` is mandatory; for edits a NULL link on a
    legacy row is preserved.
    """
    errors = []
    row_id = body.get(ROW_ID)
    row_id = int(row_id) if row_id else None

    date_str = body.get(FIELD_DATE)
    if not date_str:
        errors.append(S.ERR_DATE_REQUIRED)
    else:
        try:
            if date_type.fromisoformat(date_str) > date_type.today():
                errors.append(S.ERR_DATE_FUTURE)
        except ValueError:
            errors.append(S.ERR_DATE_REQUIRED)

    mass_q = parse_decimal(body.get(FIELD_MASS_Q)) or Decimal(0)
    if mass_q <= 0:
        errors.append(S.ERR_QUINTALS_POSITIVE)

    record1 = body.get(FIELD_RECORD1)

    # Resolve the Cantiere link.  New rows: mandatory + must be open.
    # Edits: optional (legacy NULL preserved), but if supplied the new
    # link must point at an item in state OPEN or HARVESTING.
    item_id_raw = body.get(FIELD_HARVEST_PLAN_ITEM_ID)
    item = None
    if row_id is None:
        if not item_id_raw:
            errors.append(S.ERR_CANTIERE_REQUIRED)
        else:
            item = HarvestPlanItem.objects.filter(id=int(item_id_raw)).first()
            if item is None or item.state not in _OPEN_STATES:
                errors.append(S.ERR_CANTIERE_STATE_INVALID)
    else:
        if item_id_raw:
            item = HarvestPlanItem.objects.filter(id=int(item_id_raw)).first()
            if item is None or item.state not in _OPEN_STATES:
                errors.append(S.ERR_CANTIERE_STATE_INVALID)

    parsed = {
        FIELD_DATE: date_str,
        FIELD_CREW_ID: int(body[FIELD_CREW_ID]) if body.get(FIELD_CREW_ID) else None,
        FIELD_PRODUCT_ID: int(body[FIELD_PRODUCT_ID]) if body.get(FIELD_PRODUCT_ID) else None,
        FIELD_RECORD1: int(record1) if record1 else None,
        FIELD_MASS_Q: mass_q,
        FIELD_NOTE: body.get(FIELD_NOTE, ''),
    }
    if item is not None:
        # Parcel-scoped cantiere: parcel derives from the item; any
        # submitted parcel_id is ignored on purpose so a hand-crafted
        # POST cannot mis-attribute a harvest.
        if item.parcel_id is not None:
            parsed[FIELD_PARCEL_ID] = item.parcel_id
        else:
            # Region-wide cantiere (damaged / unhealthy operation
            # spanning a whole region): the operator must still
            # point the harvest at a specific parcel inside that
            # region.
            submitted_parcel_id = int_or_none(body.get(FIELD_PARCEL_ID))
            if submitted_parcel_id is None:
                errors.append(S.ERR_PARCEL_REQUIRED_FOR_REGION_WIDE)
            else:
                parcel = Parcel.objects.filter(
                    id=submitted_parcel_id,
                ).only('id', 'region_id').first()
                if parcel is None or parcel.region_id != item.region_id:
                    errors.append(S.ERR_PARCEL_NOT_IN_REGION)
                else:
                    parsed[FIELD_PARCEL_ID] = parcel.id
        parsed['harvest_plan_item_id'] = item.id
        parsed['damaged'] = item.damaged
        parsed['unhealthy'] = item.unhealthy
        parsed['psr'] = item.psr
    elif row_id is not None:
        # Edit of a legacy row whose Cantiere stays NULL: parcel and
        # flags travel through on the existing row (handled in _update_op).
        pass

    if FIELD_RECORD2 in body:
        record2 = body[FIELD_RECORD2]
        parsed[FIELD_RECORD2] = int(record2) if record2 else None
    return row_id, parsed, errors


def _parse_percentages(body):
    sp_pcts = {}
    tr_pcts = {}
    for key, val in body.items():
        if not val:
            continue
        if key.startswith('sp_'):
            pct = int(val)
            if pct > 0:
                sp_pcts[int(key[3:])] = pct
        elif key.startswith('tr_'):
            pct = int(val)
            if pct > 0:
                tr_pcts[int(key[3:])] = pct

    errors = []
    if sum(sp_pcts.values()) != 100:
        errors.append(S.ERR_SPECIES_PCT_SUM)
    if sum(tr_pcts.values()) != 100:
        errors.append(S.ERR_TRACTOR_PCT_SUM)
    return sp_pcts, tr_pcts, errors


def _compute_volume_m3(mass_q: Decimal, sp_pcts: dict[int, int]) -> Decimal:
    """Materialised harvest volume from mass + per-species percentages, using
    current Species.density values (later density edits don't retroactively
    change stored volumes)."""
    if not sp_pcts:
        return Decimal('0')
    id_density = Species.objects.filter(id__in=sp_pcts.keys()).values_list('id', 'density')
    density_by_id = {sid: density for sid, density in id_density}
    return harvest_volume_m3(
        mass_q, ((density_by_id.get(sid), pct) for sid, pct in sp_pcts.items()),
    )


def _conflict_response(row_id, request):
    op = Harvest.objects.select_related(
        'parcel__region', 'crew', 'product', 'harvest_plan_item',
    ).get(id=row_id)
    return conflict_response(
        data_id='prelievi', row_id=row_id, record=build_harvest_record(op),
        html=_render_form(row_id, request),
    )


def _check_update_conflict(row_id, body, request):
    try:
        actual_version = Harvest.objects.values_list(VERSION, flat=True).get(id=row_id)
    except Harvest.DoesNotExist:
        return JsonResponse({MESSAGE: S.ERR_NOT_FOUND}, status=404)
    if actual_version == int(body.get(VERSION, 0)):
        return None
    return _conflict_response(row_id, request)


def _update_op(row_id, parsed, body, request):
    """Update an existing Harvest under row lock."""
    version = int(body.get(VERSION, 0))
    op = Harvest.objects.select_for_update().get(id=row_id)
    if op.version != version:
        return _conflict_response(row_id, request)

    for field, value in parsed.items():
        setattr(op, field, value)
    op.version += 1
    op.save()
    return op


def _write_junctions(op, sp_pcts, tr_pcts):
    HarvestSpecies.objects.filter(harvest=op).delete()
    HarvestTractor.objects.filter(harvest=op).delete()
    HarvestSpecies.objects.bulk_create([
        HarvestSpecies(harvest=op, species_id=sid, percent=pct)
        for sid, pct in sp_pcts.items()
    ])
    HarvestTractor.objects.bulk_create([
        HarvestTractor(harvest=op, tractor_id=tid, percent=pct)
        for tid, pct in tr_pcts.items()
    ])


def _validation_error(errors, row_id, request, vals=None):
    return validation_error(errors, html=_render_form(row_id, request, vals))
