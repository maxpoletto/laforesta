"""Prelievi API views: data, form, save, delete."""

import json
from datetime import date as date_type
from decimal import Decimal, InvalidOperation

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import build_harvest_record, mark_stale, serve_digest
from apps.base.middleware import save_nonce
from apps.base.models import Crew, Note, Parcel, Product, Region, Species, Tractor
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S


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
    return JsonResponse({S.HTML: _render_form(op_id, request)})


# ---------------------------------------------------------------------------
# Save endpoint
# ---------------------------------------------------------------------------

@login_required
@require_writer
@require_POST
def save_view(request):
    """Create or update a harvest."""
    body = json.loads(request.body)

    row_id, parsed, errors = _parse_body(body)
    if errors:
        return _validation_error(errors, row_id, request, body)

    # Pre-flight version check for updates; surface conflicts before the
    # other validations so the user re-submits against current state.
    if row_id is not None:
        conflict = _check_update_conflict(row_id, body, request)
        if conflict is not None:
            return conflict

    # VDP uniqueness
    record1 = parsed['record1']
    if record1 is not None:
        dup = Harvest.objects.filter(record1=record1)
        if row_id:
            dup = dup.exclude(id=row_id)
        if dup.exists():
            return _validation_error([S.ERR_VDP_DUPLICATE.format(record1)], row_id, request, body)

    # Species / tractor percentages
    sp_pcts, tr_pcts, pct_errors = _parse_percentages(body)
    if pct_errors:
        return _validation_error(pct_errors, row_id, request, body)

    # Materialize volume_m3 from species mix and current densities.
    parsed['volume_m3'] = _compute_volume_m3(parsed['quintals'], sp_pcts)

    with transaction.atomic():
        if row_id:
            op = _update_op(row_id, parsed, body, request)
            if isinstance(op, JsonResponse):
                return op  # race conflict
        else:
            op = Harvest.objects.create(**parsed)

        _write_junctions(op, sp_pcts, tr_pcts)
        mark_stale('prelievi', 'parcel_year_production', 'audit')

    op = Harvest.objects.select_related('parcel__region', 'crew', 'note', 'product').get(id=op.id)
    record = build_harvest_record(op)
    response_data = {S.DATA_ID: 'prelievi', S.ROW_ID: op.id, S.RECORD: record}

    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)

    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------

@login_required
@require_writer
@require_POST
def delete_view(request):
    """Delete a harvest (with version check)."""
    body = json.loads(request.body)
    row_id = int(body[S.ROW_ID])
    version = int(body.get(S.VERSION, 0))

    try:
        op = Harvest.objects.select_related('parcel__region', 'crew', 'note', 'product').get(id=row_id)
    except Harvest.DoesNotExist:
        return JsonResponse({S.MESSAGE: S.ERR_NOT_FOUND}, status=404)

    if op.version != version:
        record = build_harvest_record(op)
        return JsonResponse({
            S.STATUS: S.STATUS_CONFLICT, S.MESSAGE: S.ERROR_CONFLICT,
            S.DATA_ID: 'prelievi', S.ROW_ID: row_id, S.RECORD: record,
        }, status=400)

    with transaction.atomic():
        op.delete()
        mark_stale('prelievi', 'parcel_year_production', 'audit')

    response_data = {S.DATA_ID: 'prelievi', S.ROW_ID: row_id}

    nonce = body.get('nonce')
    if nonce:
        save_nonce(nonce, request.user, response_data)

    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _form_context(op_id=None, vals=None):
    """Build template context for the prelievi form.

    *vals* is the raw POST body dict, used to re-populate the form after a
    validation error on a new entry (where there is no *op* to read from).
    """
    op = None
    sp_pcts = {}
    tr_pcts = {}

    if op_id:
        op = Harvest.objects.select_related(
            'parcel__region', 'crew', 'note',
        ).get(id=op_id)
        sp_pcts = dict(
            HarvestSpecies.objects.filter(harvest=op).values_list('species_id', 'percent'),
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

    # vals dict for re-populating non-percentage fields on new-entry errors.
    v = vals or {}

    return {
        'op': op,
        'vals': v,
        'regions': Region.objects.order_by('name'),
        'parcels': Parcel.objects.exclude(name='X')
                        .select_related('region').order_by('region__name', 'name'),
        'crews': Crew.objects.filter(active=True).order_by('name'),
        'products': Product.objects.order_by('name'),
        'notes': Note.objects.order_by('name'),
        'species_data': [
            (sp.id, sp.common_name, sp_pcts.get(sp.id, 0))
            for sp in Species.objects.filter(active=True).order_by('sort_order')
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

    Returns (row_id, parsed_dict, error_list).
    """
    errors = []
    row_id = body.get(S.ROW_ID)
    row_id = int(row_id) if row_id else None

    date = body.get('date')
    if not date:
        errors.append(S.ERR_DATE_REQUIRED)
    else:
        try:
            if date_type.fromisoformat(date) > date_type.today():
                errors.append(S.ERR_DATE_FUTURE)
        except ValueError:
            errors.append(S.ERR_DATE_REQUIRED)

    try:
        quintals = Decimal(body.get('quintals', '0'))
        if quintals <= 0:
            errors.append(S.ERR_QUINTALS_POSITIVE)
    except InvalidOperation:
        errors.append(S.ERR_QUINTALS_POSITIVE)
        quintals = Decimal(0)

    note_id = body.get('note_id')
    record1 = body.get('record1')

    parsed = {
        'date': date,
        'parcel_id': int(body['parcel_id']),
        'crew_id': int(body['crew_id']),
        'product_id': int(body['product_id']),
        'note_id': int(note_id) if note_id else None,
        'record1': int(record1) if record1 else None,
        'quintals': quintals,
        'extra_note': body.get('extra_note', ''),
    }
    # record2 (Prot) is display-only for legacy data; only overwrite if
    # explicitly present in the submission (i.e., never from the current form).
    if 'record2' in body:
        record2 = body['record2']
        parsed['record2'] = int(record2) if record2 else None
    return row_id, parsed, errors


def _parse_percentages(body):
    """Extract species/tractor percentages from the POST body.

    Returns (sp_pcts, tr_pcts, error_list).
    """
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


def _compute_volume_m3(quintals: Decimal, sp_pcts: dict[int, int]) -> Decimal:
    """Compute the materialized harvest volume in m³.

    `volume_m3 = SUM_over_species(quintals × pct/100 / species.density)`.
    Captured at write time using current `Species.density` values; later
    density edits do not retroactively update stored volumes.
    """
    if not sp_pcts:
        return Decimal('0')
    densities = dict(
        Species.objects.filter(id__in=sp_pcts.keys()).values_list('id', 'density'),
    )
    total = Decimal('0')
    hundred = Decimal(100)
    for sid, pct in sp_pcts.items():
        density = densities.get(sid)
        if density and density > 0:
            total += quintals * Decimal(pct) / hundred / density
    return total.quantize(Decimal('0.001'))


def _conflict_response(row_id, request):
    """Build the update-flow conflict response (used on pre-check and on
    races inside the transaction)."""
    op = Harvest.objects.select_related(
        'parcel__region', 'crew', 'note', 'product',
    ).get(id=row_id)
    return JsonResponse({
        S.STATUS: S.STATUS_CONFLICT, S.MESSAGE: S.ERROR_CONFLICT,
        S.DATA_ID: 'prelievi', S.ROW_ID: row_id,
        S.RECORD: build_harvest_record(op),
        S.HTML: _render_form(row_id, request),
    }, status=400)


def _check_update_conflict(row_id, body, request):
    """Pre-flight version check.  Returns a conflict JsonResponse if the
    submitted version is stale, a 404 if the row is gone, or None if OK.
    The authoritative check inside `transaction.atomic()` still runs in
    `_update_op` to handle races with concurrent writers."""
    try:
        actual_version = Harvest.objects.values_list(S.VERSION, flat=True).get(id=row_id)
    except Harvest.DoesNotExist:
        return JsonResponse({S.MESSAGE: S.ERR_NOT_FOUND}, status=404)
    if actual_version == int(body.get(S.VERSION, 0)):
        return None
    return _conflict_response(row_id, request)


def _update_op(row_id, parsed, body, request):
    """Update an existing Harvest under row lock.  Returns the updated op,
    or a conflict JsonResponse if a concurrent writer bumped the version
    since `_check_update_conflict` passed."""
    version = int(body.get(S.VERSION, 0))
    op = Harvest.objects.select_for_update().get(id=row_id)
    if op.version != version:
        return _conflict_response(row_id, request)

    for field, value in parsed.items():
        setattr(op, field, value)
    op.version += 1
    op.save()
    return op


def _write_junctions(op, sp_pcts, tr_pcts):
    """Replace species and tractor junction records for a harvest."""
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
    return JsonResponse({
        S.STATUS: S.STATUS_VALIDATION_ERROR,
        S.MESSAGE: ' '.join(errors),
        S.HTML: _render_form(row_id, request, vals),
    }, status=400)
