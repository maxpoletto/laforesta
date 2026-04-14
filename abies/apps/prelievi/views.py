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
from apps.base.models import Crew, Note, Optype, Parcel, Region, Species, Tractor
from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor
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
    return JsonResponse({'html': _render_form(op_id, request)})


# ---------------------------------------------------------------------------
# Save endpoint
# ---------------------------------------------------------------------------

@login_required
@require_writer
@require_POST
def save_view(request):
    """Create or update a harvest operation."""
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
        dup = HarvestOp.objects.filter(record1=record1)
        if row_id:
            dup = dup.exclude(id=row_id)
        if dup.exists():
            return _validation_error([S.ERR_VDP_DUPLICATE.format(record1)], row_id, request, body)

    # Species / tractor percentages
    sp_pcts, tr_pcts, pct_errors = _parse_percentages(body)
    if pct_errors:
        return _validation_error(pct_errors, row_id, request, body)

    with transaction.atomic():
        if row_id:
            op = _update_op(row_id, parsed, body, request)
            if isinstance(op, JsonResponse):
                return op  # race conflict
        else:
            op = HarvestOp.objects.create(**parsed)

        _write_junctions(op, sp_pcts, tr_pcts)
        mark_stale('prelievi', 'parcel_year_production', 'audit')

    op = HarvestOp.objects.select_related('parcel__region', 'crew', 'note', 'optype').get(id=op.id)
    record = build_harvest_record(op)
    response_data = {'data_id': 'prelievi', 'row_id': op.id, 'record': record}

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
    """Delete a harvest operation (with version check)."""
    body = json.loads(request.body)
    row_id = int(body['row_id'])
    version = int(body.get('version', 0))

    try:
        op = HarvestOp.objects.select_related('parcel__region', 'crew', 'note', 'optype').get(id=row_id)
    except HarvestOp.DoesNotExist:
        return JsonResponse({'message': S.ERR_NOT_FOUND}, status=404)

    if op.version != version:
        record = build_harvest_record(op)
        return JsonResponse({
            'status': 'conflict', 'message': S.ERROR_CONFLICT,
            'data_id': 'prelievi', 'row_id': row_id, 'record': record,
        }, status=400)

    with transaction.atomic():
        op.delete()
        mark_stale('prelievi', 'parcel_year_production', 'audit')

    response_data = {'data_id': 'prelievi', 'row_id': row_id}

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
        op = HarvestOp.objects.select_related(
            'parcel__region', 'crew', 'note',
        ).get(id=op_id)
        sp_pcts = dict(
            HarvestSpecies.objects.filter(harvest_op=op).values_list('species_id', 'percent'),
        )
        tr_pcts = dict(
            HarvestTractor.objects.filter(harvest_op=op).values_list('tractor_id', 'percent'),
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
        'optypes': Optype.objects.order_by('name'),
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
    row_id = body.get('row_id')
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
        'optype_id': int(body['optype_id']),
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


def _conflict_response(row_id, request):
    """Build the update-flow conflict response (used on pre-check and on
    races inside the transaction)."""
    op = HarvestOp.objects.select_related(
        'parcel__region', 'crew', 'note', 'optype',
    ).get(id=row_id)
    return JsonResponse({
        'status': 'conflict', 'message': S.ERROR_CONFLICT,
        'data_id': 'prelievi', 'row_id': row_id,
        'record': build_harvest_record(op),
        'html': _render_form(row_id, request),
    }, status=400)


def _check_update_conflict(row_id, body, request):
    """Pre-flight version check.  Returns a conflict JsonResponse if the
    submitted version is stale, a 404 if the row is gone, or None if OK.
    The authoritative check inside `transaction.atomic()` still runs in
    `_update_op` to handle races with concurrent writers."""
    try:
        actual_version = HarvestOp.objects.values_list('version', flat=True).get(id=row_id)
    except HarvestOp.DoesNotExist:
        return JsonResponse({'message': S.ERR_NOT_FOUND}, status=404)
    if actual_version == int(body.get('version', 0)):
        return None
    return _conflict_response(row_id, request)


def _update_op(row_id, parsed, body, request):
    """Update an existing HarvestOp under row lock.  Returns the updated op,
    or a conflict JsonResponse if a concurrent writer bumped the version
    since `_check_update_conflict` passed."""
    version = int(body.get('version', 0))
    op = HarvestOp.objects.select_for_update().get(id=row_id)
    if op.version != version:
        return _conflict_response(row_id, request)

    for field, value in parsed.items():
        setattr(op, field, value)
    op.version += 1
    op.save()
    return op


def _write_junctions(op, sp_pcts, tr_pcts):
    """Replace species and tractor junction records for a harvest op."""
    HarvestSpecies.objects.filter(harvest_op=op).delete()
    HarvestTractor.objects.filter(harvest_op=op).delete()
    HarvestSpecies.objects.bulk_create([
        HarvestSpecies(harvest_op=op, species_id=sid, percent=pct)
        for sid, pct in sp_pcts.items()
    ])
    HarvestTractor.objects.bulk_create([
        HarvestTractor(harvest_op=op, tractor_id=tid, percent=pct)
        for tid, pct in tr_pcts.items()
    ])


def _validation_error(errors, row_id, request, vals=None):
    return JsonResponse({
        'status': 'validation_error',
        'message': ' '.join(errors),
        'html': _render_form(row_id, request, vals),
    }, status=400)
