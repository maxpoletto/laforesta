"""Squadre API views: personnel, work hours, credits, and report metadata."""

from datetime import date as date_type

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import mark_stale, prelievi_species_cols
from apps.base.models import Crew, Product
from apps.base.numparse import int_or_none, parse_decimal
from apps.base.responses import (
    conflict_response, parse_json_body, row_delete,
    save_model_response, submitted_version, success_response, validation_error,
)
from apps.mannesi.models import ProductionCredit, WorkHour
from config import strings as S
from config.constants import (
    COLUMNS, FIELD_ACTIVE, FIELD_CREW_ID, FIELD_DATE, FIELD_HOURS, FIELD_MASS_Q,
    FIELD_NAME, FIELD_NOTE, FIELD_NOTES, HTML, ROWS, ROW_ID, STATUS,
    STATUS_NOT_FOUND, VERSION, is_truthy,
)

DATA_ID_CREWS = 'crews'
DATA_ID_HOURS = 'squadre_hours'
DATA_ID_CREDITS = 'squadre_credits'

CREW_COLS = [ROW_ID, S.LABEL_NAME, S.LABEL_NOTES, S.COL_ACTIVE]
HOURS_COLS = [ROW_ID, VERSION, S.COL_DATE, S.COL_CREW, S.COL_HOURS, S.COL_NOTE]
CREDITS_COLS = [ROW_ID, VERSION, S.COL_DATE, S.COL_CREW, S.COL_CREDITS_Q, S.COL_NOTE]


def _crew_row(c):
    return [c.id, c.name, c.notes, c.active]


@login_required
def crews_data(request):
    return JsonResponse({COLUMNS: CREW_COLS, ROWS: [
        _crew_row(obj) for obj in Crew.objects.order_by('name')
    ]})


@login_required
@require_writer
def crews_form(request, obj_id=None):
    obj = Crew.objects.filter(id=obj_id).first() if obj_id else None
    html = render_to_string(
        'squadre/_crew_form.html', {'obj': obj}, request=request,
    )
    return JsonResponse({HTML: html})


@login_required
@require_writer
@require_POST
def crews_save(request):
    body, error = parse_json_body(request)
    if error:
        return error
    parsed = {
        FIELD_NAME: body.get(FIELD_NAME, '').strip(),
        FIELD_NOTES: body.get(FIELD_NOTES, ''),
        FIELD_ACTIVE: is_truthy(body.get(FIELD_ACTIVE)),
    }
    if not parsed[FIELD_NAME]:
        return validation_error([S.ERR_NAME_REQUIRED])
    return save_model_response(
        request, body, model=Crew, data_id=DATA_ID_CREWS, values=parsed,
        row_fn=_crew_row, stale=('prelievi', 'audit'),
    )


@login_required
def meta_view(request):
    """Small metadata bundle used by the report generator."""
    _, species_names, _, _ = prelievi_species_cols()
    return JsonResponse({
        'products': list(Product.objects.order_by('name').values_list('name', flat=True)),
        'species': species_names,
    })


@login_required
def hours_data(request):
    return JsonResponse({COLUMNS: HOURS_COLS, ROWS: [
        _hours_row(obj) for obj in _hours_queryset()
    ]})


@login_required
def credits_data(request):
    return JsonResponse({COLUMNS: CREDITS_COLS, ROWS: [
        _credit_row(obj) for obj in _credits_queryset()
    ]})


@login_required
@require_writer
def hours_form(request, obj_id=None):
    obj = WorkHour.objects.filter(id=obj_id).first() if obj_id else None
    return _form('hours', obj, request)


@login_required
@require_writer
def credits_form(request, obj_id=None):
    obj = ProductionCredit.objects.filter(id=obj_id).first() if obj_id else None
    return _form('credits', obj, request)


@login_required
@require_writer
@require_POST
def hours_save(request):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id, error = _entry_row_id(body)
    if error:
        return error
    parsed, errors = _parse_entry_body(
        body, value_field=FIELD_HOURS, value_error=S.ERR_HOURS_POSITIVE,
    )
    if errors:
        return validation_error(errors)
    return save_model_response(
        request, body, model=WorkHour, data_id=DATA_ID_HOURS,
        row_id=row_id, values={
            'date': parsed[FIELD_DATE],
            'crew_id': parsed[FIELD_CREW_ID],
            'hours': parsed[FIELD_HOURS],
            'note': parsed[FIELD_NOTE],
        },
        row_fn=_hours_row, stale=('audit',),
    )


@login_required
@require_writer
@require_POST
def credits_save(request):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id, error = _entry_row_id(body)
    if error:
        return error
    parsed, errors = _parse_entry_body(
        body, value_field=FIELD_MASS_Q, value_error=S.ERR_CREDITS_POSITIVE,
    )
    if errors:
        return validation_error(errors)
    return save_model_response(
        request, body, model=ProductionCredit, data_id=DATA_ID_CREDITS,
        row_id=row_id,
        values={
            'date': parsed[FIELD_DATE],
            'crew_id': parsed[FIELD_CREW_ID],
            'mass_q': parsed[FIELD_MASS_Q],
            'note': parsed[FIELD_NOTE],
        },
        row_fn=_credit_row, stale=('audit',),
    )


@login_required
@require_writer
@require_POST
def hours_delete(request):
    return _delete_entry(request, WorkHour, DATA_ID_HOURS, _hours_row)


@login_required
@require_writer
@require_POST
def credits_delete(request):
    return _delete_entry(request, ProductionCredit, DATA_ID_CREDITS, _credit_row)


def _entry_row_id(body):
    raw = body.get(ROW_ID)
    if raw in (None, ''):
        return None, None
    row_id = int_or_none(raw)
    if row_id is None:
        return None, JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    return row_id, None


def _delete_entry(request, model, data_id, row_fn):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id = int_or_none(body.get(ROW_ID))
    if row_id is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    obj = model.objects.select_related('crew').filter(id=row_id).first()
    if obj is None:
        return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    if obj.version != submitted_version(body):
        return conflict_response(data_id=data_id, row_id=obj.id, record=row_fn(obj))
    with transaction.atomic():
        obj.delete()
        mark_stale('audit')
    return success_response(
        request, body, data_id=data_id, row_id=row_id,
        deletes=[row_delete(data_id, row_id)],
    )


def _hours_queryset():
    return WorkHour.objects.select_related('crew').order_by('-date', 'id')


def _credits_queryset():
    return ProductionCredit.objects.select_related('crew').order_by('-date', 'id')


def _hours_row(obj):
    return [
        obj.id, obj.version, obj.date.isoformat(), obj.crew.name,
        float(obj.hours), obj.note,
    ]


def _credit_row(obj):
    return [
        obj.id, obj.version, obj.date.isoformat(), obj.crew.name,
        float(obj.mass_q), obj.note,
    ]


def _form(kind, obj, request):
    html = render_to_string(
        'squadre/_entry_form.html',
        {
            'kind': kind,
            'obj': obj,
            'crews': Crew.objects.filter(active=True).order_by('name'),
        },
        request=request,
    )
    return JsonResponse({HTML: html})


def _parse_entry_body(body, *, value_field: str, value_error: str):
    errors = []
    try:
        date = date_type.fromisoformat((body.get(FIELD_DATE) or '').strip())
    except ValueError:
        date = None
        errors.append(S.ERR_DATE_REQUIRED)

    crew_id = int_or_none(body.get(FIELD_CREW_ID))
    if crew_id is None or not Crew.objects.filter(id=crew_id, active=True).exists():
        errors.append(S.ERR_CREW_REQUIRED)

    value = parse_decimal(body.get(value_field))
    if value is None or value <= 0:
        errors.append(value_error)

    return {
        FIELD_DATE: date,
        FIELD_CREW_ID: crew_id,
        value_field: value,
        FIELD_NOTE: body.get(FIELD_NOTE, ''),
    }, errors
