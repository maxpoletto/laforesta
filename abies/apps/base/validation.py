"""Small, shared JSON-edge guards; domain validation remains with each caller.

No coercion of text objects/arrays, booleans to IDs, or floats to integers.
Known text nulls mean blank; unknown fields remain forward-compatible. Normal
invalid textual numbers still reach the caller's existing field-specific errors.
"""

import json
import math
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache

from django.apps import apps
from django.db.models import DecimalField
from django.http import JsonResponse

from apps.base.numparse import int_or_none, parse_decimal, to_decimal
from config import constants as C
from config import strings as S


_TEXT_FIELDS = {
    C.FIELD_NAME, C.FIELD_DESCRIPTION, C.FIELD_NOTE, C.FIELD_NOTES, C.FIELD_EXTRA_NOTE,
    C.FIELD_COMPRESA,
    C.FIELD_DATE, C.FIELD_DEFAULT_DATE, C.FIELD_OPERATOR, C.FIELD_TEXT,
    C.FIELD_MANUFACTURER, C.FIELD_MODEL, C.FIELD_COMMON_NAME, C.FIELD_LATIN_NAME,
    C.FIELD_EMAIL, C.FIELD_USERNAME, C.FIELD_FIRST_NAME, C.FIELD_LAST_NAME,
    C.FIELD_ROLE, C.FIELD_LOGIN_METHOD, C.FIELD_PASSWORD, C.FIELD_PASSWORD1,
    C.FIELD_PASSWORD2, C.FIELD_CURRENT_PASSWORD, C.FIELD_BOTANICAL_TYPE,
    C.FIELD_LANDING_PAGE, C.FIELD_DEFAULT_LANDING_PAGE, C.FIELD_DIAMETER_CLASS_MODE,
}
_INTEGER_FIELDS = {
    C.ROW_ID, C.VERSION, C.FIELD_ID, C.FIELD_PARCEL_ID, C.FIELD_REGION_ID,
    C.FIELD_SPECIES_ID, C.FIELD_CREW_ID, C.FIELD_PRODUCT_ID, C.FIELD_SAMPLE_AREA_ID,
    C.FIELD_SAMPLE_GRID_ID, C.FIELD_SURVEY_ID, C.FIELD_HARVEST_PLAN_ID,
    C.FIELD_HARVEST_PLAN_ITEM_ID, C.FIELD_D_CM, C.FIELD_SHOOT, C.FIELD_L10_MM,
    C.FIELD_ACC_M, C.FIELD_SORT_ORDER, C.FIELD_YEAR, C.FIELD_YEAR_START,
    C.FIELD_YEAR_END, C.FIELD_YEAR_PLANNED, C.FIELD_MIN_N, C.FIELD_RECORD1, C.FIELD_RECORD2,
    C.FIELD_R_M, C.FIELD_ALTITUDE, C.FIELD_ALTITUDE_M, C.FIELD_PRESERVED_NUMBER,
}
_BOOLEAN_FIELDS = {
    C.FIELD_ACTIVE, C.FIELD_IS_ACTIVE, C.FIELD_MINOR, C.FIELD_PRESERVED,
    C.FIELD_HIGHFOREST, C.FIELD_COPPICE, C.FIELD_STANDARD, C.FIELD_H_MEASURED,
    C.FIELD_OPEN, C.FIELD_DAMAGED, C.FIELD_UNHEALTHY, C.FIELD_PSR,
    C.FIELD_WARNINGS_CONFIRMED, C.FIELD_USE_FOR_HEIGHT_PLOTS,
}
# Reuse the actual storage precision rather than duplicating numeric limits.
_DECIMAL_FIELDS = {
    C.FIELD_H_M: ('base.TreeSample', 'h_m'),
    C.FIELD_PRESSLER_COEFF: ('base.TreeSample', 'pressler_coeff'),
    C.FIELD_DENSITY: ('base.Species', 'density'),
    C.FIELD_PRESSLER_DEFAULT: ('base.Species', 'pressler_default'),
    C.FIELD_MASS_Q: ('prelievi.Harvest', 'mass_q'),
    C.FIELD_HOURS: ('mannesi.WorkHour', 'hours'),
    C.FIELD_INTERVENTION_AREA_HA: ('base.HarvestPlanItem', 'intervention_area_ha'),
    C.FIELD_VOLUME_PLANNED_M3: ('base.HarvestPlanItem', 'volume_planned_m3'),
}


def validation_error(errors: list, html: str = '', *, message: str | None = None) -> JsonResponse:
    """The shared 400 envelope; no replacement HTML is needed to retry a form."""
    return JsonResponse({
        C.STATUS: C.STATUS_VALIDATION_ERROR,
        C.MESSAGE: ' '.join(errors) if message is None else message,
        C.FIELD_ERRORS: errors,
        C.HTML: html,
    }, status=400)


class InputValidationError(Exception):
    """An explicitly detected invalid input/derived value, not a server bug."""


def _field_decimal(value, field):
    if value is None or not value.is_finite():
        return None
    quantum = Decimal(1).scaleb(-field.decimal_places)
    try:
        return value.quantize(quantum, context=Context(prec=field.max_digits), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def decimal_field_value(value, model_label, field_name):
    """Quantize to a model DecimalField's precision, or return None safely."""
    return _field_decimal(value, apps.get_model(model_label)._meta.get_field(field_name))


@lru_cache(maxsize=64)
def _decimal_fields(model):
    return tuple(field for field in model._meta.concrete_fields if isinstance(field, DecimalField))


def validate_decimal_storage(instance, update_fields=None):
    """Reject unrepresentable stored/derived values before SQL is executed.

    This is not full_clean(), a scientific range policy, or a change to normal
    rounding. It checks only representability, using the field's own precision.
    """
    for field in _decimal_fields(type(instance)):
        if update_fields is not None and field.name not in update_fields:
            continue
        raw = getattr(instance, field.attname)
        if raw is not None and _field_decimal(to_decimal(raw, '.'), field) is None:
            raise InputValidationError(S.ERR_JSON_FIELD_INVALID)


def json_fields_valid(body):
    """Validate shared wire types/storage bounds before views dereference them."""
    for key in _TEXT_FIELDS & body.keys():
        if body[key] is None:
            body[key] = ''
        elif not isinstance(body[key], str):
            return False
    # Area numbers are free text, while tree numbers are integers. Both APIs
    # accept an integer rendered as text; neither may truncate a JSON float.
    for key in {C.FIELD_NUMBER, C.FIELD_PARTICELLA} & body.keys():
        value = body[key]
        if isinstance(value, bool) or (value is not None and not isinstance(value, (str, int))):
            return False
        if isinstance(value, int):
            body[key] = str(value)
    for key in _BOOLEAN_FIELDS & body.keys():
        value = body[key]
        if value is not None and value != '' and C.parse_bool(value) is None:
            return False
    for key in _INTEGER_FIELDS & body.keys():
        value = body[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (str, int)) or
            (isinstance(value, int) and int_or_none(value) is None)
        ):
            return False
    for key in (set(_DECIMAL_FIELDS) | {C.FIELD_LAT, C.FIELD_LON}) & body.keys():
        raw = body[key]
        if raw is None or raw == '' or raw == 'null':
            continue
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            return False
        value = parse_decimal(raw)
        if key in (C.FIELD_LAT, C.FIELD_LON):
            limit = 90 if key == C.FIELD_LAT else 180
            if value is None or abs(value) > limit:
                return False
        elif value is not None and decimal_field_value(value, *_DECIMAL_FIELDS[key]) is None:
            return False
    return True


def json_object_list(value):
    """Read a JSON/form array of objects, or None when structurally invalid."""
    if isinstance(value, str):
        try:
            value = _decode_json(value)
        except (ValueError, RecursionError):
            return None
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None
    return value


def _invalid_json_constant(_value):
    raise ValueError('Non-finite JSON number')


def _json_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError('Non-finite JSON number')
    return value


def _decode_json(raw):
    try:
        return json.loads(raw, parse_constant=_invalid_json_constant, parse_float=_json_float)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid JSON encoding or depth') from exc


def read_json_body(request):
    """Decode once per request; adapters keep their own protocol/error shape."""
    if not hasattr(request, '_abies_json_body'):
        request._abies_json_body = _decode_json(request.body)
    return request._abies_json_body
