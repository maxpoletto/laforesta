"""The parcel properties form shared by Bosco and Cartografia."""
from django.template.loader import render_to_string

from apps.base.models import Eclass, Parcel
from apps.base.numparse import int_or_none, parse_decimal
from apps.base.validation import decimal_field_value
from config import strings as S


FIELD_ECLASS_ID = 'eclass_id'
FIELD_AREA_HA = 'area_ha'
FIELD_AVE_AGE = 'ave_age'
FIELD_LOCATION_NAME = 'location_name'
FIELD_ALTITUDE_MIN_M = 'altitude_min_m'
FIELD_ALTITUDE_MAX_M = 'altitude_max_m'
FIELD_ASPECT = 'aspect'
FIELD_GRADE_PCT = 'grade_pct'
FIELD_DESC_VEG = 'desc_veg'
FIELD_DESC_GEO = 'desc_geo'
FIELD_CUTTING_PLAN = 'cutting_plan'
FIELD_HARVEST_MECHANISM = 'harvest_mechanism'
FIELD_INTERVENTION_INTERVAL = 'intervention_interval'
FIELD_STANDARDS_PER_HA = 'standards_per_ha'


PARCEL_METADATA_TEXT_FIELDS = {
    FIELD_LOCATION_NAME: (S.COL_LOCATION, 200),
    FIELD_ASPECT: (S.COL_ASPECT, 20),
    FIELD_DESC_VEG: (S.LABEL_BOSCO_VEG_DESC, None),
    FIELD_DESC_GEO: (S.LABEL_BOSCO_GEO_DESC, None),
    FIELD_CUTTING_PLAN: (S.LABEL_BOSCO_CUTTING_PLAN, None),
    FIELD_HARVEST_MECHANISM: (S.LABEL_BOSCO_HARVEST_MECHANISM, 200),
}

def render_parcel_metadata_form(request, parcel, values=None, *, promotion=False):
    form_values = _parcel_metadata_form_values(parcel, values)
    eclasses = list(Eclass.objects.order_by('name') if promotion else Eclass.objects.order_by('coppice', 'name'))
    selected_eclass_id = int_or_none(values.get(FIELD_ECLASS_ID)) if values else None
    if selected_eclass_id is None:
        selected_eclass_id = parcel.eclass_id or (eclasses[0].pk if eclasses else None)
    selected_eclass = next(
        (eclass for eclass in eclasses if eclass.id == selected_eclass_id),
        None,
    )
    return render_to_string('bosco/_parcel_metadata_form.html', {
        'parcel': parcel,
        'promotion': promotion,
        'destinations': Parcel.objects.filter(region=parcel.region).order_by('name') if promotion else (),
        'version': parcel.version,
        'values': form_values,
        'eclasses': eclasses,
        'selected_eclass_id': selected_eclass_id,
        'is_coppice': selected_eclass is not None and selected_eclass.coppice,
    }, request=request)


def _parcel_metadata_form_values(parcel, values: dict | None = None):
    fields = (
        (FIELD_AREA_HA, parcel.area_ha),
        (FIELD_AVE_AGE, parcel.ave_age),
        (FIELD_LOCATION_NAME, parcel.location_name),
        (FIELD_ALTITUDE_MIN_M, parcel.altitude_min_m),
        (FIELD_ALTITUDE_MAX_M, parcel.altitude_max_m),
        (FIELD_ASPECT, parcel.aspect),
        (FIELD_GRADE_PCT, parcel.grade_pct),
        (FIELD_DESC_VEG, parcel.desc_veg),
        (FIELD_DESC_GEO, parcel.desc_geo),
        (FIELD_CUTTING_PLAN, parcel.cutting_plan),
        (FIELD_HARVEST_MECHANISM, parcel.harvest_mechanism),
        (FIELD_INTERVENTION_INTERVAL, parcel.intervention_interval),
        (FIELD_STANDARDS_PER_HA, parcel.standards_per_ha),
    )
    defaults = {key: default if default is not None else '' for key, default in fields}
    return {**defaults, **(values or {})}


def parse_parcel_metadata(body: dict):
    errors = []
    eclass_id = int_or_none(body.get(FIELD_ECLASS_ID))
    if eclass_id is None:
        eclass = None
        errors.append(S.ERR_BOSCO_GOVERNANCE_REQUIRED)
    else:
        eclass = Eclass.objects.filter(id=eclass_id).first()
        if eclass is None:
            errors.append(S.ERR_BOSCO_GOVERNANCE_INVALID)

    raw_area = body.get(FIELD_AREA_HA)
    area_ha = decimal_field_value(parse_decimal(raw_area), 'base.Parcel', FIELD_AREA_HA)
    if raw_area not in (None, '') and (area_ha is None or area_ha <= 0):
        errors.append(S.ERR_BOSCO_AREA_INVALID)

    values = {
        'eclass': eclass,
        FIELD_AREA_HA: area_ha,
        FIELD_AVE_AGE: _optional_int(body, FIELD_AVE_AGE, S.LABEL_BOSCO_AVE_AGE, errors),
        FIELD_LOCATION_NAME: _text_value(body, FIELD_LOCATION_NAME, errors),
        FIELD_ALTITUDE_MIN_M: _optional_int(
            body, FIELD_ALTITUDE_MIN_M, S.LABEL_BOSCO_ALTITUDE_MIN, errors,
        ),
        FIELD_ALTITUDE_MAX_M: _optional_int(
            body, FIELD_ALTITUDE_MAX_M, S.LABEL_BOSCO_ALTITUDE_MAX, errors,
        ),
        FIELD_ASPECT: _text_value(body, FIELD_ASPECT, errors),
        FIELD_GRADE_PCT: _optional_int(body, FIELD_GRADE_PCT, S.LABEL_BOSCO_GRADE, errors),
        FIELD_DESC_VEG: _text_value(body, FIELD_DESC_VEG, errors),
        FIELD_DESC_GEO: _text_value(body, FIELD_DESC_GEO, errors),
        FIELD_CUTTING_PLAN: _text_value(body, FIELD_CUTTING_PLAN, errors),
        FIELD_HARVEST_MECHANISM: _text_value(
            body, FIELD_HARVEST_MECHANISM, errors),
        FIELD_INTERVENTION_INTERVAL: _optional_int(
            body, FIELD_INTERVENTION_INTERVAL, S.COL_INTERVENTION_INTERVAL, errors,
        ),
        FIELD_STANDARDS_PER_HA: _optional_int(
            body, FIELD_STANDARDS_PER_HA, S.COL_STANDARDS_PER_HA, errors,
        ),
    }
    if eclass is not None and eclass.coppice:
        for field, label in (
            (FIELD_INTERVENTION_INTERVAL, S.COL_INTERVENTION_INTERVAL),
            (FIELD_STANDARDS_PER_HA, S.COL_STANDARDS_PER_HA),
        ):
            if values[field] is None:
                errors.append(S.ERR_BOSCO_COPPICE_METADATA_REQUIRED.format(label))
    else:
        values[FIELD_INTERVENTION_INTERVAL] = None
        values[FIELD_STANDARDS_PER_HA] = None
    alt_min = values[FIELD_ALTITUDE_MIN_M]
    alt_max = values[FIELD_ALTITUDE_MAX_M]
    if alt_min is not None and alt_max is not None and alt_min > alt_max:
        errors.append(S.ERR_BOSCO_ALTITUDE_RANGE)
    return values, errors


def _optional_int(body: dict, key: str, label: str, errors: list[str]):
    raw = body.get(key)
    if raw in (None, ''):
        return None
    value = int_or_none(raw)
    if value is None:
        errors.append(S.ERR_BOSCO_INTEGER_REQUIRED.format(label))
    return value


def _text_value(body: dict, key: str, errors: list[str]):
    label, max_len = PARCEL_METADATA_TEXT_FIELDS[key]
    value = str(body.get(key) or '').strip()
    if max_len is not None and len(value) > max_len:
        errors.append(S.ERR_BOSCO_TEXT_TOO_LONG.format(label))
    return value
