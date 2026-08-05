"""Import cores for staged Ipso uploads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from decimal import Decimal

from apps.base.models import (
    Parcel, Sample, SampleArea, Species, Survey, TreeSample,
)
from apps.base.numparse import to_decimal
from apps.base.preserved_trees import current_preserved_number_keys
from apps.campionamenti.csv_trees import parsed_tree_row
from apps.campionamenti.tree_validation import normalize_sample_tree_values
from config import strings as S
from config.constants import (
    FIELD_ACC_M, FIELD_AREA, FIELD_COPPICE, FIELD_DATE,
    FIELD_D_CM, FIELD_H_M, FIELD_H_MEASURED,
    FIELD_LAT, FIELD_LON, FIELD_L10_MM, FIELD_NOTE,
    FIELD_NUMBER, FIELD_OPERATOR, FIELD_PARCEL, FIELD_PARCEL_ID,
    FIELD_PRESERVED, FIELD_PRESERVED_NUMBER, FIELD_PRESSLER_COEFF,
    FIELD_SAMPLE_AREA_ID, FIELD_SHOOT, FIELD_SPECIES, FIELD_SPECIES_ID, FIELD_STANDARD,
    PRESSLER_DEFAULT, RECORDS, SESSION,
)


@dataclass(frozen=True)
class TreeMeasurements:
    date: date_type
    d_cm: int
    h_m: Decimal


_SAMPLE_PARSE_NUMBER_INVALID = 'sample_number_invalid'
_SAMPLE_PARSE_NUMBER_POSITIVE = 'sample_number_positive'


def _int_ids(records: list, field: str) -> set[int]:
    ids = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        value = record.get(field)
        if type(value) is int:
            ids.add(value)
    return ids


def sample_import_rows(payload: dict, survey: Survey) -> tuple[list[dict], list[str]]:
    if survey.sample_grid_id is None:
        return [], [S.ERR_SURVEY_STRUCTURED_REQUIRED]
    records = _payload_records(payload)
    if records is None:
        return [], [S.IPSO_ERR_IMPORT_RECORDS_ARRAY]

    species_ids = _int_ids(records, FIELD_SPECIES_ID)
    area_ids = _int_ids(records, FIELD_SAMPLE_AREA_ID)
    species = {sp.id: sp for sp in Species.objects.filter(id__in=species_ids)}
    areas = {
        area.id: area
        for area in (SampleArea.objects
                     .filter(id__in=area_ids)
                     .select_related('parcel__region', 'parcel__eclass'))
    }
    existing_samples = {
        sample.sample_area_id: sample
        for sample in Sample.objects.filter(survey=survey, sample_area_id__in=area_ids)
    }
    seen_number_shoots = set(
        TreeSample.objects
        .filter(sample__survey=survey, sample__sample_area_id__in=area_ids)
        .values_list('sample__sample_area_id', FIELD_NUMBER, FIELD_SHOOT)
    )
    session = payload.get(SESSION, {}) if isinstance(payload, dict) else {}
    session_operator = (
        (session.get(FIELD_OPERATOR) or '').strip()
        if isinstance(session, dict) else ''
    )

    rows = []
    errors = []
    csv_date_by_area = {}
    for i, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(S.IPSO_ERR_IMPORT_RECORD_INVALID.format(i))
            continue
        area = areas.get(record.get(FIELD_SAMPLE_AREA_ID))
        if area is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_AREA_NOT_FOUND.format(i))
            continue
        if area.sample_grid_id != survey.sample_grid_id:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_AREA_OUT_OF_SURVEY.format(i))
            continue
        if area.parcel_id != record.get(FIELD_PARCEL_ID):
            errors.append(S.IPSO_ERR_IMPORT_RECORD_AREA_PARCEL_MISMATCH.format(i))
            continue
        sp = species.get(record.get(FIELD_SPECIES_ID))
        if sp is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SPECIES_NOT_FOUND.format(i))
            continue
        if record.get(FIELD_NUMBER) is None:
            errors.append(S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(i))
            continue
        parsed = _sample_record_values(record, area, sp, session_operator)
        if parsed is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SAMPLE_FIELDS_INVALID.format(i))
            continue
        if parsed == _SAMPLE_PARSE_NUMBER_INVALID:
            errors.append(S.IPSO_ERR_RECORD_NUMBER_INVALID.format(i))
            continue
        if parsed == _SAMPLE_PARSE_NUMBER_POSITIVE:
            errors.append(S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(i))
            continue

        existing_sample = existing_samples.get(area.id)
        row_date = parsed[FIELD_DATE]
        if existing_sample and existing_sample.date != row_date:
            errors.append(S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
                i, area.parcel.region.name, area.parcel.name, area.number,
                existing_sample.date.isoformat(),
            ))
            continue
        previous_date = csv_date_by_area.get(area.id)
        if previous_date is not None and previous_date != row_date:
            errors.append(S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
                i, area.parcel.region.name, area.parcel.name, area.number,
                previous_date.isoformat(),
            ))
            continue
        csv_date_by_area.setdefault(area.id, row_date)
        number_shoot_key = (area.id, parsed[FIELD_NUMBER], parsed[FIELD_SHOOT])
        if number_shoot_key in seen_number_shoots:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(i))
            continue
        seen_number_shoots.add(number_shoot_key)
        rows.append(parsed)
    return rows, errors


def free_survey_import_rows(payload: dict, survey: Survey) -> tuple[list[dict], list[str]]:
    if survey.sample_grid_id is not None:
        return [], [S.ERR_SURVEY_UNSTRUCTURED_REQUIRED]
    records = _payload_records(payload)
    if records is None:
        return [], [S.IPSO_ERR_IMPORT_RECORDS_ARRAY]

    session = payload.get(SESSION, {}) if isinstance(payload, dict) else {}
    session_operator = (
        (session.get(FIELD_OPERATOR) or '').strip()
        if isinstance(session, dict) else ''
    )
    species_ids = _int_ids(records, FIELD_SPECIES_ID)
    parcel_ids = _int_ids(records, FIELD_PARCEL_ID)
    species = {sp.id: sp for sp in Species.objects.filter(id__in=species_ids)}
    parcels = {
        parcel.id: parcel
        for parcel in (Parcel.objects
                       .filter(id__in=parcel_ids)
                       .select_related('region', 'eclass'))
    }
    seen_sample_numbers = set(
        TreeSample.objects
        .filter(sample__survey=survey)
        .values_list(FIELD_NUMBER, flat=True)
    )
    # Reserve every valid number explicitly supplied for an ordinary row
    # before filling blanks. Otherwise a blank between 3 and 4 would consume 4
    # during the single-pass import and make the later explicit 4 look like a
    # duplicate. Invalid values are deliberately left to the row validator.
    reserved_sample_numbers = {
        int(record[FIELD_NUMBER])
        for record in records
        if (
            isinstance(record, dict)
            and not bool(record.get(FIELD_PRESERVED))
            and _is_positive_int_value(record.get(FIELD_NUMBER))
        )
    }
    unavailable_sample_numbers = seen_sample_numbers | reserved_sample_numbers
    next_sample_number = (
        max(unavailable_sample_numbers) if unavailable_sample_numbers else 0
    ) + 1
    seen_preserved_numbers = current_preserved_number_keys(parcel_ids)

    rows = []
    errors = []
    for i, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(S.IPSO_ERR_IMPORT_RECORD_INVALID.format(i))
            continue
        parcel = parcels.get(record.get(FIELD_PARCEL_ID))
        if parcel is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_PARCEL_NOT_FOUND.format(i))
            continue
        sp = species.get(record.get(FIELD_SPECIES_ID))
        if sp is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SPECIES_NOT_FOUND.format(i))
            continue

        preserved = bool(record.get(FIELD_PRESERVED))
        number = record.get(FIELD_NUMBER)
        if number is not None:
            try:
                number = int(number)
            except (TypeError, ValueError):
                errors.append(S.IPSO_ERR_RECORD_NUMBER_INVALID.format(i))
                continue
            if number <= 0:
                errors.append(S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(i))
                continue
        if preserved and number is None:
            errors.append(S.IPSO_ERR_RECORD_NUMBER_REQUIRED.format(i))
            continue

        preserved_number = number if preserved else None
        if preserved_number is not None:
            preserved_key = (parcel.id, preserved_number)
            if preserved_key in seen_preserved_numbers:
                errors.append(S.IPSO_ERR_IMPORT_RECORD_PAI_NUMBER_DUPLICATE.format(i))
                continue
            seen_preserved_numbers.add(preserved_key)

        sample_number = None if preserved else number
        if sample_number is None:
            while next_sample_number in unavailable_sample_numbers:
                next_sample_number += 1
            sample_number = next_sample_number
            next_sample_number += 1
            unavailable_sample_numbers.add(sample_number)
        if sample_number in seen_sample_numbers:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(i))
            continue
        seen_sample_numbers.add(sample_number)

        parsed = _free_survey_record_values(
            record, parcel, sp, session_operator, sample_number,
            preserved_number,
        )
        if parsed is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_DH_DATE_INVALID.format(i))
            continue
        rows.append(parsed)
    return rows, errors


def _is_positive_int_value(value) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _payload_records(payload: dict) -> list | None:
    records = payload.get(RECORDS, []) if isinstance(payload, dict) else []
    return records if isinstance(records, list) else None


def record_measurements(record: dict) -> TreeMeasurements | None:
    try:
        row_date = date_type.fromisoformat(str(record.get(FIELD_DATE)))
        d_cm = int(record.get(FIELD_D_CM))
        h_m = to_decimal(record.get(FIELD_H_M), '.')
    except (TypeError, ValueError):
        return None
    if h_m is None or d_cm <= 0 or h_m <= 0:
        return None
    return TreeMeasurements(date=row_date, d_cm=d_cm, h_m=h_m)


def _free_survey_record_values(
        record: dict, parcel: Parcel, sp: Species, session_operator: str,
        sample_number: int, preserved_number: int | None,
) -> dict | None:
    measurements = record_measurements(record)
    if measurements is None:
        return None
    values = normalize_sample_tree_values(
        number=sample_number,
        d_cm=measurements.d_cm,
        h_m=measurements.h_m,
        shoot=0,
        l10_mm=0,
        pressler_coeff=PRESSLER_DEFAULT,
        h_measured=bool(record.get(FIELD_H_MEASURED)),
    )
    if values is None:
        return None
    row = parsed_tree_row(
        area=None, parcel=parcel, row_date=measurements.date, species=sp,
        coppice=parcel.eclass.coppice, preserved=False, number=values.number,
        shoot=values.shoot, standard=False,
        d_cm=values.d_cm, h_m=values.h_m,
        h_measured=values.h_measured,
        l10_mm=values.l10_mm, pressler_coeff=values.pressler_coeff,
        lat=record.get(FIELD_LAT), lon=record.get(FIELD_LON),
        acc_m=record.get(FIELD_ACC_M),
        operator=(record.get(FIELD_OPERATOR) or session_operator).strip(),
        note=(record.get(FIELD_NOTE) or '').strip(),
    )
    row[FIELD_PRESERVED] = preserved_number is not None
    row[FIELD_PRESERVED_NUMBER] = preserved_number
    return row


def _sample_record_values(
        record: dict, area: SampleArea, sp: Species, session_operator: str,
) -> dict | None:
    measurements = record_measurements(record)
    if measurements is None:
        return None
    try:
        number = int(record.get(FIELD_NUMBER))
    except (TypeError, ValueError):
        return _SAMPLE_PARSE_NUMBER_INVALID
    if number <= 0:
        return _SAMPLE_PARSE_NUMBER_POSITIVE
    try:
        shoot = int(record.get(FIELD_SHOOT, 0) or 0)
        l10_mm = int(record.get(FIELD_L10_MM, 0) or 0)
    except (TypeError, ValueError):
        return None
    pressler_coeff = to_decimal(record.get(FIELD_PRESSLER_COEFF), '.')
    if pressler_coeff is None:
        pressler_coeff = PRESSLER_DEFAULT
    values = normalize_sample_tree_values(
        number=number,
        d_cm=measurements.d_cm,
        h_m=measurements.h_m,
        shoot=shoot,
        l10_mm=l10_mm,
        pressler_coeff=pressler_coeff,
        h_measured=bool(record.get(FIELD_H_MEASURED)),
    )
    if values is None:
        return None
    coppice_value = record.get(FIELD_COPPICE)
    coppice = area.parcel.eclass.coppice if coppice_value is None else bool(coppice_value)
    return parsed_tree_row(
        area=area, row_date=measurements.date, species=sp, coppice=coppice,
        preserved=bool(record.get(FIELD_PRESERVED)), number=values.number,
        shoot=values.shoot, standard=bool(record.get(FIELD_STANDARD)),
        d_cm=values.d_cm, h_m=values.h_m,
        h_measured=values.h_measured,
        l10_mm=values.l10_mm, pressler_coeff=values.pressler_coeff,
        lat=record.get(FIELD_LAT),
        lon=record.get(FIELD_LON), acc_m=record.get(FIELD_ACC_M),
        operator=(record.get(FIELD_OPERATOR) or session_operator).strip(),
        note=(record.get(FIELD_NOTE) or '').strip(),
    )
