"""Import cores for staged Ipso uploads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from decimal import Decimal, ROUND_HALF_UP

from apps.base.models import (
    Parcel, Sample, SampleArea, Species, Survey, TreePreserved,
    TreeSample, next_sequence_number,
)
from apps.base.numparse import to_decimal
from apps.campionamenti import csv_preserved
from apps.campionamenti.csv_trees import parsed_tree_row
from config import strings as S
from config.constants import (
    FIELD_ACC_M, FIELD_AREA, FIELD_COPPICE, FIELD_DATE,
    FIELD_D_CM, FIELD_ESTIMATED_BIRTH_YEAR, FIELD_H_M, FIELD_H_MEASURED,
    FIELD_LAT, FIELD_LON, FIELD_L10_MM, FIELD_NOTE,
    FIELD_NUMBER, FIELD_OPERATOR, FIELD_PARCEL, FIELD_PARCEL_ID,
    FIELD_PRESERVED, FIELD_PRESSLER_COEFF, FIELD_SAMPLE_AREA_ID, FIELD_SHOOT,
    FIELD_SPECIES, FIELD_SPECIES_ID, FIELD_STANDARD,
    PRESSLER_DEFAULT, RECORDS, SESSION, TREE_H_QUANTUM,
)


@dataclass(frozen=True)
class TreeMeasurements:
    date: date_type
    d_cm: int
    h_m: Decimal


_PAI_PARSE_COORDS = 'coords'
_PAI_PARSE_DUPLICATE = 'duplicate'


def sample_import_rows(payload: dict, survey: Survey) -> tuple[list[dict], list[str]]:
    records = _payload_records(payload)
    if records is None:
        return [], [S.IPSO_ERR_IMPORT_RECORDS_ARRAY]

    species_ids = {
        r.get(FIELD_SPECIES_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SPECIES_ID)) is int
    }
    area_ids = {
        r.get(FIELD_SAMPLE_AREA_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SAMPLE_AREA_ID)) is int
    }
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
        parsed = _sample_record_values(record, area, sp)
        if parsed is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_SAMPLE_FIELDS_INVALID.format(i))
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
        rows.append(parsed)
    return rows, errors


def pai_import_rows(payload: dict) -> tuple[list[dict], list[str]]:
    records = _payload_records(payload)
    if records is None:
        return [], [S.IPSO_ERR_IMPORT_RECORDS_ARRAY]
    session = payload.get(SESSION, {}) if isinstance(payload, dict) else {}
    session_operator = (
        (session.get(FIELD_OPERATOR) or '').strip() if isinstance(session, dict) else ''
    )

    species_ids = {
        r.get(FIELD_SPECIES_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_SPECIES_ID)) is int
    }
    parcel_ids = {
        r.get(FIELD_PARCEL_ID) for r in records
        if isinstance(r, dict) and type(r.get(FIELD_PARCEL_ID)) is int
    }
    species = {sp.id: sp for sp in Species.objects.filter(id__in=species_ids)}
    parcels = {
        parcel.id: parcel
        for parcel in (Parcel.objects
                       .filter(id__in=parcel_ids)
                       .select_related('region', 'eclass'))
    }
    seen_numbers = set(
        TreePreserved.objects
        .filter(parcel_id__in=parcel_ids)
        .values_list(FIELD_PARCEL_ID, FIELD_NUMBER)
    )
    next_numbers: dict[int, int] = {}

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
        parsed = _pai_record_values(
            record, parcel, sp, session_operator, seen_numbers, next_numbers,
        )
        if parsed is None:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_DH_DATE_INVALID.format(i))
            continue
        if parsed == _PAI_PARSE_COORDS:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_COORDS_REQUIRED.format(i))
            continue
        if parsed == _PAI_PARSE_DUPLICATE:
            errors.append(S.IPSO_ERR_IMPORT_RECORD_PAI_NUMBER_DUPLICATE.format(i))
            continue
        rows.append(parsed)
    return rows, errors


def apply_pai_rows(rows: list[dict]) -> int:
    return csv_preserved.apply(rows)


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


def _sample_record_values(record: dict, area: SampleArea, sp: Species) -> dict | None:
    measurements = record_measurements(record)
    if measurements is None:
        return None
    try:
        number = int(record.get(FIELD_NUMBER))
        shoot = int(record.get(FIELD_SHOOT, 0) or 0)
        l10_mm = int(record.get(FIELD_L10_MM, 0) or 0)
    except (TypeError, ValueError):
        return None
    pressler_coeff = to_decimal(record.get(FIELD_PRESSLER_COEFF), '.')
    if pressler_coeff is None:
        pressler_coeff = PRESSLER_DEFAULT
    if number <= 0 or pressler_coeff <= 0:
        return None
    coppice_value = record.get(FIELD_COPPICE)
    coppice = area.parcel.eclass.coppice if coppice_value is None else bool(coppice_value)
    return parsed_tree_row(
        area=area, row_date=measurements.date, species=sp, coppice=coppice,
        preserved=bool(record.get(FIELD_PRESERVED)), number=number,
        shoot=shoot, standard=bool(record.get(FIELD_STANDARD)),
        d_cm=measurements.d_cm, h_m=measurements.h_m, l10_mm=l10_mm,
        pressler_coeff=pressler_coeff, lat=record.get(FIELD_LAT),
        lon=record.get(FIELD_LON), acc_m=record.get(FIELD_ACC_M),
    )


def _pai_record_values(
        record: dict, parcel: Parcel, sp: Species, session_operator: str,
        seen_numbers: set[tuple[int, int]], next_numbers: dict[int, int],
):
    measurements = record_measurements(record)
    if measurements is None:
        return None
    h_m = measurements.h_m.quantize(TREE_H_QUANTUM, rounding=ROUND_HALF_UP)
    lat = record.get(FIELD_LAT)
    lon = record.get(FIELD_LON)
    if lat is None or lon is None:
        return _PAI_PARSE_COORDS
    number = record.get(FIELD_NUMBER)
    if number is None:
        number = _next_pai_number(parcel.id, seen_numbers, next_numbers)
    else:
        try:
            number = int(number)
        except (TypeError, ValueError):
            return None
        if number <= 0:
            return None
        if (parcel.id, number) in seen_numbers:
            return _PAI_PARSE_DUPLICATE
    seen_numbers.add((parcel.id, number))
    return {
        FIELD_PARCEL: parcel,
        FIELD_SPECIES: sp,
        FIELD_NUMBER: number,
        FIELD_DATE: measurements.date,
        FIELD_ESTIMATED_BIRTH_YEAR: record.get(FIELD_ESTIMATED_BIRTH_YEAR),
        FIELD_D_CM: measurements.d_cm,
        FIELD_H_M: h_m,
        FIELD_H_MEASURED: True,
        FIELD_LAT: lat,
        FIELD_LON: lon,
        FIELD_ACC_M: record.get(FIELD_ACC_M),
        FIELD_OPERATOR: (record.get(FIELD_OPERATOR) or session_operator).strip(),
        FIELD_NOTE: (record.get(FIELD_NOTE) or '').strip(),
    }


def _next_pai_number(
        parcel_id: int, seen_numbers: set[tuple[int, int]], next_numbers: dict[int, int],
) -> int:
    number = next_numbers.get(parcel_id)
    if number is None:
        number = next_sequence_number(
            TreePreserved.objects.filter(parcel_id=parcel_id), FIELD_NUMBER,
        )
    while (parcel_id, number) in seen_numbers:
        number += 1
    next_numbers[parcel_id] = number + 1
    return number
