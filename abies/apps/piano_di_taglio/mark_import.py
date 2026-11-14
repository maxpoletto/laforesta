"""Shared write core for high-forest tree mark imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date as date_type
from decimal import Decimal

from django.db import transaction

from apps.base import csv_io
from apps.base.digests import mark_stale
from apps.base.models import (
    HarvestPlanItem, Parcel, Species, Tree, TreeMark,
    next_sequence_number,
)
from apps.base.equations import estimate_volume_and_mass
from config import strings as S
from config.constants import FIELD_NUMBER
from apps.piano_di_taglio.totals import lock_plan_items, refresh_marked_total
from apps.piano_di_taglio.dependencies import marked_tree_dependencies

MARK_CSV_SPECIES_HEADERS = (S.CSV_COL_SPECIES, S.COL_SPECIES)


@dataclass(frozen=True)
class MarkImportRow:
    date: date_type
    parcel: Parcel
    species: Species
    number: int | None
    d_cm: int
    h_m: Decimal | None
    h_measured: bool
    lat: float | None
    lon: float | None
    acc_m: int | None
    operator: str
    fingerprint: str
    legacy_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarkImportResult:
    imported: int
    skipped_duplicates: int
    errors: list[str]


def mark_parcel_matches_item(item: HarvestPlanItem, parcel: Parcel) -> bool:
    """Return whether ``parcel`` is a valid target for marks on ``item``."""
    if item.parcel_id is not None:
        return parcel.id == item.parcel_id
    return item.region_id is not None and parcel.region_id == item.region_id


def csv_mark_fingerprint(
        *, source_row: int, date: date_type, parcel_id: int, species_id: int,
        number: int | None, d_cm: int, h_m: Decimal | None, h_measured: bool,
        lat: float | None, lon: float | None, acc_m: int | None, operator: str,
) -> str:
    """Fingerprint every canonical CSV field plus its stable row position."""
    raw = json.dumps({
        'version': 2,
        'source': 'csv',
        'source_row': source_row,
        'date': date.isoformat(),
        'parcel_id': parcel_id,
        'species_id': species_id,
        'number': number,
        'd_cm': d_cm,
        'h_m': (format(h_m.quantize(Decimal('0.01')), 'f')
                if h_m is not None else None),
        'h_measured': h_measured,
        'lat': lat,
        'lon': lon,
        'acc_m': acc_m,
        'operator': operator,
    }, sort_keys=True, separators=(',', ':'))
    return f'v2:{hashlib.sha256(raw.encode()).hexdigest()}'


def legacy_csv_mark_fingerprint(
        *, date: date_type, species_name: str, d_cm: int, h_m: Decimal,
        lat: float | None, lon: float | None, operator: str,
) -> str:
    """Reproduce the pre-v2 hash so historical imports remain idempotent."""
    fp_src = f'{date}|{species_name}|{d_cm}|{h_m}|{lat}|{lon}|{operator}'
    return hashlib.sha256(fp_src.encode()).hexdigest()


def ipso_mark_fingerprint(session_id: str, record: dict) -> str:
    raw = json.dumps({
        'source': 'ipso',
        'session_id': session_id,
        'client_record_id': record.get('client_record_id'),
        'date': record.get('date'),
        'parcel_id': record.get('parcel_id'),
        'species_id': record.get('species_id'),
        'number': record.get('number'),
        'd_cm': record.get('d_cm'),
        'h_m': record.get('h_m'),
        'lat': record.get('lat'),
        'lon': record.get('lon'),
    }, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def import_mark_rows(item: HarvestPlanItem, rows: list[MarkImportRow]) -> MarkImportResult:
    with transaction.atomic():
        item = lock_plan_items([item.id])[item.id]
        if any(not mark_parcel_matches_item(item, row.parcel) for row in rows):
            return MarkImportResult(
                imported=0, skipped_duplicates=0,
                errors=[S.ERR_MARK_PARCEL_NOT_IN_TARGET],
            )
        parsed, skipped = _unskipped_import_rows(item.id, rows)
        errors = mark_number_duplicate_errors(item.id, parsed)
        if errors:
            return MarkImportResult(imported=0, skipped_duplicates=skipped, errors=errors)

        for row in parsed:
            volume_m3, mass_q = mark_volume_and_mass(row.d_cm, row.h_m, row.species)
            tree = Tree.objects.create(species=row.species)
            TreeMark.objects.create(
                harvest_plan_item=item, tree=tree, parcel=row.parcel,
                number=row.number,
                date=row.date, d_cm=row.d_cm, h_m=row.h_m,
                h_measured=row.h_measured,
                volume_m3=volume_m3, mass_q=mass_q,
                lat=row.lat, lon=row.lon, acc_m=row.acc_m,
                operator=row.operator,
                import_fingerprint=row.fingerprint,
            )

        if parsed:
            refresh_marked_total(item.id, mark_date=min(row.date for row in parsed))

        mark_stale(*marked_tree_dependencies([item.id]).data_ids)

    return MarkImportResult(imported=len(parsed), skipped_duplicates=skipped, errors=[])


def _unskipped_import_rows(
        item_id: int, rows: list[MarkImportRow],
) -> tuple[list[MarkImportRow], int]:
    persisted_fps = set(
        TreeMark.objects
        .filter(harvest_plan_item_id=item_id, import_fingerprint__isnull=False)
        .values_list('import_fingerprint', flat=True)
    )
    batch_fps: set[str] = set()
    parsed: list[MarkImportRow] = []
    skipped = 0
    for row in rows:
        persisted_aliases = {row.fingerprint, *row.legacy_fingerprints}
        if persisted_aliases & persisted_fps or row.fingerprint in batch_fps:
            skipped += 1
            continue
        parsed.append(row)
        batch_fps.add(row.fingerprint)
    return parsed, skipped


def mark_number_duplicate_errors(
        item_id: int, rows: list[MarkImportRow], *, exclude_mark_id: int | None = None,
) -> list[str]:
    numbers = [row.number for row in rows if row.number is not None]
    if not numbers:
        return []
    existing = TreeMark.objects.filter(
        harvest_plan_item_id=item_id, number__in=numbers,
    )
    if exclude_mark_id is not None:
        existing = existing.exclude(id=exclude_mark_id)
    seen = set(existing.values_list(FIELD_NUMBER, flat=True))
    errors = []
    for number in numbers:
        if number in seen:
            errors.append(S.ERR_MARK_NUMBER_DUPLICATE.format(number))
        else:
            seen.add(number)
    return errors


def mark_volume_and_mass(
        d_cm: int, h_m: Decimal | None, species: Species,
):
    return estimate_volume_and_mass(d_cm, h_m, species)


def rematerialize_volume_marked(item_id: int) -> HarvestPlanItem | None:
    """Compatibility entry point for the intervention totals service."""
    return refresh_marked_total(item_id)


def next_mark_number(item_id: int) -> int:
    """Return max(tree_mark.number)+1 for the item, or 1 if no marks exist."""
    return next_sequence_number(
        TreeMark.objects.filter(harvest_plan_item_id=item_id), FIELD_NUMBER,
    )


def render_mark_csv(item, *, row_ids=None):
    """Canonical marked-tree CSV used by table exports and intervention backups."""
    delimiter, decimal_sep = csv_io.export_format()
    marks_buf, marks_w = csv_io.csv_buffer(delimiter)
    marks_w.writerow([
        S.CSV_COL_DATA, S.CSV_COL_REGION, S.CSV_COL_PARCEL,
        S.CSV_COL_DAMAGED, S.CSV_COL_NUMBER, S.CSV_COL_SPECIES,
        S.CSV_COL_D_CM, S.CSV_COL_H_M, S.CSV_COL_H_MEASURED,
        S.CSV_COL_LAT, S.CSV_COL_LON, S.CSV_COL_ACC_M, S.CSV_COL_OPERATOR,
    ])
    marks_qs = (TreeMark.objects
                .filter(harvest_plan_item=item)
                .select_related('tree__species', 'parcel__region')
                .order_by('date', 'id'))
    if row_ids is not None:
        marks_qs = marks_qs.filter(id__in=row_ids)
    for tm in marks_qs:
        # Compresa / Particella reflect the row-level mark parcel; for
        # region-wide items, this differs from the item's missing parcel by
        # design.
        parcel = tm.parcel
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
    return marks_buf.getvalue()
