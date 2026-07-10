"""Shared write core for high-forest tree mark imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date as date_type
from decimal import Decimal

from django.db import transaction
from django.db.models import Sum

from apps.base.digests import mark_stale
from apps.base.models import (
    HarvestPlanItem, HarvestPlanItemState, Parcel, Species, Tree, TreeMark,
    next_sequence_number, tree_mass_q,
)
from apps.base.tabacchi import tabacchi_volume_m3
from config import strings as S
from config.constants import DIGEST_FUTURE_PRODUCTION, FIELD_NUMBER

MARK_CSV_SPECIES_HEADERS = (S.CSV_COL_SPECIES, S.COL_SPECIES)


@dataclass(frozen=True)
class MarkImportRow:
    date: date_type
    parcel: Parcel
    species: Species
    number: int | None
    d_cm: int
    h_m: Decimal
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
        number: int | None, d_cm: int, h_m: Decimal, h_measured: bool,
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
        'h_m': format(h_m.quantize(Decimal('0.01')), 'f'),
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
        item = HarvestPlanItem.objects.select_for_update().get(id=item.id)
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
            tree = Tree.objects.create(
                species=row.species, parcel=row.parcel,
                lat=row.lat, lon=row.lon, acc_m=row.acc_m,
            )
            TreeMark.objects.create(
                harvest_plan_item=item, tree=tree,
                number=row.number,
                date=row.date, d_cm=row.d_cm, h_m=row.h_m,
                h_measured=row.h_measured,
                volume_m3=volume_m3, mass_q=mass_q,
                lat=row.lat, lon=row.lon, acc_m=row.acc_m,
                operator=row.operator,
                import_fingerprint=row.fingerprint,
            )

        if parsed:
            auto_advance_to_marked(item, min(row.date for row in parsed))
            rematerialize_volume_marked(item.id)

        mark_stale(
            f'mark_trees_{item.id}', 'harvest_plan_items',
            DIGEST_FUTURE_PRODUCTION, 'audit',
        )

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


def mark_volume_and_mass(d_cm: int, h_m: Decimal, species: Species):
    try:
        volume_m3 = tabacchi_volume_m3(d_cm, h_m, species.common_name)
        mass_q = tree_mass_q(volume_m3, species.density)
    except (ValueError, KeyError):
        return None, None
    return volume_m3, mass_q


def auto_advance_to_marked(item: HarvestPlanItem, mark_date: date_type) -> None:
    """Auto-advance state from planned to marked on first TreeMark."""
    if item.state == HarvestPlanItemState.PLANNED:
        item.state = HarvestPlanItemState.MARKED
        item.date_actual = mark_date
        item.version += 1
        item.save()


def rematerialize_volume_marked(item_id: int) -> None:
    """Recompute volume_marked_m3 on the linked HarvestPlanItem."""
    total = (TreeMark.objects
             .filter(harvest_plan_item_id=item_id)
             .aggregate(s=Sum('volume_m3'))['s'])
    item = HarvestPlanItem.objects.select_for_update().filter(id=item_id).first()
    if item is not None and item.volume_marked_m3 != total:
        item.volume_marked_m3 = total
        item.save(update_fields=['volume_marked_m3'])


def next_mark_number(item_id: int) -> int:
    """Return max(tree_mark.number)+1 for the item, or 1 if no marks exist."""
    return next_sequence_number(
        TreeMark.objects.filter(harvest_plan_item_id=item_id), FIELD_NUMBER,
    )
