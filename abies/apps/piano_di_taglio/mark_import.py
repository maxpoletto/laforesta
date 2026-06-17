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
from config.constants import DIGEST_FUTURE_PRODUCTION, FIELD_NUMBER


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


@dataclass(frozen=True)
class MarkImportResult:
    imported: int
    skipped_duplicates: int


def csv_mark_fingerprint(
        *, date: date_type, species_name: str, d_cm: int, h_m: Decimal,
        lat: float | None, lon: float | None, operator: str,
) -> str:
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
    existing_fps = set(
        TreeMark.objects
        .filter(harvest_plan_item_id=item.id, import_fingerprint__isnull=False)
        .values_list('import_fingerprint', flat=True)
    )
    parsed: list[MarkImportRow] = []
    skipped = 0
    for row in rows:
        if row.fingerprint in existing_fps:
            skipped += 1
            continue
        parsed.append(row)
        existing_fps.add(row.fingerprint)

    with transaction.atomic():
        next_number = next_mark_number(item.id)
        for row in parsed:
            number = row.number
            if number is None:
                number = next_number
                next_number += 1
            volume_m3, mass_q = mark_volume_and_mass(row.d_cm, row.h_m, row.species)
            tree = Tree.objects.create(
                species=row.species, parcel=row.parcel,
                lat=row.lat, lon=row.lon, acc_m=row.acc_m,
            )
            TreeMark.objects.create(
                harvest_plan_item=item, tree=tree,
                number=number,
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

    return MarkImportResult(imported=len(parsed), skipped_duplicates=skipped)


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
