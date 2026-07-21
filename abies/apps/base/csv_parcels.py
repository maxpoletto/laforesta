"""Parcel CSV import core (``particelle.csv``): the three-phase contract
(``db_indexes`` -> ``validate_rows`` -> ``apply``).

Parcels carry two foreign keys (region, eclass) and a composite natural key
(name, region), so this is a bespoke core in the style of
``apps/campionamenti/csv_trees`` rather than a declarative ``RefTable`` (which is
for flat, FK-free tables).  ``validate_rows`` is pure and resolves FKs against an
injected ``ParcelIndexes``; ``apply`` is create-only for empty-instance
bootstrap, while ``update_existing`` is the explicit live-DB update path. No
synthetic 'X' parcels — the canonical contract has none.

The caller verifies required-column presence via
``csv_io.read(text, required_cols=PARCEL_CSV_REQUIRED)`` before calling
``validate_rows`` (same contract as the trees view).  Parsed rows are dicts of
Parcel model kwargs keyed by model field name.
"""

from dataclasses import dataclass

from django.db import transaction

from apps.base.digests import mark_all_stale
from apps.base.models import Eclass, Parcel, Region
from config import strings as S
from config.constants import VERSION

PARCEL_CSV_REQUIRED = [
    S.CSV_COL_REGION, S.CSV_COL_CLASS, S.CSV_COL_PARCEL, S.CSV_COL_AREA_HA,
]
PARCEL_CSV_OPTIONAL = [
    S.CSV_COL_AVE_AGE, S.CSV_COL_LOCATION, S.CSV_COL_ALT_MIN, S.CSV_COL_ALT_MAX,
    S.CSV_COL_ASPECT, S.CSV_COL_GRADE_PCT, S.CSV_COL_VEG_DESC, S.CSV_COL_GEO_DESC,
    S.CSV_COL_CUTTING_PLAN, S.CSV_COL_INTERVAL, S.CSV_COL_STANDARDS,
]
PARCEL_UPDATE_FIELDS = [
    'eclass', 'area_ha', 'ave_age', 'location_name', 'altitude_min_m',
    'altitude_max_m', 'aspect', 'grade_pct', 'desc_veg', 'desc_geo',
    'cutting_plan', 'intervention_interval', 'standards_per_ha',
]


@dataclass
class ParcelUpdate:
    parcel: Parcel
    data: dict
    fields: list[str]


@dataclass
class ParcelIndexes:
    """Lookups ``validate_rows`` needs, injected so validation runs against the
    live DB or a staged index (bootstrap)."""
    regions: dict    # region name -> Region
    eclasses: dict   # eclass name -> Eclass


def db_indexes() -> ParcelIndexes:
    return ParcelIndexes(
        regions={r.name: r for r in Region.objects.all()},
        eclasses={e.name: e for e in Eclass.objects.all()},
    )


def validate_rows(reader, idx: ParcelIndexes):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed, errors)``.  Each parsed row is a dict of Parcel model
    kwargs (region/eclass resolved to objects).  Errors name the 1-based data
    row (header is row 1) and the offending value.
    """
    errors, parsed, seen = [], [], set()
    for i, row in enumerate(reader, 2):
        region = idx.regions.get(row[S.CSV_COL_REGION].strip())
        if region is None:
            errors.append(
                S.ERR_CSV_REGION_NOT_FOUND.format(i, row[S.CSV_COL_REGION].strip()))
            continue
        eclass = idx.eclasses.get(row[S.CSV_COL_CLASS].strip())
        if eclass is None:
            errors.append(
                S.ERR_CSV_ECLASS_NOT_FOUND.format(i, row[S.CSV_COL_CLASS].strip()))
            continue
        name = row[S.CSV_COL_PARCEL].strip()
        if not name:
            errors.append(S.ERR_CSV_VALUE_REQUIRED.format(i, S.CSV_COL_PARCEL))
            continue
        key = (region.id, name)
        if key in seen:
            errors.append(S.ERR_CSV_DUPLICATE_KEY.format(i, S.CSV_COL_PARCEL, name))
            continue
        area_ha = reader.decimal(row.get(S.CSV_COL_AREA_HA))
        if area_ha is None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_AREA_HA, (row.get(S.CSV_COL_AREA_HA) or '').strip()))
            continue
        optional_ints, bad_col = {}, None
        for field_name, col in (
            ('ave_age', S.CSV_COL_AVE_AGE),
            ('altitude_min_m', S.CSV_COL_ALT_MIN),
            ('altitude_max_m', S.CSV_COL_ALT_MAX),
            ('grade_pct', S.CSV_COL_GRADE_PCT),
            ('intervention_interval', S.CSV_COL_INTERVAL),
            ('standards_per_ha', S.CSV_COL_STANDARDS),
        ):
            value, ok = reader.opt_int(row.get(col))
            if not ok:
                bad_col = col
                break
            optional_ints[field_name] = value
        if bad_col is not None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, bad_col, (row.get(bad_col) or '').strip()))
            continue
        coppice_error = False
        for field_name, col in (
            ('intervention_interval', S.CSV_COL_INTERVAL),
            ('standards_per_ha', S.CSV_COL_STANDARDS),
        ):
            value = optional_ints[field_name]
            if eclass.coppice and value is None:
                errors.append(S.ERR_CSV_COPPICE_FIELD_REQUIRED.format(i, col))
                coppice_error = True
            elif not eclass.coppice and value is not None:
                errors.append(S.ERR_CSV_HIGHFOREST_COPPICE_FIELD.format(i, col))
                coppice_error = True
        if coppice_error:
            continue
        seen.add(key)
        parsed.append({
            'name': name, 'region': region, 'eclass': eclass, 'area_ha': area_ha,
            'location_name': (row.get(S.CSV_COL_LOCATION) or '').strip(),
            'aspect': (row.get(S.CSV_COL_ASPECT) or '').strip(),
            'desc_veg': (row.get(S.CSV_COL_VEG_DESC) or '').strip(),
            'desc_geo': (row.get(S.CSV_COL_GEO_DESC) or '').strip(),
            'cutting_plan': (row.get(S.CSV_COL_CUTTING_PLAN) or '').strip(),
            **optional_ints,
        })
    return parsed, errors


def plan_existing_updates(parsed) -> tuple[list[ParcelUpdate], list[tuple[str, str]]]:
    """Plan updates for existing parcels only.

    Returns ``(updates, missing)``. ``missing`` contains ``(region, parcel)``
    keys present in the validated CSV but absent from the live DB, so callers
    can fail safely before writing partial updates.
    """
    existing = _existing_parcels(parsed)
    updates, missing = [], []
    for d in parsed:
        key = (d['region'].id, d['name'])
        parcel = existing.get(key)
        if parcel is None:
            missing.append((d['region'].name, d['name']))
            continue
        fields = _changed_fields(parcel, d)
        if fields:
            updates.append(ParcelUpdate(parcel, d, fields))
    return updates, missing


def update_existing(parsed) -> int:
    """Persist validated CSV data to existing parcels only.

    Missing live parcels are refused with ``ValueError``. Changed parcels get a
    new optimistic-lock ``version`` and normal model save/history behavior.
    Returns the number of updated rows.
    """
    updates, missing = plan_existing_updates(parsed)
    if missing:
        raise ValueError('cannot update missing parcels')
    with transaction.atomic():
        for update in updates:
            for field in update.fields:
                setattr(update.parcel, field, update.data[field])
            update.parcel.version += 1
            update.parcel.save(update_fields=[*update.fields, VERSION])
        if updates:
            mark_all_stale()
    return len(updates)


def _existing_parcels(parsed) -> dict[tuple[int, str], Parcel]:
    region_ids = {d['region'].id for d in parsed}
    names = {d['name'] for d in parsed}
    return {
        (p.region_id, p.name): p
        for p in Parcel.objects.filter(region_id__in=region_ids, name__in=names)
    }


def _changed_fields(parcel: Parcel, data: dict) -> list[str]:
    fields = []
    for field in PARCEL_UPDATE_FIELDS:
        if field == 'eclass':
            if parcel.eclass_id != data[field].id:
                fields.append(field)
        elif getattr(parcel, field) != data[field]:
            fields.append(field)
    return fields


def apply(parsed) -> int:
    """Persist validated parcels, inserting on (name, region).  Create-only: a
    row whose key already exists is left unchanged (re-import is idempotent but
    applies no corrections — bootstrap loads into an empty instance; unlike the
    Settings-editable reference tables, whose ``csv_reference.apply`` does
    update).  Returns the number created (no tuple, since nothing is updated).
    Wrapped in a transaction; marks digests stale.
    """
    created = 0
    with transaction.atomic():
        for d in parsed:
            _, was_created = Parcel.objects.get_or_create(
                name=d['name'], region=d['region'],
                defaults={k: v for k, v in d.items()
                          if k not in ('name', 'region')},
            )
            if was_created:
                created += 1
        mark_all_stale()
    return created
