"""Harvest-plan CSV import core: alias dicts, read/parse helpers, flag
decoder, index builder, and the write transaction, shared by the in-app
upload view and bootstrap.

``parse_fustaia_rows`` and ``parse_ceduo_rows`` are pure — no DB writes,
foreign keys resolved against an injected ``PlanIndexes`` — so the same
code can back a true ``--check`` dry-run against a staged index.
"""

from dataclasses import dataclass
from datetime import date as date_type

from django.db import transaction

from apps.base import csv_io
from apps.base.digests import mark_stale
from apps.base.models import (
    HarvestDetail,
    HarvestPlan,
    HarvestPlanItem,
    Parcel,
    ParcelPlanDetail,
    Region,
)
from config import strings as S
from config.constants import (
    DIGEST_FUTURE_PRODUCTION,
    FIELD_DAMAGED,
    FIELD_INTERVENTION_AREA_HA,
    FIELD_NOTE,
    FIELD_PARCEL_ID,
    FIELD_PERIOD_Y,
    FIELD_PSR,
    FIELD_REGION_ID,
    FIELD_UNHEALTHY,
    FIELD_VOLUME_PLANNED_M3,
    FIELD_YEAR_PLANNED,
    ROW_ID,
)

# Column aliases — each logical field accepts the legacy pdg-2026 name
# AND the display-name used by the per-table CSV export, so a freshly
# exported plan zip round-trips through the importer.  `required` raises
# a missing-column error pre-parse; `optional` keys are absent from row
# dicts when the CSV doesn't carry them.
HIGHFOREST_REQUIRED = {
    'compresa':   [S.CSV_COL_REGION],
    'particella': [S.CSV_COL_PARCEL],
    'anno':       [S.COL_YEAR_PLANNED, S.CSV_COL_YEAR],
    'prelievo':   [S.CSV_COL_HARVEST_M3, S.COL_VOLUME_PLANNED],
}
HIGHFOREST_OPTIONAL = {
    # Present in Abies plan exports. Used as row identity when re-importing
    # rows that otherwise share the same plan/parcel/year.
    'row_id':     [S.COL_ID, ROW_ID],
    # Holds the flag string ("Catastrofato" / "Fitosanitario" / "PSR")
    # in Abies exports; required for region-wide rows (Particella = 'X').
    'note':       [S.COL_NOTE],
    # Free text; separate from `note` because Abies exports use `note` for
    # the flag string. Legacy pdg-2026 fustaia CSVs simply omit this.
    'free_note':  [S.CSV_COL_EXTRA_NOTE],
}
COPPICE_REQUIRED = {
    'anno':       [S.COL_YEAR_PLANNED, S.CSV_COL_YEAR],
    'compresa':   [S.CSV_COL_REGION],
    'particella': [S.CSV_COL_PARCEL],
    'superficie': [S.CSV_COL_SURFACE_HA],                     # 'Superficie intervento (ha)'
    'turno':      [S.CSV_COL_PERIOD_Y],                           # 'Turno (a)'
}
COPPICE_OPTIONAL = {
    # Present in Abies plan exports. Used as row identity when re-importing
    # rows that otherwise share the same plan/parcel/year.
    'row_id':     [S.COL_ID, ROW_ID],
    # In Abies exports: 'Altre note' = free-text, 'Note' = flag string.
    # In legacy pdg-2026 exports: only 'Note' is present and is itself
    # free-text — handled by checking `has_altre_note` at parse time.
    'free_note':  [S.CSV_COL_EXTRA_NOTE],                        # 'Altre note'
    'flag_note':  [S.COL_NOTE],                                  # 'Note' (flag string)
}


# Minimal required columns: plan, region, parcel, year — all that
# load_canonical_items needs before it can branch on row type.
PLAN_ITEMS_CSV_REQUIRED = [S.CSV_COL_PLAN, S.CSV_COL_REGION, S.CSV_COL_PARCEL, S.CSV_COL_YEAR]


class CsvError:
    def __init__(self, message: str):
        self.message = message


@dataclass
class AliasedCsv:
    """A parsed CSV paired with its rows remapped to logical alias names, so
    cell parsing goes through ``reader`` (which carries the detected decimal
    separator) and header presence through ``reader.fieldnames``."""
    reader: csv_io.CsvReader
    rows: list[dict]


def read_optional(upload, required, optional=None) -> AliasedCsv | CsvError | None:
    """Parse an uploaded CSV if present.

    Delimiter is auto-detected (``,`` or ``;``) so files exported in
    either Italian (``;``) or pdg-2026's English (``,``) style both
    round-trip.  ``required`` and ``optional`` map logical alias names
    to candidate column-name lists; the first candidate that appears in
    the header wins.  Returns an `AliasedCsv` whose rows are alias-keyed
    dicts (optional aliases that did not resolve are absent from each row),
    `None` if no file was uploaded, or `CsvError` on a malformed file.
    """
    if upload is None:
        return None
    try:
        reader = csv_io.read(upload)
    except csv_io.CsvError as exc:
        return CsvError(str(exc))
    fieldset = set(reader.fieldnames)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for alias, candidates in required.items():
        hit = next((c for c in candidates if c in fieldset), None)
        if hit is None:
            missing.append(candidates[0])
        else:
            resolved[alias] = hit
    if missing:
        return CsvError(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
    for alias, candidates in (optional or {}).items():
        hit = next((c for c in candidates if c in fieldset), None)
        if hit is not None:
            resolved[alias] = hit
    rows = [{alias: r.get(col, '') for alias, col in resolved.items()}
            for r in reader]
    return AliasedCsv(reader, rows)


# Substrings (case-insensitive) recognised in the Note column for
# flag inference on import.  Matches the per-flag labels used by
# `render_flag_note` so a round-trip preserves the boolean.
_FLAG_KEYWORDS = {
    'damaged':   S.FLAG_DAMAGED.lower(),
    'unhealthy': S.FLAG_UNHEALTHY.lower(),
    'psr':       S.FLAG_PSR.lower(),
}


def parse_flag_keywords(note: str) -> tuple[bool, bool, bool]:
    """Scan a Note cell for flag keywords. Returns (damaged, unhealthy, psr)."""
    s = (note or '').lower()
    return (
        _FLAG_KEYWORDS['damaged']   in s,
        _FLAG_KEYWORDS['unhealthy'] in s,
        _FLAG_KEYWORDS['psr']       in s,
    )


def _optional_row_id(data, row, row_number, errors):
    raw = row.get('row_id', '')
    if raw in (None, ''):
        return None, True
    value = data.reader.integer(raw)
    if value is None:
        errors.append(S.ERR_CSV_VALUE_PARSE.format(row_number, S.COL_ID, raw))
        return None, False
    return value, True


def _optional_decimal(reader, row, column, row_number, errors):
    value, ok = reader.opt_decimal(row.get(column))
    if not ok:
        errors.append(S.ERR_CSV_VALUE_PARSE.format(
            row_number, column, row.get(column, ''),
        ))
    return value, ok


def parse_fustaia_rows(data, parcel_cache, region_cache, errors):
    """Parse fustaia.csv rows.  A row is either parcel-scoped (the
    common case) or whole-region (Particella = ``X`` — see
    ``S.PARCEL_WHOLE_REGION_MARK``).  The Note column, when present,
    is scanned for ``Catastrofato`` / ``Fitosanitario`` / ``PSR``
    substrings to drive the damaged / unhealthy / psr flags — required
    for whole-region rows, optional for parcel-scoped ones (preserves
    flags on round-trip).
    """
    out = []
    for i, row in enumerate(data.rows, 2):
        compresa = (row.get('compresa') or '').strip()
        particella = (row.get('particella') or '').strip()
        row_id, row_id_ok = _optional_row_id(data, row, i, errors)
        if not row_id_ok:
            continue
        note_raw = (row.get('note') or '').strip()
        damaged, unhealthy, psr = parse_flag_keywords(note_raw)
        is_whole_region = (
            particella.upper() == S.PARCEL_WHOLE_REGION_MARK.upper()
        )
        if is_whole_region:
            region = region_cache.get(compresa.lower())
            if region is None:
                errors.append(S.ERR_CSV_REGION_NOT_FOUND.format(i, compresa))
                continue
            if not (damaged or unhealthy):
                errors.append(S.ERR_CSV_WHOLE_REGION_REQUIRES_FLAG.format(
                    i, particella,
                ))
                continue
            parcel = None
        else:
            region = None
            parcel = parcel_cache.get((compresa.lower(), particella))
            if parcel is None:
                errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                    i, compresa, particella,
                ))
                continue
        year = data.reader.integer(row.get('anno'))
        if year is None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_YEAR, row.get('anno', ''),
            ))
            continue
        prelievo = data.reader.decimal(row.get('prelievo'))
        parsed = {
            ROW_ID: row_id,
            FIELD_REGION_ID: region,
            FIELD_PARCEL_ID: parcel,
            FIELD_YEAR_PLANNED: year,
            FIELD_VOLUME_PLANNED_M3: prelievo,
            FIELD_DAMAGED:   damaged,
            FIELD_UNHEALTHY: unhealthy,
            FIELD_PSR:       psr,
        }
        if 'free_note' in row:
            parsed[FIELD_NOTE] = (row.get('free_note') or '').strip()
        out.append(parsed)
    return out


def parse_ceduo_rows(data, parcel_cache, errors):
    """Parse ceduo.csv rows.  Disambiguates Abies-exported (`Altre note`
    = free-text + `Note` = flag string) from legacy pdg-2026 (`Note` =
    free-text, no flag column) via header presence: if `Altre note` is
    in the header, treat `Note` as the flag string; otherwise treat
    `Note` as free-text and skip flag parsing.
    """
    out = []
    has_altre_note = S.CSV_COL_EXTRA_NOTE in data.reader.fieldnames
    for i, row in enumerate(data.rows, 2):
        row_id, row_id_ok = _optional_row_id(data, row, i, errors)
        if not row_id_ok:
            continue
        compresa = (row.get('compresa') or '').strip()
        particella = (row.get('particella') or '').strip()
        parcel = parcel_cache.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                i, compresa, particella,
            ))
            continue
        year = data.reader.integer(row.get('anno'))
        interval = data.reader.integer(row.get('turno'))
        area = data.reader.decimal(row.get('superficie'))
        if year is None or interval is None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_PERIOD_Y, row.get('turno', ''),
            ))
            continue
        if has_altre_note:
            free_note = (row.get('free_note') or '').strip()
            damaged, unhealthy, psr = parse_flag_keywords(
                row.get('flag_note') or '',
            )
        else:
            # Legacy pdg-2026: 'Note' is free-text and there are no flags.
            free_note = (row.get('flag_note') or row.get('free_note') or '').strip()
            damaged = unhealthy = psr = False
        out.append({
            ROW_ID: row_id,
            FIELD_PARCEL_ID: parcel,
            FIELD_YEAR_PLANNED: year,
            FIELD_INTERVENTION_AREA_HA: area,
            FIELD_PERIOD_Y: interval,
            FIELD_NOTE: free_note,
            FIELD_DAMAGED:   damaged,
            FIELD_UNHEALTHY: unhealthy,
            FIELD_PSR:       psr,
        })
    return out


@dataclass
class PlanIndexes:
    """Lookups the parse helpers need, injected so they run against the live DB
    (view) or a staged index (bootstrap)."""
    parcels: dict   # (region_name.lower(), parcel_name) -> Parcel
    regions: dict   # region_name.lower() -> Region


def db_indexes() -> PlanIndexes:
    """Build ``PlanIndexes`` from the live database."""
    regions = {r.name.lower(): r for r in Region.objects.all()}
    parcels = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region', 'eclass')
    }
    return PlanIndexes(parcels, regions)


def apply(*, target_plan, name, description, fustaia_parsed, ceduo_parsed):
    """Create or upsert a HarvestPlan and its items from parsed rows.

    Returns ``(plan, n_items)``.  Rows exported by Abies carry an optional
    item ID, which lets re-import update the exact row even when several
    interventions share the same plan/parcel/year.  Hand-authored CSVs with
    no ID keep the legacy upsert key, so revised rows still overwrite the
    matching calendar item.
    """
    all_years = (
        [r[FIELD_YEAR_PLANNED] for r in fustaia_parsed]
        + [r[FIELD_YEAR_PLANNED] for r in ceduo_parsed]
    )
    with transaction.atomic():
        if target_plan is None:
            year_start = min(all_years) if all_years else date_type.today().year
            year_end = max(all_years) if all_years else year_start
            plan = HarvestPlan.objects.create(
                name=name, description=description,
                year_start=year_start, year_end=year_end,
            )
        else:
            plan = target_plan
            if all_years:
                new_start = min(plan.year_start, *all_years)
                new_end = max(plan.year_end, *all_years)
                if new_start != plan.year_start or new_end != plan.year_end:
                    plan.year_start = new_start
                    plan.year_end = new_end
                    plan.version += 1
                    plan.save()

        # Coppice rows need a HarvestDetail per (description, interval).
        # Seed the cache from existing details under this plan so re-imports
        # reuse the same rows.
        detail_cache: dict[tuple[str, int], HarvestDetail] = {
            (ppd.harvest_detail.description, ppd.harvest_detail.interval):
                ppd.harvest_detail
            for ppd in ParcelPlanDetail.objects
                .filter(harvest_plan=plan)
                .select_related('harvest_detail')
            if ppd.harvest_detail.interval is not None
        }
        for r in ceduo_parsed:
            interval = r[FIELD_PERIOD_Y]
            desc = f'{S.HARVEST_DETAIL} {interval}a'
            key = (desc, interval)
            hd = detail_cache.get(key)
            if hd is None:
                hd = HarvestDetail.objects.create(
                    description=desc, interval=interval,
                )
                detail_cache[key] = hd
            ParcelPlanDetail.objects.update_or_create(
                harvest_plan=plan, parcel=r[FIELD_PARCEL_ID],
                defaults={'harvest_detail': hd},
            )

        def save_plan_item(row, identity, defaults):
            fields = {**identity, **defaults}
            row_id = row.get(ROW_ID)
            if row_id:
                item = HarvestPlanItem.objects.filter(
                    id=row_id, harvest_plan=plan,
                ).first()
                if item is not None:
                    for field, value in fields.items():
                        setattr(item, field, value)
                    item.save()
                    return item
                item = HarvestPlanItem.objects.filter(**fields).first()
                if item is not None:
                    return item
                return HarvestPlanItem.objects.create(**fields)
            item, _ = HarvestPlanItem.objects.update_or_create(
                **identity, defaults=defaults,
            )
            return item

        n_items = 0
        for r in fustaia_parsed:
            flag_defaults = {
                'volume_planned_m3': r[FIELD_VOLUME_PLANNED_M3],
                'damaged':   r[FIELD_DAMAGED],
                'unhealthy': r[FIELD_UNHEALTHY],
                'psr':       r[FIELD_PSR],
            }
            if FIELD_NOTE in r:
                flag_defaults['note'] = r[FIELD_NOTE]
            if r[FIELD_PARCEL_ID] is not None:
                identity = {
                    'harvest_plan': plan,
                    'region': None,
                    'parcel': r[FIELD_PARCEL_ID],
                    'year_planned': r[FIELD_YEAR_PLANNED],
                }
            else:
                identity = {
                    'harvest_plan': plan,
                    'region': r[FIELD_REGION_ID],
                    'parcel': None,
                    'year_planned': r[FIELD_YEAR_PLANNED],
                }
            save_plan_item(r, identity, flag_defaults)
            n_items += 1
        for r in ceduo_parsed:
            save_plan_item(r, {
                'harvest_plan': plan,
                'region': None,
                'parcel': r[FIELD_PARCEL_ID],
                'year_planned': r[FIELD_YEAR_PLANNED],
            }, {
                'intervention_area_ha': r[FIELD_INTERVENTION_AREA_HA],
                'note':      r[FIELD_NOTE],
                'damaged':   r[FIELD_DAMAGED],
                'unhealthy': r[FIELD_UNHEALTHY],
                'psr':       r[FIELD_PSR],
            })
            n_items += 1

        mark_stale('harvest_plans', 'harvest_plan_items', DIGEST_FUTURE_PRODUCTION, 'audit')

    return plan, n_items


@dataclass
class CanonicalItemsForPlan:
    plan: HarvestPlan
    fustaia: list
    ceduo: list


def validate_canonical_items(reader, indexes: PlanIndexes, plans: dict):
    """Validate the unified bootstrap ``harvest_plan_items.csv``.

    Returns ``(rows_by_plan, errors)`` without writing to the database.  The
    caller decides whether validated rows are safe to apply.
    """
    errors = []
    rows_by_plan: dict[str, CanonicalItemsForPlan] = {}

    def group_for(plan_name: str) -> CanonicalItemsForPlan:
        plan = plans[plan_name]
        group = rows_by_plan.get(plan_name)
        if group is None:
            group = CanonicalItemsForPlan(plan=plan, fustaia=[], ceduo=[])
            rows_by_plan[plan_name] = group
        return group

    for i, row in enumerate(reader, 2):
        plan_name = (row.get(S.CSV_COL_PLAN) or '').strip()
        if plan_name not in plans:
            errors.append(S.ERR_CSV_PLAN_NOT_FOUND.format(i, plan_name))
            continue

        compresa = (row.get(S.CSV_COL_REGION) or '').strip()
        particella = (row.get(S.CSV_COL_PARCEL) or '').strip()
        is_region_wide = not particella

        year = reader.integer(row.get(S.CSV_COL_YEAR))
        if year is None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(
                i, S.CSV_COL_YEAR, row.get(S.CSV_COL_YEAR, '')))
            continue

        damaged_val, damaged_ok = reader.opt_bool(
            row.get(S.CSV_COL_HARVEST_DAMAGED, ''))
        unhealthy_val, unhealthy_ok = reader.opt_bool(
            row.get(S.CSV_COL_HARVEST_UNHEALTHY, ''))
        psr_val, psr_ok = reader.opt_bool(row.get(S.CSV_COL_HARVEST_PSR, ''))
        if not (damaged_ok and unhealthy_ok and psr_ok):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_HARVEST_DAMAGED}/'
                   f'{S.CSV_COL_HARVEST_UNHEALTHY}/{S.CSV_COL_HARVEST_PSR}'))
            continue
        damaged = bool(damaged_val)
        unhealthy = bool(unhealthy_val)
        psr = bool(psr_val)

        note = (row.get(S.CSV_COL_NOTE) or '').strip()
        group = group_for(plan_name)

        if is_region_wide:
            region = indexes.regions.get(compresa.lower())
            if region is None:
                errors.append(S.ERR_CSV_REGION_NOT_FOUND.format(i, compresa))
                continue
            if not (damaged or unhealthy):
                errors.append(S.ERR_CSV_PLAN_ITEM_REGION_REQUIRES_FLAG.format(i))
                continue
            volume, ok = _optional_decimal(
                reader, row, S.CSV_COL_HARVEST_M3, i, errors,
            )
            if not ok:
                continue
            group.fustaia.append({
                FIELD_REGION_ID: region,
                FIELD_PARCEL_ID: None,
                FIELD_YEAR_PLANNED: year,
                FIELD_VOLUME_PLANNED_M3: volume,
                FIELD_DAMAGED: damaged,
                FIELD_UNHEALTHY: unhealthy,
                FIELD_PSR: psr,
            })
        else:
            parcel = indexes.parcels.get((compresa.lower(), particella))
            if parcel is None:
                errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                    i, compresa, particella))
                continue
            if parcel.eclass.coppice:
                area, ok = _optional_decimal(
                    reader, row, S.CSV_COL_SURFACE_HA, i, errors,
                )
                if not ok:
                    continue
                interval = reader.integer(row.get(S.CSV_COL_PERIOD_Y))
                if interval is None:
                    errors.append(S.ERR_CSV_VALUE_PARSE.format(
                        i, S.CSV_COL_PERIOD_Y, row.get(S.CSV_COL_PERIOD_Y, '')))
                    continue
                group.ceduo.append({
                    FIELD_PARCEL_ID: parcel,
                    FIELD_YEAR_PLANNED: year,
                    FIELD_INTERVENTION_AREA_HA: area,
                    FIELD_PERIOD_Y: interval,
                    FIELD_NOTE: note,
                    FIELD_DAMAGED: damaged,
                    FIELD_UNHEALTHY: unhealthy,
                    FIELD_PSR: psr,
                })
            else:
                volume, ok = _optional_decimal(
                    reader, row, S.CSV_COL_HARVEST_M3, i, errors,
                )
                if not ok:
                    continue
                group.fustaia.append({
                    FIELD_REGION_ID: None,
                    FIELD_PARCEL_ID: parcel,
                    FIELD_YEAR_PLANNED: year,
                    FIELD_VOLUME_PLANNED_M3: volume,
                    FIELD_DAMAGED: damaged,
                    FIELD_UNHEALTHY: unhealthy,
                    FIELD_PSR: psr,
                })

    return rows_by_plan, errors


def apply_canonical_items(rows_by_plan: dict[str, CanonicalItemsForPlan]) -> int:
    """Apply validated canonical plan items.  Returns rows applied."""
    total_items = 0
    for group in rows_by_plan.values():
        plan = group.plan
        _, n = apply(
            target_plan=plan,
            name=plan.name,
            description=plan.description,
            fustaia_parsed=group.fustaia,
            ceduo_parsed=group.ceduo,
        )
        total_items += n
    return total_items


def load_canonical_items(reader, indexes: PlanIndexes, plans: dict):
    """Validate and apply the unified bootstrap ``harvest_plan_items.csv``.

    Kept as the older convenience API for tests and callers that want the legacy
    combined behavior.  New bootstrap orchestration calls validate/apply
    separately so it can uniformly skip applying a file when validation reports
    any error.
    """
    rows_by_plan, errors = validate_canonical_items(reader, indexes, plans)
    if errors:
        return 0, errors
    return apply_canonical_items(rows_by_plan), errors
