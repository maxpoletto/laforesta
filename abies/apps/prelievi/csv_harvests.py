"""Harvests CSV import core: the three-phase contract
(``resolve_columns`` → ``validate_rows`` → ``apply``) shared by the in-app
upload view and bootstrap.  ``validate_rows`` is pure (no DB writes); foreign
keys resolve against an injected ``HarvestIndexes``, so the same code can back
a ``--check`` dry-run.  Mirrors ``apps.campionamenti.csv_trees``.

Dynamic columns
---------------
Headers starting with ``S.CSV_COL_SPECIES_PREFIX`` or
``S.CSV_COL_TRACTOR_PREFIX`` are dynamic: each carries an integer percentage.
The key (header suffix after the prefix, whitespace-stripped) must match
``Species.common_name`` or ``Tractor.name`` exactly (case-sensitive).
"""

from dataclasses import dataclass
from datetime import date as date_type

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import Crew, Parcel, Product, Region, Species, Tractor
from apps.prelievi.models import (
    Harvest, HarvestSpecies, HarvestTractor, harvest_volume_m3,
)
from config import strings as S

# Required static columns — blank Particella is permitted (region-wide row).
HARVEST_CSV_REQUIRED = [
    S.CSV_COL_REGION,
    S.CSV_COL_PARCEL,
    S.CSV_COL_DATA,
    S.CSV_COL_CREW,
    S.CSV_COL_PRODUCT,
    S.CSV_COL_QUINTALS,
]

# Static optional columns that we recognise by name.
_OPTIONAL_STATIC = {
    S.CSV_COL_VDP,
    S.CSV_COL_PROT,
    S.CSV_COL_HARVEST_DAMAGED,
    S.CSV_COL_HARVEST_UNHEALTHY,
    S.CSV_COL_HARVEST_PSR,
    S.CSV_COL_EXTRA_NOTE,
}

# Dynamic-column kind labels.
_KIND_SPECIES = 'species'
_KIND_TRACTOR = 'tractor'


@dataclass
class HarvestIndexes:
    """Lookups ``validate_rows`` needs, injected so validation runs against the
    live DB (view) or a staged index (bootstrap)."""
    # (region.name.lower(), parcel.name) -> Parcel
    parcels: dict
    # region.name.lower() -> Region
    regions: dict
    # Crew.name -> Crew
    crews: dict
    # Product.name -> Product
    products: dict
    # Species.common_name -> Species (exact-case key)
    species: dict
    # Tractor.name -> Tractor (exact-case key, name may be None — excluded)
    tractors: dict


def db_indexes() -> HarvestIndexes:
    """Build ``HarvestIndexes`` from the live database."""
    parcels = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region').all()
    }
    regions = {r.name.lower(): r for r in Region.objects.all()}
    crews = {c.name: c for c in Crew.objects.all()}
    products = {p.name: p for p in Product.objects.all()}
    species = {sp.common_name: sp for sp in Species.objects.all()}
    tractors = {
        t.name: t for t in Tractor.objects.all() if t.name
    }
    return HarvestIndexes(parcels, regions, crews, products, species, tractors)


def resolve_columns(fieldnames):
    """Classify CSV fieldnames into static columns, dynamic columns, and missing required.

    Returns ``(static_cols, dyn, missing)`` where:

    - ``static_cols`` is the set of recognised static fieldnames present in the header.
    - ``dyn`` is a list of ``(kind, key, header)`` tuples for each dynamic column,
      where ``kind`` is ``'species'`` or ``'tractor'``, ``key`` is the trimmed
      suffix after the prefix, and ``header`` is the original header string.
    - ``missing`` is the list of required static column names that are absent.
    """
    fieldnames_set = set(fieldnames)
    static_cols = (set(HARVEST_CSV_REQUIRED) | _OPTIONAL_STATIC) & fieldnames_set
    missing = [c for c in HARVEST_CSV_REQUIRED if c not in fieldnames_set]

    dyn = []
    for header in fieldnames:
        if header.startswith(S.CSV_COL_SPECIES_PREFIX):
            key = header[len(S.CSV_COL_SPECIES_PREFIX):].strip()
            dyn.append((_KIND_SPECIES, key, header))
        elif header.startswith(S.CSV_COL_TRACTOR_PREFIX):
            key = header[len(S.CSV_COL_TRACTOR_PREFIX):].strip()
            dyn.append((_KIND_TRACTOR, key, header))

    return static_cols, dyn, missing


def validate_rows(reader, static_cols, dyn, idx: HarvestIndexes):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed_rows, errors)``.  Each parsed row is a dict ready for
    ``apply``.  ``errors`` are user-facing strings.
    """
    errors = []

    # --- Resolve dynamic columns against the DB and detect duplicates. ---
    # key = (kind, trimmed_key); value = resolved model object
    dyn_objects = {}   # (kind, key) -> Species or Tractor
    dyn_headers = {}   # (kind, key) -> first header seen (for duplicate detection)

    for kind, key, header in dyn:
        ck = (kind, key)
        if ck in dyn_headers:
            errors.append(S.ERR_CSV_DUPLICATE_DYN_COL.format(header))
            continue
        dyn_headers[ck] = header
        if kind == _KIND_SPECIES:
            obj = idx.species.get(key)
            if obj is None:
                errors.append(S.ERR_CSV_UNKNOWN_SPECIES_COL.format(key))
                continue
        else:  # _KIND_TRACTOR
            obj = idx.tractors.get(key)
            if obj is None:
                errors.append(S.ERR_CSV_UNKNOWN_TRACTOR_COL.format(key))
                continue
        dyn_objects[ck] = obj

    # If there were header-level errors, stop before attempting row parsing.
    if errors:
        return [], errors

    # Build ordered lists for row iteration (preserve column order).
    species_dyn = [
        (dyn_objects[(_KIND_SPECIES, key)], header)
        for kind, key, header in dyn
        if kind == _KIND_SPECIES and (_KIND_SPECIES, key) in dyn_objects
    ]
    tractor_dyn = [
        (dyn_objects[(_KIND_TRACTOR, key)], header)
        for kind, key, header in dyn
        if kind == _KIND_TRACTOR and (_KIND_TRACTOR, key) in dyn_objects
    ]

    parsed = []
    for i, row in enumerate(reader, 2):
        # --- Location resolution. ---
        compresa = (row.get(S.CSV_COL_REGION) or '').strip()
        particella = (row.get(S.CSV_COL_PARCEL) or '').strip()

        if not compresa:
            errors.append(S.ERR_CSV_HARVEST_LOCATION.format(i, S.CSV_COL_REGION))
            continue

        if particella:
            parcel = idx.parcels.get((compresa.lower(), particella))
            if parcel is None:
                errors.append(S.ERR_CSV_HARVEST_LOCATION.format(
                    i, f'{compresa}/{particella}',
                ))
                continue
            region = None
        else:
            parcel = None
            region = idx.regions.get(compresa.lower())
            if region is None:
                errors.append(S.ERR_CSV_HARVEST_LOCATION.format(i, compresa))
                continue

        # --- Date. ---
        raw_date = (row.get(S.CSV_COL_DATA) or '').strip()
        try:
            row_date = date_type.fromisoformat(raw_date)
        except ValueError:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_COL_DATA))
            continue

        # --- Crew and product. ---
        crew_name = (row.get(S.CSV_COL_CREW) or '').strip()
        crew = idx.crews.get(crew_name)
        if crew is None:
            errors.append(S.ERR_CSV_UNKNOWN_CREW.format(i, crew_name))
            continue

        product_name = (row.get(S.CSV_COL_PRODUCT) or '').strip()
        product = idx.products.get(product_name)
        if product is None:
            errors.append(S.ERR_CSV_UNKNOWN_PRODUCT.format(i, product_name))
            continue

        # --- Quintals (required). ---
        mass_q = reader.decimal(row.get(S.CSV_COL_QUINTALS))
        if mass_q is None or mass_q < 0:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_COL_QUINTALS))
            continue

        # --- Optional integer fields. ---
        record1, r1_ok = reader.opt_int(row.get(S.CSV_COL_VDP))
        record2, r2_ok = reader.opt_int(row.get(S.CSV_COL_PROT))
        if not (r1_ok and r2_ok):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_VDP}/{S.CSV_COL_PROT}',
            ))
            continue

        # --- Boolean flags (blank → False). ---
        damaged_raw, dmg_ok = reader.opt_bool(row.get(S.CSV_COL_HARVEST_DAMAGED, ''))
        unhealthy_raw, unh_ok = reader.opt_bool(row.get(S.CSV_COL_HARVEST_UNHEALTHY, ''))
        psr_raw, psr_ok = reader.opt_bool(row.get(S.CSV_COL_HARVEST_PSR, ''))
        if not (dmg_ok and unh_ok and psr_ok):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_HARVEST_DAMAGED}/{S.CSV_COL_HARVEST_UNHEALTHY}/{S.CSV_COL_HARVEST_PSR}',
            ))
            continue
        damaged = bool(damaged_raw)
        unhealthy = bool(unhealthy_raw)
        psr = bool(psr_raw)

        note = (row.get(S.CSV_COL_EXTRA_NOTE) or '').strip()

        # --- Species percentages. ---
        species_pcts = []  # [(Species, int_pct)]
        ok = True
        for sp, hdr in species_dyn:
            pct, pct_ok = reader.opt_int(row.get(hdr, ''))
            if not pct_ok:
                errors.append(S.ERR_CSV_ROW_PARSE.format(i, hdr))
                ok = False
                break
            pct = pct or 0
            if pct < 0 or pct > 100:
                errors.append(S.ERR_CSV_ROW_PARSE.format(i, hdr))
                ok = False
                break
            species_pcts.append((sp, pct))
        if not ok:
            continue

        if mass_q > 0:
            sp_sum = sum(pct for _, pct in species_pcts)
            if sp_sum != 100:
                errors.append(S.ERR_CSV_SPECIES_PCT_SUM.format(i, sp_sum))
                continue

        # --- Tractor percentages. ---
        tractor_pcts = []  # [(Tractor, int_pct)]
        ok = True
        for tr, hdr in tractor_dyn:
            pct, pct_ok = reader.opt_int(row.get(hdr, ''))
            if not pct_ok:
                errors.append(S.ERR_CSV_ROW_PARSE.format(i, hdr))
                ok = False
                break
            pct = pct or 0
            if pct < 0 or pct > 100:
                errors.append(S.ERR_CSV_ROW_PARSE.format(i, hdr))
                ok = False
                break
            tractor_pcts.append((tr, pct))
        if not ok:
            continue

        if tractor_pcts:
            tr_sum = sum(pct for _, pct in tractor_pcts)
            if tr_sum not in (0, 100):
                errors.append(S.ERR_CSV_TRACTOR_PCT_SUM.format(i, tr_sum))
                continue

        # --- Volume. ---
        volume_m3 = harvest_volume_m3(
            mass_q,
            ((sp.density, pct) for sp, pct in species_pcts),
        )

        parsed.append({
            'date': row_date,
            'product': product,
            'parcel': parcel,
            'region': region,
            'crew': crew,
            'record1': record1,
            'record2': record2,
            'mass_q': mass_q,
            'volume_m3': volume_m3,
            'damaged': damaged,
            'unhealthy': unhealthy,
            'psr': psr,
            'note': note,
            'species_pcts': species_pcts,
            'tractor_pcts': tractor_pcts,
        })

    return parsed, errors


def apply(parsed: list) -> int:
    """Persist validated rows: one Harvest per row, plus HarvestSpecies /
    HarvestTractor children.  Returns the number of Harvests created.
    """
    with transaction.atomic():
        harvests = [
            Harvest.objects.create(
                date=r['date'],
                product=r['product'],
                parcel=r['parcel'],
                region=r['region'],
                crew=r['crew'],
                record1=r['record1'],
                record2=r['record2'],
                mass_q=r['mass_q'],
                volume_m3=r['volume_m3'],
                damaged=r['damaged'],
                unhealthy=r['unhealthy'],
                psr=r['psr'],
                note=r['note'],
            )
            for r in parsed
        ]

        species_rows = []
        tractor_rows = []
        for harvest, r in zip(harvests, parsed):
            for sp, pct in r['species_pcts']:
                if pct > 0:
                    species_rows.append(
                        HarvestSpecies(harvest=harvest, species=sp, percent=pct)
                    )
            for tr, pct in r['tractor_pcts']:
                if pct > 0:
                    tractor_rows.append(
                        HarvestTractor(harvest=harvest, tractor=tr, percent=pct)
                    )

        if species_rows:
            HarvestSpecies.objects.bulk_create(species_rows)
        if tractor_rows:
            HarvestTractor.objects.bulk_create(tractor_rows)

        mark_stale('prelievi', 'audit')

    return len(harvests)
