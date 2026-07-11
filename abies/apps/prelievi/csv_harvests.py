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

import hashlib
import json

from dataclasses import dataclass
from datetime import date as date_type
from decimal import Decimal

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import Crew, Parcel, Product, Region, Species, Tractor
from apps.prelievi.models import (
    Harvest, HarvestSpecies, HarvestTractor, harvest_volume_m3,
)
from apps.prelievi.harvest_validation import (
    percentages_sum_to_0_or_100, percentages_sum_to_100,
    valid_harvest_mass, valid_percentage,
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
_FIELD_SOURCE_ROW = 'source_row'
_FINGERPRINT_VERSION = 'v1'
_MASS_Q_QUANTUM = Decimal('0.01')
_VOLUME_M3_QUANTUM = Decimal('0.001')


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
        if not valid_harvest_mass(mass_q):
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
            if not valid_percentage(pct):
                errors.append(S.ERR_CSV_ROW_PARSE.format(i, hdr))
                ok = False
                break
            species_pcts.append((sp, pct))
        if not ok:
            continue

        sp_sum = sum(pct for _, pct in species_pcts)
        if not percentages_sum_to_100(pct for _, pct in species_pcts):
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
            if not valid_percentage(pct):
                errors.append(S.ERR_CSV_ROW_PARSE.format(i, hdr))
                ok = False
                break
            tractor_pcts.append((tr, pct))
        if not ok:
            continue

        tr_sum = sum(pct for _, pct in tractor_pcts)
        if not percentages_sum_to_0_or_100(pct for _, pct in tractor_pcts):
            errors.append(S.ERR_CSV_TRACTOR_PCT_SUM.format(i, tr_sum))
            continue

        # --- Volume. ---
        volume_m3 = harvest_volume_m3(
            mass_q,
            ((sp.density, pct) for sp, pct in species_pcts),
        )

        parsed.append({
            _FIELD_SOURCE_ROW: i,
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


def harvest_import_fingerprint(row: dict) -> str:
    data = {
        'version': _FINGERPRINT_VERSION,
        _FIELD_SOURCE_ROW: row.get(_FIELD_SOURCE_ROW),
        'date': row['date'].isoformat(),
        'product_id': row['product'].id,
        'parcel_id': row['parcel'].id if row['parcel'] is not None else None,
        'region_id': row['region'].id if row['region'] is not None else None,
        'crew_id': row['crew'].id,
        'record1': row['record1'],
        'record2': row['record2'],
        'mass_q': _decimal_fingerprint(row['mass_q'], _MASS_Q_QUANTUM),
        'volume_m3': _decimal_fingerprint(row['volume_m3'], _VOLUME_M3_QUANTUM),
        'damaged': row['damaged'],
        'unhealthy': row['unhealthy'],
        'psr': row['psr'],
        'note': row['note'],
        'species_pcts': [(sp.id, pct) for sp, pct in row['species_pcts']],
        'tractor_pcts': [(tr.id, pct) for tr, pct in row['tractor_pcts']],
    }
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return f'{_FINGERPRINT_VERSION}:{digest}'


def _decimal_fingerprint(value: Decimal, quantum: Decimal) -> str:
    return format(value.quantize(quantum), 'f')


def apply(parsed: list) -> int:
    """Persist validated rows idempotently: one new Harvest per new row
    fingerprint, plus HarvestSpecies / HarvestTractor children.

    Returns the number of Harvests created.
    """
    with transaction.atomic():
        rows = [(harvest_import_fingerprint(r), r) for r in parsed]
        fingerprints = [fingerprint for fingerprint, _row in rows]
        existing = set(
            Harvest.objects
            .filter(import_fingerprint__in=fingerprints)
            .values_list('import_fingerprint', flat=True)
        )
        rows = [
            (fingerprint, r)
            for fingerprint, r in rows
            if fingerprint not in existing
        ]

        harvests = [
            Harvest.objects.create(
                import_fingerprint=fingerprint,
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
            for fingerprint, r in rows
        ]

        species_rows = []
        tractor_rows = []
        for harvest, (_fingerprint, r) in zip(harvests, rows):
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

        if harvests:
            mark_stale('prelievi', 'audit')

    return len(harvests)
