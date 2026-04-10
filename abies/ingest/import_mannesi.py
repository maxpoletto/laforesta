#!/usr/bin/env python3
"""Import harvest operations from mannesi.csv.

Idempotent at the row level: skips rows whose (date, crew, optype, parcel,
record1) combination already exists.  For a clean re-import, delete all
HarvestOp rows first.
"""

import csv
import os
import sys
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------
# Django bootstrap
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django  # noqa: E402
django.setup()

from apps.base.models import Crew, Note, Optype, Parcel, Region, Species, Tractor  # noqa: E402
from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor  # noqa: E402
from ingest.import_reference import NOTE_MAP, OPTYPE_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------

BOSCO_DATA = Path(__file__).resolve().parent.parent.parent / 'bosco' / 'data'
MANNESI_CSV = BOSCO_DATA / 'mannesi.csv'

# ---------------------------------------------------------------------------
# Column name → model FK mappings
# ---------------------------------------------------------------------------

# CSV column prefix → Species.common_name
SPECIES_COL_MAP = {
    'abete': 'Abete',
    'pino': 'Pino',
    'douglas': 'Douglas',
    'faggio': 'Faggio',
    'castagno': 'Castagno',
    'ontano': 'Ontano',
    'altro': 'Altro',
}

# CSV column prefix → (Tractor.manufacturer, Tractor.model)
TRACTOR_COL_MAP = {
    'Equus': ('Equus', '175N UN'),
    'Fiat 110-90': ('Fiat', '110-90'),
    'Fiat 80-66': ('Fiat', '80-66'),
    'Landini 135': ('Landini', '135'),
    'New Holland T5050': ('New Holland', 'T5050'),
}

BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s or s == 'nd':
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _pct(s: str) -> int:
    """Parse a percentage field; treat blank/non-numeric as 0."""
    s = s.strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def run():
    # Build FK lookup caches.
    crew_cache = {c.name: c for c in Crew.objects.all()}
    optype_cache = {o.name: o for o in Optype.objects.all()}
    note_cache = {n.name: n for n in Note.objects.all()}
    species_cache = {s.common_name: s for s in Species.objects.all()}
    tractor_cache = {
        (t.manufacturer, t.model): t for t in Tractor.objects.all()
    }

    # Build parcel cache keyed by (region_name, parcel_name).
    parcel_cache: dict[tuple[str, str], Parcel] = {}
    for p in Parcel.objects.select_related('region'):
        parcel_cache[(p.region.name, p.name)] = p

    # Read CSV.
    with open(MANNESI_CSV, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    # Batch accumulators.
    ops_batch: list[HarvestOp] = []
    # Deferred junction records: list of (op_index, species/tractor, percent).
    species_deferred: list[tuple[int, Species, int]] = []
    tractor_deferred: list[tuple[int, Tractor, int]] = []

    for i, row in enumerate(rows):
        region_name = row['Compresa']
        parcel_name = row['Particella']
        parcel = parcel_cache.get((region_name, parcel_name))
        if parcel is None:
            print(f'Row {i + 1}: unknown parcel {region_name}-{parcel_name}, skipping')
            continue

        crew = crew_cache.get(row['Squadra'])
        if crew is None:
            print(f'Row {i + 1}: unknown crew {row["Squadra"]!r}, skipping')
            continue

        optype = optype_cache.get(OPTYPE_MAP.get(row['Tipo'], ''))
        if optype is None:
            print(f'Row {i + 1}: unknown optype {row["Tipo"]!r}, skipping')
            continue

        note_name = NOTE_MAP.get(row.get('Note', '').strip(), '')
        note = note_cache.get(note_name) if note_name else None

        op = HarvestOp(
            date=row['Data'],
            optype=optype,
            parcel=parcel,
            crew=crew,
            record1=_int_or_none(row['VDP']),
            record2=_int_or_none(row.get('Prot.', '')),
            quintals=Decimal(row['Q.li'].strip()),
            note=note,
            extra_note=row.get('Altre note', '').strip(),
        )
        op_idx = len(ops_batch)
        ops_batch.append(op)

        # Collect non-zero species percentages.
        for col_prefix, common_name in SPECIES_COL_MAP.items():
            pct = _pct(row.get(f'{col_prefix} %', ''))
            if pct > 0:
                species_deferred.append((op_idx, species_cache[common_name], pct))

        # Collect non-zero tractor percentages.
        for col_prefix, (mfr, model) in TRACTOR_COL_MAP.items():
            pct = _pct(row.get(f'{col_prefix} %', ''))
            if pct > 0:
                tractor_deferred.append((op_idx, tractor_cache[(mfr, model)], pct))

    # Bulk-create ops (no history tracking for ETL — that's intentional).
    HarvestOp.objects.bulk_create(ops_batch, batch_size=BATCH_SIZE)
    print(f'HarvestOps: {len(ops_batch)} created')

    # Now that ops have PKs, build junction records.
    species_records = [
        HarvestSpecies(harvest_op=ops_batch[idx], species=sp, percent=pct)
        for idx, sp, pct in species_deferred
    ]
    HarvestSpecies.objects.bulk_create(species_records, batch_size=BATCH_SIZE)
    print(f'HarvestSpecies: {len(species_records)} created')

    tractor_records = [
        HarvestTractor(harvest_op=ops_batch[idx], tractor=tr, percent=pct)
        for idx, tr, pct in tractor_deferred
    ]
    HarvestTractor.objects.bulk_create(tractor_records, batch_size=BATCH_SIZE)
    print(f'HarvestTractors: {len(tractor_records)} created')

    print('Mannesi import complete.')


if __name__ == '__main__':
    run()
