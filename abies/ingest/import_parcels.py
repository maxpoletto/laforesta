#!/usr/bin/env python3
"""Import parcels from particelle.csv, plus synthetic 'X' parcels.

Idempotent: safe to re-run.  Uses get_or_create throughout.
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

from apps.base.models import Eclass, Parcel, Region  # noqa: E402

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------

BOSCO_DATA = Path(__file__).resolve().parent.parent.parent / 'bosco' / 'data'
PARTICELLE_CSV = BOSCO_DATA / 'particelle.csv'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def run():
    region_cache: dict[str, Region] = {r.name: r for r in Region.objects.all()}
    eclass_cache: dict[str, Eclass] = {e.name: e for e in Eclass.objects.all()}

    created = 0
    with open(PARTICELLE_CSV, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            region = region_cache[row['Compresa']]
            eclass = eclass_cache[row['Comparto']]
            _, was_created = Parcel.objects.get_or_create(
                name=row['Particella'],
                region=region,
                defaults={
                    'eclass': eclass,
                    'area_ha': Decimal(row['Area (ha)'].strip()),
                    'ave_age': _int_or_none(row['Età media']),
                    'location_name': row.get('Località', '').strip(),
                    'altitude_min_m': _int_or_none(row['Altitudine min']),
                    'altitude_max_m': _int_or_none(row['Altitudine max']),
                    'aspect': row.get('Esposizione', '').strip(),
                    'grade_pct': _int_or_none(row['Pendenza %']),
                    'desc_veg': row.get('Soprassuolo', '').strip(),
                    'desc_geo': row.get('Stazione', '').strip(),
                },
            )
            if was_created:
                created += 1

    # Synthetic "X" parcels for catastrophe harvests.
    eclass_a = eclass_cache['A']
    for region in region_cache.values():
        _, was_created = Parcel.objects.get_or_create(
            name='X',
            region=region,
            defaults={'eclass': eclass_a, 'area_ha': Decimal('0')},
        )
        if was_created:
            created += 1

    total = Parcel.objects.count()
    print(f'Parcels: {total} ({created} created)')


if __name__ == '__main__':
    run()
