"""Import parcels from particelle.csv, plus synthetic 'X' parcels for
catastrophe harvests.

Idempotent: safe to re-run. Uses get_or_create throughout.
"""

import csv
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base.models import Eclass, Parcel, Region


def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


class Command(BaseCommand):
    help = "Import parcels from particelle.csv in <data_dir>."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help="Directory containing particelle.csv.",
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        particelle_csv = data_dir / 'particelle.csv'
        if not particelle_csv.is_file():
            raise CommandError(f'{particelle_csv} not found')

        region_cache = {r.name: r for r in Region.objects.all()}
        eclass_cache = {e.name: e for e in Eclass.objects.all()}
        if not region_cache or not eclass_cache:
            raise CommandError(
                'Regions and eclasses must be imported first. '
                'Run import_reference before import_parcels.'
            )

        created = 0
        with open(particelle_csv, encoding='utf-8-sig') as f:
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

        # Synthetic "X" parcels for catastrophe harvests, one per region.
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
        self.stdout.write(f'Parcels: {total} ({created} created)')
