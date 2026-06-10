"""Import parcels from particelle.csv, plus synthetic 'X' parcels for
catastrophe harvests.

Idempotent: safe to re-run. Uses get_or_create throughout.
"""

from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base import csv_io
from apps.base.models import Eclass, Parcel, Region
from config import strings as S


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
            reader = csv_io.read(f.read())
            for row in reader:
                region = region_cache[row[S.CSV_COL_REGION]]
                eclass = eclass_cache[row[S.CSV_COL_CLASS]]
                _, was_created = Parcel.objects.get_or_create(
                    name=row[S.CSV_COL_PARCEL],
                    region=region,
                    defaults={
                        'eclass': eclass,
                        'area_ha': reader.decimal(row[S.CSV_COL_AREA_HA]),
                        'ave_age': reader.integer(row[S.CSV_COL_AVE_AGE]),
                        'location_name': row.get(S.CSV_COL_LOCATION, '').strip(),
                        'altitude_min_m': reader.integer(row[S.CSV_COL_ALT_MIN]),
                        'altitude_max_m': reader.integer(row[S.CSV_COL_ALT_MAX]),
                        'aspect': row.get(S.CSV_COL_ASPECT, '').strip(),
                        'grade_pct': reader.integer(row[S.CSV_COL_GRADE_PCT]),
                        'desc_veg': row.get(S.CSV_COL_VEG_DESC, '').strip(),
                        'desc_geo': row.get(S.CSV_COL_GEO_DESC, '').strip(),
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

        from apps.base.digests import mark_all_stale
        mark_all_stale()
