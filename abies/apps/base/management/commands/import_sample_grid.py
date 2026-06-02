"""Import sample-area locations into one SampleGrid.

Reads bosco/data/aree-di-saggio.csv (header:
Compresa,Particella,CP,Area saggio,UTMLon,UTMLat,GaussLon,GaussLat,Lon,Lat,Quota).

Idempotent: re-running produces no duplicates.  Creates one
SampleGrid named "Aree di saggio PDG 2026" with one SampleArea row
per CSV row.  The CSV lacks `Raggio`, so `r_m` defaults to 12 (per
spec).

Area numbers must be unique per (grid, compresa); the loader aborts
(rolling back) if the CSV assigns one number to two particelle of the
same compresa.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.base import csv_io
from apps.base.models import Parcel, Region, SampleArea, SampleGrid
from config import strings as S
from config.constants import FIELD_ALTITUDE_M, FIELD_LAT, FIELD_LON, FIELD_NOTE, FIELD_R_M


GRID_NAME = 'Aree di saggio PDG 2026'
GRID_DESC = ('Griglia di aree di saggio generata per il PDG 2026, '
             'caricata dal file aree-di-saggio.csv.')
DEFAULT_RADIUS_M = 12


def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


class Command(BaseCommand):
    help = "Import sample areas from aree-di-saggio.csv into one SampleGrid."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help='Directory containing aree-di-saggio.csv.',
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        csv_path = data_dir / 'aree-di-saggio.csv'
        if not csv_path.is_file():
            raise CommandError(f'{csv_path} not found')

        region_cache = {r.name: r for r in Region.objects.all()}
        parcel_cache = {
            (p.region.name, p.name): p
            for p in Parcel.objects.select_related('region')
        }
        if not parcel_cache:
            raise CommandError(
                'Reference + parcel data must be loaded first. '
                'Run import_reference and import_parcels before '
                'import_sample_grid.'
            )

        with open(csv_path, encoding='utf-8-sig') as f:
            rows = csv_io.read(f.read()).rows

        with transaction.atomic():
            grid, created = SampleGrid.objects.get_or_create(
                name=GRID_NAME,
                defaults={'description': GRID_DESC},
            )
            if created:
                self.stdout.write(f'Created grid: {GRID_NAME}')

            # Area numbers must be unique per (grid, region) — see SampleArea.
            # `owner` maps (region_id, number) → the parcel that first claimed
            # it, so we can detect a CSV that hands one number to two parcels of
            # the same compresa.  Pre-loaded from the grid for idempotent
            # re-runs.
            owner: dict[tuple[int, str], int] = {
                (region_id, num): pid
                for pid, region_id, num in (
                    SampleArea.objects.filter(sample_grid=grid)
                    .values_list('parcel_id', 'parcel__region_id', 'number')
                )
            }

            n_created = 0
            n_skipped = 0
            collisions = []
            for i, row in enumerate(rows, 1):
                region_name = row[S.CSV_COL_COMPRESA]
                parcel_name = row[S.CSV_COL_PARTICELLA]
                parcel = parcel_cache.get((region_name, parcel_name))
                if parcel is None:
                    self.stdout.write(
                        f'Row {i}: unknown parcel '
                        f'{region_name}-{parcel_name}, skipping'
                    )
                    n_skipped += 1
                    continue
                number = row[S.CSV_COL_AREA_SAGGIO].strip()
                key = (parcel.region_id, number)
                if owner.setdefault(key, parcel.id) != parcel.id:
                    collisions.append(
                        f'Row {i}: area number {number} in compresa '
                        f'{region_name} is already used by another particella'
                    )
                    continue
                obj, was_created = SampleArea.objects.get_or_create(
                    sample_grid=grid, parcel=parcel, number=number,
                    defaults={
                        FIELD_LAT: float(row[S.CSV_COL_LAT]),
                        FIELD_LON: float(row[S.CSV_COL_LON]),
                        FIELD_ALTITUDE_M: _int_or_none(row[S.CSV_COL_QUOTA]),
                        FIELD_R_M: DEFAULT_RADIUS_M,
                        FIELD_NOTE: '',
                    },
                )
                if was_created:
                    n_created += 1

            if collisions:
                raise CommandError(
                    'Area numbers must be unique per compresa; '
                    f'found {len(collisions)} collision(s):\n  '
                    + '\n  '.join(collisions)
                )

        self.stdout.write(
            f'SampleAreas: {n_created} created, {n_skipped} skipped'
        )

        from apps.base.digests import mark_all_stale
        mark_all_stale()
