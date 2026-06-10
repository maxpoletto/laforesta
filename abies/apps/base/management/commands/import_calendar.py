"""Import harvest calendars into a HarvestPlan.

Reads piano_fustaia.csv and piano_ceduo.csv from <csv_dir>, creates a
HarvestPlan named "Piano 2026-2040", and populates it with
HarvestPlanItems.

Idempotent: deletes and recreates the plan on each run.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.base import csv_io
from apps.base.models import HarvestPlan, HarvestPlanItem, Parcel
from config import strings as S

PLAN_NAME = 'Piano 2026-2040'
PLAN_YEAR_START = 2026
PLAN_YEAR_END = 2040


class Command(BaseCommand):
    help = "Import fustaia/ceduo calendars into a HarvestPlan."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help="Directory containing piano.csv and ceduo.csv.",
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        piano_csv = data_dir / 'piano_fustaia.csv'
        ceduo_csv = data_dir / 'piano_ceduo.csv'
        if not piano_csv.is_file():
            raise CommandError(f'{piano_csv} not found')
        if not ceduo_csv.is_file():
            raise CommandError(f'{ceduo_csv} not found')

        parcel_cache = {
            (p.region.name, p.name): p
            for p in Parcel.objects.select_related('region')
        }
        if not parcel_cache:
            raise CommandError(
                'Reference + parcel data must be loaded first.'
            )

        with transaction.atomic():
            HarvestPlan.objects.filter(name=PLAN_NAME).delete()
            plan = HarvestPlan.objects.create(
                name=PLAN_NAME,
                year_start=PLAN_YEAR_START,
                year_end=PLAN_YEAR_END,
            )

            n_fustaia = self._import_fustaia(piano_csv, plan, parcel_cache)
            n_ceduo = self._import_ceduo(ceduo_csv, plan, parcel_cache)

        from apps.base.digests import mark_all_stale
        mark_all_stale()

        self.stdout.write(
            f'Calendar: plan "{PLAN_NAME}" with '
            f'{n_fustaia} fustaia + {n_ceduo} ceduo items'
        )

    def _import_fustaia(self, csv_path, plan, parcel_cache):
        n = 0
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv_io.read(f.read())
            for row in reader:
                parcel = parcel_cache.get(
                    (row[S.CSV_COL_REGION], row[S.CSV_COL_PARCEL])
                )
                if parcel is None:
                    continue
                HarvestPlanItem.objects.create(
                    harvest_plan=plan,
                    parcel=parcel,
                    year_planned=reader.integer(row[S.CSV_COL_YEAR]),
                    volume_planned_m3=reader.decimal(row[S.CSV_COL_HARVEST_M3]),
                )
                n += 1
        return n

    def _import_ceduo(self, csv_path, plan, parcel_cache):
        n = 0
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv_io.read(f.read())
            for row in reader:
                parcel = parcel_cache.get(
                    (row[S.CSV_COL_REGION], row[S.CSV_COL_PARCEL])
                )
                if parcel is None:
                    continue
                HarvestPlanItem.objects.create(
                    harvest_plan=plan,
                    parcel=parcel,
                    year_planned=reader.integer(row[S.CSV_COL_YEAR]),
                    intervention_area_ha=reader.decimal(
                        row[S.CSV_COL_SURFACE_HA]
                    ),
                    note=row.get(S.CSV_COL_NOTE, '').strip(),
                )
                n += 1
        return n
