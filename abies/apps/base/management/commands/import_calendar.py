"""Import fustaia and ceduo harvest calendars into a HarvestPlan.

Reads piano_fustaia.csv and piano_ceduo.csv from <csv_dir>,
creates a HarvestPlan named "Piano 2026-2040", and populates it with
HarvestPlanItems.

Idempotent: deletes and recreates the plan on each run.
"""

import csv
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

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
        ceduo_csv = data_dir / 'ceduo_ceduo.csv'
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
            for row in csv.DictReader(f):
                parcel = parcel_cache.get(
                    (row[S.CSV_COL_COMPRESA], row[S.CSV_COL_PARTICELLA])
                )
                if parcel is None:
                    continue
                HarvestPlanItem.objects.create(
                    harvest_plan=plan,
                    parcel=parcel,
                    year_planned=int(row[S.CSV_COL_ANNO]),
                    volume_planned_m3=Decimal(row[S.CSV_COL_PRELIEVO_M3]),
                )
                n += 1
        return n

    def _import_ceduo(self, csv_path, plan, parcel_cache):
        n = 0
        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                parcel = parcel_cache.get(
                    (row[S.CSV_COL_COMPRESA], row[S.CSV_COL_PARTICELLA])
                )
                if parcel is None:
                    continue
                HarvestPlanItem.objects.create(
                    harvest_plan=plan,
                    parcel=parcel,
                    year_planned=int(row[S.CSV_COL_ANNO]),
                    intervention_area_ha=Decimal(
                        row[S.CSV_COL_SUPERFICIE_HA]
                    ),
                    note=row.get(S.CSV_COL_NOTE, '').strip(),
                )
                n += 1
        return n
