"""manage.py bootstrap <data_dir> [--check]

Load a canonical CSV data dir (see the import-contract spec) into an EMPTY
instance.  Atomic-transaction load: every present file is validated and its valid
rows applied in dependency order inside one transaction; errors are accumulated
across all files; on any error — or with --check — the whole transaction rolls
back, so --check never persists and a failed load never leaves a half-loaded
instance.

Load-into-empty-only: the command refuses if any forestry domain is non-empty
(it has no wipe, so it never deletes data it cannot restore).  Loaded: reference,
parcels, containers (sample_grids/harvest_plans/surveys), and bulk
(sample_areas/sampled-trees/hypso).  preserved-trees / harvest_plan_items /
harvests are reported as not yet supported (later increments).
"""

from dataclasses import dataclass, field
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.base import (
    csv_containers as cc, csv_io, csv_parcels, csv_reference, hypsometry,
)
from apps.base.models import (
    Crew, Eclass, HarvestDetail, HarvestPlan, HarvestPlanItem,
    HypsoParamSet, HypsoParamSource, Parcel, Product, Region, Sample,
    SampleArea, SampleGrid, Species, Survey, Tractor, Tree, TreeMark,
)
from apps.campionamenti import csv_grid, csv_trees
from apps.prelievi.models import Harvest
from config import strings as S

# Comprehensive emptiness guard: if ANY of these holds a row, the instance is not
# empty and bootstrap refuses (it has no wipe).  Listed verbose_names are shown.
GUARD_MODELS = [
    Region, Eclass, Crew, Tractor, Species, Product, Parcel, SampleGrid,
    SampleArea, Survey, Sample, Tree, TreeMark, HarvestPlan, HarvestPlanItem,
    HarvestDetail, Harvest, HypsoParamSet,
]

DEFERRED_FILES = [
    S.CSV_FILE_PRESERVED_TREES, S.CSV_FILE_HARVEST_PLAN_ITEMS, S.CSV_FILE_HARVESTS,
]


@dataclass
class FileReport:
    name: str
    required: bool
    present: bool = False
    loaded: int = 0
    errors: list = field(default_factory=list)
    note: str = ''


class Command(BaseCommand):
    help = "Load a canonical CSV data dir into this (empty) instance."

    def add_arguments(self, parser):
        parser.add_argument('data_dir', type=Path)
        parser.add_argument('--check', action='store_true',
                            help='Validate and report only; persist nothing.')

    def handle(self, *args, data_dir, check, **opts):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        populated = [str(m._meta.verbose_name) for m in GUARD_MODELS
                     if m.objects.exists()]
        if populated:
            raise CommandError(
                S.ERR_BOOTSTRAP_NOT_EMPTY.format(', '.join(populated)))

        reports = []
        try:
            with transaction.atomic():
                survey_dates: dict = {}
                for table in csv_reference.ALL_TABLES:
                    reports.append(self._load_reftable(
                        data_dir, table, required=table in (
                            csv_reference.REGIONS, csv_reference.ECLASSES)))
                reports.append(self._load_parcels(data_dir))
                reports.append(self._load_reftable(data_dir, cc.SAMPLE_GRIDS, required=False))
                reports.append(self._load_reftable(data_dir, cc.HARVEST_PLANS, required=False))
                reports.append(self._load_surveys(data_dir, survey_dates))
                reports.append(self._load_sample_areas(data_dir))
                reports.append(self._load_sampled_trees(data_dir, survey_dates))
                reports.append(self._load_hypso(data_dir))
                reports.extend(self._note_deferred(data_dir))
                if sum(len(r.errors) for r in reports) or check:
                    transaction.set_rollback(True)
        except Exception as exc:
            # An unexpected error (e.g. a DB constraint a core's validate missed)
            # aborts and rolls back the transaction; surface it in the report
            # rather than dying with a traceback.
            reports.append(FileReport(name=S.BOOTSTRAP_INTERNAL, required=False,
                                      errors=[str(exc)]))

        total_errors = sum(len(r.errors) for r in reports)
        self._print_report(reports, check, total_errors)
        if total_errors:
            raise CommandError(S.ERR_BOOTSTRAP_FAILED.format(total_errors))

    # --- file reading -------------------------------------------------------

    def _open(self, data_dir, filename, required, required_cols=None):
        """Return (reader, report); reader is None if absent or unreadable."""
        report = FileReport(name=filename, required=required)
        path = data_dir / filename
        if not path.is_file():
            if required:
                report.errors.append(
                    S.ERR_BOOTSTRAP_REQUIRED_FILE.format(filename))
            else:
                report.note = S.BOOTSTRAP_OPTIONAL_SKIPPED
            return None, report
        report.present = True
        try:
            reader = csv_io.read(path.read_text(encoding='utf-8-sig'),
                                 required_cols=required_cols)
        except csv_io.CsvError as exc:
            report.errors.append(f'{filename}: {exc}')
            return None, report
        return reader, report

    # --- reference / container RefTables ------------------------------------

    def _load_reftable(self, data_dir, table, required):
        reader, report = self._open(data_dir, f'{table.name}.csv', required)
        if reader is None:
            return report
        cols, missing = csv_reference.resolve_columns(table, reader.fieldnames)
        if missing:
            report.errors.append(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
            return report
        parsed, errors = csv_reference.validate_rows(table, reader, cols)
        report.errors.extend(errors)
        if parsed:
            created, _ = csv_reference.apply(table, parsed)
            report.loaded = created
        return report

    # --- parcels ------------------------------------------------------------

    def _load_parcels(self, data_dir):
        reader, report = self._open(
            data_dir, S.CSV_FILE_PARCELS, required=True,
            required_cols=csv_parcels.PARCEL_CSV_REQUIRED)
        if reader is None:
            return report
        parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
        report.errors.extend(errors)
        report.loaded = csv_parcels.apply(parsed)
        return report

    # --- surveys ------------------------------------------------------------

    def _load_surveys(self, data_dir, survey_dates):
        reader, report = self._open(
            data_dir, S.CSV_FILE_SURVEYS, required=False,
            required_cols=cc.SURVEY_CSV_REQUIRED)
        if reader is None:
            return report
        parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
        report.errors.extend(errors)
        report.loaded = cc.apply_surveys(parsed)
        for d in parsed:
            survey_dates[d['name']] = d['date']
        return report

    # --- bulk: sample areas (grouped by grid) -------------------------------

    def _load_sample_areas(self, data_dir):
        reader, report = self._open(
            data_dir, S.CSV_FILE_SAMPLE_AREAS, required=False,
            required_cols=[S.CSV_COL_GRID] + csv_grid.GRID_CSV_REQUIRED)
        if reader is None:
            return report
        cols, missing = csv_grid.resolve_columns(reader.fieldnames)
        if missing:  # same columns for every group → resolve once
            report.errors.append(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
            return report
        grids = {g.name: g for g in SampleGrid.objects.all()}
        for grid_name, rows in _group(reader, S.CSV_COL_GRID).items():
            grid = grids.get(grid_name)
            if grid is None:
                report.errors.append(S.ERR_BOOTSTRAP_UNKNOWN_GRID.format(
                    S.CSV_FILE_SAMPLE_AREAS, grid_name))
                continue
            sub = csv_io.CsvReader(rows, reader.delimiter, reader.fieldnames)
            parsed, errors = csv_grid.validate_rows(sub, cols, csv_grid.db_indexes(grid))
            report.errors.extend(errors)
            if parsed:
                csv_grid.apply(grid, parsed)
                report.loaded += len(parsed)
        return report

    # --- bulk: sampled trees (grouped by survey) ----------------------------

    def _load_sampled_trees(self, data_dir, survey_dates):
        reader, report = self._open(
            data_dir, S.CSV_FILE_SAMPLED_TREES, required=False,
            required_cols=[S.CSV_COL_SURVEY] + csv_trees.TREE_CSV_REQUIRED)
        if reader is None:
            return report
        surveys = {s.name: s for s in Survey.objects.all()}
        has_date = S.CSV_COL_DATA in (reader.fieldnames or [])
        for survey_name, rows in _group(reader, S.CSV_COL_SURVEY).items():
            survey = surveys.get(survey_name)
            if survey is None:
                report.errors.append(S.ERR_BOOTSTRAP_UNKNOWN_SURVEY.format(
                    S.CSV_FILE_SAMPLED_TREES, survey_name))
                continue
            sub = csv_io.CsvReader(rows, reader.delimiter, reader.fieldnames)
            parsed, errors = csv_trees.validate_rows(
                sub, csv_trees.db_indexes(survey),
                has_date_column=has_date,
                default_date=survey_dates.get(survey_name))
            report.errors.extend(errors)
            if parsed:
                report.loaded += csv_trees.apply(survey, parsed)['n_trees']
        return report

    # --- bulk: hypso --------------------------------------------------------

    def _load_hypso(self, data_dir):
        report = FileReport(name=S.CSV_FILE_HYPSO, required=False)
        path = data_dir / S.CSV_FILE_HYPSO
        if not path.is_file():
            report.note = S.BOOTSTRAP_OPTIONAL_SKIPPED
            return report
        report.present = True
        rows, errors = hypsometry.parse_param_csv(path.read_text(encoding='utf-8-sig'))
        report.errors.extend(errors)
        if rows and not errors:
            hypsometry.replace_active_set(
                rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[])
            report.loaded = len(rows)
        return report

    # --- deferred -----------------------------------------------------------

    def _note_deferred(self, data_dir):
        out = []
        for filename in DEFERRED_FILES:
            if (data_dir / filename).is_file():
                out.append(FileReport(name=filename, required=False, present=True,
                                      note=S.BOOTSTRAP_NOT_SUPPORTED))
        return out

    # --- report -------------------------------------------------------------

    def _print_report(self, reports, check, total_errors):
        for r in reports:
            loaded = (S.BOOTSTRAP_LOADED.format(r.loaded) if r.present
                      else S.BOOTSTRAP_ABSENT)
            self.stdout.write(f'  {r.name:<26} {r.note or loaded}')
            for err in r.errors:
                self.stdout.write(f'      ! {err}')
        if check:
            self.stdout.write(S.BOOTSTRAP_CHECK_NOTICE)
        elif not total_errors:
            self.stdout.write(S.BOOTSTRAP_DONE)


def _group(reader, col):
    """Partition reader rows by the (stripped) value of `col`."""
    groups: dict = {}
    for row in reader:
        groups.setdefault((row.get(col) or '').strip(), []).append(row)
    return groups
