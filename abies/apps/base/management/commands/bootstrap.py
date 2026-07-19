"""manage.py bootstrap <data_dir> [--check]

Load a canonical CSV data dir (see docs/bootstrap.md) into an EMPTY
instance.  Atomic-transaction load: every present file is validated; clean
files are applied in dependency order inside one transaction; errors are
accumulated across all files; on any error — or with --check — the whole
transaction rolls
back, so --check never persists and a failed load never leaves a half-loaded
instance.

Load-into-empty-only: the command refuses if any forestry domain is non-empty
(it has no wipe, so it never deletes data it cannot restore).  Loaded: reference,
parcels, containers (sample_grids/harvest_plans/surveys), bulk
(sample_areas/sampled-trees/hypso), harvest_plan_items, preserved-trees,
and harvests.
"""

from dataclasses import dataclass, field
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import OperationalError, ProgrammingError, transaction

from apps.base import (
    csv_containers as cc, csv_io, csv_parcels, csv_reference, hypsometry, refdata,
)
from apps.base.models import (
    Crew, Eclass, HarvestDetail, HarvestPlan, HarvestPlanItem,
    HypsoParamSet, HypsoParamSource, Parcel, Product, Region, Sample,
    SampleArea, SampleGrid, Species, Survey, Tractor, Tree, TreeMark,
    TreePreserved, TreeSample,
)
from apps.campionamenti import csv_grid, csv_trees
from apps.campionamenti import csv_preserved
from apps.piano_di_taglio import csv_plan
from apps.prelievi import csv_harvests
from apps.prelievi.models import Harvest
from config import strings as S
from config.constants import (
    FIELD_COMMON_NAME, FIELD_DENSITY, FIELD_LATIN_NAME, FIELD_MINOR,
    FIELD_NAME, FIELD_PRESSLER_DEFAULT, FIELD_SORT_ORDER,
)

# Comprehensive emptiness guard: if ANY of these holds a row, the instance is not
# empty and bootstrap refuses (it has no wipe).  Listed verbose_names are shown.
GUARD_MODELS = [
    Region, Eclass, Crew, Tractor, Species, Product, Parcel, SampleGrid,
    SampleArea, Survey, Sample, Tree, TreeMark, TreePreserved, TreeSample,
    HarvestPlan,
    HarvestPlanItem, HarvestDetail, Harvest, HypsoParamSet,
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
        try:
            populated = [str(m._meta.verbose_name) for m in GUARD_MODELS
                         if m.objects.exists()]
        except (OperationalError, ProgrammingError) as exc:
            raise CommandError(S.ERR_BOOTSTRAP_SCHEMA.format(exc)) from exc
        if populated:
            raise CommandError(
                S.ERR_BOOTSTRAP_NOT_EMPTY.format(', '.join(populated)))

        reports = []
        try:
            with transaction.atomic():
                survey_dates: dict = {}
                for table in csv_reference.ALL_TABLES:
                    report = self._load_reftable(
                        data_dir, table, required=table in (
                            csv_reference.REGIONS, csv_reference.ECLASSES))
                    if not report.present and not report.errors:
                        self._seed_default(table, report)
                    reports.append(report)
                reports.append(self._load_parcels(data_dir))
                reports.append(self._load_reftable(data_dir, cc.SAMPLE_GRIDS, required=False))
                reports.append(self._load_reftable(data_dir, cc.HARVEST_PLANS, required=False))
                reports.append(self._load_surveys(data_dir, survey_dates))
                reports.append(self._load_sample_areas(data_dir))
                reports.append(self._load_sampled_trees(data_dir, survey_dates))
                reports.append(self._load_hypso(data_dir))
                reports.append(self._load_harvest_plan_items(data_dir))
                reports.append(self._load_preserved_trees(data_dir))
                reports.append(self._load_harvests(data_dir))
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

    def _missing_columns(self, report, missing):
        if not missing:
            return False
        report.errors.append(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
        return True

    def _apply_validated(self, report, parsed, errors, apply_fn):
        """Attach validation errors and apply only if this file is clean."""
        report.errors.extend(errors)
        if report.errors or not parsed:
            return False
        report.loaded = apply_fn(parsed)
        return True

    def _load_grouped(self, reader, report, *, group_col, targets, unknown_error,
                      validate_group, apply_group):
        """Validate grouped child rows, then apply all groups only if clean."""
        groups = []
        for group_name, rows in _group(reader, group_col).items():
            target = targets.get(group_name)
            if target is None:
                report.errors.append(unknown_error(group_name))
                continue
            sub = csv_io.CsvReader(rows, reader.delimiter, reader.fieldnames)
            parsed, errors = validate_group(target, sub, group_name)
            report.errors.extend(errors)
            if parsed:
                groups.append((target, parsed))
        if report.errors:
            return report
        for target, parsed in groups:
            report.loaded += apply_group(target, parsed)
        return report

    # --- reference / container RefTables ------------------------------------

    def _load_reftable(self, data_dir, table, required):
        reader, report = self._open(data_dir, f'{table.name}.csv', required)
        if reader is None:
            return report
        cols, missing = csv_reference.resolve_columns(table, reader.fieldnames)
        if self._missing_columns(report, missing):
            return report
        parsed, errors = csv_reference.validate_rows(table, reader, cols)
        self._apply_validated(
            report, parsed, errors,
            lambda rows: csv_reference.apply(table, rows)[0],
        )
        return report

    def _seed_default(self, table, report):
        """Seed species/products from the in-repo canonical defaults when their
        optional CSV is absent.  This mainly serves minimal or hand-built data
        dirs.  Other absent reference tables have no default and are left
        empty."""
        if table is csv_reference.SPECIES:
            parsed = [
                {FIELD_COMMON_NAME: common, FIELD_LATIN_NAME: latin,
                 FIELD_SORT_ORDER: order, FIELD_DENSITY: density,
                 FIELD_PRESSLER_DEFAULT: pressler_default, FIELD_MINOR: minor}
                for common, latin, order, density, pressler_default, minor
                in refdata.load_species()
            ]
        elif table is csv_reference.PRODUCTS:
            parsed = [{FIELD_NAME: name}
                      for name in dict.fromkeys(refdata.PRODUCT_MAP.values())]
        else:
            return
        report.loaded, _ = csv_reference.apply(table, parsed)
        report.note = S.BOOTSTRAP_DEFAULT_SEEDED

    # --- parcels ------------------------------------------------------------

    def _load_parcels(self, data_dir):
        reader, report = self._open(
            data_dir, S.CSV_FILE_PARCELS, required=True,
            required_cols=csv_parcels.PARCEL_CSV_REQUIRED)
        if reader is None:
            return report
        parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
        self._apply_validated(report, parsed, errors, csv_parcels.apply)
        return report

    # --- surveys ------------------------------------------------------------

    def _load_surveys(self, data_dir, survey_dates):
        reader, report = self._open(
            data_dir, S.CSV_FILE_SURVEYS, required=False,
            required_cols=cc.SURVEY_CSV_REQUIRED)
        if reader is None:
            return report
        parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
        if self._apply_validated(report, parsed, errors, cc.apply_surveys):
            for d in parsed:
                survey_dates[d['name']] = d['date']
        return report

    # --- bulk: sample areas (grouped by grid) -------------------------------

    def _load_sample_areas(self, data_dir):
        filename = S.CSV_FILE_SAMPLE_AREAS
        reader, report = self._open(
            data_dir, filename, required=False,
            required_cols=[S.CSV_COL_GRID] + csv_grid.GRID_CSV_REQUIRED)
        if reader is None:
            return report
        cols, missing = csv_grid.resolve_columns(reader.fieldnames)
        if self._missing_columns(report, missing):
            return report
        grids = {g.name: g for g in SampleGrid.objects.all()}
        return self._load_grouped(
            reader, report,
            group_col=S.CSV_COL_GRID,
            targets=grids,
            unknown_error=lambda name: S.ERR_BOOTSTRAP_UNKNOWN_GRID.format(
                filename, name),
            validate_group=lambda grid, sub, _name: csv_grid.validate_rows(
                sub, cols, csv_grid.db_indexes(grid)),
            apply_group=lambda grid, parsed: (
                csv_grid.apply(grid, parsed) or len(parsed)
            ),
        )

    # --- bulk: sampled trees (grouped by survey) ----------------------------

    def _load_sampled_trees(self, data_dir, survey_dates):
        filename = S.CSV_FILE_SAMPLED_TREES
        reader, report = self._open(
            data_dir, filename, required=False,
            required_cols=[S.CSV_COL_SURVEY] + csv_trees.TREE_CSV_REQUIRED)
        if reader is None:
            return report
        surveys = {s.name: s for s in Survey.objects.all()}
        has_date = S.CSV_COL_DATA in (reader.fieldnames or [])
        return self._load_grouped(
            reader, report,
            group_col=S.CSV_COL_SURVEY,
            targets=surveys,
            unknown_error=lambda name: S.ERR_BOOTSTRAP_UNKNOWN_SURVEY.format(
                filename, name),
            validate_group=lambda survey, sub, name: csv_trees.validate_rows(
                sub, csv_trees.db_indexes(survey),
                has_date_column=has_date,
                default_date=survey_dates.get(name)),
            apply_group=lambda survey, parsed: (
                csv_trees.apply(survey, parsed)['n_trees']
            ),
        )

    # --- bulk: hypso --------------------------------------------------------

    def _load_hypso(self, data_dir):
        report = FileReport(name=S.CSV_FILE_HYPSO, required=False)
        path = data_dir / S.CSV_FILE_HYPSO
        if not path.is_file():
            report.note = S.BOOTSTRAP_OPTIONAL_SKIPPED
            return report
        report.present = True
        rows, errors = hypsometry.parse_param_csv(path.read_text(encoding='utf-8-sig'))
        self._apply_validated(report, rows, errors, _replace_hypso)
        return report

    # --- harvest plan items -------------------------------------------------

    def _load_harvest_plan_items(self, data_dir):
        reader, report = self._open(
            data_dir, S.CSV_FILE_HARVEST_PLAN_ITEMS, required=False,
            required_cols=csv_plan.PLAN_ITEMS_CSV_REQUIRED)
        if reader is None:
            return report
        parsed, errors = csv_plan.validate_canonical_items(
            reader, csv_plan.db_indexes(),
            {p.name: p for p in HarvestPlan.objects.all()},
        )
        self._apply_validated(
            report, parsed, errors, csv_plan.apply_canonical_items,
        )
        return report

    # --- preserved trees ----------------------------------------------------

    def _load_preserved_trees(self, data_dir):
        reader, report = self._open(
            data_dir, S.CSV_FILE_PRESERVED_TREES, required=False,
            required_cols=csv_preserved.PRESERVED_CSV_REQUIRED)
        if reader is None:
            return report
        parsed, errors = csv_preserved.validate_rows(
            reader, csv_preserved.db_indexes())
        self._apply_validated(report, parsed, errors, csv_preserved.apply)
        return report

    # --- harvests -----------------------------------------------------------

    def _load_harvests(self, data_dir):
        reader, report = self._open(
            data_dir, S.CSV_FILE_HARVESTS, required=False,
            required_cols=csv_harvests.HARVEST_CSV_REQUIRED)
        if reader is None:
            return report
        cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
        if self._missing_columns(report, missing):
            return report
        parsed, errors = csv_harvests.validate_rows(
            reader, cols, dyn, csv_harvests.db_indexes())
        self._apply_validated(report, parsed, errors, csv_harvests.apply)
        return report

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


def _replace_hypso(rows):
    hypsometry.replace_active_set(
        rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[],
    )
    return len(rows)


def _group(reader, col):
    """Partition reader rows by the (stripped) value of `col`."""
    groups: dict = {}
    for row in reader:
        groups.setdefault((row.get(col) or '').strip(), []).append(row)
    return groups
