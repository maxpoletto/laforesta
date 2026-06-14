"""Named-container CSV import cores for bootstrap: sample_grids, harvest_plans,
surveys.  These create the container rows that the bulk domain cores
(``csv_grid``/``csv_trees``/``csv_plan``) then fill.

``sample_grids`` and ``harvest_plans`` are flat, FK-free, single-key tables, so
they reuse the declarative ``RefTable`` engine from ``apps.base.csv_reference``
(load them with ``ref.resolve_columns`` / ``ref.validate_rows`` / ``ref.apply``).
``surveys`` carries a foreign key to its grid, so it is a small bespoke
three-phase core in the style of ``apps/campionamenti/csv_trees``.

The surveys loader also parses each survey's default sample date (the ``Data``
column) and carries it on the parsed row as ``'date'``.  That value is the
per-survey default fed to ``csv_trees`` at bootstrap; it is NOT a ``Survey``
column (``Survey`` has no date field), so ``apply_surveys`` does not persist it.
"""

from dataclasses import dataclass
from datetime import date as date_type

from django.db import transaction

from apps.base.csv_reference import KIND_INT, KIND_STR, RefColumn, RefTable
from apps.base.digests import mark_all_stale
from apps.base.models import HarvestPlan, SampleGrid, Survey
from config import strings as S
from config.constants import (
    FIELD_DESCRIPTION, FIELD_NAME, FIELD_YEAR_END, FIELD_YEAR_START,
)

# Flat, FK-free containers — declarative RefTables (loaded via the csv_reference
# engine: resolve_columns / validate_rows / apply).
SAMPLE_GRIDS = RefTable('sample_grids', SampleGrid, (
    RefColumn(S.CSV_COL_GRID, FIELD_NAME, KIND_STR),
    RefColumn(S.CSV_COL_DESCRIPTION, FIELD_DESCRIPTION, KIND_STR,
              required=False, default=''),
))

HARVEST_PLANS = RefTable('harvest_plans', HarvestPlan, (
    RefColumn(S.CSV_COL_PLAN, FIELD_NAME, KIND_STR),
    RefColumn(S.CSV_COL_YEAR_START, FIELD_YEAR_START, KIND_INT),
    RefColumn(S.CSV_COL_YEAR_END, FIELD_YEAR_END, KIND_INT),
    RefColumn(S.CSV_COL_DESCRIPTION, FIELD_DESCRIPTION, KIND_STR,
              required=False, default=''),
))


# --- surveys (bespoke; FK to grid) -----------------------------------------

SURVEY_CSV_REQUIRED = [S.CSV_COL_SURVEY, S.CSV_COL_GRID]
SURVEY_CSV_OPTIONAL = [S.CSV_COL_DESCRIPTION, S.CSV_COL_DATA]


@dataclass
class SurveyIndexes:
    grids: dict   # grid name -> SampleGrid


def survey_db_indexes() -> SurveyIndexes:
    return SurveyIndexes(grids={g.name: g for g in SampleGrid.objects.all()})


def validate_surveys(reader, idx: SurveyIndexes):
    """Validate ``surveys.csv`` rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed, errors)``.  Each parsed row is a dict of Survey model
    kwargs (``name``, ``sample_grid``, ``description``) plus ``date`` — the
    optional per-survey default sample date carried for the orchestrator (not a
    Survey column).  Errors name the 1-based data row.
    """
    errors, parsed, seen = [], [], set()
    for i, row in enumerate(reader, 2):
        name = row[S.CSV_COL_SURVEY].strip()
        if not name:
            errors.append(S.ERR_CSV_VALUE_REQUIRED.format(i, S.CSV_COL_SURVEY))
            continue
        grid = idx.grids.get(row[S.CSV_COL_GRID].strip())
        if grid is None:
            errors.append(
                S.ERR_CSV_GRID_NOT_FOUND.format(i, row[S.CSV_COL_GRID].strip()))
            continue
        if name in seen:
            errors.append(S.ERR_CSV_DUPLICATE_KEY.format(i, S.CSV_COL_SURVEY, name))
            continue
        default_date = None
        raw_date = (row.get(S.CSV_COL_DATA) or '').strip()
        if raw_date:
            try:
                default_date = date_type.fromisoformat(raw_date)
            except ValueError:
                errors.append(
                    S.ERR_CSV_VALUE_PARSE.format(i, S.CSV_COL_DATA, raw_date))
                continue
        seen.add(name)
        parsed.append({
            'name': name, 'sample_grid': grid,
            'description': (row.get(S.CSV_COL_DESCRIPTION) or '').strip(),
            'date': default_date,
        })
    return parsed, errors


def apply_surveys(parsed) -> int:
    """Persist validated surveys, inserting on name.  Create-only: a row whose
    name already exists is left unchanged (idempotent re-import, no corrections
    — bootstrap loads into an empty instance).  The ``date`` on each parsed row
    is downstream metadata and is not stored.  Returns the number created; marks
    digests stale."""
    created = 0
    with transaction.atomic():
        for d in parsed:
            _, was_created = Survey.objects.get_or_create(
                name=d['name'],
                defaults={'sample_grid': d['sample_grid'],
                          'description': d['description']},
            )
            if was_created:
                created += 1
        mark_all_stale()
    return created
