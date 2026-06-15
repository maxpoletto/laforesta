"""Unit tests for the named-container CSV cores (apps/base/csv_containers)."""

from datetime import date

import pytest

from apps.base import csv_io, csv_containers as cc
from apps.base import csv_reference as ref
from apps.base.models import HarvestPlan, SampleGrid, Survey
from config import strings as S


def _reader(text):
    return csv_io.read(text)


# --- sample_grids (RefTable via the engine) --------------------------------

@pytest.mark.django_db
def test_sample_grids_create():
    reader = _reader(
        f'{S.CSV_COL_GRID},{S.CSV_COL_DESCRIPTION}\n'
        'Griglia 2024,Prima griglia\n'
        'Griglia 2025,\n'
    )
    cols, missing = ref.resolve_columns(cc.SAMPLE_GRIDS, reader.fieldnames)
    assert not missing
    parsed, errors = ref.validate_rows(cc.SAMPLE_GRIDS, reader, cols)
    assert errors == []
    assert ref.apply(cc.SAMPLE_GRIDS, parsed) == (2, 0)
    g = SampleGrid.objects.get(name='Griglia 2024')
    assert g.description == 'Prima griglia'
    assert SampleGrid.objects.get(name='Griglia 2025').description == ''


# --- harvest_plans (RefTable via the engine) -------------------------------

@pytest.mark.django_db
def test_harvest_plans_create():
    reader = _reader(
        f'{S.CSV_COL_PLAN},{S.CSV_COL_YEAR_START},{S.CSV_COL_YEAR_END},'
        f'{S.CSV_COL_DESCRIPTION}\n'
        'PDG 2026,2026,2040,Piano decennale\n'
    )
    cols, missing = ref.resolve_columns(cc.HARVEST_PLANS, reader.fieldnames)
    assert not missing
    parsed, errors = ref.validate_rows(cc.HARVEST_PLANS, reader, cols)
    assert errors == []
    ref.apply(cc.HARVEST_PLANS, parsed)
    plan = HarvestPlan.objects.get(name='PDG 2026')
    assert plan.year_start == 2026 and plan.year_end == 2040
    assert plan.description == 'Piano decennale'
    assert plan.active is False


@pytest.mark.django_db
def test_harvest_plans_missing_required_year_flagged():
    reader = _reader(
        f'{S.CSV_COL_PLAN},{S.CSV_COL_YEAR_START},{S.CSV_COL_YEAR_END}\n'
        'PDG 2026,,2040\n'                     # blank required year_start
    )
    cols, _ = ref.resolve_columns(cc.HARVEST_PLANS, reader.fieldnames)
    parsed, errors = ref.validate_rows(cc.HARVEST_PLANS, reader, cols)
    assert parsed == []
    assert len(errors) == 1


# --- surveys (bespoke, grid FK) --------------------------------------------

@pytest.mark.django_db
def test_surveys_create_with_grid_and_date():
    grid = SampleGrid.objects.create(name='Griglia 2024')
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID},{S.CSV_COL_DESCRIPTION},{S.CSV_COL_DATA}\n'
        f'Campagna 2024,{grid.name},Misure 2024,2024-09-15\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert errors == []
    assert parsed[0]['date'] == date(2024, 9, 15)      # carried for the orchestrator
    assert cc.apply_surveys(parsed) == 1
    s = Survey.objects.get(name='Campagna 2024')
    assert s.sample_grid == grid
    assert s.description == 'Misure 2024'
    assert s.active is False


@pytest.mark.django_db
def test_surveys_active_column_persisted():
    grid = SampleGrid.objects.create(name='Griglia 2024')
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID},{S.CSV_COL_ACTIVE}\n'
        f'Campagna 2024,{grid.name},True\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert errors == []
    assert cc.apply_surveys(parsed) == 1
    assert Survey.objects.get(name='Campagna 2024').active is True


@pytest.mark.django_db
def test_surveys_invalid_active_flagged():
    grid = SampleGrid.objects.create(name='Griglia 2024')
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID},{S.CSV_COL_ACTIVE}\n'
        f'Campagna 2024,{grid.name},maybe\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_surveys_unknown_grid_flagged():
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID}\n'
        'Campagna 2024,Nessuna griglia\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_surveys_blank_name_flagged():
    grid = SampleGrid.objects.create(name='Griglia 2024')
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID}\n'
        f',{grid.name}\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_surveys_bad_date_flagged():
    grid = SampleGrid.objects.create(name='Griglia 2024')
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID},{S.CSV_COL_DATA}\n'
        f'Campagna 2024,{grid.name},2024-13-99\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_surveys_date_optional():
    grid = SampleGrid.objects.create(name='Griglia 2024')
    reader = _reader(
        f'{S.CSV_COL_SURVEY},{S.CSV_COL_GRID}\n'
        f'Campagna 2024,{grid.name}\n'
    )
    parsed, errors = cc.validate_surveys(reader, cc.survey_db_indexes())
    assert errors == []
    assert parsed[0]['date'] is None
    cc.apply_surveys(parsed)
    assert Survey.objects.get(name='Campagna 2024').sample_grid == grid
