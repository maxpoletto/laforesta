"""Tests for shared query selectors."""

from apps.base.models import HarvestPlan, SampleGrid, Survey
from apps.base.selectors import (
    active_or_default_harvest_plan, active_or_default_survey_ids,
)


def test_active_or_default_harvest_plan_prefers_active(db):
    fallback = HarvestPlan.objects.create(
        name='Current', year_start=2020, year_end=2030,
    )
    active = HarvestPlan.objects.create(
        name='Manual', year_start=2010, year_end=2015, active=True,
    )

    assert active_or_default_harvest_plan() == active
    assert active_or_default_harvest_plan() != fallback


def test_active_or_default_harvest_plan_uses_current_year(monkeypatch, db):
    class FixedDate:
        year = 2026

    monkeypatch.setattr('apps.base.selectors.timezone.localdate', lambda: FixedDate())
    HarvestPlan.objects.create(name='Old', year_start=2010, year_end=2020)
    shorter = HarvestPlan.objects.create(name='Shorter', year_start=2024, year_end=2026)
    longer = HarvestPlan.objects.create(name='Longer', year_start=2024, year_end=2028)

    assert active_or_default_harvest_plan() == longer
    assert active_or_default_harvest_plan() != shorter


def test_active_or_default_survey_ids_prefers_active_sorted_by_name(db):
    grid = SampleGrid.objects.create(name='Grid')
    zeta = Survey.objects.create(name='Zeta', sample_grid=grid, active=True)
    alpha = Survey.objects.create(name='Alpha', sample_grid=grid, active=True)
    Survey.objects.create(name='Inactive', sample_grid=grid)

    assert active_or_default_survey_ids() == [alpha.id, zeta.id]


def test_active_or_default_survey_ids_falls_back_to_first_by_name(db):
    grid = SampleGrid.objects.create(name='Grid')
    beta = Survey.objects.create(name='Beta', sample_grid=grid)
    alpha = Survey.objects.create(name='Alpha', sample_grid=grid)

    assert active_or_default_survey_ids() == [alpha.id]
    assert active_or_default_survey_ids() != [beta.id]


def test_active_or_default_survey_ids_empty(db):
    assert active_or_default_survey_ids() == []
