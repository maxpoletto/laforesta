"""CSV totals use the same parent serialization and no-op rules as forms."""
from datetime import date
from decimal import Decimal

import pytest
from django.db.models.query import QuerySet

from apps.base.models import HarvestPlan, HarvestPlanItem, HarvestPlanItemState
from apps.prelievi import csv_harvests
from apps.prelievi.models import Harvest

pytestmark = pytest.mark.django_db


def test_csv_locks_parent_before_children_and_preserves_state_and_replay(
        parcels, species, crews, products, tractors, monkeypatch):
    plan = HarvestPlan.objects.create(name='CSV clocks', year_start=2024, year_end=2030)
    item = HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcels[0], year_planned=2026)
    row = dict(date=date(2026, 10, 1), product=products[0], parcel=parcels[0],
        region=None, crew=crews[0], harvest_plan_item=item, record1=1, record2=None,
        mass_q=Decimal('18'), volume_m3=Decimal('2'), damaged=False, unhealthy=False,
        psr=False, note='', species_pcts=[(species[0], 100)], tractor_pcts=[(tractors[0], 100)])
    original = QuerySet.select_for_update
    completed_writer = False
    def lock_after_writer(queryset, *args, **kwargs):
        nonlocal completed_writer
        if queryset.model is HarvestPlanItem and not completed_writer:
            completed_writer = True
            Harvest.objects.create(date=row['date'], parcel=parcels[0], crew=crews[0],
                product=products[0], harvest_plan_item=item, mass_q=27, volume_m3=3)
        return original(queryset, *args, **kwargs)
    monkeypatch.setattr(QuerySet, 'select_for_update', lock_after_writer)
    assert csv_harvests.apply([row]) == 1
    item.refresh_from_db()
    assert item.volume_actual_m3 == 5
    assert item.state == HarvestPlanItemState.PLANNED and item.version == 1
    unchanged = (item.modified_at, item.history.count())
    assert csv_harvests.apply([row]) == 0
    item.refresh_from_db()
    assert (item.modified_at, item.history.count()) == unchanged
