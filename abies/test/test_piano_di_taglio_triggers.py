"""Schema-level invariants for Piano di taglio tables.

Covers the SQLite trigger families landed in the initial migrations:
- harvest_plan_item: region XOR parcel
- harvest: parcel must match linked plan item's parcel or region
- harvest: damaged/unhealthy/psr flags must match linked plan item's flags

Sample-grid match triggers are tested in test_sample_invariants.py.
"""

from decimal import Decimal

import pytest
from django.db import DatabaseError, IntegrityError, transaction

from apps.base.models import (
    HarvestPlan, HarvestPlanItem, HarvestPlanItemState, Parcel,
)
from apps.prelievi.models import Harvest


# ---------------------------------------------------------------------------
# HarvestPlanItem.region XOR parcel
# ---------------------------------------------------------------------------

@pytest.fixture
def plan(db):
    return HarvestPlan.objects.create(
        name='Test Plan', year_start=2020, year_end=2030,
    )


class TestRegionXorParcel:
    def test_parcel_only_ok(self, plan, parcels):
        HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
        )

    def test_region_only_ok(self, plan, regions):
        HarvestPlanItem.objects.create(
            harvest_plan=plan, region=regions[0], year_planned=2025,
        )

    def test_both_blocks(self, plan, regions, parcels):
        with pytest.raises((IntegrityError, DatabaseError)):
            with transaction.atomic():
                HarvestPlanItem.objects.create(
                    harvest_plan=plan, region=regions[0], parcel=parcels[0],
                    year_planned=2025,
                )

    def test_neither_blocks(self, plan):
        with pytest.raises((IntegrityError, DatabaseError)):
            with transaction.atomic():
                HarvestPlanItem.objects.create(
                    harvest_plan=plan, year_planned=2025,
                )


# ---------------------------------------------------------------------------
# Harvest.parcel consistency with linked plan item
# ---------------------------------------------------------------------------

@pytest.fixture
def open_item_parcel(plan, parcels):
    """A plan item bound to parcels[0], state=open."""
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2025,
        state=HarvestPlanItemState.OPEN,
    )


@pytest.fixture
def open_item_region(plan, regions):
    """A region-wide plan item on regions[0], state=open."""
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, region=regions[0], year_planned=2025,
        state=HarvestPlanItemState.OPEN,
    )


class TestHarvestParcelConsistency:
    def test_parcel_bound_match_ok(self, open_item_parcel, parcels, products, crews):
        Harvest.objects.create(
            date='2025-06-01', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal('10'),
            harvest_plan_item=open_item_parcel,
        )

    def test_parcel_bound_mismatch_blocks(self, open_item_parcel, parcels, products, crews):
        # open_item_parcel is bound to parcels[0]; passing parcels[1] mismatches.
        with pytest.raises((IntegrityError, DatabaseError)):
            with transaction.atomic():
                Harvest.objects.create(
                    date='2025-06-01', parcel=parcels[1], crew=crews[0],
                    product=products[0], mass_q=Decimal('10'),
                    harvest_plan_item=open_item_parcel,
                )

    def test_region_wide_match_ok(self, open_item_region, parcels, products, crews):
        # parcels[0] and parcels[1] are both in regions[0].
        Harvest.objects.create(
            date='2025-06-01', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal('10'),
            harvest_plan_item=open_item_region,
        )

    def test_region_wide_mismatch_blocks(self, open_item_region, parcels, products, crews):
        # parcels[2] is in regions[1], not regions[0].
        with pytest.raises((IntegrityError, DatabaseError)):
            with transaction.atomic():
                Harvest.objects.create(
                    date='2025-06-01', parcel=parcels[2], crew=crews[0],
                    product=products[0], mass_q=Decimal('10'),
                    harvest_plan_item=open_item_region,
                )

    def test_null_link_skips_check(self, parcels, products, crews):
        """Historical harvests have no plan_item link; trigger must not fire."""
        Harvest.objects.create(
            date='2025-06-01', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal('10'),
        )


# ---------------------------------------------------------------------------
# Harvest.{damaged,unhealthy,psr} consistency with plan item
# ---------------------------------------------------------------------------

class TestHarvestFlagsConsistency:
    def test_all_match_ok(self, plan, parcels, products, crews):
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
            state=HarvestPlanItemState.OPEN,
            damaged=True, unhealthy=False, psr=False,
        )
        Harvest.objects.create(
            date='2025-06-01', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal('10'),
            harvest_plan_item=item,
            damaged=True, unhealthy=False, psr=False,
        )

    def test_damaged_mismatch_blocks(self, plan, parcels, products, crews):
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
            state=HarvestPlanItemState.OPEN, damaged=True,
        )
        with pytest.raises((IntegrityError, DatabaseError)):
            with transaction.atomic():
                Harvest.objects.create(
                    date='2025-06-01', parcel=parcels[0], crew=crews[0],
                    product=products[0], mass_q=Decimal('10'),
                    harvest_plan_item=item, damaged=False,
                )

    def test_psr_mismatch_blocks(self, plan, parcels, products, crews):
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
            state=HarvestPlanItemState.OPEN, psr=True,
        )
        with pytest.raises((IntegrityError, DatabaseError)):
            with transaction.atomic():
                Harvest.objects.create(
                    date='2025-06-01', parcel=parcels[0], crew=crews[0],
                    product=products[0], mass_q=Decimal('10'),
                    harvest_plan_item=item, psr=False,
                )

    def test_null_link_skips_check(self, parcels, products, crews):
        """Historical harvests can have any flag combination."""
        Harvest.objects.create(
            date='2025-06-01', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal('10'),
            damaged=True, psr=True,
        )


# ---------------------------------------------------------------------------
# HarvestPlanItem.state monotonicity (Python-level validator)
# ---------------------------------------------------------------------------

class TestStateMonotonic:
    def test_advance_ok(self, plan, parcels):
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
            state=HarvestPlanItemState.PLANNED,
        )
        item.state = HarvestPlanItemState.OPEN
        item.full_clean()  # raises if monotonic violated

    def test_regress_blocks(self, plan, parcels):
        from django.core.exceptions import ValidationError
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
            state=HarvestPlanItemState.HARVESTING,
        )
        item.state = HarvestPlanItemState.OPEN
        with pytest.raises(ValidationError):
            item.full_clean()
