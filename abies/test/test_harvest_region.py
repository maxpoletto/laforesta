"""Tests for region-wide Harvest (nullable parcel, region FK)."""

import pytest
from decimal import Decimal
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from apps.prelievi.models import Harvest
from apps.base.models import Product, Crew


@pytest.mark.django_db
def test_region_wide_harvest_allowed(regions, parcels):
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    h = Harvest.objects.create(
        date='2024-01-10', product=product, crew=crew, region=regions[0],
        parcel=None, mass_q=Decimal('10.0'), volume_m3=Decimal('0'),
    )
    assert h.region_id == regions[0].id and h.parcel_id is None


@pytest.mark.django_db
def test_harvest_both_parcel_and_region_rejected_by_trigger(regions, parcels):
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Harvest.objects.create(
                date='2024-01-10', product=product, crew=crew,
                region=regions[0], parcel=parcels[0],
                mass_q=Decimal('1'), volume_m3=Decimal('0'))


@pytest.mark.django_db
def test_harvest_neither_parcel_nor_region_rejected(regions, parcels):
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Harvest.objects.create(
                date='2024-01-10', product=product, crew=crew,
                region=None, parcel=None,
                mass_q=Decimal('1'), volume_m3=Decimal('0'))


@pytest.mark.django_db
def test_harvest_clean_rejects_both(regions, parcels):
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    h = Harvest(date='2024-01-10', product=product, crew=crew,
                region=regions[0], parcel=parcels[0],
                mass_q=Decimal('1'), volume_m3=Decimal('0'))
    with pytest.raises(ValidationError):
        h.clean()


@pytest.mark.django_db
def test_harvest_clean_rejects_neither(regions, parcels):
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    h = Harvest(date='2024-01-10', product=product, crew=crew,
                region=None, parcel=None,
                mass_q=Decimal('1'), volume_m3=Decimal('0'))
    with pytest.raises(ValidationError):
        h.clean()


@pytest.mark.django_db
def test_region_wide_harvest_digest_row(regions, parcels, species):
    from apps.base.models import Product, Crew
    from apps.prelievi.models import Harvest
    from apps.base.digests import build_harvest_record
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    h = Harvest.objects.create(date='2024-01-10', product=product, crew=crew,
                               region=regions[0], parcel=None,
                               mass_q=Decimal('10.0'), volume_m3=Decimal('0'))
    # Reload with select_related to mirror what the digest generator does.
    h = Harvest.objects.select_related(
        'parcel__region', 'region', 'crew', 'product',
    ).get(pk=h.pk)
    row = build_harvest_record(h)            # must not raise
    # region name present; Particella renders as the whole-region mark 'X'.
    assert regions[0].name in row
    assert 'X' in row


@pytest.mark.django_db
def test_harvest_region_xor_parcel_consistency_trigger(regions, parcels):
    """A region-wide harvest linked to a matching region-wide HarvestPlanItem
    is accepted; one linked to a different region's item is rejected."""
    from apps.base.models import HarvestPlan, HarvestPlanItem
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')

    plan = HarvestPlan.objects.create(name='Piano 2024', year_start=2024, year_end=2030)
    # Region-wide plan item in regions[0].
    hpi = HarvestPlanItem.objects.create(
        harvest_plan=plan, region=regions[0], year_planned=2024, damaged=True,
    )

    # Harvest in the same region (region[0]) linked to the region-wide item: allowed.
    h = Harvest.objects.create(
        date='2024-01-10', product=product, crew=crew,
        region=regions[0], parcel=None,
        harvest_plan_item=hpi,
        mass_q=Decimal('10.0'), volume_m3=Decimal('0'),
        damaged=True,
    )
    assert h.pk is not None

    # Harvest in a *different* region linked to that item: rejected.
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            Harvest.objects.create(
                date='2024-01-11', product=product, crew=crew,
                region=regions[1], parcel=None,
                harvest_plan_item=hpi,
                mass_q=Decimal('5.0'), volume_m3=Decimal('0'),
                damaged=True,
            )
