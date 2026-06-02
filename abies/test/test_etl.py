"""Tests for ETL management commands (against real CSV data)."""

import pytest
from pathlib import Path

from django.core.management import call_command

from apps.base.models import (
    Crew, Eclass, Product, Parcel, Region, Species, Tractor,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor

BOSCO_DATA = Path(__file__).resolve().parent.parent.parent / 'bosco' / 'data'
ABIES_DATA = Path(__file__).resolve().parent.parent.parent / 'abies-data'
HAS_CSV = (BOSCO_DATA / 'mannesi.csv').exists()

skip_no_csv = pytest.mark.skipif(not HAS_CSV, reason='bosco/data CSVs not available')
skip_no_calendar = pytest.mark.skipif(
    not (ABIES_DATA / 'piano_fustaia.csv').exists(),
    reason='abies-data calendar CSVs not available',
)


# ---------------------------------------------------------------------------
# import_reference
# ---------------------------------------------------------------------------

@skip_no_csv
class TestImportReference:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))

    def test_region_count(self):
        assert Region.objects.count() == 3

    def test_eclass_count(self):
        assert Eclass.objects.count() == 6

    def test_eclass_f_is_coppice(self):
        assert Eclass.objects.get(name='F').coppice is True

    def test_species_count(self):
        assert Species.objects.count() == 22

    def test_species_sort_order(self):
        other = Species.objects.get(common_name='Altro')
        assert other.sort_order == 999

    def test_tractor_count(self):
        assert Tractor.objects.count() == 5

    def test_product_count(self):
        assert Product.objects.count() == 5

    def test_crew_count(self):
        assert Crew.objects.count() >= 20  # at least 20 unique crews

    def test_idempotent(self, db):
        """Running twice doesn't duplicate rows."""
        call_command('import_reference', str(BOSCO_DATA))
        assert Region.objects.count() == 3
        assert Species.objects.count() == 22


# ---------------------------------------------------------------------------
# import_parcels
# ---------------------------------------------------------------------------

@skip_no_csv
class TestImportParcels:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))
        call_command('import_parcels', str(BOSCO_DATA))

    def test_parcel_count(self):
        # 85 real + 3 synthetic X = 88
        assert Parcel.objects.count() == 88

    def test_synthetic_x_parcels(self):
        x_parcels = Parcel.objects.filter(name='X')
        assert x_parcels.count() == 3
        for p in x_parcels:
            assert p.area_ha == 0

    def test_parcel_has_region(self):
        p = Parcel.objects.exclude(name='X').first()
        assert p.region is not None
        assert p.region.name in ('Capistrano', 'Fabrizia', 'Serra')


# ---------------------------------------------------------------------------
# import_mannesi
# ---------------------------------------------------------------------------

@skip_no_csv
class TestImportMannesi:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))
        call_command('import_parcels', str(BOSCO_DATA))
        call_command('import_mannesi', str(BOSCO_DATA))

    def test_harvest_count(self):
        assert Harvest.objects.count() == 11941

    def test_harvest_species_created(self):
        assert HarvestSpecies.objects.count() > 10000

    def test_harvest_tractor_created(self):
        assert HarvestTractor.objects.count() > 3000

    def test_vdp_nd_maps_to_null(self):
        """'nd' VDP values should be stored as NULL."""
        null_vdps = Harvest.objects.filter(record1__isnull=True)
        assert null_vdps.exists()

    def test_species_percentages_positive(self):
        """All stored species percentages should be > 0."""
        assert not HarvestSpecies.objects.filter(percent__lte=0).exists()

    def test_tractor_percentages_positive(self):
        assert not HarvestTractor.objects.filter(percent__lte=0).exists()

    def test_harvest_has_volume_m3(self):
        """volume_m3 should be materialized at write time."""
        h = Harvest.objects.exclude(volume_m3=0).first()
        assert h is not None
        assert h.volume_m3 > 0


# ---------------------------------------------------------------------------
# import_calendar (real piano CSVs live in abies-data/, not bosco/data)
# ---------------------------------------------------------------------------

@skip_no_csv
@skip_no_calendar
class TestImportCalendar:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))
        call_command('import_parcels', str(BOSCO_DATA))
        call_command('import_calendar', str(ABIES_DATA))

    def test_creates_plan_with_items(self):
        from apps.base.management.commands.import_calendar import PLAN_NAME
        from apps.base.models import HarvestPlan, HarvestPlanItem
        assert HarvestPlan.objects.filter(name=PLAN_NAME).exists()
        assert HarvestPlanItem.objects.count() > 0

    def test_parses_fustaia_volume(self):
        from decimal import Decimal

        from apps.base.models import HarvestPlanItem
        # piano_fustaia.csv row 1: Serra / 26 / 2027 / Prelievo (m³) = 3842.4
        parcel = Parcel.objects.filter(region__name='Serra', name='26').first()
        if parcel is None:
            pytest.skip('parcel Serra/26 not in dataset')
        item = HarvestPlanItem.objects.get(parcel=parcel, year_planned=2027)
        assert item.volume_planned_m3 == Decimal('3842.4')
