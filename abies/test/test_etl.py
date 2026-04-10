"""Tests for ETL scripts (against real CSV data)."""

import pytest
from pathlib import Path

from apps.base.models import (
    Crew, Eclass, Note, Optype, Parcel, Region, Species, Tractor,
)
from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor

BOSCO_DATA = Path(__file__).resolve().parent.parent.parent / 'bosco' / 'data'
HAS_CSV = (BOSCO_DATA / 'mannesi.csv').exists()

skip_no_csv = pytest.mark.skipif(not HAS_CSV, reason='bosco/data CSVs not available')


# ---------------------------------------------------------------------------
# import_reference
# ---------------------------------------------------------------------------

@skip_no_csv
class TestImportReference:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        from ingest.import_reference import run
        run()

    def test_region_count(self):
        assert Region.objects.count() == 3

    def test_eclass_count(self):
        assert Eclass.objects.count() == 6

    def test_eclass_f_is_coppice(self):
        assert Eclass.objects.get(name='F').coppice is True

    def test_species_count(self):
        assert Species.objects.count() == 7

    def test_species_sort_order(self):
        altro = Species.objects.get(common_name='Altro')
        assert altro.sort_order == 999

    def test_tractor_count(self):
        assert Tractor.objects.count() == 5

    def test_optype_count(self):
        assert Optype.objects.count() == 5

    def test_note_count(self):
        assert Note.objects.count() == 3

    def test_crew_count(self):
        assert Crew.objects.count() >= 20  # at least 20 unique crews

    def test_idempotent(self, db):
        """Running twice doesn't duplicate rows."""
        from ingest.import_reference import run
        run()
        assert Region.objects.count() == 3
        assert Species.objects.count() == 7


# ---------------------------------------------------------------------------
# import_parcels
# ---------------------------------------------------------------------------

@skip_no_csv
class TestImportParcels:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        from ingest.import_reference import run as ref_run
        ref_run()
        from ingest.import_parcels import run
        run()

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
        from ingest.import_reference import run as ref_run
        ref_run()
        from ingest.import_parcels import run as parcel_run
        parcel_run()
        from ingest.import_mannesi import run
        run()

    def test_harvest_op_count(self):
        assert HarvestOp.objects.count() == 11941

    def test_harvest_species_created(self):
        assert HarvestSpecies.objects.count() > 10000

    def test_harvest_tractor_created(self):
        assert HarvestTractor.objects.count() > 3000

    def test_vdp_nd_maps_to_null(self):
        """'nd' VDP values should be stored as NULL."""
        null_vdps = HarvestOp.objects.filter(record1__isnull=True)
        assert null_vdps.exists()

    def test_species_percentages_positive(self):
        """All stored species percentages should be > 0."""
        assert not HarvestSpecies.objects.filter(percent__lte=0).exists()

    def test_tractor_percentages_positive(self):
        assert not HarvestTractor.objects.filter(percent__lte=0).exists()
