"""ETL tests for the new Campionamenti import commands."""

from decimal import Decimal
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.base.models import (
    Parcel, Sample, SampleArea, SampleGrid, Survey, Tree, TreeSample,
)

BOSCO_DATA = Path(__file__).resolve().parent.parent.parent / 'bosco' / 'data'
HAS_CSV = (BOSCO_DATA / 'aree-di-saggio.csv').exists()

skip_no_csv = pytest.mark.skipif(
    not HAS_CSV, reason='bosco/data CSVs not available',
)


@skip_no_csv
class TestImportSampleGrid:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))
        call_command('import_parcels', str(BOSCO_DATA))
        call_command('import_sample_grid', str(BOSCO_DATA))

    def test_creates_one_grid(self):
        assert SampleGrid.objects.count() == 1

    def test_sample_area_count(self):
        assert SampleArea.objects.count() == 177

    def test_default_radius(self):
        # The PDG 2026 CSV lacks Raggio; this one-off loader uses 20 m.
        sa = SampleArea.objects.first()
        assert sa.r_m == 20

    def test_idempotent(self):
        """Re-running doesn't duplicate."""
        call_command('import_sample_grid', str(BOSCO_DATA))
        assert SampleArea.objects.count() == 177

    def test_bis_numbers_preserved(self):
        """Alphanumeric area numbers like '27 bis' should round-trip."""
        assert SampleArea.objects.filter(number__contains='bis').exists()


class TestImportSampleGridPerRegion:
    """The loader enforces per-(grid, compresa) area-number uniqueness,
    aborting if the source CSV gives one number to two particelle of the
    same compresa.  Builds its own CSV, so it runs without bosco/data."""

    HEADER = 'Compresa,Particella,Area saggio,Lon,Lat,Quota\n'

    def _write_csv(self, data_dir, rows):
        (data_dir / 'aree-di-saggio.csv').write_text(
            self.HEADER + ''.join(rows), encoding='utf-8-sig',
        )

    def test_same_number_two_parcels_one_region_aborts(
        self, db, tmp_path, regions, eclasses,
    ):
        region = regions[0]
        for name in ('P1', 'P2'):
            Parcel.objects.create(
                name=name, region=region, eclass=eclasses[0],
                area_ha=Decimal('1.0'),
            )
        self._write_csv(tmp_path, [
            f'{region.name},P1,5,16.1,38.5,500\n',
            f'{region.name},P2,5,16.2,38.6,520\n',
        ])
        with pytest.raises(CommandError, match='unique per compresa'):
            call_command('import_sample_grid', str(tmp_path))
        assert SampleArea.objects.count() == 0          # rolled back

    def test_same_number_different_regions_ok(
        self, db, tmp_path, regions, eclasses,
    ):
        Parcel.objects.create(name='P1', region=regions[0],
                              eclass=eclasses[0], area_ha=Decimal('1.0'))
        Parcel.objects.create(name='P2', region=regions[1],
                              eclass=eclasses[0], area_ha=Decimal('1.0'))
        self._write_csv(tmp_path, [
            f'{regions[0].name},P1,5,16.1,38.5,500\n',
            f'{regions[1].name},P2,5,16.2,38.6,520\n',
        ])
        call_command('import_sample_grid', str(tmp_path))
        assert SampleArea.objects.count() == 2


@skip_no_csv
class TestImportSampledTrees:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))
        call_command('import_parcels', str(BOSCO_DATA))
        call_command('import_sample_grid', str(BOSCO_DATA))
        call_command('import_sampled_trees', str(BOSCO_DATA))

    def test_creates_one_survey(self):
        assert Survey.objects.count() == 1

    def test_samples_per_visited_area(self):
        # 150 visited areas (alberi-calcolati.csv covers a subset of the
        # 177 sample areas).
        assert Sample.objects.count() == 150

    def test_tree_samples_count(self):
        # 6087 fustaia trees made it through species mapping.
        assert TreeSample.objects.count() == 6087

    def test_volume_materialized_for_fustaia(self):
        """Fustaia rows have non-null V and m."""
        ts = TreeSample.objects.exclude(volume_m3=None).first()
        assert ts is not None
        assert ts.volume_m3 > 0
        assert ts.mass_q is not None
        assert ts.mass_q > 0


@skip_no_csv
class TestImportPai:
    @pytest.fixture(autouse=True)
    def run_import(self, db):
        call_command('import_reference', str(BOSCO_DATA))
        call_command('import_parcels', str(BOSCO_DATA))
        call_command('import_pai', str(BOSCO_DATA))

    def test_pai_tree_count(self):
        assert Tree.objects.filter(preserved=True).count() == 1494

    def test_no_tree_samples(self):
        """PAI doesn't create TreeSample rows."""
        assert TreeSample.objects.count() == 0

    def test_idempotent(self):
        call_command('import_pai', str(BOSCO_DATA))
        assert Tree.objects.filter(preserved=True).count() == 1494
