"""ETL tests for the new Campionamenti import commands."""

from pathlib import Path

import pytest
from django.core.management import call_command

from apps.base.models import (
    Sample, SampleArea, SampleGrid, Survey, Tree, TreeSample,
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
        sa = SampleArea.objects.first()
        assert sa.r_m == 12

    def test_idempotent(self):
        """Re-running doesn't duplicate."""
        call_command('import_sample_grid', str(BOSCO_DATA))
        assert SampleArea.objects.count() == 177

    def test_bis_numbers_preserved(self):
        """Alphanumeric area numbers like '27 bis' should round-trip."""
        assert SampleArea.objects.filter(number__contains='bis').exists()


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
