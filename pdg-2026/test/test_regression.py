"""
Regression tests for pdg calculate_* functions.

These tests compare the output of calculate_* functions against golden reference
CSVs generated from a 3-parcel subset of the real data:
  - Serra/1      (comparto A, 1 sample area, 59 trees, 2 species)
  - Fabrizia/1   (comparto E, 3 sample areas, 109 trees, 4 species)
  - Capistrano/3 (comparto D, 3 sample areas, 79 trees, 5 species)

The flag combinations tested match the actual @@directive invocations in
template/tex/*.tex (see template grep for details).

To regenerate golden files after an intentional change, run:
    make regenerate-golden
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pdg.computation import COL_PARTICELLA, COL_COMPRESA, COL_GENERE, COL_CD_CM
from pdg.io import file_cache, load_csv, load_trees
from pdg.simulation import calculate_pct_growth_table
from pdg.core import (
    region_cache, parcel_data,
    calculate_volume_table, calculate_harvest_table,
    calculate_diameter_class_data,
)
from pdg.harvest_rules import max_harvest

TEST_DIR = Path(__file__).parent / "data"

# Tolerance: golden CSVs are written with %.6f, so round-trip introduces
# errors up to ~3e-6 relative.  Use 1e-5 to stay safely above that.
RTOL = 1e-5


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def regression_env():
    """Load the regression test data set (3 parcels from real data)."""
    file_cache.clear()
    region_cache.clear()
    trees_df = load_trees(['regression-alberi.csv'], TEST_DIR)
    particelle_df = load_csv('regression-particelle.csv', TEST_DIR)
    particelle_df[COL_PARTICELLA] = particelle_df[COL_PARTICELLA].astype(str)
    yield trees_df, particelle_df
    file_cache.clear()
    region_cache.clear()


def _parcel_data(regression_env, regions=None, parcels=None):
    trees_df, particelle_df = regression_env
    region_cache.clear()
    return parcel_data(
        ['regression-alberi.csv'], trees_df, particelle_df,
        regions=regions or [], parcels=parcels or [], species=[])


@pytest.fixture(scope="module")
def data_all(regression_env):
    return _parcel_data(regression_env)


@pytest.fixture(scope="module")
def data_serra(regression_env):
    return _parcel_data(regression_env, regions=['Serra'])


@pytest.fixture(scope="module")
def data_fab1(regression_env):
    return _parcel_data(regression_env, regions=['Fabrizia'], parcels=['1'])


@pytest.fixture(scope="module")
def data_cap3(regression_env):
    return _parcel_data(regression_env, regions=['Capistrano'], parcels=['3'])


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_golden(name: str) -> pd.DataFrame:
    """Load a golden reference CSV."""
    path = TEST_DIR / f"golden-{name}.csv"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")
    return pd.read_csv(path)


def _load_golden_indexed(name: str) -> pd.DataFrame:
    """Load a golden reference CSV with index column 0."""
    path = TEST_DIR / f"golden-{name}.csv"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")
    return pd.read_csv(path, index_col=0)


def _assert_frames_close(actual: pd.DataFrame, expected: pd.DataFrame,
                          label: str):
    """Assert two DataFrames are close, with informative error messages."""
    # Same shape
    assert actual.shape == expected.shape, (
        f"{label}: shape mismatch: {actual.shape} vs {expected.shape}")

    # Compare numeric columns with tolerance, string columns exactly
    for col in expected.columns:
        if expected[col].dtype in (np.float64, float):
            np.testing.assert_allclose(
                actual[col].values, expected[col].values,
                rtol=RTOL, err_msg=f"{label}, column '{col}'")
        else:
            act_vals = actual[col].astype(str).values
            exp_vals = expected[col].astype(str).values
            assert list(act_vals) == list(exp_vals), (
                f"{label}, column '{col}': {act_vals} != {exp_vals}")


# ── @@volumi (calculate_volume_table) ──────────────────────────────────────────────

class TestTsvRegression:
    """Regression tests matching @@volumi invocations in templates."""

    def test_per_compresa(self, data_all):
        """sec-volumi.tex: per_compresa=si, per_particella=no, with CI."""
        actual = calculate_volume_table(data_all,
            group_cols=[COL_COMPRESA],
            calc_margin=True, calc_total=True)
        expected = _load_golden('tsv-per_compresa')
        _assert_frames_close(actual, expected, 'tsv-per_compresa')

    def test_serra_per_particella(self, data_serra):
        """sec-volumi.tex: single compresa, per_particella=si, with CI."""
        actual = calculate_volume_table(data_serra,
            group_cols=[COL_PARTICELLA],
            calc_margin=True, calc_total=True)
        expected = _load_golden('tsv-serra-per_particella')
        _assert_frames_close(actual, expected, 'tsv-serra-per_particella')

    def test_fab1_per_genere(self, data_fab1):
        """particella.tex: single particella, per_genere=si, with CI."""
        actual = calculate_volume_table(data_fab1,
            group_cols=[COL_GENERE],
            calc_margin=True, calc_total=True)
        expected = _load_golden('tsv-fab1-per_genere')
        _assert_frames_close(actual, expected, 'tsv-fab1-per_genere')


# ── @@prelievi (calculate_harvest_table) ──────────────────────────────────────────────

class TestTptRegression:
    """Regression tests matching @@prelievi invocations in templates."""

    def test_per_compresa(self, data_all):
        """sec-ripresa.tex: per_compresa=si, per_particella=no."""
        actual = calculate_harvest_table(data_all, max_harvest,
            group_cols=[COL_COMPRESA])
        expected = _load_golden('tpt-per_compresa')
        _assert_frames_close(actual, expected, 'tpt-per_compresa')

    def test_serra_per_particella(self, data_serra):
        """sec-ripresa.tex: single compresa, per_particella=si."""
        actual = calculate_harvest_table(data_serra, max_harvest,
            group_cols=[COL_PARTICELLA])
        expected = _load_golden('tpt-serra-per_particella')
        _assert_frames_close(actual, expected, 'tpt-serra-per_particella')

    def test_cap3_per_genere(self, data_cap3):
        """particella.tex: single particella, per_genere=si."""
        actual = calculate_harvest_table(data_cap3, max_harvest,
            group_cols=[COL_GENERE])
        expected = _load_golden('tpt-cap3-per_genere')
        _assert_frames_close(actual, expected, 'tpt-cap3-per_genere')

    def test_serra_per_particella_genere(self, data_serra):
        """relazione.tex: per_particella=si, per_genere=si."""
        actual = calculate_harvest_table(data_serra, max_harvest,
            group_cols=[COL_PARTICELLA, COL_GENERE])
        expected = _load_golden('tpt-serra-per_particella_genere')
        _assert_frames_close(actual, expected, 'tpt-serra-per_particella_genere')


# ── @@tabella_incremento_percentuale (calculate_pct_growth_table) ───────────────────────────────────────────

class TestTipRegression:
    """Regression tests matching @@tabella_incremento_percentuale invocations in templates."""

    def test_fab1(self, data_fab1):
        """particella.tex: single particella, genere+cd_cm."""
        actual = calculate_pct_growth_table(data_fab1,
            group_cols=[COL_GENERE, COL_CD_CM],
            stime_totali=True)
        expected = _load_golden('tip-fab1')
        _assert_frames_close(actual, expected, 'tip-fab1')

    def test_cap3(self, data_cap3):
        """particella.tex: single particella, genere+cd_cm."""
        actual = calculate_pct_growth_table(data_cap3,
            group_cols=[COL_GENERE, COL_CD_CM],
            stime_totali=True)
        expected = _load_golden('tip-cap3')
        _assert_frames_close(actual, expected, 'tip-cap3')


# ── @@tabella_classi_diametriche (calculate_diameter_class_data) ────────────────────────────────────────────────

class TestTcdRegression:
    """Regression tests matching @@tabella_classi_diametriche invocations in templates."""

    @pytest.mark.parametrize("metrica", ['alberi_ha', 'volume_ha', 'G_ha', 'altezza'])
    def test_fab1_coarse(self, data_fab1, metrica):
        """particella.tex: single particella, coarse bins, each metric."""
        actual = calculate_diameter_class_data(data_fab1,
            metrica=metrica, stime_totali=True, fine=False)
        expected = _load_golden_indexed(f'tcd-fab1-{metrica}')
        # Align index types (coarse bins are strings)
        actual.index = actual.index.astype(str)
        expected.index = expected.index.astype(str)
        pd.testing.assert_frame_equal(actual, expected, rtol=RTOL,
                                       check_names=False)

    def test_all_volume_tot_fine(self, data_all):
        """Cross-check: fine volume_tot over all parcels."""
        actual = calculate_diameter_class_data(data_all,
            metrica='volume_tot', stime_totali=True, fine=True)
        expected = _load_golden_indexed('tcd-all-volume_tot-fine')
        pd.testing.assert_frame_equal(actual, expected, rtol=RTOL,
                                       check_names=False)
