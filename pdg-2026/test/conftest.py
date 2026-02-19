"""
Pytest configuration and shared fixtures for acc.py tests.
"""

import math
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import acc

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def clear_caches():
    """Clear module-level caches before tests."""
    acc.region_cache.clear()
    acc.file_cache.clear()
    yield
    acc.region_cache.clear()
    acc.file_cache.clear()


@pytest.fixture(scope="module")
def trees_df(clear_caches):
    """Load test tree data and compute volumes."""
    df = acc.load_trees(["alberi.csv"], TEST_DATA_DIR)
    df = acc.calculate_all_trees_volume(df)
    return df


@pytest.fixture(scope="module")
def particelle_df(clear_caches):
    """Load test parcel metadata."""
    return acc.load_csv("particelle.csv", TEST_DATA_DIR)


@pytest.fixture(scope="module")
def data_all(trees_df, particelle_df):
    """Get parcel_data for all parcels, all species."""
    return acc.parcel_data(
        ["alberi.csv"], trees_df, particelle_df,
        regions=["Test"], parcels=[], species=[]
    )


@pytest.fixture(scope="module")
def data_parcel_a(trees_df, particelle_df):
    """Get parcel_data for parcel A only."""
    return acc.parcel_data(
        ["alberi.csv"], trees_df, particelle_df,
        regions=["Test"], parcels=["A"], species=[]
    )


@pytest.fixture(scope="module")
def data_parcel_b(trees_df, particelle_df):
    """Get parcel_data for parcel B only."""
    return acc.parcel_data(
        ["alberi.csv"], trees_df, particelle_df,
        regions=["Test"], parcels=["B"], species=[]
    )


@pytest.fixture(scope="module")
def data_parcel_c(trees_df, particelle_df):
    """Get parcel_data for parcel C only."""
    return acc.parcel_data(
        ["alberi.csv"], trees_df, particelle_df,
        regions=["Test"], parcels=["C"], species=[]
    )


@pytest.fixture(scope="module")
def data_parcel_d(trees_df, particelle_df):
    """Get parcel_data for parcel D only (age=20, basal area 15% rule)."""
    return acc.parcel_data(
        ["alberi.csv"], trees_df, particelle_df,
        regions=["Test"], parcels=["D"], species=[]
    )


@pytest.fixture(scope="module")
def data_parcel_e(trees_df, particelle_df):
    """Get parcel_data for parcel E only (age=45, basal area 20% rule)."""
    return acc.parcel_data(
        ["alberi.csv"], trees_df, particelle_df,
        regions=["Test"], parcels=["E"], species=[]
    )


# Harvest-related fixtures

# Test harvest rules: single comparto 'A' with provv_min=200,
# same age/volume thresholds as production.
_PROVV_MINIMA_TEST = {'A': 20}
_MIN_VOLUME_PCT_TEST = 120
_VOLUME_RULES_TEST = [(180, 25), (160, 20), (140, 15), (_MIN_VOLUME_PCT_TEST, 10), (0, 0)]
_AGE_RULES_TEST = [(60, None), (30, 20), (0, 15)]

def _volume_pp_max_test(volume_per_ha, provv_min):
    for threshold_ppm, pp_max in _VOLUME_RULES_TEST:
        if volume_per_ha > threshold_ppm * provv_min / 100:
            return pp_max
    return 0.0

def _test_prelievo_massimo(comparto, eta_media, volume_per_ha, area_basimetrica_per_ha):
    """Test harvest rules matching production logic."""
    provv_min = _PROVV_MINIMA_TEST.get(comparto)
    if provv_min is None:
        raise ValueError(f"Comparto sconosciuto: {comparto}")
    if provv_min < 0:
        return 0.0, 0.0
    pp_max = _volume_pp_max_test(volume_per_ha, provv_min)
    vol_max = volume_per_ha * pp_max / 100
    for min_age, pp_max_basal in _AGE_RULES_TEST:
        if eta_media >= min_age:
            if pp_max_basal is None:
                return vol_max, math.inf
            else:
                return vol_max, area_basimetrica_per_ha * pp_max_basal / 100
    return 0.0, 0.0

@pytest.fixture(scope="module")
def harvest_rules():
    """Harvest rules function for tests (replaces old comparti/provv_vol/provv_eta fixtures)."""
    return _test_prelievo_massimo
