"""
Pytest configuration and shared fixtures for acc.py tests.
"""

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
