#!/usr/bin/env python3
"""Tests for sentinel.py — rasterize_parcels and compute_timeseries.

Focuses on the multi-polygon parcel case: multiple GeoJSON features sharing
the same name should be merged into a single parcel for rasterization and
timeseries computation.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from sentinel import (
    BANDS,
    INDICES,
    index_to_uint8,
    rasterize_parcels,
    compute_timeseries,
    reflectance_to_uint8,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic GeoJSON with multi-polygon parcels
# ---------------------------------------------------------------------------

# Two small non-overlapping squares, both named "ParcelA".
# One square named "ParcelB".
# Grid: 10x10 pixels, bbox [0, 0, 1, 1].
# ParcelA-left:  [0.0, 0.0] to [0.3, 0.5]  (left side)
# ParcelA-right: [0.7, 0.0] to [1.0, 0.5]  (right side)
# ParcelB:       [0.3, 0.5] to [0.7, 1.0]  (center-top)

BBOX = [0.0, 0.0, 1.0, 1.0]
W, H = 10, 10

def _poly(west, south, east, north):
    """Build a GeoJSON Polygon geometry from bbox corners."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south], [east, south], [east, north],
            [west, north], [west, south],
        ]],
    }

GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "ParcelA", "layer": "x", "id": 1},
            "geometry": _poly(0.0, 0.0, 0.3, 0.5),
        },
        {
            "type": "Feature",
            "properties": {"name": "ParcelA", "layer": "x", "id": 2},
            "geometry": _poly(0.7, 0.0, 1.0, 0.5),
        },
        {
            "type": "Feature",
            "properties": {"name": "ParcelB", "layer": "x", "id": 3},
            "geometry": _poly(0.3, 0.5, 0.7, 1.0),
        },
    ],
}


@pytest.fixture
def geojson_path(tmp_path):
    """Write the test GeoJSON to a temp file and return its path."""
    p = tmp_path / "test.geojson"
    p.write_text(json.dumps(GEOJSON))
    return p


# ---------------------------------------------------------------------------
# Tests: rasterize_parcels
# ---------------------------------------------------------------------------

class TestRasterizeParcels:
    def test_unique_parcel_names(self, geojson_path):
        """Returned parcel names should be deduplicated."""
        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        assert names == sorted(set(names)), (
            f"parcel_names should be unique and sorted, got {names}"
        )
        assert "ParcelA" in names
        assert "ParcelB" in names
        assert len(names) == 2

    def test_all_fragments_rasterized(self, geojson_path):
        """Both ParcelA polygons must appear in the mask."""
        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        # The mask should have nonzero values in the regions of BOTH
        # ParcelA polygons (left and right sides of the grid).
        assert np.any(mask > 0), "mask should have some parcel pixels"

        # ParcelA is the first name (sorted), so it gets index 1.
        parcela_idx = names.index("ParcelA") + 1
        parcela_mask = mask == parcela_idx

        # Check left region (columns 0..2) and right region (columns 7..9)
        # have some ParcelA pixels each.
        left_has_a = np.any(parcela_mask[:, :3])
        right_has_a = np.any(parcela_mask[:, 7:])
        assert left_has_a, "ParcelA left fragment not in mask"
        assert right_has_a, "ParcelA right fragment not in mask"

    def test_parcelb_separate(self, geojson_path):
        """ParcelB should have its own distinct index."""
        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        parcelb_idx = names.index("ParcelB") + 1
        parcelb_pixels = np.sum(mask == parcelb_idx)
        assert parcelb_pixels > 0, "ParcelB should have pixels in the mask"


# ---------------------------------------------------------------------------
# Tests: compute_timeseries
# ---------------------------------------------------------------------------

def _write_tiff(path, data, bbox, w, h):
    """Write a single-band uint8 GeoTIFF."""
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path), "w", driver="GTiff", dtype="uint8",
        width=w, height=h, count=1, crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def satellite_dir(tmp_path, geojson_path):
    """Create a fake satellite data directory with one date."""
    date = "2024-01-01"
    date_dir = tmp_path / "sat" / date
    date_dir.mkdir(parents=True)

    # Create uniform rasters for each band and index.
    # Bands: constant reflectance = 0.5 -> uint8 = 128
    # Indices: constant index = 0.0 -> uint8 = 128
    uniform = np.full((H, W), 128, dtype=np.uint8)
    for b in BANDS:
        _write_tiff(date_dir / f"{b.lower()}.tif", uniform, BBOX, W, H)
    for idx in INDICES:
        _write_tiff(date_dir / f"{idx}.tif", uniform, BBOX, W, H)

    return tmp_path / "sat", [date], geojson_path


class TestComputeTimeseries:
    def test_no_duplicate_parcels_in_output(self, satellite_dir):
        """The timeseries parcels list should have no duplicates."""
        sat_dir, dates, geojson_path = satellite_dir
        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        ts = compute_timeseries(mask, names, dates, sat_dir)

        assert ts["parcels"] == sorted(set(ts["parcels"])), (
            f"timeseries parcels should be unique, got {ts['parcels']}"
        )

    def test_multi_polygon_parcel_uses_all_fragments(self, satellite_dir):
        """Per-parcel average should reflect pixels from ALL fragments.

        With uniform rasters the values are the same everywhere, so we can't
        distinguish by value. Instead, we create a raster where the two
        ParcelA regions have different values and verify the average spans both.
        """
        sat_dir, dates, geojson_path = satellite_dir

        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        parcela_idx = names.index("ParcelA") + 1

        # Create a non-uniform NDVI raster:
        # Left half = index +0.5 (uint8 = 191), right half = index -0.5 (uint8 = 64)
        ndvi = np.full((H, W), 128, dtype=np.uint8)
        ndvi[:, :5] = 191   # left half: +0.5
        ndvi[:, 5:] = 64    # right half: -0.5

        date_dir = sat_dir / dates[0]
        _write_tiff(date_dir / "ndvi.tif", ndvi, BBOX, W, H)

        ts = compute_timeseries(mask, names, dates, sat_dir)

        # ParcelA spans both left and right regions. If only the last fragment
        # is used (the bug), we'd get either ~+0.5 or ~-0.5. With both
        # fragments, the average should be somewhere in between.
        parcela_ndvi = ts["means"]["parcels"]["ParcelA"]["ndvi"][0]

        # Get the individual fragment values for comparison
        parcela_bool = mask == parcela_idx
        left_count = np.sum(parcela_bool[:, :5])
        right_count = np.sum(parcela_bool[:, 5:])

        # With the bug: only one fragment is used, so the average would be
        # close to +0.5 or -0.5. With the fix, it should be between them.
        assert left_count > 0 and right_count > 0, (
            "Both ParcelA fragments should have pixels "
            f"(left={left_count}, right={right_count})"
        )

        # The average should not be close to either extreme alone.
        # With both fragments, if roughly equal pixel counts, it should be ~0.
        # We just check it's strictly between the two extremes.
        assert -0.4 < parcela_ndvi < 0.4, (
            f"ParcelA NDVI average should use both fragments, got {parcela_ndvi}"
        )

    def test_forest_average_includes_all_pixels(self, satellite_dir):
        """Forest-wide average should include pixels from all parcels."""
        sat_dir, dates, geojson_path = satellite_dir
        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        ts = compute_timeseries(mask, names, dates, sat_dir)

        forest_count = np.sum(mask > 0)
        assert forest_count > 0
        # With uniform data (128 everywhere), forest average should be ~0.5
        # for reflectance bands, ~0.0 for index bands
        b02_avg = ts["means"]["forest"]["b02"][0]
        assert abs(b02_avg - 0.5020) < 0.02, f"forest b02 avg should be ~0.5, got {b02_avg}"

    def test_per_parcel_keys_match_parcel_list(self, satellite_dir):
        """means.parcels keys should match the parcels list exactly."""
        sat_dir, dates, geojson_path = satellite_dir
        mask, names = rasterize_parcels(geojson_path, BBOX, W, H)
        ts = compute_timeseries(mask, names, dates, sat_dir)

        assert set(ts["means"]["parcels"].keys()) == set(ts["parcels"])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
