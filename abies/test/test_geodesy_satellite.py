"""Satellite dates and derived data must use the canonical WGS84 raster grid."""
from unittest.mock import Mock

import pytest
import rasterio
from affine import Affine

from apps.bosco import satellite
from test.test_build_satellite import _feature, _write_complete_date, _write_geojson


def test_satellite_dimensions_use_wgs84_ground_distances():
    # Independent GeographicLib WGS84 inverse reference at the bbox midlines.
    assert satellite.pixel_dims([16.3, 38.5, 16.36, 38.545]) == (523, 500)


def test_precompute_rejects_noncanonical_raster_grid(tmp_path):
    geojson = tmp_path / 'terreni.geojson'
    _write_geojson(geojson, [_feature('Forest', '1')])
    _write_complete_date(tmp_path, geojson, 'Forest', '2026-01-01', dimensions=(7, 9))

    with pytest.raises(satellite.SatelliteError, match='grid'):
        satellite.precompute_region(geojson, tmp_path, 'Forest')


def test_force_fetch_rebuilds_all_requested_dates_on_canonical_grid(tmp_path, monkeypatch):
    geojson = tmp_path / 'terreni.geojson'
    _write_geojson(geojson, [_feature('Forest', '1')])
    _write_complete_date(tmp_path, geojson, 'Forest', '2026-01-01', dimensions=(7, 9))
    monkeypatch.setattr(satellite, 'get_access_token', lambda: 'token')
    fetch = Mock(return_value=True)
    monkeypatch.setattr(satellite, 'fetch_one_date', fetch)

    assert satellite.fetch_region(geojson, tmp_path, 'Forest', ['2026-01-01', '2026-02-01'], force=True) == (2, 0)

    assert all(call.args[2:4] == (11, 11) for call in fetch.call_args_list)


def test_fetch_rejects_incompatible_retained_dates(tmp_path, monkeypatch):
    geojson = tmp_path / 'terreni.geojson'
    _write_geojson(geojson, [_feature('Forest', '1')])
    _write_complete_date(tmp_path, geojson, 'Forest', '2026-01-01', dimensions=(7, 9))
    fetch = Mock()
    monkeypatch.setattr(satellite, 'fetch_one_date', fetch)
    with pytest.raises(satellite.SatelliteError, match='grid'):
        satellite.fetch_region(geojson, tmp_path, 'Forest', ['2026-02-01'])
    fetch.assert_not_called()


def test_precompute_rejects_mixed_raster_grids(tmp_path):
    geojson = tmp_path / 'terreni.geojson'
    _write_geojson(geojson, [_feature('Forest', '1')])
    _write_complete_date(tmp_path, geojson, 'Forest', '2026-01-01')
    _write_complete_date(tmp_path, geojson, 'Forest', '2026-02-01', dimensions=(12, 11))
    (tmp_path / 'manifest.json').write_text('previous manifest')
    (tmp_path / 'parcel-mask.tif').write_bytes(b'previous mask')

    with pytest.raises(satellite.SatelliteError, match='grid'):
        satellite.precompute_region(geojson, tmp_path, 'Forest')
    assert (tmp_path / 'manifest.json').read_text() == 'previous manifest'
    assert (tmp_path / 'parcel-mask.tif').read_bytes() == b'previous mask'


@pytest.mark.parametrize('change', ['transform', 'crs'])
def test_precompute_rejects_shifted_or_wrong_crs_band(tmp_path, change):
    geojson = tmp_path / 'terreni.geojson'
    _write_geojson(geojson, [_feature('Forest', '1')])
    _write_complete_date(tmp_path, geojson, 'Forest', '2026-01-01')
    with rasterio.open(tmp_path / '2026-01-01' / 'ndvi.tif', 'r+') as raster:
        if change == 'crs':
            raster.crs = rasterio.CRS.from_epsg(3857)
        else:
            raster.transform = raster.transform @ Affine.translation(1, 0)

    with pytest.raises(satellite.SatelliteError, match='grid'):
        satellite.precompute_region(geojson, tmp_path, 'Forest')
    assert not (tmp_path / 'manifest.json').exists()


@pytest.mark.parametrize('bbox', [[16.4, 38.5, 16.3, 38.6], [16.3, 38.6, 16.4, 38.5]])
def test_satellite_dimensions_reject_reversed_bounds(bbox):
    with pytest.raises(satellite.SatelliteError, match='bounds'):
        satellite.pixel_dims(bbox)
