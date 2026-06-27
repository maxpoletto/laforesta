"""Tests for the Bosco satellite build command."""

import json
from io import StringIO

import numpy as np
from django.core.management import call_command

from apps.bosco import satellite


def _feature(region, parcel, x0=0.0):
    x1 = x0 + 0.001
    return {
        'type': 'Feature',
        'properties': {'layer': region, 'name': f'{region}-{parcel}'},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [x0, 0.0], [x1, 0.0], [x1, 0.001], [x0, 0.001], [x0, 0.0],
            ]],
        },
    }


def _write_geojson(path, features):
    path.write_text(json.dumps({
        'type': 'FeatureCollection',
        'features': features,
    }), encoding='utf-8')


def _read_lines(path):
    return path.read_text(encoding='utf-8').splitlines()


def _write_complete_date(region_dir, geojson, region, date, value=128):
    bbox = satellite.bbox_from_geojson(geojson, region)
    width, height = satellite.pixel_dims(bbox)
    arr = np.full((height, width), value, dtype=np.uint8)
    for band in [b.lower() for b in satellite.BANDS]:
        satellite.save_geotiff(region_dir / date / f'{band}.tif', arr, bbox, width, height)
    for index in satellite.INDICES:
        satellite.save_geotiff(region_dir / date / f'{index}.tif', arr, bbox, width, height)


def test_regions_from_geojson_returns_sorted_layers(tmp_path):
    geojson = tmp_path / 'terreni.geojson'
    _write_geojson(geojson, [_feature('Serra', '1'), _feature('Capistrano', '2')])

    assert satellite.regions_from_geojson(geojson) == ['Capistrano', 'Serra']


def test_build_satellite_dates_writes_window_and_combined_files(tmp_path, monkeypatch):
    geojson = tmp_path / 'terreni.geojson'
    output = tmp_path / 'satellite'
    _write_geojson(geojson, [_feature('Serra', '1')])

    def fake_find(geojson_path, region, months, max_cloud, year_start, year_end):
        assert geojson_path == geojson
        assert region == 'Serra'
        assert year_start == 2024
        assert year_end == 2025
        if months == {6, 7}:
            assert max_cloud == 1.0
            return ['2024-07-01', '2025-06-20']
        if months == {1, 2}:
            assert max_cloud == 10.0
            return ['2024-02-15', '2025-01-12']
        raise AssertionError(months)

    monkeypatch.setattr(satellite, 'find_region_dates', fake_find)

    call_command(
        'build_satellite', 'dates', '--geojson', str(geojson), '--output-dir', str(output),
        '--year-start', '2024', '--year-end', '2025', stdout=StringIO(),
    )

    region_dir = output / 'Serra'
    assert _read_lines(region_dir / 'dates-summer.txt') == ['2024-07-01', '2025-06-20']
    assert _read_lines(region_dir / 'dates-winter.txt') == ['2024-02-15', '2025-01-12']
    assert _read_lines(region_dir / 'dates.txt') == [
        '2024-02-15', '2024-07-01', '2025-01-12', '2025-06-20',
    ]


def test_build_satellite_fetch_skips_complete_dates_without_token(tmp_path, monkeypatch):
    geojson = tmp_path / 'terreni.geojson'
    output = tmp_path / 'satellite'
    region_dir = output / 'Serra'
    _write_geojson(geojson, [_feature('Serra', '1')])
    _write_complete_date(region_dir, geojson, 'Serra', '2026-07-01')
    satellite.write_dates_file(region_dir / 'dates.txt', ['2026-07-01'])

    def fail_token():
        raise AssertionError('complete dates should not request CDSE credentials')

    monkeypatch.setattr(satellite, 'get_access_token', fail_token)
    stdout = StringIO()

    call_command(
        'build_satellite', 'fetch', '--geojson', str(geojson), '--output-dir', str(output),
        '--region', 'Serra', stdout=stdout,
    )

    assert 'fetched 0 date(s); skipped 1' in stdout.getvalue()


def test_build_satellite_precompute_writes_manifest_mask_and_timeseries(tmp_path):
    geojson = tmp_path / 'terreni.geojson'
    output = tmp_path / 'satellite'
    region_dir = output / 'Serra'
    _write_geojson(geojson, [_feature('Serra', '1')])
    _write_complete_date(region_dir, geojson, 'Serra', '2026-07-01', value=192)

    call_command(
        'build_satellite', 'precompute', '--geojson', str(geojson), '--output-dir', str(output),
        '--region', 'Serra', stdout=StringIO(),
    )

    manifest = json.loads((region_dir / 'manifest.json').read_text(encoding='utf-8'))
    timeseries = json.loads((region_dir / 'timeseries.json').read_text(encoding='utf-8'))
    assert manifest['dates'] == ['2026-07-01']
    assert manifest['bands'] == ['b02', 'b04', 'b08', 'b11']
    assert (region_dir / 'parcel-mask.tif').is_file()
    assert timeseries['dates'] == ['2026-07-01']
    assert timeseries['parcels'] == ['Serra-1']
    assert timeseries['means']['forest']['ndvi'] == [round(192 / 127.5 - 1, 4)]
    assert timeseries['means']['parcels']['Serra-1']['b02'] == [round(192 / 255, 4)]
