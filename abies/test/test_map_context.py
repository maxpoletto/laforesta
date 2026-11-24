"""Vector-map builds are reproducible, geometrically sound and atomically served."""
import gzip
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, shape

from apps.base.map_keys import initial_map_key
from apps.bosco import map_context as M
from apps.bosco.map_context_data import (
    CONTEXT_FILE, MapContextError, publish_context, read_context, region_directory, region_geometry,
)


def polygon(west=16, south=38, east=16.01, north=38.01):
    return {'type': 'Polygon', 'coordinates': [[[west, south], [east, south],
            [east, north], [west, north], [west, south]]]}


@pytest.fixture
def geojson():
    return {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'layer': 'Forest'}, 'geometry': polygon()},
    ]}


def osm_geometry(points):
    return [{'lon': x, 'lat': y} for x, y in points]


def test_region_extent_identity_and_safe_paths(geojson, tmp_path):
    fingerprint, bounds = region_geometry(geojson, 'Forest')
    assert bounds == [16, 38, 16.01, 38.01]
    enlarged = M.buffered_bounds(bounds, 500)
    assert enlarged[0] < 16 < 16.01 < enlarged[2]
    assert enlarged[1] < 38 < 38.01 < enlarged[3]
    geojson['features'][0]['geometry']['coordinates'][0][1][0] += .001
    assert region_geometry(geojson, 'Forest')[0] != fingerprint
    for name in ('..', '.', '../outside', '/tmp', 'nested/region'):
        with pytest.raises(MapContextError):
            region_directory(tmp_path, name)
    (tmp_path / 'symlink').symlink_to('/tmp')
    with pytest.raises(MapContextError):
        region_directory(tmp_path, 'symlink')
    with pytest.raises(MapContextError):
        region_geometry(geojson, 'Missing')
    for buffer in (-1, float('nan'), 10001):
        with pytest.raises(MapContextError):
            M.buffered_bounds(bounds, buffer)


def test_osm_clips_lines_and_preserves_water_holes():
    outer = [[15.999, 37.999], [16.009, 37.999], [16.009, 38.009], [15.999, 38.009], [15.999, 37.999]]
    inner = [[16.003, 38.003], [16.005, 38.003], [16.005, 38.005], [16.003, 38.005], [16.003, 38.003]]
    payload = {'elements': [
        {'type': 'way', 'id': 1, 'tags': {'highway': 'track'},
         'geometry': osm_geometry([[15.99, 38.001], [16.02, 38.001]])},
        {'type': 'way', 'id': 2, 'tags': {'waterway': 'stream'},
         'geometry': osm_geometry([[16.001, 38.001], [16.005, 38.005]])},
        {'type': 'way', 'id': 3, 'tags': {'natural': 'water'}, 'geometry': osm_geometry(outer)},
        {'type': 'relation', 'id': 9, 'tags': {'natural': 'water'}, 'members': [
            {'type': 'way', 'ref': 3, 'role': 'outer', 'geometry': osm_geometry(outer[:3])},
            {'type': 'way', 'ref': 4, 'role': 'outer', 'geometry': osm_geometry(outer[2:])},
            {'type': 'way', 'ref': 5, 'role': 'inner', 'geometry': osm_geometry(inner)},
        ]},
    ]}
    features = M.osm_features(payload, [16, 38, 16.01, 38.01])
    assert len(features) == 3, 'do not fill a multipolygon hole with a duplicate member-way polygon'
    road, stream, water = features
    assert road['geometry']['coordinates'] == [[16, 38.001], [16.01, 38.001]]
    assert stream['properties']['layer'] == 'water'
    lake = shape(water['geometry'])
    assert lake.is_valid and len(lake.interiors) == 1
    assert not lake.contains(Point(16.004, 38.004))
    assert lake.contains(Point(16.002, 38.002))
    payload['elements'][-1]['members'].pop(1)
    with pytest.raises(MapContextError, match='Incomplete'):
        M.osm_features(payload, [16, 38, 16.01, 38.01])


def test_osm_download_rejects_partial_results(monkeypatch):
    response = Mock()
    response.json.return_value = {'elements': [], 'remark': 'runtime error: timeout'}
    monkeypatch.setattr(M.requests, 'post', Mock(return_value=response))
    with pytest.raises(MapContextError, match='incomplete'):
        M.fetch_osm([16, 38, 16.01, 38.01])
    response.json.return_value = {'elements': []}
    assert M.fetch_osm([16, 38, 16.01, 38.01]) == {'elements': []}
    query = M.requests.post.call_args.kwargs['data']['data']
    assert '(38,16,38.01,16.01)' in query
    assert 'relation' in query and 'out body geom' in query


def test_contours_interpolate_pixel_centers_and_do_not_bridge_nodata():
    data = np.tile(np.arange(5, dtype=float) * 10, (5, 1))
    transform = from_origin(16, 38.01, .001, .001)
    features = M.contour_features(data, transform, [16, 38, 16.01, 38.01], 20)
    twenty = next(f for f in features if f['properties']['elevation'] == 20)
    assert all(point[0] == pytest.approx(16.0025) for point in twenty['geometry']['coordinates'])
    assert all(38.005 <= point[1] <= 38.01 for point in twenty['geometry']['coordinates'])
    masked = np.ma.array(data, mask=False)
    masked.mask[2, :] = True
    segments = [shape(f['geometry']) for f in M.contour_features(masked, transform,
                [16, 38, 16.01, 38.01], 20) if f['properties']['elevation'] == 20]
    assert len(segments) == 2
    assert not any(line.bounds[1] < 38.0075 < line.bounds[3] for line in segments)
    for interval in (0, -1, 30):
        with pytest.raises(MapContextError):
            M.contour_features(data, transform, [16, 38, 16.01, 38.01], interval)
    assert 'N38_00_E016_00_DEM' in M.copernicus_tiles([16, 38, 16.01, 38.01])[0]
    assert len(M.copernicus_tiles([15.99, 37.99, 16.01, 38.01])) == 4


def test_build_atomic_reuse_and_failure_preserves_snapshot(geojson, tmp_path, monkeypatch):
    osm = tmp_path / 'osm.json'
    osm.write_text('{"elements": []}')
    dem = tmp_path / 'dem.tif'
    data = np.tile(np.arange(100, dtype=np.float32), (100, 1))
    with rasterio.open(dem, 'w', driver='GTiff', height=100, width=100, count=1,
                       dtype='float32', crs='EPSG:4326',
                       transform=from_origin(15.99, 38.02, 1 / 3600, 1 / 3600)) as dataset:
        dataset.write(data, 1)
    output = tmp_path / 'context'
    payload, changed = M.build_region(geojson, 'Forest', output, buffer_m=0, osm_file=osm, dem=dem)
    assert changed and payload['features']
    target = output / str(initial_map_key('Forest')) / CONTEXT_FILE
    before = target.read_bytes()
    assert read_context(target) == payload
    assert M.build_region(geojson, 'Forest', output, buffer_m=0, osm_file=osm, dem=dem)[1] is False
    monkeypatch.setattr(M, 'elevation_grid', Mock(side_effect=MapContextError('Offline')))
    with pytest.raises(MapContextError):
        M.build_region(geojson, 'Forest', output, buffer_m=0, osm_file=osm, dem=dem, force=True)
    assert target.read_bytes() == before
    with pytest.raises(ValueError):
        publish_context(target, {'invalid': float('nan')})
    assert target.read_bytes() == before
    assert list(target.parent.glob('.context-*')) == []


@pytest.mark.django_db
def test_context_reader_access_conditional_get_and_stale_geometry(client, reader_user, regions, settings):
    from django.urls import reverse
    region = regions[0]
    geojson = {'features': [{'properties': {'layer': region.name}, 'geometry': polygon()}]}
    from apps.base.models import MapFeature
    feature = MapFeature.objects.create(members={'type': 'Feature', 'properties': {'layer': region.name}},
                                        geometry=polygon())
    target = settings.MAP_CONTEXT_DIR / str(region.map_key) / CONTEXT_FILE
    fingerprint = region_geometry(geojson, region.name)[0]
    publish_context(target, {'available': True, 'schema_version': 1, 'region_key': str(region.map_key),
                            'build': {'geometry_hash': fingerprint}, 'features': []})
    url = reverse('bosco-map-context', args=[region.id])
    assert client.get(url).status_code == 302
    client.force_login(reader_user)
    response = client.get(url)
    assert response.status_code == 200
    assert response['Content-Encoding'] == 'gzip'
    assert json.loads(gzip.decompress(b''.join(response.streaming_content)))['available']
    assert client.get(url, HTTP_IF_NONE_MATCH=response['ETag']).status_code == 304
    feature.geometry = polygon(east=16.02)
    feature.save()
    assert client.get(url).json() == {'available': False, 'stale': True}
    target.unlink()
    assert client.get(url).json() == {'available': False}
    assert client.get(reverse('bosco-map-context', args=[999999])).status_code == 404


@pytest.mark.django_db
def test_context_request_streams_the_snapshot_it_validated(client, reader_user, regions, settings, monkeypatch):
    from django.urls import reverse

    from apps.base.models import MapFeature
    from apps.bosco import map_context_data as storage

    region = regions[0]
    feature = {'properties': {'layer': region.name}, 'geometry': polygon()}
    MapFeature.objects.create(members={'properties': feature['properties']}, geometry=feature['geometry'])
    fingerprint, _ = region_geometry({'features': [feature]}, region.name)
    target = settings.MAP_CONTEXT_DIR / str(region.map_key) / CONTEXT_FILE
    snapshot = {'schema_version': 1, 'region_key': str(region.map_key), 'available': True,
                'build': {'geometry_hash': fingerprint}, 'features': []}
    replacement = {**snapshot, 'build': {'geometry_hash': 'wrong-geometry'}}
    publish_context(target, snapshot)
    read_stream = storage.read_context_stream

    def replace_after_read(stream):
        result = read_stream(stream)
        publish_context(target, replacement)
        return result

    url = reverse('bosco-map-context', args=[region.pk])
    client.force_login(reader_user)
    with monkeypatch.context() as race:
        race.setattr(storage, 'read_context_stream', replace_after_read)
        response = client.get(url)
        assert response.status_code == 200
        assert json.loads(gzip.decompress(b''.join(response.streaming_content))) == snapshot
    assert client.get(url).json() == {'available': False, 'stale': True}


def test_command_builds_selected_regions_and_keeps_reusable_output(db, geojson, tmp_path, monkeypatch):
    from django.core.management import call_command, CommandError
    from apps.base.models import Region
    from apps.bosco.management.commands import build_map_context as command
    source = tmp_path / 'geo.json'
    source.write_text(json.dumps(geojson))
    builder = Mock(return_value=({'features': []}, True))
    monkeypatch.setattr(command, 'build_region', builder)
    call_command('build_map_context', geojson=source, output_dir=tmp_path / 'output', region=['Forest'])
    assert builder.call_args.args[:2] == (geojson, 'Forest')
    assert builder.call_args.kwargs['buffer_m'] == 500
    assert builder.call_args.kwargs['interval'] == 20
    assert builder.call_args.kwargs['region_key'] == initial_map_key('Forest')
    region = Region.objects.create(name='Old name')
    region.name = 'Forest'
    region.save()
    call_command('build_map_context', geojson=source, output_dir=tmp_path / 'output')
    assert builder.call_args.kwargs['region_key'] == region.map_key
    builder.side_effect = MapContextError('Remote data incomplete')
    with pytest.raises(CommandError, match='incomplete'):
        call_command('build_map_context', geojson=source, output_dir=tmp_path / 'output')
    geojson['features'][0]['properties']['layer'] = 'Old name'
    source.write_text(json.dumps(geojson))
    with pytest.raises(CommandError, match='Forest'):
        call_command('build_map_context', geojson=source, output_dir=tmp_path / 'output')


@pytest.mark.parametrize('payload', [[], {}, {'elements': None}, {'elements': [], 'remark': 'timeout'}])
def test_osm_download_rejects_malformed_responses(payload, monkeypatch):
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(M.requests, 'post', Mock(return_value=response))
    with pytest.raises(MapContextError, match='invalid result'):
        M.fetch_osm([16, 38, 16.01, 38.01])


@pytest.mark.parametrize('error', [M.requests.Timeout('Timed out'), ValueError('Invalid JSON')])
def test_osm_network_and_decode_errors_are_reported(error, monkeypatch):
    monkeypatch.setattr(M.requests, 'post', Mock(side_effect=error))
    with pytest.raises(MapContextError, match='OSM download failed'):
        M.fetch_osm([16, 38, 16.01, 38.01])


@pytest.mark.parametrize('coordinates', [None, [{'lon': 16}], [{'lat': 38}],
                                       [{'lon': 181, 'lat': 38}], [{'lon': 16, 'lat': float('nan')}]])
def test_osm_rejects_missing_and_invalid_coordinates(coordinates):
    payload = {'elements': [{'type': 'way', 'id': 1, 'tags': {'highway': 'track'},
                            'geometry': coordinates}]}
    with pytest.raises(MapContextError, match='OSM'):
        M.osm_features(payload, [16, 38, 16.01, 38.01])


def test_water_ways_multipart_clipping_and_irrelevant_elements():
    # This concave water body is split in two by clipping to the upper strip.
    coast = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4), (0, 0)]
    payload = {'elements': [
        {'type': 'way', 'id': 1, 'tags': {'landuse': 'reservoir'}, 'geometry': osm_geometry(coast)},
        {'type': 'way', 'id': 2, 'tags': {'highway': 'track'},
         'geometry': osm_geometry([(-2, 3), (-1, 3)])},
        {'type': 'node', 'id': 3, 'tags': {}},
        {'type': 'way', 'id': 4, 'tags': {'highway': 'track'},
         'geometry': osm_geometry([(-1, 1), (0, 2)])},
    ]}
    features = M.osm_features(payload, [0, 2, 4, 4])
    assert len(features) == 2
    assert all(f['geometry']['type'] == 'Polygon' and shape(f['geometry']).is_valid for f in features)
    assert sum(shape(f['geometry']).area for f in features) == pytest.approx(4)
    payload['elements'][0]['geometry'] = osm_geometry(coast[:-1])
    with pytest.raises(MapContextError, match='not closed'):
        M.osm_features(payload, [0, 2, 4, 4])


def test_water_relation_without_outer_boundary_is_rejected():
    closed = osm_geometry([(16, 38), (16.01, 38), (16.01, 38.01), (16, 38)])
    relation = {'type': 'relation', 'id': 1, 'tags': {'natural': 'water'}, 'members': [
        {'type': 'node', 'ref': 2, 'role': 'label'},
        {'type': 'way', 'ref': 3, 'role': 'inner', 'geometry': closed},
    ]}
    with pytest.raises(MapContextError, match='no outer boundary'):
        M.osm_features({'elements': [relation]}, [16, 38, 16.01, 38.01])


def test_geometry_identity_handles_multipart_order_and_rejects_nonpolygons(geojson):
    first = geojson['features'][0]
    second = {'properties': {'layer': 'Forest'}, 'geometry': {'type': 'MultiPolygon',
              'coordinates': [polygon(16.02, 38.02, 16.03, 38.03)['coordinates']]}}
    geojson['features'].append(second)
    fingerprint, bounds = region_geometry(geojson, 'Forest')
    assert bounds == [16, 38, 16.03, 38.03]
    geojson['features'].reverse()
    first['properties']['parcel'] = 'Renamed'
    assert region_geometry(geojson, 'Forest')[0] == fingerprint
    first['geometry'] = {'type': 'Point', 'coordinates': [16, 38]}
    with pytest.raises(MapContextError, match='polygons'):
        region_geometry(geojson, 'Forest')
    geojson['features'] = [{'properties': {'layer': 'Forest'},
                           'geometry': {'type': 'Polygon', 'coordinates': []}}]
    with pytest.raises(MapContextError, match='Empty'):
        region_geometry(geojson, 'Forest')


@pytest.mark.parametrize('bounds', [[16, 38, 16, 38.01], [16, 90, 16.01, 91],
                                  [float('nan'), 38, 16.01, 38.01]])
def test_invalid_extents_are_rejected(bounds):
    with pytest.raises(MapContextError):
        M.buffered_bounds(bounds, 0)
    with pytest.raises(MapContextError):
        M.copernicus_tiles(bounds)


def test_elevation_input_failures_and_limits(tmp_path, monkeypatch):
    with pytest.raises(MapContextError, match='four million'):
        M.elevation_grid([16, 38, 20, 40])
    with pytest.raises(MapContextError, match='download/read failed'):
        M.elevation_grid([16, 38, 16.01, 38.01], tmp_path / 'missing.tif')
    dem = tmp_path / 'input.tif'
    for crs, data, message in [
        ('EPSG:3857', np.ones((10, 10), dtype='float32'), 'WGS84'),
        ('EPSG:4326', np.full((10, 10), -32767, dtype='float32'), 'full extent'),
    ]:
        with rasterio.open(dem, 'w', driver='GTiff', width=10, height=10, count=1,
                           dtype='float32', nodata=-32767, crs=crs,
                           transform=from_origin(16, 38.01, .001, .001)) as dataset:
            dataset.write(data, 1)
        with pytest.raises(MapContextError, match=message):
            M.elevation_grid([16, 38, 16.01, 38.01], dem)
    assert 'S39_00_W017_00_DEM' in M.copernicus_tiles([-16.9, -38.9, -16.8, -38.8])[0]
    with pytest.raises(MapContextError, match='Too many'):
        M.contour_features(np.array([[0, 10000], [0, 10000]]),
                           from_origin(16, 38.01, .001, .001), [16, 38, 16.01, 38.01], 1)


def test_default_source_build_rebuilds_changed_metadata_and_rejects_partial_osm(geojson, tmp_path, monkeypatch):
    fetch = Mock(return_value={'elements': [], 'osm3s': {'timestamp_osm_base': '2026-11-16T00:00:00Z'}})
    monkeypatch.setattr(M, 'fetch_osm', fetch)
    monkeypatch.setattr(M, 'elevation_grid', Mock(return_value=(
        np.tile(np.arange(5) * 20, (5, 1)), from_origin(16, 38.01, .001, .001))))
    payload, _ = M.build_region(geojson, 'Forest', tmp_path)
    assert payload['build']['dem_source'] == 'copernicus-glo-30'
    assert payload['osm_timestamp'] == '2026-11-16T00:00:00Z'
    target = tmp_path / str(initial_map_key('Forest')) / CONTEXT_FILE
    previous = dict(payload, schema_version=0)
    publish_context(target, previous)
    assert M.build_region(geojson, 'Forest', tmp_path)[1] is True
    assert fetch.call_count == 2
    osm = tmp_path / 'partial.json'
    osm.write_text('{"elements": [], "remark": "timeout"}')
    with pytest.raises(MapContextError, match='Incomplete OSM'):
        M.build_region(geojson, 'Forest', tmp_path, osm_file=osm)


def test_corrupt_snapshots_and_failed_publication_keep_previous_file(tmp_path, monkeypatch):
    from apps.bosco import map_context_data as storage
    target = tmp_path / CONTEXT_FILE
    publish_context(target, {'ready': True})
    previous = target.read_bytes()
    monkeypatch.setattr(storage.os, 'replace', Mock(side_effect=OSError('No space')))
    with pytest.raises(OSError, match='No space'):
        publish_context(target, {'ready': False})
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob('.context-*'))
    with gzip.open(target, 'wt') as stream:
        json.dump([], stream)
    with pytest.raises(MapContextError, match='Invalid map context'):
        read_context(target)


@pytest.mark.django_db
def test_context_corrupt_file_falls_back_to_plain_map(client, reader_user, regions, settings):
    from django.urls import reverse
    region = regions[0]
    target = settings.MAP_CONTEXT_DIR / str(region.map_key) / CONTEXT_FILE
    target.parent.mkdir(parents=True)
    target.write_bytes(b'Not a gzip file')
    client.force_login(reader_user)
    response = client.get(reverse('bosco-map-context', args=[region.id]))
    assert response.json() == {'available': False, 'stale': True}
    assert response['Cache-Control'] == 'no-cache'


def test_public_cog_reads_use_requested_window_and_preserve_heights(tmp_path, monkeypatch):
    dem = tmp_path / 'public-tile.tif'
    with rasterio.open(dem, 'w', driver='GTiff', width=100, height=100, count=1,
                       dtype='float32', crs='EPSG:4326',
                       transform=from_origin(16, 38.02, 1 / 3600, 1 / 3600)) as dataset:
        dataset.write(np.full((100, 100), 1800, dtype='float32'), 1)
    original_open = rasterio.open
    opened = []
    def open_public_tile(url):
        opened.append(url)
        return original_open(dem)
    monkeypatch.setattr(M.rasterio, 'open', open_public_tile)
    data, transform = M.elevation_grid([16.001, 38.001, 16.01, 38.01])
    assert len(opened) == 1 and opened[0].startswith(M.COPERNICUS_URL)
    assert data.max() == data.min() == 1800
    assert data.shape == (32, 32), 'read the local extent, rather than the entire source tile'
    assert transform.c == pytest.approx(16.001)
    assert transform.f == pytest.approx(38.01)
    assert M.contour_features(data, transform, [16.001, 38.001, 16.01, 38.01], 20) == []


def test_scottish_cog_windows_join_across_a_western_tile_seam(tmp_path, monkeypatch):
    arcsec = 1 / 3600
    seam = -4 - .75 * arcsec
    top = 57 + 41.5 * arcsec
    heights = np.repeat((180 + 10 * np.arange(24))[:, None], 16, axis=1)
    tiles = {}
    for longitude, left, offset in [('W005', seam - 24 * arcsec, 0), ('W004', seam, 5)]:
        path = tmp_path / f'{longitude}.tif'
        # GLO-30 uses 1.5″ longitude cells at this latitude and 1″ latitude cells.
        with rasterio.open(path, 'w', driver='GTiff', width=16, height=24, count=1,
                           dtype='float32', crs='EPSG:4326',
                           transform=from_origin(left, top, 1.5 * arcsec, arcsec)) as dataset:
            dataset.write((heights + offset).astype('float32'), 1)
        name = f'Copernicus_DSM_COG_10_N57_00_{longitude}_00_DEM'
        tiles[f'{M.COPERNICUS_URL}/{name}/{name}.tif'] = path

    open_raster = rasterio.open
    monkeypatch.setattr(M.rasterio, 'open', lambda url: open_raster(tiles[url]))
    bounds = [seam - 9 * arcsec, top - 20 * arcsec, seam + 12 * arcsec, top - 4 * arcsec]
    data, transform = M.elevation_grid(bounds)
    assert data.shape == (16, 21)
    assert (transform.a, transform.e) == pytest.approx((arcsec, -arcsec))
    assert np.all(data[:, :9] == heights[4:20, :1])
    assert np.all(data[:, 9:] == heights[4:20, :1] + 5)

    contour, = [feature for feature in M.contour_features(data, transform, bounds, 20)
                if feature['properties']['elevation'] == 300]
    points = contour['geometry']['coordinates']
    west_end, east_end = sorted((points[0], points[-1]))
    # Context coordinates are rounded to six decimal places.
    assert west_end == pytest.approx([bounds[0] + .5 * arcsec, top - 12.5 * arcsec], abs=5e-7)
    assert east_end == pytest.approx([bounds[2] - .5 * arcsec, top - 12 * arcsec], abs=5e-7)
