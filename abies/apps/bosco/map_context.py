"""Build local OSM roads/water and Copernicus DSM contour vectors."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import contourpy
from geographiclib.geodesic import Geodesic
import numpy as np
import rasterio
from rasterio.merge import merge
import requests
from shapely import make_valid
from shapely.geometry import LineString, Polygon, box, mapping
from shapely.ops import polygonize_full, unary_union

from .map_context_data import (
    CONTEXT_FILE, SCHEMA_VERSION, MapContextError, publish_context, read_context,
    region_directory, region_geometry,
)

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'
COPERNICUS_URL = 'https://copernicus-dem-30m.s3.amazonaws.com'
ROAD_TYPES = (
    'motorway|trunk|primary|secondary|tertiary|unclassified|residential|'
    'service|track|path|footway|bridleway|steps|living_street|'
    'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link'
)


def buffered_bounds(bounds, meters):
    if not math.isfinite(meters) or not 0 <= meters <= 10000:
        raise MapContextError('Buffer must be between 0 and 10000 meters')
    west, south, east, north = bounds
    if not all(math.isfinite(v) for v in bounds) or not (-180 < west < east < 180 and -85 < south < north < 85):
        raise MapContextError('Invalid parcel bounds')
    geod = Geodesic.WGS84
    return [
        geod.Direct((south + north) / 2, west, 270, meters)['lon2'],
        geod.Direct(south, (west + east) / 2, 180, meters)['lat2'],
        geod.Direct((south + north) / 2, east, 90, meters)['lon2'],
        geod.Direct(north, (west + east) / 2, 0, meters)['lat2'],
    ]


def overpass_query(bounds):
    west, south, east, north = bounds
    bbox = f'({south},{west},{north},{east})'
    return (
        '[out:json][timeout:90];('
        f'way[highway~"^({ROAD_TYPES})$"]{bbox};'
        f'way[waterway~"^(river|stream|canal|drain|ditch)$"]{bbox};'
        f'way[natural=water]{bbox};relation[type=multipolygon][natural=water]{bbox};'
        f'way[waterway=riverbank]{bbox};relation[type=multipolygon][waterway=riverbank]{bbox};'
        f'way[landuse=reservoir]{bbox};relation[type=multipolygon][landuse=reservoir]{bbox};'
        ');out body geom;'
    )


def fetch_osm(bounds, endpoint=OVERPASS_URL):
    try:
        response = requests.post(endpoint, data={'data': overpass_query(bounds)},
                                 headers={'User-Agent': 'Abies map-context builder'}, timeout=(15, 120))
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MapContextError(f'OSM download failed: {exc}') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('elements'), list) or payload.get('remark'):
        raise MapContextError('OSM returned an incomplete or invalid result')
    return payload


def _coordinates(geometry):
    if not geometry or any('lon' not in p or 'lat' not in p for p in geometry):
        raise MapContextError('Incomplete OSM geometry')
    result = [(p['lon'], p['lat']) for p in geometry]
    if any(not math.isfinite(x) or not math.isfinite(y) or abs(x) > 180 or abs(y) > 90
           for x, y in result):
        raise MapContextError('Invalid OSM coordinates')
    return result


def _water_polygon(element):
    if element['type'] == 'way':
        points = _coordinates(element.get('geometry'))
        if len(points) < 4 or points[0] != points[-1]:
            raise MapContextError('An OSM water polygon is not closed')
        return make_valid(Polygon(points))
    rings = {'outer': [], 'inner': []}
    for member in element.get('members', []):
        role = member.get('role') or 'outer'
        if member['type'] == 'way' and role in rings:
            rings[role].append(LineString(_coordinates(member.get('geometry'))))
    areas = {}
    for role, lines in rings.items():
        polygons, cuts, dangles, invalid = polygonize_full(lines)
        if lines and (polygons.is_empty or not cuts.is_empty or not dangles.is_empty or not invalid.is_empty):
            raise MapContextError('Incomplete OSM water multipolygon')
        areas[role] = unary_union(polygons)
    if areas['outer'].is_empty:
        raise MapContextError('OSM water multipolygon has no outer boundary')
    return make_valid(areas['outer'].difference(areas['inner']))


def _parts(geometry, kind):
    if geometry.is_empty:
        return
    if geometry.geom_type == kind:
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _parts(part, kind)


def _feature(geometry, properties):
    def rounded(value):
        if isinstance(value, (list, tuple)):
            return [rounded(v) for v in value]
        return round(float(value), 6)
    geo = mapping(geometry)
    return {'type': 'Feature', 'properties': properties,
            'geometry': {'type': geo['type'], 'coordinates': rounded(geo['coordinates'])}}


def osm_features(payload, bounds):
    extent = box(*bounds)
    elements = payload.get('elements', [])
    water = lambda tags: tags.get('natural') == 'water' or tags.get('waterway') == 'riverbank' or tags.get('landuse') == 'reservoir'
    member_ids = {m['ref'] for e in elements if e['type'] == 'relation' and water(e.get('tags', {}))
                  for m in e.get('members', []) if m['type'] == 'way'}
    features = []
    for element in elements:
        tags = element.get('tags', {})
        if element['type'] == 'way' and tags.get('highway') in ROAD_TYPES.split('|'):
            layer, kind, geometry = 'roads', tags['highway'], LineString(_coordinates(element.get('geometry')))
        elif element['type'] == 'way' and tags.get('waterway') in ('river', 'stream', 'canal', 'drain', 'ditch'):
            layer, kind, geometry = 'water', tags['waterway'], LineString(_coordinates(element.get('geometry')))
        elif water(tags) and (element['type'] == 'relation' or element['id'] not in member_ids):
            layer, kind, geometry = 'water', 'water', _water_polygon(element)
        else:
            continue
        geometry = make_valid(geometry).intersection(extent).simplify(.00001, preserve_topology=True)
        for part in _parts(geometry, 'Polygon' if kind == 'water' else 'LineString'):
            features.append(_feature(part, {'layer': layer, 'kind': kind}))
    return features


def copernicus_tiles(bounds):
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -85 < south < north < 85):
        raise MapContextError('Invalid elevation extent')
    urls = []
    for latitude in range(math.floor(south), math.ceil(north)):
        for longitude in range(math.floor(west), math.ceil(east)):
            tile = (f'Copernicus_DSM_COG_10_{"N" if latitude >= 0 else "S"}{abs(latitude):02}_00_'
                    f'{"E" if longitude >= 0 else "W"}{abs(longitude):03}_00_DEM')
            urls.append(f'{COPERNICUS_URL}/{tile}/{tile}.tif')
    return urls


def elevation_grid(bounds, dem=None):
    """Read only COG windows for this extent; no account or entire tile download."""
    if (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) * 3600 ** 2 > 4_000_000:
        raise MapContextError('Elevation extent exceeds four million cells')
    sources = [str(dem)] if dem else copernicus_tiles(bounds)
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          GDAL_HTTP_TIMEOUT='90', GDAL_HTTP_MAX_RETRY='2'):
            with ExitStack() as stack:
                datasets = [stack.enter_context(rasterio.open(source)) for source in sources]
                if any(dataset.crs != rasterio.CRS.from_epsg(4326) for dataset in datasets):
                    raise MapContextError('Elevation input must use WGS84 (EPSG:4326)')
                grid, transform = merge(datasets, bounds=bounds, res=1 / 3600,
                                        nodata=-32767, masked=True, dtype='float32')
        data = np.ma.masked_invalid(grid[0])
        if data.count() == 0 or np.any(np.ma.getmaskarray(data)):
            raise MapContextError('Elevation source does not cover the full extent')
        return data, transform
    except rasterio.errors.RasterioError as exc:
        raise MapContextError(f'Elevation download/read failed: {exc}') from exc


def contour_features(data, transform, bounds, interval):
    if not math.isfinite(interval) or interval <= 0 or 100 % interval:
        raise MapContextError('Contour interval must be a positive divisor of 100 meters')
    ny, nx = data.shape
    x = transform.c + (np.arange(nx) + .5) * transform.a
    y = transform.f + (np.arange(ny) + .5) * transform.e
    generator = contourpy.contour_generator(x=x, y=y, z=data, corner_mask=False, line_type='Separate')
    low, high = math.ceil(float(data.min()) / interval), math.floor(float(data.max()) / interval)
    if high - low > 2000:
        raise MapContextError('Too many contour levels')
    extent = box(*bounds)
    features = []
    for level in range(low, high + 1):
        elevation = level * interval
        for points in generator.lines(elevation):
            line = LineString(points).intersection(extent).simplify(.00001)
            for part in _parts(line, 'LineString'):
                features.append(_feature(part, {'layer': 'contours', 'elevation': elevation}))
    return features


def build_region(geojson, region, output_dir, *, buffer_m=500, interval=20,
                 osm_file=None, dem=None, force=False, endpoint=OVERPASS_URL):
    fingerprint, parcel_bounds = region_geometry(geojson, region)
    bounds = buffered_bounds(parcel_bounds, buffer_m)
    target = region_directory(output_dir, region) / CONTEXT_FILE
    parameters = {'geometry_hash': fingerprint, 'buffer_m': buffer_m, 'contour_interval_m': interval,
                  'dem_source': str(dem) if dem else 'copernicus-glo-30',
                  'osm_source': str(osm_file) if osm_file else endpoint}
    if target.is_file() and not force:
        previous = read_context(target)
        if previous.get('schema_version') == SCHEMA_VERSION and previous.get('build') == parameters:
            return previous, False
    osm = json.loads(Path(osm_file).read_text()) if osm_file else fetch_osm(bounds, endpoint)
    if not isinstance(osm.get('elements'), list) or osm.get('remark'):
        raise MapContextError('Incomplete OSM data')
    features = osm_features(osm, bounds)
    data, transform = elevation_grid(bounds, dem)
    features.extend(contour_features(data, transform, bounds, interval))
    payload = {'type': 'FeatureCollection', 'available': True, 'schema_version': SCHEMA_VERSION,
               'region': region, 'bbox': bounds, 'parcel_bbox': parcel_bounds, 'build': parameters,
               'built_at': datetime.now(timezone.utc).isoformat(),
               'osm_timestamp': osm.get('osm3s', {}).get('timestamp_osm_base'),
               'features': features}
    publish_context(target, payload)
    return payload, True
