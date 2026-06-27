"""Sentinel-2 satellite data builder for Bosco."""

from __future__ import annotations

import io
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.transform import from_bounds


CATALOG_URL = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
TOKEN_URL = (
    'https://identity.dataspace.copernicus.eu'
    '/auth/realms/CDSE/protocol/openid-connect/token'
)
PROCESS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/process'

BANDS = ['B02', 'B04', 'B08', 'B11']
INDICES = ['ndvi', 'ndmi', 'evi']
EXPECTED_TIFFS = [b.lower() + '.tif' for b in BANDS] + [i + '.tif' for i in INDICES]
RESOLUTION_M = 10
FIRST_YEAR = 2015
DATE_DIR_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
WGS84 = rasterio.CRS.from_epsg(4326)


class SatelliteError(Exception):
    """Raised when satellite build input or remote data is invalid."""


def regions_from_geojson(path: Path) -> list[str]:
    """Return sorted region names from GeoJSON feature properties.layer."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    regions = {
        feature.get('properties', {}).get('layer')
        for feature in data.get('features') or []
    }
    return sorted(r for r in regions if r)


def bbox_from_geojson(path: Path, region: str | None = None) -> list[float]:
    """Return [west, south, east, north] bbox, optionally for one region."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    features = data.get('features') or []
    if region:
        features = [
            f for f in features
            if f.get('properties', {}).get('layer') == region
        ]
        if not features:
            raise SatelliteError(f"no features for region '{region}'")

    lons = []
    lats = []
    for feature in features:
        geometry = feature.get('geometry') or {}
        if geometry.get('type') == 'Polygon':
            rings = geometry.get('coordinates') or []
        elif geometry.get('type') == 'MultiPolygon':
            rings = [
                ring
                for polygon in geometry.get('coordinates') or []
                for ring in polygon
            ]
        else:
            continue
        for ring in rings:
            for coord in ring:
                lons.append(coord[0])
                lats.append(coord[1])

    if not lons or not lats:
        scope = f" for region '{region}'" if region else ''
        raise SatelliteError(f'no polygon coordinates found{scope}')
    return [min(lons), min(lats), max(lons), max(lats)]


def pixel_dims(bbox: list[float]) -> tuple[int, int]:
    """Compute (width, height) in pixels for a WGS84 bbox at RESOLUTION_M."""
    west, south, east, north = bbox
    mid_lat_rad = math.radians((south + north) / 2)
    width = round((east - west) * 111320 * math.cos(mid_lat_rad) / RESOLUTION_M)
    height = round((north - south) * 111320 / RESOLUTION_M)
    if width <= 0 or height <= 0:
        raise SatelliteError(f'invalid raster dimensions {width}x{height}')
    return width, height


def parse_months(value: str) -> set[int]:
    try:
        months = {int(v.strip()) for v in value.split(',') if v.strip()}
    except ValueError as exc:
        raise SatelliteError(f'invalid month list: {value}') from exc
    if not months or min(months) < 1 or max(months) > 12:
        raise SatelliteError(f'invalid month list: {value}')
    return months


def complete_date_dir(path: Path) -> bool:
    if not path.is_dir() or not DATE_DIR_RE.fullmatch(path.name):
        return False
    existing = {p.name for p in path.iterdir()}
    return all(name in existing for name in EXPECTED_TIFFS)


def discover_dates(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return [
        entry.name for entry in sorted(output_dir.iterdir())
        if complete_date_dir(entry)
    ]


def write_dates_file(path: Path, dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for date in sorted(set(dates)):
            f.write(date + '\n')


def read_dates_file(path: Path) -> list[str]:
    if not path.is_file():
        raise SatelliteError(f'{path} not found')
    dates = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                dates.append(line)
    if not dates:
        raise SatelliteError(f'no dates in {path}')
    return dates


def get_access_token() -> str:
    client_id = os.environ.get('CDSE_CLIENT_ID')
    client_secret = os.environ.get('CDSE_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise SatelliteError(
            'set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in the satellite env file'
        )
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SatelliteError(f'CDSE auth failed: {exc}') from exc
    if response.status_code != 200:
        raise SatelliteError(f'CDSE auth failed ({response.status_code}): {response.text}')
    return response.json()['access_token']


def find_region_dates(
        geojson_path: Path, region: str, months: set[int], max_cloud: float,
        year_start: int = FIRST_YEAR, year_end: int | None = None) -> list[str]:
    """Find one lowest-cloud scene per year for a region and month window."""
    bbox = bbox_from_geojson(geojson_path, region)
    year_end = year_end or datetime.now().year
    west, south, east, north = bbox
    wkt = (
        f'POLYGON(({west} {south},{east} {south},'
        f'{east} {north},{west} {north},{west} {south}))'
    )
    month_start = min(months)
    month_end = max(months)
    scenes = []

    for year in range(year_start, year_end + 1):
        date_start = f'{year}-{month_start:02d}-01T00:00:00.000Z'
        if month_end == 12:
            date_end = f'{year + 1}-01-01T00:00:00.000Z'
        else:
            date_end = f'{year}-{month_end + 1:02d}-01T00:00:00.000Z'

        filter_parts = [
            "Collection/Name eq 'SENTINEL-2'",
            "contains(Name, 'MSIL2A')",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
            f'ContentDate/Start gt {date_start}',
            f'ContentDate/Start lt {date_end}',
            (
                'Attributes/OData.CSC.DoubleAttribute/any('
                "att:att/Name eq 'cloudCover' and "
                f'att/OData.CSC.DoubleAttribute/Value lt {max_cloud})'
            ),
        ]
        try:
            response = requests.get(
                CATALOG_URL,
                params={
                    '$filter': ' and '.join(filter_parts),
                    '$top': 50,
                    '$expand': 'Attributes',
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SatelliteError(
                f'catalog search failed for {region} {year}: {exc}'
            ) from exc
        for product in response.json().get('value', []):
            date_str = product['ContentDate']['Start'][:10]
            if int(date_str[5:7]) not in months:
                continue
            cloud = next(
                (
                    attr['Value']
                    for attr in product.get('Attributes', [])
                    if attr.get('Name') == 'cloudCover'
                ),
                100,
            )
            scenes.append({'date': date_str, 'cloud': cloud})

    best_by_year = {}
    for scene in scenes:
        key = scene['date'][:4]
        if key not in best_by_year or scene['cloud'] < best_by_year[key]['cloud']:
            best_by_year[key] = scene
    return sorted(scene['date'] for scene in best_by_year.values())


def reflectance_to_uint8(arr: np.ndarray) -> np.ndarray:
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def uint8_to_reflectance(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64) / 255.0


def index_to_uint8(arr: np.ndarray) -> np.ndarray:
    return ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def uint8_to_index(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64) / 127.5 - 1


def fetch_band(
        bbox: list[float], width: int, height: int, date_str: str,
        band_name: str, token: str) -> np.ndarray:
    evalscript = f"""//VERSION=3
function setup() {{
  return {{
    input: [{{bands: ["{band_name}"]}}],
    output: {{bands: 1, sampleType: "FLOAT32"}}
  }};
}}
function evaluatePixel(s) {{
  return [s.{band_name}];
}}"""
    body = {
        'input': {
            'bounds': {
                'bbox': bbox,
                'properties': {'crs': 'http://www.opengis.net/def/crs/EPSG/0/4326'},
            },
            'data': [{
                'type': 'sentinel-2-l2a',
                'dataFilter': {
                    'timeRange': {
                        'from': f'{date_str}T00:00:00Z',
                        'to': f'{date_str}T23:59:59Z',
                    },
                },
            }],
        },
        'output': {
            'width': width,
            'height': height,
            'responses': [{
                'identifier': 'default',
                'format': {'type': 'image/tiff'},
            }],
        },
        'evalscript': evalscript,
    }
    try:
        response = requests.post(
            PROCESS_URL,
            json=body,
            headers={'Authorization': f'Bearer {token}', 'Accept': 'image/tiff'},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise SatelliteError(
            f'error fetching {band_name} for {date_str}: {exc}'
        ) from exc
    if response.status_code != 200:
        raise SatelliteError(
            f'error fetching {band_name} for {date_str} '
            f'({response.status_code}): {response.text[:500]}'
        )

    with rasterio.open(io.BytesIO(response.content)) as src:
        return src.read(1)


def compute_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    b02, b04, b08, b11 = bands['B02'], bands['B04'], bands['B08'], bands['B11']
    eps = np.float32(1e-10)
    return {
        'ndvi': np.clip((b08 - b04) / (b08 + b04 + eps), -1, 1),
        'ndmi': np.clip((b08 - b11) / (b08 + b11 + eps), -1, 1),
        'evi': np.clip(2.5 * (b08 - b04) / (b08 + 6 * b04 - 7.5 * b02 + 1 + eps), -1, 1),
    }


def save_geotiff(
        path: Path, data_uint8: np.ndarray, bbox: list[float],
        width: int, height: int) -> None:
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
            path, 'w', driver='GTiff', dtype='uint8', width=width, height=height,
            count=1, crs=WGS84, transform=transform, compress='deflate',
    ) as dst:
        dst.write(data_uint8, 1)


def fetch_one_date(
        date_str: str, bbox: list[float], width: int, height: int,
        output_dir: Path, token: str, log=None) -> bool:
    log = log or (lambda message: None)
    bands = {}
    for band in BANDS:
        log(f'  downloading {band}')
        bands[band] = fetch_band(bbox, width, height, date_str, band, token)

    if all(band.max() == 0 for band in bands.values()):
        log(f'  skipping {date_str}: all bands are zero')
        return False

    out_dir = output_dir / date_str
    for name, arr in bands.items():
        save_geotiff(
            out_dir / f'{name.lower()}.tif',
            reflectance_to_uint8(arr), bbox, width, height,
        )
    for name, arr in compute_indices(bands).items():
        save_geotiff(out_dir / f'{name}.tif', index_to_uint8(arr), bbox, width, height)
    return True


def fetch_region(
        geojson_path: Path, output_dir: Path, region: str, dates: list[str],
        force: bool = False, log=None) -> tuple[int, int]:
    """Fetch missing dates for a region. Returns (fetched, skipped)."""
    log = log or (lambda message: None)
    missing_dates = []
    skipped = 0
    for date in dates:
        if not force and complete_date_dir(output_dir / date):
            skipped += 1
        else:
            missing_dates.append(date)

    if not missing_dates:
        return 0, skipped

    bbox = bbox_from_geojson(geojson_path, region)
    width, height = pixel_dims(bbox)
    token = get_access_token()
    fetched = 0
    for date in missing_dates:
        log(f'{region}: {date}')
        if fetch_one_date(date, bbox, width, height, output_dir, token, log=log):
            fetched += 1
    return fetched, skipped


def rasterize_parcels(
        geojson_path: Path, bbox: list[float], width: int, height: int,
        region: str | None = None) -> tuple[np.ndarray, list[str]]:
    with open(geojson_path, encoding='utf-8') as f:
        data = json.load(f)

    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    features = data.get('features') or []
    if region:
        features = [
            f for f in features
            if f.get('properties', {}).get('layer') == region
        ]
    features = sorted(features, key=lambda f: f.get('properties', {}).get('name') or '')
    parcel_names = sorted({
        f.get('properties', {}).get('name')
        for f in features
        if f.get('properties', {}).get('name')
    })
    name_to_idx = {name: idx for idx, name in enumerate(parcel_names, start=1)}
    shapes = [
        (feature['geometry'], name_to_idx[feature['properties']['name']])
        for feature in features
        if feature.get('geometry')
        and feature.get('properties', {}).get('name') in name_to_idx
    ]
    mask = rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0,
        dtype=np.uint8,
    )
    return mask, parcel_names


def compute_timeseries(
        mask: np.ndarray, parcel_names: list[str], dates: list[str],
        output_dir: Path) -> dict:
    bands_lower = [b.lower() for b in BANDS]
    layers = bands_lower + INDICES
    index_set = set(INDICES)
    forest_pixels = mask > 0
    forest_count = int(np.sum(forest_pixels))
    parcel_masks = {
        name: (mask == idx, int(np.sum(mask == idx)))
        for idx, name in enumerate(parcel_names, start=1)
    }

    means_forest = {layer: [] for layer in layers}
    means_parcels = {name: {layer: [] for layer in layers} for name in parcel_names}

    for date in dates:
        for layer in layers:
            with rasterio.open(output_dir / date / f'{layer}.tif') as src:
                data = src.read(1)
            real = uint8_to_index(data) if layer in index_set else uint8_to_reflectance(data)
            forest_mean = float(np.mean(real[forest_pixels])) if forest_count > 0 else 0.0
            means_forest[layer].append(round(forest_mean, 4))
            for name, (parcel_mask, parcel_count) in parcel_masks.items():
                value = float(np.mean(real[parcel_mask])) if parcel_count > 0 else 0.0
                means_parcels[name][layer].append(round(value, 4))

    return {
        'parcels': parcel_names,
        'dates': dates,
        'layers': layers,
        'means': {
            'forest': means_forest,
            'parcels': means_parcels,
        },
    }


def precompute_region(geojson_path: Path, output_dir: Path, region: str) -> int:
    dates = discover_dates(output_dir)
    if not dates:
        raise SatelliteError(f'no complete date directories found in {output_dir}')

    bbox = bbox_from_geojson(geojson_path, region)
    width, height = pixel_dims(bbox)
    manifest = {
        'dates': dates,
        'bands': [b.lower() for b in BANDS],
        'indices': INDICES,
        'bbox': [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        'width': width,
        'height': height,
        'resolution_m': RESOLUTION_M,
    }
    with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')

    mask, parcel_names = rasterize_parcels(geojson_path, bbox, width, height, region)
    save_geotiff(output_dir / 'parcel-mask.tif', mask, bbox, width, height)

    timeseries = compute_timeseries(mask, parcel_names, dates, output_dir)
    with open(output_dir / 'timeseries.json', 'w', encoding='utf-8') as f:
        json.dump(timeseries, f, indent=2)
        f.write('\n')
    return len(dates)
