"""Build a Bosco hillshade from INGV's TINITALY 1.1 bare-earth elevations."""
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, PngImagePlugin
import requests
from django.conf import settings
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds


WCS_URL = 'https://tinitaly.pi.ingv.it/TINItaly_1_1/wcs'
DEM_COVERAGE = 'TINItaly_1_1__tinitaly_dem'
NATIVE_CRS = 'EPSG:32632'
WEB_CRS = 'EPSG:3857'
MAP_CRS = 'EPSG:4326'
CELL_M = 10
MAX_PIXELS = 16_000_000
FILENAME = 'hillshade.png'
BOUNDS_HEADER = 'X-Abies-Bounds'


class HillshadeError(Exception):
    """The elevation source did not supply a usable raster."""


def output_path(map_key):
    return settings.SATELLITE_DIR / str(map_key) / FILENAME


def native_bounds(bounds):
    """Align the request to the native grid and pad it for slope calculation."""
    west, south, east, north = transform_bounds(MAP_CRS, NATIVE_CRS, *bounds)
    return (CELL_M * (math.floor(west / CELL_M) - 2), CELL_M * (math.floor(south / CELL_M) - 2),
            CELL_M * (math.ceil(east / CELL_M) + 2), CELL_M * (math.ceil(north / CELL_M) + 2))


def download_dem(bounds):
    west, south, east, north = native_bounds(bounds)
    if (east - west) * (north - south) > MAX_PIXELS * CELL_M ** 2:
        raise HillshadeError('Requested TINITALY raster is too large')
    try:
        response = requests.get(WCS_URL, params={
            'service': 'WCS', 'version': '2.0.1', 'request': 'GetCoverage',
            'coverageId': DEM_COVERAGE, 'format': 'image/tiff',
            'subset': (f'E({west},{east})', f'N({south},{north})'),
        }, timeout=45)
        response.raise_for_status()
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        raise HillshadeError(f'TINITALY download failed (HTTP {status})' if status else
                             'TINITALY download failed') from exc
    return response.content


def render_dem(raw):
    with MemoryFile(raw) as file, file.open() as src:
        if (src.count != 1 or src.crs != NATIVE_CRS or src.res != (CELL_M, CELL_M)
                or min(src.shape) < 3 or src.width * src.height > MAX_PIXELS):
            raise HillshadeError('Unexpected TINITALY elevation grid')
        elevation = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform, width, height = calculate_default_transform(
            src.crs, WEB_CRS, src.width, src.height, *src.bounds,
        )
        source_transform = src.transform

    # Raster rows run south. Compute slopes in east/north coordinates and
    # dot the unit surface normal with a light 45° above the northwest horizon.
    north_slope, east_slope = np.gradient(elevation, -CELL_M, CELL_M)
    altitude, azimuth = math.radians(45), math.radians(315)
    light = (math.sin(altitude) - math.cos(altitude) * (
        math.sin(azimuth) * east_slope + math.cos(azimuth) * north_slope
    )) / np.sqrt(1 + east_slope ** 2 + north_slope ** 2)
    valid = np.isfinite(elevation) & np.isfinite(light)
    pixels = np.zeros(elevation.shape, dtype=np.uint8)
    pixels[valid] = 1 + np.rint(254 * np.clip(light[valid], 0, 1)).astype(np.uint8)
    result = np.zeros((height, width), dtype=np.uint8)
    reproject(pixels, result, src_transform=source_transform, src_crs=NATIVE_CRS,
              dst_transform=transform, dst_crs=WEB_CRS, src_nodata=0, dst_nodata=0,
              resampling=Resampling.bilinear)
    if not result.any():
        raise HillshadeError('TINITALY has no usable data for this extent')

    west, south, east, north = transform_bounds(WEB_CRS, MAP_CRS, *array_bounds(height, width, transform))
    image = Image.fromarray(result).convert('LA')
    image.putalpha(Image.fromarray(np.where(result > 0, 255, 0).astype(np.uint8)))
    return image, [[south, west], [north, east]]


def metadata(stream):
    with Image.open(stream, formats=('PNG',)) as image:
        return image.info['bounds'], json.loads(image.info['requested_bounds'])


def refresh(map_key, bounds):
    path = output_path(map_key)
    try:
        with path.open('rb') as stream:
            _, (west, south, east, north) = metadata(stream)
        if west <= bounds[0] and south <= bounds[1] and east >= bounds[2] and north >= bounds[3]:
            return
    except (OSError, KeyError, ValueError):
        pass

    image, image_bounds = render_dem(download_dem(bounds))
    info = PngImagePlugin.PngInfo()
    info.add_text('bounds', json.dumps(image_bounds))
    info.add_text('requested_bounds', json.dumps(bounds))
    info.add_text('source', 'INGV TINITALY 1.1 · https://doi.org/10.13127/tinitaly/1.1 · CC BY 4.0')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.hillshade-', dir=path.parent) as directory:
        staged = Path(directory) / FILENAME
        with staged.open('wb') as stream:
            image.save(stream, format='PNG', pnginfo=info)
            stream.flush()
            os.fsync(stream.fileno())
        staged.replace(path)
