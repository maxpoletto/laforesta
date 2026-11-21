"""Aligned rectangular coverage and staged acquisition downloads."""
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window, bounds as window_bounds

from apps.bosco import satellite


TILE_SIZE = 1024
COVERAGE = 'coverage.tif'


class MissingCoverage(satellite.SatelliteError):
    """The acquisition has missing pixels over published parcels."""


@dataclass(frozen=True)
class Grid:
    bounds: tuple
    width: int
    height: int

    @property
    def transform(self):
        return from_bounds(*self.bounds, self.width, self.height)

    @property
    def dimensions(self):
        return self.width, self.height

    def expand(self, extent):
        west, south, east, north = extent
        inverse = ~self.transform
        left, top = inverse @ (west, north)
        right, bottom = inverse @ (east, south)
        col = min(0, math.floor(round(left, 8)))
        row = min(0, math.floor(round(top, 8)))
        width = max(self.width, math.ceil(round(right, 8))) - col
        height = max(self.height, math.ceil(round(bottom, 8))) - row
        if col == row == 0 and (width, height) == self.dimensions:
            return self
        window = Window(col, row, width, height)
        return Grid(window_bounds(window, self.transform), width, height)

    def window(self, old):
        col, row = ~self.transform @ (old.bounds[0], old.bounds[3])
        return Window(round(col), round(row), old.width, old.height)

    @classmethod
    def read(cls, path):
        with rasterio.open(path) as src:
            grid = cls(tuple(src.bounds), src.width, src.height)
            if src.crs != satellite.WGS84 or not src.transform.almost_equals(grid.transform, precision=1e-9):
                raise satellite.SatelliteError('Incompatible satellite grid')
            return grid


def missing_windows(grid, known):
    if known is None:
        return [Window(0, 0, grid.width, grid.height)]
    left, top, width, height = known.flatten()
    right, bottom = left + width, top + height
    return [window for window in [
        Window(0, 0, grid.width, top),
        Window(0, bottom, grid.width, grid.height - bottom),
        Window(0, top, left, height),
        Window(right, top, grid.width - right, height),
    ] if window.width and window.height]


def tiles(window):
    x, y, width, height = map(int, window.flatten())
    for row in range(y, y + height, TILE_SIZE):
        for col in range(x, x + width, TILE_SIZE):
            yield Window(col, row, min(TILE_SIZE, x + width - col), min(TILE_SIZE, y + height - row))


def usable_coverage(path, parcel_mask):
    if not path.is_file():
        return  # Legacy rasters predate coverage masks.
    with rasterio.open(path) as source:
        for _, window in source.block_windows():
            rows, cols = window.toslices()
            if np.any((source.read(1, window=window) == 0) & parcel_mask[rows, cols]):
                raise MissingCoverage('Acquisition does not cover the parcels')


def reuse_date(source, dest, parcel_mask, *, immutable):
    usable_coverage(source / COVERAGE, parcel_mask)
    dest.mkdir()
    for name in [*satellite.EXPECTED_TIFFS, COVERAGE]:
        path = source / name
        if path.is_file():
            (os.link if immutable else copyfile)(path, dest / name)


def _copy_grid(source, writers, target_window):
    with ExitStack() as stack:
        for name, writer in writers.items():
            filename = source / name
            reader = stack.enter_context(rasterio.open(filename)) if filename.is_file() else None
            for window in tiles(Window(0, 0, target_window.width, target_window.height)):
                pixels = (reader.read(1, window=window) if reader is not None
                          else np.ones((int(window.height), int(window.width)), dtype=np.uint8))
                writer.write(pixels, 1, window=Window(
                    target_window.col_off + window.col_off, target_window.row_off + window.row_off,
                    window.width, window.height,
                ))


def _download_tile(writers, grid, window, date, token, max_cloud):
    extent = window_bounds(window, grid.transform)
    width, height = int(window.width), int(window.height)

    def fetch(name):
        pixels = satellite.fetch_band(extent, width, height, date, name, token, max_cloud=max_cloud)
        if pixels.shape != (height, width):
            raise satellite.SatelliteError('Copernicus returned unexpected raster dimensions')
        return pixels

    valid = fetch('dataMask') > 0
    writers[COVERAGE].write(valid.astype(np.uint8), 1, window=window)
    if not valid.any():
        return
    bands = {name: fetch(name) for name in satellite.BANDS}
    for name, values in bands.items():
        writers[f'{name.lower()}.tif'].write(satellite.reflectance_to_uint8(values), 1, window=window)
    for name, values in satellite.compute_indices(bands).items():
        writers[f'{name}.tif'].write(satellite.index_to_uint8(values), 1, window=window)


def download_date(dest, grid, parcel_mask, token, max_cloud, *, previous=None, previous_grid=None):
    known = grid.window(previous_grid) if previous_grid else None
    with TemporaryDirectory(dir=dest.parent) as temp:
        stage = Path(temp) / dest.name
        stage.mkdir()
        with ExitStack() as stack:
            writers = {name: stack.enter_context(rasterio.open(
                stage / name, 'w', driver='GTiff', dtype='uint8', width=grid.width, height=grid.height,
                count=1, crs=satellite.WGS84, transform=grid.transform, compress='deflate',
            )) for name in [*satellite.EXPECTED_TIFFS, COVERAGE]}
            if known is not None:
                _copy_grid(previous, writers, known)
            for area in missing_windows(grid, known):
                for window in tiles(area):
                    _download_tile(writers, grid, window, dest.name, token, max_cloud)
        usable_coverage(stage / COVERAGE, parcel_mask)
        stage.replace(dest)
