"""Build and atomically publish a compresa's complete satellite revision."""
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import numpy as np
import rasterio
from django.conf import settings
from django.db import transaction
from django.db.models import F
from django.utils import timezone
from shapely import make_valid
from shapely.errors import ShapelyError
from shapely.geometry import shape
from shapely.ops import unary_union

from apps.base.models import SatelliteRevision, SatelliteSeason, TerrainRefresh
from apps.bosco import satellite, satellite_grid as rasters, terrain_queue as queue
from apps.bosco.satellite_credentials import load_credentials
from apps.bosco.satellite_data import parcel_snapshot_key


logger = logging.getLogger(__name__)
State = TerrainRefresh.State
MARGIN = 0.001
RETENTION = timedelta(hours=24)
MAX_GRID_PIXELS = 16_000_000


@dataclass(frozen=True)
class MapInputs:
    source: str | None
    parcels: list
    bounds: tuple | None


def map_inputs(region_id):
    rows = list(queue.map_features(region_id).values_list(
        'pk', 'version', 'parcel_id', 'parcel__region_id', 'region_id', 'geometry',
    ))
    source = queue.signatures(row[:5] for row in rows).get(region_id)
    parcels = [(str(row[2]), row[5]) for row in rows if row[2]]
    if not parcels:
        return MapInputs(None, [], None)
    bounds = [shape(row[5]).bounds for row in rows]
    west = min(box[0] for box in bounds)
    south = min(box[1] for box in bounds)
    east = max(box[2] for box in bounds)
    north = max(box[3] for box in bounds)
    coverage = (max(-180, west - MARGIN), max(-90, south - MARGIN),
                min(180, east + MARGIN), min(90, north + MARGIN))
    return MapInputs(source, parcels, coverage)


def read_json(path):
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def existing_grids(root):
    grids = {}
    for name in satellite.discover_dates(root):
        folder = root / name
        try:
            grid = rasters.Grid.read(folder / satellite.EXPECTED_TIFFS[0])
            satellite.validate_raster_grid(root, [name], grid.bounds, grid.dimensions)
            grids[name] = grid
        except (OSError, rasterio.errors.RasterioError, satellite.SatelliteError):
            logger.warning('Ignoring incompatible satellite rasters for %s in %s', name, root.name)
    return grids


class CredentialsNeeded(Exception):
    """The remaining work needs a valid Copernicus client."""


class Provider:
    def __init__(self):
        self.credentials = load_credentials()
        self.token = None
        self.expires = 0

    def access_token(self):
        if not self.credentials:
            raise CredentialsNeeded
        if time.monotonic() >= self.expires:
            self.token = satellite.get_access_token(self.credentials)
            self.expires = time.monotonic() + 30 * 60
        return self.token


def _date_grid(stage, name, grid, mask, provider, max_cloud, old=None, old_grid=None, immutable=False):
    dest = stage / name
    if dest.exists():
        return True
    if old_grid == grid:
        try:
            rasters.reuse_date(old, dest, mask, immutable=immutable)
            return True
        except rasters.MissingCoverage:
            return False
    try:
        rasters.download_date(dest, grid, mask, provider.access_token(), max_cloud,
                              previous=old, previous_grid=old_grid)
    except rasters.MissingCoverage:
        return False
    return True


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_tree(path):
    for root, _, files in os.walk(path, topdown=False):
        for filename in files:
            _sync(Path(root) / filename)
        _sync(Path(root))


@transaction.atomic
def finish(job, inputs, state, now, *, revision=None, outcomes=(), through=None):
    rows = TerrainRefresh.objects.filter(pk=job.pk)
    if not rows.update(state=F('state')):
        return
    if queue.map_signatures(job.region_id).get(job.region_id) != inputs.source:
        rows.update(state=State.QUEUED, retry_at=None)
        return
    if revision:
        SatelliteRevision.objects.create(pk=revision, region_id=job.region_id)
        if job.revision_id:
            SatelliteRevision.objects.filter(pk=job.revision_id).update(superseded_at=now)
    for start, picked, geometry in outcomes:
        SatelliteSeason.objects.update_or_create(region_id=job.region_id, start=start,
                                                  defaults={'date': picked, 'geometry': geometry})
    rows.update(state=state, source=inputs.source or '', finished_at=now,
                retry_at=now + timedelta(hours=1) if state == State.FAILED else None,
                searched_through=through or job.searched_through, revision_id=revision or job.revision_id)


def build(job, inputs, requested_at, progress):
    if not inputs.parcels:
        finish(job, inputs, State.EMPTY, requested_at)
        return
    root = settings.SATELLITE_DIR / str(job.region.map_key)
    revisions = root / 'revisions'
    revisions.mkdir(parents=True, exist_ok=True)
    previous = root
    if job.revision_id:
        path = queue.revision_dir(job.region.map_key, job.revision_id)
        if (path / 'manifest.json').is_file():
            previous = path
    available = existing_grids(previous)
    base_grid = next(iter(available.values()), None)
    provider = Provider()
    waiting = False
    if base_grid is None and not provider.credentials:
        finish(job, inputs, State.WAITING, requested_at)
        return
    footprint = unary_union([make_valid(shape(geom)) for _, geom in inputs.parcels])
    if base_grid:
        grid = base_grid.expand(inputs.bounds)
        if grid != base_grid and not provider.credentials:
            if base_grid.expand(footprint.bounds) != base_grid:
                finish(job, inputs, State.WAITING, requested_at)
                return
            waiting, grid = True, base_grid
        # Legacy inputs could span multiple builds. Reuse those on the adopted grid.
        available = {name: stored for name, stored in available.items() if stored == base_grid}
    else:
        grid = rasters.Grid(inputs.bounds, *satellite.pixel_dims(inputs.bounds))
    if grid.width * grid.height > MAX_GRID_PIXELS:
        raise satellite.SatelliteError('Satellite grid exceeds 16 million pixels')
    mask, _ = satellite.rasterize_parcels(inputs.parcels.copy(), grid.bounds, *grid.dimensions)
    parcel_mask = mask != 0
    del mask
    geometry = parcel_snapshot_key(inputs.parcels)
    known = {row.start: row for row in SatelliteSeason.objects.filter(region_id=job.region_id)}
    through = None
    outcomes = []
    with TemporaryDirectory(prefix='.staging-', dir=revisions) as directory:
        stage = Path(directory)
        for name, old_grid in available.items():
            try:
                _date_grid(stage, name, grid, parcel_mask, provider, None, previous / name, old_grid,
                           immutable=previous != root)
            except CredentialsNeeded:
                waiting = True
            progress()
        for start, end, limit in queue.seasons(requested_at):
            already = known.get(start)
            stored = sorted(name for name in satellite.discover_dates(stage) if str(start) <= name < str(end))
            picked = already.date if already and str(already.date) in stored else (date.fromisoformat(stored[0]) if stored else None)
            if picked is None and (already is None or already.geometry != geometry or already.date is not None):
                if not provider.credentials:
                    waiting = True
                    continue
                saved = str(already.date) if already and already.date else None
                if saved and saved not in available and _date_grid(stage, saved, grid, parcel_mask, provider, None):
                    picked = already.date
                else:
                    for candidate, cloud in satellite.catalog_dates(grid.bounds, footprint, start, end, limit):
                        if _date_grid(stage, candidate, grid, parcel_mask, provider, cloud):
                            picked = date.fromisoformat(candidate)
                            break
                        progress()
            outcomes.append((start, picked, geometry))
            through = start
            progress()

        dates = satellite.discover_dates(stage)
        current = read_json(previous / 'manifest.json')
        previous_summary = read_json(previous / 'timeseries.json')
        state = State.WAITING if waiting else (State.CURRENT if dates else State.EMPTY)
        if not dates or (previous != root and (previous / 'parcel-mask.tif').is_file()
                         and dates == current.get('dates')
                         and grid == base_grid and geometry == previous_summary.get('parcel_snapshot')):
            finish(job, inputs, state, timezone.now(), outcomes=outcomes, through=through)
            return

        revision = uuid4()
        satellite.write_parcel_products(stage, dates, grid.bounds, grid.dimensions, inputs.parcels.copy(),
                                        stable_ids=True, revision=revision)
        sync_tree(stage)
        stage.replace(revisions / str(revision))
        _sync(revisions)
        _sync(root)
        _sync(root.parent)
        finish(job, inputs, state, timezone.now(), revision=revision, outcomes=outcomes, through=through)


def refresh(job, now, progress=lambda: None):
    inputs = None
    try:
        inputs = map_inputs(job.region_id)
        build(job, inputs, now, progress)
    except (CredentialsNeeded, satellite.SatelliteAuthError):
        finish(job, inputs, State.WAITING, timezone.now())
    except (OSError, ValueError, rasterio.errors.RasterioError, satellite.SatelliteError, ShapelyError) as exc:
        logger.warning('Satellite refresh failed for compresa %s: %s', job.region_id, exc)
        if inputs is None:
            inputs = MapInputs(queue.map_signatures(job.region_id).get(job.region_id), [], None)
        finish(job, inputs, State.FAILED, timezone.now())


def prune(now):
    cutoff = now - RETENTION
    expired = SatelliteRevision.objects.filter(superseded_at__lte=cutoff, terrainrefresh__isnull=True)
    for row in expired.select_related('region'):
        directory = queue.revision_dir(row.region.map_key, row.pk)
        if directory.exists():
            rmtree(directory)
        row.delete()
    known = {str(key) for key in SatelliteRevision.objects.values_list('pk', flat=True)}
    for directory in settings.SATELLITE_DIR.glob('*/revisions/*'):
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            UUID(directory.name)
        except ValueError:
            continue
        if directory.name not in known and directory.stat().st_mtime < cutoff.timestamp():
            rmtree(directory)


def discard_interrupted_staging():
    for directory in settings.SATELLITE_DIR.glob('*/revisions/.staging-*'):
        if directory.is_dir() and not directory.is_symlink():
            rmtree(directory)
