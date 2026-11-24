"""Identity and publication contract for locally built printable-map layers."""
import gzip
import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path

from django.db.models import Q

from apps.base.models import MapFeature

SCHEMA_VERSION = 1
CONTEXT_FILE = 'context.json.gz'


class MapContextError(Exception):
    """Invalid map input, incomplete remote data, or an unsuccessful build."""


def region_names(geojson):
    layers = ((feature.get('properties') or {}).get('layer') for feature in geojson['features'])
    return sorted({name for name in layers if isinstance(name, str) and name})


def region_geometry(geojson, region):
    geometries = [f['geometry'] for f in geojson.get('features', [])
                  if (f.get('properties') or {}).get('layer') == region]
    return geometry_snapshot(geometries, region)


def published_region_geometry(region):
    parcels = MapFeature.objects.filter(published=True, road_type__isnull=True).filter(
        Q(parcel__region_id=region.pk) | Q(parcel__isnull=True, members__properties__layer=region.name),
    )
    return geometry_snapshot(list(parcels.values_list('geometry', flat=True)), region.name)


def geometry_snapshot(geometries, region):
    if not geometries:
        raise MapContextError(f'No parcel geometry for {region}')
    encoded = sorted(json.dumps(g, sort_keys=True, separators=(',', ':')) for g in geometries)
    fingerprint = hashlib.sha256('\n'.join(encoded).encode()).hexdigest()
    points = []
    for geometry in geometries:
        if geometry['type'] == 'Polygon':
            rings = geometry['coordinates']
        elif geometry['type'] == 'MultiPolygon':
            rings = [r for p in geometry['coordinates'] for r in p]
        else:
            raise MapContextError('Parcel geometries must be polygons')
        points.extend(p for ring in rings for p in ring)
    if not points:
        raise MapContextError('Empty parcel geometry')
    west, south = map(min, zip(*[p[:2] for p in points]))
    east, north = map(max, zip(*[p[:2] for p in points]))
    return fingerprint, [west, south, east, north]


def region_directory(root, region):
    root = Path(root).resolve()
    path = (root / region).resolve()
    if path == root or path.parent != root:
        raise MapContextError('Invalid region directory')
    return path


def context_path(root, map_key):
    return region_directory(root, str(map_key)) / CONTEXT_FILE


def read_context(path):
    with open(path, 'rb') as stream:
        return read_context_stream(stream)


def read_context_stream(stream):
    try:
        with gzip.GzipFile(fileobj=stream) as compressed:
            payload = json.load(compressed)
    except (EOFError, OSError, ValueError, zlib.error) as exc:
        raise MapContextError('Invalid map context snapshot') from exc
    if not isinstance(payload, dict):
        raise MapContextError('Invalid map context snapshot')
    return payload


def context_matches(payload, region_key, fingerprint):
    build = payload.get('build')
    return (payload.get('available') is True and isinstance(payload.get('features'), list)
            and payload.get('schema_version') == SCHEMA_VERSION
            and payload.get('region_key') == str(region_key)
            and isinstance(build, dict) and build.get('geometry_hash') == fingerprint)


def context_is_current(region, output_dir, *, snapshot=None):
    try:
        payload = read_context(context_path(output_dir, region.map_key))
        fingerprint, _ = snapshot if snapshot is not None else published_region_geometry(region)
    except (MapContextError, OSError, ValueError, KeyError):
        return False
    return context_matches(payload, region.map_key, fingerprint)


def publish_context(path, payload):
    """A failed build leaves the preceding complete snapshot available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.context-')
    try:
        with os.fdopen(descriptor, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
                compressed.write(json.dumps(payload, ensure_ascii=False,
                                            allow_nan=False, separators=(',', ':')).encode())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
