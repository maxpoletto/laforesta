"""Identity and publication contract for locally built printable-map layers."""
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
CONTEXT_FILE = 'context.json.gz'


class MapContextError(Exception):
    """Invalid map input, incomplete remote data, or an unsuccessful build."""


def region_geometry(geojson, region):
    features = [f for f in geojson.get('features', [])
                if f.get('properties', {}).get('layer') == region]
    if not features:
        raise MapContextError(f'No parcel geometry for {region}')
    geometries = [f['geometry'] for f in features]
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


def read_context(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise MapContextError('Invalid map context snapshot')
    return payload


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
