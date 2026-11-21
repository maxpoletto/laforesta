"""Geometry of editable drawings. Each part may carry one GPS fix per vertex."""
from itertools import groupby

from shapely.geometry import shape

from apps.base.geodesy import valid_position
from config import strings as S


def walk_parts(records):
    parts = []
    for _, group in groupby(records, lambda r: ('trace', r['segment']) if r['segment']
                            else ('point', r['client_record_id'])):
        rows = list(group)
        coords = [[r['lon'], r['lat']] for r in rows]
        parts.append({'type': 'Point' if len(rows) == 1 else 'LineString',
                      'coordinates': coords[0] if len(rows) == 1 else coords,
                      'fixes': [[r['acc_m'], r['note']] for r in rows]})
    return parts


def valid_part(part):
    if not isinstance(part, dict):
        return False
    kind, coords = part.get('type'), part.get('coordinates')
    if not isinstance(coords, list):
        return False
    if kind == 'Point':
        points = [coords]
    elif kind == 'LineString' and len(coords) >= 2:
        points = coords
    elif kind == 'Polygon' and coords and all(
            isinstance(r, list) and len(r) >= 4 and r[0] == r[-1] for r in coords):
        points = [p for ring in coords for p in ring]
    else:
        return False
    if not all(map(valid_position, points)):
        return False
    fixes = part.get('fixes')
    if fixes is not None and not (isinstance(fixes, list) and len(fixes) == len(points)
            and all(f is None or (isinstance(f, list) and len(f) == 2
                    and (f[0] is None or type(f[0]) in (float, int) and 0 <= f[0] < float('inf'))
                    and isinstance(f[1], str)) for f in fixes)):
        return False
    return set(part) <= {'type', 'coordinates', 'fixes'}


def checked_parts(parts):
    if not isinstance(parts, list) or not parts or not all(map(valid_part, parts)):
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    return parts


def feature_geometry(parts, kind):
    """Only completed, valid lines or polygon pieces can become live map features."""
    if not parts or any(p['type'] != kind for p in parts):
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    if len(parts) == 1:
        geometry = {'type': kind, 'coordinates': parts[0]['coordinates']}
    elif kind == 'Polygon':
        geometry = {'type': 'MultiPolygon', 'coordinates': [p['coordinates'] for p in parts]}
    else:
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    if not shape(geometry).is_valid:
        raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
    return geometry
