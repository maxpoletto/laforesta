"""Read drawing files and locate the measured positions of a tree survey."""

import json
import re
from base64 import b64decode
from collections import defaultdict
from math import isfinite
from xml.etree import ElementTree

from django.db.models import Count
from rasterio._err import CPLE_BaseError
from rasterio.crs import CRS
from rasterio.errors import RasterioError
from rasterio.warp import transform

from apps.base.models import Sample, TreeSample
from apps.impostazioni.drawing_geometry import checked_parts
from config import strings as S


def _xy(values, depth):
    if not isinstance(values, list):
        raise ValueError
    if depth == 0:
        if len(values) < 2:
            raise ValueError
        return values[:2]
    return [_xy(value, depth - 1) for value in values]


def _geojson_parts(value):
    if not isinstance(value, dict):
        raise ValueError
    kind = value.get('type')
    crs = value.get('crs')
    wgs84_names = ('CRS84', 'urn:ogc:def:crs:OGC:1.3:CRS84', 'EPSG:4326', 'urn:ogc:def:crs:EPSG::4326')
    if crs is not None and not any(crs == {'type': 'name', 'properties': {'name': name}} for name in wgs84_names):
        raise ValueError
    if kind == 'Feature':
        if value.get('geometry') is not None:
            yield from _geojson_parts(value['geometry'])
    elif kind in ('FeatureCollection', 'GeometryCollection'):
        collection = value.get('features' if kind == 'FeatureCollection' else 'geometries')
        if not isinstance(collection, list):
            raise ValueError
        for member in collection:
            yield from _geojson_parts(member)
    elif kind in ('Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon'):
        single = kind.removeprefix('Multi')
        depth = ('Point', 'LineString', 'Polygon').index(single)
        coordinates = value.get('coordinates')
        if single == kind:
            coordinates = [coordinates]
        if not isinstance(coordinates, list):
            raise ValueError
        for component in coordinates:
            yield {'type': single, 'coordinates': _xy(component, depth)}
    else:
        raise ValueError


def _gpx_position(node):
    return [float(node.attrib['lon']), float(node.attrib['lat'])]


def _gpx_path(nodes):
    points = [_gpx_position(node) for node in nodes]
    if len(points) == 1:
        yield {'type': 'Point', 'coordinates': points[0]}
    elif points:
        yield {'type': 'LineString', 'coordinates': points}


def _gpx_parts(text):
    if '<!DOCTYPE' in text.upper():
        raise ValueError
    root = ElementTree.fromstring(text)
    if root.tag.rsplit('}', 1)[-1] != 'gpx':
        raise ValueError
    for node in root:
        kind = node.tag.rsplit('}', 1)[-1]
        if kind == 'wpt':
            yield {'type': 'Point', 'coordinates': _gpx_position(node)}
        elif kind == 'rte':
            yield from _gpx_path(node.findall('{*}rtept'))
        elif kind == 'trk':
            for segment in node.findall('{*}trkseg'):
                yield from _gpx_path(segment.findall('{*}trkpt'))


def _dxf_entities(text):
    lines = iter(text.splitlines())
    pairs = ((int(code), value.strip()) for code, value in zip(lines, lines, strict=True))
    for pair in pairs:
        if pair == (0, 'SECTION') and next(pairs, None) == (2, 'ENTITIES'):
            break
    else:
        raise ValueError
    kind = None
    fields = defaultdict(list)
    for code, value in pairs:
        if code:
            fields[code].append(value)
            continue
        if kind:
            yield kind, fields
        if value == 'ENDSEC':
            return
        kind, fields = value, defaultdict(list)
    raise ValueError


def _dxf_flag(fields):
    return int(fields.get(70, [0])[0])


def _dxf_xy(fields):
    return [float(fields[10][0]), float(fields[20][0])]


def _dxf_planar(fields):
    if any(float(fields.get(code, [default])[0]) != default
           for code, default in ((210, 0), (220, 0), (230, 1))):
        raise ValueError(S.CARTOGRAPHY_DXF_UNSUPPORTED)
    if any(float(value) for value in fields.get(42, ())):
        raise ValueError(S.CARTOGRAPHY_DXF_UNSUPPORTED)


def _dxf_path(points, closed):
    if len(points) < 2:
        raise ValueError
    if not closed:
        return {'type': 'LineString', 'coordinates': points}
    if points[0] != points[-1]:
        points.append(points[0].copy())
    return {'type': 'Polygon', 'coordinates': [points]}


def _dxf_parts(text, epsg):
    if not isinstance(epsg, str) or not re.fullmatch(r'[0-9]{1,6}', epsg.strip()):
        raise ValueError(S.CARTOGRAPHY_DXF_CRS_REQUIRED)
    try:
        crs = CRS.from_epsg(int(epsg))
    except (ValueError, RasterioError):
        raise ValueError(S.CARTOGRAPHY_DXF_CRS_REQUIRED) from None
    if not (crs.is_geographic or crs.is_projected):
        raise ValueError(S.CARTOGRAPHY_DXF_CRS_REQUIRED)
    parts, positions = [], []
    entities = _dxf_entities(text)
    for kind, fields in entities:
        if kind in ('TEXT', 'MTEXT', 'DIMENSION') or fields.get(67) == ['1']:
            continue
        _dxf_planar(fields)
        if kind == 'POINT':
            point = _dxf_xy(fields)
            parts.append({'type': 'Point', 'coordinates': point})
            positions.append(point)
            continue
        if kind == 'LINE':
            points = [_dxf_xy(fields), [float(fields[11][0]), float(fields[21][0])]]
        elif kind == 'LWPOLYLINE':
            points = [[float(x), float(y)] for x, y in zip(fields[10], fields[20], strict=True)]
            if len(points) != int(fields[90][0]):
                raise ValueError
        elif kind == 'POLYLINE':
            if _dxf_flag(fields) & ~129:
                raise ValueError(S.CARTOGRAPHY_DXF_UNSUPPORTED)
            points = []
            for item, vertex in entities:
                if item == 'SEQEND':
                    break
                if item != 'VERTEX' or _dxf_flag(vertex):
                    raise ValueError(S.CARTOGRAPHY_DXF_UNSUPPORTED)
                _dxf_planar(vertex)
                points.append(_dxf_xy(vertex))
            else:
                raise ValueError
        else:
            raise ValueError(S.CARTOGRAPHY_DXF_UNSUPPORTED)
        part = _dxf_path(points, kind != 'LINE' and bool(_dxf_flag(fields) & 1))
        parts.append(part)
        positions.extend(points)
    if positions:
        # PROJ wraps some longitudes by subtracting full turns. Keep the
        # coordinates within terrestrial scales before passing them to PROJ.
        limit = 1e9 / crs.linear_units_factor[1] if crs.is_projected else 400
        if not all(isfinite(value) and abs(value) <= limit for point in positions for value in point):
            raise ValueError
        xs, ys = zip(*positions, strict=True)
        lons, lats = transform(crs, 'EPSG:4326', xs, ys)
        for point, lon, lat in zip(positions, lons, lats, strict=True):
            point[:] = [lon, lat]
    return parts


def file_parts(body):
    fmt = body.get('format')
    if fmt not in ('geojson', 'gpx', 'dxf'):
        raise ValueError(S.CARTOGRAPHY_FILE_INVALID)
    try:
        text = b64decode(body.get('file', ''), validate=True).decode('utf-8-sig')
        if fmt == 'geojson':
            parts = list(_geojson_parts(json.loads(text)))
        elif fmt == 'gpx':
            parts = list(_gpx_parts(text))
        else:
            parts = _dxf_parts(text, body.get('epsg'))
        return checked_parts(parts)
    except (ElementTree.ParseError, KeyError, IndexError, TypeError, ValueError,
            OverflowError, RecursionError, RasterioError, CPLE_BaseError) as exc:
        if str(exc) in (S.CARTOGRAPHY_DXF_CRS_REQUIRED, S.CARTOGRAPHY_DXF_UNSUPPORTED):
            raise
        raise ValueError(S.CARTOGRAPHY_FILE_INVALID) from None


def _trees(region_id):
    return TreeSample.objects.filter(sample__survey__sample_grid__isnull=True, parcel__region_id=region_id,
                                     lon__range=(-180, 180), lat__range=(-90, 90))


def _plots(region_id):
    return Sample.objects.filter(survey__sample_grid__isnull=False, sample_area__parcel__region_id=region_id,
                                 sample_area__lon__range=(-180, 180), sample_area__lat__range=(-90, 90))


def surveys_with_positions(region_id):
    trees = _trees(region_id).values('sample__survey_id', 'sample__survey__name').annotate(
        count=Count('tree_id', distinct=True))
    plots = _plots(region_id).values('survey_id', 'survey__name').annotate(count=Count('id'))
    return sorted([
        *({'id': row['sample__survey_id'], 'name': row['sample__survey__name'],
           'structured': False, 'count': row['count']} for row in trees),
        *({'id': row['survey_id'], 'name': row['survey__name'],
           'structured': True, 'count': row['count']} for row in plots),
    ], key=lambda row: row['name'].casefold())


def survey_parts(survey, region_id):
    if survey.sample_grid_id:
        rows = _plots(region_id).filter(survey=survey).order_by('sample_area_id').values_list(
            'sample_area__lon', 'sample_area__lat', 'sample_area__note')
        parts = [{'type': 'Point', 'coordinates': [lon, lat], 'fixes': [[None, note]]}
                 for lon, lat, note in rows.iterator()]
    else:
        rows = _trees(region_id).filter(sample__survey=survey).order_by('pk').values_list(
            'tree_id', 'lon', 'lat', 'acc_m', 'note')
        parts, seen = [], set()
        for tree_id, lon, lat, accuracy, note in rows.iterator():
            if tree_id in seen:
                continue
            seen.add(tree_id)
            accuracy = accuracy if accuracy is not None and accuracy >= 0 else None
            parts.append({'type': 'Point', 'coordinates': [lon, lat], 'fixes': [[accuracy, note]]})
    if not parts:
        raise ValueError(S.CARTOGRAPHY_SURVEY_NO_POINTS)
    return checked_parts(parts)
