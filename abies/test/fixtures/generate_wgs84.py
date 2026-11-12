"""Reproduce independent JS test vectors with GeographicLib Python 2.1.

Run: python test/fixtures/generate_wgs84.py
Boundary references minimize exact ellipsoidal distance along each geodesic
edge, rather than using the JavaScript local-plane approximation.
"""
import json
import math
from pathlib import Path

from geographiclib.geodesic import Geodesic

GEOD = Geodesic.WGS84


def area(ring):
    polygon = GEOD.Polygon()
    for lon, lat in ring:
        polygon.AddPoint(lat, lon)
    return abs(polygon.Compute(False, True)[2])


def boundary_distance(query, geometry):
    polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
    best = math.inf
    for rings in polygons:
        for ring in rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                line = GEOD.InverseLine(a[1], a[0], b[1], b[0])

                def distance(s):
                    point = line.Position(s)
                    return GEOD.Inverse(query[1], query[0], point['lat2'], point['lon2'])['s12']

                lo, hi = 0, line.s13
                for _ in range(80):
                    c, d = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                    if distance(c) < distance(d):
                        hi = d
                    else:
                        lo = c
                best = min(best, distance(0), distance(line.s13), distance((lo + hi) / 2))
    return best


def local_point(origin, x, y):
    result = GEOD.Direct(*origin, math.degrees(math.atan2(x, y)), math.hypot(x, y))
    return [result['lon2'], result['lat2']]


def square(origin, half_width, rotation=0):
    angle = math.radians(rotation)
    points = [local_point(origin, x * math.cos(angle) - y * math.sin(angle),
                          x * math.sin(angle) + y * math.cos(angle))
              for x, y in [(-half_width, -half_width), (half_width, -half_width),
                           (half_width, half_width), (-half_width, half_width)]]
    return points + [points[0]]


def main():
    pairs = [(0, 0, 0, 1), (0, 0, 1, 0), (38.5, 16.3, 38.5, 16.36),
             (38.5, 16.3, 38.545, 16.3), (38.5, 16.3, 38.53, 16.35),
             (0, 0, 0, 180), (0, 0, .001, 179.999), (89.99, 0, 89.99, 180),
             (10, 179.99, 10, -179.99), (-41.32, 174.81, 40.96, -5.50),
             (38.5, 16.3, 38.5, 16.3)]
    rings = [[[0, 0], [1, 0], [1, 1], [0, 1]],
             [[16.3, 38.5], [16.35, 38.5], [16.35, 38.545], [16.3, 38.545]],
             [[179.99, 10], [-179.99, 10], [-179.99, 10.01], [179.99, 10.01]],
             [[0, 89], [120, 89], [-120, 89]]]
    hole = [[16.31, 38.51], [16.32, 38.51], [16.32, 38.52], [16.31, 38.52]]
    data = {
        'source': 'GeographicLib Python 2.1, Geodesic.WGS84; a=6378137, f=1/298.257223563.',
        'generator': 'test/fixtures/generate_wgs84.py',
        'reference_url': 'https://geographiclib.sourceforge.io/JavaScript/doc/',
        'distances': [{'points': p, 'metres': GEOD.Inverse(*p)['s12']} for p in pairs],
        'areas': [{'ring': ring, 'area': area(ring)} for ring in rings],
        'polygons': [
            {'geometry': {'type': 'Polygon', 'coordinates': [rings[1], hole]},
             'area': area(rings[1]) - area(hole)},
            {'geometry': {'type': 'MultiPolygon', 'coordinates': [[rings[0]], [rings[1], hole]]},
             'area': area(rings[0]) + area(rings[1]) - area(hole)},
        ],
        'destinations': [], 'boundary_polygons': [], 'boundaries': [],
    }
    starts = [(38.5, 16.3, az, s) for az in [0, 90, 180, 270, 45] for s in [12, 5000]]
    starts += [(89.9, 179.9, 45, 5000), (-38, -179.9, -90, 5000)]
    for lat, lon, az, s in starts:
        point = GEOD.Direct(lat, lon, az, s)
        data['destinations'].append({'start': [lat, lon], 'azimuth': az, 'metres': s,
                                     'point': [point['lat2'], point['lon2']]})
    for origin, rotation in [((38.5, 16.3), 0), ((38.5, 16.3), 30),
                             ((70, 16.3), 30), ((-38.5, 16.3), 30)]:
        index = len(data['boundary_polygons'])
        geometry = {'type': 'Polygon', 'coordinates': [square(origin, 2500, rotation)]}
        data['boundary_polygons'].append(geometry)
        for x, y in [(0, 0), (2800, 0), (0, 2800), (3000, 3000), (0, .01)]:
            query = local_point(origin, x, y)
            data['boundaries'].append({'polygon': index, 'query': query,
                                      'metres': boundary_distance(query, geometry)})
    origin = (38.5, 16.3)
    for geometry in [
        {'type': 'Polygon', 'coordinates': [square(origin, 2500), square(origin, 500)]},
        {'type': 'MultiPolygon', 'coordinates': [[square(origin, 250)],
                                               [square((38.53, 16.33), 250)]]},
    ]:
        index = len(data['boundary_polygons'])
        data['boundary_polygons'].append(geometry)
        for x, y in [(0, 0), (1000, 1000)]:
            query = local_point(origin, x, y)
            data['boundaries'].append({'polygon': index, 'query': query,
                                      'metres': boundary_distance(query, geometry)})
    Path(__file__).with_name('wgs84-geodesy.json').write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
