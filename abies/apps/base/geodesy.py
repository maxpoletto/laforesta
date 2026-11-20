"""WGS84 positions and ellipsoidal measurements."""
from geographiclib.geodesic import Geodesic

_WGS84 = Geodesic.WGS84


def valid_position(point, *, extra_dimensions=False):
    return (isinstance(point, list) and len(point) >= 2
            and (extra_dimensions or len(point) == 2)
            and all(type(value) in (int, float) for value in point)
            and -180 <= point[0] <= 180 and -90 <= point[1] <= 90)


def distance_m(lat1, lon1, lat2, lon2):
    """Shortest horizontal ellipsoidal surface distance, in meters."""
    return _WGS84.Inverse(lat1, lon1, lat2, lon2)['s12']
