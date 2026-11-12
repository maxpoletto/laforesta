"""WGS84 ellipsoidal measurements, using the same GeographicLib model as JS."""
from geographiclib.geodesic import Geodesic

_WGS84 = Geodesic.WGS84


def distance_m(lat1, lon1, lat2, lon2):
    """Shortest horizontal ellipsoidal surface distance, in meters."""
    return _WGS84.Inverse(lat1, lon1, lat2, lon2)['s12']
