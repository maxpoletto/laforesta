"""Backend GeoJSON helpers for parcel point lookups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from apps.base.models import Parcel
from config.constants import IPSO_TERRENI_GEOJSON


@dataclass(frozen=True)
class ParcelFeature:
    parcel: Parcel
    geometry: dict
    bbox: tuple[float, float, float, float]


class ParcelGeometryIndex:
    """Resolve lat/lon points to DB Parcel rows using terreni.geojson.

    Missing or stale geometry is non-fatal: callers get ``None`` and can skip
    geometry-derived checks instead of blocking imports.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or Path(settings.GEO_DIR) / IPSO_TERRENI_GEOJSON
        self.features = self._load_features()

    def parcel_at(self, lat, lon) -> Parcel | None:
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            return None

        for feature in self.features:
            min_lon, min_lat, max_lon, max_lat = feature.bbox
            if not (min_lon <= lon_f <= max_lon and min_lat <= lat_f <= max_lat):
                continue
            if point_in_polygon(lon_f, lat_f, feature.geometry):
                return feature.parcel
        return None

    def _load_features(self) -> list[ParcelFeature]:
        try:
            with open(self.path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

        parcels = {
            (p.region.name, p.name): p
            for p in Parcel.objects.select_related('region')
        }
        features = []
        for feature in data.get('features') or []:
            key = parcel_key(feature)
            parcel = parcels.get(key)
            bbox = feature_bbox(feature.get('geometry'))
            if parcel is None or bbox is None:
                continue
            features.append(ParcelFeature(parcel, feature.get('geometry') or {}, bbox))

        # Prefer the most specific match if geometries overlap.
        features.sort(key=lambda f: _bbox_area(f.bbox))
        return features


def parcel_key(feature: dict) -> tuple[str, str]:
    """Return (compresa, particella), mirroring frontend geo.parcelNames()."""
    props = feature.get('properties') or {}
    region = props.get('layer') or props.get('compresa') or props.get('Compresa') or ''
    full_name = props.get('name') or props.get('particella') or props.get('Particella') or ''
    dash = full_name.find('-')
    parcel = full_name[dash + 1:] if dash >= 0 else full_name
    return region, parcel


def feature_bbox(geometry: dict | None) -> tuple[float, float, float, float] | None:
    coords = list(_iter_geometry_coords(geometry))
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def point_in_polygon(lon: float, lat: float, geometry: dict | None) -> bool:
    return any(_point_in_polygon_rings(lon, lat, rings)
               for rings in geometry_polygons(geometry))


def geometry_polygons(geometry: dict | None) -> list:
    if not geometry:
        return []
    if geometry.get('type') == 'Polygon':
        return [geometry.get('coordinates') or []]
    if geometry.get('type') == 'MultiPolygon':
        return geometry.get('coordinates') or []
    return []


def _point_in_polygon_rings(lon: float, lat: float, rings: list) -> bool:
    if not rings or not point_in_ring(lon, lat, rings[0]):
        return False
    return not any(point_in_ring(lon, lat, ring) for ring in rings[1:])


def point_in_ring(lon: float, lat: float, ring: list) -> bool:
    if len(ring) < 3:
        return False
    inside = False
    j = len(ring) - 1
    for i, coord in enumerate(ring):
        xi, yi = coord[0], coord[1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)
                and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _iter_geometry_coords(geometry: dict | None):
    for polygon in geometry_polygons(geometry):
        for ring in polygon:
            for coord in ring:
                if len(coord) >= 2:
                    yield coord


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    return (max_lon - min_lon) * (max_lat - min_lat)
