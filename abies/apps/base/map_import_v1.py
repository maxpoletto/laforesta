"""Initial parcel-map import, shared with migration 0035. Keep its schema stable."""

import json
from pathlib import Path

from django.db import transaction

from apps.base.geodesy import valid_position


def initial_map_path(directory):
    for name in ('map.geojson', 'terreni.geojson'):
        path = directory / name
        try:
            path.stat()
        except FileNotFoundError:
            continue
        return path
    return None


def parcel_key(feature):
    props = feature.get('properties') or {}
    region = props.get('layer') or props.get('compresa') or props.get('Compresa') or ''
    name = props.get('name') or props.get('particella') or props.get('Particella') or ''
    if not isinstance(region, str) or not isinstance(name, str):
        return '', ''
    if region and name.startswith(region + '-'):
        return region, name[len(region) + 1:]
    return region, name.split('-', 1)[-1]


def _valid_ring(ring):
    return (isinstance(ring, list) and len(ring) >= 4
            and ring[0] == ring[-1] and all(valid_position(point, extra_dimensions=True) for point in ring))


def read_collection(path):
    try:
        with Path(path).open(encoding='utf-8-sig') as stream:
            data = json.load(stream)
        # Python accepts NaN/Infinity even in foreign members. GeoJSON and
        # SQLite JSON fields both require finite JSON numbers.
        json.dumps(data, allow_nan=False)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    if (not isinstance(data, dict) or data.get('type') != 'FeatureCollection'
            or not isinstance(data.get('features'), list)):
        raise ValueError(f'{path}: expected a GeoJSON FeatureCollection with a features array')
    for index, feature in enumerate(data['features'], 1):
        if (not isinstance(feature, dict) or feature.get('type') != 'Feature'
                or not isinstance(feature.get('properties'), (dict, type(None)))):
            raise ValueError(f'{path}, feature {index}: invalid GeoJSON Feature')
        geometry = feature.get('geometry')
        if not isinstance(geometry, dict) or geometry.get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError(f'{path}, feature {index}: parcel geometry must be Polygon or MultiPolygon')
        coords = geometry.get('coordinates')
        polygons = [coords] if geometry['type'] == 'Polygon' else coords
        if not (isinstance(polygons, list) and polygons
                and all(isinstance(p, list) and p and all(_valid_ring(r) for r in p) for p in polygons)):
            raise ValueError(f'{path}, feature {index}: expected closed rings of WGS84 positions')
    return data


def _members(feature, linked):
    members = {k: v for k, v in feature.items() if k != 'geometry'}
    if linked:
        # Served from Eclass. The old generated file contains this field while
        # the canonical input usually does not.
        props = dict(members.get('properties') or {})
        props.pop('coppice', None)
        members['properties'] = props
    return members


def _same_json(left, right):
    """Compare JSON values by type, treating integer and decimal spellings as numbers."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    # Python treats True and False as the numbers 1 and 0.
    return isinstance(left, bool) == isinstance(right, bool) and left == right


def _same_features(rows, existing):
    return len(rows) == len(existing) and all(
        incoming.parcel_id == stored.parcel_id and _same_json(incoming.geometry, stored.geometry)
        and _same_json(incoming.members, _members(stored.members, stored.parcel_id is not None))
        for incoming, stored in zip(rows, existing)
    )


def import_initial_map(apps, alias, path, *, existing_filter=None):
    """Import once, preserving order, IDs, holes, disconnected pieces and foreign members."""
    data = read_collection(path)
    features = data.pop('features')
    collections = apps.get_model('base', 'MapCollection').objects.using(alias)
    Feature = apps.get_model('base', 'MapFeature')
    History = apps.get_model('base', 'HistoricalMapFeature')
    parcels = apps.get_model('base', 'Parcel').objects.using(alias)
    with transaction.atomic(using=alias):
        # Reserve SQLite's write lock before checking whether another import won.
        exists = collections.filter(pk=1).update(id=1)
        ids = {(region, name): pk for region, name, pk in
               parcels.values_list('region__name', 'name', 'pk')}
        rows = []
        for feature in features:
            parcel_id = ids.get(parcel_key(feature))
            rows.append(Feature(parcel_id=parcel_id, geometry=feature['geometry'],
                                members=_members(feature, parcel_id is not None)))
        existing = Feature.objects.using(alias).filter(**(existing_filter or {}))
        if exists or existing.exists():
            members = collections.filter(pk=1).values_list('members', flat=True).first()
            if _same_json(members, data) and _same_features(rows, list(existing.order_by('pk'))):
                return False
            raise ValueError('The database already contains a different map. Initial import cannot replace it.')
        collections.create(pk=1, members=data)
        Feature.objects.using(alias).bulk_create(rows)
        History.objects.using(alias).bulk_create([
            History(id=r.pk, parcel_id=r.parcel_id, members=r.members, geometry=r.geometry,
                    version=r.version, created_at=r.created_at, modified_at=r.modified_at,
                    history_date=r.created_at, history_type='+',
                    history_change_reason='initial-audit-baseline') for r in rows
        ])
    return True
