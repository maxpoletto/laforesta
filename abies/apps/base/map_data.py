"""Database-backed GeoJSON shared by desktop, offline devices and GIS builds."""

import hashlib
import json
from pathlib import Path

from django.apps import apps
from django.core.management.base import CommandError
from django.db import DEFAULT_DB_ALIAS, connection, transaction
from django.db.models import Q
from django.http import HttpResponse
from django.utils.cache import get_conditional_response, patch_vary_headers
from django.views.decorators.gzip import gzip_page

from apps.base.http import CACHE_NO_CACHE, apply_cache_control
from apps.base.map_import_v1 import import_initial_map, parcel_key
from apps.base.map_keys import initial_map_key
from apps.base.models import MapCollection, MapFeature, Parcel, Region


def region_map_keys(names, *, file_input=False):
    """Resolve build inputs, including a standalone GeoJSON without a catalog."""
    keys = {name: initial_map_key(name) for name in names}
    if file_input:
        db = connection.settings_dict['NAME']
        if ((not connection.is_in_memory_db() and not Path(db).is_file())
                or Region._meta.db_table not in connection.introspection.table_names()):
            return keys
    catalog = Region.objects.filter(Q(name__in=keys) | Q(map_key__in=keys.values()))
    owners = {}
    for name, key in catalog.values_list('name', 'map_key'):
        owners[key] = name
        if name in keys:
            keys[name] = key
    for name, key in keys.items():
        if key in owners and owners[key] != name:
            raise CommandError(f'Map layer {name!r} was renamed to {owners[key]!r}. Use its current name.')
    return keys


def import_parcel_map(path):
    return import_initial_map(apps, DEFAULT_DB_ALIAS, path, existing_filter={'road_type__isnull': True})


def link_imported_parcels():
    """Bind a map-only installation to the parcel catalog during CSV bootstrap."""
    unlinked = list(MapFeature.objects.filter(parcel__isnull=True, road_type__isnull=True))
    if not unlinked:
        return
    parcels = {(r, n): pk for r, n, pk in Parcel.objects.values_list('region__name', 'name', 'pk')}
    changed = False
    for feature in unlinked:
        parcel_id = parcels.get(parcel_key(feature.members))
        if parcel_id is not None:
            feature.parcel_id = parcel_id
            feature.version += 1
            feature.save(update_fields=['parcel', 'version'])
            changed = True
    if changed:
        from apps.base.digests import mark_stale
        mark_stale('audit')


@transaction.atomic
def map_geojson(*, include_parcel_ids=False):
    collection = MapCollection.objects.filter(pk=1).values_list('members', flat=True).first()
    features = []
    for row in MapFeature.objects.filter(published=True, road_type__isnull=True).values_list(
            'members', 'geometry', 'parcel__region__name', 'parcel__name', 'parcel__eclass__coppice', 'parcel_id'):
        members, geometry, region, parcel, coppice, parcel_id = row
        if parcel is not None:
            members['properties'] = {**(members.get('properties') or {}), 'layer': region,
                                     'name': f'{region}-{parcel}', 'coppice': coppice}
        if include_parcel_ids:
            members['properties'] = {**(members.get('properties') or {}), 'parcel_id': parcel_id}
        features.append({**members, 'geometry': geometry})
    return {**(collection or {'type': 'FeatureCollection'}), 'features': features}


def road_geojson():
    roads = MapFeature.objects.filter(published=True, road_type__isnull=False).select_related('region', 'road_type')
    features = [{
        'type': 'Feature', 'id': road.pk, 'geometry': road.geometry,
        'properties': {
            'name': road.name, 'region_id': road.region_id, 'region': road.region.name,
            'road_type_id': road.road_type_id, 'road_type': road.road_type.name,
            'sort_order': road.road_type.sort_order,
        },
    } for road in roads]
    return {'type': 'FeatureCollection', 'features': features}


@gzip_page
def map_response(request, cache_control=CACHE_NO_CACHE, *, roads=False):
    raw = json.dumps(road_geojson() if roads else map_geojson(), ensure_ascii=False, separators=(',', ':')).encode()
    etag = 'W/"' + hashlib.sha256(raw).hexdigest() + '"'
    response = HttpResponse(raw, content_type='application/geo+json')
    response['ETag'] = etag
    apply_cache_control(response, cache_control)
    patch_vary_headers(response, ['Accept-Encoding'])
    return get_conditional_response(request, etag=etag, response=response)
