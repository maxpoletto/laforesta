"""Database-backed GeoJSON shared by desktop, offline devices and GIS builds."""

import hashlib
import json

from django.db import transaction
from django.http import HttpResponse
from django.utils.cache import get_conditional_response, patch_vary_headers
from django.views.decorators.gzip import gzip_page

from apps.base.http import CACHE_NO_CACHE, apply_cache_control
from apps.base.map_import_v1 import parcel_key
from apps.base.models import MapCollection, MapFeature, Parcel


def link_imported_parcels():
    """Bind a map-only installation to the parcel catalog during CSV bootstrap."""
    unlinked = list(MapFeature.objects.filter(parcel__isnull=True))
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
def map_geojson():
    collection = MapCollection.objects.filter(pk=1).values_list('members', flat=True).first()
    features = []
    for row in MapFeature.objects.values_list(
            'members', 'geometry', 'parcel__region__name', 'parcel__name', 'parcel__eclass__coppice'):
        members, geometry, region, parcel, coppice = row
        if parcel is not None:
            members['properties'] = {**(members.get('properties') or {}), 'layer': region,
                                     'name': f'{region}-{parcel}', 'coppice': coppice}
        features.append({**members, 'geometry': geometry})
    return {**(collection or {'type': 'FeatureCollection'}), 'features': features}


@gzip_page
def map_response(request, cache_control=CACHE_NO_CACHE):
    raw = json.dumps(map_geojson(), ensure_ascii=False, separators=(',', ':')).encode()
    etag = 'W/"' + hashlib.sha256(raw).hexdigest() + '"'
    response = HttpResponse(raw, content_type='application/geo+json')
    response['ETag'] = etag
    apply_cache_control(response, cache_control)
    patch_vary_headers(response, ['Accept-Encoding'])
    return get_conditional_response(request, etag=etag, response=response)
