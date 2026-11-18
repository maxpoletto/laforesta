"""Map delivery must work offline of the input file and revalidate actual content."""
import gzip
import json

import pytest
from django.test import Client, override_settings

from apps.base.models import MapFeature


@pytest.mark.django_db
@override_settings(IPSO_SECRET='field-secret')
def test_map_routes_enforce_existing_access(reader_user):
    client = Client()
    for name in ('terreni.geojson', 'map.geojson'):
        assert client.get('/api/geo/' + name).status_code == 302
        assert client.get('/ipso/' + name).status_code == 401
    client.force_login(reader_user)
    for name in ('terreni.geojson', 'map.geojson'):
        response = client.get('/api/geo/' + name)
        assert response.json() == {'type': 'FeatureCollection', 'features': []}
        assert 'no-cache' in response['Cache-Control']
        field = client.get('/ipso/' + name, HTTP_AUTHORIZATION='Bearer field-secret')
        assert field.json() == response.json()
        assert 'no-store' in field['Cache-Control'] and 'Authorization' in field['Vary']
    assert client.get('/api/geo/secret.txt').status_code == 404


def test_revalidate_geometry_and_parcel_metadata(reader_user, parcels, import_parcel_map):
    parcel = parcels[0]
    import_parcel_map(*parcels)
    client = Client()
    client.force_login(reader_user)
    url = '/api/geo/map.geojson'
    first = client.get(url)
    assert first['Content-Type'] == 'application/geo+json'
    assert client.get(url, HTTP_IF_NONE_MATCH=first['ETag']).status_code == 304
    part = MapFeature.objects.get(parcel=parcel)
    part.geometry['coordinates'][0].reverse()
    part.version += 1
    part.save()
    changed = client.get(url, HTTP_IF_NONE_MATCH=first['ETag'])
    assert changed.status_code == 200 and changed['ETag'] != first['ETag']
    parcel.name = '01b'
    parcel.save()
    renamed = client.get(url, HTTP_IF_NONE_MATCH=changed['ETag'], HTTP_ACCEPT_ENCODING='gzip')
    assert renamed.status_code == 200 and renamed['Content-Encoding'] == 'gzip'
    assert json.loads(gzip.decompress(renamed.content))['features'][0]['properties']['name'].endswith('-01b')
    cached = client.get(url, HTTP_IF_NONE_MATCH=renamed['ETag'], HTTP_ACCEPT_ENCODING='gzip')
    assert cached.status_code == 304 and cached['ETag'] == renamed['ETag']
    assert 'Accept-Encoding' in cached['Vary']


@override_settings(IPSO_SECRET='field-secret')
def test_populated_map_revalidates_across_desktop_and_ipso(reader_user, parcels, eclasses, import_parcel_map):
    import_parcel_map(*parcels)
    desktop, ipso = Client(), Client()
    desktop.force_login(reader_user)
    url = '/api/geo/map.geojson'
    token = {'HTTP_AUTHORIZATION': 'Bearer field-secret'}
    original = desktop.get(url)
    for name in ('map.geojson', 'terreni.geojson'):
        mobile = ipso.get('/ipso/' + name, **token)
        assert mobile.content == original.content
        assert mobile['ETag'] == original['ETag']

    cached = {'HTTP_IF_NONE_MATCH': original['ETag']}
    response = ipso.get('/ipso/map.geojson', **token, **cached)
    assert response.status_code == 304 and 'no-store' in response['Cache-Control']
    assert 'Authorization' in response['Vary']
    assert ipso.get('/ipso/map.geojson', **cached).status_code == 401

    region = parcels[0].region
    region.name = 'Costa-Nord'
    region.save()
    parcels[0].eclass = eclasses[2]
    parcels[0].intervention_interval = 18
    parcels[0].standards_per_ha = 75
    parcels[0].save()
    updated = ipso.get('/ipso/map.geojson', **token, **cached)
    assert updated.status_code == 200 and updated['ETag'] != original['ETag']
    assert desktop.get(url, **cached).content == updated.content
    labels = [(f['properties']['name'], f['properties']['coppice']) for f in updated.json()['features']]
    assert labels == [('Costa-Nord-1', True), ('Costa-Nord-2', False), ('Fabrizia-1', False)]
    assert MapFeature.history.count() == len(parcels)


def test_public_roads_are_separate_from_parcel_outlines(reader_user, regions):
    from apps.base.models import RoadType

    client = Client()
    client.force_login(reader_user)
    kind = RoadType.objects.first()
    road = MapFeature.objects.create(
        region=regions[0], road_type=kind, name='Pista est', published=False,
        geometry={'type': 'LineString', 'coordinates': [[16, 38], [16.1, 38.1]]},
    )
    url = '/api/geo/roads.geojson'
    assert client.get(url).json()['features'] == []
    road.published = True
    road.save()
    feature, = client.get(url).json()['features']
    assert feature['id'] == road.pk
    assert feature['properties'] == {
        'name': road.name, 'region_id': regions[0].pk, 'region': regions[0].name,
        'road_type_id': kind.pk, 'road_type': kind.name, 'sort_order': kind.sort_order,
    }
    assert feature['geometry'] == road.geometry
    assert client.get('/api/geo/map.geojson').json()['features'] == []
