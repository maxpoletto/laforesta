"""Tests for the Abies-served Ipso PWA."""

import json
from decimal import Decimal

from django.test import Client

from apps.base.models import (
    HYPSO_FUNC_LN, HypsoParam, HypsoParamSet, HypsoParamSource,
)


def _body(resp) -> bytes:
    return b''.join(resp.streaming_content)


def test_index_is_public_and_uses_relative_assets(db):
    resp = Client().get('/ipso/')

    assert resp.status_code == 200
    body = _body(resp).decode()
    assert '<title>Ipso' in body
    assert 'href="style.css"' in body
    assert 'src="app.js"' in body


def test_service_worker_served_from_ipso_scope(db):
    resp = Client().get('/ipso/sw.js')

    assert resp.status_code == 200
    assert resp['Content-Type'].startswith('text/javascript')
    assert b'ipso service worker' in _body(resp)


def test_manifest_is_served(db):
    resp = Client().get('/ipso/manifest.webmanifest')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/manifest+json'
    assert json.loads(_body(resp))['short_name'] == 'Ipso'


def test_reference_json_comes_from_abies_data(db, regions, parcels, species):
    active = HypsoParamSet.objects.create(source=HypsoParamSource.IMPORTED)
    HypsoParam.objects.create(
        param_set=active, region=regions[0], species=species[0],
        func=HYPSO_FUNC_LN, a=Decimal('7.0000'), b=Decimal('-4.0000'),
        r2=Decimal('0.5000'), n=12,
    )

    resp = Client().get('/ipso/reference.json')

    assert resp.status_code == 200
    data = resp.json()
    assert data['schema_version'] == 1
    assert data['species'][0]['common'] == 'Abete'
    assert {'compresa': 'Capistrano', 'particella': '1'} in data['parcels']
    assert data['ipsometrica']['Capistrano']['Abete'] == {'a': 7.0, 'b': -4.0}


def test_terreni_geojson_has_empty_fallback(db):
    resp = Client().get('/ipso/terreni.geojson')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/geo+json'
    assert resp.json() == {'type': 'FeatureCollection', 'features': []}
