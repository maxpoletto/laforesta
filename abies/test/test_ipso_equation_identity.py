import json

import pytest
from django.test import Client, override_settings

from apps.base import hypsometry

pytestmark = pytest.mark.django_db


@override_settings(IPSO_SECRET='equation-test')
def test_reference_contains_id_equations_and_legacy_projection(species, regions):
    hypsometry.merge_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, .6, 20),
    ], source='imported')
    sp = species[0]
    sp.common_name = 'Abete bianco'
    sp.save()
    response = Client().get('/ipso/reference.json', HTTP_AUTHORIZATION='Bearer equation-test')
    assert response.status_code == 200
    data = json.loads(response.content)
    equation = data['hypsometry_by_id'][str(regions[0].pk)][str(sp.pk)]
    assert equation['a'] == 7 and equation['b'] == -4
    assert data['ipsometrica'][regions[0].name]['Abete bianco'] == equation
