import base64
import json

import pytest
from django.test import Client

from apps.base import hypsometry
from apps.base.models import HypsoParam

pytestmark = pytest.mark.django_db


@pytest.fixture
def client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


def post(client, path, body):
    return client.post('/api/impostazioni/hypso-params/' + path + '/',
                       json.dumps(body), content_type='application/json')


def test_csv_update_preserves_an_unmentioned_species_region(client, species, regions):
    hypsometry.replace_active_set([
        hypsometry.ParamRow(regions[1], species[1], 8, -5, .7, 30),
    ], source='imported', min_n=None, survey_ids=[])
    text = f'compresa,genere,funzione,a,b,r2,n\n{regions[0].name},Abete,ln,7,-4,0.6,20\n'
    assert post(client, 'import', {'file': base64.b64encode(text.encode()).decode()}).status_code == 200
    assert HypsoParam.objects.filter(param_set__superseded_at__isnull=True).count() == 2


def test_computed_update_preserves_unfitted_pairs_and_detects_changed_preview(
        client, hypso_samples, species, regions):
    old = hypsometry.replace_active_set([
        hypsometry.ParamRow(regions[2], species[1], 8, -5, .7, 30),
    ], source='imported', min_n=None, survey_ids=[])
    body = {'survey_ids': [hypso_samples['survey'].pk], 'min_n': 5}
    response = post(client, 'compute', body)
    assert response.status_code == 200
    candidate = response.json()
    response = post(client, 'accept', {**body, 'equation_state': candidate['equation_state'],
                                      'candidate_state': candidate['candidate_state']})
    assert response.status_code == 200, response.content
    assert HypsoParam.objects.filter(param_set__superseded_at__isnull=True,
                                    region=regions[2], species=species[1], origin_set=old).exists()
    assert post(client, 'accept', {**body, 'equation_state': candidate['equation_state'],
                                'candidate_state': candidate['candidate_state']}).status_code == 400
