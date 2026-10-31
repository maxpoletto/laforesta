import base64
import json

import pytest
from django.test import Client

from apps.base import hypsometry
from apps.base.models import HypsoParam, HypsoParamSet, VolumeEquation

pytestmark = pytest.mark.django_db
BASE = '/api/impostazioni/equations/import/'


@pytest.fixture
def client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


def post(client, path, body):
    return client.post(BASE + path + '/', json.dumps(body), content_type='application/json')


def preview(client, kind, text, **options):
    response = post(client, 'preview', dict(kind=kind,
        file=base64.b64encode(text.encode()).decode(), **options))
    assert response.status_code == 200, response.content
    return response.json()


def test_height_preview_is_read_only_and_confirmation_preserves_unsupplied_rows(client, species, regions):
    old = hypsometry.replace_active_set([
        hypsometry.ParamRow(regions[1], species[1], 8, -5, .7, 30),
    ], source='imported', min_n=None, survey_ids=[])
    data = preview(client, 'height',
        f'compresa,genere,funzione,a,b,n,r2\n{regions[0].name},Abete,ln,7,-4,20,0.6\n')
    assert HypsoParamSet.objects.count() == 1
    assert data['token'] and not data['errors']
    response = post(client, 'confirm', dict(token=data['token'], nonce='confirm-height'))
    assert response.status_code == 200, response.content
    active = hypsometry.active_set()
    assert HypsoParam.objects.filter(param_set=active).count() == 2
    assert HypsoParam.objects.get(param_set=active, species=species[1]).origin_set_id == old.pk
    replay = post(client, 'confirm', dict(token=data['token'], nonce='confirm-height'))
    assert replay.json() == response.json() and HypsoParamSet.objects.count() == 2


def test_unmatched_species_requires_explicit_mapping(client, species, regions):
    text = f'compresa,genere,funzione,a,b\n{regions[0].name},Former name,ln,7,-4\n'
    data = preview(client, 'height', text)
    assert data['errors'] and not data['token']
    data = preview(client, 'height', text, species_map={'2': species[0].pk})
    assert data['token'] and not data['errors']
    assert post(client, 'confirm', dict(token=data['token'])).status_code == 200
    assert HypsoParam.objects.get(param_set__superseded_at__isnull=True).species_id == species[0].pk


def test_volume_preview_exposes_old_new_values_and_preserves_other_species(client, species):
    old = species[1].volume_equation.b0
    data = preview(client, 'volume', 'genere,b0,b1,b2,fonte\nAbete,1000,0,,Example\n')
    assert data['token']
    assert VolumeEquation.objects.get(species=species[0]).b0 != 1000
    assert post(client, 'confirm', dict(token=data['token'])).status_code == 200
    assert VolumeEquation.objects.get(species=species[0]).b0 == 1000
    assert VolumeEquation.objects.get(species=species[1]).b0 == old


@pytest.mark.parametrize('text', [
    'genere,b0,b1\nAbete,NaN,0\n',
    'genere,b0,b1\nAbete,1000,0\nAbete,2000,0\n',
])
def test_invalid_or_duplicate_rows_cannot_be_confirmed(client, species, text):
    before = VolumeEquation.history.count()
    data = preview(client, 'volume', text)
    assert data['errors'] and not data['token']
    assert VolumeEquation.history.count() == before


def test_stale_preview_is_rejected_before_any_write(client, species):
    data = preview(client, 'volume', 'genere,b0,b1\nAbete,1000,0\n')
    obj = species[1]
    obj.common_name = 'New name'
    obj.version += 1
    obj.save()
    before = VolumeEquation.history.count()
    response = post(client, 'confirm', dict(token=data['token'], nonce='stale-import'))
    assert response.status_code == 400 and response.json()['status'] == 'conflict'
    assert VolumeEquation.history.count() == before


def test_preview_token_cannot_be_changed_or_used_by_another_writer(client, species, admin_user):
    data = preview(client, 'volume', 'genere,b0,b1\nAbete,1000,0\n')
    assert post(client, 'confirm', dict(token=data['token'] + 'changed')).status_code == 400
    other = Client()
    other.force_login(admin_user)
    assert post(other, 'confirm', dict(token=data['token'])).status_code == 400


def test_reader_cannot_import(reader_user):
    client = Client()
    client.force_login(reader_user)
    assert post(client, 'preview', dict(kind='volume', file='')).status_code == 403


@pytest.mark.parametrize('token', [None, [], {}, 123])
def test_malformed_preview_token_returns_normal_validation(client, token):
    assert post(client, 'confirm', dict(token=token)).status_code == 400


def test_proxy_import_and_export_round_trip(client, species):
    text = 'genere,b0,b1,b2,fonte,riferimento\nCastagno,,,,Proxy example,Abete\n'
    data = preview(client, 'volume', text)
    assert not data['errors']
    assert post(client, 'confirm', dict(token=data['token'])).status_code == 200
    assert VolumeEquation.objects.get(species=species[1]).proxy_species_id == species[0].pk
    response = client.get('/api/impostazioni/equations/volume/export/')
    assert response.status_code == 200
    data = preview(client, 'volume', response.content.decode())
    assert not data['errors'], data['errors']


def test_circular_proxy_import_has_no_confirmable_token(client, species):
    data = preview(client, 'volume',
        'genere,b0,b1,riferimento\nAbete,,,Castagno\nCastagno,,,Abete\n')
    assert data['errors'] and not data['token']


def test_manual_height_export_remains_importable_without_fit_statistics(client, species, regions):
    hypsometry.merge_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, None, None),
    ], source='manual')
    response = client.get('/api/impostazioni/hypso-params/export/')
    assert response.status_code == 200
    data = preview(client, 'height', response.content.decode())
    assert data['token'] and not data['errors']


def test_preview_localizes_coefficients_and_unknown_statistics(client, species, regions):
    data = preview(client, 'height',
        f'compresa,genere,funzione,a,b,n,r2\n{regions[0].name},Abete,ln,7.25,-4,,\n')
    proposed = data['rows'][0][5]
    assert 'None' not in proposed
    assert '7,25' in proposed


def test_late_confirmation_failure_rolls_back_equations_history_and_nonce(client, species, monkeypatch):
    from apps.base.models import UsedNonce
    data = preview(client, 'volume', 'genere,b0,b1\nAbete,1000,0\n')
    before = list(VolumeEquation.objects.order_by('pk').values())
    count = VolumeEquation.history.count()
    def fail():
        raise RuntimeError('Injected invalidation failure')
    monkeypatch.setattr('apps.impostazioni.equation_imports.invalidate_equations', fail)
    client.raise_request_exception = False
    response = post(client, 'confirm', dict(token=data['token'], nonce='failed-confirm'))
    assert response.status_code == 500
    assert list(VolumeEquation.objects.order_by('pk').values()) == before
    assert VolumeEquation.history.count() == count
    assert not UsedNonce.objects.filter(nonce='failed-confirm').exists()
