"""Malformed JSON must yield usable 400 responses, not coercion or partial writes."""

import json
import sys

import pytest
from django.test import Client

from apps.base.models import (
    HarvestPlan, HarvestPlanItem, Sample, SampleArea, SampleGrid, Species, Survey,
    Tree, TreeMark, TreeSample, UsedNonce, User,
)
from config.constants import FIELD_ERRORS


@pytest.fixture
def inputs(admin_user, parcels, species):
    client = Client(raise_request_exception=False)
    client.force_login(admin_user)
    grid = SampleGrid.objects.create(name='Input grid')
    area = SampleArea.objects.create(sample_grid=grid, parcel=parcels[0], number='1', lat=0, lon=0)
    survey = Survey.objects.create(name='Input survey', sample_grid=grid)
    plan = HarvestPlan.objects.create(name='Input plan', year_start=2026, year_end=2027)
    item = HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcels[0], year_planned=2026)
    return client, {
        'tree': ('/api/campionamenti/tree/save/', {
            'survey_id': survey.pk, 'sample_area_id': area.pk, 'number': 1,
            'species_id': species[0].pk, 'd_cm': 30, 'h_m': '20', 'fustaia': 'true',
        }),
        'mark': ('/api/piano-di-taglio/mark/save/', {
            'harvest_plan_item_id': item.pk, 'species_id': species[0].pk,
            'd_cm': 30, 'h_m': '20', 'date': '2026-10-25', 'operator': 'Test',
        }),
        'grid': ('/api/campionamenti/grid/save/', {'name': 'New grid'}),
        'auto-grid': ('/api/campionamenti/grid/save-auto/', {'name': 'New auto grid'}),
        'species': ('/api/impostazioni/species/save/', {
            'common_name': 'New species', 'latin_name': '', 'density': '9',
            'pressler_default': '2', 'botanical_type': 'conifer', 'active': True,
        }),
        'user': ('/api/impostazioni/users/save/', {
            'username': 'newuser', 'email': 'new@example.com', 'role': 'reader',
            'login_method': 'password', 'password1': 'securePassword99!',
            'password2': 'securePassword99!', 'is_active': True,
        }),
    }


@pytest.mark.parametrize('kind,override', [
    ('tree', {'d_cm': 1.9}), ('tree', {'d_cm': True}),
    ('tree', {'survey_id': float('inf')}), ('tree', {'survey_id': 10 ** 100}),
    ('tree', {'h_m': '1e29'}), ('tree', {'lat': '1e29'}), ('tree', {'lat': '91'}),
    ('tree', {'pressler_coeff': 'invalid'}),
    ('tree', {'fustaia': 'false', 'shoots': [1]}),
    ('tree', {'fustaia': 'false', 'shoots': [None]}),
    ('tree', {'fustaia': 'false', 'shoots': [{'shoot': 1, 'd_cm': 1.9, 'h_m': 20}]}),
    ('grid', {'name': {'wrong': 'type'}}), ('grid', {'description': [1]}),
    ('species', {'common_name': 123}), ('species', {'density': '1e29'}),
    ('user', {'email': []}), ('user', {'username': 123}),
    ('user', {'role': 'root'}), ('user', {'login_method': 'unknown'}),
    ('user', {'password1': ['not', 'text']}),
    ('mark', {'operator': {'wrong': 'type'}}), ('mark', {'h_m': '1e29'}),
    ('mark', {'d_cm': 10 ** 15}), ('tree', {'d_cm': 10 ** 15}),
    ('grid', {'active': [1]}),
    ('auto-grid', {'points': [1]}),
    ('auto-grid', {'points': [{'compresa': {}, 'particella': '1', 'lat': 0, 'lon': 0}]}),
    ('auto-grid', {'points': [{'compresa': 'Capistrano', 'particella': '1', 'lat': '1e29', 'lon': 0}]}),
])
def test_bad_inputs_return_validation_error_without_writes(inputs, kind, override):
    client, payloads = inputs
    url, body = payloads[kind]
    models = [Tree, TreeSample, Sample, TreeMark, Species, SampleGrid, User, UsedNonce]
    before = [model.objects.count() for model in models]
    response = client.post(url, json.dumps({**body, **override, 'nonce': 'invalid-input'}),
                           content_type='application/json')
    assert response.status_code == 400, response.content[:200]
    data = response.json()
    assert data['status'] == 'validation_error'
    assert data['message'] and data[FIELD_ERRORS]
    assert not data.get('patches') and not data.get('deletes')
    assert [model.objects.count() for model in models] == before


def test_false_string_is_not_a_standard_tree(inputs):
    client, payloads = inputs
    url, body = payloads['tree']
    body = {**body, 'fustaia': 'false', 'shoots': [
        {'shoot': 1, 'd_cm': 10, 'h_m': '5', 'standard': 'false'},
    ]}
    response = client.post(url, json.dumps(body), content_type='application/json')
    assert response.status_code == 200, response.content
    assert TreeSample.objects.get().standard is False


def test_deep_json_remains_a_validation_error_with_nonce(inputs):
    client, _ = inputs
    depth = sys.getrecursionlimit() + 100
    raw = b'{"name":"Deep","nonce":"deep","extra":' + b'[' * depth + b'0' + b']' * depth + b'}'
    response = client.post('/api/campionamenti/grid/save/', raw, content_type='application/json')
    assert response.status_code == 400
    assert response.json()['status'] == 'validation_error'


def test_request_json_is_decoded_once():
    from unittest.mock import patch
    from django.test import RequestFactory
    from apps.base.validation import read_json_body

    request = RequestFactory().post('/api/test/', '{"id":1}', content_type='application/json')
    with patch('apps.base.validation.json.loads', wraps=json.loads) as decode:
        assert read_json_body(request) is read_json_body(request)
        assert decode.call_count == 1


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '-Infinity', '1e400'])
def test_nonfinite_json_numbers_are_rejected_everywhere(inputs, number):
    client, _ = inputs
    raw = '{"name":"Nonfinite","extra":' + number + '}'
    response = client.post('/api/campionamenti/grid/save/', raw, content_type='application/json')
    assert response.status_code == 400
    assert response.json()['status'] == 'validation_error'


@pytest.mark.parametrize('kind', ['tree', 'mark'])
@pytest.mark.parametrize('diameter', [10000, 10 ** 15])
def test_derived_storage_overflow_rolls_back_without_a_nonce(inputs, kind, diameter):
    client, payloads = inputs
    url, body = payloads[kind]
    before = [model.objects.count() for model in (Tree, Sample, TreeSample, TreeMark)]
    response = client.post(url, json.dumps({**body, 'd_cm': diameter}),
                           content_type='application/json')
    assert response.status_code == 400, response.content[:100]
    assert response.json()['status'] == 'validation_error'
    assert [model.objects.count() for model in (Tree, Sample, TreeSample, TreeMark)] == before


def test_late_import_overflow_rolls_back_all_rows(inputs):
    from datetime import date
    from decimal import Decimal
    from apps.base.validation import InputValidationError
    from apps.piano_di_taglio.mark_import import MarkImportRow, import_mark_rows

    _, payloads = inputs
    body = payloads['mark'][1]
    item = HarvestPlanItem.objects.get(pk=body['harvest_plan_item_id'])
    species = Species.objects.get(pk=body['species_id'])
    rows = [MarkImportRow(
        date=date(2026, 10, 25), parcel=item.parcel, species=species,
        number=index, d_cm=diameter, h_m=Decimal('20'), h_measured=True,
        lat=None, lon=None, acc_m=None, operator='Test', fingerprint=f'overflow-{index}',
    ) for index, diameter in enumerate([30, 10000], start=1)]
    with pytest.raises(InputValidationError):
        import_mark_rows(item, rows)
    assert not Tree.objects.exists()
    assert not TreeMark.objects.exists()
