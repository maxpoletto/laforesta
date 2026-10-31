import json
from decimal import Decimal

import pytest
from django.test import Client

from apps.base import hypsometry
from apps.base.models import HypsoParam, HypsoParamSet, VolumeEquation, Species

pytestmark = pytest.mark.django_db


@pytest.fixture
def client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


def post(client, body):
    return client.post('/api/impostazioni/species/save/', json.dumps(body), content_type='application/json')


def form_state(client, species):
    response = client.get(f'/api/impostazioni/species/form/{species.pk}/')
    import re
    from html import unescape
    html = response.json()['html']
    match = re.search(r'name="equation_state" value="([^"]+)"', html)
    assert match, 'Unified form must carry equation concurrency state'
    return unescape(match.group(1))


def body(client, species, **changes):
    return dict(row_id=species.pk, version=species.version,
                common_name=species.common_name, latin_name=species.latin_name,
                density=str(species.density), pressler_default='2',
                botanical_type=species.botanical_type, active=True,
                equation_state=form_state(client, species), **changes)


def test_combined_edit_persists_species_and_equations_together(client, species, regions):
    fir = species[0]
    payload = body(client, fir,
        height_equations=[dict(region_id=regions[0].pk, enabled=True, a='7', b='-4', n='', r2='')],
        volume_equation=dict(mode='own', b0='1000', b1='0', b2='', source='Example source'))
    payload['common_name'] = 'Abete bianco'
    response = post(client, payload)
    assert response.status_code == 200, response.content
    fir.refresh_from_db()
    assert fir.common_name == 'Abete bianco' and fir.version == 2
    assert fir.volume_equation.b0 == 1000
    assert fir.volume_equation.source == 'Example source'
    p = HypsoParam.objects.get(param_set__superseded_at__isnull=True, species=fir)
    assert p.region_id == regions[0].pk and p.a == 7 and p.source == 'manual'
    assert p.n is None and p.r2 is None


def test_form_exposes_all_equations_and_per_region_coverage(client, species, regions):
    response = client.get(f'/api/impostazioni/species/form/{species[0].pk}/')
    html = response.json()['html']
    assert 'volume_b0' in html and 'volume_source' in html
    for region in regions:
        assert f'data-region-id="{region.pk}"' in html
    data = client.get('/api/impostazioni/species/data/').json()
    assert 'Eq. altezza' in data['columns'] and 'Eq. volume' in data['columns']


@pytest.mark.parametrize('bad', [
    dict(mode='own', b0='NaN', b1='0'),
    dict(mode='own', b0='1', b1='bad'),
    dict(mode='proxy', proxy_species_id=999999),
])
def test_invalid_equation_leaves_metadata_and_history_unchanged(client, species, bad):
    fir = species[0]
    payload = body(client, fir, volume_equation=bad)
    payload['common_name'] = 'Must not persist'
    before = (fir.history.count(), VolumeEquation.history.count(), HypsoParamSet.objects.count())
    response = post(client, payload)
    assert response.status_code == 400
    fir.refresh_from_db()
    assert fir.common_name == 'Abete' and fir.version == 1
    assert before == (fir.history.count(), VolumeEquation.history.count(), HypsoParamSet.objects.count())


def test_stale_equation_snapshot_cannot_overwrite_newer_changes(client, species):
    fir = species[0]
    payload = body(client, fir, volume_equation=dict(mode='none'))
    equation = fir.volume_equation
    equation.b0 = Decimal('12')
    equation.version += 1
    equation.save()
    response = post(client, payload)
    assert response.status_code == 400 and response.json()['status'] == 'conflict'
    equation.refresh_from_db()
    assert equation.b0 == 12


def test_cannot_remove_an_equation_needed_by_a_proxy(client, species):
    response = post(client, body(client, species[0], volume_equation=dict(mode='none')))
    assert response.status_code == 400
    assert VolumeEquation.objects.filter(species=species[0]).exists()


def test_height_removal_is_scoped_and_other_rows_retain_provenance(client, species, regions):
    old = hypsometry.replace_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, .6, 20),
        hypsometry.ParamRow(regions[1], species[1], 8, -5, .7, 30),
    ], source='imported', min_n=None, survey_ids=[])
    response = post(client, body(client, species[0],
        height_equations=[dict(region_id=regions[0].pk, enabled=False)]))
    assert response.status_code == 200, response.content
    current = HypsoParam.objects.get(param_set__superseded_at__isnull=True)
    assert current.species_id == species[1].pk and current.origin_set_id == old.pk
    assert current.source == 'imported' and current.n == 30
    assert HypsoParam.objects.filter(param_set=old).count() == 2


def test_reader_cannot_edit_equations(reader_user, species):
    client = Client()
    client.force_login(reader_user)
    assert client.get(f'/api/impostazioni/species/form/{species[0].pk}/').status_code == 403


def test_form_snapshot_precedes_loading_its_equation_values(species, regions, monkeypatch):
    from apps.impostazioni import species_equations as editor
    hypsometry.merge_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, None, None),
    ], source='manual')
    original = editor.equation_state
    changed = False
    def concurrent_change():
        nonlocal changed
        if not changed:
            changed = True
            hypsometry.merge_active_set([
                hypsometry.ParamRow(regions[0], species[0], 8, -4, None, None),
            ], source='manual')
        return original()
    monkeypatch.setattr(editor, 'equation_state', concurrent_change)
    context = editor.editor_context(species[0])
    row = next(row for row in context['height_regions'] if row['region'].pk == regions[0].pk)
    assert row['equation'].a == 8, 'A fresh token must not accompany old displayed coefficients'
