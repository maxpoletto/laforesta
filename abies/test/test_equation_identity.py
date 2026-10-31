"""Species renames must preserve equation identity and recorded estimates."""
from decimal import Decimal
import json

import pytest
from django.test import Client

from apps.base import hypsometry
from apps.base.models import HypsoParam, HypsoParamSet, Species, User, Role
from apps.piano_di_taglio.mark_import import mark_volume_and_mass
from apps.campionamenti.csv_trees import tree_volume_and_mass
from config.constants import COLUMNS, ROWS

pytestmark = pytest.mark.django_db


def test_volume_equation_has_a_species_identity_and_survives_rename(species):
    fir = species[0]
    equation = fir.volume_equation
    assert equation.species_id == fir.pk
    assert equation.coefficients == (Decimal('-1.8381'), Decimal('0.037836'), Decimal('0.39934'))
    before = mark_volume_and_mass(30, Decimal('20'), fir)
    fir.common_name = 'Abete bianco'
    fir.save()
    fir.refresh_from_db()
    assert mark_volume_and_mass(30, Decimal('20'), fir) == before
    assert tree_volume_and_mass(False, 30, Decimal('20'), fir) == before


def test_explicit_equation_changes_drive_calculation_instead_of_name(species):
    fir = species[0]
    equation = fir.volume_equation
    equation.b0 = 1000
    equation.b1 = 0
    equation.b2 = None
    equation.save()
    assert mark_volume_and_mass(30, Decimal('20'), fir)[0] == Decimal('1.0000')
    equation.delete()
    fir.refresh_from_db()
    assert mark_volume_and_mass(30, Decimal('20'), fir) == (None, None)


def test_volume_proxy_keeps_its_own_species_and_density(species):
    fir, bucket = species[0], species[4]
    assert bucket.volume_equation.species_id == bucket.pk
    assert bucket.volume_equation.proxy_species_id == fir.pk
    bucket.density = Decimal('7.25')
    bucket.save()
    volume, mass = mark_volume_and_mass(30, Decimal('20'), bucket)
    assert volume == mark_volume_and_mass(30, Decimal('20'), fir)[0]
    assert mass == (volume * bucket.density).quantize(Decimal('.001'))


def test_unsupported_species_does_not_gain_an_equation_from_classification():
    spruce = Species.objects.create(common_name='Abete rosso', latin_name='Picea abies',
                                    botanical_type='conifer')
    assert mark_volume_and_mass(30, Decimal('20'), spruce) == (None, None)


def test_renaming_refreshes_hypso_digest_without_changing_foreign_keys(species, regions):
    fir = species[0]
    param_set = hypsometry.replace_active_set([
        hypsometry.ParamRow(regions[0], fir, 7, -4, .6, 20),
    ], source='imported', min_n=None, survey_ids=[])
    client = Client()
    client.force_login(User.objects.create_user(username='equations', role=Role.ADMIN))
    url = '/api/impostazioni/hypso-params/data/'
    old = client.get(url)
    assert old.status_code == 200
    response = client.post('/api/impostazioni/species/save/', json.dumps({
        'row_id': fir.pk, 'version': fir.version, 'common_name': 'Abete bianco',
        'latin_name': fir.latin_name, 'density': str(fir.density),
        'pressler_default': '2', 'botanical_type': 'conifer', 'active': True,
    }), content_type='application/json')
    assert response.status_code == 200
    # The server digest is gzip; Django's client does not decode it for json().
    import gzip
    digest = json.loads(gzip.decompress(b''.join(client.get(url).streaming_content)))
    columns, rows = digest[COLUMNS], digest[ROWS]
    assert rows[0][columns.index('species_id')] == fir.pk
    assert rows[0][columns.index('region_id')] == regions[0].pk
    assert 'Abete bianco' in rows[0]
    assert HypsoParam.objects.get(param_set=param_set).species_id == fir.pk
    assert HypsoParamSet.objects.count() == 1


def test_height_edit_preserves_other_regions_species_and_archives_previous(species, regions):
    old = hypsometry.replace_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, .6, 20),
        hypsometry.ParamRow(regions[1], species[0], 8, -5, .7, 30),
        hypsometry.ParamRow(regions[0], species[1], 9, -6, .8, 40),
    ], source='imported', min_n=None, survey_ids=[])
    new = hypsometry.merge_active_set([
        hypsometry.ParamRow(regions[0], species[0], 10, -3, None, None),
    ], source='manual')
    assert HypsoParam.objects.filter(param_set=new).count() == 3
    assert HypsoParam.objects.get(param_set=new, region=regions[1], species=species[0]).a == 8
    assert HypsoParam.objects.get(param_set=new, region=regions[0], species=species[1]).a == 9
    assert HypsoParam.objects.get(param_set=old, region=regions[0], species=species[0]).a == 7
    old.refresh_from_db()
    assert old.superseded_at is not None


def test_height_proxy_is_not_implied_by_volume_proxy(species, regions):
    active = hypsometry.merge_active_set([
        hypsometry.ParamRow(regions[0], species[0], 7, -4, .6, 20),
    ], source='manual')
    assert not HypsoParam.objects.filter(param_set=active, species=species[4]).exists()
