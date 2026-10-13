"""Species classification, catch-all defaults, and the legacy boundary."""

import ast
import gzip
import importlib
import inspect
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from django.apps import apps
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import Client, override_settings

from apps.base import csv_io, csv_reference
from apps.base.digests import generate_species
from apps.base.models import (
    DigestStatus, HarvestPlan, HarvestPlanItem, HarvestPlanItemState, Sample, SampleArea, SampleGrid,
    Species, SpeciesType, Survey, Tree, TreeMark, TreeSample, tree_mass_q,
)
from apps.base.species import ensure_other_species, initial_botanical_type, selectable_species
from apps.base.tabacchi import tabacchi_volume_m3
from apps.campionamenti.csv_trees import tree_volume_and_mass
from apps.piano_di_taglio.mark_import import mark_volume_and_mass
from apps.prelievi.models import Harvest, HarvestSpecies
from config import strings as S

pytestmark = pytest.mark.django_db


@pytest.fixture
def client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


def post(client, url, payload):
    return client.post(url, json.dumps(payload), content_type='application/json')


def digest(response):
    assert response.status_code == 200
    return json.loads(gzip.decompress(response.getvalue()))


@pytest.mark.parametrize('name,latin,expected', [
    ('Abete', 'Abies alba', SpeciesType.CONIFER),
    ('Larice', 'Larix decidua', SpeciesType.CONIFER),
    ('Tasso', '', SpeciesType.CONIFER),
    ('Faggio', 'Fagus sylvatica', SpeciesType.BROADLEAF),
    ('Acero', '', SpeciesType.BROADLEAF),
    (S.SPECIES_OTHER_CONIFER, '', SpeciesType.CONIFER),
    (S.SPECIES_OTHER_BROADLEAF, '', SpeciesType.BROADLEAF),
    (S.SPECIES_OTHER, '', ''),
    ('Custom unknown', '', ''),
])
def test_initial_classification(name, latin, expected):
    assert initial_botanical_type(name, latin) == expected


def test_common_name_fallbacks_live_in_the_localized_catalog():
    module = importlib.import_module('apps.base.species')
    literals = {
        node.value for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    common_names = {*S.SPECIES_CONIFER_NAMES, *S.SPECIES_BROADLEAF_NAMES,
                    S.SPECIES_OTHER, S.SPECIES_OTHER_CONIFER, S.SPECIES_OTHER_BROADLEAF}
    assert not literals & common_names
    for names, kind in ((S.SPECIES_CONIFER_NAMES, SpeciesType.CONIFER),
                        (S.SPECIES_BROADLEAF_NAMES, SpeciesType.BROADLEAF)):
        assert all(initial_botanical_type(name) == kind for name in names)


def test_defaults_copy_current_reference_values_once():
    fir = Species.objects.create(common_name='Abete bianco', latin_name='Abies alba',
                                 density='8.75', pressler_default='1.75')
    beech = Species.objects.create(common_name='Faggio', latin_name='Fagus sylvatica',
                                   density='11.25', pressler_default='2.25')
    assert ensure_other_species() == 3
    assert not Species.objects.get(common_name=S.SPECIES_OTHER).selectable
    for name, source, kind in (
        (S.SPECIES_OTHER_CONIFER, fir, SpeciesType.CONIFER),
        (S.SPECIES_OTHER_BROADLEAF, beech, SpeciesType.BROADLEAF),
    ):
        row = Species.objects.get(common_name=name)
        source.refresh_from_db()
        assert (row.density, row.pressler_default) == (source.density, source.pressler_default)
        assert row.botanical_type == kind
        assert row.active and not row.minor
    conifer = Species.objects.get(common_name=S.SPECIES_OTHER_CONIFER)
    conifer.density = Decimal('7.50')
    conifer.save()
    assert ensure_other_species() == 0
    conifer.refresh_from_db()
    assert conifer.density == Decimal('7.50')


def test_legacy_choice_is_existing_only(species):
    legacy = species[3]
    assert legacy.active and not legacy.selectable
    assert legacy not in list(selectable_species())
    assert legacy in list(selectable_species(existing_id=legacy.id))
    assert species[4].selectable and species[5].selectable


@pytest.mark.parametrize('index,proxy', [(4, S.SPECIES_SILVER_FIR), (5, S.SPECIES_BEECH)])
def test_catch_all_mass_uses_its_own_density(species, index, proxy):
    bucket = species[index]
    bucket.density = Decimal('7.25')
    expected = tabacchi_volume_m3(30, 20, proxy)
    for result in (mark_volume_and_mass(30, 20, bucket),
                   tree_volume_and_mass(False, 30, 20, bucket)):
        assert result == (expected, tree_mass_q(expected, bucket.density))
    assert tree_volume_and_mass(True, 30, 20, bucket) == (None, None)
    assert mark_volume_and_mass(30, None, bucket) == (None, None)


def test_settings_rows_and_digest_have_identical_shape(client, species, settings):
    data = client.get('/api/impostazioni/species/data/').json()
    generate_species()
    with gzip.open(settings.DIGEST_DIR / 'species.json.gz', 'rt') as stream:
        lookup = json.load(stream)
    assert data['columns'] == lookup['columns']
    assert sorted(data['rows']) == sorted(lookup['rows'])
    rows = {row[data['columns'].index(S.COL_NAME)]: row for row in data['rows']}
    assert S.COL_CONIFER not in data['columns']
    assert S.COL_BROADLEAF not in data['columns']
    for name, kind in (
        (S.SPECIES_OTHER, ''),
        (S.SPECIES_OTHER_CONIFER, S.COL_CONIFER),
        (S.SPECIES_OTHER_BROADLEAF, S.COL_BROADLEAF),
    ):
        assert rows[name][data['columns'].index(S.COL_TYPE)] == kind


def test_inline_type_save_is_versioned_audited_and_field_scoped(client, species, writer_user):
    row = species[2]
    old_density, old_name, old_minor = row.density, row.common_name, row.minor
    DigestStatus.objects.create(name='prelievi')
    response = post(client, '/api/impostazioni/species/type/save/', {
        'row_id': row.id, 'version': row.version, 'botanical_type': 'conifer',
    })
    assert response.status_code == 200, response.content
    row.refresh_from_db()
    assert row.version == 2 and row.botanical_type == SpeciesType.CONIFER
    assert (row.density, row.common_name, row.minor) == (old_density, old_name, old_minor)
    assert row.history.first().history_user == writer_user
    assert DigestStatus.objects.get(name='prelievi').stale
    audit = digest(client.get('/api/controllo/data/'))
    rendered = json.dumps(audit, ensure_ascii=False)
    assert S.COL_BOTANICAL_TYPE in rendered
    assert S.COL_CONIFER in rendered and S.COL_BROADLEAF in rendered
    assert response.json()['patches'][0]['record'][1] == row.version
    columns = client.get('/api/impostazioni/species/data/').json()['columns']
    assert response.json()['patches'][0]['record'][columns.index(S.COL_TYPE)] == S.COL_CONIFER


def test_inline_type_rejects_stale_save(client, species):
    response = post(client, '/api/impostazioni/species/type/save/', {
        'row_id': species[0].id, 'version': 99, 'botanical_type': 'broadleaf',
    })
    assert response.status_code == 400
    assert response.json()['status'] == 'conflict'
    species[0].refresh_from_db()
    assert species[0].botanical_type == SpeciesType.CONIFER


@pytest.mark.parametrize('index,kind', [(3, 'conifer'), (4, 'broadleaf'), (5, 'conifer')])
def test_bucket_classifications_are_fixed(client, species, index, kind):
    response = post(client, '/api/impostazioni/species/type/save/', {
        'row_id': species[index].id, 'version': 1, 'botanical_type': kind,
    })
    assert response.status_code == 400


def test_reader_cannot_change_species_type(reader_user, species):
    reader = Client()
    reader.force_login(reader_user)
    assert post(reader, '/api/impostazioni/species/type/save/', {
        'row_id': species[0].id, 'version': 1, 'botanical_type': 'broadleaf',
    }).status_code == 403


def test_new_species_requires_one_classification(client):
    body = {'common_name': 'New', 'density': '9.0', 'active': True}
    assert post(client, '/api/impostazioni/species/save/', body).status_code == 400
    body['botanical_type'] = 'conifer'
    assert post(client, '/api/impostazioni/species/save/', body).status_code == 200
    assert Species.objects.get(common_name='New').botanical_type == SpeciesType.CONIFER


@pytest.fixture
def item(parcels):
    plan = HarvestPlan.objects.create(name='Species plan', year_start=2024, year_end=2034)
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2024,
        state=HarvestPlanItemState.OPEN,
    )


def test_minor_groups_split_without_relabeling_legacy_records(client, species, parcels, crews, products):
    larch = Species.objects.create(common_name='Larice', latin_name='Larix decidua',
                                  minor=True, botanical_type=SpeciesType.CONIFER)
    harvest = Harvest.objects.create(date='2024-06-01', parcel=parcels[0],
                                     crew=crews[0], product=products[0],
                                     mass_q=100, volume_m3='12.345')
    for row, pct in ((larch, 45), (species[2], 35), (species[3], 20)):
        HarvestSpecies.objects.create(harvest=harvest, species=row, percent=pct)
    data = digest(client.get('/api/prelievi/data/'))
    record = data['rows'][0]
    for name, expected in ((S.SPECIES_OTHER_CONIFER, 45),
                           (S.SPECIES_OTHER_BROADLEAF, 35), (S.SPECIES_OTHER, 20)):
        assert record[data['columns'].index(name)] == expected
        assert record[data['columns'].index(f'{name} %')] == expected
    assert list(HarvestSpecies.objects.filter(harvest=harvest).order_by('id')
                .values_list('species_id', flat=True)) == [larch.id, species[2].id, species[3].id]
    harvest.refresh_from_db()
    assert harvest.volume_m3 == Decimal('12.345')
    form = client.get(f'/api/prelievi/form/{harvest.id}/').json()['html']
    assert f'name="sp_{species[3].id}"' in form
    assert f'name="sp_{species[4].id}"' in form
    assert f'name="sp_{species[5].id}"' in form
    # Changing classification must regroup an already-materialized digest on
    # the next read, without changing captured junctions or harvest volume.
    response = post(client, '/api/impostazioni/species/type/save/', {
        'row_id': larch.id, 'version': larch.version, 'botanical_type': 'broadleaf',
    })
    assert response.status_code == 200, response.content
    updated = digest(client.get('/api/prelievi/data/'))
    assert updated['rows'][0][updated['columns'].index(S.SPECIES_OTHER_CONIFER)] == 0
    assert updated['rows'][0][updated['columns'].index(S.SPECIES_OTHER_BROADLEAF)] == 80
    assert updated['rows'][0][updated['columns'].index(S.SPECIES_OTHER)] == 20
    harvest.refresh_from_db()
    assert harvest.volume_m3 == Decimal('12.345')


def test_new_harvest_rejects_legacy_and_accepts_new_buckets(client, species, item, crews, products):
    form = client.get('/api/prelievi/form/').json()['html']
    assert f'name="sp_{species[3].id}"' not in form
    body = {'date': '2024-06-01', 'harvest_plan_item_id': item.id,
            'crew_id': crews[0].id, 'product_id': products[0].id, 'mass_q': '90',
            f'sp_{species[3].id}': 100}
    response = post(client, '/api/prelievi/save/', body)
    assert response.status_code == 400, response.content
    assert Harvest.objects.count() == 0
    del body[f'sp_{species[3].id}']
    body[f'sp_{species[4].id}'] = 60
    body[f'sp_{species[5].id}'] = 40
    response = post(client, '/api/prelievi/save/', body)
    assert response.status_code == 200, response.content


@pytest.mark.parametrize('index,status', [(3, 400), (4, 200), (5, 200)])
def test_new_marks_use_only_new_choices(client, species, item, index, status):
    form = client.get(f'/api/piano-di-taglio/mark/form/?item={item.id}').json()['html']
    assert f'value="{species[3].id}"' not in form
    assert f'value="{species[4].id}"' in form and f'value="{species[5].id}"' in form
    response = post(client, '/api/piano-di-taglio/mark/save/', {
        'harvest_plan_item_id': item.id, 'species_id': species[index].id,
        'date': '2024-06-01', 'd_cm': 30, 'h_m': '20.0', 'h_measured': True,
        'operator': 'Species test',
    })
    assert response.status_code == status, response.content
    assert TreeMark.objects.count() == (status == 200)
    if status == 200:
        mark = TreeMark.objects.get()
        expected = tabacchi_volume_m3(30, 20, species[index].common_name)
        assert mark.volume_m3 == expected
        assert mark.mass_q == tree_mass_q(expected, species[index].density)


def test_existing_legacy_mark_can_be_edited(client, species, item):
    tree = Tree.objects.create(species=species[3])
    mark = TreeMark.objects.create(harvest_plan_item=item, parcel=item.parcel,
                                   tree=tree, date='2024-06-01', d_cm=30, h_m='20.0')
    form = client.get(f'/api/piano-di-taglio/mark/form/{mark.id}/').json()['html']
    assert f'value="{species[3].id}"' in form
    response = post(client, '/api/piano-di-taglio/mark/save/', {
        'row_id': mark.id, 'version': mark.version,
        'harvest_plan_item_id': item.id, 'species_id': species[3].id,
        'date': '2024-06-01', 'd_cm': 32, 'h_m': '21.0', 'h_measured': True,
        'operator': 'Species test',
    })
    assert response.status_code == 200, response.content
    tree.refresh_from_db()
    assert tree.species_id == species[3].id


@pytest.mark.parametrize('index,status', [(3, 400), (4, 200), (5, 200)])
def test_new_sampled_trees_use_new_choices(client, species, parcels, index, status):
    survey = Survey.objects.create(name='Free species survey')
    form = client.get(f'/api/campionamenti/tree/form/?survey={survey.id}').json()['html']
    select = form.split('<select id="tf-species"', 1)[1].split('</select>', 1)[0]
    assert f'value="{species[3].id}"' not in select
    assert f'value="{species[4].id}"' in select and f'value="{species[5].id}"' in select
    response = post(client, '/api/campionamenti/tree/save/', {
        'survey_id': survey.id, 'parcel_id': parcels[0].id, 'species_id': species[index].id,
        'date': '2024-06-01', 'number': 1, 'd_cm': 30, 'h_m': '20.0', 'fustaia': True,
    })
    assert response.status_code == status, response.content
    assert Tree.objects.count() == (status == 200)
    if status == 200:
        row = TreeSample.objects.get()
        expected = tabacchi_volume_m3(30, 20, species[index].common_name)
        assert row.volume_m3 == expected
        assert row.mass_q == tree_mass_q(expected, species[index].density)


def test_existing_legacy_tree_can_be_resampled_without_exposing_new_choice(client, species, parcels):
    grid = SampleGrid.objects.create(name='Species grid')
    area = SampleArea.objects.create(sample_grid=grid, parcel=parcels[0], number='1', lat=0, lon=0)
    old_survey = Survey.objects.create(name='Old survey', sample_grid=grid)
    new_survey = Survey.objects.create(name='New survey', sample_grid=grid)
    sample = Sample.objects.create(survey=old_survey, sample_area=area, date='2024-01-01')
    tree = Tree.objects.create(species=species[3])
    TreeSample.objects.create(sample=sample, tree=tree, parcel=parcels[0],
                               number=1, shoot=0, d_cm=30, h_m='20.0')
    form = client.get(f'/api/campionamenti/tree/form/?survey={new_survey.id}&area={area.id}').json()['html']
    assert f'<option value="{species[3].id}" hidden disabled' in form
    response = post(client, '/api/campionamenti/tree/save/', {
        'survey_id': new_survey.id, 'sample_area_id': area.id,
        'tree_pick': tree.id, 'number': 1, 'date': '2024-06-01',
        'd_cm': 32, 'h_m': '21.0', 'fustaia': True,
    })
    assert response.status_code == 200, response.content
    assert Tree.objects.count() == 1
    assert TreeSample.objects.get(id=response.json()['row_id']).tree_id == tree.id


@override_settings(IPSO_SECRET='test-token')
def test_mobile_reference_retains_legacy_metadata_but_disables_selection(species):
    legacy = species[3]
    legacy.active = False
    legacy.save()
    data = Client().get('/ipso/reference.json',
                        HTTP_AUTHORIZATION='Bearer test-token').json()
    rows = {row['common']: row for row in data['species']}
    assert rows[S.SPECIES_OTHER]['selectable'] is False
    assert rows[S.SPECIES_OTHER_CONIFER]['selectable'] is True
    assert rows[S.SPECIES_OTHER_BROADLEAF]['selectable'] is True


@pytest.mark.parametrize('index,status', [(3, 400), (4, 200), (5, 200)])
def test_new_pai_trees_use_new_choices(client, species, parcels, index, status):
    form = client.get('/api/bosco/pai/form/').json()['html']
    assert f'value="{species[3].id}"' not in form
    assert f'value="{species[4].id}"' in form and f'value="{species[5].id}"' in form
    response = post(client, '/api/bosco/pai/save/', {
        'species_id': species[index].id, 'parcel_id': parcels[0].id,
        'number': 7, 'date': '2024-06-01', 'd_cm': 32, 'h_m': '21.0',
        'lat': '38.5', 'lon': '16.3',
    })
    assert response.status_code == status, response.content
    assert Tree.objects.count() == (status == 200)


def test_legacy_pai_tree_can_still_be_edited(client, species, parcels):
    survey = Survey.objects.create(name='Legacy PAI')
    sample = Sample.objects.create(survey=survey, date='2024-01-01')
    tree = Tree.objects.create(species=species[3])
    row = TreeSample.objects.create(sample=sample, tree=tree, parcel=parcels[0],
                                    number=7, preserved_number=7, shoot=0,
                                    d_cm=30, h_m='20.0', lat=38.5, lon=16.3)
    form = client.get(f'/api/bosco/pai/form/{row.id}/').json()['html']
    assert f'value="{species[3].id}"' in form
    response = post(client, '/api/bosco/pai/save/', {
        'row_id': row.id, 'version': row.version, 'species_id': species[3].id,
        'parcel_id': parcels[0].id, 'number': 7, 'date': '2024-06-01',
        'd_cm': 32, 'h_m': '21.0', 'lat': '38.5', 'lon': '16.3',
    })
    assert response.status_code == 200, response.content
    tree.refresh_from_db()
    assert tree.species_id == species[3].id


def test_old_reference_csv_infers_new_rows_but_preserves_edits():
    reader = csv_io.read('Genere,Nome latino\nAbete,Abies alba\n')
    cols, _ = csv_reference.resolve_columns(csv_reference.SPECIES, reader.fieldnames)
    rows, errors = csv_reference.validate_rows(csv_reference.SPECIES, reader, cols)
    assert not errors
    assert csv_reference.apply(csv_reference.SPECIES, rows) == (1, 0)
    fir = Species.objects.get(common_name='Abete')
    assert fir.botanical_type == SpeciesType.CONIFER
    fir.botanical_type = SpeciesType.BROADLEAF
    fir.save()
    assert csv_reference.apply(csv_reference.SPECIES, rows) == (0, 0)
    fir.refresh_from_db()
    assert fir.botanical_type == SpeciesType.BROADLEAF


def test_migration_preserves_ids_values_and_copies_current_defaults(parcels):
    fir = Species.objects.create(common_name='Abete', latin_name='Abies alba',
                                 density='8.25', pressler_default='1.25')
    beech = Species.objects.create(common_name='Faggio', latin_name='Fagus sylvatica',
                                   density='11.00', pressler_default='2.25')
    legacy = Species.objects.create(common_name=S.SPECIES_OTHER, density='9.00')
    tree = Tree.objects.create(species=legacy)
    unknown = Species.objects.create(common_name='Unknown custom')
    DigestStatus.objects.create(name='prelievi')
    state_apps = MigrationExecutor(connection).loader.project_state([
        ('base', '0026_species_botanical_type'),
    ]).apps
    migration = importlib.import_module('apps.base.migrations.0026_species_botanical_type')
    migration.classify_species(state_apps, SimpleNamespace(connection=connection))
    fir.refresh_from_db()
    beech.refresh_from_db()
    legacy.refresh_from_db()
    unknown.refresh_from_db()
    tree.refresh_from_db()
    assert tree.species_id == legacy.id
    assert legacy.botanical_type == unknown.botanical_type == ''
    assert fir.botanical_type == SpeciesType.CONIFER
    assert beech.botanical_type == SpeciesType.BROADLEAF
    assert fir.history.first().history_change_reason == 'initial-audit-baseline'
    assert fir.version == 2
    for name, source in ((S.SPECIES_OTHER_CONIFER, fir), (S.SPECIES_OTHER_BROADLEAF, beech)):
        added = Species.objects.get(common_name=name)
        assert (added.density, added.pressler_default) == (source.density, source.pressler_default)
        assert added.history.count() == 1
    assert DigestStatus.objects.get(name='prelievi').stale


def test_migration_leaves_empty_species_table_for_bootstrap():
    migration = importlib.import_module('apps.base.migrations.0026_species_botanical_type')
    migration.classify_species(apps, SimpleNamespace(connection=connection))
    assert not Species.objects.exists()
