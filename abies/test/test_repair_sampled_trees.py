"""Tests for the one-off sampled-tree repair command."""

from decimal import Decimal

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.base.models import (
    Eclass, HarvestPlan, HarvestPlanItem, Parcel, Region, Sample, SampleArea,
    SampleGrid, Species, Survey, Tree, TreeMark, TreeSample,
)


def _write_canonical(tmp_path, sampled_trees: str):
    (tmp_path / 'surveys.csv').write_text(
        'Rilevamento,Griglia,Data\n'
        'Campagna 2024,Griglia 1,2024-09-15\n',
        encoding='utf-8',
    )
    (tmp_path / 'sampled-trees.csv').write_text(sampled_trees, encoding='utf-8')
    return tmp_path


def _setup_domain():
    region = Region.objects.create(name='Capistrano')
    eclass = Eclass.objects.create(name='A', coppice=False)
    parcel = Parcel.objects.create(
        region=region, eclass=eclass, name='1', area_ha=Decimal('10.5'),
    )
    species = Species.objects.create(common_name='Abete', density=Decimal('9.0'))
    grid = SampleGrid.objects.create(name='Griglia 1')
    survey = Survey.objects.create(name='Campagna 2024', sample_grid=grid)
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='10',
        lon=16.1, lat=38.5, r_m=15,
    )
    sample = Sample.objects.create(
        sample_area=area, survey=survey, date='2024-09-15',
    )
    sampled_tree = Tree.objects.create(species=species)
    TreeSample.objects.create(
        sample=sample, tree=sampled_tree, parcel=parcel, number=1, shoot=0,
        d_cm=30, h_m=Decimal('15.0'),
    )
    preserved_tree = Tree.objects.create(
        species=species, coppice=False,
    )
    pai_survey = Survey.objects.create(name='PAI')
    pai_sample = Sample.objects.create(
        sample_area=None, survey=pai_survey, date='2024-09-15',
    )
    TreeSample.objects.create(
        sample=pai_sample, tree=preserved_tree, parcel=parcel,
        number=1, preserved_number=1, d_cm=40, h_m=Decimal('18.5'),
        h_measured=True, lat=38.5, lon=16.1,
    )
    return {
        'sampled_tree_id': sampled_tree.id,
        'preserved_tree_id': preserved_tree.id,
        'parcel_id': parcel.id,
    }


@pytest.mark.django_db
def test_repair_sampled_trees_check_persists_nothing(tmp_path):
    ids = _setup_domain()
    data_dir = _write_canonical(tmp_path, _replacement_csv())

    call_command('repair_sampled_trees', data_dir, '--check')

    assert TreeSample.objects.count() == 2
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 1
    assert Tree.objects.filter(id=ids['sampled_tree_id']).exists()
    assert Tree.objects.filter(id=ids['preserved_tree_id']).exists()


@pytest.mark.django_db
def test_repair_sampled_trees_refuses_sampled_tree_referenced_by_mark(tmp_path):
    ids = _setup_domain()
    sampled_tree = Tree.objects.get(id=ids['sampled_tree_id'])
    parcel = Parcel.objects.get(id=ids['parcel_id'])
    plan = HarvestPlan.objects.create(
        name='PDG 2026', year_start=2026, year_end=2040,
    )
    item = HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcel, year_planned=2026,
    )
    TreeMark.objects.create(
        harvest_plan_item=item, tree=sampled_tree, parcel=parcel,
        number=1, date='2026-01-01', d_cm=30, h_m=Decimal('18.0'),
        operator='Rossi',
    )
    data_dir = _write_canonical(tmp_path, _replacement_csv())

    with pytest.raises(CommandError, match='referenced outside samples'):
        call_command('repair_sampled_trees', data_dir)

    assert TreeSample.objects.count() == 2
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 1
    assert Tree.objects.filter(id=ids['sampled_tree_id']).exists()


@pytest.mark.django_db
def test_repair_sampled_trees_refuses_sampled_tree_referenced_by_pai_sample(tmp_path):
    ids = _setup_domain()
    sampled_tree = Tree.objects.get(id=ids['sampled_tree_id'])
    parcel = Parcel.objects.get(id=ids['parcel_id'])
    pai_survey = Survey.objects.create(name='PAI shared tree')
    pai_sample = Sample.objects.create(
        sample_area=None, survey=pai_survey, date='2024-10-01',
    )
    TreeSample.objects.create(
        sample=pai_sample, tree=sampled_tree, parcel=parcel,
        number=99, preserved_number=99, d_cm=35, h_m=Decimal('16.0'),
        h_measured=True,
    )
    data_dir = _write_canonical(tmp_path, _replacement_csv())

    with pytest.raises(CommandError, match='referenced outside samples'):
        call_command('repair_sampled_trees', data_dir)

    assert Tree.objects.filter(id=ids['sampled_tree_id']).exists()


@pytest.mark.django_db
def test_repair_sampled_trees_replaces_sample_rows_only(tmp_path):
    ids = _setup_domain()
    data_dir = _write_canonical(tmp_path, _replacement_csv())

    call_command('repair_sampled_trees', data_dir)

    assert Sample.objects.count() == 2
    assert TreeSample.objects.count() == 3
    assert not Tree.objects.filter(id=ids['sampled_tree_id']).exists()
    assert Tree.objects.filter(id=ids['preserved_tree_id']).exists()
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 1
    assert list(
        TreeSample.objects
        .filter(preserved_number__isnull=True)
        .order_by('number')
        .values_list('number', 'd_cm')
    ) == [(1, 31), (2, 32)]


def _replacement_csv():
    return (
        'Rilevamento,Compresa,Particella,Area saggio,Albero,Pollone,'
        'Matricina,D_cm,H_m,L10_mm,Pressler,Genere,Fustaia\n'
        'Campagna 2024,Capistrano,1,10,1,0,False,31,16.0,0,2,Abete,True\n'
        'Campagna 2024,Capistrano,1,10,2,0,False,32,17.0,0,2,Abete,True\n'
    )
