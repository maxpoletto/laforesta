"""Tests for the `bootstrap` management command (apps/base/.../bootstrap)."""

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.base.models import (
    Crew, Eclass, HarvestPlan, Parcel, Region, SampleArea, SampleGrid,
    Species, Survey, Tree,
)
from apps.base.hypsometry import active_set

# A small, internally consistent canonical data dir.  Each value is the file's
# full text; tests write these to a tmp dir and may override one file (None to omit).
CANONICAL = {
    'regions.csv': 'Compresa\nCapistrano\n',
    'eclasses.csv': 'Comparto,Ceduo\nA,0\n',
    'crews.csv': 'Squadra\nAlfa\n',
    'species.csv': 'Genere,Densità (q/m³)\nAbete,9.0\n',
    'products.csv': 'Tipo\nTronchi\n',
    'particelle.csv': 'Compresa,Comparto,Particella,Area (ha)\nCapistrano,A,1,10.5\n',
    'sample_grids.csv': 'Griglia\nGriglia 1\n',
    'harvest_plans.csv': 'Piano,Anno inizio,Anno fine\nPDG 2026,2026,2040\n',
    'surveys.csv': 'Rilevamento,Griglia,Data\nCampagna 2024,Griglia 1,2024-09-15\n',
    'sample_areas.csv': (
        'Griglia,Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        'Griglia 1,Capistrano,1,10,16.1,38.5,500,15\n'),
    'sampled-trees.csv': (
        'Rilevamento,Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
        'D_cm,H_m,L10_mm,Genere,Fustaia\n'
        'Campagna 2024,Capistrano,1,10,1,0,False,30,15.0,0,Abete,True\n'),
}


def _make_dir(tmp_path, **overrides):
    files = dict(CANONICAL, **overrides)
    for name, text in files.items():
        if text is not None:
            (tmp_path / name).write_text(text, encoding='utf-8')
    return str(tmp_path)


@pytest.mark.django_db
def test_happy_path_loads_everything(tmp_path):
    call_command('bootstrap', _make_dir(tmp_path))
    assert Region.objects.count() == 1
    assert Eclass.objects.count() == 1
    assert Crew.objects.count() == 1
    assert Species.objects.filter(common_name='Abete').exists()
    assert Parcel.objects.filter(name='1', region__name='Capistrano').exists()
    assert SampleGrid.objects.filter(name='Griglia 1').exists()
    assert HarvestPlan.objects.filter(name='PDG 2026').exists()
    assert Survey.objects.filter(name='Campagna 2024').exists()
    assert SampleArea.objects.count() == 1
    assert Tree.objects.count() == 1


@pytest.mark.django_db
def test_check_persists_nothing(tmp_path):
    call_command('bootstrap', _make_dir(tmp_path), '--check')
    assert Region.objects.count() == 0
    assert Parcel.objects.count() == 0
    assert Tree.objects.count() == 0


@pytest.mark.django_db
def test_non_empty_instance_refused(tmp_path):
    Region.objects.create(name='Existing')
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(tmp_path))
    assert Region.objects.filter(name='Existing').exists()   # untouched


@pytest.mark.django_db
def test_non_empty_unsupported_domain_refused(tmp_path):
    """A populated *unsupported* domain (a Tractor) still blocks bootstrap."""
    from apps.base.models import Tractor
    Tractor.objects.create(manufacturer='Fiat', model='110-90')
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(tmp_path))


@pytest.mark.django_db
def test_missing_required_file_errors(tmp_path):
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(tmp_path, **{'regions.csv': None}))
    assert Region.objects.count() == 0          # rolled back


@pytest.mark.django_db
def test_error_in_one_file_rolls_back_all(tmp_path):
    bad_parcels = 'Compresa,Comparto,Particella,Area (ha)\nCapistrano,Z,1,10.5\n'
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(tmp_path, **{'particelle.csv': bad_parcels}))
    assert Region.objects.count() == 0          # nothing persisted
    assert Parcel.objects.count() == 0


@pytest.mark.django_db
def test_optional_files_absent_skipped(tmp_path):
    only_required = {k: (v if k in ('regions.csv', 'eclasses.csv', 'particelle.csv')
                         else None) for k, v in CANONICAL.items()}
    call_command('bootstrap', _make_dir(tmp_path, **only_required))
    assert Parcel.objects.count() == 1
    assert SampleGrid.objects.count() == 0
    assert Survey.objects.count() == 0


@pytest.mark.django_db
def test_sample_areas_unknown_grid_errors(tmp_path):
    bad_areas = (
        'Griglia,Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        'Nessuna griglia,Capistrano,1,10,16.1,38.5,500,15\n')
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(tmp_path, **{'sample_areas.csv': bad_areas}))
    assert SampleArea.objects.count() == 0


@pytest.mark.django_db
def test_hypso_loaded_when_present(tmp_path):
    hypso = 'Compresa,Genere,funzione,a,b,r2,n\nCapistrano,Abete,ln,7.0,-4.0,0.90,12\n'
    call_command('bootstrap', _make_dir(tmp_path, **{'hypso_params.csv': hypso}))
    s = active_set()
    assert s is not None
    assert s.params.count() == 1


@pytest.mark.django_db
def test_deferred_file_noted_not_loaded(tmp_path, capsys):
    call_command('bootstrap', _make_dir(tmp_path, **{'harvests.csv': 'Compresa\nCapistrano\n'}))
    out = capsys.readouterr().out
    assert 'harvests.csv' in out
    assert Region.objects.filter(name='Capistrano').exists()   # rest loaded
