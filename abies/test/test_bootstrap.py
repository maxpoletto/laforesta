"""Tests for the `bootstrap` management command (apps/base/.../bootstrap)."""

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import OperationalError

from apps.base.models import (
    Crew, Eclass, HarvestPlan, HarvestPlanItem, Parcel, Product, Region,
    SampleArea, SampleGrid, Species, Survey, Tractor, Tree, TreeSample,
)
from apps.base.hypsometry import active_set
from apps.base.refdata import PRODUCT_MAP, load_species
from apps.prelievi.models import Harvest
from config import strings as S

# A small, internally consistent canonical data dir.  Each value is the file's
# full text; tests write these to a tmp dir and may override one file (None to omit).
CANONICAL = {
    'regions.csv': 'Compresa\nCapistrano\n',
    'eclasses.csv': 'Comparto,Ceduo\nA,0\n',
    'crews.csv': 'Squadra\nAlfa\n',
    'tractors.csv': 'Trattore,Produttore,Modello,Anno\nT1,Fiat,110-90,1990\n',
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
        'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia\n'
        'Campagna 2024,Capistrano,1,10,1,0,False,30,15.0,0,2,Abete,True\n'),
    'harvest_plan_items.csv': (
        'Piano,Compresa,Particella,Anno,Prelievo (m³),Danneggiato,Fitosanitario,PSR\n'
        'PDG 2026,Capistrano,1,2027,50.0,0,0,0\n'),
    'preserved-trees.csv': (
        'Compresa,Particella,Numero,Genere,Data,Anno di nascita stimato,'
        'D_cm,H_m,H_measured,Lon,Lat,Acc_m,Operatore,Note\n'
        'Capistrano,1,1,Abete,2024-09-15,1920,40,18.5,1,16.1,38.5,,Rossi,nota\n'),
    'harvests.csv': (
        'Compresa,Particella,Data,Squadra,Tipo,Q.li,Specie: Abete\n'
        'Capistrano,1,2027-03-01,Alfa,Tronchi,10.0,100\n'
        'Capistrano,,2027-03-02,Alfa,Tronchi,5.0,100\n'),
}


def _report_line(output, filename):
    return next(line for line in output.splitlines() if filename in line)


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
    assert Tractor.objects.filter(name='T1').exists()
    assert Species.objects.filter(common_name='Abete').exists()
    assert Parcel.objects.filter(name='1', region__name='Capistrano').exists()
    assert SampleGrid.objects.filter(name='Griglia 1').exists()
    assert HarvestPlan.objects.filter(name='PDG 2026').exists()
    assert HarvestPlanItem.objects.count() == 1
    assert Survey.objects.filter(name='Campagna 2024').exists()
    assert SampleArea.objects.count() == 1
    assert Tree.objects.filter(preserved=False).count() == 1   # sampled tree
    assert Tree.objects.filter(preserved=True).count() == 1    # preserved tree
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 1
    assert Harvest.objects.count() == 2


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
def test_missing_schema_reports_actionable_error(tmp_path, monkeypatch):
    def broken_exists():
        raise OperationalError('no such table: base_product')

    monkeypatch.setattr(Product.objects, 'exists', broken_exists)
    with pytest.raises(CommandError, match='Schema database'):
        call_command('bootstrap', _make_dir(tmp_path))


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
def test_file_with_row_errors_reports_zero_loaded(tmp_path, capsys):
    optional_absent = {
        k: None for k in CANONICAL
        if k not in ('regions.csv', 'eclasses.csv', 'particelle.csv')
    }
    bad_parcels = (
        'Compresa,Comparto,Particella,Area (ha)\n'
        'Capistrano,A,1,10.5\n'
        'Capistrano,Z,2,11.5\n'
    )
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(
            tmp_path, **optional_absent, **{'particelle.csv': bad_parcels},
        ))
    line = _report_line(capsys.readouterr().out, 'particelle.csv')
    assert S.BOOTSTRAP_LOADED.format(0) in line


@pytest.mark.django_db
def test_grouped_file_with_row_errors_reports_zero_loaded(tmp_path, capsys):
    keep = ('regions.csv', 'eclasses.csv', 'particelle.csv',
            'sample_grids.csv', 'sample_areas.csv')
    optional_absent = {k: None for k in CANONICAL if k not in keep}
    bad_areas = (
        'Griglia,Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        'Griglia 1,Capistrano,1,10,16.1,38.5,500,15\n'
        'Nessuna griglia,Capistrano,1,11,16.2,38.6,510,15\n'
    )
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(
            tmp_path, **optional_absent, **{'sample_areas.csv': bad_areas},
        ))
    line = _report_line(capsys.readouterr().out, 'sample_areas.csv')
    assert S.BOOTSTRAP_LOADED.format(0) in line


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
def test_tractors_load_via_reference_loop(tmp_path):
    """tractors.csv loads via the reference table loop (TRACTORS in ALL_TABLES)."""
    call_command('bootstrap', _make_dir(tmp_path))
    assert Tractor.objects.filter(name='T1').exists()


@pytest.mark.django_db
def test_harvest_plan_items_load(tmp_path):
    """harvest_plan_items.csv wires in and creates HarvestPlanItem rows."""
    call_command('bootstrap', _make_dir(tmp_path))
    assert HarvestPlanItem.objects.count() == 1


@pytest.mark.django_db
def test_preserved_trees_load(tmp_path):
    """preserved-trees.csv wires in and creates preserved Tree rows."""
    call_command('bootstrap', _make_dir(tmp_path))
    trees = Tree.objects.filter(preserved=True)
    assert trees.count() == 1
    pai = TreeSample.objects.get(preserved_number__isnull=False)
    assert pai.tree == trees.get()
    assert pai.preserved_number == 1
    assert pai.number == 1
    assert pai.sample.date.isoformat() == '2024-09-15'
    assert pai.d_cm == 40
    assert str(pai.h_m) == '18.50'
    assert pai.note == 'nota'


@pytest.mark.django_db
def test_harvests_load(tmp_path):
    """harvests.csv wires in and creates Harvest rows."""
    call_command('bootstrap', _make_dir(tmp_path))
    assert Harvest.objects.count() == 2


@pytest.mark.django_db
def test_region_wide_harvest_persists_parcel_none(tmp_path):
    """A harvest row with blank Particella persists with parcel=None, region set."""
    call_command('bootstrap', _make_dir(tmp_path))
    region_wide = Harvest.objects.filter(parcel__isnull=True)
    assert region_wide.count() == 1
    assert region_wide.first().region is not None
    assert region_wide.first().region.name == 'Capistrano'


@pytest.mark.django_db
def test_harvest_bad_parcel_region_mismatch_rejected(tmp_path):
    """A harvest row with a parcel that does not exist in the given region is rejected."""
    bad_harvests = (
        'Compresa,Particella,Data,Squadra,Tipo,Q.li,Specie: Abete\n'
        'Capistrano,99,2027-03-01,Alfa,Tronchi,10.0,100\n'
    )
    with pytest.raises(CommandError):
        call_command('bootstrap', _make_dir(tmp_path, **{'harvests.csv': bad_harvests}))
    assert Harvest.objects.count() == 0


@pytest.mark.django_db
def test_species_products_seeded_from_in_repo_default_when_absent(tmp_path, capsys):
    """Omitting species.csv/products.csv seeds the in-repo canonical defaults
    rather than skipping (spec §3)."""
    call_command('bootstrap', _make_dir(
        tmp_path, **{'species.csv': None, 'products.csv': None}))
    assert Species.objects.count() == len(load_species())
    assert Species.objects.filter(common_name='Abete').exists()
    assert Species.objects.get(common_name='Pino Nero').minor is False
    assert Product.objects.count() == len(set(PRODUCT_MAP.values()))
    # The default species set includes 'Abete', so the sampled tree still loads.
    assert Tree.objects.filter(preserved=False).count() == 1   # sampled tree
    # The report flags that defaults were seeded (not silently skipped).
    assert S.BOOTSTRAP_DEFAULT_SEEDED in capsys.readouterr().out


@pytest.mark.django_db
def test_present_but_empty_species_not_defaulted(tmp_path):
    """A present-but-empty species.csv is authoritative: the in-repo default is
    seeded only when the file is *absent*, not when it is present with no rows."""
    only = {k: (v if k in ('regions.csv', 'eclasses.csv', 'particelle.csv')
                else None) for k, v in CANONICAL.items()}
    only['species.csv'] = 'Genere,Densità (q/m³)\n'   # header only, no data rows
    call_command('bootstrap', _make_dir(tmp_path, **only))
    assert Species.objects.count() == 0
