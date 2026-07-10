"""Unit tests for the harvests CSV import core (apps/prelievi/csv_harvests)."""

import pytest

from apps.base import csv_io
from apps.base.models import Crew, Product, Tractor
from apps.prelievi import csv_harvests
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_header(*extra_cols):
    """Build a harvest CSV header with optional dynamic columns appended."""
    static = (
        f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_DATA},'
        f'{S.CSV_COL_CREW},{S.CSV_COL_PRODUCT},{S.CSV_COL_QUINTALS},'
        f'{S.CSV_COL_VDP},{S.CSV_COL_PROT},'
        f'{S.CSV_COL_HARVEST_DAMAGED},{S.CSV_COL_HARVEST_UNHEALTHY},'
        f'{S.CSV_COL_HARVEST_PSR},{S.CSV_COL_EXTRA_NOTE}'
    )
    if extra_cols:
        return static + ',' + ','.join(extra_cols)
    return static


def _row(region='Capistrano', parcel='1', date='2024-01-10',
         crew='Alfa', product='Tronchi', quintals='100',
         vdp='5', prot='', damaged='false', unhealthy='false',
         psr='false', extra_note='', extra_vals=()):
    static = f'{region},{parcel},{date},{crew},{product},{quintals},{vdp},{prot},{damaged},{unhealthy},{psr},{extra_note}'
    if extra_vals:
        return static + ',' + ','.join(str(v) for v in extra_vals)
    return static


@pytest.fixture
def base_setup(db):
    """Crew, Product, and one Tractor with a name — the minimal happy-path setup."""
    crew = Crew.objects.create(name='Alfa')
    product = Product.objects.create(name='Tronchi')
    tractor = Tractor.objects.create(manufacturer='Fiat', model='110-90', name='Fiat 110-90')
    return {'crew': crew, 'product': product, 'tractor': tractor}


# ---------------------------------------------------------------------------
# resolve_columns
# ---------------------------------------------------------------------------

def test_resolve_columns_all_static(species, db):
    header = _make_header()
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    assert missing == []
    assert dyn == []
    assert S.CSV_COL_REGION in cols
    assert S.CSV_COL_PARCEL in cols
    assert S.CSV_COL_DATA in cols


def test_resolve_columns_missing_required(db):
    # Omit Q.li (CSV_COL_QUINTALS)
    header = (
        f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_DATA},'
        f'{S.CSV_COL_CREW},{S.CSV_COL_PRODUCT}'
    )
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    assert S.CSV_COL_QUINTALS in missing


def test_resolve_columns_species_dynamic(species, db):
    header = _make_header('Specie: Abete', 'Specie: Castagno')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    assert missing == []
    kinds = [(kind, key) for kind, key, _hdr in dyn]
    assert ('species', 'Abete') in kinds
    assert ('species', 'Castagno') in kinds


def test_resolve_columns_tractor_dynamic(base_setup, db):
    header = _make_header('Trattore: Fiat 110-90')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    assert missing == []
    kinds = [(kind, key) for kind, key, _hdr in dyn]
    assert ('tractor', 'Fiat 110-90') in kinds


def test_resolve_columns_strips_whitespace(db):
    """Key is trimmed: 'Specie:  Abete ' → key='Abete'."""
    header = _make_header('Specie:  Abete ')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    keys = [key for _kind, key, _hdr in dyn]
    assert 'Abete' in keys


# ---------------------------------------------------------------------------
# validate_rows: unknown / duplicate dynamic columns
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_unknown_species_col(parcels, species, base_setup):
    header = _make_header('Specie: Quercia')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert any('Quercia' in e for e in errors)


@pytest.mark.django_db
def test_unknown_tractor_col(parcels, species, base_setup):
    header = _make_header('Trattore: Trattore Inesistente')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert any('Trattore Inesistente' in e for e in errors)


@pytest.mark.django_db
def test_duplicate_species_col(parcels, species, base_setup):
    """Two headers with the same trimmed key → duplicate error."""
    header = _make_header('Specie: Abete', 'Specie:  Abete ')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert any('Abete' in e for e in errors)


@pytest.mark.django_db
def test_duplicate_tractor_col(parcels, species, base_setup):
    """Two tractor headers with same key → duplicate error."""
    tname = base_setup['tractor'].name
    header = _make_header(f'Trattore: {tname}', f'Trattore:  {tname} ')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert any(tname in e for e in errors)


@pytest.mark.django_db
def test_bare_species_prefix_column_errors(parcels, species, base_setup):
    """A header 'Specie:' with empty key (no name after colon) produces an error."""
    header = _make_header('Specie:')
    reader = csv_io.read(header + '\n')
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors


# ---------------------------------------------------------------------------
# validate_rows: location resolution
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_harvest_parcel_row(parcels, species, base_setup):
    """A row with known Compresa+Particella resolves parcel, no errors."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='50', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(
        reader, cols, dyn, csv_harvests.db_indexes(),
    )
    assert errors == []
    assert parsed[0]['parcel'] == p
    assert parsed[0]['region'] is None


@pytest.mark.django_db
def test_harvest_region_wide_row(parcels, species, base_setup):
    """Blank Particella → region-wide row (parcel=None, region set)."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel='',
        quintals='50', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    assert parsed[0]['parcel'] is None
    assert parsed[0]['region'] == p.region


@pytest.mark.django_db
def test_unknown_region_error(parcels, species, base_setup):
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(region='RegioneSconosciuta', parcel='', extra_vals=('100',))
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_unknown_parcel_error(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(region=p.region.name, parcel='999', extra_vals=('100',))
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_blank_compresa_error(parcels, species, base_setup):
    """Blank Compresa (not just blank Particella) is an error."""
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(region='', parcel='', extra_vals=('100',))
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


# ---------------------------------------------------------------------------
# validate_rows: unknown crew / product
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_unknown_crew_error(parcels, species, base_setup):
    """Unknown crew name produces ERR_CSV_UNKNOWN_CREW with the offending value."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        crew='SquadraSconosciuta', quintals='100', extra_vals=('100',),
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []
    assert any(S.ERR_CSV_UNKNOWN_CREW.format(2, 'SquadraSconosciuta') == e for e in errors)


@pytest.mark.django_db
def test_unknown_product_error(parcels, species, base_setup):
    """Unknown product name produces ERR_CSV_UNKNOWN_PRODUCT with the offending value."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        product='TipoProdottoSconosciuto', quintals='100', extra_vals=('100',),
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []
    assert any(S.ERR_CSV_UNKNOWN_PRODUCT.format(2, 'TipoProdottoSconosciuto') == e for e in errors)


# ---------------------------------------------------------------------------
# validate_rows: species / tractor pct validation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_species_pct_sum_not_100_when_mass_positive(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete', 'Specie: Castagno')
    # pcts sum to 60, mass=100 -> error
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='100', extra_vals=('40', '20')
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_species_pct_range_checked_even_when_sum_100(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete', 'Specie: Castagno')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='100', extra_vals=('150', '-50')
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_negative_quintals_rejected(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='-1', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_species_pct_sum_ok_when_mass_zero(parcels, species, base_setup):
    """When Q.li=0, species %-sum is not checked."""
    p = parcels[0]
    header = _make_header('Specie: Abete', 'Specie: Castagno')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='0', extra_vals=('40', '20')
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []


@pytest.mark.django_db
def test_tractor_pct_partial_sum_error(parcels, species, base_setup):
    """Tractor pct sum of 50 (not 0 or 100) → error."""
    p = parcels[0]
    tname = base_setup['tractor'].name
    header = _make_header('Specie: Abete', f'Trattore: {tname}')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='100', extra_vals=('100', '50')
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_tractor_pct_all_blank_ok(parcels, species, base_setup):
    """Tractor pct all blank → sum=0, which is valid."""
    p = parcels[0]
    tname = base_setup['tractor'].name
    header = _make_header('Specie: Abete', f'Trattore: {tname}')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='100', extra_vals=('100', '')
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []


@pytest.mark.django_db
def test_tractor_pct_100_ok(parcels, species, base_setup):
    p = parcels[0]
    tname = base_setup['tractor'].name
    header = _make_header('Specie: Abete', f'Trattore: {tname}')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='100', extra_vals=('100', '100')
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []


# ---------------------------------------------------------------------------
# validate_rows: volume computation
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_volume_computed_positive(parcels, species, base_setup):
    """volume_m3 > 0 when mass_q > 0 and species pct = 100."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='100', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    assert parsed[0]['volume_m3'] > 0


@pytest.mark.django_db
def test_volume_zero_when_mass_zero(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        quintals='0', extra_vals=('0',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    assert parsed[0]['volume_m3'] == 0


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_harvest_happy_path(parcels, species, base_setup):
    """Full happy-path: parse + apply creates Harvest + child rows."""
    p = parcels[0]
    text = (
        'Compresa,Particella,Data,Squadra,Tipo,Q.li,VDP,Prot.,'
        'Danneggiato,Fitosanitario,PSR,Altre note,Specie: Abete,Trattore: Fiat 110-90\n'
        f'{p.region.name},{p.name},2024-01-10,Alfa,Tronchi,100,5,,'
        'false,false,false,,100,100\n'
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    assert not missing
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == [] and len(parsed) == 1
    assert parsed[0]['volume_m3'] > 0
    n = csv_harvests.apply(parsed)
    assert n == 1
    h = Harvest.objects.get()
    assert h.parcel == p
    assert h.region is None
    assert h.record1 == 5
    assert h.record2 is None
    assert h.mass_q == 100
    assert h.volume_m3 > 0
    assert h.import_fingerprint.startswith('v1:')
    assert not h.damaged
    assert not h.unhealthy
    assert not h.psr
    assert HarvestSpecies.objects.filter(harvest=h).count() == 1
    assert HarvestTractor.objects.filter(harvest=h).count() == 1


@pytest.mark.django_db
def test_apply_region_wide(parcels, species, base_setup):
    """Region-wide row: harvest has parcel=None, region set."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel='',
        quintals='50', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    n = csv_harvests.apply(parsed)
    assert n == 1
    h = Harvest.objects.get()
    assert h.parcel is None
    assert h.region == p.region


@pytest.mark.django_db
def test_apply_multiple_rows(parcels, species, base_setup):
    """Multiple rows → correct harvest count."""
    p0, p1 = parcels[0], parcels[1]
    header = _make_header('Specie: Abete')
    text = (
        header + '\n'
        + _row(region=p0.region.name, parcel=p0.name, quintals='10', extra_vals=('100',))
        + '\n'
        + _row(region=p1.region.name, parcel=p1.name, quintals='20', extra_vals=('100',))
        + '\n'
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    assert csv_harvests.apply(parsed) == 2
    assert Harvest.objects.count() == 2


@pytest.mark.django_db
def test_apply_is_idempotent_for_same_csv_retry(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name, quintals='10', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []

    assert csv_harvests.apply(parsed) == 1
    assert csv_harvests.apply(parsed) == 0

    assert Harvest.objects.count() == 1
    h = Harvest.objects.get()
    assert HarvestSpecies.objects.filter(harvest=h).count() == 1


@pytest.mark.django_db
def test_apply_keeps_identical_rows_in_same_csv(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header('Specie: Abete')
    row = _row(region=p.region.name, parcel=p.name, quintals='10', extra_vals=('100',))
    text = f'{header}\n{row}\n{row}\n'
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(
        reader, cols, dyn, csv_harvests.db_indexes(),
    )
    assert errors == []

    assert csv_harvests.apply(parsed) == 2

    assert Harvest.objects.count() == 2
    fingerprints = set(Harvest.objects.values_list('import_fingerprint', flat=True))
    assert len(fingerprints) == 2


@pytest.mark.django_db
def test_apply_boolean_flags(parcels, species, base_setup):
    """damaged=true, psr=true are stored correctly."""
    p = parcels[0]
    header = _make_header('Specie: Abete')
    text = header + '\n' + _row(
        region=p.region.name, parcel=p.name,
        damaged='true', psr='true', quintals='100', extra_vals=('100',)
    )
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    csv_harvests.apply(parsed)
    h = Harvest.objects.get()
    assert h.damaged is True
    assert h.unhealthy is False
    assert h.psr is True


@pytest.mark.django_db
def test_positive_mass_requires_species_breakdown(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header()
    text = header + '\n' + _row(region=p.region.name, parcel=p.name, quintals='100')
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors and parsed == []


@pytest.mark.django_db
def test_zero_mass_without_species_breakdown_ok(parcels, species, base_setup):
    p = parcels[0]
    header = _make_header()
    text = header + '\n' + _row(region=p.region.name, parcel=p.name, quintals='0')
    reader = csv_io.read(text)
    cols, dyn, missing = csv_harvests.resolve_columns(reader.fieldnames)
    parsed, errors = csv_harvests.validate_rows(reader, cols, dyn, csv_harvests.db_indexes())
    assert errors == []
    csv_harvests.apply(parsed)
    assert HarvestSpecies.objects.count() == 0
