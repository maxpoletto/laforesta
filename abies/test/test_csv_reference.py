"""Unit tests for the reference-table CSV import cores (apps/base/csv_reference)."""

from decimal import Decimal

import pytest

from apps.base import csv_io
from apps.base import csv_reference as ref
from apps.base.models import Crew, Eclass, Product, Region, Species, Tractor
from config import strings as S
from config.constants import PRESSLER_DEFAULT
from config.constants import FIELD_ACTIVE, FIELD_COPPICE, FIELD_NAME, FIELD_NOTES


def _reader(text):
    return csv_io.read(text)


# --- resolve_columns -------------------------------------------------------

def test_resolve_columns_all_present():
    fn = [S.CSV_COL_CLASS, S.CSV_COL_COPPICE, S.CSV_COL_MIN_VOLUME]
    found, missing = ref.resolve_columns(ref.ECLASSES, fn)
    assert missing == []
    assert found[FIELD_NAME] == S.CSV_COL_CLASS
    assert found[FIELD_COPPICE] == S.CSV_COL_COPPICE


def test_resolve_columns_missing_required_reported():
    _, missing = ref.resolve_columns(ref.ECLASSES, [S.CSV_COL_CLASS])
    assert S.CSV_COL_COPPICE in missing          # required
    assert S.CSV_COL_MIN_VOLUME not in missing   # optional


def test_resolve_columns_optional_absent_ok():
    found, missing = ref.resolve_columns(ref.CREWS, [S.CSV_COL_CREW])
    assert missing == []
    assert FIELD_NOTES not in found
    assert FIELD_ACTIVE not in found


# --- regions ---------------------------------------------------------------

@pytest.mark.django_db
def test_regions_validate_and_apply():
    reader = _reader(f'{S.CSV_COL_REGION}\nCapistrano\nFabrizia\n')
    cols, missing = ref.resolve_columns(ref.REGIONS, reader.fieldnames)
    assert not missing
    parsed, errors = ref.validate_rows(ref.REGIONS, reader, cols)
    assert errors == []
    assert ref.apply(ref.REGIONS, parsed) == (2, 0)
    assert set(Region.objects.values_list('name', flat=True)) == {
        'Capistrano', 'Fabrizia'}


# --- eclasses --------------------------------------------------------------

@pytest.mark.django_db
def test_eclasses_coppice_bool_and_optional_volume():
    reader = _reader(
        f'{S.CSV_COL_CLASS},{S.CSV_COL_COPPICE},{S.CSV_COL_MIN_VOLUME}\n'
        'A,0,\n'
        'F,1,50\n'
    )
    cols, _ = ref.resolve_columns(ref.ECLASSES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.ECLASSES, reader, cols)
    assert errors == []
    ref.apply(ref.ECLASSES, parsed)
    a = Eclass.objects.get(name='A')
    f = Eclass.objects.get(name='F')
    assert a.coppice is False and a.min_harvest_volume == 0   # blank → default 0
    assert f.coppice is True and f.min_harvest_volume == 50


@pytest.mark.django_db
def test_eclasses_blank_required_coppice_flagged():
    reader = _reader(f'{S.CSV_COL_CLASS},{S.CSV_COL_COPPICE}\nA,\n')
    cols, _ = ref.resolve_columns(ref.ECLASSES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.ECLASSES, reader, cols)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_eclasses_unrecognised_coppice_flagged():
    """A non-blank but unrecognised boolean is an error, not a silent False."""
    reader = _reader(f'{S.CSV_COL_CLASS},{S.CSV_COL_COPPICE}\nA,maybe\n')
    cols, _ = ref.resolve_columns(ref.ECLASSES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.ECLASSES, reader, cols)
    assert parsed == []
    assert len(errors) == 1


# --- crews -----------------------------------------------------------------

@pytest.mark.django_db
def test_crews_defaults_applied():
    reader = _reader(f'{S.CSV_COL_CREW}\nAlfa\n')   # only the key column
    cols, _ = ref.resolve_columns(ref.CREWS, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.CREWS, reader, cols)
    assert errors == []
    ref.apply(ref.CREWS, parsed)
    crew = Crew.objects.get(name='Alfa')
    assert crew.active is True and crew.notes == ''


@pytest.mark.django_db
def test_crews_active_false_honored():
    reader = _reader(f'{S.CSV_COL_CREW},{S.CSV_COL_ACTIVE}\nBeta,0\n')
    cols, _ = ref.resolve_columns(ref.CREWS, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.CREWS, reader, cols)
    assert errors == []
    ref.apply(ref.CREWS, parsed)
    assert Crew.objects.get(name='Beta').active is False


# --- species ---------------------------------------------------------------

@pytest.mark.django_db
def test_species_blank_density_uses_model_default():
    reader = _reader(
        f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY},{S.CSV_COL_MINOR}\n'
        'Abete,,1\n'
    )
    cols, _ = ref.resolve_columns(ref.SPECIES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.SPECIES, reader, cols)
    assert errors == []
    ref.apply(ref.SPECIES, parsed)
    sp = Species.objects.get(common_name='Abete')
    assert sp.density == Decimal('5.00')   # model default (column blank)
    assert sp.pressler_default == PRESSLER_DEFAULT
    assert sp.minor is True


@pytest.mark.django_db
def test_species_density_and_pressler_parsed():
    reader = _reader(
        f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY},{S.CSV_COL_PRESSLER}\n'
        'Castagno,9.2,1.5\n'
    )
    cols, _ = ref.resolve_columns(ref.SPECIES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.SPECIES, reader, cols)
    assert errors == []
    ref.apply(ref.SPECIES, parsed)
    sp = Species.objects.get(common_name='Castagno')
    assert sp.density == Decimal('9.20')
    assert sp.pressler_default == Decimal('1.50')


@pytest.mark.django_db
def test_species_unparseable_density_flagged():
    reader = _reader(f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY}\nAbete,abc\n')
    cols, _ = ref.resolve_columns(ref.SPECIES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.SPECIES, reader, cols)
    assert parsed == []
    assert len(errors) == 1


# --- products --------------------------------------------------------------

@pytest.mark.django_db
def test_products_single_column():
    reader = _reader(f'{S.CSV_COL_PRODUCT}\nTronchi\nCippato\n')
    cols, missing = ref.resolve_columns(ref.PRODUCTS, reader.fieldnames)
    assert not missing
    parsed, errors = ref.validate_rows(ref.PRODUCTS, reader, cols)
    assert errors == []
    ref.apply(ref.PRODUCTS, parsed)
    assert set(Product.objects.values_list('name', flat=True)) == {
        'Tronchi', 'Cippato'}


# --- generic validation ----------------------------------------------------

@pytest.mark.django_db
def test_duplicate_key_within_file_flagged():
    reader = _reader(f'{S.CSV_COL_REGION}\nCapistrano\nCapistrano\n')
    cols, _ = ref.resolve_columns(ref.REGIONS, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.REGIONS, reader, cols)
    assert len(parsed) == 1     # first accepted
    assert len(errors) == 1     # duplicate flagged


@pytest.mark.django_db
def test_blank_key_flagged():
    reader = _reader(f'{S.CSV_COL_CLASS},{S.CSV_COL_COPPICE}\n,1\n')
    cols, _ = ref.resolve_columns(ref.ECLASSES, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.ECLASSES, reader, cols)
    assert parsed == []
    assert len(errors) == 1


# --- apply idempotency -----------------------------------------------------

@pytest.mark.django_db
def test_apply_idempotent_and_updates():
    cols, _ = ref.resolve_columns(
        ref.SPECIES, [S.CSV_COL_SPECIES, S.CSV_COL_DENSITY])
    r1 = _reader(f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY}\nAbete,9.0\n')
    parsed, _ = ref.validate_rows(ref.SPECIES, r1, cols)
    assert ref.apply(ref.SPECIES, parsed) == (1, 0)        # created
    # Re-apply identical → no change.
    r1b = _reader(f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY}\nAbete,9.0\n')
    parsed_b, _ = ref.validate_rows(ref.SPECIES, r1b, cols)
    assert ref.apply(ref.SPECIES, parsed_b) == (0, 0)
    # Re-apply with a changed density → update, no duplicate row.
    r2 = _reader(f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY}\nAbete,8.0\n')
    parsed2, _ = ref.validate_rows(ref.SPECIES, r2, cols)
    assert ref.apply(ref.SPECIES, parsed2) == (0, 1)       # updated
    assert Species.objects.filter(common_name='Abete').count() == 1
    assert Species.objects.get(common_name='Abete').density == Decimal('8.00')


@pytest.mark.django_db
def test_apply_update_bumps_version_on_timestamped():
    cols, _ = ref.resolve_columns(ref.SPECIES, [S.CSV_COL_SPECIES, S.CSV_COL_DENSITY])
    r1 = _reader(f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY}\nAbete,9.0\n')
    ref.apply(ref.SPECIES, ref.validate_rows(ref.SPECIES, r1, cols)[0])
    v0 = Species.objects.get(common_name='Abete').version
    r2 = _reader(f'{S.CSV_COL_SPECIES},{S.CSV_COL_DENSITY}\nAbete,8.0\n')
    ref.apply(ref.SPECIES, ref.validate_rows(ref.SPECIES, r2, cols)[0])
    assert Species.objects.get(common_name='Abete').version == v0 + 1


# --- tractors ---------------------------------------------------------------

@pytest.mark.django_db
def test_tractors_minimal_key_only():
    """Tractor with only the required Trattore column; optional fields use model defaults."""
    reader = _reader(f'{S.CSV_COL_TRACTOR_NAME}\nFiat 110-90\n')
    cols, missing = ref.resolve_columns(ref.TRACTORS, reader.fieldnames)
    assert missing == []
    parsed, errors = ref.validate_rows(ref.TRACTORS, reader, cols)
    assert errors == []
    assert ref.apply(ref.TRACTORS, parsed) == (1, 0)
    t = Tractor.objects.get(name='Fiat 110-90')
    assert t.manufacturer == ''
    assert t.model == ''
    assert t.year is None


@pytest.mark.django_db
def test_tractors_full_row():
    """All four columns parsed correctly."""
    csv_text = (
        f'{S.CSV_COL_TRACTOR_NAME},{S.CSV_COL_MANUFACTURER},'
        f'{S.CSV_COL_MODEL},{S.CSV_COL_YEAR}\n'
        'Fiat 110-90,Fiat,110-90,1995\n'
    )
    reader = _reader(csv_text)
    cols, missing = ref.resolve_columns(ref.TRACTORS, reader.fieldnames)
    assert missing == []
    parsed, errors = ref.validate_rows(ref.TRACTORS, reader, cols)
    assert errors == []
    ref.apply(ref.TRACTORS, parsed)
    t = Tractor.objects.get(name='Fiat 110-90')
    assert t.manufacturer == 'Fiat'
    assert t.model == '110-90'
    assert t.year == 1995


@pytest.mark.django_db
def test_tractors_update_idempotent():
    """Re-importing with changed year updates the row."""
    csv1 = (
        f'{S.CSV_COL_TRACTOR_NAME},{S.CSV_COL_YEAR}\n'
        'Landini 135,2000\n'
    )
    csv2 = (
        f'{S.CSV_COL_TRACTOR_NAME},{S.CSV_COL_YEAR}\n'
        'Landini 135,2001\n'
    )
    r1 = _reader(csv1)
    cols, _ = ref.resolve_columns(ref.TRACTORS, r1.fieldnames)
    assert ref.apply(ref.TRACTORS, ref.validate_rows(ref.TRACTORS, r1, cols)[0]) == (1, 0)
    r2 = _reader(csv2)
    cols2, _ = ref.resolve_columns(ref.TRACTORS, r2.fieldnames)
    assert ref.apply(ref.TRACTORS, ref.validate_rows(ref.TRACTORS, r2, cols2)[0]) == (0, 1)
    assert Tractor.objects.get(name='Landini 135').year == 2001


@pytest.mark.django_db
def test_tractors_blank_key_flagged():
    """A blank Trattore name (explicit comma-padded cell) is a required-column error."""
    reader = _reader(f'{S.CSV_COL_TRACTOR_NAME},{S.CSV_COL_MODEL}\n,110-90\n')
    cols, _ = ref.resolve_columns(ref.TRACTORS, reader.fieldnames)
    parsed, errors = ref.validate_rows(ref.TRACTORS, reader, cols)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_tractors_in_all_tables():
    """TRACTORS appears in ALL_TABLES after CREWS."""
    names = [t.name for t in ref.ALL_TABLES]
    assert 'tractors' in names
    crews_idx = names.index('crews')
    tractors_idx = names.index('tractors')
    assert tractors_idx == crews_idx + 1
