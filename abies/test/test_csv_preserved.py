"""Unit tests for the preserved-trees CSV import core (apps/campionamenti/csv_preserved)."""

import pytest

from apps.base import csv_io
from apps.base.models import Tree, TreePreserved
from apps.campionamenti import csv_preserved
from config import strings as S


def _reader(text):
    return csv_io.read(text)


def _csv(*rows):
    """Build a preserved-trees CSV string from header + data rows."""
    header = (
        f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_NUMBER},'
        f'{S.CSV_COL_SPECIES},{S.CSV_COL_LON},{S.CSV_COL_LAT},'
        f'{S.CSV_COL_DATA},{S.CSV_COL_ESTIMATED_BIRTH_YEAR},'
        f'{S.CSV_COL_D_CM},{S.CSV_COL_H_M},{S.CSV_COL_NOTE}'
    )
    return '\n'.join([header] + list(rows))


@pytest.mark.django_db
def test_preserved_happy_path(parcels, species):
    """Valid row creates a preserved=True, coppice=False Tree."""
    reader = _reader(_csv('Capistrano,1,7,Abete,16.12345,38.45678,2024-09-15,1920,42,18.5,nota'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert errors == []
    assert len(parsed) == 1
    n = csv_preserved.apply(parsed)
    assert n == 1
    t = Tree.objects.get()
    assert t.preserved is True
    assert t.coppice is False
    assert t.species.common_name == 'Abete'
    assert t.parcel == parcels[0]
    assert t.estimated_birth_year == 1920
    assert t.lat == pytest.approx(38.45678, abs=1e-4)
    assert t.lon == pytest.approx(16.12345, abs=1e-4)
    pai = TreePreserved.objects.get(tree=t)
    assert pai.number == 7
    assert pai.date.isoformat() == '2024-09-15'
    assert pai.d_cm == 42
    assert str(pai.h_m) == '18.50'
    assert pai.note == 'nota'


@pytest.mark.django_db
def test_preserved_duplicate_number_in_file_flagged(parcels, species):
    csv_text = _csv(
        'Capistrano,1,1,Abete,16.1,38.4,,,,,',
        'Capistrano,1,1,Castagno,16.2,38.5,,,,,',
    )
    reader = _reader(csv_text)
    idx = csv_preserved.db_indexes()

    parsed, errors = csv_preserved.validate_rows(reader, idx)

    assert len(parsed) == 1
    assert len(errors) == 1
    assert S.ERR_CSV_DUPLICATE_KEY.format(
        3, f'{S.CSV_COL_PARCEL}/{S.CSV_COL_NUMBER}', 'Capistrano 1/1',
    ) in errors


@pytest.mark.django_db
def test_preserved_duplicate_number_in_db_flagged(parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        lat=38.1, lon=16.1,
    )
    TreePreserved.objects.create(
        tree=tree, parcel=parcels[0], number=1, lat=38.1, lon=16.1,
    )
    reader = _reader(_csv('Capistrano,1,1,Castagno,16.2,38.5,,,,,'))
    idx = csv_preserved.db_indexes()

    parsed, errors = csv_preserved.validate_rows(reader, idx)

    assert parsed == []
    assert len(errors) == 1
    assert S.ERR_CSV_DUPLICATE_KEY.format(
        2, f'{S.CSV_COL_PARCEL}/{S.CSV_COL_NUMBER}', 'Capistrano 1/1',
    ) in errors


@pytest.mark.django_db
def test_preserved_unknown_species_flagged(parcels, species):
    """Unknown Genere produces an error."""
    reader = _reader(_csv('Capistrano,1,1,Quercia,16.12345,38.45678,,,,,'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert parsed == []
    assert len(errors) == 1
    assert 'Quercia' in errors[0]


@pytest.mark.django_db
def test_preserved_unknown_parcel_flagged(parcels, species):
    """Unknown parcel produces an error."""
    reader = _reader(_csv('Capistrano,999,1,Abete,16.12345,38.45678,,,,,'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_preserved_unknown_region_flagged(parcels, species):
    """Unknown region produces an error."""
    reader = _reader(_csv('Sconosciuta,1,1,Abete,16.12345,38.45678,,,,,'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_preserved_multiple_rows(parcels, species):
    """Multiple valid rows all create Trees."""
    csv_text = _csv(
        'Capistrano,1,1,Abete,16.1,38.4,,,,,',
        'Capistrano,2,1,Castagno,16.2,38.5,,,,,',
        'Fabrizia,1,1,Abete,16.3,38.6,,,,,',
    )
    reader = _reader(csv_text)
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert errors == []
    assert len(parsed) == 3
    assert csv_preserved.apply(parsed) == 3
    assert Tree.objects.count() == 3
    assert TreePreserved.objects.count() == 3


@pytest.mark.django_db
def test_preserved_missing_required_col(parcels, species):
    """CSV missing a required column raises CsvError on read."""
    from apps.base.csv_io import CsvError
    bad_csv = f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_LON},{S.CSV_COL_LAT}\nCapistrano,1,16.1,38.4\n'
    with pytest.raises(CsvError):
        csv_io.read(bad_csv, required_cols=csv_preserved.PRESERVED_CSV_REQUIRED)
