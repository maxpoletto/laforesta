"""Unit tests for the preserved-trees CSV import core (apps/campionamenti/csv_preserved)."""

import pytest

from apps.base import csv_io
from apps.base.models import Tree
from apps.campionamenti import csv_preserved
from config import strings as S


def _reader(text):
    return csv_io.read(text)


def _csv(*rows):
    """Build a preserved-trees CSV string from header + data rows."""
    header = (
        f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},'
        f'{S.CSV_COL_SPECIES},{S.CSV_COL_LON},{S.CSV_COL_LAT}'
    )
    return '\n'.join([header] + list(rows))


@pytest.mark.django_db
def test_preserved_happy_path(parcels, species):
    """Valid row creates a preserved=True, coppice=False Tree."""
    reader = _reader(_csv('Capistrano,1,Abete,16.12345,38.45678'))
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
    assert t.lat == pytest.approx(38.45678, abs=1e-4)
    assert t.lon == pytest.approx(16.12345, abs=1e-4)


@pytest.mark.django_db
def test_preserved_unknown_species_flagged(parcels, species):
    """Unknown Genere produces an error."""
    reader = _reader(_csv('Capistrano,1,Quercia,16.12345,38.45678'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert parsed == []
    assert len(errors) == 1
    assert 'Quercia' in errors[0]


@pytest.mark.django_db
def test_preserved_unknown_parcel_flagged(parcels, species):
    """Unknown parcel produces an error."""
    reader = _reader(_csv('Capistrano,999,Abete,16.12345,38.45678'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_preserved_unknown_region_flagged(parcels, species):
    """Unknown region produces an error."""
    reader = _reader(_csv('Sconosciuta,1,Abete,16.12345,38.45678'))
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_preserved_multiple_rows(parcels, species):
    """Multiple valid rows all create Trees."""
    csv_text = _csv(
        'Capistrano,1,Abete,16.1,38.4',
        'Capistrano,2,Castagno,16.2,38.5',
        'Fabrizia,1,Abete,16.3,38.6',
    )
    reader = _reader(csv_text)
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert errors == []
    assert len(parsed) == 3
    assert csv_preserved.apply(parsed) == 3
    assert Tree.objects.count() == 3


@pytest.mark.django_db
def test_preserved_missing_required_col(parcels, species):
    """CSV missing a required column raises CsvError on read."""
    from apps.base.csv_io import CsvError
    bad_csv = f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_LON},{S.CSV_COL_LAT}\nCapistrano,1,16.1,38.4\n'
    with pytest.raises(CsvError):
        csv_io.read(bad_csv, required_cols=csv_preserved.PRESERVED_CSV_REQUIRED)
