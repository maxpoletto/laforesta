"""Unit tests for the preserved-trees CSV import core (apps/campionamenti/csv_preserved)."""

import pytest

from apps.base import csv_io
from apps.base.models import Sample, Survey, Tree, TreeSample
from apps.base.preserved_trees import PRESERVED_LEGACY_UNKNOWN_DATE
from apps.campionamenti import csv_preserved
from config import strings as S


def _pai_row(tree, parcel, *, number=1):
    survey = Survey.objects.create(name=f'csv-preserved-existing-{tree.id}-{number}')
    sample = Sample.objects.create(
        sample_area=None, survey=survey, date='2024-09-14',
    )
    return TreeSample.objects.create(
        sample=sample, tree=tree, parcel=parcel, number=number,
        preserved_number=number, d_cm=40, h_m='18.00',
        h_measured=True, lat=38.1, lon=16.1,
    )


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
    pai = TreeSample.objects.get(tree=t)
    assert pai.preserved_number == 7
    assert pai.sample.date.isoformat() == '2024-09-15'
    assert pai.d_cm == 42
    assert str(pai.h_m) == '18.50'
    assert pai.note == 'nota'


@pytest.mark.django_db
def test_preserved_legacy_blank_date_and_height_imports(parcels, species):
    reader = _reader(_csv('Capistrano,1,7,Abete,16.12345,38.45678,,,42,,nota'))
    idx = csv_preserved.db_indexes()

    parsed, errors = csv_preserved.validate_rows(reader, idx)

    assert errors == []
    assert len(parsed) == 1
    assert parsed[0]['date'] == PRESERVED_LEGACY_UNKNOWN_DATE
    assert parsed[0]['h_m'] is None
    assert parsed[0]['h_measured'] is False

    assert csv_preserved.apply(parsed) == 1
    pai = TreeSample.objects.get(preserved_number=7)
    assert pai.sample.date == PRESERVED_LEGACY_UNKNOWN_DATE
    assert pai.h_m is None
    assert pai.h_measured is False


@pytest.mark.django_db
def test_preserved_blank_height_cannot_be_marked_measured(parcels, species):
    header = (
        f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_NUMBER},'
        f'{S.CSV_COL_SPECIES},{S.CSV_COL_LON},{S.CSV_COL_LAT},'
        f'{S.CSV_COL_DATA},{S.CSV_COL_ESTIMATED_BIRTH_YEAR},'
        f'{S.CSV_COL_D_CM},{S.CSV_COL_H_M},{S.CSV_COL_H_MEASURED}'
    )
    reader = _reader('\n'.join([
        header,
        'Capistrano,1,7,Abete,16.12345,38.45678,2024-09-15,,42,,true',
    ]))
    idx = csv_preserved.db_indexes()

    parsed, errors = csv_preserved.validate_rows(reader, idx)

    assert parsed == []
    assert errors == [S.ERR_CSV_ROW_PARSE.format(2, S.CSV_COL_H_M)]


@pytest.mark.django_db
def test_preserved_duplicate_number_in_file_flagged(parcels, species):
    csv_text = _csv(
        'Capistrano,1,1,Abete,16.1,38.4,2024-09-15,,42,18.5,',
        'Capistrano,1,1,Castagno,16.2,38.5,2024-09-15,,43,19.0,',
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
    _pai_row(tree, parcels[0], number=1)
    reader = _reader(_csv('Capistrano,1,1,Castagno,16.2,38.5,2024-09-15,,43,19.0,'))
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
        'Capistrano,1,1,Abete,16.1,38.4,2024-09-15,,42,18.5,',
        'Capistrano,2,1,Castagno,16.2,38.5,2024-09-15,,43,19.0,',
        'Fabrizia,1,1,Abete,16.3,38.6,2024-09-15,,44,20.0,',
    )
    reader = _reader(csv_text)
    idx = csv_preserved.db_indexes()
    parsed, errors = csv_preserved.validate_rows(reader, idx)
    assert errors == []
    assert len(parsed) == 3
    assert csv_preserved.apply(parsed) == 3
    assert Tree.objects.count() == 3
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 3


@pytest.mark.django_db
def test_preserved_missing_required_col(parcels, species):
    """CSV missing a required column raises CsvError on read."""
    from apps.base.csv_io import CsvError
    bad_csv = (
        f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},'
        f'{S.CSV_COL_LON},{S.CSV_COL_LAT}\nCapistrano,1,16.1,38.4\n'
    )
    with pytest.raises(CsvError):
        csv_io.read(bad_csv, required_cols=csv_preserved.PRESERVED_CSV_REQUIRED)
