"""Unit tests for the sampled-trees CSV import core (apps/campionamenti/csv_trees)."""

from datetime import date

import pytest

from apps.base import csv_io
from apps.base.models import Sample, SampleArea, SampleGrid, Survey, TreeSample
from apps.campionamenti import csv_trees
from config import strings as S


# Header with all TREE_CSV_REQUIRED columns; no Data column (use default_date).
TREE_HEADER = ('Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
               'D_cm,H_m,L10_mm,Genere,Fustaia')


@pytest.fixture
def survey_with_area(parcels, species):
    """A survey + grid + one sample area on parcels[0] (Capistrano/1)."""
    grid = SampleGrid.objects.create(name='trees-core-grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcels[0], number='1',
        lat=38.5, lon=16.1, r_m=12,
    )
    survey = Survey.objects.create(name='trees-core-survey', sample_grid=grid)
    return {'survey': survey, 'grid': grid, 'area': area, 'parcel': parcels[0]}


def _reader(csv_text):
    return csv_io.read(csv_text, csv_trees.TREE_CSV_REQUIRED)


@pytest.mark.django_db
def test_db_indexes_scopes_to_survey(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    area = survey_with_area['area']
    key = (area.parcel.region.name.lower(), area.parcel.name, area.number)
    assert key in idx.area_cache
    assert 'abete' in idx.species_cache  # from the `species` fixture


def _validate(csv_text, idx, *, default_date=date(2024, 9, 15)):
    reader = _reader(csv_text)
    has_date = bool(reader) and S.CSV_COL_DATA in reader[0]
    return csv_trees.validate_rows(
        reader, idx, has_date_column=has_date, default_date=default_date)


@pytest.mark.django_db
def test_validate_rows_happy_path(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert errors == []
    assert len(parsed) == 1
    assert parsed[0][csv_trees.FIELD_NUMBER] == 1
    assert parsed[0][csv_trees.FIELD_D_CM] == 30


@pytest.mark.django_db
def test_validate_rows_unknown_area_flagged(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},999,1,0,False,30,15.5,250,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_unknown_species_flagged(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,Zzz,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_parse_error_flagged(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,,0,False,,15.5,250,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_missing_date_flagged(survey_with_area):
    """No Data column and no default date → each row flagged."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx, default_date=None)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_unrecognised_fustaia_flagged(survey_with_area):
    """A non-blank but unrecognised Fustaia is an error, not a silent False."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,Abete,maybe\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_invalid_l10_flagged(survey_with_area):
    """A non-blank but invalid L10_mm is an error, not a silent 0."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,abc,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_invalid_matricina_flagged(survey_with_area):
    """A non-blank but unrecognised Matricina is an error, not a silent False."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,maybe,30,15.5,250,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_invalid_pai_flagged(survey_with_area):
    """A non-blank but unrecognised PAI value is an error, not a silent False."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + f',{S.CSV_COL_PRESERVED}\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,Abete,True,maybe\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_blank_optionals_default(survey_with_area):
    """Blank Pollone/L10_mm default to 0 and blank Matricina defaults to False."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,,,30,15.5,,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert errors == []
    assert len(parsed) == 1
    assert parsed[0][csv_trees.FIELD_SHOOT] == 0
    assert parsed[0][csv_trees.FIELD_L10_MM] == 0
    assert parsed[0][csv_trees.FIELD_STANDARD] is False


@pytest.mark.django_db
def test_apply_creates_sample_and_treesample(survey_with_area):
    survey = survey_with_area['survey']
    idx = csv_trees.db_indexes(survey)
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert errors == []
    counts = csv_trees.apply(survey, parsed)
    assert counts == {'n_samples': 1, 'n_trees': 1}
    sample = Sample.objects.get(survey=survey, sample_area=survey_with_area['area'])
    ts = TreeSample.objects.get(sample=sample)
    assert ts.number == 1
    assert ts.d_cm == 30
