"""Unit tests for the sampled-trees CSV import core (apps/campionamenti/csv_trees)."""

from datetime import date

import pytest

from apps.base import csv_io
from apps.base.models import Sample, SampleArea, SampleGrid, Survey, Tree, TreeSample
from apps.campionamenti import csv_trees
from config import strings as S


# Header with all TREE_CSV_REQUIRED columns; no Data column (use default_date).
TREE_HEADER = ('Compresa,Particella,Area saggio,Albero,Pollone,Matricina,'
               'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia')
FREE_TREE_HEADER = ('Compresa,Particella,Albero,Pollone,Matricina,'
                    'D_cm,H_m,L10_mm,Pressler,Genere,Fustaia')


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


def _free_reader(csv_text):
    return csv_io.read(csv_text, csv_trees.FREE_TREE_CSV_REQUIRED)


@pytest.mark.django_db
def test_db_indexes_scopes_to_survey(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    area = survey_with_area['area']
    key = (area.parcel.region.name.lower(), area.parcel.name, area.number)
    assert key in idx.area_cache
    assert 'abete' in idx.species_cache  # from the `species` fixture


def _validate(csv_text, idx, *, default_date=date(2024, 9, 15)):
    reader = _reader(csv_text)
    has_date = S.CSV_COL_DATA in (reader.fieldnames or [])
    return csv_trees.validate_rows(
        reader, idx, has_date_column=has_date, default_date=default_date)


def _validate_free(csv_text, survey, *, default_date=date(2024, 9, 15)):
    reader = _free_reader(csv_text)
    has_date = S.CSV_COL_DATA in (reader.fieldnames or [])
    return csv_trees.validate_rows(
        reader, csv_trees.db_indexes(survey),
        has_date_column=has_date, default_date=default_date,
    )


@pytest.mark.django_db
def test_validate_rows_happy_path(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        f'{TREE_HEADER},{S.CSV_COL_OPERATOR},{S.CSV_COL_NOTE}\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,'
        'Abete,True,Mario Rossi,nota campo\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert errors == []
    assert len(parsed) == 1
    assert parsed[0][csv_trees.FIELD_NUMBER] == 1
    assert parsed[0][csv_trees.FIELD_D_CM] == 30
    assert parsed[0][csv_trees.FIELD_PRESSLER_COEFF] == 2


@pytest.mark.django_db
def test_validate_rows_unknown_area_flagged(survey_with_area):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},999,1,0,False,30,15.5,250,2,Abete,True\n'
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
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,Zzz,True\n'
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
        f'{parcel.region.name},{parcel.name},1,,0,False,,15.5,250,2,Abete,True\n'
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
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,Abete,True\n'
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
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,Abete,maybe\n'
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
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,abc,2,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_invalid_pressler_flagged(survey_with_area):
    """Pressler must parse and be positive."""
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,0,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.parametrize(('number', 'd_cm', 'h_m'), [
    ('0', '30', '15.5'),
    ('1', '0', '15.5'),
    ('1', '30', '0'),
])
@pytest.mark.django_db
def test_validate_rows_rejects_non_positive_required_measurements(
        survey_with_area, number, d_cm, h_m):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,{number},0,False,'
        f'{d_cm},{h_m},250,2,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.parametrize(('shoot', 'l10_mm'), [
    ('-1', '250'),
    ('0', '-1'),
])
@pytest.mark.django_db
def test_validate_rows_rejects_negative_optional_measurements(
        survey_with_area, shoot, l10_mm):
    idx = csv_trees.db_indexes(survey_with_area['survey'])
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,{shoot},False,30,15.5,'
        f'{l10_mm},2,Abete,True\n'
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
        f'{parcel.region.name},{parcel.name},1,1,0,maybe,30,15.5,250,2,Abete,True\n'
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
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,Abete,True,maybe\n'
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
        f'{parcel.region.name},{parcel.name},1,1,,,30,15.5,,2,Abete,True\n'
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
        f'{TREE_HEADER},{S.CSV_COL_OPERATOR},{S.CSV_COL_NOTE}\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,'
        'Abete,True,Mario Rossi,nota campo\n'
    )
    parsed, errors = _validate(csv_text, idx)
    assert errors == []
    counts = csv_trees.apply(survey, parsed)
    assert counts == {'n_samples': 1, 'n_trees': 1}
    sample = Sample.objects.get(survey=survey, sample_area=survey_with_area['area'])
    ts = TreeSample.objects.get(sample=sample)
    assert ts.number == 1
    assert ts.d_cm == 30
    assert ts.pressler_coeff == 2
    assert ts.parcel == parcel
    assert ts.operator == 'Mario Rossi'
    assert ts.note == 'nota campo'


@pytest.mark.django_db
def test_apply_reuses_tree_identity_across_surveys(survey_with_area):
    survey = survey_with_area['survey']
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,0,False,30,15.5,250,2,Abete,True\n'
    )
    parsed, errors = _validate(csv_text, csv_trees.db_indexes(survey))
    assert errors == []
    csv_trees.apply(survey, parsed)

    next_survey = Survey.objects.create(
        name='trees-core-follow-up', sample_grid=survey.sample_grid,
    )
    parsed, errors = _validate(csv_text, csv_trees.db_indexes(next_survey))
    assert errors == []
    csv_trees.apply(next_survey, parsed)

    rows = list(TreeSample.objects.order_by('sample__survey_id'))
    assert len(rows) == 2
    assert {row.tree_id for row in rows} == {rows[0].tree_id}


@pytest.mark.django_db
def test_apply_groups_coppice_shoots_under_one_tree(survey_with_area):
    survey = survey_with_area['survey']
    parcel = survey_with_area['parcel']
    csv_text = (
        TREE_HEADER + '\n'
        f'{parcel.region.name},{parcel.name},1,1,1,False,30,15.5,250,2,Abete,False\n'
        f'{parcel.region.name},{parcel.name},1,1,2,False,29,14.5,240,2,Abete,False\n'
    )
    parsed, errors = _validate(csv_text, csv_trees.db_indexes(survey))
    assert errors == []

    counts = csv_trees.apply(survey, parsed)

    assert counts == {'n_samples': 1, 'n_trees': 2}
    rows = list(TreeSample.objects.select_related('tree').order_by('shoot'))
    assert [row.shoot for row in rows] == [1, 2]
    assert {row.tree_id for row in rows} == {rows[0].tree_id}
    assert all(row.tree.coppice for row in rows)


@pytest.mark.django_db
def test_validate_free_rows_happy_path(parcels, species):
    survey = Survey.objects.create(name='free-csv-survey')
    parcel = parcels[0]
    csv_text = (
        f'{FREE_TREE_HEADER},{S.CSV_COL_H_MEASURED},{S.CSV_COL_LAT},'
        f'{S.CSV_COL_LON},{S.CSV_COL_ACC_M},{S.CSV_COL_OPERATOR},'
        f'{S.CSV_COL_NOTE}\n'
        f'{parcel.region.name},{parcel.name},1,0,False,30,15.5,250,2,'
        'Abete,True,true,38.5,16.1,7,Mario Rossi,nota campo\n'
    )

    parsed, errors = _validate_free(csv_text, survey)

    assert errors == []
    assert len(parsed) == 1
    row = parsed[0]
    assert row[csv_trees.FIELD_AREA] is None
    assert row[csv_trees.FIELD_PARCEL] == parcel
    assert row[csv_trees.FIELD_H_MEASURED] is True
    assert row[csv_trees.FIELD_LAT] == pytest.approx(38.5)
    assert row[csv_trees.FIELD_LON] == pytest.approx(16.1)
    assert row[csv_trees.FIELD_ACC_M] == 7
    assert row[csv_trees.FIELD_OPERATOR] == 'Mario Rossi'
    assert row[csv_trees.FIELD_NOTE] == 'nota campo'


@pytest.mark.django_db
def test_validate_free_rows_rejects_mixed_dates(parcels, species):
    survey = Survey.objects.create(name='free-csv-mixed-dates')
    parcel = parcels[0]
    csv_text = (
        f'{FREE_TREE_HEADER},{S.CSV_COL_DATA}\n'
        f'{parcel.region.name},{parcel.name},1,0,False,30,15.5,250,2,'
        'Abete,True,2024-09-15\n'
        f'{parcel.region.name},{parcel.name},2,0,False,32,16.5,250,2,'
        'Abete,True,2024-09-16\n'
    )

    parsed, errors = _validate_free(csv_text, survey)

    assert len(parsed) == 1
    assert errors == [
        S.ERR_CSV_ROW_FREE_SAMPLE_DATE_CONFLICT.format(3, '2024-09-15'),
    ]


@pytest.mark.django_db
def test_validate_free_rows_rejects_one_sided_coordinates(parcels, species):
    survey = Survey.objects.create(name='free-csv-one-sided-coord')
    parcel = parcels[0]
    csv_text = (
        f'{FREE_TREE_HEADER},{S.CSV_COL_LAT}\n'
        f'{parcel.region.name},{parcel.name},1,0,False,30,15.5,250,2,'
        'Abete,True,38.5\n'
    )

    parsed, errors = _validate_free(csv_text, survey)

    assert parsed == []
    assert errors == [
        S.ERR_CSV_ROW_PARSE.format(2, f'{S.CSV_COL_LAT}/{S.CSV_COL_LON}'),
    ]


@pytest.mark.django_db
def test_apply_free_creates_one_sample_and_new_tree_per_ordinary_row(parcels, species):
    survey = Survey.objects.create(name='free-csv-apply')
    parcel = parcels[0]
    csv_text = (
        f'{FREE_TREE_HEADER},{S.CSV_COL_H_MEASURED}\n'
        f'{parcel.region.name},{parcel.name},1,0,False,30,15.5,250,2,'
        'Abete,True,true\n'
        f'{parcel.region.name},{parcel.name},2,0,False,32,16.5,250,2,'
        'Abete,True,false\n'
    )
    parsed, errors = _validate_free(csv_text, survey, default_date=date(2024, 9, 20))
    assert errors == []

    counts = csv_trees.apply(survey, parsed)

    assert counts == {'n_samples': 1, 'n_trees': 2}
    sample = Sample.objects.get(survey=survey, sample_area__isnull=True)
    assert sample.date == date(2024, 9, 20)
    rows = list(TreeSample.objects.filter(sample=sample).order_by('number'))
    assert [row.number for row in rows] == [1, 2]
    assert rows[0].tree_id != rows[1].tree_id
    assert [row.h_measured for row in rows] == [True, False]


@pytest.mark.django_db
def test_apply_free_reuses_preserved_tree_identity(parcels, species):
    parcel = parcels[0]
    existing_tree = Tree.objects.create(
        species=species[0], coppice=False,
    )
    pai_survey = Survey.objects.create(name='existing-pai-survey')
    pai_sample = Sample.objects.create(
        sample_area=None, survey=pai_survey, date=date(2024, 9, 1),
    )
    TreeSample.objects.create(
        sample=pai_sample, tree=existing_tree, parcel=parcel,
        number=7, preserved_number=7, d_cm=40, h_m='18.00', h_measured=True,
    )
    survey = Survey.objects.create(name='free-csv-preserved-reuse')
    csv_text = (
        f'{FREE_TREE_HEADER},{S.CSV_COL_PRESERVED}\n'
        f'{parcel.region.name},{parcel.name},7,0,False,42,19.5,250,2,'
        'Abete,True,true\n'
    )
    parsed, errors = _validate_free(csv_text, survey)
    assert errors == []

    counts = csv_trees.apply(survey, parsed)

    assert counts == {'n_samples': 1, 'n_trees': 1}
    row = TreeSample.objects.get(sample__survey=survey)
    assert row.tree == existing_tree
    assert row.preserved_number == 7


@pytest.mark.django_db
def test_validate_free_rejects_preserved_species_mismatch(parcels, species):
    parcel = parcels[0]
    existing_tree = Tree.objects.create(
        species=species[0], coppice=False,
    )
    pai_survey = Survey.objects.create(name='existing-pai-species-survey')
    pai_sample = Sample.objects.create(
        sample_area=None, survey=pai_survey, date=date(2024, 9, 1),
    )
    TreeSample.objects.create(
        sample=pai_sample, tree=existing_tree, parcel=parcel,
        number=7, preserved_number=7, d_cm=40, h_m='18.00', h_measured=True,
    )
    survey = Survey.objects.create(name='free-csv-preserved-species-mismatch')
    csv_text = (
        f'{FREE_TREE_HEADER},{S.CSV_COL_PRESERVED}\n'
        f'{parcel.region.name},{parcel.name},7,0,False,42,19.5,250,2,'
        f'{species[1].common_name},True,true\n'
    )

    parsed, errors = _validate_free(csv_text, survey)

    assert parsed == []
    assert errors == [
        S.ERR_CSV_ROW_PAI_SPECIES_CONFLICT.format(
            2, parcel.region.name, parcel.name, 7, species[0].common_name,
        ),
    ]
