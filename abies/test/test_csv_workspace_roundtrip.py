"""Public workspace exports must be accepted by their matching imports."""
from datetime import date

import pytest
from django.utils import translation

from apps.base import csv_io
from apps.base.models import Sample, SampleArea, SampleGrid, Survey, Tree, TreeSample
from apps.campionamenti import csv_trees


@pytest.mark.django_db
@pytest.mark.parametrize('locale', ['it', 'en'])
@pytest.mark.parametrize('same_date', [False, True])
def test_free_survey_export_retains_sample_groups_and_restarted_numbers(parcels, species, same_date, locale):
    source = Survey.objects.create(name='Free export source')
    for index in range(2):
        sample = Sample.objects.create(survey=source, date=date(2026, 11, 1 if same_date else index + 1))
        TreeSample.objects.create(sample=sample, parcel=parcels[0],
            tree=Tree.objects.create(species=species[0]), number=1,
            d_cm=30, h_m='20', h_measured=True, note=f'Sample {index}')
    with translation.override(locale):
        exported = csv_trees.render_csv(source)
    target = Survey.objects.create(name='Free restore target')
    parsed, errors = csv_trees.validate_rows(csv_io.read(exported), csv_trees.db_indexes(target),
        has_date_column=True, default_date=None)
    assert not errors, errors
    csv_trees.apply_unstructured(target, parsed)
    assert Sample.objects.filter(survey=target).count() == 2
    assert TreeSample.objects.filter(sample__survey=target).values('tree_id').distinct().count() == 2
    assert list(TreeSample.objects.filter(sample__survey=target).order_by('id').values_list('number', flat=True)) == [1, 1]
    with translation.override(locale):
        assert csv_trees.render_csv(target) == exported


@pytest.mark.django_db
@pytest.mark.parametrize('locale', ['it', 'en'])
@pytest.mark.parametrize('structured', [False, True])
def test_sampled_tree_export_retains_distinct_preserved_number(parcels, species, locale, structured):
    grid = SampleGrid.objects.create(name='Preserved grid') if structured else None
    area = SampleArea.objects.create(sample_grid=grid, parcel=parcels[0],
        number='1', lat=38.5, lon=16.1, r_m=12) if structured else None
    source = Survey.objects.create(name='Preserved source', sample_grid=grid)
    sample = Sample.objects.create(survey=source, sample_area=area, date=date(2026, 11, 1))
    TreeSample.objects.create(sample=sample, parcel=parcels[0],
        tree=Tree.objects.create(species=species[0]), number=7, preserved_number=1440,
        d_cm=30, h_m='20', h_measured=True)
    target = Survey.objects.create(name='Preserved restore target', sample_grid=grid)
    with translation.override(locale):
        exported = csv_trees.render_csv(source)
    parsed, errors = csv_trees.validate_rows(csv_io.read(exported),
        csv_trees.db_indexes(target), has_date_column=True, default_date=None)
    assert not errors, errors
    assert parsed[0]['number'] == 7
    assert parsed[0]['preserved_number'] == 1440
    csv_trees.apply(target, parsed)
    restored = TreeSample.objects.get(sample__survey=target)
    assert restored.tree_id == TreeSample.objects.get(sample=sample).tree_id
    assert (restored.number, restored.preserved_number) == (7, 1440)
    with translation.override(locale):
        assert csv_trees.render_csv(target) == exported


@pytest.mark.django_db
@pytest.mark.parametrize('case', ['group_date', 'duplicate_pai', 'pai_flag', 'auto_numbers', 'repeated_pai'])
def test_optional_sample_identity_columns_validate_and_scope_rows(parcels, species, case):
    from config import strings as S

    survey = Survey.objects.create(name='Grouped CSV validation')
    first = dict(zip(csv_trees.FREE_TREE_CSV_REQUIRED,
        [parcels[0].region.name, parcels[0].name, '1', '0', '0', '30', '20', '0', '2', species[0].common_name, '1']))
    first.update({S.CSV_COL_DATA: '2026-11-01', S.CSV_COL_SAMPLE_GROUP: 'first',
                  S.CSV_COL_PRESERVED: '1', S.CSV_COL_PRESERVED_NUMBER: '1440'})
    second = {**first, S.CSV_COL_TREE: '2'}
    if case == 'group_date':
        second[S.CSV_COL_DATA] = '2026-11-02'
    elif case == 'pai_flag':
        second[S.CSV_COL_PRESERVED] = '0'
    elif case == 'repeated_pai':
        second[S.CSV_COL_SAMPLE_GROUP] = 'second'
    elif case == 'auto_numbers':
        first.update({S.CSV_COL_TREE: '3', S.CSV_COL_PRESERVED: '0', S.CSV_COL_PRESERVED_NUMBER: ''})
        second = {**first, S.CSV_COL_TREE: ''}
    buffer, writer = csv_io.csv_buffer(';')
    writer.writerow(first)
    writer.writerows([[row[key] for key in first] for row in [first, second]])
    parsed, errors = csv_trees.validate_rows(csv_io.read(buffer.getvalue()), csv_trees.db_indexes(survey),
        has_date_column=True, default_date=None)
    if case in {'group_date', 'duplicate_pai', 'pai_flag'}:
        assert errors and 'Riga 3:' in errors[0]
        assert not TreeSample.objects.exists()
    else:
        assert not errors, errors
        csv_trees.apply_unstructured(survey, parsed)
        rows = TreeSample.objects.filter(sample__survey=survey).order_by('id')
        if case == 'auto_numbers':
            assert list(rows.values_list('number', flat=True)) == [3, 4]
        else:
            assert rows.values('sample_id').distinct().count() == 2
            assert rows.values('tree_id').distinct().count() == 1
            assert set(rows.values_list('preserved_number', flat=True)) == {1440}
