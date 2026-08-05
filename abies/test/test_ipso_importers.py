"""Focused tests for Ipso staged-upload import cores."""

from datetime import date

import pytest

from apps.base.models import Sample, SampleArea, SampleGrid, Survey, Tree, TreeSample
from apps.ipso.importers import (
    free_survey_import_rows, record_measurements, sample_import_rows,
)
from config import strings as S
from config.constants import (
    FIELD_DATE, FIELD_D_CM, FIELD_H_M, FIELD_NUMBER, FIELD_PARCEL_ID,
    FIELD_PRESERVED,
    FIELD_PRESSLER_COEFF,
    FIELD_SAMPLE_AREA_ID, FIELD_SPECIES_ID, RECORDS, SESSION,
)


def _sample_context(parcel):
    grid = SampleGrid.objects.create(name='Ipso importer grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='1',
        lat=38.51234, lon=16.12345, r_m=15,
    )
    survey = Survey.objects.create(name='Ipso importer survey', sample_grid=grid)
    return survey, area


def _sample_record(area, species, **overrides):
    record = {
        FIELD_SAMPLE_AREA_ID: area.id,
        FIELD_PARCEL_ID: area.parcel_id,
        FIELD_SPECIES_ID: species.id,
        FIELD_NUMBER: 1,
        FIELD_DATE: '2026-06-17',
        FIELD_D_CM: 42,
        FIELD_H_M: '22',
    }
    record.update(overrides)
    return record


def _free_record(parcel, species, number):
    return {
        FIELD_PARCEL_ID: parcel.id,
        FIELD_SPECIES_ID: species.id,
        FIELD_NUMBER: number,
        FIELD_PRESERVED: False,
        FIELD_DATE: '2026-06-17',
        FIELD_D_CM: 42,
        FIELD_H_M: '22',
    }



@pytest.mark.django_db
def test_sample_import_rejects_non_array_records(parcels):
    survey, _ = _sample_context(parcels[0])

    rows, errors = sample_import_rows({RECORDS: {}}, survey)

    assert rows == []
    assert errors == [S.IPSO_ERR_IMPORT_RECORDS_ARRAY]


@pytest.mark.django_db
def test_sample_import_collects_lookup_and_shape_errors(parcels, species):
    survey, area = _sample_context(parcels[0])

    rows, errors = sample_import_rows({
        RECORDS: [
            'not-a-record',
            _sample_record(area, species[0], **{FIELD_SAMPLE_AREA_ID: 999999}),
            _sample_record(area, species[0], **{FIELD_PARCEL_ID: parcels[1].id}),
            _sample_record(area, species[0], **{FIELD_SPECIES_ID: 999999}),
            _sample_record(area, species[0], **{FIELD_DATE: 'not-a-date'}),
            _sample_record(area, species[0], **{FIELD_NUMBER: 'x'}),
            _sample_record(area, species[0], **{FIELD_NUMBER: 0}),
            _sample_record(area, species[0], **{FIELD_PRESSLER_COEFF: '0'}),
        ],
    }, survey)

    assert rows == []
    assert errors == [
        S.IPSO_ERR_IMPORT_RECORD_INVALID.format(1),
        S.IPSO_ERR_IMPORT_RECORD_AREA_NOT_FOUND.format(2),
        S.IPSO_ERR_IMPORT_RECORD_AREA_PARCEL_MISMATCH.format(3),
        S.IPSO_ERR_IMPORT_RECORD_SPECIES_NOT_FOUND.format(4),
        S.IPSO_ERR_IMPORT_RECORD_SAMPLE_FIELDS_INVALID.format(5),
        S.IPSO_ERR_RECORD_NUMBER_INVALID.format(6),
        S.IPSO_ERR_RECORD_NUMBER_POSITIVE.format(7),
        S.IPSO_ERR_IMPORT_RECORD_SAMPLE_FIELDS_INVALID.format(8),
    ]


@pytest.mark.django_db
def test_sample_import_defaults_pressler_and_rejects_date_conflicts(
        parcels, species):
    survey, area = _sample_context(parcels[0])
    Sample.objects.create(survey=survey, sample_area=area, date=date(2026, 6, 16))

    rows, errors = sample_import_rows({
        RECORDS: [
            _sample_record(area, species[0], **{FIELD_NUMBER: 1}),
            _sample_record(
                area, species[0],
                **{FIELD_NUMBER: 2, FIELD_DATE: '2026-06-18'},
            ),
        ],
    }, survey)

    assert rows == []
    assert errors == [
        S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
            1, area.parcel.region.name, area.parcel.name, area.number,
            '2026-06-16',
        ),
        S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
            2, area.parcel.region.name, area.parcel.name, area.number,
            '2026-06-16',
        ),
    ]


@pytest.mark.django_db
def test_sample_import_rejects_in_payload_date_conflict(parcels, species):
    survey, area = _sample_context(parcels[0])

    rows, errors = sample_import_rows({
        RECORDS: [
            _sample_record(area, species[0], **{FIELD_NUMBER: 1}),
            _sample_record(
                area, species[0],
                **{FIELD_NUMBER: 2, FIELD_DATE: '2026-06-18'},
            ),
        ],
    }, survey)

    assert len(rows) == 1
    assert errors == [
        S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
            2, area.parcel.region.name, area.parcel.name, area.number,
            '2026-06-17',
        ),
    ]


@pytest.mark.django_db
def test_free_survey_import_reserves_later_explicit_numbers(parcels, species):
    survey = Survey.objects.create(name='Mixed-number free survey')
    payload = {
        RECORDS: [
            _free_record(parcels[0], species[0], number)
            for number in (1, 2, 3, None, 4, 5)
        ],
    }

    rows, errors = free_survey_import_rows(payload, survey)

    assert errors == []
    assert [row[FIELD_NUMBER] for row in rows] == [1, 2, 3, 6, 4, 5]


@pytest.mark.django_db
def test_free_survey_import_still_rejects_explicit_duplicates(parcels, species):
    survey = Survey.objects.create(name='Duplicate-number free survey')
    payload = {
        RECORDS: [
            _free_record(parcels[0], species[0], number)
            for number in (4, 4)
        ],
    }

    rows, errors = free_survey_import_rows(payload, survey)

    assert [row[FIELD_NUMBER] for row in rows] == [4]
    assert errors == [S.IPSO_ERR_IMPORT_RECORD_SAMPLE_NUMBER_DUPLICATE.format(2)]


@pytest.mark.django_db
def test_free_survey_numbers_may_repeat_in_a_later_sample(parcels, species):
    survey = Survey.objects.create(name='Repeated-number free survey')
    previous_sample = Sample.objects.create(
        survey=survey, sample_area=None, date=date(2026, 6, 16),
    )
    TreeSample.objects.create(
        sample=previous_sample,
        tree=Tree.objects.create(species=species[0]),
        parcel=parcels[0], number=1, d_cm=30, h_m='20.00',
    )
    payload = {
        RECORDS: [
            _free_record(parcels[0], species[0], 1),
            _free_record(parcels[0], species[0], None),
        ],
    }

    rows, errors = free_survey_import_rows(payload, survey)

    assert errors == []
    assert [row[FIELD_NUMBER] for row in rows] == [1, 2]


def test_record_measurements_rejects_bad_and_non_positive_values():
    assert record_measurements({
        FIELD_DATE: 'bad-date', FIELD_D_CM: 42, FIELD_H_M: '22',
    }) is None
    assert record_measurements({
        FIELD_DATE: '2026-06-17', FIELD_D_CM: 0, FIELD_H_M: '22',
    }) is None
    assert record_measurements({
        FIELD_DATE: '2026-06-17', FIELD_D_CM: 42, FIELD_H_M: '0',
    }) is None
