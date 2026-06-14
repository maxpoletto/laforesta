"""Unit tests for the harvest-plan CSV import core (apps/piano_di_taglio/csv_plan)."""

import io

import pytest

from apps.base.models import HarvestPlanItem, ParcelPlanDetail
from apps.piano_di_taglio import csv_plan
from config import strings as S


FUSTAIA_CSV = (
    'Compresa,Particella,Anno,Prelievo (m³)\r\n'
    'Capistrano,1,2027,250\r\n'
)
CEDUO_CSV = (
    'Anno,Compresa,Particella,Superficie intervento (ha),Turno (a),Note\r\n'
    '2028,Capistrano,1,2.5,18,Cont.\r\n'
)


@pytest.mark.django_db
def test_db_indexes_has_parcels_and_regions(parcels):
    idx = csv_plan.db_indexes()
    assert ('capistrano', '1') in idx.parcels
    assert 'capistrano' in idx.regions


def _fustaia(csv_text):
    return csv_plan.read_optional(
        io.BytesIO(csv_text.encode('utf-8')),
        csv_plan.HIGHFOREST_REQUIRED, csv_plan.HIGHFOREST_OPTIONAL)


def _ceduo(csv_text):
    return csv_plan.read_optional(
        io.BytesIO(csv_text.encode('utf-8')),
        csv_plan.COPPICE_REQUIRED, csv_plan.COPPICE_OPTIONAL)


@pytest.mark.django_db
def test_parse_fustaia_happy(parcels):
    idx = csv_plan.db_indexes()
    errors = []
    parsed = csv_plan.parse_fustaia_rows(_fustaia(FUSTAIA_CSV), idx.parcels, idx.regions, errors)
    assert errors == []
    assert len(parsed) == 1
    assert parsed[0][csv_plan.FIELD_YEAR_PLANNED] == 2027


@pytest.mark.django_db
def test_parse_fustaia_unknown_parcel_flagged(parcels):
    idx = csv_plan.db_indexes()
    errors = []
    bad = 'Compresa,Particella,Anno,Prelievo (m³)\r\nCapistrano,999,2027,250\r\n'
    parsed = csv_plan.parse_fustaia_rows(_fustaia(bad), idx.parcels, idx.regions, errors)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_parse_fustaia_whole_region_with_flag(parcels):
    idx = csv_plan.db_indexes()
    errors = []
    csv_in = (
        f'Compresa,Particella,Anno,Prelievo (m³),Note\r\n'
        f'Capistrano,{S.PARCEL_WHOLE_REGION_MARK},2027,250,{S.FLAG_DAMAGED}\r\n'
    )
    parsed = csv_plan.parse_fustaia_rows(_fustaia(csv_in), idx.parcels, idx.regions, errors)
    assert errors == []
    assert len(parsed) == 1
    assert parsed[0][csv_plan.FIELD_PARCEL_ID] is None
    assert parsed[0][csv_plan.FIELD_REGION_ID] is not None
    assert parsed[0][csv_plan.FIELD_DAMAGED] is True


@pytest.mark.django_db
def test_parse_fustaia_whole_region_without_flag_flagged(parcels):
    idx = csv_plan.db_indexes()
    errors = []
    csv_in = (
        f'Compresa,Particella,Anno,Prelievo (m³),Note\r\n'
        f'Capistrano,{S.PARCEL_WHOLE_REGION_MARK},2027,250,\r\n'
    )
    parsed = csv_plan.parse_fustaia_rows(_fustaia(csv_in), idx.parcels, idx.regions, errors)
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_apply_creates_plan_items_and_detail(parcels):
    idx = csv_plan.db_indexes()
    errors = []
    fustaia = csv_plan.parse_fustaia_rows(_fustaia(FUSTAIA_CSV), idx.parcels, idx.regions, errors)
    ceduo = csv_plan.parse_ceduo_rows(_ceduo(CEDUO_CSV), idx.parcels, errors)
    assert errors == []
    plan, n_items = csv_plan.apply(
        target_plan=None, name='Unit plan', description='',
        fustaia_parsed=fustaia, ceduo_parsed=ceduo)
    assert n_items == 2
    assert plan.year_start == 2027 and plan.year_end == 2028
    assert HarvestPlanItem.objects.filter(harvest_plan=plan).count() == 2
    assert ParcelPlanDetail.objects.get(harvest_plan=plan).harvest_detail.interval == 18
