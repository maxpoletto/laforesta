"""Unit tests for the load_canonical_items loader in apps/piano_di_taglio/csv_plan.py."""

from decimal import Decimal

import pytest

from apps.base import csv_io
from apps.base.models import Eclass, HarvestPlan, HarvestPlanItem, Parcel, Region
from apps.piano_di_taglio import csv_plan
from config import strings as S
from config.constants import (
    FIELD_DAMAGED, FIELD_INTERVENTION_AREA_HA, FIELD_PARCEL_ID,
    FIELD_PERIOD_Y, FIELD_PSR, FIELD_REGION_ID, FIELD_UNHEALTHY,
    FIELD_VOLUME_PLANNED_M3, FIELD_YEAR_PLANNED,
)


# Header for the unified harvest_plan_items.csv
ITEMS_HEADER = (
    f'{S.CSV_COL_PLAN},{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},'
    f'{S.CSV_COL_YEAR},{S.CSV_COL_HARVEST_M3},{S.CSV_COL_SURFACE_HA},'
    f'{S.CSV_COL_PERIOD_Y},{S.CSV_COL_HARVEST_DAMAGED},'
    f'{S.CSV_COL_HARVEST_UNHEALTHY},{S.CSV_COL_HARVEST_PSR},{S.CSV_COL_NOTE}'
)


def _reader(text):
    return csv_io.read(text)


def _make_plans(names):
    """Create HarvestPlan objects for the given names."""
    return {
        name: HarvestPlan.objects.create(
            name=name, year_start=2025, year_end=2030)
        for name in names
    }


@pytest.mark.django_db
def test_load_items_highforest(parcels, species):
    """A fustaia row (non-coppice parcel) creates one HarvestPlanItem."""
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,1,2027,250,,,,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert errors == [], errors
    assert n == 1
    item = HarvestPlanItem.objects.get(harvest_plan=plans['Piano A'])
    assert item.parcel == parcels[0]
    assert item.year_planned == 2027
    assert item.volume_planned_m3 == Decimal('250')
    assert item.damaged is False
    assert item.psr is False


@pytest.mark.django_db
def test_load_items_coppice(regions, eclasses, species):
    """A coppice row (coppice eclass) creates a ceduo HarvestPlanItem with area+turno."""
    coppice_parcel = Parcel.objects.create(
        name='9', region=regions[0], eclass=eclasses[2], area_ha=Decimal('5.0')
    )
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        f'Piano A,{regions[0].name},9,2028,,3.5,18,0,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert errors == [], errors
    assert n == 1
    item = HarvestPlanItem.objects.get(harvest_plan=plans['Piano A'])
    assert item.parcel == coppice_parcel
    assert item.year_planned == 2028
    assert item.intervention_area_ha == Decimal('3.5')


@pytest.mark.django_db
def test_load_items_region_wide(parcels, species):
    """A region-wide row (blank Particella + Danneggiato=1) creates a region-scoped item."""
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,,2026,100,,,1,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert errors == [], errors
    assert n == 1
    item = HarvestPlanItem.objects.get(harvest_plan=plans['Piano A'])
    assert item.parcel is None
    assert item.region is not None
    assert item.region.name == 'Capistrano'
    assert item.damaged is True
    assert item.year_planned == 2026


@pytest.mark.django_db
def test_load_items_optional_volume_blank_ok(parcels, species):
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,1,2027,,,,0,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert errors == [], errors
    assert n == 1
    item = HarvestPlanItem.objects.get(harvest_plan=plans['Piano A'])
    assert item.volume_planned_m3 is None


@pytest.mark.django_db
def test_load_items_highforest_invalid_volume_rejected(parcels, species):
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,1,2027,nope,,,0,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert n == 0
    assert len(errors) == 1
    assert S.CSV_COL_HARVEST_M3 in errors[0]
    assert 'nope' in errors[0]
    assert HarvestPlanItem.objects.count() == 0


@pytest.mark.django_db
def test_load_items_region_wide_invalid_volume_rejected(parcels, species):
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,,2027,nope,,,1,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert n == 0
    assert len(errors) == 1
    assert S.CSV_COL_HARVEST_M3 in errors[0]
    assert 'nope' in errors[0]
    assert HarvestPlanItem.objects.count() == 0


@pytest.mark.django_db
def test_load_items_coppice_invalid_surface_rejected(regions, eclasses, species):
    Parcel.objects.create(
        name='9', region=regions[0], eclass=eclasses[2], area_ha=Decimal('5.0')
    )
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        f'Piano A,{regions[0].name},9,2028,,nope,18,0,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert n == 0
    assert len(errors) == 1
    assert S.CSV_COL_SURFACE_HA in errors[0]
    assert 'nope' in errors[0]
    assert HarvestPlanItem.objects.count() == 0


@pytest.mark.django_db
def test_load_items_two_plans(parcels, species):
    """Rows belonging to two different Piano values create items under each plan."""
    plans = _make_plans(['Piano A', 'Piano B'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,1,2027,200,,,,0,0,',
        'Piano B,Capistrano,2,2028,150,,,,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert errors == [], errors
    assert n == 2
    assert HarvestPlanItem.objects.filter(harvest_plan=plans['Piano A']).count() == 1
    assert HarvestPlanItem.objects.filter(harvest_plan=plans['Piano B']).count() == 1


@pytest.mark.django_db
def test_load_items_unknown_plan_flagged(parcels, species):
    """An unknown Piano value produces an error."""
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano X,Capistrano,1,2027,200,,,,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert n == 0
    assert len(errors) == 1
    assert 'Piano X' in errors[0]


@pytest.mark.django_db
def test_load_items_flags_parsed(parcels, species):
    """Danneggiato/Fitosanitario/PSR boolean columns are parsed correctly.

    Header column positions (1-based):
    Piano(1) Compresa(2) Particella(3) Anno(4) Prelievo(5) Superficie(6)
    Turno(7) Danneggiato(8) Fitosanitario(9) PSR(10) Note(11)
    """
    plans = _make_plans(['Piano A'])
    # After Prelievo (pos5), leave Superficie(6)/Turno(7)/Danneggiato(8) blank,
    # then supply Fitosanitario(9) / PSR(10) / Note(11).
    csv_text = '\n'.join([
        ITEMS_HEADER,
        # Particella=1: Danneggiato='', Fitosanitario=0, PSR=0 → all false
        'Piano A,Capistrano,1,2027,200,,,,0,0,',
        # Particella=2: Danneggiato='', Fitosanitario=0, PSR=1 → psr only
        'Piano A,Capistrano,2,2028,150,,,,0,1,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert errors == [], errors
    assert n == 2
    items = {i.parcel.name: i
             for i in HarvestPlanItem.objects.filter(harvest_plan=plans['Piano A'])}
    assert items['1'].damaged is False
    assert items['1'].unhealthy is False
    assert items['1'].psr is False
    assert items['2'].psr is True
    assert items['2'].unhealthy is False


@pytest.mark.django_db
def test_load_items_unknown_parcel_flagged(parcels, species):
    """Unknown parcel produces an error."""
    plans = _make_plans(['Piano A'])
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,999,2027,200,,,,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert n == 0
    assert len(errors) == 1


@pytest.mark.django_db
def test_load_items_region_wide_missing_flag_includes_row_number(parcels, species):
    """A region-wide row with no Danneggiato/Fitosanitario flag produces an error
    that includes the 1-based row number so operators can locate the offending line."""
    plans = _make_plans(['Piano A'])
    # Row 2 (header is row 1): blank Particella, both Danneggiato and Fitosanitario are 0.
    csv_text = '\n'.join([
        ITEMS_HEADER,
        'Piano A,Capistrano,,2026,100,,,0,0,0,',
    ])
    reader = _reader(csv_text)
    idx = csv_plan.db_indexes()
    n, errors = csv_plan.load_canonical_items(reader, idx, plans)
    assert n == 0
    assert len(errors) == 1
    # The error must mention the row number (2) so the operator can find the line.
    assert '2' in errors[0]
