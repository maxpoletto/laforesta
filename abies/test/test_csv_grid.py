"""Unit tests for the sample-grid CSV import core (apps/campionamenti/csv_grid)."""

import pytest

from apps.base import csv_io
from apps.base.models import SampleArea, SampleGrid
from apps.campionamenti import csv_grid
from config import strings as S


def test_resolve_columns_canonical_headers():
    fieldnames = ['Compresa', 'Particella', 'Area saggio', 'Lon', 'Lat',
                  'Quota', 'Raggio']
    cols, missing = csv_grid.resolve_columns(fieldnames)
    assert missing == []
    assert cols[S.CSV_COL_REGION] == 'Compresa'
    assert cols[S.CSV_COL_RADIUS] == 'Raggio'


def test_resolve_columns_missing_required_reported():
    _, missing = csv_grid.resolve_columns(['Compresa', 'Particella'])
    # Required columns absent from the header are reported; Raggio is optional.
    assert S.CSV_COL_LAT in ' '.join(missing) or any('Lat' in m for m in missing)
    assert missing  # non-empty


def _reader(csv_text):
    return csv_io.read(csv_text)


@pytest.mark.django_db
def test_validate_rows_happy_path(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-happy')
    compresa = parcel.region.name
    particella = parcel.name
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{compresa},{particella},10,16.1,38.5,500,15\n'
    )
    cols, missing = csv_grid.resolve_columns(reader.fieldnames)
    assert not missing
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert errors == []
    assert len(parsed) == 1
    assert parsed[0][csv_grid.FIELD_NUMBER] == '10'
    assert parsed[0][csv_grid.FIELD_R_M] == 15


@pytest.mark.django_db
def test_validate_rows_blank_radius_flagged(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-blank-radius')
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{parcel.region.name},{parcel.name},10,16.1,38.5,500,\n'
    )
    cols, missing = csv_grid.resolve_columns(reader.fieldnames)
    assert not missing
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_unparseable_radius_flagged(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-bad-radius')
    compresa = parcel.region.name
    particella = parcel.name
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{compresa},{particella},10,16.1,38.5,500,abc\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert parsed == []
    assert len(errors) == 1


def test_resolve_columns_radius_now_required():
    _, missing = csv_grid.resolve_columns(
        ['Compresa', 'Particella', 'Area saggio', 'Lon', 'Lat', 'Quota'])
    assert any('Raggio' in m for m in missing)


@pytest.mark.django_db
def test_validate_rows_invalid_quota_flagged(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-bad-quota')
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{parcel.region.name},{parcel.name},10,16.1,38.5,abc,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_blank_quota_ok(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-blank-quota')
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{parcel.region.name},{parcel.name},10,16.1,38.5,,15\n'
    )
    cols, missing = csv_grid.resolve_columns(reader.fieldnames)
    assert not missing
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert errors == []
    assert parsed[0][csv_grid.FIELD_ALTITUDE] is None


@pytest.mark.django_db
def test_validate_rows_unknown_parcel_flagged(parcels):
    grid = SampleGrid.objects.create(name='vr-bad-parcel')
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        'Nowhere,999,10,16.1,38.5,500,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_duplicate_within_upload_flagged(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-dup')
    compresa = parcel.region.name
    particella = parcel.name
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{compresa},{particella},10,16.1,38.5,500,15\n'
        f'{compresa},{particella},10,16.2,38.6,510,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert len(parsed) == 1   # first row accepted
    assert len(errors) == 1   # second row flagged as duplicate


@pytest.mark.django_db
def test_validate_rows_existing_db_key_flagged(parcels):
    """A row matching an area already in the grid (DB) is flagged, not created."""
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='vr-existing-key')
    SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='10',
        lat=16.1, lon=38.5, r_m=12,
    )
    compresa = parcel.region.name
    particella = parcel.name
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{compresa},{particella},10,16.2,38.6,510,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_validate_rows_same_number_two_parcels_one_region_flagged(parcels):
    """Area numbers are unique per (grid, compresa): two parcels of the same
    region cannot share a number, even within one upload (region-keyed dedup)."""
    p1, p2 = parcels[0], parcels[1]          # both in regions[0]
    assert p1.region_id == p2.region_id
    grid = SampleGrid.objects.create(name='vr-region-dup')
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{p1.region.name},{p1.name},5,16.1,38.5,500,15\n'
        f'{p2.region.name},{p2.name},5,16.2,38.6,510,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert len(parsed) == 1   # first parcel's area accepted
    assert len(errors) == 1   # same number under a second parcel of the region flagged


@pytest.mark.django_db
def test_validate_rows_same_number_different_regions_ok(parcels):
    """The same area number under two different compresa is allowed."""
    p1, p3 = parcels[0], parcels[2]          # regions[0] and regions[1]
    assert p1.region_id != p3.region_id
    grid = SampleGrid.objects.create(name='vr-region-ok')
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{p1.region.name},{p1.name},5,16.1,38.5,500,15\n'
        f'{p3.region.name},{p3.name},5,16.2,38.6,510,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert errors == []
    assert len(parsed) == 2


@pytest.mark.django_db
def test_apply_creates_sample_areas(parcels):
    parcel = parcels[0]
    grid = SampleGrid.objects.create(name='apply-create')
    compresa = parcel.region.name
    particella = parcel.name
    reader = _reader(
        'Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio\n'
        f'{compresa},{particella},10,16.1,38.5,500,15\n'
    )
    cols, _ = csv_grid.resolve_columns(reader.fieldnames)
    parsed, errors = csv_grid.validate_rows(reader, cols, csv_grid.db_indexes(grid))
    assert errors == []
    csv_grid.apply(grid, parsed)
    area = SampleArea.objects.get(sample_grid=grid, number='10')
    assert area.r_m == 15
    assert area.altitude_m == 500
