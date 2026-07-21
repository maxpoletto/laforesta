"""Tests for the live-DB parcel CSV update command."""

import gzip
import json
from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.base.models import DigestStatus, Parcel
from config import strings as S
from config.constants import COLUMNS, DIGEST_PARCELS, ROWS


HEADER = ','.join([
    S.CSV_COL_REGION, S.CSV_COL_CLASS, S.CSV_COL_PARCEL,
    S.CSV_COL_AREA_HA, S.CSV_COL_AVE_AGE, S.CSV_COL_LOCATION,
    S.CSV_COL_ALT_MIN, S.CSV_COL_ALT_MAX, S.CSV_COL_ASPECT,
    S.CSV_COL_GRADE_PCT, S.CSV_COL_GEO_DESC, S.CSV_COL_VEG_DESC,
    S.CSV_COL_CUTTING_PLAN, S.CSV_COL_INTERVAL, S.CSV_COL_STANDARDS,
])


def _write_csv(path, rows):
    path.write_text(HEADER + '\n' + '\n'.join(rows) + '\n')


@pytest.mark.django_db
def test_update_parcels_from_csv_dry_run_does_not_write(tmp_path, parcels):
    csv_path = tmp_path / 'particelle.csv'
    _write_csv(csv_path, [
        'Capistrano,A,1,11.5,45,Costa,700,900,N,10,Geo,Veg,Piano,,',
    ])
    out = StringIO()

    call_command('update_parcels_from_csv', csv_path, stdout=out)

    parcels[0].refresh_from_db()
    assert 'Would update 1 parcel row(s)' in out.getvalue()
    assert parcels[0].area_ha == Decimal('10.50')
    assert parcels[0].version == 1


@pytest.mark.django_db
def test_update_parcels_from_csv_apply_updates_existing_rows(
        tmp_path, parcels, settings):
    csv_path = tmp_path / 'particelle.csv'
    _write_csv(csv_path, [
        'Capistrano,A,1,11.5,45,Costa,700,900,N,10,Geo,Veg,Piano,,',
    ])
    out = StringIO()

    call_command('update_parcels_from_csv', csv_path, '--apply', stdout=out)

    parcels[0].refresh_from_db()
    assert 'Updated 1 parcel row(s)' in out.getvalue()
    assert 'Regenerated parcels digest.' in out.getvalue()
    assert parcels[0].area_ha == Decimal('11.50')
    assert parcels[0].ave_age == 45
    assert parcels[0].location_name == 'Costa'
    assert parcels[0].altitude_min_m == 700
    assert parcels[0].altitude_max_m == 900
    assert parcels[0].aspect == 'N'
    assert parcels[0].grade_pct == 10
    assert parcels[0].desc_geo == 'Geo'
    assert parcels[0].desc_veg == 'Veg'
    assert parcels[0].cutting_plan == 'Piano'
    assert parcels[0].version == 2
    assert DigestStatus.objects.get(name=DIGEST_PARCELS).stale is False
    with gzip.open(settings.DIGEST_DIR / f'{DIGEST_PARCELS}.json.gz', 'rt') as fh:
        digest = json.load(fh)
    assert S.COL_CUTTING_PLAN in digest[COLUMNS]
    row = next(r for r in digest[ROWS] if r[0] == parcels[0].id)
    assert row[digest[COLUMNS].index(S.COL_CUTTING_PLAN)] == 'Piano'


@pytest.mark.django_db
def test_update_parcels_from_csv_apply_regenerates_digest_when_unchanged(
        tmp_path, parcels):
    csv_path = tmp_path / 'particelle.csv'
    _write_csv(csv_path, [','.join(['Capistrano', 'A', '1', '10.5'] + [''] * 11)])
    out = StringIO()

    call_command('update_parcels_from_csv', csv_path, '--apply', stdout=out)

    assert 'Updated 0 parcel row(s)' in out.getvalue()
    assert 'Regenerated parcels digest.' in out.getvalue()
    assert DigestStatus.objects.get(name=DIGEST_PARCELS).stale is False


@pytest.mark.django_db
def test_update_parcels_from_csv_refuses_missing_parcels(tmp_path, parcels):
    csv_path = tmp_path / 'particelle.csv'
    _write_csv(csv_path, [
        'Capistrano,A,missing,11.5,45,Costa,700,900,N,10,Geo,Veg,Piano,,',
    ])

    with pytest.raises(CommandError, match='not found in the DB'):
        call_command('update_parcels_from_csv', csv_path, '--apply')

    assert Parcel.objects.filter(name='missing').exists() is False


@pytest.mark.django_db
def test_update_parcels_from_csv_updates_coppice_metadata(
        tmp_path, regions, eclasses):
    parcel = Parcel.objects.create(
        name='C1', region=regions[0], eclass=eclasses[2],
        area_ha=Decimal('1.00'), intervention_interval=18,
        standards_per_ha=75,
    )
    csv_path = tmp_path / 'particelle.csv'
    _write_csv(csv_path, [
        'Capistrano,F,C1,1.0,,Ceduo,,,,,Geo,Veg,Piano ceduo,20,80',
    ])

    call_command('update_parcels_from_csv', csv_path, '--apply')

    parcel.refresh_from_db()
    assert parcel.intervention_interval == 20
    assert parcel.standards_per_ha == 80
    assert parcel.cutting_plan == 'Piano ceduo'
