from datetime import timedelta
from decimal import Decimal

import pytest
from django.utils import timezone

from apps.base import csv_parcels

pytestmark = pytest.mark.django_db


def test_parcel_csv_clock_and_unchanged_retry(parcels, monkeypatch):
    parcel = parcels[0]
    before = (parcel.created_at, parcel.modified_at, parcel.history.count())
    later = before[1] + timedelta(minutes=1)
    monkeypatch.setattr(timezone, 'now', lambda: later)
    row = {field: getattr(parcel, field) for field in csv_parcels.PARCEL_UPDATE_FIELDS}
    row.update(region=parcel.region, name=parcel.name, area_ha=Decimal('12.50'))
    assert csv_parcels.update_existing([row]) == 1
    parcel.refresh_from_db()
    assert parcel.modified_at == later and parcel.created_at == before[0]
    assert parcel.version == 2 and parcel.history.count() == before[2] + 1
    assert csv_parcels.update_existing([row]) == 0
    parcel.refresh_from_db()
    assert parcel.modified_at == later and parcel.history.count() == before[2] + 1
