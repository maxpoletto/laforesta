"""The row clock records partial writes without becoming an edit token."""
from datetime import timedelta

import pytest
from django.utils import timezone

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize('fields', [('name',), iter(['name'])])
def test_partial_save_updates_clock_without_changing_edit_token(parcels, monkeypatch, fields):
    parcel = parcels[0]
    before = (parcel.version, parcel.created_at, parcel.modified_at, parcel.history.count())
    later = before[2] + timedelta(days=1)
    monkeypatch.setattr(timezone, 'now', lambda: later)
    parcel.name = '1b'
    parcel.save(update_fields=fields)
    parcel.refresh_from_db()
    assert parcel.modified_at == later
    assert (parcel.version, parcel.created_at) == before[:2]
    assert parcel.history.count() == before[3] + 1


def test_empty_partial_save_is_noop(parcels, monkeypatch):
    parcel = parcels[0]
    before = (parcel.version, parcel.created_at, parcel.modified_at, parcel.history.count())
    monkeypatch.setattr(timezone, 'now', lambda: before[2] + timedelta(days=1))
    parcel.name = 'Must not persist'
    parcel.save(update_fields=[])
    parcel.refresh_from_db()
    assert parcel.name == '1'
    assert (parcel.version, parcel.created_at, parcel.modified_at, parcel.history.count()) == before
