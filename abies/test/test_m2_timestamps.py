"""The row clock records partial writes without becoming an edit token."""
from datetime import timedelta
import json

import pytest
from django.utils import timezone
from django.test import Client

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


def test_parcel_metadata_endpoint_advances_clock_once_and_stale_edit_is_noop(
        parcels, writer_user, monkeypatch):
    parcel = parcels[0]
    client = Client()
    client.force_login(writer_user)
    before = (parcel.created_at, parcel.modified_at, parcel.history.count())
    later = before[1] + timedelta(minutes=1)
    monkeypatch.setattr(timezone, 'now', lambda: later)
    body = dict(row_id=parcel.pk, version=parcel.version, eclass_id=parcel.eclass_id,
                area_ha='12.5', location_name='Updated location')
    response = client.post('/api/bosco/parcels/metadata/save/',
                           json.dumps(body), content_type='application/json')
    assert response.status_code == 200, response.content
    parcel.refresh_from_db()
    assert parcel.modified_at == later and parcel.created_at == before[0]
    assert parcel.version == 2 and parcel.history.count() == before[2] + 1
    assert client.post('/api/bosco/parcels/metadata/save/',
        json.dumps(body), content_type='application/json').status_code == 400
    parcel.refresh_from_db()
    assert parcel.modified_at == later and parcel.history.count() == before[2] + 1
