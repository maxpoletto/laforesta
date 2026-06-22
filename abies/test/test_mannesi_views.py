"""Tests for Mannesi page endpoints."""

import json
from decimal import Decimal

import pytest
from django.test import Client

from apps.mannesi.models import ProductionCredit, WorkHour
from config.constants import (
    DATA_ID, DELETES, FIELD_CREW_ID, FIELD_DATE, FIELD_HOURS, FIELD_MASS_Q,
    FIELD_NONCE, FIELD_NOTE, HTML, PATCHES,
    RECORD, ROWS, ROW_ID, STATUS, STATUS_CONFLICT, STATUS_NOT_FOUND, STATUS_VALIDATION_ERROR,
    VERSION,
)


@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


@pytest.fixture
def reader_client(reader_user):
    c = Client()
    c.force_login(reader_user)
    return c


def _post(client, url, data):
    return client.post(url, data=json.dumps(data), content_type='application/json')


def test_meta_returns_receipt_support_data(writer_client, products, species):
    resp = writer_client.get('/api/mannesi/meta/')

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {'products', 'species'}
    assert 'Tronchi' in data['products']
    assert 'Abete' in data['species']


class TestHours:
    def test_form(self, writer_client, crews):
        resp = writer_client.get('/api/mannesi/hours/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[HTML]
        assert 'Alfa' in resp.json()[HTML]

    def test_save_create_and_data(self, writer_client, crews):
        resp = _post(writer_client, '/api/mannesi/hours/save/', {
            FIELD_DATE: '2026-01-15', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '7,5', FIELD_NOTE: 'bosco', FIELD_NONCE: 'hours-1',
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data[DATA_ID] == 'mannesi_hours'
        assert data[PATCHES][0][RECORD][3] == 'Alfa'
        assert data[PATCHES][0][RECORD][4] == 7.5
        obj = WorkHour.objects.get()
        assert obj.hours == Decimal('7.5')

        resp = writer_client.get('/api/mannesi/hours/data/')
        assert resp.status_code == 200
        assert resp.json()[ROWS][0][0] == obj.id

    def test_save_validation_error(self, writer_client, crews):
        resp = _post(writer_client, '/api/mannesi/hours/save/', {
            FIELD_DATE: '2026-01-15', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '0', FIELD_NOTE: '',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

    def test_save_malformed_row_id_returns_not_found(self, writer_client, crews):
        resp = _post(writer_client, '/api/mannesi/hours/save/', {
            ROW_ID: 'not-an-id', FIELD_DATE: '2026-01-15',
            FIELD_CREW_ID: str(crews[0].id), FIELD_HOURS: '5',
            FIELD_NOTE: '',
        })
        assert resp.status_code == 404
        assert resp.json()[STATUS] == STATUS_NOT_FOUND

    def test_update_conflict(self, writer_client, crews):
        obj = WorkHour.objects.create(
            date='2026-01-15', crew=crews[0], hours=Decimal('4'),
        )
        resp = _post(writer_client, '/api/mannesi/hours/save/', {
            ROW_ID: str(obj.id), VERSION: '999',
            FIELD_DATE: '2026-01-16', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '5', FIELD_NOTE: '',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        obj.refresh_from_db()
        assert str(obj.date) == '2026-01-15'

    def test_delete(self, writer_client, crews):
        obj = WorkHour.objects.create(
            date='2026-01-15', crew=crews[0], hours=Decimal('4'),
        )
        resp = _post(writer_client, '/api/mannesi/hours/delete/', {
            ROW_ID: str(obj.id), VERSION: str(obj.version), FIELD_NONCE: 'del-h',
        })
        assert resp.status_code == 200
        assert resp.json()[DELETES] == [{DATA_ID: 'mannesi_hours', ROW_ID: obj.id}]
        assert not WorkHour.objects.filter(id=obj.id).exists()

    def test_delete_malformed_row_id_returns_not_found(self, writer_client):
        resp = _post(writer_client, '/api/mannesi/hours/delete/', {
            ROW_ID: 'not-an-id', VERSION: '1', FIELD_NONCE: 'del-bad',
        })
        assert resp.status_code == 404
        assert resp.json()[STATUS] == STATUS_NOT_FOUND

    def test_reader_cannot_save(self, reader_client, crews):
        resp = _post(reader_client, '/api/mannesi/hours/save/', {
            FIELD_DATE: '2026-01-15', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '2', FIELD_NOTE: '',
        })
        assert resp.status_code == 403


def test_credits_save_create(writer_client, crews):
    resp = _post(writer_client, '/api/mannesi/credits/save/', {
        FIELD_DATE: '2026-01-20', FIELD_CREW_ID: str(crews[1].id),
        FIELD_MASS_Q: '12.25', FIELD_NOTE: 'anticipo', FIELD_NONCE: 'credit-1',
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data[DATA_ID] == 'mannesi_credits'
    assert data[PATCHES][0][RECORD][3] == 'Beta'
    assert data[PATCHES][0][RECORD][4] == 12.25
    assert ProductionCredit.objects.get().mass_q == Decimal('12.25')
