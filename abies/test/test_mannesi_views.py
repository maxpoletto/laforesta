"""Tests for Squadre page endpoints."""

import json
from decimal import Decimal

import pytest
from django.test import Client

from apps.base.models import Crew
from apps.mannesi.models import ProductionCredit, WorkHour
from config import strings as S
from config.constants import (
    COLUMNS, DATA_ID, DELETES, FIELD_ACTIVE, FIELD_CREW_ID, FIELD_DATE,
    FIELD_HOURS, FIELD_MASS_Q, FIELD_NAME, FIELD_NONCE, FIELD_NOTE,
    FIELD_NOTES, HTML, MESSAGE, PATCHES, RECORD, ROWS, ROW_ID, STATUS,
    STATUS_CONFLICT, STATUS_NOT_FOUND, STATUS_VALIDATION_ERROR, VERSION,
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


def test_meta_returns_report_support_data(writer_client, products, species):
    resp = writer_client.get('/api/squadre/meta/')

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {'products', 'species'}
    assert 'Tronchi' in data['products']
    assert 'Abete' in data['species']


class TestCrews:
    def test_data(self, writer_client, crews):
        resp = writer_client.get('/api/squadre/crews/data/')
        assert resp.status_code == 200
        data = resp.json()
        assert data[COLUMNS][0] == ROW_ID
        assert len(data[ROWS]) == 2

    def test_form_add(self, writer_client, db):
        resp = writer_client.get('/api/squadre/crews/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[HTML]

    def test_form_edit(self, writer_client, crews):
        resp = writer_client.get(f'/api/squadre/crews/form/{crews[0].id}/')
        assert resp.status_code == 200
        assert crews[0].name in resp.json()[HTML]

    def test_save_create(self, writer_client, db):
        resp = _post(writer_client, '/api/squadre/crews/save/', {
            FIELD_NAME: 'Gamma', FIELD_NOTES: 'test notes', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data[DATA_ID] == 'crews'
        # _crew_row returns [id, name, notes, active]; name is at index 1.
        assert data[PATCHES][0][RECORD][1] == 'Gamma'
        assert Crew.objects.filter(name='Gamma').exists()

    def test_save_update(self, writer_client, crews):
        resp = _post(writer_client, '/api/squadre/crews/save/', {
            ROW_ID: str(crews[0].id), VERSION: str(crews[0].version),
            FIELD_NAME: 'Renamed', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 200
        crews[0].refresh_from_db()
        assert crews[0].name == 'Renamed'
        assert crews[0].version == 2

    def test_save_conflict(self, writer_client, crews):
        resp = _post(writer_client, '/api/squadre/crews/save/', {
            ROW_ID: str(crews[0].id), VERSION: '999',
            FIELD_NAME: 'Conflict', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT

    def test_save_non_numeric_version_conflicts(self, writer_client, crews):
        resp = _post(writer_client, '/api/squadre/crews/save/', {
            ROW_ID: str(crews[0].id), VERSION: 'not-a-number',
            FIELD_NAME: 'Conflict', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        crews[0].refresh_from_db()
        assert crews[0].name != 'Conflict'

    def test_save_validation_error(self, writer_client, db):
        resp = _post(writer_client, '/api/squadre/crews/save/', {
            FIELD_NAME: '', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

    def test_save_malformed_json_returns_validation_error(self, writer_client, db):
        resp = writer_client.post(
            '/api/squadre/crews/save/',
            data='{',
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert resp.json()[MESSAGE] == S.ERR_JSON_INVALID

    def test_reader_can_read_data(self, reader_client, crews):
        resp = reader_client.get('/api/squadre/crews/data/')
        assert resp.status_code == 200
        assert len(resp.json()[ROWS]) == 2

    def test_reader_cannot_open_form(self, reader_client, crews):
        resp = reader_client.get(f'/api/squadre/crews/form/{crews[0].id}/')
        assert resp.status_code == 403

    def test_reader_cannot_save(self, reader_client, db):
        resp = _post(reader_client, '/api/squadre/crews/save/', {
            FIELD_NAME: 'Gamma', FIELD_NOTES: '', FIELD_ACTIVE: 'true',
        })
        assert resp.status_code == 403


class TestHours:
    def test_form(self, writer_client, crews):
        resp = writer_client.get('/api/squadre/hours/form/')
        assert resp.status_code == 200
        assert '<form' in resp.json()[HTML]
        assert 'Alfa' in resp.json()[HTML]

    def test_save_create_and_data(self, writer_client, crews):
        resp = _post(writer_client, '/api/squadre/hours/save/', {
            FIELD_DATE: '2026-01-15', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '7,5', FIELD_NOTE: 'bosco', FIELD_NONCE: 'hours-1',
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data[DATA_ID] == 'squadre_hours'
        assert data[PATCHES][0][RECORD][3] == 'Alfa'
        assert data[PATCHES][0][RECORD][4] == 7.5
        obj = WorkHour.objects.get()
        assert obj.hours == Decimal('7.5')

        resp = writer_client.get('/api/squadre/hours/data/')
        assert resp.status_code == 200
        assert resp.json()[ROWS][0][0] == obj.id

    def test_save_validation_error(self, writer_client, crews):
        resp = _post(writer_client, '/api/squadre/hours/save/', {
            FIELD_DATE: '2026-01-15', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '0', FIELD_NOTE: '',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR

    def test_save_malformed_row_id_returns_not_found(self, writer_client, crews):
        resp = _post(writer_client, '/api/squadre/hours/save/', {
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
        resp = _post(writer_client, '/api/squadre/hours/save/', {
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
        resp = _post(writer_client, '/api/squadre/hours/delete/', {
            ROW_ID: str(obj.id), VERSION: str(obj.version), FIELD_NONCE: 'del-h',
        })
        assert resp.status_code == 200
        assert resp.json()[DELETES] == [{DATA_ID: 'squadre_hours', ROW_ID: obj.id}]
        assert not WorkHour.objects.filter(id=obj.id).exists()

    def test_delete_malformed_row_id_returns_not_found(self, writer_client):
        resp = _post(writer_client, '/api/squadre/hours/delete/', {
            ROW_ID: 'not-an-id', VERSION: '1', FIELD_NONCE: 'del-bad',
        })
        assert resp.status_code == 404
        assert resp.json()[STATUS] == STATUS_NOT_FOUND

    def test_reader_cannot_save(self, reader_client, crews):
        resp = _post(reader_client, '/api/squadre/hours/save/', {
            FIELD_DATE: '2026-01-15', FIELD_CREW_ID: str(crews[0].id),
            FIELD_HOURS: '2', FIELD_NOTE: '',
        })
        assert resp.status_code == 403


def test_credits_save_create(writer_client, crews):
    resp = _post(writer_client, '/api/squadre/credits/save/', {
        FIELD_DATE: '2026-01-20', FIELD_CREW_ID: str(crews[1].id),
        FIELD_MASS_Q: '12.25', FIELD_NOTE: 'anticipo', FIELD_NONCE: 'credit-1',
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data[DATA_ID] == 'squadre_credits'
    assert data[PATCHES][0][RECORD][3] == 'Beta'
    assert data[PATCHES][0][RECORD][4] == 12.25
    assert ProductionCredit.objects.get().mass_q == Decimal('12.25')
