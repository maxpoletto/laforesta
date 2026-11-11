"""Bosco's shared harvest-plan source, persistence and permissions."""

import json
from datetime import date

import pytest
from django.test import Client

from apps.base.models import DigestStatus, HarvestPlan
from config import strings as S
from config.constants import (
    DATA_ID, DIGEST_FUTURE_PRODUCTION, FIELD_DATA_IDS, FIELD_HARVEST_PLAN_ID,
    FIELD_NONCE, INVALIDATES, MESSAGE, PATCHES, RECORD, ROW_ID,
)


@pytest.fixture
def writer_client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


@pytest.fixture
def reader_client(reader_user):
    client = Client()
    client.force_login(reader_user)
    return client


def _post(client, url, data):
    return client.post(url, data=json.dumps(data), content_type='application/json')


class TestBoscoPlanSource:
    DATA_URL = '/api/bosco/plan-source/data/'
    SAVE_URL = '/api/bosco/plan-source/save/'

    def test_data_defaults_to_current_plan_with_greatest_end_year(
            self, writer_client, monkeypatch,
    ):
        monkeypatch.setattr(
            'apps.base.selectors.timezone.localdate',
            lambda: date(2026, 6, 10),
        )
        HarvestPlan.objects.create(name='Old', year_start=2010, year_end=2020)
        short = HarvestPlan.objects.create(
            name='Current short', year_start=2020, year_end=2030,
        )
        long = HarvestPlan.objects.create(
            name='Current long', year_start=2020, year_end=2035,
        )

        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data['active_id'] == long.id
        active_by_id = {p['id']: p['active'] for p in data['plans']}
        assert active_by_id[long.id] is True
        assert active_by_id[short.id] is False

    def test_data_prefers_runtime_active_plan(self, writer_client, monkeypatch):
        monkeypatch.setattr(
            'apps.base.selectors.timezone.localdate',
            lambda: date(2026, 6, 10),
        )
        manual = HarvestPlan.objects.create(
            name='Manual', year_start=2010, year_end=2020, active=True,
        )
        current = HarvestPlan.objects.create(
            name='Current', year_start=2020, year_end=2035,
        )

        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        data = resp.json()
        assert data['active_id'] == manual.id
        active_by_id = {p['id']: p['active'] for p in data['plans']}
        assert active_by_id[manual.id] is True
        assert active_by_id[current.id] is False

    def test_data_empty_is_robust(self, writer_client, db):
        resp = writer_client.get(self.DATA_URL)

        assert resp.status_code == 200
        assert resp.json() == {'active_id': None, 'plans': []}

    def test_save_sets_single_active_plan_and_marks_digests(self, writer_client):
        old = HarvestPlan.objects.create(
            name='Old active', year_start=2010, year_end=2020, active=True,
        )
        new = HarvestPlan.objects.create(
            name='New active', year_start=2021, year_end=2030,
        )

        resp = _post(writer_client, self.SAVE_URL, {
            FIELD_HARVEST_PLAN_ID: str(new.id), FIELD_NONCE: 'future-1',
        })

        assert resp.status_code == 200, resp.content
        assert resp.json()[MESSAGE] == S.BOSCO_PLAN_SOURCE_SAVED
        old.refresh_from_db()
        new.refresh_from_db()
        assert old.active is False
        assert new.active is True
        assert old.version == 2
        assert new.version == 2
        patches = {
            p[ROW_ID]: p[RECORD]
            for p in resp.json()[PATCHES]
            if p[DATA_ID] == 'harvest_plans'
        }
        assert patches[old.id][-1] is False
        assert patches[new.id][-1] is True
        for name in (
                DIGEST_FUTURE_PRODUCTION, 'harvest_plans', 'audit',
        ):
            assert DigestStatus.objects.get(name=name).stale is True

    def test_reader_can_read_but_cannot_save(self, reader_client, db):
        assert reader_client.get(self.DATA_URL).status_code == 200
        assert _post(reader_client, self.SAVE_URL, {}).status_code == 403

    def test_save_invalidates_loaded_bosco_data(self, writer_client):
        plan = HarvestPlan.objects.create(name='New source', year_start=2026, year_end=2040)
        response = _post(writer_client, self.SAVE_URL, {
            FIELD_HARVEST_PLAN_ID: plan.id, FIELD_NONCE: 'change-source',
        })
        assert response.status_code == 200
        assert response.json()[INVALIDATES][FIELD_DATA_IDS] == [DIGEST_FUTURE_PRODUCTION]

    def test_reselecting_saved_plan_does_not_recompute(self, writer_client):
        plan = HarvestPlan.objects.create(name='Unchanged', year_start=2026, year_end=2040, active=True)
        status = DigestStatus.objects.create(name=DIGEST_FUTURE_PRODUCTION, stale=False)
        response = _post(writer_client, self.SAVE_URL, {
            FIELD_HARVEST_PLAN_ID: plan.id, FIELD_NONCE: 'same-source',
        })
        assert response.status_code == 200
        plan.refresh_from_db()
        status.refresh_from_db()
        assert plan.version == 1
        assert plan.history.count() == 1
        assert status.stale is False

    def test_bosco_control_replaces_settings_section(self, writer_client):
        html = writer_client.get('/bosco').content.decode()
        assert 'data-role="plan-source"' in html
        assert html.index('data-role="plan-source"') < html.index('data-role="cadastral-toggle"')
        assert 'data-settings-section="future-production"' not in html
