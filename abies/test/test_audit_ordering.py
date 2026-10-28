"""Audit order and event identity must survive presentation and regeneration."""

import gzip
import importlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.base import digests
from apps.base.models import Crew, DigestStatus, HarvestPlan, User
from config import strings as S
from config.constants import COLUMNS, ROWS


pytestmark = pytest.mark.django_db


def audit(settings):
    digests.generate_audit()
    with gzip.open(settings.DIGEST_DIR / 'audit.json.gz', 'rt') as source:
        data = json.load(source)
    return [dict(zip(data[COLUMNS], row)) for row in data[ROWS]]


def set_event_time(obj, instant):
    entry = obj.history.first()
    obj.history.filter(history_id=entry.history_id).update(history_date=instant)
    return entry.history_id


@pytest.mark.parametrize('difference', [timedelta(seconds=58), timedelta(microseconds=100)])
def test_later_edit_precedes_insert_in_same_minute(settings, difference):
    early = datetime(2026, 10, 1, 12, 0, 1, 100, tzinfo=timezone.utc)
    crew = Crew.objects.create(name='Before')
    set_event_time(crew, early)
    crew.name = 'After'
    crew.save()
    set_event_time(crew, early + difference)

    rows = audit(settings)

    assert [row[S.COL_ACTION] for row in rows] == [S.AUDIT_UPDATE, S.AUDIT_INSERT]


def test_newest_order_spans_different_models(settings):
    early = datetime(2026, 10, 1, 12, 0, 1, tzinfo=timezone.utc)
    crew = Crew.objects.create(name='Older crew')
    set_event_time(crew, early)
    plan = HarvestPlan.objects.create(name='Newer plan', year_start=2026, year_end=2026)
    set_event_time(plan, early + timedelta(seconds=1))

    assert [row[S.COL_TABLE] for row in audit(settings)] == [S.TABLE_HARVEST_PLAN, S.TABLE_CREW]


def test_equal_time_ties_and_ids_do_not_depend_on_config_order(settings, monkeypatch):
    instant = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    crew = Crew.objects.create(name='Crew')
    crew_event = set_event_time(crew, instant)
    plan = HarvestPlan.objects.create(name='Plan', year_start=2026, year_end=2026)
    plan_event = set_event_time(plan, instant)
    before = audit(settings)
    configs = digests._audit_configs()
    monkeypatch.setattr(digests, '_audit_configs', lambda: list(reversed(configs)))

    assert audit(settings) == before
    assert {row['row_id'] for row in before} == {
        f'base.crew:{crew_event}', f'base.harvestplan:{plan_event}',
    }


def test_event_id_is_preserved_when_an_older_event_is_added(settings):
    instant = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    plan = HarvestPlan.objects.create(name='Existing event', year_start=2026, year_end=2026)
    set_event_time(plan, instant)
    original_id = audit(settings)[0]['row_id']
    crew = Crew.objects.create(name='Earlier event in another table')
    set_event_time(crew, instant - timedelta(days=1))

    current = next(row for row in audit(settings) if row[S.COL_TABLE] == S.TABLE_HARVEST_PLAN)
    assert current['row_id'] == original_id


def test_wire_timestamp_keeps_utc_microseconds_and_local_display_seconds(settings):
    settings.TIME_ZONE = 'Europe/Rome'
    crew = Crew.objects.create(name='Precise event')
    set_event_time(crew, datetime(2026, 1, 1, 12, 0, 3, 123456, tzinfo=timezone.utc))

    row = audit(settings)[0]

    assert row[S.COL_TIMESTAMP] == '2026-01-01T12:00:03.123456+00:00'
    assert row['timestamp_label'] == '2026-01-01 13:00:03'


def test_order_survives_repeated_local_hour_at_daylight_saving_end(settings):
    settings.TIME_ZONE = 'Europe/Rome'
    early = datetime(2026, 10, 25, 0, 50, tzinfo=timezone.utc)
    crew = Crew.objects.create(name='Earlier summer-time event')
    set_event_time(crew, early)
    plan = HarvestPlan.objects.create(name='Later winter-time event', year_start=2026, year_end=2026)
    set_event_time(plan, early + timedelta(minutes=20))

    rows = audit(settings)

    assert [row[S.COL_TABLE] for row in rows] == [S.TABLE_HARVEST_PLAN, S.TABLE_CREW]


def test_upgrade_invalidates_existing_audit_without_new_history(settings):
    crew = Crew.objects.create(name='Existing history')
    before = list(crew.history.values())
    DigestStatus.objects.update_or_create(name='audit', defaults={'stale': False, 'dirty_seq': 7})
    unrelated = DigestStatus.objects.create(name='parcels', stale=False, dirty_seq=3)
    state = MigrationExecutor(connection).loader.project_state([('base', '0029_usednonce_request_hash')]).apps
    migration = importlib.import_module('apps.base.migrations.0030_refresh_audit_ordering')

    migration.mark_audit_stale(state, SimpleNamespace(connection=connection))

    status = DigestStatus.objects.get(name='audit')
    assert status.stale and status.dirty_seq == 8
    unrelated.refresh_from_db()
    assert not unrelated.stale and unrelated.dirty_seq == 3
    assert list(crew.history.values()) == before


def test_upgrade_registers_an_untracked_digest_for_existing_users():
    User.objects.create_user(username='existing-admin')
    DigestStatus.objects.filter(name='audit').delete()
    state = MigrationExecutor(connection).loader.project_state([('base', '0029_usednonce_request_hash')]).apps
    migration = importlib.import_module('apps.base.migrations.0030_refresh_audit_ordering')
    migration.mark_audit_stale(state, SimpleNamespace(connection=connection))
    status = DigestStatus.objects.get(name='audit')
    assert status.stale and status.dirty_seq == 1


def test_upgrade_leaves_a_fresh_installation_unchanged():
    assert not User.objects.exists()
    DigestStatus.objects.filter(name='audit').delete()
    state = MigrationExecutor(connection).loader.project_state([('base', '0029_usednonce_request_hash')]).apps
    migration = importlib.import_module('apps.base.migrations.0030_refresh_audit_ordering')
    migration.mark_audit_stale(state, SimpleNamespace(connection=connection))
    assert not DigestStatus.objects.filter(name='audit').exists()
