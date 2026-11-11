"""Existing forecast files must gain the selected-plan/calendar snapshot on upgrade."""
import importlib
from types import SimpleNamespace

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.base.models import DigestStatus, HarvestPlan

pytestmark = pytest.mark.django_db


@pytest.fixture
def refresh_calendar_digest():
    state_apps = MigrationExecutor(connection).loader.project_state([
        ('base', '0032_species_equations'),
    ]).apps
    migration = importlib.import_module('apps.base.migrations.0033_refresh_calendar_digest')
    return lambda: migration.mark_calendar_digest_stale(
        state_apps, SimpleNamespace(connection=connection),
    )


@pytest.mark.parametrize('existing_stale', [None, False, True])
def test_upgrade_refreshes_forecast_snapshot_only(refresh_calendar_digest, existing_stale):
    HarvestPlan.objects.create(name='Existing plan', year_start=2026, year_end=2040)
    if existing_stale is not None:
        DigestStatus.objects.create(name='future_production', stale=existing_stale, dirty_seq=7)
    unrelated = DigestStatus.objects.create(name='parcels', stale=False, dirty_seq=4)

    refresh_calendar_digest()

    status = DigestStatus.objects.get(name='future_production')
    assert status.stale
    assert status.dirty_seq == (1 if existing_stale is None else 8)
    unrelated.refresh_from_db()
    assert not unrelated.stale
    assert unrelated.dirty_seq == 4


def test_upgrade_refreshes_empty_untracked_snapshot_too(refresh_calendar_digest, reader_user):
    # generate_all can publish an empty file with no status and no plans.
    refresh_calendar_digest()

    status = DigestStatus.objects.get(name='future_production')
    assert status.stale
    assert status.dirty_seq == 1


def test_fresh_install_needs_no_upgrade_refresh(refresh_calendar_digest):
    refresh_calendar_digest()

    assert not DigestStatus.objects.filter(name='future_production').exists()
