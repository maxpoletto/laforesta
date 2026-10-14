"""Invalidate stored parcel summaries when correcting coppice stump counts."""

import importlib
from types import SimpleNamespace

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.base.models import DigestStatus


pytestmark = pytest.mark.django_db


@pytest.fixture
def refresh_stump_digest():
    state_apps = MigrationExecutor(connection).loader.project_state([
        ('base', '0027_other_species_volumes'),
    ]).apps
    migration = importlib.import_module(
        'apps.base.migrations.0028_refresh_coppice_stump_digest',
    )
    return lambda: migration.mark_parcel_stump_digest_stale(
        state_apps, SimpleNamespace(connection=connection),
    )


@pytest.mark.parametrize('existing_stale', [None, False, True],
                         ids=['untracked', 'fresh', 'already-stale'])
def test_refresh_invalidates_only_parcels(
        refresh_stump_digest, parcels, existing_stale):
    if existing_stale is not None:
        DigestStatus.objects.create(
            name='parcels', stale=existing_stale, dirty_seq=7,
        )
    unrelated = DigestStatus.objects.create(name='surveys', stale=False, dirty_seq=4)

    refresh_stump_digest()

    status = DigestStatus.objects.get(name='parcels')
    assert status.stale
    assert status.dirty_seq == (1 if existing_stale is None else 8)
    unrelated.refresh_from_db()
    assert not unrelated.stale
    assert unrelated.dirty_seq == 4


def test_refresh_does_not_register_a_digest_for_an_empty_install(refresh_stump_digest):
    refresh_stump_digest()

    assert not DigestStatus.objects.filter(name='parcels').exists()
