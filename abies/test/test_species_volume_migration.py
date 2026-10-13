"""Narrow repair of missing estimates from the first catch-all deployment."""

import importlib
from decimal import Decimal
from types import SimpleNamespace

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.base.models import (
    DigestStatus, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    Sample, Survey, Tree, TreeMark, TreeSample, tree_mass_q,
)
from apps.base.tabacchi import tabacchi_volume_m3

pytestmark = pytest.mark.django_db


@pytest.fixture
def repair():
    state_apps = MigrationExecutor(connection).loader.project_state([
        ('base', '0027_other_species_volumes'),
    ]).apps
    migration = importlib.import_module('apps.base.migrations.0027_other_species_volumes')
    return lambda: migration.fill_other_species_volumes(
        state_apps, SimpleNamespace(connection=connection),
    )


@pytest.fixture
def item(parcels):
    plan = HarvestPlan.objects.create(name='Volume repair', year_start=2024, year_end=2034)
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2024,
        state=HarvestPlanItemState.OPEN, date_actual='2024-06-01',
        volume_actual_m3='12.345',
    )


@pytest.fixture
def observations(item):
    sample = Sample.objects.create(survey=Survey.objects.create(name='Volume repair'),
                                   date='2024-06-01')

    def create(Model, species, *, coppice=False, **fields):
        parent = {'harvest_plan_item': item, 'date': '2024-06-01'} if Model is TreeMark else {
            'sample': sample, 'number': TreeSample.objects.count() + 1,
        }
        return Model.objects.create(
            tree=Tree.objects.create(species=species, coppice=coppice),
            parcel=item.parcel, d_cm=30, h_m='20.00', operator='Original operator',
            import_fingerprint='original-fingerprint', **parent, **fields,
        )

    return create


@pytest.mark.parametrize('index', [4, 5])
def test_repair_matches_equations_and_current_bucket_density(
        repair, observations, species, item, index):
    bucket = species[index]
    bucket.density = Decimal('7.25')  # independent of the reference species
    bucket.save()
    mark = observations(TreeMark, bucket)
    row = observations(TreeSample, bucket, pressler_coeff='1.75', l10_mm=4)
    before = (mark.modified_at, row.modified_at)
    status = DigestStatus.objects.create(name='bosco', stale=False, dirty_seq=7)

    repair()

    expected = tabacchi_volume_m3(30, 20, bucket.common_name)
    for obj, timestamp in zip((mark, row), before):
        obj.refresh_from_db()
        assert obj.volume_m3 == expected
        assert obj.mass_q == tree_mass_q(expected, bucket.density)
        assert obj.version == 2 and obj.modified_at > timestamp
        assert obj.operator == 'Original operator'
        assert obj.import_fingerprint == 'original-fingerprint'
        assert obj.tree.species_id == bucket.id
    assert row.pressler_coeff == Decimal('1.75') and row.l10_mm == 4
    item.refresh_from_db()
    assert item.volume_marked_m3 == expected.quantize(Decimal('0.001'))
    assert item.volume_actual_m3 == Decimal('12.345')
    assert item.state == HarvestPlanItemState.OPEN
    assert item.date_actual.isoformat() == '2024-06-01'
    assert item.version == 2
    baseline = item.history.first()
    assert baseline.history_change_reason == 'initial-audit-baseline'
    assert baseline.volume_marked_m3 == item.volume_marked_m3
    status.refresh_from_db()
    assert status.stale and status.dirty_seq == 8

    repair()  # No re-estimation, version churn or repeated invalidation.
    mark.refresh_from_db()
    item.refresh_from_db()
    status.refresh_from_db()
    assert mark.version == item.version == 2
    assert item.history.count() == 2
    assert status.dirty_seq == 8


@pytest.mark.parametrize('Model', [TreeMark, TreeSample])
def test_repair_preserves_legacy_coppice_and_preexisting_values(
        repair, observations, species, item, Model):
    legacy = observations(Model, species[3])
    unsupported = observations(Model, species[2])
    coppice = observations(Model, species[5], coppice=True)
    captured = observations(Model, species[4], volume_m3='0.1234', mass_q='1.234')
    zero = observations(Model, species[5], volume_m3=0, mass_q=0)
    partial = observations(Model, species[4], volume_m3='0.1234')
    unknown = observations(Model, species[4], **(
        {'preserved_number': 99} if Model is TreeSample else {}
    ))
    unknown.h_m = None
    unknown.save()
    original = [(obj.volume_m3, obj.mass_q, obj.version, obj.modified_at)
                for obj in (legacy, unsupported, coppice, captured, zero, partial, unknown)]
    status = DigestStatus.objects.create(name='species', stale=False, dirty_seq=0)

    repair()

    for obj, values in zip((legacy, unsupported, coppice, captured, zero, partial, unknown), original):
        obj.refresh_from_db()
        assert (obj.volume_m3, obj.mass_q, obj.version, obj.modified_at) == (
            Decimal(str(values[0])) if values[0] is not None else None,
            Decimal(str(values[1])) if values[1] is not None else None,
            *values[2:],
        )
    item.refresh_from_db()
    assert item.version == 1 and item.volume_marked_m3 is None
    status.refresh_from_db()
    assert not status.stale and status.dirty_seq == 0


def test_parent_total_includes_preexisting_marks(repair, observations, species, item):
    old = observations(TreeMark, species[0], volume_m3='1.2345', mass_q='11.111')
    new = observations(TreeMark, species[4])

    repair()

    item.refresh_from_db()
    new.refresh_from_db()
    old.refresh_from_db()
    assert item.volume_marked_m3 == (old.volume_m3 + new.volume_m3).quantize(Decimal('0.001'))
    assert old.version == 1


def test_repair_leaves_empty_database_unchanged(repair):
    repair()
    assert not TreeMark.objects.exists() and not TreeSample.objects.exists()
