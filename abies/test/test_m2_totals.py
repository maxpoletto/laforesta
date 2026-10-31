"""Parent clocks, edit tokens and complete state/total history events."""
import json
from datetime import date
from decimal import Decimal

import pytest
from django.test import Client
from django.urls import reverse

from apps.base.models import HarvestPlan, HarvestPlanItem, HarvestPlanItemState, Tree, TreeMark
from apps.piano_di_taglio.mark_import import MarkImportRow, import_mark_rows, rematerialize_volume_marked
from apps.prelievi.models import Harvest
from apps.prelievi.views import _rematerialize_volume_actual

pytestmark = pytest.mark.django_db


@pytest.fixture
def item(parcels):
    plan = HarvestPlan.objects.create(name='M2 plan', year_start=2024, year_end=2030)
    return HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcels[0], year_planned=2026)


@pytest.fixture
def client(writer_user):
    result = Client()
    result.force_login(writer_user)
    return result


def mark_body(item, species):
    return dict(harvest_plan_item_id=item.pk, species_id=species.pk,
                parcel_id=item.parcel_id, d_cm=30, h_m=20, h_measured=True,
                date='2026-10-01', operator='M2 operator')


@pytest.mark.parametrize('entry', ['manual', 'import'])
def test_first_mark_has_one_complete_parent_history_event(item, species, client, entry):
    before = item.history.count()
    if entry == 'manual':
        response = client.post(reverse('piano-di-taglio-mark-save'),
            json.dumps(mark_body(item, species[0])), content_type='application/json')
        assert response.status_code == 200, response.content
    else:
        row = MarkImportRow(date(2026, 10, 1), item.parcel, species[0], 1,
            30, Decimal('20'), True, None, None, None, 'M2 operator', 'm2-mark')
        assert import_mark_rows(item, [row]).imported == 1
    item.refresh_from_db()
    assert item.version == 2 and item.state == HarvestPlanItemState.MARKED
    assert item.date_actual == date(2026, 10, 1) and item.volume_marked_m3 > 0
    assert item.history.count() == before + 1
    event = item.history.first()
    assert event.volume_marked_m3 == item.volume_marked_m3
    assert event.state == item.state and event.version == item.version


def add_mark(item, species, volume):
    return TreeMark.objects.create(harvest_plan_item=item, parcel=item.parcel,
        tree=Tree.objects.create(species=species), date='2026-10-01',
        d_cm=30, h_m=20, volume_m3=volume)


def test_later_mark_aggregates_after_parent_lock(item, species, monkeypatch):
    from django.db.models.query import QuerySet
    add_mark(item, species[0], 2)
    original = QuerySet.select_for_update
    completed_writer = False
    def lock_after_writer(queryset, *args, **kwargs):
        nonlocal completed_writer
        if queryset.model is HarvestPlanItem and not completed_writer:
            completed_writer = True
            add_mark(item, species[0], 3)
        return original(queryset, *args, **kwargs)
    monkeypatch.setattr(QuerySet, 'select_for_update', lock_after_writer)
    rematerialize_volume_marked(item.pk)
    item.refresh_from_db()
    assert item.volume_marked_m3 == 5, 'Include the child writer that finished before our parent lock'
    assert item.version == 1


def test_total_only_mark_write_preserves_metadata_token_and_noop_history(item, species):
    add_mark(item, species[0], 2)
    before = (item.version, item.created_at, item.modified_at)
    rematerialize_volume_marked(item.pk)
    item.refresh_from_db()
    assert (item.version, item.created_at) == before[:2]
    assert item.modified_at > before[2]
    unchanged = (item.modified_at, item.history.count())
    rematerialize_volume_marked(item.pk)
    item.refresh_from_db()
    assert (item.modified_at, item.history.count()) == unchanged
    TreeMark.objects.filter(harvest_plan_item=item).delete()
    rematerialize_volume_marked(item.pk)
    item.refresh_from_db()
    assert item.volume_marked_m3 is None and item.version == before[0]


def test_first_harvest_has_one_complete_parent_event(item, species, crews, products, tractors, client):
    item.state = HarvestPlanItemState.OPEN
    item.save()
    before = item.history.count()
    body = dict(date='2026-10-01', harvest_plan_item_id=item.pk,
        crew_id=crews[0].pk, product_id=products[0].pk, mass_q='18',
        **{f'sp_{species[0].pk}': '100', f'tr_{tractors[0].pk}': '100'})
    response = client.post(reverse('prelievi-save'), json.dumps(body), content_type='application/json')
    assert response.status_code == 200, response.content
    item.refresh_from_db()
    assert item.state == HarvestPlanItemState.HARVESTING and item.version == 2
    assert item.volume_actual_m3 == 2
    assert item.history.count() == before + 1
    assert item.history.first().volume_actual_m3 == item.volume_actual_m3


def test_later_actual_total_reads_after_parent_lock(item, crews, products, monkeypatch):
    from django.db.models.query import QuerySet
    def add(volume):
        return Harvest.objects.create(harvest_plan_item=item, parcel=item.parcel,
            crew=crews[0], product=products[0], date='2026-10-01', mass_q=18, volume_m3=volume)
    add(2)
    original = QuerySet.select_for_update
    completed_writer = False
    def lock_after_writer(queryset, *args, **kwargs):
        nonlocal completed_writer
        if queryset.model is HarvestPlanItem and not completed_writer:
            completed_writer = True
            add(3)
        return original(queryset, *args, **kwargs)
    monkeypatch.setattr(QuerySet, 'select_for_update', lock_after_writer)
    _rematerialize_volume_actual(item.pk)
    item.refresh_from_db()
    assert item.volume_actual_m3 == 5
    assert item.version == 1


def test_actual_empty_total_is_zero_and_repeated_materialization_is_noop(item):
    before = item.version
    _rematerialize_volume_actual(item.pk)
    item.refresh_from_db()
    assert item.volume_actual_m3 == 0 and item.version == before
    unchanged = (item.modified_at, item.history.count())
    _rematerialize_volume_actual(item.pk)
    item.refresh_from_db()
    assert (item.modified_at, item.history.count()) == unchanged
