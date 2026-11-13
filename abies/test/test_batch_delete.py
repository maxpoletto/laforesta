"""Selected-row deletion is scoped, versioned, audited and atomic."""
import json
from dataclasses import dataclass
from decimal import Decimal

import pytest
from django.test import Client

from apps.base.models import (
    HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    Sample, SampleArea, SampleGrid, Survey, Tree, TreeMark, TreeSample, UsedNonce,
)
from apps.prelievi.models import Harvest

pytestmark = pytest.mark.django_db


@dataclass
class Selection:
    url: str
    scope: dict
    data_id: str
    rows: list
    parent: object

    def body(self, rows=None, **extra):
        selected = self.rows[:2] if rows is None else rows
        return {
            **self.scope,
            'rows': [{'row_id': row.pk, 'version': row.version} for row in selected],
            **extra,
        }


@pytest.fixture
def writer(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


@pytest.fixture(params=['harvest', 'item', 'mark', 'sample'])
def selection(request, parcels, species, crews, products):
    kind = request.param
    plan = HarvestPlan.objects.create(name='Batch plan', year_start=2026, year_end=2035)
    item = HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2027,
        state=HarvestPlanItemState.OPEN,
    )
    if kind == 'item':
        rows = [item] + [HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=year,
        ) for year in (2028, 2029)]
        return Selection('/api/piano-di-taglio/item/delete/',
                         {'harvest_plan_id': plan.pk}, 'harvest_plan_items', rows, plan)
    if kind == 'harvest':
        rows = [Harvest.objects.create(
            harvest_plan_item=item, parcel=parcels[0], crew=crews[0],
            product=products[0], date='2026-11-01', mass_q=18, volume_m3=volume,
        ) for volume in (1, 2, 3)]
        return Selection('/api/prelievi/delete/', {}, 'prelievi', rows, item)
    if kind == 'mark':
        rows = [TreeMark.objects.create(
            harvest_plan_item=item, parcel=parcels[0],
            tree=Tree.objects.create(species=species[0]), number=number,
            date='2026-11-01', d_cm=30, h_m=20, volume_m3=number,
        ) for number in (1, 2, 3)]
        return Selection('/api/piano-di-taglio/mark/delete/',
                         {'harvest_plan_item_id': item.pk}, f'mark_trees_{item.pk}', rows, item)
    grid = SampleGrid.objects.create(name='Batch grid')
    survey = Survey.objects.create(name='Batch survey', sample_grid=grid)
    area = SampleArea.objects.create(sample_grid=grid, parcel=parcels[0], number='1',
                                     lat=38, lon=16, r_m=20)
    sample = Sample.objects.create(sample_area=area, survey=survey, date='2026-11-01')
    rows = [TreeSample.objects.create(
        sample=sample, parcel=parcels[0], tree=Tree.objects.create(species=species[0]),
        number=number, d_cm=30, h_m=20, volume_m3=Decimal('1.2'),
    ) for number in (1, 2, 3)]
    return Selection('/api/campionamenti/tree/delete/',
                     {'survey_id': survey.pk}, f'sampled_trees_{survey.pk}', rows, sample)


def post(client, selection, body):
    return client.post(selection.url, json.dumps(body), content_type='application/json')


def test_deletes_only_selected_rows_and_records_each_audit_event(writer, selection, writer_user):
    ids = [row.pk for row in selection.rows[:2]]
    response = post(writer, selection, selection.body(nonce='selected-delete'))
    assert response.status_code == 200, response.content
    model = type(selection.rows[0])
    assert list(model.objects.filter(pk__in=[r.pk for r in selection.rows])
                .values_list('pk', flat=True)) == [selection.rows[2].pk]
    assert {(entry['data_id'], entry['row_id']) for entry in response.json()['deletes']} >= {
        (selection.data_id, pk) for pk in ids
    }
    history = model.history.filter(id__in=ids, history_type='-') if hasattr(model, 'history') else None
    if history is not None:
        assert history.count() == 2
        assert set(history.values_list('history_user_id', flat=True)) == {writer_user.pk}
    # Replaying the completed batch returns its receipt despite rows being gone.
    replay = post(writer, selection, selection.body(nonce='selected-delete'))
    assert replay.status_code == 200
    assert replay.json() == response.json()
    if history is not None:
        assert history.count() == 2


@pytest.mark.parametrize('nonce', [None, 'stale-selection'])
def test_one_stale_version_rejects_entire_batch(writer, selection, nonce):
    body = selection.body()
    body['rows'][1]['version'] += 1
    if nonce:
        body['nonce'] = nonce
    response = post(writer, selection, body)
    assert response.status_code == 400, response.content
    assert response.json()['status'] == 'conflict'
    assert type(selection.rows[0]).objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3
    model = type(selection.rows[0])
    if hasattr(model, 'history'):
        assert not model.history.filter(history_type='-').exists()
    assert not UsedNonce.objects.filter(nonce=nonce).exists()


def test_missing_record_rejects_entire_batch(writer, selection):
    body = selection.body()
    body['rows'][1]['row_id'] = 999999
    response = post(writer, selection, body)
    assert response.status_code == 404
    assert type(selection.rows[0]).objects.filter(pk=selection.rows[0].pk).exists()


@pytest.mark.parametrize('invalid', [[], 'all', [None], [{'row_id': True, 'version': 1}],
                                  [{'row_id': 1.5, 'version': 1}]])
def test_malformed_selection_is_rejected(writer, selection, invalid):
    response = post(writer, selection, {**selection.scope, 'rows': invalid})
    assert response.status_code == 400
    assert type(selection.rows[0]).objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3


def test_duplicate_ids_are_rejected(writer, selection):
    body = selection.body()
    body['rows'].append(body['rows'][0])
    response = post(writer, selection, body)
    assert response.status_code == 400
    assert type(selection.rows[0]).objects.filter(pk=selection.rows[0].pk).exists()


def test_reader_cannot_delete_selection(reader_user, selection):
    client = Client()
    client.force_login(reader_user)
    response = post(client, selection, selection.body())
    assert response.status_code == 403


def test_scope_is_required_and_must_match(writer, selection):
    if not selection.scope:
        return
    response = post(writer, selection, {'rows': selection.body()['rows']})
    assert response.status_code == 400
    response = post(writer, selection, {
        **selection.body(), **{key: 999999 for key in selection.scope},
    })
    assert response.status_code == 404
    assert type(selection.rows[0]).objects.filter(pk=selection.rows[0].pk).exists()


def test_parent_totals_and_tree_lifetimes(writer, selection):
    if selection.data_id in ('prelievi', 'harvest_plan_items'):
        field = 'volume_actual_m3' if selection.data_id == 'prelievi' else None
    else:
        field = 'volume_marked_m3' if selection.data_id.startswith('mark_trees') else None
    tree_ids = [r.tree_id for r in selection.rows if hasattr(r, 'tree_id')]
    response = post(writer, selection, selection.body())
    assert response.status_code == 200, response.content
    selection.parent.refresh_from_db()
    if field:
        assert getattr(selection.parent, field) == 3
        assert selection.parent.version == 1  # Derived total is not a domain edit.
    if selection.data_id.startswith('mark_trees'):
        assert not Tree.objects.filter(pk__in=tree_ids[:2]).exists()
    elif selection.data_id.startswith('sampled_trees'):
        assert Tree.objects.filter(pk__in=tree_ids).count() == 3
        assert Sample.objects.filter(pk=selection.parent.pk).exists()


def test_dependency_blocks_whole_intervention_batch(writer, selection, species):
    if selection.data_id != 'harvest_plan_items':
        return
    blocked = selection.rows[1]
    TreeMark.objects.create(harvest_plan_item=blocked, parcel=blocked.parcel,
        tree=Tree.objects.create(species=species[0]), date='2026-11-01', d_cm=30)
    response = post(writer, selection, selection.body(nonce='blocked-delete'))
    assert response.status_code == 400
    assert str(blocked.year_planned) in response.json()['message']
    assert f'ID {blocked.pk}' in response.json()['message']
    assert HarvestPlanItem.objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3
    assert not UsedNonce.objects.filter(nonce='blocked-delete').exists()


def test_closed_worksite_blocks_whole_mark_batch(writer, selection):
    if not selection.data_id.startswith('mark_trees'):
        return
    selection.parent.state = HarvestPlanItemState.CLOSED
    selection.parent.save()
    response = post(writer, selection, selection.body())
    assert response.status_code == 400
    assert TreeMark.objects.filter(harvest_plan_item=selection.parent).count() == 3


@pytest.mark.parametrize('nonce', [None, 'interrupted-delete'])
def test_failure_during_deletion_rolls_back_every_row(writer, selection, monkeypatch, nonce):
    model = type(selection.rows[0])
    original = model.delete
    calls = 0

    def fail_on_second(obj, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated deletion failure')
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(model, 'delete', fail_on_second)
    body = selection.body()
    if nonce:
        body['nonce'] = nonce
    with pytest.raises(RuntimeError, match='simulated deletion failure'):
        post(writer, selection, body)
    assert model.objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3
    assert not UsedNonce.objects.filter(nonce=nonce).exists()
    if hasattr(model, 'history'):
        assert not model.history.filter(history_type='-').exists()


def test_response_failure_rolls_back_selection_without_nonce(writer, selection, monkeypatch):
    module = ('apps.prelievi.views' if selection.data_id == 'prelievi' else
              'apps.campionamenti.views' if selection.data_id.startswith('sampled_trees') else
              'apps.piano_di_taglio.views')

    def fail(*args, **kwargs):
        raise RuntimeError('simulated response failure')

    monkeypatch.setattr(f'{module}.success_response', fail)
    with pytest.raises(RuntimeError, match='simulated response failure'):
        post(writer, selection, selection.body())
    assert type(selection.rows[0]).objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3


def test_missing_version_does_not_weaken_batch_locking(writer, selection):
    body = selection.body()
    del body['rows'][1]['version']
    response = post(writer, selection, body)
    assert response.status_code == 400
    assert response.json()['status'] == 'conflict'
    assert type(selection.rows[0]).objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3


def test_batch_rechecks_versions_at_the_lock_boundary(writer, selection, monkeypatch):
    from django.db.models import F
    from django.db.models.query import QuerySet

    original = QuerySet.select_for_update
    completed_writer = False
    model = type(selection.rows[0])

    def lock_after_writer(queryset, *args, **kwargs):
        nonlocal completed_writer
        if queryset.model is model and not completed_writer:
            completed_writer = True
            model.objects.filter(pk=selection.rows[1].pk).update(version=F('version') + 1)
        return original(queryset, *args, **kwargs)

    monkeypatch.setattr(QuerySet, 'select_for_update', lock_after_writer)
    response = post(writer, selection, selection.body())
    assert response.status_code == 400
    assert response.json()['status'] == 'conflict'
    assert model.objects.filter(pk__in=[r.pk for r in selection.rows]).count() == 3


def test_harvest_batch_updates_every_affected_parent_once(writer, selection, monkeypatch):
    if selection.data_id != 'prelievi':
        return
    from apps.prelievi import views

    other = HarvestPlanItem.objects.create(
        harvest_plan=selection.parent.harvest_plan, parcel=selection.parent.parcel,
        year_planned=2028, state=HarvestPlanItemState.OPEN,
    )
    Harvest.objects.filter(pk=selection.rows[1].pk).update(harvest_plan_item=other)
    original = views._rematerialize_volume_actual
    refreshed = []

    def track(item_id):
        refreshed.append(item_id)
        return original(item_id)

    monkeypatch.setattr(views, '_rematerialize_volume_actual', track)
    response = post(writer, selection, selection.body())
    assert response.status_code == 200, response.content
    assert sorted(refreshed) == sorted([selection.parent.pk, other.pk])
    selection.parent.refresh_from_db()
    other.refresh_from_db()
    assert selection.parent.volume_actual_m3 == 3
    assert other.volume_actual_m3 == 0
    assert {patch['row_id'] for patch in response.json()['patches']
            if patch['data_id'] == 'harvest_plan_items'} == {selection.parent.pk, other.pk}


def test_sample_batch_updates_preserved_tree_representative(writer, selection):
    if not selection.data_id.startswith('sampled_trees'):
        return
    current = selection.rows[0]
    current.preserved_number = 7
    current.save(update_fields=['preserved_number'])
    earlier_survey = Survey.objects.create(name='Earlier visit')
    earlier_sample = Sample.objects.create(survey=earlier_survey, date='2025-11-01')
    earlier = TreeSample.objects.create(
        sample=earlier_sample, parcel=current.parcel, tree=current.tree,
        number=1, preserved_number=7, d_cm=29,
    )
    response = post(writer, selection, selection.body())
    assert response.status_code == 200, response.content
    payload = response.json()
    assert {'data_id': 'preserved_trees', 'row_id': current.pk} in payload['deletes']
    assert any(patch['data_id'] == 'preserved_trees' and patch['row_id'] == earlier.pk
               for patch in payload['patches'])
    sample_patch = next(patch for patch in payload['patches']
                        if patch['data_id'] == 'samples')
    assert sample_patch['record'][-1] == 1
    assert Tree.objects.filter(pk=current.tree_id).exists()


@pytest.mark.parametrize('use', ['mark', 'sample'])
def test_mark_batch_preserves_tree_still_in_use(writer, selection, use):
    if not selection.data_id.startswith('mark_trees'):
        return
    first = selection.rows[0]
    if use == 'mark':
        other = HarvestPlanItem.objects.create(
            harvest_plan=selection.parent.harvest_plan, parcel=selection.parent.parcel,
            year_planned=2028,
        )
        retained = TreeMark.objects.create(
            harvest_plan_item=other, parcel=first.parcel, tree=first.tree,
            number=1, date='2026-11-02', d_cm=30,
        )
    else:
        survey = Survey.objects.create(name='Shared physical tree')
        sample = Sample.objects.create(survey=survey, date='2026-11-02')
        retained = TreeSample.objects.create(
            sample=sample, parcel=first.parcel, tree=first.tree,
            number=1, d_cm=30, h_m=20,
        )
    response = post(writer, selection, selection.body())
    assert response.status_code == 200, response.content
    assert type(retained).objects.filter(pk=retained.pk).exists()
    assert Tree.objects.filter(pk=first.tree_id).exists()
