"""Mutation/dirty-mark atomicity, including callers without a retry nonce."""

import importlib
import json
from unittest.mock import patch

import pytest
from django.db.models import F
from django.test import Client
from django.urls import reverse

from apps.base import digests, hypsometry
from apps.base.models import (
    Crew, DigestStatus, HarvestPlan, HarvestPlanItem, HypsoParamSet,
    HypsoParamSource, Product, Sample, SampleArea, SampleGrid, Survey, Tree,
    TreeMark, TreeSample,
)
from apps.mannesi.models import ProductionCredit, WorkHour
from apps.prelievi.models import Harvest
from config.constants import DIGEST_HYPSO_PARAMS, DIGEST_PARCEL_DENDROMETRY_POINTS

pytestmark = pytest.mark.django_db
DELETE_KINDS = ['tree', 'area', 'grid', 'survey', 'plan', 'item', 'mark',
                'harvest', 'hours', 'credits']


@pytest.fixture
def client(writer_user):
    client = Client(raise_request_exception=False)
    client.force_login(writer_user)
    return client


@pytest.fixture(params=DELETE_KINDS)
def deletion(request, parcels, species):
    kind = request.param
    parcel = parcels[0]
    module = 'campionamenti'
    if kind in ('tree', 'area', 'grid', 'survey'):
        grid = SampleGrid.objects.create(name='Atomic deletion')
        obj = grid
        if kind in ('area', 'tree'):
            obj = SampleArea.objects.create(
                sample_grid=grid, parcel=parcel, number='1', lat=0, lon=0,
            )
        if kind in ('survey', 'tree'):
            survey = Survey.objects.create(name='Atomic survey', sample_grid=grid)
            if kind == 'survey':
                obj = survey
            else:
                sample = Sample.objects.create(
                    survey=survey, sample_area=obj, date='2026-10-25',
                )
                tree = Tree.objects.create(species=species[0])
                obj = TreeSample.objects.create(
                    sample=sample, tree=tree, parcel=parcel, number=1,
                    d_cm=30, h_m=20,
                )
        url = reverse(f'campionamenti-{kind}-delete', args=[obj.pk])
    elif kind in ('plan', 'item', 'mark'):
        module = 'piano_di_taglio'
        obj = HarvestPlan.objects.create(name='Atomic plan', year_start=2026, year_end=2027)
        if kind in ('item', 'mark'):
            obj = HarvestPlanItem.objects.create(
                harvest_plan=obj, parcel=parcel, year_planned=2026,
            )
        if kind == 'mark':
            obj = TreeMark.objects.create(
                harvest_plan_item=obj, tree=Tree.objects.create(species=species[0]),
                parcel=parcel, date='2026-10-25', d_cm=30, h_m=20,
            )
        args = [] if kind == 'mark' else [obj.pk]
        url = reverse(f'piano-di-taglio-{kind}-delete', args=args)
    else:
        crew = Crew.objects.create(name='Atomic crew')
        if kind == 'harvest':
            module = 'prelievi'
            obj = Harvest.objects.create(
                date='2026-10-25', parcel=parcel, crew=crew, mass_q=10,
                product=Product.objects.create(name='Atomic product'),
            )
            url = reverse('prelievi-delete')
        else:
            module = 'mannesi'
            model, fields = ((WorkHour, {'hours': 8}) if kind == 'hours'
                             else (ProductionCredit, {'mass_q': 10}))
            obj = model.objects.create(date='2026-10-25', crew=crew, **fields)
            url = reverse(f'squadre-{kind}-delete')
    return obj, url, importlib.import_module(f'apps.{module}.views')


def digest_state():
    return list(DigestStatus.objects.order_by('pk').values())


def test_delete_rolls_back_when_dirty_marking_fails(client, deletion):
    obj, url, views = deletion
    before = digest_state()
    original = views.mark_stale

    def mark_then_fail(*names):
        original(*names)
        raise RuntimeError('Interrupted dirty marking')

    with patch.object(views, 'mark_stale', side_effect=mark_then_fail):
        response = client.post(url, json.dumps({'row_id': obj.pk, 'version': obj.version}),
                               content_type='application/json')
    assert response.status_code == 500
    assert type(obj).objects.filter(pk=obj.pk).exists()
    assert digest_state() == before


@pytest.mark.parametrize('removed', [False, True])
def test_delete_rechecks_row_at_write_boundary(client, deletion, removed):
    obj, url, views = deletion
    original = views.begin_nonce_write

    def competing_write(request):
        # Deterministically simulate another writer after the preflight read,
        # before this request starts its write transaction.
        if removed:
            type(obj).objects.filter(pk=obj.pk).delete()
        else:
            type(obj).objects.filter(pk=obj.pk).update(version=F('version') + 1)
        original(request)

    with patch.object(views, 'begin_nonce_write', side_effect=competing_write):
        response = client.post(url, json.dumps({'row_id': obj.pk, 'version': obj.version}),
                               content_type='application/json')
    assert response.status_code == (404 if removed else 400)
    if not removed:
        obj.refresh_from_db()
        assert obj.version == 2
        assert response.json()['status'] == 'conflict'


@pytest.mark.parametrize('operation', ['replace', 'clear'])
@pytest.mark.parametrize('fail', [False, True])
def test_hypsometry_service_owns_dirty_marks(hypso_samples, operation, fail):
    rows = hypsometry.compute_params([hypso_samples['survey'].pk], min_n=5)
    original_set = hypsometry.replace_active_set(
        rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[],
    )
    names = [DIGEST_HYPSO_PARAMS, DIGEST_PARCEL_DENDROMETRY_POINTS, 'audit']
    for name in names:
        DigestStatus.objects.update_or_create(name=name, defaults={'stale': False})
    before = digest_state()

    def apply():
        if operation == 'clear':
            hypsometry.clear_active_set()
        else:
            hypsometry.replace_active_set(
                rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[],
            )

    original = digests.mark_stale

    def mark_then_fail(*names):
        original(*names)
        raise RuntimeError('Interrupted dirty marking')

    if fail:
        with patch.object(digests, 'mark_stale', side_effect=mark_then_fail):
            with pytest.raises(RuntimeError, match='Interrupted dirty marking'):
                apply()
        assert hypsometry.active_set() == original_set
        assert HypsoParamSet.objects.count() == 1
        assert digest_state() == before
    else:
        apply()
        assert DigestStatus.objects.filter(name__in=names, stale=True).count() == 3
