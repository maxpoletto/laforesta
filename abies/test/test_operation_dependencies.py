"""Stored dirtiness and optimistic client updates follow one operation family."""
import json

import pytest
from django.test import Client

from apps.base.models import HarvestPlan, HarvestPlanItem, DigestStatus

pytestmark = pytest.mark.django_db


def test_mark_write_invalidates_dependents_but_preserves_patched_datasets(parcels, species, writer_user):
    plan = HarvestPlan.objects.create(name='Dependency plan', year_start=2024, year_end=2030)
    item = HarvestPlanItem.objects.create(harvest_plan=plan, parcel=parcels[0], year_planned=2026)
    client = Client()
    client.force_login(writer_user)
    body = dict(harvest_plan_item_id=item.pk, species_id=species[0].pk,
        parcel_id=parcels[0].pk, date='2026-10-01', operator='Dependency operator', d_cm=30, h_m=20)
    response = client.post('/api/piano-di-taglio/mark/save/', json.dumps(body), content_type='application/json')
    assert response.status_code == 200, response.content
    data = response.json()
    invalidated = set(data.get('invalidates', {}).get('data_ids', []))
    patched = {patch['data_id'] for patch in data['patches']}
    assert {'future_production', 'audit'} <= invalidated
    assert not invalidated & patched
    assert {f'mark_trees_{item.pk}', 'harvest_plan_items'} <= patched
    assert set(DigestStatus.objects.filter(stale=True).values_list('name', flat=True)) >= invalidated | patched
