"""Parent clocks, edit tokens and complete state/total history events."""
import json
from datetime import date
from decimal import Decimal

import pytest
from django.test import Client
from django.urls import reverse

from apps.base.models import HarvestPlan, HarvestPlanItem, HarvestPlanItemState
from apps.piano_di_taglio.mark_import import MarkImportRow, import_mark_rows

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
