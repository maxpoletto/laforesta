"""One cached plan source supplies forecasts and the complete parcel calendar."""

import gzip
import json
from decimal import Decimal

from django.test import Client

from apps.base.digests import generate_future_production
from apps.base.models import HarvestPlan, HarvestPlanItem, HarvestPlanItemState, Parcel
from config import strings as S


def read_digest(settings):
    generate_future_production()
    with gzip.open(settings.DIGEST_DIR / 'future_production.json.gz', 'rt') as stream:
        return json.load(stream)


def test_plan_calendar_includes_all_parcel_items_without_changing_forecast(
        parcels, eclasses, settings,
):
    plan = HarvestPlan.objects.create(name='Selected', year_start=2026, year_end=2040, active=True)
    other = HarvestPlan.objects.create(name='Other', year_start=2041, year_end=2055)
    highforest = parcels[0]
    coppice = Parcel.objects.create(
        name='C1', region=highforest.region, eclass=eclasses[2],
        area_ha=Decimal('3'), intervention_interval=18, standards_per_ha=75,
    )
    item = HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=highforest, year_planned=2030,
        volume_planned_m3=Decimal('12.5'),
    )
    HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=highforest, year_planned=2028,
        state=HarvestPlanItemState.CLOSED,
    )
    HarvestPlanItem.objects.create(harvest_plan=plan, parcel=highforest, year_planned=2030)
    HarvestPlanItem.objects.create(harvest_plan=plan, parcel=coppice, year_planned=2027)
    HarvestPlanItem.objects.create(
        harvest_plan=plan, region=highforest.region, year_planned=2026, damaged=True,
    )
    HarvestPlanItem.objects.create(harvest_plan=other, parcel=highforest, year_planned=2042)

    data = read_digest(settings)

    assert data['selected_plan'] == {
        'id': plan.id, 'name': 'Selected', 'year_start': 2026, 'year_end': 2040,
    }
    calendar = {row['parcel_id']: row['years'] for row in data['calendar']}
    assert calendar == {highforest.id: [2028, 2030], coppice.id: [2027]}
    assert [row[0] for row in data['rows']] == [item.id]
    assert data['rows'][0][data['columns'].index(S.COL_VOLUME_PLANNED)] == 12.5


def test_empty_selected_plan_still_identifies_its_source(db, settings):
    plan = HarvestPlan.objects.create(name='Empty plan', year_start=2026, year_end=2040, active=True)
    data = read_digest(settings)
    assert data['selected_plan']['id'] == plan.id
    assert data['calendar'] == []
    assert data['rows'] == []


def test_no_plans_has_explicit_empty_source(db, settings):
    data = read_digest(settings)
    assert data['selected_plan'] is None
    assert data['calendar'] == []


def test_switching_plan_refreshes_both_calendar_and_volume(parcels, writer_user):
    first = HarvestPlan.objects.create(name='First', year_start=2026, year_end=2040, active=True)
    second = HarvestPlan.objects.create(name='Second', year_start=2041, year_end=2055)
    for plan, year, volume in ((first, 2028, '10'), (second, 2042, '20')):
        HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=year,
            volume_planned_m3=Decimal(volume),
        )
    client = Client()
    client.force_login(writer_user)

    def get():
        response = client.get('/api/bosco/future-production/data/')
        assert response.status_code == 200
        return json.loads(gzip.decompress(b''.join(response.streaming_content)))

    assert get()['selected_plan']['id'] == first.id
    response = client.post('/api/bosco/plan-source/save/',
        data=json.dumps({'harvest_plan_id': second.id, 'nonce': 'switch-cached-plan'}),
        content_type='application/json')
    assert response.status_code == 200
    updated = get()
    assert updated['selected_plan']['id'] == second.id
    assert updated['calendar'] == [{'parcel_id': parcels[0].id, 'years': [2042]}]
    assert updated['rows'][0][-1] == 20
