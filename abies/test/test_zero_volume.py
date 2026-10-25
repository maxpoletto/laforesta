"""Small measured trees can have zero, never negative, harvestable volume."""

import json

import pytest
from django.test import Client

from apps.base.models import (
    HarvestPlan, HarvestPlanItem, SampleArea, SampleGrid, Species, Survey,
    TreeMark, TreeSample,
)


@pytest.mark.parametrize('kind', ['mark', 'sample'])
def test_zero_volume_and_mass_are_persisted(kind, writer_user, parcels):
    species = Species.objects.create(common_name='Ontano', density=9)
    body = {'species_id': species.pk, 'number': 1, 'd_cm': 5, 'h_m': '3',
            'h_measured': True, 'operator': 'Test', 'date': '2026-10-25'}
    if kind == 'mark':
        plan = HarvestPlan.objects.create(name='Small trees', year_start=2026, year_end=2027)
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2026,
        )
        body['harvest_plan_item_id'] = item.pk
        url, model = '/api/piano-di-taglio/mark/save/', TreeMark
    else:
        grid = SampleGrid.objects.create(name='Small trees')
        area = SampleArea.objects.create(sample_grid=grid, parcel=parcels[0],
                                         number='1', lat=0, lon=0)
        survey = Survey.objects.create(name='Small trees', sample_grid=grid)
        body.update({'survey_id': survey.pk, 'sample_area_id': area.pk})
        url, model = '/api/campionamenti/tree/save/', TreeSample
    client = Client()
    client.force_login(writer_user)
    response = client.post(url, json.dumps(body), content_type='application/json')
    assert response.status_code == 200, response.content
    row = model.objects.get()
    assert row.d_cm == 5 and row.h_m == 3
    assert row.volume_m3 == row.mass_q == 0
    if kind == 'mark':
        item.refresh_from_db()
        assert item.volume_marked_m3 == 0
