from django.urls import reverse

from apps.base.models import HarvestPlan, HarvestPlanItem
from apps.prelievi.models import Harvest


def test_region_history_uses_harvest_dates_and_filters_exceptional_cuts(
    client, reader_user, parcels, crews, products,
):
    parcel, _, other_region = parcels
    plan = HarvestPlan.objects.create(name='Piano', year_start=2020, year_end=2030)
    worksite = HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcel, year_planned=2023, intervention_area_ha=4,
    )
    shared = {'crew': crews[0], 'product': products[0], 'mass_q': 80, 'volume_m3': 10}
    Harvest.objects.create(date='2024-12-01', parcel=parcel, harvest_plan_item=worksite, **shared)
    Harvest.objects.create(date='2024-12-15', parcel=parcel, harvest_plan_item=worksite, **shared)
    Harvest.objects.create(date='2025-01-01', parcel=parcel, harvest_plan_item=worksite, **shared)
    Harvest.objects.create(date='2019-06-01', parcel=parcel, **shared)
    Harvest.objects.create(date='2026-01-01', parcel=parcel, damaged=True, **shared)
    Harvest.objects.create(date='2026-02-01', parcel=parcel, unhealthy=True, **shared)
    Harvest.objects.create(date='2026-03-01', region=parcel.region, **shared)
    Harvest.objects.create(date='2026-04-01', parcel=other_region, **shared)

    url = reverse('bosco-region-production', args=[parcel.region_id])
    assert client.get(url).status_code == 302
    client.force_login(reader_user)
    shell = client.get('/bosco')
    assert b'Riepilogo compresa' in shell.content
    assert b'data-target="production-yield"' in shell.content
    response = client.get(url)
    assert response.status_code == 200
    assert 'no-store' in response.headers['Cache-Control']
    rows = response.json()['rows']
    assert [(row['month'], row['worksite_id'], row['volume_m3']) for row in rows] == [
        ('2019-06', None, 10), ('2024-12', worksite.id, 20), ('2025-01', worksite.id, 10),
    ]
    assert rows[1] == {
        'month': '2024-12', 'parcel_id': parcel.id, 'parcel': parcel.name,
        'coppice': False, 'parcel_area_ha': 10.5, 'worksite_id': worksite.id,
        'intervention_area_ha': 4, 'mass_q': 160, 'volume_m3': 20,
    }
    assert client.get(url, headers={'If-None-Match': response.headers['ETag']}).status_code == 304

    worksite.intervention_area_ha = 5
    worksite.save()
    changed = client.get(url, headers={'If-None-Match': response.headers['ETag']})
    assert changed.status_code == 200
    assert changed.json()['rows'][1]['intervention_area_ha'] == 5
