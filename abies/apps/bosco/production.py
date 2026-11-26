"""Monthly harvest data for the historical yield report."""

import hashlib

from django.contrib.auth.decorators import login_required
from django.db.models import Sum
from django.db.models.functions import TruncMonth
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils.cache import get_conditional_response
from django.views.decorators.gzip import gzip_page
from django.views.decorators.http import require_GET

from apps.base.http import CACHE_NO_STORE, apply_cache_control
from apps.base.models import Region
from apps.base.numparse import float_or_none
from apps.prelievi.models import Harvest


@login_required
@require_GET
@gzip_page
def region_production_view(request, region_id):
    get_object_or_404(Region.objects.only('id'), pk=region_id)
    rows = (
        Harvest.objects.filter(parcel__region_id=region_id, damaged=False, unhealthy=False)
        .annotate(month=TruncMonth('date'))
        .values('month', 'parcel_id', 'parcel__name', 'parcel__area_ha',
                'parcel__eclass__coppice', 'harvest_plan_item_id',
                'harvest_plan_item__intervention_area_ha')
        .annotate(monthly_mass=Sum('mass_q'), monthly_volume=Sum('volume_m3'))
        .order_by('month', 'parcel_id', 'harvest_plan_item_id')
    )
    response = JsonResponse({'rows': [{
        'month': row['month'].isoformat()[:7],
        'parcel_id': row['parcel_id'],
        'parcel': row['parcel__name'],
        'coppice': row['parcel__eclass__coppice'],
        'parcel_area_ha': float_or_none(row['parcel__area_ha']),
        'worksite_id': row['harvest_plan_item_id'],
        'intervention_area_ha': float_or_none(row['harvest_plan_item__intervention_area_ha']),
        'mass_q': float(row['monthly_mass']),
        'volume_m3': float(row['monthly_volume']),
    } for row in rows]})
    etag = f'W/"{hashlib.sha256(response.content).hexdigest()}"'
    response['ETag'] = etag
    apply_cache_control(response, CACHE_NO_STORE)
    return get_conditional_response(request, etag=etag, response=response)
