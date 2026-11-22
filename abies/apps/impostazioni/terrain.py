"""Queue and inspect a compresa's terrain refresh."""
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_http_methods

from apps.base.auth import require_writer
from apps.base.models import Region, TerrainRefresh
from apps.base.responses import validation_error
from apps.bosco.terrain_queue import enqueue, has_parcels
from config import strings as S


def job_status(region_id, job=None):
    if job is None:
        job = TerrainRefresh.objects.filter(pk=region_id).first()
    available = has_parcels(region_id)
    state = job.state if job and available else ''
    message = S.TERRAIN_REFRESH_MESSAGES.get(state, '')
    if state in (TerrainRefresh.State.FAILED, TerrainRefresh.State.WAITING) and job.failure:
        message = S.TERRAIN_REFRESH_FAILURES.get(job.failure.get('kind'), message)
        status = job.failure.get('status')
        if status == 429:
            message = S.TERRAIN_REFRESH_QUOTA
        elif status:
            message = S.TERRAIN_REFRESH_HTTP_STATUS.format(message=message.rstrip('.'), status=status)
    return {
        'available': available,
        'state': state,
        'message': message,
    }


@login_required
@require_writer
@never_cache
@require_http_methods(['GET', 'POST'])
def refresh(request, region_id):
    get_object_or_404(Region, pk=region_id)
    if request.method == 'POST':
        job = enqueue(region_id, requested_by=request.user)
        if job is None:
            return validation_error([S.TERRAIN_REFRESH_NO_PARCELS])
        return JsonResponse(job_status(region_id, job), status=202)
    return JsonResponse(job_status(region_id))
