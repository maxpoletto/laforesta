"""Writer workspace for the selected compresa."""
from itertools import groupby

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import F, Q
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET, require_POST

from apps.base.auth import require_writer
from apps.base.idempotency import begin_nonce_write
from apps.base.models import MapDrawing, MapFeature, MapPointSet, Region, RoadType
from apps.base.numparse import int_or_none
from apps.base.responses import parse_json_body, success_response, validation_error
from apps.ipso.models import IpsoUpload, IpsoUploadMode, IpsoUploadState, IpsoUploadTargetType
from apps.ipso.staging import claim_upload_import, read_staged_payload
from config import strings as S


@login_required
@require_writer
@require_GET
def catalog(request):
    return JsonResponse({'regions': list(Region.objects.order_by('name').values('id', 'name')),
                         'road_types': list(RoadType.objects.filter(active=True).values('id', 'name'))})


def pending_uploads(region_id):
    pending = IpsoUpload.objects.filter(mode=IpsoUploadMode.POINTS, state=IpsoUploadState.RECEIVED)
    rows = []
    for upload in pending.filter(Q(point_region_id=region_id) | Q(point_region_id__isnull=True)):
        # Older inbox rows predate the region index. Trust the recorded session.
        if upload.point_region_id is None:
            payload, error = read_staged_payload(upload)
            if error or payload['session'].get('region_id') != region_id:
                continue
        rows.append({'id': upload.pk, 'name': upload.work_package_id,
                     'operator': upload.operator, 'date': upload.record_date})
    return rows


@login_required
@require_writer
@require_GET
def workspace(request, region_id):
    region = get_object_or_404(Region, pk=region_id)
    return JsonResponse({
        'uploads': pending_uploads(region.pk),
        'sources': list(MapPointSet.objects.filter(region=region).order_by('-date', '-pk')
                        .values('id', 'name', 'operator', 'date')),
        'drawings': list(MapDrawing.objects.filter(region=region).values(
            'id', 'source_id', 'sequence', 'name', 'geometry', 'fixes', 'version')),
        'roads': list(MapFeature.objects.filter(region=region).values(
            'id', 'name', 'road_type_id', 'geometry', 'published', 'version')),
    })


def restore_drawings(source):
    existing = set(MapDrawing.objects.filter(source=source).values_list('sequence', flat=True))
    drawings = []
    # Manual points are separate even when consecutive. A pause always breaks a line.
    groups = groupby(source.records, key=lambda r: ('trace', r['segment']) if r['segment']
                     else ('point', r['client_record_id']))
    for seq, (_, records) in enumerate(groups, 1):
        if seq in existing:
            continue
        rows = list(records)
        coords = [[r['lon'], r['lat']] for r in rows]
        geometry = ({'type': 'Point', 'coordinates': coords[0]} if len(coords) == 1 else
                    {'type': 'LineString', 'coordinates': coords})
        drawings.append(MapDrawing(region_id=source.region_id, source=source,
                                   name=source.name, sequence=seq, geometry=geometry,
                                   fixes=[[r['acc_m'], r['note']] for r in rows]))
    MapDrawing.objects.bulk_create(drawings)
    return len(drawings)


def import_points(region_id, body, user):
    upload = get_object_or_404(IpsoUpload, pk=int_or_none(body.get('upload_id')))
    if upload.mode != IpsoUploadMode.POINTS or upload.state != IpsoUploadState.RECEIVED:
        raise ValueError(S.IPSO_ERR_UPLOAD_NOT_RECEIVED)
    payload, error = read_staged_payload(upload)
    if error:
        raise ValueError(error)
    session = payload['session']
    if session.get('region_id') != region_id:
        raise ValueError(S.CARTOGRAPHY_WRONG_REGION)
    source, _ = MapPointSet.objects.get_or_create(session_id=upload.session_id, defaults={
        'region_id': region_id, 'name': session['name'], 'date': upload.record_date,
        'operator': upload.operator, 'records': payload['records'],
    })
    if source.region_id != region_id or source.records != payload['records']:
        raise ValueError(S.IPSO_ERR_IMPORT_CONTEXT_CHANGED)
    added = restore_drawings(source)
    claim_upload_import(upload, user, IpsoUploadTargetType.POINT_SET, source.pk)
    return {'added': added, 'source_id': source.pk}


def valid_edit(drawing, geometry):
    if not isinstance(geometry, dict) or geometry.get('type') != drawing.geometry['type']:
        return False
    coords = geometry.get('coordinates')
    points = [coords] if geometry['type'] == 'Point' else coords
    return (isinstance(points, list) and len(points) == len(drawing.fixes)
            and all(isinstance(p, list) and len(p) == 2 and all(type(n) in (float, int) for n in p)
                    and -180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in points))


def check_selection(rows, selected):
    versions = [list(pair) for pair in rows.order_by('pk').values_list('pk', 'version')]
    if selected != versions or not versions:
        raise ValueError(S.CARTOGRAPHY_STALE)


def edit_workspace(region_id, action, body, user):
    if action == 'import':
        return import_points(region_id, body, user)
    if action in ('restore', 'forget'):
        source = get_object_or_404(MapPointSet, region_id=region_id, pk=int_or_none(body.get('source_id')))
        if action == 'restore':
            return {'added': restore_drawings(source)}
        source.delete()
        return {}
    drawings = MapDrawing.objects.filter(region_id=region_id)
    if action == 'clear':
        check_selection(drawings, body.get('versions'))
        drawings.delete()
        return {}
    if action == 'adjust':
        drawing = get_object_or_404(drawings, pk=int_or_none(body.get('id')))
        if drawing.version != int_or_none(body.get('version')):
            raise ValueError(S.CARTOGRAPHY_STALE)
        geometry = body.get('geometry')
        if not valid_edit(drawing, geometry):
            raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
        drawing.geometry = {'type': geometry['type'], 'coordinates': geometry['coordinates']}
        drawing.version += 1
        drawing.save(update_fields=['geometry', 'version'])
        return {}
    raise Http404


@login_required
@require_writer
@require_POST
def change(request, region_id, action):
    body, error = parse_json_body(request)
    if error:
        return error
    begin_nonce_write(request)
    try:
        with transaction.atomic():
            # Reserve the SQLite writer before checking the editor's snapshot.
            if not Region.objects.filter(pk=region_id).update(name=F('name')):
                raise Http404
            extra = edit_workspace(region_id, action, body, request.user)
            return success_response(request, body, extra=extra)
    except ValueError as exc:
        return validation_error([str(exc)])
