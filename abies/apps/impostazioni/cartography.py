"""Writer workspace for the selected compresa."""
from itertools import groupby

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import F, Q
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET, require_POST

from apps.base.auth import require_writer
from apps.base.digests import mark_stale
from apps.base.idempotency import begin_nonce_write
from apps.base.models import MapDrawing, MapFeature, MapPointSet, Region, RoadType
from apps.base.numparse import int_or_none
from apps.base.responses import parse_json_body, success_response, validation_error
from apps.ipso.models import IpsoUpload, IpsoUploadMode, IpsoUploadState, IpsoUploadTargetType
from apps.ipso.staging import claim_upload_import, lock_prepared_upload, read_staged_payload
from config import strings as S
from config.constants import DATA_ID_IPSO_UPLOADS, DATA_ID_ROAD_TYPES


@login_required
@require_writer
@require_GET
def catalog(request):
    return JsonResponse({'regions': list(Region.objects.order_by('name').values('id', 'name')),
                         'road_types': list(RoadType.objects.filter(active=True).values('id', 'name'))})


def required_name(body):
    name = body.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError(S.ERR_NAME_REQUIRED)
    if len(name.strip()) > 100:
        raise ValueError(S.ERR_NAME_MAX_LENGTH.format(100))
    return name.strip()


@login_required
@require_writer
@require_POST
def publish_region(request):
    body, error = parse_json_body(request)
    if error:
        return error
    try:
        name = required_name(body)
    except ValueError as exc:
        return validation_error([str(exc)])
    begin_nonce_write(request)
    with transaction.atomic():
        region, created = Region.objects.get_or_create(name=name)
        if not created:
            return validation_error([S.CARTOGRAPHY_REGION_EXISTS])
        return success_response(request, body, extra={'region_id': region.pk})


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
@transaction.atomic
def workspace(request, region_id):
    region = get_object_or_404(Region, pk=region_id)
    return JsonResponse({
        'uploads': pending_uploads(region.pk),
        'sources': list(MapPointSet.objects.filter(region=region).order_by('-date', '-pk')
                        .values('id', 'name', 'operator', 'date')),
        'drawings': list(MapDrawing.objects.filter(region=region).order_by('source_id', 'sequence', 'pk').values(
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


def prepare_point_import(region_id, body):
    upload = get_object_or_404(IpsoUpload, pk=int_or_none(body.get('upload_id')))
    if upload.mode != IpsoUploadMode.POINTS or upload.state != IpsoUploadState.RECEIVED:
        raise ValueError(S.IPSO_ERR_UPLOAD_NOT_RECEIVED)
    payload, error = read_staged_payload(upload)
    if error:
        raise ValueError(error)
    session = payload['session']
    if session.get('region_id') != region_id:
        raise ValueError(S.CARTOGRAPHY_WRONG_REGION)
    return upload, payload


def import_points(region_id, upload, payload, user):
    session = payload['session']
    source, _ = MapPointSet.objects.get_or_create(session_id=upload.session_id, defaults={
        'region_id': region_id, 'name': session['name'], 'date': upload.record_date,
        'operator': upload.operator, 'records': payload['records'],
    })
    if source.region_id != region_id or source.records != payload['records']:
        raise ValueError(S.IPSO_ERR_IMPORT_CONTEXT_CHANGED)
    added = restore_drawings(source)
    claim_upload_import(upload, user, IpsoUploadTargetType.POINT_SET, source.pk)
    mark_stale('audit')
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
    if (not isinstance(selected, list) or not versions or selected != versions
            or any(type(value) is not int for pair in selected for value in pair)):
        raise ValueError(S.CARTOGRAPHY_STALE)


def edit_workspace(region_id, action, body):
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
    if action == 'publish':
        drafts = MapFeature.objects.filter(region_id=region_id, published=False)
        check_selection(drafts, body.get('versions'))
        for road in drafts:
            road.published = True
            road.version += 1
            road.save(update_fields=['published', 'version'])
        mark_stale('audit')
        return {}
    if action == 'discard-road':
        road = get_object_or_404(MapFeature, pk=int_or_none(body.get('id')), region_id=region_id)
        if road.published or road.version != int_or_none(body.get('version')):
            raise ValueError(S.CARTOGRAPHY_STALE)
        road.delete()
        mark_stale('audit')
        return {}
    if action in ('adjust', 'road', 'remove-drawing'):
        drawing = get_object_or_404(drawings, pk=int_or_none(body.get('id')))
        if drawing.version != int_or_none(body.get('version')):
            raise ValueError(S.CARTOGRAPHY_STALE)
        if action == 'remove-drawing':
            drawing.delete()
            return {}
        if action == 'road':
            if (drawing.geometry['type'] != 'LineString'
                    or len(set(map(tuple, drawing.geometry['coordinates']))) < 2):
                raise ValueError(S.CARTOGRAPHY_GEOMETRY_INVALID)
            kind = RoadType.objects.filter(pk=int_or_none(body.get('road_type_id')), active=True).first()
            if kind is None:
                raise ValueError(S.CARTOGRAPHY_ROAD_TYPE_REQUIRED)
            road = MapFeature.objects.create(region_id=region_id, road_type=kind, name=required_name(body),
                                             geometry=drawing.geometry, published=False)
            drawing.delete()
            mark_stale('audit')
            return {'road_id': road.pk}
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
    try:
        if action == 'import':
            prepared, payload = prepare_point_import(region_id, body)
        begin_nonce_write(request)
        with transaction.atomic():
            # Reserve the SQLite writer before checking the editor's snapshot.
            if not Region.objects.filter(pk=region_id).update(name=F('name')):
                raise Http404
            if action == 'import':
                upload, error = lock_prepared_upload(prepared)
                if error:
                    return error
                extra = import_points(region_id, upload, payload, request.user)
            else:
                extra = edit_workspace(region_id, action, body)
            invalidations = {'import': [DATA_ID_IPSO_UPLOADS, 'audit'],
                             'publish': ['roads', 'audit'], 'road': [DATA_ID_ROAD_TYPES, 'audit'],
                             'discard-road': [DATA_ID_ROAD_TYPES, 'audit']}
            return success_response(request, body, extra=extra,
                                    invalidates={'data_ids': invalidations.get(action, [])})
    except ValueError as exc:
        return validation_error([str(exc)])
