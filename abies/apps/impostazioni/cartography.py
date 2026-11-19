"""Writer workspace for the selected compresa."""

from shapely.geometry import shape

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import F, Q
from django.db.models.functions import Lower
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET, require_POST

from apps.base.auth import require_writer
from apps.base.digests import mark_stale
from apps.base.idempotency import begin_nonce_write
from apps.base.models import MapDrawing, MapFeature, MapPointSet, Parcel, Region, RoadType
from apps.base.numparse import int_or_none
from apps.base.responses import parse_json_body, success_response, validation_error
from apps.impostazioni.drawing_geometry import checked_parts, feature_geometry, walk_parts
from apps.base.map_import_v1 import parcel_key
from apps.bosco.parcel_metadata import parse_parcel_metadata, render_parcel_metadata_form
from apps.impostazioni.parcel_boundaries import boundary_workspace, parcel_outlines, save_boundaries
from apps.impostazioni.regions import invalidate_map_labels, required_name, unlinked_outlines
from apps.ipso.models import IpsoUpload, IpsoUploadMode, IpsoUploadState, IpsoUploadTargetType
from apps.ipso.staging import claim_upload_import, lock_prepared_upload, read_staged_payload
from config import strings as S
from config.constants import DATA_ID_ECLASSES, DATA_ID_IPSO_UPLOADS, DATA_ID_ROAD_TYPES, DIGEST_FUTURE_PRODUCTION, DIGEST_PARCELS, HTML, PENDING_COUNT


@login_required
@require_writer
@require_GET
def catalog(request):
    return JsonResponse({'regions': list(Region.objects.order_by(Lower('name'), 'name').values('id', 'name')),
                         'road_types': list(RoadType.objects.filter(active=True).values('id', 'name'))})


@login_required
@require_writer
@require_GET
def parcel_form(request, region_id):
    parcel = Parcel(region=get_object_or_404(Region, pk=region_id))
    return JsonResponse({HTML: render_parcel_metadata_form(request, parcel, promotion=True)})


def parcel_used(parcel):
    # Do not let a deletion cascade through an otherwise unprotected grid or plan.
    return any(
        relation.related_model._base_manager.filter(**{relation.field.name: parcel}).exists()
        for relation in Parcel._meta.related_objects if relation.get_accessor_name() != 'map_features'
    )


@login_required
@require_writer
@require_GET
@transaction.atomic
def parcel_details(request, region_id, parcel_id, *, boundaries=False):
    parcel = get_object_or_404(Parcel.objects.select_related('region', 'eclass'),
                               region_id=region_id, pk=parcel_id)
    if boundaries:
        return JsonResponse(boundary_workspace(parcel, parcel_outlines(region_id)))
    return JsonResponse({HTML: render_parcel_metadata_form(request, parcel, edit_name=True,
                                                          can_delete=not parcel_used(parcel))})


def parcel_name(body, region, existing_id=None):
    name = required_name(body)
    limit = Parcel._meta.get_field('name').max_length
    if len(name) > limit:
        raise ValueError(S.ERR_NAME_MAX_LENGTH.format(limit))
    if Parcel.objects.filter(region=region, name=name).exclude(pk=existing_id).exists() or any(
            parcel_key(row.members)[1] == name for row in unlinked_outlines(region.name)):
        raise ValueError(S.CARTOGRAPHY_PARCEL_EXISTS)
    return name


def road_type(body, current=None):
    pk = int_or_none(body.get('road_type_id'))
    types = RoadType.objects.filter(Q(active=True) | Q(pk=current))
    kind = types.filter(pk=pk).first()
    if kind is None:
        raise ValueError(S.CARTOGRAPHY_ROAD_TYPE_REQUIRED)
    return kind


def edit_road(region_id, action, body):
    road = get_object_or_404(MapFeature, region_id=region_id, pk=int_or_none(body.get('id')))
    if road.version != int_or_none(body.get('version')):
        raise ValueError(S.CARTOGRAPHY_STALE)
    if action == 'remove-road':
        road.delete()
        return {}
    if action == 'road-properties':
        road.name = required_name(body)
        road.road_type = road_type(body, road.road_type_id)
        fields = ['name', 'road_type']
    else:
        road.geometry = feature_geometry(checked_parts(body.get('parts')), 'LineString')
        fields = ['geometry']
    road.version += 1
    road.save(update_fields=[*fields, 'version'])
    return {'road_id': road.pk}


def edit_parcel(region_id, action, body):
    parcel = get_object_or_404(Parcel, region_id=region_id, pk=int_or_none(body.get('id')))
    if action == 'parcel-boundaries':
        save_boundaries(parcel, body)
        return {'parcel_id': parcel.pk, 'parcel_name': parcel.name}
    if parcel.version != int_or_none(body.get('version')):
        raise ValueError(S.CARTOGRAPHY_STALE)
    if action == 'remove-parcel':
        if parcel_used(parcel):
            raise ValueError(S.CARTOGRAPHY_PARCEL_IN_USE)
        parcel.map_features.all().delete()
        parcel.delete()
        return {'parcel_map_changed': True}
    name = required_name(body)
    if name != parcel.name:
        if body.get('confirm_rename') is not True:
            raise ValueError(S.CARTOGRAPHY_CONFIRM_RENAME)
        name = parcel_name(body, parcel.region, parcel.pk)
    values, errors = parse_parcel_metadata(body)
    if errors:
        raise ValueError(' '.join(errors))
    for field, value in values.items():
        setattr(parcel, field, value)
    parcel.name = name
    parcel.version += 1
    parcel.save(update_fields=['name', *values, 'version'])
    return {'parcel_id': parcel.pk, 'parcel_name': parcel.name}


def promote_parcel(drawing, parts, body):
    geometry = feature_geometry(parts, 'Polygon')
    targets = Parcel.objects.filter(region=drawing.region)
    if body.get('parcel_id') not in (None, ''):
        parcel = get_object_or_404(targets, pk=int_or_none(body['parcel_id']))
        if parcel.version != int_or_none(body.get('parcel_version')):
            raise ValueError(S.CARTOGRAPHY_STALE)
        piece = shape(geometry)
        existing = parcel.map_features.filter(published=True).values_list('geometry', flat=True)
        if any(piece.relate_pattern(shape(outline), 'T********') for outline in existing):
            raise ValueError(S.CARTOGRAPHY_PIECE_OVERLAPS)
        parcel.version += 1
        parcel.save(update_fields=['version'])
    else:
        name = parcel_name(body, drawing.region)
        values, errors = parse_parcel_metadata(body)
        if errors:
            raise ValueError(' '.join(errors))
        parcel = targets.create(region=drawing.region, name=name, **values)
    MapFeature.objects.create(parcel=parcel, geometry=geometry, members={'type': 'Feature'})
    drawing.delete()
    return {'parcel_id': parcel.pk, 'parcel_name': parcel.name}


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
        'drawings': list(MapDrawing.objects.filter(region=region).order_by('pk').values(
            'id', 'source_id', 'name', 'parts', 'version')),
        'roads': list(MapFeature.objects.filter(region=region).values(
            'id', 'name', 'road_type_id', 'road_type__name', 'geometry', 'published', 'version')),
        'parcels': list(Parcel.objects.filter(region=region).values('id', 'name')),
    })


def restore_drawings(source):
    drawing, created = MapDrawing.objects.get_or_create(
        source=source, defaults={'region_id': source.region_id, 'name': source.name,
                                 'parts': lambda: walk_parts(source.records)})
    return {'drawing_id': drawing.pk, 'added': int(created)}


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
    result = restore_drawings(source)
    claim_upload_import(upload, user, IpsoUploadTargetType.POINT_SET, source.pk)
    mark_stale('audit')
    return {**result, 'source_id': source.pk,
            PENDING_COUNT: IpsoUpload.objects.filter(state=IpsoUploadState.RECEIVED).count()}


def check_selection(rows, selected):
    versions = [list(pair) for pair in rows.order_by('pk').values_list('pk', 'version')]
    if (not isinstance(selected, list) or not versions or selected != versions
            or any(type(value) is not int for pair in selected for value in pair)):
        raise ValueError(S.CARTOGRAPHY_STALE)


def edit_workspace(region_id, action, body):
    if action in ('road-properties', 'road-points', 'remove-road'):
        return edit_road(region_id, action, body)
    if action in ('parcel-properties', 'parcel-boundaries', 'remove-parcel'):
        return edit_parcel(region_id, action, body)
    if action in ('restore', 'forget'):
        source = get_object_or_404(MapPointSet, region_id=region_id, pk=int_or_none(body.get('source_id')))
        if action == 'restore':
            return restore_drawings(source)
        source.delete()
        return {}
    drawings = MapDrawing.objects.filter(region_id=region_id)
    if action == 'clear':
        check_selection(drawings, body.get('versions'))
        drawings.delete()
        return {}
    if action == 'create':
        row = drawings.create(region_id=region_id, name=required_name(body), parts=checked_parts(body.get('parts')))
        return {'drawing_id': row.pk}
    if action in ('adjust', 'road', 'parcel', 'copy', 'remove-drawing'):
        drawing = get_object_or_404(drawings, pk=int_or_none(body.get('id')))
        if drawing.version != int_or_none(body.get('version')):
            raise ValueError(S.CARTOGRAPHY_STALE)
        if action == 'remove-drawing':
            drawing.delete()
            return {}
        parts = checked_parts(body.get('parts', drawing.parts if action in ('road', 'parcel') else None))
        if action == 'copy':
            copy = drawings.create(region_id=region_id, name=required_name(body), parts=parts)
            return {'drawing_id': copy.pk}
        if action == 'parcel':
            return promote_parcel(drawing, parts, body)
        if action == 'road':
            road = MapFeature.objects.create(region_id=region_id, road_type=road_type(body), name=required_name(body),
                                             geometry=feature_geometry(parts, 'LineString'))
            drawing.delete()
            return {'road_id': road.pk}
        drawing.parts = parts
        drawing.version += 1
        drawing.save(update_fields=['parts', 'version'])
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
            if action in ('parcel-properties', 'remove-parcel'):
                invalidates = invalidate_map_labels()
            elif action in ('parcel', 'parcel-boundaries'):
                digests = [DIGEST_PARCELS, 'audit']
                if action == 'parcel':
                    digests.extend(['harvest_plan_items', DIGEST_FUTURE_PRODUCTION])
                mark_stale(*digests)
                invalidates = {'data_ids': ['terreni', DATA_ID_ECLASSES, *digests]}
            elif action in ('road', 'road-properties', 'road-points', 'remove-road'):
                mark_stale('audit')
                invalidates = {'data_ids': ['roads', DATA_ID_ROAD_TYPES, 'audit']}
            elif action == 'import':
                invalidates = {'data_ids': [DATA_ID_IPSO_UPLOADS, 'audit']}
            else:
                invalidates = {}
            return success_response(request, body, extra=extra, invalidates=invalidates)

    except ValueError as exc:
        return validation_error([str(exc)])
