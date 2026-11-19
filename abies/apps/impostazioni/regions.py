"""Compresa forms and reviewed deletion of their cartography."""
import hashlib
import json

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import F, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET, require_POST

from apps.base.auth import require_writer
from apps.base.digests import mark_stale, mark_stale_prefixes
from apps.base.idempotency import begin_nonce_write
from apps.base.map_import_v1 import parcel_key
from apps.base.models import (
    HarvestPlanItem, HypsoParam, MapDrawing, MapFeature, MapPointSet, Observation,
    Parcel, ParcelPlanDetail, Region, SampleArea, TreeMark, TreeSample,
)
from apps.base.responses import conflict_response, parse_json_body, submitted_version, success_response, validation_error
from apps.ipso.staging import region_has_pending_fieldwork
from apps.prelievi.models import Harvest
from config import strings as S
from config.constants import (
    BOSCO_TREE_DIGESTS, DATA_ID_ECLASSES, DATA_ID_ROAD_TYPES, DIGEST_FUTURE_PRODUCTION,
    DIGEST_HYPSO_PARAMS, DIGEST_PARCELS, DIGEST_PREFIX_MARK_TREES, DIGEST_PREFIX_SAMPLED_TREES,
)


def required_name(body):
    name = body.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError(S.ERR_NAME_REQUIRED)
    if len(name.strip()) > 100:
        raise ValueError(S.ERR_NAME_MAX_LENGTH.format(100))
    return name.strip()


def unlinked_outlines(name):
    # Map-only installations can have parcel outlines before the CSV parcel catalog exists.
    features = MapFeature.objects.filter(parcel__isnull=True, road_type__isnull=True).filter(
        Q(members__properties__layer=name) | Q(members__properties__compresa=name)
        | Q(members__properties__Compresa=name))
    return [row for row in features.defer('geometry') if parcel_key(row.members)[0] == name]


def has_fieldwork(region):
    for model in (HarvestPlanItem, Harvest):
        if model.objects.filter(Q(region=region) | Q(parcel__region=region)).exists():
            return True
    for model in (SampleArea, TreeSample, TreeMark, ParcelPlanDetail):
        if model.objects.filter(parcel__region=region).exists():
            return True
    return (Observation.objects.filter(region=region).exists() or MapPointSet.objects.filter(region=region).exists()
            or HypsoParam.objects.filter(region=region).exists() or region_has_pending_fieldwork(region.pk))


def region_map(region):
    outlines = [row.pk for row in unlinked_outlines(region.name)]
    return {
        'parcels': Parcel.objects.filter(region=region),
        'outlines': MapFeature.objects.filter(Q(parcel__region=region) | Q(pk__in=outlines)),
        'roads': MapFeature.objects.filter(region=region),
        'drawings': MapDrawing.objects.filter(region=region),
    }


def deletion_preview(content):
    snapshot = {key: list(rows.order_by('pk').values_list('pk', 'version')) for key, rows in content.items()}
    digest = hashlib.sha256(json.dumps(snapshot, separators=(',', ':')).encode()).hexdigest()
    return {**{key: len(rows) for key, rows in snapshot.items()}, 'review': digest}


@login_required
@require_writer
@require_GET
@transaction.atomic
def details(request, region_id):
    region = get_object_or_404(Region, pk=region_id)
    return JsonResponse({'id': region.pk, 'name': region.name, 'description': region.description,
                         'version': region.version,
                         'deletion': None if has_fieldwork(region) else deletion_preview(region_map(region))})


def invalidate_map_labels():
    # Region names are displayed on both region-wide entries and parcel-specific records.
    digests = ['prelievi', DIGEST_PARCELS, 'grids', 'sample_areas', 'samples', 'harvest_plan_items',
               DIGEST_HYPSO_PARAMS, DIGEST_FUTURE_PRODUCTION, *BOSCO_TREE_DIGESTS, 'audit']
    prefixes = [DIGEST_PREFIX_SAMPLED_TREES, DIGEST_PREFIX_MARK_TREES]
    mark_stale(*digests)
    mark_stale_prefixes(*prefixes)
    return {'data_ids': [*digests, 'terreni', 'roads', DATA_ID_ECLASSES, DATA_ID_ROAD_TYPES], 'prefixes': prefixes}


@login_required
@require_writer
@require_POST
def save(request, region_id=None, delete=False):
    body, error = parse_json_body(request)
    if error:
        return error
    if not delete:
        try:
            name = required_name(body)
        except ValueError as exc:
            return validation_error([str(exc)])
        description = body.get('description', '')
        if not isinstance(description, str):
            return validation_error([S.ERR_JSON_FIELD_INVALID])
    begin_nonce_write(request)
    with transaction.atomic():
        region = None
        if region_id is not None:
            if not Region.objects.filter(pk=region_id).update(version=F('version')):
                return JsonResponse({'status': 'not_found', 'message': S.ERROR_CONFLICT}, status=404)
            region = Region.objects.get(pk=region_id)
            if region.version != submitted_version(body):
                return conflict_response(data_id='regions', row_id=region.pk, record=[region.pk, region.version])
        if delete:
            if has_fieldwork(region):
                return validation_error([S.CARTOGRAPHY_REGION_IN_USE])
            content = region_map(region)
            if body.get('review') != deletion_preview(content)['review']:
                return validation_error([S.CARTOGRAPHY_REGION_REVIEW])
            for key in ('drawings', 'roads', 'outlines', 'parcels'):
                content[key].delete()
            region.delete()
            return success_response(request, body, invalidates=invalidate_map_labels())

        if region is None:
            # Start with a write even when the name is free so the SQLite snapshot
            # cannot become stale between checking the old map and inserting the row.
            if Region.objects.filter(name=name).update(version=F('version')):
                return validation_error([S.CARTOGRAPHY_REGION_EXISTS])
            region = Region.objects.create(name=name, description=description.strip())
        else:
            if region.name != name:
                if Region.objects.filter(name=name).exists() or unlinked_outlines(name):
                    return validation_error([S.CARTOGRAPHY_REGION_EXISTS])
                for outline in unlinked_outlines(region.name):
                    _, parcel = parcel_key(outline.members)
                    outline.members['properties'].update(layer=name, name=f'{name}-{parcel}')
                    outline.version += 1
                    outline.save(update_fields=['members', 'version'])
            renamed = region.name != name
            region.name = name
            region.description = description.strip()
            region.version += 1
            region.save()
            if renamed:
                return success_response(request, body, extra={'region_id': region.pk},
                                        invalidates=invalidate_map_labels())
        mark_stale('audit')
        return success_response(request, body, extra={'region_id': region.pk}, invalidates={'data_ids': ['audit']})
