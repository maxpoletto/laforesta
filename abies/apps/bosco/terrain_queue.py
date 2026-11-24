"""Schedule and claim SQLite-backed work for Cartografia and the terrain worker."""
import hashlib
import json
from collections import defaultdict
from datetime import date, time

from django.conf import settings
from django.db import transaction
from django.db.models import F, Q
from django.utils import timezone

from apps.base.digests import mark_stale
from apps.base.models import MapFeature, Region, TerrainRefresh
from apps.bosco.hillshade import output_path as hillshade_path
from apps.bosco.map_context_data import context_is_current


State = TerrainRefresh.State


def seasons(now):
    local = timezone.localtime(now)
    today = local.date()
    for year in range(2015, today.year + 1):
        for start, end, cloud in ((date(year, 1, 1), date(year, 3, 1), 20),
                                  (date(year, 6, 1), date(year, 8, 1), 10)):
            if today > end or (today == end and local.time() >= time(2)):
                yield start, end, cloud


def map_features(region_id=None):
    rows = MapFeature.objects.filter(published=True).filter(
        Q(parcel__isnull=False) | Q(region__isnull=False),
    )
    if region_id is not None:
        rows = rows.filter(Q(parcel__region_id=region_id) | Q(region_id=region_id))
    return rows.order_by('pk')


def has_parcels(region_id):
    return MapFeature.objects.filter(published=True, parcel__region_id=region_id).exists()


def signatures(rows):
    by_region = defaultdict(list)
    eligible = set()
    for pk, version, parcel, parcel_region, road_region in rows:
        region_id = parcel_region if parcel else road_region
        by_region[region_id].append((pk, version, parcel))
        if parcel:
            eligible.add(region_id)
    return {region: hashlib.sha256(json.dumps(by_region[region]).encode()).hexdigest()
            for region in eligible}


def map_signatures(region_id=None):
    return signatures(map_features(region_id).values_list('pk', 'version', 'parcel_id', 'parcel__region_id', 'region_id'))


def revision_dir(map_key, revision):
    return settings.SATELLITE_DIR / str(map_key) / 'revisions' / str(revision)


@transaction.atomic
def enqueue(region_id, *, requested_by=None):
    # Establish the SQLite write lock before checking a concurrent request.
    if not Region.objects.filter(pk=region_id).update(name=F('name')):
        return None
    if not has_parcels(region_id):
        return None
    job = TerrainRefresh.objects.filter(region_id=region_id).first()
    if job and job.state in (State.QUEUED, State.RUNNING):
        return job
    job = job or TerrainRefresh(region_id=region_id)
    job.state, job.retry_at, job.failure = State.QUEUED, None, {}
    job.requested_by = requested_by
    if requested_by:
        job._history_user = requested_by
        job.save()
        mark_stale('audit')
    else:
        job.save_without_historical_record()
    return job


def schedule(now):
    inputs = map_signatures()
    completed = list(seasons(now))
    overnight = 2 <= timezone.localtime(now).hour < 6
    due_season = completed[-1][0] if overnight and completed else None
    jobs = {job.region_id: job for job in TerrainRefresh.objects.select_related('region')
            .filter(region_id__in=inputs)}
    for region_id, source in inputs.items():
        job = jobs.get(region_id)
        if job is None:
            enqueue(region_id)
            continue
        if job.state in (State.QUEUED, State.RUNNING):
            continue
        changed = job.source != source
        missing_hillshade = job.state != State.FAILED and not hillshade_path(job.region.map_key).is_file()
        stale_context = (not changed and job.state != State.FAILED
                         and not context_is_current(job.region, settings.MAP_CONTEXT_DIR))
        if job.state == State.WAITING and not (changed or missing_hillshade or stale_context):
            continue
        missing_revision = job.revision_id and not all(
            (revision_dir(job.region.map_key, job.revision_id) / name).is_file()
            for name in ('manifest.json', 'parcel-mask.tif', 'timeseries.json')
        )
        if (changed or missing_hillshade or stale_context or missing_revision or
                (job.state == State.FAILED and job.retry_at and job.retry_at <= now) or
                (job.state != State.FAILED and due_season and
                 (job.searched_through is None or job.searched_through < due_season))):
            enqueue(region_id)


@transaction.atomic
def claim():
    rows = TerrainRefresh.objects.filter(state=State.QUEUED)
    rows.update(state=F('state'))
    job = rows.select_related('region').order_by('finished_at', 'region_id').first()
    if job:
        job.state = State.RUNNING
        rows.filter(pk=job.pk).update(state=State.RUNNING)
    return job
