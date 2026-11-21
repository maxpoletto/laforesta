"""The writer, scheduler, provider and Bosco share a durable refresh."""
import gzip
import json
from datetime import date, datetime, timedelta, timezone as tz

import numpy as np
import pytest
import rasterio
from django.core.management import call_command
from django.utils import timezone
from shapely.geometry import box, mapping

from apps.base import digests
from apps.base.models import MapFeature, SatelliteRevision, SatelliteSeason, TerrainRefresh
from apps.bosco import satellite, satellite_grid, satellite_refresh, terrain_queue
from config import strings as S


pytestmark = pytest.mark.django_db


def polygon(x=16):
    return {'type': 'Polygon', 'coordinates': [[[x, 38], [16.004, 38], [16.004, 38.004], [x, 38.004], [x, 38]]]}


def test_writer_can_queue_only_published_parcels(client, reader_user, writer_user, parcels):
    parcel = parcels[0]
    url = f'/api/impostazioni/cartography/{parcel.region_id}/terrain/'
    client.force_login(reader_user)
    assert client.get(url).status_code == 403
    assert client.post(url, {}, content_type='application/json').status_code == 403
    client.force_login(writer_user)
    assert not client.get(url).json()['available']
    assert client.post(url, {}, content_type='application/json').status_code == 400
    outline = MapFeature.objects.create(parcel=parcel, geometry=polygon(), published=False)
    assert not client.get(url).json()['available']
    outline.published = True
    outline.save(update_fields=['published'])
    queued = client.post(url, {}, content_type='application/json')
    assert queued.status_code == 202
    assert queued.json() == {'available': True, 'state': 'queued', 'message': 'Aggiornamento in elaborazione.'}
    assert client.get(url.removesuffix('terrain/')).json()['terrain'] == queued.json()
    assert client.post(url, {}, content_type='application/json').status_code == 202
    assert terrain_queue.claim().region_id == parcel.region_id
    assert client.post(url, {}, content_type='application/json').json()['state'] == 'running'
    assert TerrainRefresh.objects.count() == 1
    assert TerrainRefresh.history.count() == 1


def test_nightly_schedule_checks_missed_windows_and_detects_map_edits(parcels, settings):
    settings.TIME_ZONE = 'UTC'
    outline = MapFeature.objects.create(parcel=parcels[0], geometry=polygon())
    region_id = parcels[0].region_id
    now = datetime(2026, 8, 2, 12, tzinfo=tz.utc)
    TerrainRefresh.objects.create(region_id=region_id, state='current',
                                 source=terrain_queue.map_signatures()[region_id],
                                 searched_through=date(2026, 1, 1))
    terrain_queue.schedule(now)
    assert terrain_queue.claim() is None
    terrain_queue.schedule(now.replace(hour=2))
    assert terrain_queue.claim().region_id == region_id
    job = TerrainRefresh.objects.get()
    job.state, job.searched_through = 'current', date(2026, 6, 1)
    job.save()
    outline.version += 1
    outline.save(update_fields=['version'])
    terrain_queue.schedule(now)
    assert terrain_queue.claim().region_id == region_id
    windows = list(terrain_queue.seasons(now))
    assert windows[0] == (date(2015, 1, 1), date(2015, 3, 1), 20)
    assert windows[-1] == (date(2026, 6, 1), date(2026, 8, 1), 10)


def test_catalog_ranks_using_every_page_and_enough_tiles(monkeypatch):
    from unittest.mock import Mock

    def product(day, footprint, cloud):
        return {'ContentDate': {'Start': day + 'T10:00:00Z'}, 'GeoFootprint': mapping(footprint),
                'Attributes': [{'Name': 'cloudCover', 'Value': cloud}]}

    second_page = satellite.CATALOG_URL + '?$skip=50'
    responses = [
        Mock(**{'json.return_value': {'value': [
            product('2024-01-03', box(0, 0, 0.5, 1), 1),
            product('2024-01-04', box(0, 0, 1, 1), 11),
        ], '@odata.nextLink': second_page}}),
        Mock(**{'json.return_value': {'value': [
            product('2024-01-03', box(0.5, 0, 1, 1), 9),
            product('2024-01-05', box(0, 0, 1, 1), 21),
        ]}}),
    ]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(satellite.requests, 'get', get)
    assert satellite.catalog_dates((0, 0, 1, 1), box(0, 0, 1, 1),
                                   date(2024, 1, 1), date(2024, 3, 1), 20) == [
        ('2024-01-03', 9), ('2024-01-04', 11),
    ]
    assert get.call_args.args == (second_page,)


def test_catalog_http_error_keeps_the_response_status(monkeypatch):
    response = satellite.requests.Response()
    response.status_code = 503
    monkeypatch.setattr(satellite.requests, 'get', lambda *args, **kwargs: response)
    with pytest.raises(satellite.SatelliteError) as error:
        satellite.catalog_dates((0, 0, 1, 1), box(0, 0, 1, 1), date(2024, 1, 1), date(2024, 3, 1), 20)
    assert (error.value.kind, error.value.status) == ('catalog', 503)


def test_expansion_downloads_only_uncovered_windows(tmp_path, monkeypatch):
    old = satellite_grid.Grid((16, 38, 16.004, 38.004), 4, 4)
    new = old.expand((15.998, 37.998, 16.006, 38.006))
    source = tmp_path / 'source'
    source.mkdir()
    for name in satellite.EXPECTED_TIFFS:
        satellite.save_geotiff(source / name, np.full((4, 4), 123, dtype=np.uint8),
                               old.bounds, *old.dimensions)
    requested = []

    def fetch(bounds, width, height, date, band, token, **_):
        if band == 'dataMask':
            requested.append((bounds, width, height))
        return np.ones((height, width), dtype=np.float32)

    monkeypatch.setattr(satellite, 'fetch_band', fetch)
    dest = tmp_path / '2026-01-01'
    satellite_grid.download_date(dest, new, np.ones(new.dimensions[::-1], dtype=bool), 'token', 20,
                                 previous=source, previous_grid=old)
    assert len(requested) == 4
    assert sum(width * height for _, width, height in requested) == new.width * new.height - 16
    with rasterio.open(dest / 'ndvi.tif') as raster:
        assert raster.read(1, window=new.window(old)).tolist() == [[123] * 4] * 4


def test_worker_publishes_and_bosco_pins_revisions(client, writer_user, parcels, monkeypatch, settings):
    parcel = parcels[0]
    region = parcel.region
    geometry = polygon()
    # Imported outlines can mix coordinates with and without elevation.
    geometry['coordinates'][0][1].append(866)
    outline = MapFeature.objects.create(parcel=parcel, geometry=geometry)
    refresh_url = f'/api/impostazioni/cartography/{region.pk}/terrain/'
    base = f'/api/bosco/satellite/{region.pk}/'
    client.force_login(writer_user)
    assert client.post(refresh_url, {}, content_type='application/json').status_code == 202
    # Simulate restarting while the job was claimed.
    TerrainRefresh.objects.update(state='running')
    monkeypatch.setattr(terrain_queue, 'seasons', lambda now: [(date(2015, 1, 1), date(2015, 3, 1), 20)])
    monkeypatch.setattr(satellite_refresh, 'load_credentials', lambda: ('local-id', 'local-secret'))
    monkeypatch.setattr(satellite, 'get_access_token', lambda pair: 'token')
    candidates = []

    def catalog(bounds, footprint, start, end, cloud):
        candidates.append((start, cloud))
        return [('2015-02-01', 2), ('2015-02-02', 3)]

    def fetch(bounds, width, height, day, band, token, **_):
        return np.full((height, width), 0 if day.endswith('-01') else 1, dtype=np.float32)

    monkeypatch.setattr(satellite, 'catalog_dates', catalog)
    monkeypatch.setattr(satellite, 'fetch_band', fetch)
    call_command('run_terrain_worker', once=True)
    published = json.loads(b''.join(client.get(base + 'manifest/').streaming_content))
    revision = published['revision']
    assert published['dates'] == ['2015-02-02']
    assert SatelliteSeason.objects.get().date == date(2015, 2, 2)
    assert client.get(refresh_url).json()['state'] == 'current'
    assert client.get(base + f'timeseries/?revision={revision}').status_code == 200
    assert client.get(base + f'raw/ndvi/2015-02-02.json?revision={revision}').status_code == 200

    with monkeypatch.context() as retry:
        def fail(*_):
            raise satellite.SatelliteError('Private upstream details', kind='download', status=503)

        retry.setattr(satellite_refresh, 'build', fail)
        assert client.post(refresh_url, {}, content_type='application/json').status_code == 202
        call_command('run_terrain_worker', once=True)
        assert client.get(refresh_url).json() == {
            'available': True, 'state': 'failed',
            'message': 'Impossibile scaricare le immagini da Copernicus (HTTP 503).',
        }
        assert TerrainRefresh.objects.get().revision_id == SatelliteRevision.objects.get().pk

    digests.generate_audit()
    with gzip.open(settings.DIGEST_DIR / 'audit.json.gz', 'rt') as stream:
        audit = json.load(stream)
    events = [dict(zip(audit['columns'], row)) for row in audit['rows']
              if S.TERRAIN_REFRESH_AUDIT in row]
    assert len(events) == 4
    assert [event[S.COL_USER] for event in events] == [writer_user.get_full_name() or writer_user.username] * 4
    assert all(region.name in event[S.COL_NEW_VALUE] for event in events)
    assert S.TERRAIN_STATE_FAILED in events[0][S.COL_NEW_VALUE]
    assert 'Stato: In elaborazione' in events[1][S.COL_NEW_VALUE]
    assert S.TERRAIN_STATE_CURRENT in events[2][S.COL_NEW_VALUE]

    outline.geometry = polygon(16.001)
    outline.version += 1
    outline.save(update_fields=['geometry', 'version'])
    assert client.get(base + f'timeseries/?revision={revision}').status_code == 404
    call_command('run_terrain_worker', once=True)
    latest = json.loads(b''.join(client.get(base + 'manifest/').streaming_content))
    assert latest['revision'] != revision
    assert latest['dates'] == published['dates']
    assert client.get(base + f'raw/ndvi/2015-02-02.json?revision={revision}').status_code == 200
    assert candidates == [(date(2015, 1, 1), 20)]

    SatelliteRevision.objects.filter(pk=revision).update(superseded_at=timezone.now() - timedelta(hours=25))
    satellite_refresh.prune(timezone.now())
    assert client.get(base + f'raw/ndvi/2015-02-02.json?revision={revision}').status_code == 404
    assert json.loads(b''.join(client.get(base + 'manifest/').streaming_content))['revision'] == latest['revision']


def test_existing_imagery_is_reused_without_cloud_credentials(parcels, settings, monkeypatch):
    parcel = parcels[0]
    MapFeature.objects.create(parcel=parcel, geometry=polygon())
    root = settings.SATELLITE_DIR / str(parcel.region.map_key)
    old_date = root / '2019-02-01'
    old_date.mkdir(parents=True)
    for filename in satellite.EXPECTED_TIFFS:
        satellite.save_geotiff(old_date / filename, np.full((8, 8), 123, dtype=np.uint8),
                               (15.99, 37.99, 16.01, 38.01), 8, 8)

    def unexpected(*_args, **_kwargs):
        pytest.fail('A local refresh must not contact Copernicus')

    monkeypatch.setattr(satellite, 'get_access_token', unexpected)
    monkeypatch.setattr(satellite, 'catalog_dates', unexpected)
    call_command('run_terrain_worker', once=True)
    job = TerrainRefresh.objects.get()
    assert job.state == 'credentials'
    revision = terrain_queue.revision_dir(parcel.region.map_key, job.revision_id)
    adopted = revision / old_date.name / 'ndvi.tif'
    assert adopted.read_bytes() == (old_date / 'ndvi.tif').read_bytes()
    assert adopted.stat().st_ino != (old_date / 'ndvi.tif').stat().st_ino
