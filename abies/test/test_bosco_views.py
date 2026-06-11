"""Tests for Bosco API views."""

import io

import numpy as np
import pytest
import rasterio
from django.test import Client
from PIL import Image
from rasterio.transform import from_origin

from apps.base.digests import (
    PRESERVED_TREE_COLUMNS, build_parcel_record, build_preserved_tree_record,
)
from apps.base.models import DigestStatus, Region, Tree
from config.constants import (
    DATA_ID, DELETES, DIGEST_PARCELS, DIGEST_PRESERVED_TREES, FIELD_LAT,
    FIELD_LON, FIELD_NONCE, FIELD_PARCEL_ID, FIELD_SPECIES_ID, FIELD_YEAR,
    HTML, PATCHES, RECORD, ROW_ID, STATUS, STATUS_CONFLICT, VERSION,
)


@pytest.fixture
def reader_client(reader_user):
    c = Client()
    c.force_login(reader_user)
    return c


@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


@pytest.mark.parametrize('path', [
    '/api/bosco/parcels/data/',
    '/api/bosco/species/data/',
    '/api/bosco/preserved-trees/data/',
    '/api/bosco/future-production/data/',
    '/api/bosco/parcel-dendrometry/data/',
    '/api/bosco/parcel-dendrometry-points/data/',
])
def test_bosco_digest_endpoints_reader_access(
        reader_client, path, parcels, species, tmp_path, settings,
):
    settings.DIGEST_DIR = tmp_path

    resp = reader_client.get(path)

    assert resp.status_code == 200
    assert resp['Content-Encoding'] == 'gzip'
    assert resp['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('path', [
    '/api/bosco/parcels/data/',
    '/api/bosco/species/data/',
    '/api/bosco/preserved-trees/data/',
    '/api/bosco/future-production/data/',
    '/api/bosco/parcel-dendrometry/data/',
    '/api/bosco/parcel-dendrometry-points/data/',
])
def test_bosco_digest_endpoints_require_login(client, path):
    resp = client.get(path)
    assert resp.status_code == 302
    assert '/login/' in resp.url


def test_parcel_metadata_form_requires_writer(reader_client, parcels):
    resp = reader_client.get(f'/api/bosco/parcels/metadata/form/{parcels[0].id}/')
    assert resp.status_code == 403


def test_parcel_metadata_form_writer_access(writer_client, parcels):
    resp = writer_client.get(f'/api/bosco/parcels/metadata/form/{parcels[0].id}/')

    assert resp.status_code == 200
    html = resp.json()[HTML]
    assert 'id="bosco-parcel-metadata-form"' in html
    assert f'value="{parcels[0].id}"' in html


def test_parcel_metadata_save_updates_parcel_and_returns_patch(writer_client, parcels):
    parcel = parcels[0]
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'area_ha': '12,50', 'ave_age': '44', 'location_name': 'Costa alta',
        'altitude_min_m': '700', 'altitude_max_m': '920',
        'aspect': 'NE', 'grade_pct': '35',
        'desc_veg': 'Abete e faggio.', 'desc_geo': 'Calcare.',
        FIELD_NONCE: 'parcel-metadata-save',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    parcel.refresh_from_db()
    assert str(parcel.area_ha) == '12.50'
    assert parcel.ave_age == 44
    assert parcel.location_name == 'Costa alta'
    assert parcel.altitude_min_m == 700
    assert parcel.altitude_max_m == 920
    assert parcel.aspect == 'NE'
    assert parcel.grade_pct == 35
    assert parcel.desc_veg == 'Abete e faggio.'
    assert parcel.desc_geo == 'Calcare.'
    assert parcel.version == 2
    data = resp.json()
    patch = data[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_PARCELS
    assert patch[ROW_ID] == parcel.id
    assert patch[RECORD] == build_parcel_record(parcel)
    assert DigestStatus.objects.get(name=DIGEST_PARCELS).stale is True
    assert DigestStatus.objects.get(name='audit').stale is True


def test_parcel_metadata_save_stale_conflicts(writer_client, parcels):
    parcel = parcels[0]
    parcel.version = 3
    parcel.save(update_fields=[VERSION])
    body = {
        ROW_ID: str(parcel.id), VERSION: '2', 'area_ha': '12.50',
        'ave_age': '', 'location_name': '', 'altitude_min_m': '',
        'altitude_max_m': '', 'aspect': '', 'grade_pct': '',
        'desc_veg': '', 'desc_geo': '', FIELD_NONCE: 'parcel-conflict',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    data = resp.json()
    assert data[STATUS] == STATUS_CONFLICT
    assert data[PATCHES][0][DATA_ID] == DIGEST_PARCELS
    assert 'bosco-parcel-metadata-form' in data[HTML]


def test_parcel_metadata_save_validation_error_rerenders(writer_client, parcels):
    parcel = parcels[0]
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version), 'area_ha': '',
        'ave_age': 'abc', 'location_name': '', 'altitude_min_m': '',
        'altitude_max_m': '', 'aspect': '', 'grade_pct': '',
        'desc_veg': '', 'desc_geo': '', FIELD_NONCE: 'parcel-invalid',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    data = resp.json()
    assert 'Superficie obbligatoria.' in data['message']
    assert 'Età media deve essere un numero intero.' in data['message']
    assert 'bosco-parcel-metadata-form' in data[HTML]


def test_parcel_metadata_save_rejects_inverted_altitude(writer_client, parcels):
    parcel = parcels[0]
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version), 'area_ha': '12.50',
        'ave_age': '', 'location_name': '', 'altitude_min_m': '900',
        'altitude_max_m': '800', 'aspect': '', 'grade_pct': '',
        'desc_veg': '', 'desc_geo': '', FIELD_NONCE: 'parcel-altitude-invalid',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    assert 'Altitudine minima maggiore della massima.' in resp.json()['message']


def test_pai_form_requires_writer(reader_client, regions):
    resp = reader_client.get(f'/api/bosco/pai/form/?region_id={regions[0].id}')
    assert resp.status_code == 403


def test_pai_form_writer_access(writer_client, regions, parcels, species):
    resp = writer_client.get(f'/api/bosco/pai/form/?region_id={regions[0].id}')

    assert resp.status_code == 200
    html = resp.json()[HTML]
    assert 'id="bosco-pai-form"' in html
    assert 'Capistrano 1' in html


def test_pai_save_creates_preserved_tree(writer_client, parcels, species):
    body = {
        FIELD_SPECIES_ID: str(species[0].id),
        FIELD_PARCEL_ID: str(parcels[0].id),
        FIELD_YEAR: '2026',
        FIELD_LAT: '38,123456',
        FIELD_LON: '16.123456',
        FIELD_NONCE: 'pai-create',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree = Tree.objects.get(species=species[0], parcel=parcels[0])
    assert tree.preserved is True
    assert tree.year == 2026
    assert tree.lat == 38.12346
    data = resp.json()
    patch = data[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_PRESERVED_TREES
    assert patch[ROW_ID] == tree.id
    assert patch[RECORD] == build_preserved_tree_record(tree)
    assert len(patch[RECORD]) == len(PRESERVED_TREE_COLUMNS)


def test_pai_save_stale_edit_conflicts(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        year=2025, lat=38.1, lon=16.1, version=2,
    )
    body = {
        ROW_ID: str(tree.id), VERSION: '1',
        FIELD_SPECIES_ID: str(species[1].id),
        FIELD_PARCEL_ID: str(parcels[1].id),
        FIELD_YEAR: '2026', FIELD_LAT: '38.2', FIELD_LON: '16.2',
        FIELD_NONCE: 'pai-conflict',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    data = resp.json()
    assert data[STATUS] == STATUS_CONFLICT
    patch = data[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_PRESERVED_TREES
    assert patch[ROW_ID] == tree.id
    assert patch[RECORD] == build_preserved_tree_record(tree)
    assert len(patch[RECORD]) == len(PRESERVED_TREE_COLUMNS)
    assert 'bosco-pai-form' in data[HTML]


def test_pai_delete_clears_preserved_flag(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        year=2025, lat=38.1, lon=16.1, version=3,
    )
    body = {ROW_ID: str(tree.id), VERSION: '3', FIELD_NONCE: 'pai-delete'}

    resp = writer_client.post('/api/bosco/pai/delete/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree.refresh_from_db()
    assert tree.preserved is False
    assert tree.version == 4
    assert resp.json()[DELETES] == [{
        DATA_ID: DIGEST_PRESERVED_TREES,
        ROW_ID: tree.id,
    }]


def _stream_text(resp):
    return b''.join(resp.streaming_content).decode('utf-8')


def _write_test_tif(path, values):
    arr = np.array(values, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
            path, 'w', driver='GTiff', height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype, transform=from_origin(10, 10, 1, 1),
    ) as dst:
        dst.write(arr, 1)


def test_satellite_manifest_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    region_dir.mkdir()
    (region_dir / 'manifest.json').write_text(
        '{"dates":["2026-01-01"],"bbox":[[38,16],[39,17]]}',
    )
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/manifest/')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/json'
    assert resp['Cache-Control'] == 'no-cache'
    assert '"dates"' in _stream_text(resp)


def test_satellite_timeseries_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    region_dir.mkdir()
    (region_dir / 'timeseries.json').write_text(
        '{"dates":["2026-01-01"],"means":{"parcels":{}}}',
    )
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/timeseries/')

    assert resp.status_code == 200
    assert '"means"' in _stream_text(resp)


def test_satellite_diff_png_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    _write_test_tif(region_dir / '2026-01-01' / 'ndvi.tif', [[100, 100], [100, 100]])
    _write_test_tif(region_dir / '2026-07-01' / 'ndvi.tif', [[100, 150], [80, 100]])
    _write_test_tif(region_dir / 'parcel-mask.tif', [[1, 1], [0, 1]])
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(
        f'/api/bosco/satellite/{regions[0].id}/diff/ndvi/2026-01-01/2026-07-01.png',
    )

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'image/png'
    assert resp['Cache-Control'] == 'no-cache'
    assert float(resp['X-Bosco-Max-Abs']) > 0
    image = Image.open(io.BytesIO(resp.content))
    assert image.mode == 'RGBA'
    assert image.size == (2, 2)
    assert image.getpixel((0, 0))[3] == 210
    assert image.getpixel((0, 1))[3] == 60


@pytest.mark.parametrize('url', [
    '/api/bosco/satellite/{id}/diff/bad/2026-01-01/2026-07-01.png',
    '/api/bosco/satellite/{id}/diff/ndvi/20260101/2026-07-01.png',
])
def test_satellite_diff_png_rejects_invalid_segments(reader_client, regions, tmp_path, settings, url):
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(url.format(id=regions[0].id))

    assert resp.status_code == 404


def test_satellite_manifest_conditional_get(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    region_dir.mkdir()
    (region_dir / 'manifest.json').write_text('{"dates":["2026-01-01"]}')
    settings.SATELLITE_DIR = tmp_path

    r1 = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/manifest/')
    r2 = reader_client.get(
        f'/api/bosco/satellite/{regions[0].id}/manifest/',
        HTTP_IF_MODIFIED_SINCE=r1['Last-Modified'],
    )

    assert r2.status_code == 304
    assert r2['Cache-Control'] == 'no-cache'


def test_satellite_unknown_region_404(reader_client, tmp_path, settings):
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get('/api/bosco/satellite/999999/manifest/')

    assert resp.status_code == 404


@pytest.mark.parametrize('path_suffix', [
    'manifest/',
    'timeseries/',
    'diff/ndvi/2026-01-01/2026-07-01.png',
])
def test_satellite_endpoints_require_login(client, regions, path_suffix):
    resp = client.get(f'/api/bosco/satellite/{regions[0].id}/{path_suffix}')
    assert resp.status_code == 302
    assert '/login/' in resp.url


def test_satellite_endpoint_404s_for_missing_file(reader_client, regions, tmp_path, settings):
    (tmp_path / regions[0].name).mkdir()
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{regions[0].id}/manifest/')

    assert resp.status_code == 404


def test_satellite_region_name_cannot_escape_base(reader_client, tmp_path, settings):
    region = Region.objects.create(name='../outside')
    outside = tmp_path.parent / 'outside'
    outside.mkdir(exist_ok=True)
    (outside / 'manifest.json').write_text('{"dates":[]}')
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(f'/api/bosco/satellite/{region.id}/manifest/')

    assert resp.status_code == 404
