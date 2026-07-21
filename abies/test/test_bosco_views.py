"""Tests for Bosco API views."""

import base64
import re
from datetime import date as date_type
from decimal import Decimal

import numpy as np
import pytest
import rasterio
from django.test import Client
from rasterio.transform import from_origin

from apps.base.digests import (
    PRESERVED_TREE_COLUMNS, build_parcel_record, build_preserved_tree_record,
)
from apps.base.models import DigestStatus, Parcel, Region, Sample, Survey, Tree, TreeSample
from config import strings as S
from config.constants import (
    DATA_ID, DELETES, DIGEST_PARCELS, DIGEST_PRESERVED_TREES, FIELD_DATE,
    FIELD_D_CM, FIELD_ESTIMATED_BIRTH_YEAR, FIELD_ACC_M, FIELD_H_M, FIELD_LAT,
    FIELD_LON, FIELD_NONCE, FIELD_NOTE, FIELD_NUMBER, FIELD_OPERATOR,
    FIELD_PARCEL_ID, FIELD_SPECIES_ID, HTML, MESSAGE, PATCHES, RECORD, ROW_ID,
    STATUS, STATUS_CONFLICT, VERSION,
)


def _pai_row(
        tree, parcel, *, number=7, sample_date='2024-09-15', d_cm=42,
        h_m=Decimal('18.50'), h_measured=True, lat=38.1, lon=16.1,
        acc_m=None, operator='', note='', version=1,
):
    survey = Survey.objects.create(
        name=f'PAI test survey {tree.id}-{number}-{sample_date}',
    )
    if isinstance(sample_date, str):
        sample_date = date_type.fromisoformat(sample_date)
    sample = Sample.objects.create(
        sample_area=None, survey=survey, date=sample_date,
    )
    return TreeSample.objects.create(
        sample=sample, tree=tree, parcel=parcel, number=number,
        preserved_number=number, d_cm=d_cm, h_m=h_m,
        h_measured=h_measured, lat=lat, lon=lon, acc_m=acc_m,
        operator=operator, note=note, version=version,
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
    assert 'name="cutting_plan"' in html
    assert 'name="intervention_interval"' not in html
    assert 'name="standards_per_ha"' not in html


def test_parcel_metadata_form_shows_coppice_fields(writer_client, regions, eclasses):
    parcel = Parcel.objects.create(
        name='C1', region=regions[0], eclass=eclasses[2],
        area_ha=Decimal('1.00'), intervention_interval=18, standards_per_ha=75,
    )

    resp = writer_client.get(f'/api/bosco/parcels/metadata/form/{parcel.id}/')

    assert resp.status_code == 200
    html = resp.json()[HTML]
    assert 'name="intervention_interval"' in html
    assert 'name="standards_per_ha"' in html


def test_parcel_metadata_save_updates_parcel_and_returns_patch(writer_client, parcels):
    parcel = parcels[0]
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'area_ha': '12,50', 'ave_age': '44', 'location_name': 'Costa alta',
        'altitude_min_m': '700', 'altitude_max_m': '920',
        'aspect': 'NE', 'grade_pct': '35',
        'desc_veg': 'Abete e faggio.', 'desc_geo': 'Calcare.',
        'cutting_plan': 'Diradamento selettivo.',
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
    assert parcel.cutting_plan == 'Diradamento selettivo.'
    assert parcel.intervention_interval is None
    assert parcel.standards_per_ha is None
    assert parcel.version == 2
    data = resp.json()
    patch = data[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_PARCELS
    assert patch[ROW_ID] == parcel.id
    assert patch[RECORD] == build_parcel_record(parcel)
    assert DigestStatus.objects.get(name=DIGEST_PARCELS).stale is True
    assert DigestStatus.objects.get(name='audit').stale is True


def test_parcel_metadata_save_updates_coppice_fields(writer_client, regions, eclasses):
    parcel = Parcel.objects.create(
        name='C1', region=regions[0], eclass=eclasses[2],
        area_ha=Decimal('1.00'), intervention_interval=18, standards_per_ha=75,
    )
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'area_ha': '2.00', 'ave_age': '', 'location_name': '',
        'altitude_min_m': '', 'altitude_max_m': '',
        'aspect': '', 'grade_pct': '', 'desc_veg': '', 'desc_geo': '',
        'cutting_plan': 'Taglio di ceduo.',
        'intervention_interval': '12', 'standards_per_ha': '30',
        FIELD_NONCE: 'parcel-coppice-save',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    parcel.refresh_from_db()
    assert parcel.cutting_plan == 'Taglio di ceduo.'
    assert parcel.intervention_interval == 12
    assert parcel.standards_per_ha == 30


def test_parcel_metadata_save_requires_coppice_fields(writer_client, regions, eclasses):
    parcel = Parcel.objects.create(
        name='C1', region=regions[0], eclass=eclasses[2],
        area_ha=Decimal('1.00'), intervention_interval=18, standards_per_ha=75,
    )
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'area_ha': '2.00', 'ave_age': '', 'location_name': '',
        'altitude_min_m': '', 'altitude_max_m': '',
        'aspect': '', 'grade_pct': '', 'desc_veg': '', 'desc_geo': '',
        'cutting_plan': '', 'intervention_interval': '', 'standards_per_ha': '',
        FIELD_NONCE: 'parcel-coppice-invalid',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    assert S.COL_INTERVENTION_INTERVAL in resp.json()[MESSAGE]
    assert S.COL_STANDARDS_PER_HA in resp.json()[MESSAGE]


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
    assert S.ERR_BOSCO_AREA_REQUIRED in data[MESSAGE]
    assert (S.ERR_BOSCO_INTEGER_REQUIRED.format(S.LABEL_BOSCO_AVE_AGE)
            in data[MESSAGE])
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
    assert S.ERR_BOSCO_ALTITUDE_RANGE in resp.json()[MESSAGE]


def test_pai_form_requires_writer(reader_client, regions):
    resp = reader_client.get(f'/api/bosco/pai/form/?region_id={regions[0].id}')
    assert resp.status_code == 403


def test_pai_form_writer_access(writer_client, regions, parcels, species):
    resp = writer_client.get(
        f'/api/bosco/pai/form/?region_id={regions[0].id}'
        f'&{FIELD_PARCEL_ID}={parcels[0].id}'
        f'&{FIELD_LAT}=38.12345&{FIELD_LON}=16.12345',
    )

    assert resp.status_code == 200
    html = resp.json()[HTML]
    assert 'id="bosco-pai-form"' in html
    assert 'Capistrano 1' in html
    assert f'value="{parcels[0].id}" data-region="{regions[0].id}"\n            selected' in html
    assert re.search(
        r'<input[^>]*id="id_pai_lat"[^>]*name="lat"[^>]*required[^>]*value="38\.12345(?:0)?"',
        html,
    )
    assert re.search(
        r'<input[^>]*id="id_pai_lon"[^>]*name="lon"[^>]*required[^>]*value="16\.12345(?:0)?"',
        html,
    )


def test_pai_save_creates_preserved_tree(writer_client, parcels, species):
    body = {
        FIELD_SPECIES_ID: str(species[0].id),
        FIELD_PARCEL_ID: str(parcels[0].id),
        FIELD_NUMBER: '7',
        FIELD_DATE: '2024-09-15',
        FIELD_ESTIMATED_BIRTH_YEAR: '1920',
        FIELD_D_CM: '42',
        FIELD_H_M: '18,5',
        FIELD_LAT: '38,123456',
        FIELD_LON: '16.123456',
        FIELD_NOTE: 'chioma secca',
        FIELD_NONCE: 'pai-create',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree = Tree.objects.get(species=species[0], parcel=parcels[0])
    assert tree.preserved is True
    assert tree.estimated_birth_year == 1920
    assert tree.lat == 38.12346
    pai = TreeSample.objects.get(tree=tree)
    assert pai.preserved_number == 7
    assert pai.sample.date.isoformat() == '2024-09-15'
    assert pai.d_cm == 42
    assert str(pai.h_m) == '18.50'
    assert pai.note == 'chioma secca'
    data = resp.json()
    patch = data[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_PRESERVED_TREES
    assert patch[ROW_ID] == pai.id
    assert patch[RECORD] == build_preserved_tree_record(pai)
    assert len(patch[RECORD]) == len(PRESERVED_TREE_COLUMNS)


def test_pai_save_defaults_blank_number_to_next_in_parcel(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        lat=38.1, lon=16.1,
    )
    _pai_row(tree, parcels[0], number=7)
    body = {
        FIELD_SPECIES_ID: str(species[1].id),
        FIELD_PARCEL_ID: str(parcels[0].id),
        FIELD_NUMBER: '',
        FIELD_DATE: '2024-09-16',
        FIELD_D_CM: '43',
        FIELD_H_M: '19.0',
        FIELD_LAT: '38.2',
        FIELD_LON: '16.2',
        FIELD_NONCE: 'pai-default-number',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    pai = TreeSample.objects.get(preserved_number=8)
    assert pai.parcel == parcels[0]
    data = resp.json()
    assert data[PATCHES][0][RECORD] == build_preserved_tree_record(pai)


def test_pai_save_updates_preserved_tree(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        estimated_birth_year=1920, lat=38.1, lon=16.1, acc_m=5,
    )
    pai = _pai_row(
        tree, parcels[0], number=7, sample_date='2024-09-15',
        d_cm=42, h_m=Decimal('18.50'), h_measured=True,
        lat=38.1, lon=16.1, acc_m=5, operator='Mario', note='old',
    )
    body = {
        ROW_ID: str(pai.id), VERSION: str(pai.version),
        FIELD_SPECIES_ID: str(species[1].id),
        FIELD_PARCEL_ID: str(parcels[1].id),
        FIELD_NUMBER: '11',
        FIELD_DATE: '2024-10-02',
        FIELD_ESTIMATED_BIRTH_YEAR: '1935',
        FIELD_D_CM: '55',
        FIELD_H_M: '24,75',
        FIELD_LAT: '38,22222',
        FIELD_LON: '16,33333',
        FIELD_ACC_M: '9',
        FIELD_OPERATOR: 'Luisa',
        FIELD_NOTE: 'updated',
        FIELD_NONCE: 'pai-edit-success',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree.refresh_from_db()
    pai.refresh_from_db()
    assert tree.species == species[1]
    assert tree.parcel == parcels[1]
    assert tree.estimated_birth_year == 1935
    assert tree.lat == 38.22222
    assert tree.lon == 16.33333
    assert tree.acc_m == 9
    assert tree.preserved is True
    assert tree.version == 2
    assert pai.parcel == parcels[1]
    assert pai.preserved_number == 11
    assert pai.sample.date.isoformat() == '2024-10-02'
    assert pai.d_cm == 55
    assert str(pai.h_m) == '24.75'
    assert pai.operator == 'Luisa'
    assert pai.note == 'updated'
    assert pai.version == 2
    patch = resp.json()[PATCHES][0]
    assert patch == {
        DATA_ID: DIGEST_PRESERVED_TREES,
        ROW_ID: pai.id,
        RECORD: build_preserved_tree_record(pai),
    }
    assert DigestStatus.objects.get(name=DIGEST_PRESERVED_TREES).stale is True


def test_pai_save_rejects_duplicate_number_in_parcel(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        lat=38.1, lon=16.1,
    )
    _pai_row(tree, parcels[0], number=7)
    body = {
        FIELD_SPECIES_ID: str(species[1].id),
        FIELD_PARCEL_ID: str(parcels[0].id),
        FIELD_NUMBER: '7',
        FIELD_DATE: '2024-09-16',
        FIELD_D_CM: '43',
        FIELD_H_M: '19.0',
        FIELD_LAT: '38.2',
        FIELD_LON: '16.2',
        FIELD_NONCE: 'pai-duplicate-number',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    assert S.ERR_BOSCO_PAI_NUMBER_DUPLICATE in resp.json()[MESSAGE]
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 1


def test_pai_save_stale_edit_conflicts(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        estimated_birth_year=1920, lat=38.1, lon=16.1, version=2,
    )
    pai = _pai_row(
        tree, parcels[0], number=3, sample_date='2024-09-15',
        d_cm=42, h_m=Decimal('18.50'), lat=38.1, lon=16.1, version=2,
    )
    body = {
        ROW_ID: str(pai.id), VERSION: '1',
        FIELD_SPECIES_ID: str(species[1].id),
        FIELD_PARCEL_ID: str(parcels[1].id),
        FIELD_NUMBER: '4', FIELD_DATE: '2024-09-16',
        FIELD_D_CM: '43', FIELD_H_M: '19.0',
        FIELD_LAT: '38.2', FIELD_LON: '16.2',
        FIELD_NONCE: 'pai-conflict',
    }

    resp = writer_client.post('/api/bosco/pai/save/', body,
                              content_type='application/json')

    assert resp.status_code == 400
    data = resp.json()
    assert data[STATUS] == STATUS_CONFLICT
    patch = data[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_PRESERVED_TREES
    assert patch[ROW_ID] == pai.id
    assert patch[RECORD] == build_preserved_tree_record(pai)
    assert len(patch[RECORD]) == len(PRESERVED_TREE_COLUMNS)
    assert 'bosco-pai-form' in data[HTML]


def test_pai_delete_clears_preserved_flag(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0], parcel=parcels[0], preserved=True,
        estimated_birth_year=1925, lat=38.1, lon=16.1, version=3,
    )
    pai = _pai_row(
        tree, parcels[0], number=3, sample_date='2024-09-15',
        d_cm=42, h_m=Decimal('18.50'), lat=38.1, lon=16.1, version=3,
    )
    body = {ROW_ID: str(pai.id), VERSION: '3', FIELD_NONCE: 'pai-delete'}

    resp = writer_client.post('/api/bosco/pai/delete/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    tree.refresh_from_db()
    assert tree.preserved is False
    assert tree.version == 4
    assert TreeSample.objects.filter(preserved_number__isnull=False).count() == 0
    assert resp.json()[DELETES] == [{
        DATA_ID: DIGEST_PRESERVED_TREES,
        ROW_ID: pai.id,
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


def test_satellite_raw_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    _write_test_tif(region_dir / '2026-07-01' / 'ndvi.tif', [[100, 150], [80, 100]])
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(
        f'/api/bosco/satellite/{regions[0].id}/raw/ndvi/2026-07-01.json',
    )

    assert resp.status_code == 200
    assert resp['Cache-Control'] == 'no-cache'
    payload = resp.json()
    assert payload['width'] == 2
    assert payload['height'] == 2
    assert payload['bbox'] == [[8.0, 10.0], [10.0, 12.0]]
    assert base64.b64decode(payload['data']) == bytes([100, 150, 80, 100])


def test_satellite_mask_raw_reader_access(reader_client, regions, tmp_path, settings):
    region_dir = tmp_path / regions[0].name
    _write_test_tif(region_dir / 'parcel-mask.tif', [[1, 0], [1, 1]])
    settings.SATELLITE_DIR = tmp_path

    resp = reader_client.get(
        f'/api/bosco/satellite/{regions[0].id}/raw/parcel-mask.json',
    )

    assert resp.status_code == 200
    assert resp['Cache-Control'] == 'no-cache'
    payload = resp.json()
    assert payload['width'] == 2
    assert payload['height'] == 2
    assert base64.b64decode(payload['data']) == bytes([1, 0, 1, 1])


@pytest.mark.parametrize('url', [
    '/api/bosco/satellite/{id}/raw/bad/2026-07-01.json',
    '/api/bosco/satellite/{id}/raw/ndvi/20260701.json',
])
def test_satellite_raster_endpoints_reject_invalid_segments(reader_client, regions, tmp_path, settings, url):
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
    'raw/parcel-mask.json',
    'raw/ndvi/2026-07-01.json',
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
