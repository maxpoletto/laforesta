"""Tests for Bosco API views."""

import base64
import re
from datetime import date as date_type, datetime, timezone
from decimal import Decimal

import numpy as np
import pytest
import rasterio
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from rasterio.transform import from_origin

from apps.base import csv_io
from apps.base.digests import (
    PRESERVED_TREE_COLUMNS, build_observation_record, build_parcel_record,
    build_preserved_tree_record,
)
from apps.base.models import (
    DigestStatus, Observation, ObservationCategory, ObservationCategoryAssignment,
    ObservationPhoto, Parcel, Region, Sample, Survey, Tree, TreeSample,
)
from config import strings as S
from config.constants import (
    DATA_ID, DELETES, DIGEST_OBSERVATIONS, DIGEST_PARCELS,
    DIGEST_PRESERVED_TREES, FIELD_ACC_M,
    FIELD_CATEGORIES, FIELD_CATEGORY_IDS, FIELD_CLIENT_RECORD_ID,
    FIELD_CONTENT_TYPE, FIELD_DATE,
    FIELD_D_CM, FIELD_ESTIMATED_BIRTH_YEAR, FIELD_H_M, FIELD_HEIGHT_PX,
    FIELD_EXISTING_PHOTO_IDS, FIELD_ID, FIELD_LAT, FIELD_LON, FIELD_NAME,
    FIELD_NONCE, FIELD_NOTE,
    FIELD_NUMBER, FIELD_OPERATOR, FIELD_ORIGINAL_FILENAME, FIELD_PARCEL_ID,
    FIELD_PHOTOS, FIELD_REGION_ID, FIELD_SIZE_BYTES, FIELD_SOURCE,
    FIELD_SPECIES_ID, FIELD_TAKEN_AT,
    FIELD_TEXT, FIELD_URL, FIELD_WIDTH_PX, HTML, MESSAGE, PATCHES, RECORD,
    ROW_ID, STATUS, STATUS_CONFLICT, VERSION,
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
    '/api/bosco/observations/data/',
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


def test_observation_detail_and_photo_reader_access(
        reader_client, parcels, tmp_path, settings):
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    category = ObservationCategory.objects.create(name='sentieri-test', sort_order=10)
    observation = Observation.objects.create(
        date='2026-07-25', text='Frana sul sentiero', lat=38.5, lon=16.3,
        region=parcels[0].region, acc_m=4, operator='Mario', source='ipso',
        client_record_id='rec-1',
    )
    ObservationCategoryAssignment.objects.create(
        observation=observation, category=category,
    )
    photo_path = tmp_path / str(observation.id) / 'photo.jpg'
    photo_path.parent.mkdir(parents=True)
    photo_path.write_bytes(b'jpg')
    photo = ObservationPhoto.objects.create(
        observation=observation,
        file_path=f'{observation.id}/photo.jpg',
        content_type='image/jpeg',
        size_bytes=3,
        width_px=20,
        height_px=10,
        checksum='a' * 64,
        original_filename='campo.jpg',
        lat=38.51, lon=16.31,
        taken_at=datetime(2026, 7, 31, 10, 15, 30, tzinfo=timezone.utc),
    )

    detail = reader_client.get(
        f'/api/bosco/observations/{observation.id}/detail/',
    )

    assert detail.status_code == 200
    payload = detail.json()
    assert payload[ROW_ID] == observation.id
    assert payload[FIELD_DATE] == '2026-07-25'
    assert payload[FIELD_TEXT] == 'Frana sul sentiero'
    assert payload[FIELD_LAT] == 38.5
    assert payload[FIELD_LON] == 16.3
    assert payload[FIELD_REGION_ID] == parcels[0].region_id
    assert payload[FIELD_ACC_M] == 4
    assert payload[FIELD_OPERATOR] == 'Mario'
    assert payload[FIELD_SOURCE] == 'ipso'
    assert payload[FIELD_CLIENT_RECORD_ID] == 'rec-1'
    assert payload[FIELD_CATEGORIES] == [
        {FIELD_ID: category.id, FIELD_NAME: 'sentieri-test'},
    ]
    assert payload[FIELD_PHOTOS] == [{
        FIELD_ID: photo.id,
        FIELD_URL: f'/api/bosco/observations/photos/{photo.id}/',
        FIELD_CONTENT_TYPE: 'image/jpeg',
        FIELD_SIZE_BYTES: 3,
        FIELD_WIDTH_PX: 20,
        FIELD_HEIGHT_PX: 10,
        FIELD_ORIGINAL_FILENAME: 'campo.jpg',
        FIELD_LAT: 38.51,
        FIELD_LON: 16.31,
        FIELD_TAKEN_AT: '2026-07-31T10:15:30+00:00',
    }]

    resp = reader_client.get(payload[FIELD_PHOTOS][0][FIELD_URL])

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'image/jpeg'
    assert resp['Cache-Control'] == 'no-store'
    assert resp['X-Content-Type-Options'] == 'nosniff'
    assert b''.join(resp.streaming_content) == b'jpg'


def test_observation_photo_missing_file_404(reader_client, tmp_path, settings):
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    observation = Observation.objects.create(
        date='2026-07-25', text='Frana sul sentiero', lat=38.5, lon=16.3,
    )
    photo = ObservationPhoto.objects.create(
        observation=observation, file_path='missing/photo.jpg',
        content_type='image/jpeg', size_bytes=3, checksum='a' * 64,
    )

    resp = reader_client.get(f'/api/bosco/observations/photos/{photo.id}/')

    assert resp.status_code == 404


def test_observation_photo_unsafe_content_type_downloads_attachment(
        reader_client, tmp_path, settings):
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    observation = Observation.objects.create(
        date='2026-07-25', text='Frana sul sentiero', lat=38.5, lon=16.3,
    )
    photo_path = tmp_path / str(observation.id) / 'payload.html'
    photo_path.parent.mkdir(parents=True)
    photo_path.write_text('<script>alert(1)</script>')
    photo = ObservationPhoto.objects.create(
        observation=observation, file_path=f'{observation.id}/payload.html',
        content_type='text/html', size_bytes=25, checksum='b' * 64,
    )

    resp = reader_client.get(f'/api/bosco/observations/photos/{photo.id}/')

    assert resp.status_code == 200
    assert resp['Content-Type'] == 'application/octet-stream'
    assert resp['Content-Disposition'] == 'attachment'
    assert resp['Cache-Control'] == 'no-store'
    assert resp['X-Content-Type-Options'] == 'nosniff'


def test_observation_form_requires_writer(reader_client, regions):
    resp = reader_client.get(
        f'/api/bosco/observations/form/?{FIELD_REGION_ID}={regions[0].id}',
    )

    assert resp.status_code == 403


def test_observation_form_writer_access(writer_client, regions, parcels):
    category = ObservationCategory.objects.create(name='rifiuti-test')
    observation = Observation.objects.create(
        date='2026-07-25', text='Rifiuti', lat=38.5, lon=16.3,
        region=regions[0], acc_m=4, version=3,
    )
    observation.categories.add(category)

    add = writer_client.get(
        f'/api/bosco/observations/form/?{FIELD_REGION_ID}={regions[0].id}'
        f'&{FIELD_LAT}=38.12345&{FIELD_LON}=16.12345',
    )
    edit = writer_client.get(
        f'/api/bosco/observations/form/{observation.id}/',
    )

    assert add.status_code == 200
    add_html = add.json()[HTML]
    assert 'id="bosco-observation-form"' in add_html
    assert f'name="region_id" value="{regions[0].id}"' in add_html
    assert 'readonly' in add_html
    assert 'value="38.12345' in add_html
    assert edit.status_code == 200
    edit_html = edit.json()[HTML]
    assert S.BOSCO_OBSERVATION_EDIT_TITLE in edit_html
    assert f'<select id="id_observation_region" name="region_id" required>' in edit_html
    assert f'value="{category.id}"\n              checked' in edit_html


def test_observation_save_creates_manual_observation_with_photo(
        writer_client, writer_user, parcels, tmp_path, settings):
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    category = ObservationCategory.objects.create(name='fitosanitario-test')
    content = b'\xff\xd8\xffjpeg'
    body = {
        FIELD_REGION_ID: str(parcels[0].region_id),
        FIELD_DATE: '2026-08-01',
        FIELD_TEXT: 'Ramo pericolante',
        FIELD_CATEGORY_IDS: [str(category.id)],
        FIELD_LAT: '38,123456',
        FIELD_LON: '16.123456',
        FIELD_NONCE: 'observation-create',
        FIELD_PHOTOS: _jpeg_upload('ramo.jpg', content),
    }

    resp = writer_client.post('/api/bosco/observations/save/', body)

    assert resp.status_code == 200
    observation = Observation.objects.get(text='Ramo pericolante')
    assert observation.region == parcels[0].region
    assert observation.date.isoformat() == '2026-08-01'
    assert observation.lat == 38.12346
    assert observation.lon == 16.12346
    assert observation.source == 'bosco'
    assert observation.created_by == writer_user
    assert observation.operator == writer_user.username
    assert list(observation.categories.all()) == [category]
    photo = ObservationPhoto.objects.get(observation=observation)
    assert photo.content_type == 'image/jpeg'
    assert photo.original_filename == 'ramo.jpg'
    assert (tmp_path / photo.file_path).read_bytes() == content
    patch = resp.json()[PATCHES][0]
    assert patch[DATA_ID] == DIGEST_OBSERVATIONS
    assert patch[ROW_ID] == observation.id
    assert patch[RECORD] == _observation_digest_record(observation)


def test_observation_save_updates_categories_and_photos(
        writer_client, regions, tmp_path, settings):
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    old_category = ObservationCategory.objects.create(name='vecchia')
    new_category = ObservationCategory.objects.create(name='nuova')
    observation = Observation.objects.create(
        date='2026-07-25', text='Prima nota', lat=38.5, lon=16.3,
        region=regions[0], acc_m=4, source='ipso', operator='Mario',
    )
    observation.categories.add(old_category)
    photo1 = _stored_observation_photo(
        tmp_path, observation, 'keep.jpg', b'\xff\xd8\xffkeep', 'a' * 64,
    )
    photo2 = _stored_observation_photo(
        tmp_path, observation, 'drop.jpg', b'\xff\xd8\xffdrop', 'b' * 64,
    )
    body = {
        ROW_ID: str(observation.id),
        VERSION: str(observation.version),
        FIELD_REGION_ID: str(regions[1].id),
        FIELD_DATE: '2026-08-02',
        FIELD_TEXT: 'Nota aggiornata',
        FIELD_CATEGORY_IDS: [str(new_category.id)],
        FIELD_EXISTING_PHOTO_IDS: [str(photo1.id)],
        FIELD_LAT: '38.7',
        FIELD_LON: '16.7',
        FIELD_ACC_M: '9',
        FIELD_NONCE: 'observation-update',
        FIELD_PHOTOS: _jpeg_upload('new.jpg', b'\xff\xd8\xffnew'),
    }

    resp = writer_client.post('/api/bosco/observations/save/', body)

    assert resp.status_code == 200
    observation.refresh_from_db()
    assert observation.region == regions[1]
    assert observation.date.isoformat() == '2026-08-02'
    assert observation.text == 'Nota aggiornata'
    assert observation.acc_m == 9
    assert observation.source == 'ipso'
    assert observation.operator == 'Mario'
    assert observation.version == 2
    assert list(observation.categories.all()) == [new_category]
    assert ObservationPhoto.objects.filter(id=photo1.id).exists()
    assert not ObservationPhoto.objects.filter(id=photo2.id).exists()
    assert ObservationPhoto.objects.filter(observation=observation).count() == 2
    assert resp.json()[PATCHES][0][RECORD] == _observation_digest_record(observation)


def test_observation_save_stale_edit_conflicts(writer_client, regions):
    observation = Observation.objects.create(
        date='2026-07-25', text='Prima nota', lat=38.5, lon=16.3,
        region=regions[0], version=2,
    )
    body = {
        ROW_ID: str(observation.id),
        VERSION: '1',
        FIELD_REGION_ID: str(regions[0].id),
        FIELD_DATE: '2026-08-02',
        FIELD_TEXT: 'Nota aggiornata',
        FIELD_LAT: '38.7',
        FIELD_LON: '16.7',
        FIELD_NONCE: 'observation-conflict',
    }

    resp = writer_client.post('/api/bosco/observations/save/', body)

    assert resp.status_code == 400
    data = resp.json()
    assert data[STATUS] == STATUS_CONFLICT
    assert data[PATCHES][0][DATA_ID] == DIGEST_OBSERVATIONS
    assert data[PATCHES][0][ROW_ID] == observation.id
    assert data[PATCHES][0][RECORD] == _observation_digest_record(observation)
    assert 'bosco-observation-form' in data[HTML]


def test_observation_delete_removes_observation(writer_client, tmp_path, settings):
    settings.OBSERVATION_MEDIA_DIR = tmp_path
    observation = Observation.objects.create(
        date='2026-07-25', text='Da eliminare', lat=38.5, lon=16.3,
        version=3,
    )
    _stored_observation_photo(
        tmp_path, observation, 'delete.jpg', b'\xff\xd8\xffdelete', 'c' * 64,
    )
    body = {ROW_ID: str(observation.id), VERSION: '3', FIELD_NONCE: 'obs-delete'}

    resp = writer_client.post('/api/bosco/observations/delete/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    assert Observation.objects.count() == 0
    assert ObservationPhoto.objects.count() == 0
    assert resp.json()[DELETES] == [{
        DATA_ID: DIGEST_OBSERVATIONS,
        ROW_ID: observation.id,
    }]


def _jpeg_upload(name='photo.jpg', content=b'\xff\xd8\xffjpeg'):
    return SimpleUploadedFile(name, content, content_type='image/jpeg')


def _stored_observation_photo(tmp_path, observation, filename, content, checksum):
    relative = f'{observation.id}/{filename}'
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ObservationPhoto.objects.create(
        observation=observation,
        file_path=relative,
        content_type='image/jpeg',
        size_bytes=len(content),
        checksum=checksum,
        original_filename=filename,
    )


def _observation_digest_record(observation):
    observation = (
        Observation.objects.prefetch_related('categories')
        .get(id=observation.id)
    )
    observation.photo_count = observation.photos.count()
    return build_observation_record(observation)


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


def test_parcel_metadata_export_requires_login(client, regions):
    resp = client.get(f'/api/bosco/parcels/export/?region_id={regions[0].id}')

    assert resp.status_code == 302
    assert '/login/' in resp.url


def test_parcel_metadata_export_region(reader_client, parcels):
    parcels[0].desc_geo = 'Stazione test'
    parcels[0].desc_veg = 'Soprassuolo test'
    parcels[0].cutting_plan = 'Diradamento test'
    parcels[0].harvest_mechanism = 'Strascico con trattori'
    parcels[0].save(update_fields=[
        'desc_geo', 'desc_veg', 'cutting_plan', 'harvest_mechanism',
    ])

    resp = reader_client.get(
        f'/api/bosco/parcels/export/?region_id={parcels[0].region_id}',
    )

    assert resp.status_code == 200
    assert resp['Cache-Control'] == 'no-store'
    assert 'particelle-Capistrano.csv' in resp['Content-Disposition']
    reader = csv_io.read(resp.content.decode('utf-8'))
    assert reader.fieldnames == [
        S.CSV_COL_REGION, S.CSV_COL_PARCEL, S.CSV_COL_CLASS,
        S.CSV_COL_GOVERNANCE, S.CSV_COL_AREA_HA, S.CSV_COL_AVE_AGE,
        S.CSV_COL_LOCATION, S.CSV_COL_ALT_MIN, S.CSV_COL_ALT_MAX,
        S.CSV_COL_ASPECT, S.CSV_COL_GRADE_PCT, S.CSV_COL_GEO_DESC,
        S.CSV_COL_VEG_DESC, S.CSV_COL_CUTTING_PLAN,
        S.CSV_COL_HARVEST_MECHANISM, S.CSV_COL_INTERVAL,
        S.CSV_COL_STANDARDS,
    ]
    rows = list(reader)
    assert [row[S.CSV_COL_PARCEL] for row in rows] == ['1', '2']
    assert rows[0][S.CSV_COL_CLASS] == 'A'
    assert rows[0][S.CSV_COL_GOVERNANCE] == S.TYPE_HIGHFOREST
    assert rows[0][S.CSV_COL_GEO_DESC] == 'Stazione test'
    assert rows[0][S.CSV_COL_VEG_DESC] == 'Soprassuolo test'
    assert rows[0][S.CSV_COL_CUTTING_PLAN] == 'Diradamento test'
    assert rows[0][S.CSV_COL_HARVEST_MECHANISM] == 'Strascico con trattori'
    assert rows[0][S.CSV_COL_INTERVAL] == ''
    assert rows[0][S.CSV_COL_STANDARDS] == ''


def test_parcel_metadata_export_single_parcel(reader_client, parcels):
    resp = reader_client.get(
        f'/api/bosco/parcels/export/?region_id={parcels[0].region_id}'
        f'&parcel_id={parcels[1].id}',
    )

    assert resp.status_code == 200
    assert 'particella-Capistrano-2.csv' in resp['Content-Disposition']
    rows = list(csv_io.read(resp.content.decode('utf-8')))
    assert len(rows) == 1
    assert rows[0][S.CSV_COL_REGION] == 'Capistrano'
    assert rows[0][S.CSV_COL_PARCEL] == '2'


def test_parcel_metadata_export_all_regions(reader_client, parcels):
    resp = reader_client.get('/api/bosco/parcels/export/?all=1')

    assert resp.status_code == 200
    assert 'particelle.csv' in resp['Content-Disposition']
    rows = list(csv_io.read(resp.content.decode('utf-8')))
    assert [
        (row[S.CSV_COL_REGION], row[S.CSV_COL_PARCEL]) for row in rows
    ] == [('Capistrano', '1'), ('Capistrano', '2'), ('Fabrizia', '1')]


def test_parcel_metadata_form_requires_writer(reader_client, parcels):
    resp = reader_client.get(f'/api/bosco/parcels/metadata/form/{parcels[0].id}/')
    assert resp.status_code == 403


def test_parcel_metadata_form_writer_access(writer_client, parcels):
    resp = writer_client.get(f'/api/bosco/parcels/metadata/form/{parcels[0].id}/')

    assert resp.status_code == 200
    html = resp.json()[HTML]
    assert 'id="bosco-parcel-metadata-form"' in html
    assert f'value="{parcels[0].id}"' in html
    assert 'name="eclass_id"' in html
    assert f'value="{parcels[0].eclass_id}"' in html
    assert 'Comparto' in html
    assert 'A — Fustaia' in html
    assert 'name="cutting_plan"' in html
    assert 'name="harvest_mechanism"' in html
    assert 'name="intervention_interval"' in html
    assert 'name="standards_per_ha"' in html
    assert 'data-target="coppice-metadata-fields"' in html
    assert 'hidden' in html


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
        'eclass_id': str(parcel.eclass_id),
        'area_ha': '12,50', 'ave_age': '44', 'location_name': 'Costa alta',
        'altitude_min_m': '700', 'altitude_max_m': '920',
        'aspect': 'NE', 'grade_pct': '35',
        'desc_veg': 'Abete e faggio.', 'desc_geo': 'Calcare.',
        'cutting_plan': 'Diradamento selettivo.',
        'harvest_mechanism': 'Strascico con trattori',
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
    assert parcel.harvest_mechanism == 'Strascico con trattori'
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
        'eclass_id': str(parcel.eclass_id),
        'area_ha': '2.00', 'ave_age': '', 'location_name': '',
        'altitude_min_m': '', 'altitude_max_m': '',
        'aspect': '', 'grade_pct': '', 'desc_veg': '', 'desc_geo': '',
        'cutting_plan': 'Taglio di ceduo.',
        'harvest_mechanism': 'Verricello',
        'intervention_interval': '12', 'standards_per_ha': '30',
        FIELD_NONCE: 'parcel-coppice-save',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    parcel.refresh_from_db()
    assert parcel.cutting_plan == 'Taglio di ceduo.'
    assert parcel.harvest_mechanism == 'Verricello'
    assert parcel.intervention_interval == 12
    assert parcel.standards_per_ha == 30


def test_parcel_metadata_save_updates_governance(writer_client, parcels, eclasses):
    parcel = parcels[0]
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'eclass_id': str(eclasses[2].id),
        'area_ha': str(parcel.area_ha), 'ave_age': '', 'location_name': '',
        'altitude_min_m': '', 'altitude_max_m': '',
        'aspect': '', 'grade_pct': '', 'desc_veg': '', 'desc_geo': '',
        'cutting_plan': '', 'intervention_interval': '14',
        'standards_per_ha': '50', FIELD_NONCE: 'parcel-governance-save',
    }

    resp = writer_client.post('/api/bosco/parcels/metadata/save/', body,
                              content_type='application/json')

    assert resp.status_code == 200
    parcel.refresh_from_db()
    assert parcel.eclass == eclasses[2]
    assert parcel.intervention_interval == 14
    assert parcel.standards_per_ha == 50


def test_parcel_metadata_save_requires_coppice_fields(writer_client, regions, eclasses):
    parcel = Parcel.objects.create(
        name='C1', region=regions[0], eclass=eclasses[2],
        area_ha=Decimal('1.00'), intervention_interval=18, standards_per_ha=75,
    )
    body = {
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'eclass_id': str(parcel.eclass_id),
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
        ROW_ID: str(parcel.id), VERSION: '2',
        'eclass_id': str(parcel.eclass_id), 'area_ha': '12.50',
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
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'eclass_id': str(parcel.eclass_id), 'area_ha': '',
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
        ROW_ID: str(parcel.id), VERSION: str(parcel.version),
        'eclass_id': str(parcel.eclass_id), 'area_ha': '12.50',
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
    pai = TreeSample.objects.select_related('tree').get(
        tree__species=species[0], parcel=parcels[0], preserved_number=7,
    )
    tree = pai.tree
    assert tree.estimated_birth_year == 1920
    assert pai.lat == 38.12346
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
        species=species[0],
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
        species=species[0],
        estimated_birth_year=1920,
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
    assert tree.estimated_birth_year == 1935
    assert tree.version == 2
    assert pai.parcel == parcels[1]
    assert pai.lat == 38.22222
    assert pai.lon == 16.33333
    assert pai.acc_m == 9
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
        species=species[0],
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
        species=species[0],
        estimated_birth_year=1920, version=2,
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


def test_pai_delete_removes_preserved_sample(writer_client, parcels, species):
    tree = Tree.objects.create(
        species=species[0],
        estimated_birth_year=1925, version=3,
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
    assert tree.version == 3
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
