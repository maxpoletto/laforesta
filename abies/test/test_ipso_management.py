"""Tests for Ipso management commands."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import Client
from django.urls import reverse

from apps.base.models import (
    HarvestPlan, HarvestPlanItem, HarvestPlanItemState, TreeMark,
)
from apps.ipso.models import IpsoUpload, IpsoUploadState
from config import strings as S
from config.constants import IPSO_REFERENCE_LEGACY_CONVERTED, RECORDS, SESSION, UPLOAD

pytestmark = pytest.mark.django_db


@pytest.fixture
def writer_client(writer_user):
    client = Client()
    client.force_login(writer_user)
    return client


def test_stage_marks_uploads_creates_inbox_without_source(settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'

    call_command('stage_marks_uploads', str(tmp_path / 'missing'))

    assert settings.IPSO_INBOX_DIR.is_dir()
    assert IpsoUpload.objects.count() == 0


def test_stage_marks_uploads_imports_converted_csv(
        writer_client, parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    source_dir = tmp_path / 'marks'
    source_dir.mkdir()
    session_id = '05a42dbc-9d9e-45a3-94f5-3632c64404c5'
    csv_text = (
        f'{S.CSV_COL_DATA},{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},'
        f'{S.CSV_COL_DAMAGED},{S.CSV_COL_NUMBER},{S.CSV_COL_SPECIES},'
        f'{S.CSV_COL_D_CM},{S.CSV_COL_H_M},{S.CSV_COL_H_MEASURED},'
        f'{S.CSV_COL_LAT},{S.CSV_COL_LON},{S.CSV_COL_ACC_M},{S.CSV_COL_OPERATOR}\n'
        '15/06/2026,Capistrano,1,0,1,Abete,42,22.5,0,38.570850,16.300260,2,Valerio\n'
        '15/06/2026,Capistrano,1,0,2,Abete,44,23.5,1,38.570803,16.300360,3,Giulia\n'
    )
    (source_dir / f'{session_id}.csv').write_text(csv_text, encoding='utf-8')

    call_command('stage_marks_uploads', str(source_dir))
    call_command('stage_marks_uploads', str(source_dir))

    upload = IpsoUpload.objects.get(session_id=session_id)
    assert IpsoUpload.objects.count() == 1
    assert upload.mode == 'martellate'
    assert upload.state == IpsoUploadState.RECEIVED
    assert upload.record_count == 2
    assert upload.record_date == '2026-06-15'
    assert upload.operator == ''
    upload_dir = Path(upload.inbox_path)
    payload = json.loads((upload_dir / 'upload.json').read_text(encoding='utf-8'))
    assert (upload_dir / 'upload.sha256').is_file()
    assert (upload_dir / 'export.csv').read_text(encoding='utf-8') == csv_text
    assert payload[SESSION]['reference_version'] == IPSO_REFERENCE_LEGACY_CONVERTED
    assert payload[RECORDS][0]['operator'] == 'Valerio'
    assert payload[RECORDS][0]['date'] == '2026-06-15'
    assert payload[RECORDS][0]['parcel_id'] == parcels[0].id
    assert payload[RECORDS][0]['species_id'] == species[0].id
    assert payload[RECORDS][0]['h_m'] == '22.50'

    detail = writer_client.get(reverse('ipso-upload-detail', args=[upload.id])).json()
    assert detail[UPLOAD]['mode_label'] == S.IPSO_MODE_MARTELLATE_LABEL
    assert detail[UPLOAD]['reference_version_label'] == S.IPSO_REFERENCE_LEGACY_CONVERTED
    assert detail[UPLOAD]['record_date'] == '2026-06-15'

    plan = HarvestPlan.objects.create(
        name='PDG test', year_start=2026, year_end=2026,
    )
    item = HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2026,
        state=HarvestPlanItemState.PLANNED,
        volume_planned_m3=Decimal('0'),
    )
    resp = writer_client.post(
        reverse('ipso-upload-import-martellate', args=[upload.id]),
        data=json.dumps({'harvest_plan_item_id': item.id}),
        content_type='application/json',
    )

    assert resp.status_code == 200
    assert resp.json()['imported'] == 2
    marks = list(TreeMark.objects.order_by('number'))
    assert [m.operator for m in marks] == ['Valerio', 'Giulia']
    assert [m.number for m in marks] == [1, 2]


def test_stage_marks_uploads_rejects_blank_parcel(parcels, species, settings, tmp_path):
    settings.IPSO_INBOX_DIR = tmp_path / 'inbox'
    source_dir = tmp_path / 'marks'
    source_dir.mkdir()
    csv_text = (
        f'{S.CSV_COL_DATA},{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},'
        f'{S.CSV_COL_DAMAGED},{S.CSV_COL_NUMBER},{S.CSV_COL_SPECIES},'
        f'{S.CSV_COL_D_CM},{S.CSV_COL_H_M},{S.CSV_COL_H_MEASURED},'
        f'{S.CSV_COL_LAT},{S.CSV_COL_LON},{S.CSV_COL_ACC_M},{S.CSV_COL_OPERATOR}\n'
        '15/06/2026,Capistrano,,0,1,Abete,42,22.5,0,38.570850,16.300260,2,Valerio\n'
    )
    (source_dir / 'blank-parcel.csv').write_text(csv_text, encoding='utf-8')

    with pytest.raises(CommandError, match='unknown parcel Capistrano/'):
        call_command('stage_marks_uploads', str(source_dir))

    assert IpsoUpload.objects.count() == 0
