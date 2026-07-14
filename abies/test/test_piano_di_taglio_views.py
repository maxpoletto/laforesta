"""Tests for the Piano di taglio backend endpoints.

Covers plan CRUD, plan CSV import, plan-level Esporta, plan-item
CRUD (including the state-gated delete), per-item Esporta, and the
cantiere transition save view.  All write paths share the digest-stale
contract and the nonce-idempotency contract.
"""

import base64
import csv
import gzip
import io
import json
import zipfile
from datetime import date as date_type
from decimal import Decimal

import pytest
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.test import Client

from apps.base import csv_io
from apps.base.digests import build_tree_mark_record
from apps.base.models import (
    DigestStatus, HarvestDetail, HarvestPlan, HarvestPlanItem,
    HarvestPlanItemState, HarvestTransition, Parcel, ParcelPlanDetail, Tree,
    TreeMark, UsedNonce,
)
from apps.piano_di_taglio.mark_import import (
    csv_mark_fingerprint, legacy_csv_mark_fingerprint,
)
from apps.prelievi.models import Harvest, HarvestTractor
from config import strings as S
from config.constants import (
    COLUMNS, DATA_ID, FIELD_ACC_M, FIELD_COPPICE_FILE, FIELD_CREW_ID,
    FIELD_D_CM, FIELD_DAMAGED, FIELD_DATE, FIELD_DESCRIPTION,
    FIELD_FILE, FIELD_HIGHFOREST_FILE, FIELD_H_M, FIELD_H_MEASURED,
    FIELD_HARVEST_PLAN_ID, FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_INTERVENTION_AREA_HA, FIELD_LAT, FIELD_LON, FIELD_MASS_Q,
    FIELD_NAME, FIELD_NONCE, FIELD_NOTE, FIELD_NUMBER, FIELD_OPEN, FIELD_OPERATOR,
    FIELD_PARCEL_ID, FIELD_PRODUCT_ID, FIELD_PSR, FIELD_REGION_ID,
    FIELD_SPECIES_ID, FIELD_UNHEALTHY,
    FIELD_VOLUME_M3, FIELD_VOLUME_PLANNED_M3, FIELD_YEAR_END,
    FIELD_YEAR_PLANNED, FIELD_YEAR_START, HTML, MESSAGE, PATCHES,
    RECORD, ROW_ID, ROWS, STATUS, STATUS_CONFLICT,
    STATUS_VALIDATION_ERROR, TRANSITION_RECORDS, VERSION,
)


def _csv_b64(value):
    if hasattr(value, 'getvalue'):
        raw = value.getvalue()
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = str(value).encode('utf-8')
    return base64.b64encode(raw).decode('ascii')


def _post_plan_csv_import(client, **kwargs):
    body = {}
    for key, value in kwargs.items():
        if key in (FIELD_HIGHFOREST_FILE, FIELD_COPPICE_FILE):
            body[key] = _csv_b64(value)
        else:
            body[key] = value
    return client.post(
        '/api/piano-di-taglio/plan/import-csv/',
        data=json.dumps(body),
        content_type='application/json',
    )


def _csv_rows(raw):
    delimiter, _ = csv_io.export_format()
    return list(csv.reader(
        io.StringIO(raw.decode('utf-8')), delimiter=delimiter,
    ))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


@pytest.fixture
def reader_client(reader_user):
    c = Client()
    c.force_login(reader_user)
    return c


@pytest.fixture
def plan(db):
    return HarvestPlan.objects.create(
        name='Plan 2024-2034', year_start=2024, year_end=2034,
        description='A plan.',
    )


@pytest.fixture
def planned_item(plan, parcels):
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2025,
        volume_planned_m3=Decimal('100.0'),
        state=HarvestPlanItemState.PLANNED,
    )



# ---------------------------------------------------------------------------
# Plan CRUD
# ---------------------------------------------------------------------------

class TestPlanCRUD:
    def _post(self, client, payload):
        return client.post(
            '/api/piano-di-taglio/plan/save/',
            data=json.dumps(payload), content_type='application/json',
        )

    def test_create(self, writer_client, db):
        resp = self._post(writer_client, {
            FIELD_NAME: 'New plan',
            FIELD_DESCRIPTION: 'Notes.',
            FIELD_YEAR_START: 2026, FIELD_YEAR_END: 2036,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data[DATA_ID] == 'harvest_plans'
        plan = HarvestPlan.objects.get(id=data[ROW_ID])
        assert plan.year_start == 2026
        assert DigestStatus.objects.get(name='harvest_plans').stale is True

    def test_create_duplicate_name(self, writer_client, plan):
        resp = self._post(writer_client, {
            FIELD_NAME: plan.name,
            FIELD_YEAR_START: 2030, FIELD_YEAR_END: 2040,
        })
        assert resp.status_code == 400
        assert S.ERR_PLAN_NAME_DUPLICATE in resp.json()[MESSAGE]

    def test_create_bad_year_range(self, writer_client, db):
        resp = self._post(writer_client, {
            FIELD_NAME: 'Bad range',
            FIELD_YEAR_START: 2030, FIELD_YEAR_END: 2020,
        })
        assert resp.status_code == 400
        assert S.ERR_PLAN_YEAR_RANGE in resp.json()[MESSAGE]

    def test_update(self, writer_client, plan):
        resp = self._post(writer_client, {
            ROW_ID: plan.id, VERSION: plan.version,
            FIELD_NAME: 'Renamed', FIELD_DESCRIPTION: 'Updated',
            FIELD_YEAR_START: 2024, FIELD_YEAR_END: 2034,
        })
        assert resp.status_code == 200
        plan.refresh_from_db()
        assert plan.name == 'Renamed'
        assert plan.version == 2

    def test_update_stale_version(self, writer_client, plan):
        resp = self._post(writer_client, {
            ROW_ID: plan.id, VERSION: 999,
            FIELD_NAME: 'X', FIELD_YEAR_START: 2024, FIELD_YEAR_END: 2034,
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT

    def test_delete_allowed_when_all_planned(self, writer_client, plan, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/plan/delete/{plan.id}/',
            data=json.dumps({VERSION: plan.version}),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert not HarvestPlan.objects.filter(id=plan.id).exists()

    def test_delete_saves_nonce(self, writer_client, plan, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/plan/delete/{plan.id}/',
            data=json.dumps({
                VERSION: plan.version,
                FIELD_NONCE: 'plan-delete-nonce',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='plan-delete-nonce').exists()

    def test_delete_stale_version_conflicts(self, writer_client, plan, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/plan/delete/{plan.id}/',
            data=json.dumps({VERSION: plan.version + 1}),
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert HarvestPlan.objects.filter(id=plan.id).exists()

    def test_delete_blocked_when_active_item(self, writer_client, plan, planned_item):
        planned_item.state = HarvestPlanItemState.OPEN
        planned_item.save()
        resp = writer_client.post(
            f'/api/piano-di-taglio/plan/delete/{plan.id}/',
            data=json.dumps({VERSION: plan.version}),
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert S.ERR_PLAN_HAS_ACTIVE_ITEMS in resp.json()[MESSAGE]
        assert HarvestPlan.objects.filter(id=plan.id).exists()


# ---------------------------------------------------------------------------
# Plan CSV import + export
# ---------------------------------------------------------------------------

FUSTAIA_CSV = (
    'Compresa,Particella,Anno,Prelievo (m³)\r\n'
    'Capistrano,1,2027,250\r\n'
)
CEDUO_CSV = (
    'Anno,Compresa,Particella,Superficie intervento (ha),Turno (a),Note\r\n'
    '2028,Capistrano,1,2.5,18,Cont.\r\n'
)
class TestPlanCSVImport:
    def _upload(self, client, **kwargs):
        return _post_plan_csv_import(client, **kwargs)

    def test_import_calendar_files(self, writer_client, parcels, species):
        f = self._upload(
            writer_client,
            name='CSV plan',
            description='From CSV.',
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
            ceduo_file=io.BytesIO(CEDUO_CSV.encode('utf-8')),
        )
        assert f.status_code == 200, f.json()
        data = f.json()
        assert data['n_items'] == 2
        plan = HarvestPlan.objects.get(id=data[ROW_ID])
        assert plan.year_start == 2027
        assert plan.year_end == 2028
        # Coppice item gets a ParcelPlanDetail with interval 18.
        ppd = ParcelPlanDetail.objects.get(harvest_plan=plan)
        assert ppd.harvest_detail.interval == 18

    def test_import_no_files(self, writer_client, db):
        resp = self._upload(writer_client, name='No files plan')
        assert resp.status_code == 400
        assert S.ERR_CSV_NO_FILES in resp.json()[MESSAGE]

    def test_import_duplicate_name(self, writer_client, plan, parcels):
        resp = self._upload(
            writer_client, name=plan.name,
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
        )
        assert resp.status_code == 400
        assert S.ERR_PLAN_NAME_DUPLICATE in resp.json()[MESSAGE]

    def test_import_unknown_parcel(self, writer_client, db):
        bad_csv = (
            'Compresa,Particella,Anno,Prelievo (m³)\r\n'
            'Nowhere,99,2027,250\r\n'
        )
        resp = self._upload(
            writer_client, name='Bad plan',
            fustaia_file=io.BytesIO(bad_csv.encode('utf-8')),
        )
        assert resp.status_code == 400

    def test_import_rejects_invalid_planned_volume(self, writer_client, parcels):
        bad_csv = (
            'Compresa,Particella,Anno,Prelievo (m³)\r\n'
            'Capistrano,1,2027,abc\r\n'
        )
        resp = self._upload(
            writer_client, name='Bad volume plan',
            fustaia_file=io.BytesIO(bad_csv.encode('utf-8')),
        )

        assert resp.status_code == 400
        assert 'abc' in resp.json()[MESSAGE]

    def test_import_rejects_negative_planned_volume(self, writer_client, parcels):
        bad_csv = (
            'Compresa,Particella,Anno,Prelievo (m³)\r\n'
            'Capistrano,1,2027,-1\r\n'
        )
        resp = self._upload(
            writer_client, name='Negative volume plan',
            fustaia_file=io.BytesIO(bad_csv.encode('utf-8')),
        )

        assert resp.status_code == 400
        assert '-1' in resp.json()[MESSAGE]

    def test_import_whole_region_fustaia(
        self, writer_client, plan, regions,
    ):
        # Particella = 'X' marks a whole-region item.  Note column must
        # contain "Catastrofato" or "Fitosanitario".
        csv_in = (
            f'Compresa;Particella;Anno;{S.COL_VOLUME_PLANNED};Note\r\n'
            f'Capistrano;{S.PARCEL_WHOLE_REGION_MARK};2029;;{S.FLAG_DAMAGED}\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 200, r.json()
        items = HarvestPlanItem.objects.filter(
            harvest_plan=plan, parcel__isnull=True,
        )
        assert items.count() == 1
        it = items[0]
        assert it.region.name == 'Capistrano'
        assert it.damaged is True
        assert it.unhealthy is False
        assert it.year_planned == 2029

    def test_import_whole_region_requires_flag(
        self, writer_client, plan, regions,
    ):
        csv_in = (
            f'Compresa;Particella;Anno;{S.COL_VOLUME_PLANNED};Note\r\n'
            f'Capistrano;{S.PARCEL_WHOLE_REGION_MARK};2029;;\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 400
        # Whole-region row with empty Note is rejected.
        assert 'Compresa' in r.json()[MESSAGE] or 'compresa' in r.json()[MESSAGE]

    def test_import_parcel_scoped_note_sets_flags(
        self, writer_client, plan, parcels,
    ):
        # Round-trip: a parcel-scoped row whose Note column says
        # "Catastrofato" sets damaged=True on the resulting item.
        csv_in = (
            f'Compresa;Particella;Anno;{S.COL_VOLUME_PLANNED};Note\r\n'
            f'{parcels[0].region.name};{parcels[0].name};2030;50;{S.FLAG_DAMAGED}\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 200, r.json()
        it = HarvestPlanItem.objects.get(
            harvest_plan=plan, parcel=parcels[0], year_planned=2030,
        )
        assert it.damaged is True

    def test_import_current_volume_header(self, writer_client, plan, parcels):
        csv_in = (
            f'Compresa;Particella;Anno;{S.COL_VOLUME_PLANNED};Note\r\n'
            f'{parcels[0].region.name};{parcels[0].name};2032;75;\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 200, r.json()
        it = HarvestPlanItem.objects.get(
            harvest_plan=plan, parcel=parcels[0], year_planned=2032,
        )
        assert it.volume_planned_m3 == Decimal('75')

    def test_import_rejects_target_change_when_linked_harvest_exists(
        self, writer_client, plan, planned_item, parcels, products, crews,
    ):
        Harvest.objects.create(
            date=date_type(2025, 6, 1), product=products[0], crew=crews[0],
            parcel=planned_item.parcel, harvest_plan_item=planned_item,
            mass_q=Decimal('10.00'), volume_m3=Decimal('1.000'),
            damaged=planned_item.damaged, unhealthy=planned_item.unhealthy,
            psr=planned_item.psr,
        )
        csv_in = (
            f'{S.COL_ID};Compresa;Particella;Anno;{S.COL_VOLUME_PLANNED};Note\r\n'
            f'{planned_item.id};{parcels[1].region.name};{parcels[1].name};'
            f'{planned_item.year_planned};175;\r\n'
        )

        resp = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(csv_in.encode('utf-8')),
        )

        assert resp.status_code == 400
        assert S.ERR_PLAN_ITEM_LINKED_HARVESTS_INVARIANT in resp.json()[MESSAGE]
        planned_item.refresh_from_db()
        assert planned_item.parcel_id == parcels[0].id
        assert planned_item.volume_planned_m3 == Decimal('100.000')

    def test_import_total_volume_header_rejected(self, writer_client, plan, parcels):
        csv_in = (
            'Compresa;Particella;Anno;Volume (m³);Note\r\n'
            f'{parcels[0].region.name};{parcels[0].name};2033;900;\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 400
        assert S.CSV_COL_HARVEST_M3 in r.json()[MESSAGE]
        assert not HarvestPlanItem.objects.filter(
            harvest_plan=plan, parcel=parcels[0], year_planned=2033,
        ).exists()

    def test_import_ceduo_altre_note_disambiguates(
        self, writer_client, plan, parcels, eclasses, regions,
    ):
        # When 'Altre note' is present in the header, 'Note' is the flag
        # string and 'Altre note' is free-text.
        from apps.base.models import Parcel
        coppice_eclass = next(e for e in eclasses if e.coppice)
        coppice_parcel = Parcel.objects.create(
            name='cop-1', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('3.0'),
        )
        csv_in = (
            f'Anno;Compresa;Particella;Superficie intervento (ha);'
            f'Turno (a);Note;Altre note\r\n'
            f'2031;{coppice_parcel.region.name};{coppice_parcel.name};'
            f'2,5;18;{S.FLAG_DAMAGED};Cont. 2032\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            ceduo_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 200, r.json()
        it = HarvestPlanItem.objects.get(
            harvest_plan=plan, parcel=coppice_parcel, year_planned=2031,
        )
        assert it.damaged is True
        assert it.note == 'Cont. 2032'

    def test_import_ceduo_legacy_note_is_free_text(
        self, writer_client, plan, parcels, eclasses, regions,
    ):
        # Legacy pdg-2026 ceduo CSV: 'Note' is free-text, no flag column.
        from apps.base.models import Parcel
        coppice_eclass = next(e for e in eclasses if e.coppice)
        coppice_parcel = Parcel.objects.create(
            name='cop-2', region=regions[0], eclass=coppice_eclass,
            area_ha=Decimal('3.0'),
        )
        csv_in = (
            f'Anno,Compresa,Particella,Superficie intervento (ha),'
            f'Turno (a),Note\r\n'
            f'2032,{coppice_parcel.region.name},{coppice_parcel.name},'
            f'2.5,18,Cont. 2033\r\n'
        )
        r = self._upload(
            writer_client, harvest_plan_id=plan.id,
            ceduo_file=io.BytesIO(csv_in.encode('utf-8')),
        )
        assert r.status_code == 200, r.json()
        it = HarvestPlanItem.objects.get(
            harvest_plan=plan, parcel=coppice_parcel, year_planned=2032,
        )
        assert it.damaged is False
        assert it.note == 'Cont. 2033'

    def test_import_into_existing_plan_upserts(
        self, writer_client, plan, parcels, species,
    ):
        # First import: append 1 fustaia row.
        r1 = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
        )
        assert r1.status_code == 200, r1.json()
        assert r1.json()[ROW_ID] == plan.id
        items = HarvestPlanItem.objects.filter(harvest_plan=plan).order_by('id')
        assert items.count() == 1
        assert items[0].year_planned == 2027
        assert float(items[0].volume_planned_m3) == 250.0

        # Same file again — idempotent, count stays at 1.
        r2 = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
        )
        assert r2.status_code == 200
        assert HarvestPlanItem.objects.filter(harvest_plan=plan).count() == 1

        # Revised file (same parcel + year, different volume) — overwrite.
        revised = (
            'Compresa,Particella,Anno,Prelievo (m³)\r\n'
            'Capistrano,1,2027,500\r\n'
        )
        r3 = self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(revised.encode('utf-8')),
        )
        assert r3.status_code == 200
        items = HarvestPlanItem.objects.filter(harvest_plan=plan)
        assert items.count() == 1
        assert float(items[0].volume_planned_m3) == 500.0

    def test_import_into_existing_widens_year_range(
        self, writer_client, plan, parcels,
    ):
        # Plan starts at 2024-2034.  Importing a 2027 row stays inside.
        old_start, old_end = plan.year_start, plan.year_end
        self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
        )
        plan.refresh_from_db()
        assert plan.year_start == old_start
        assert plan.year_end == old_end

        # 2040 is past year_end → widens.
        future = (
            'Compresa,Particella,Anno,Prelievo (m³)\r\n'
            'Capistrano,1,2040,100\r\n'
        )
        self._upload(
            writer_client, harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(future.encode('utf-8')),
        )
        plan.refresh_from_db()
        assert plan.year_end == 2040

    def test_import_into_unknown_plan(self, writer_client, db):
        resp = self._upload(
            writer_client, harvest_plan_id=9999,
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
        )
        assert resp.status_code == 400
        assert S.ERR_PLAN_NOT_FOUND in resp.json()[MESSAGE]


class TestPlanExport:
    def test_round_trip(self, writer_client, plan, planned_item):
        resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        assert resp.status_code == 200
        assert resp['Content-Type'] == 'application/zip'
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        assert set(zf.namelist()) == {'fustaia.csv', 'ceduo.csv'}
        # planned_item is fustaia → lands in piano.csv.
        piano = zf.read('fustaia.csv').decode('utf-8').splitlines()
        assert any(planned_item.parcel.region.name in line for line in piano)

    def test_export_uses_italian_locale(
        self, writer_client, plan, planned_item,
    ):
        # Italian locale: ';' field separator, ',' decimal mark.  Mirrors
        # TABLE_CSV_FORMAT used by the per-table CSV export.
        resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        piano = zf.read('fustaia.csv').decode('utf-8')
        header, data, *_ = piano.splitlines()
        assert ';' in header and ',' not in header
        # planned_item volume_planned_m3 = 100.0 → '100' in the export.
        # Render a non-integer to assert decimal mark.
        planned_item.volume_planned_m3 = Decimal('12.5')
        planned_item.save()
        resp2 = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        piano2 = zf.read('fustaia.csv') if False else (
            zipfile.ZipFile(io.BytesIO(resp2.content)).read('fustaia.csv')
            .decode('utf-8')
        )
        assert '12,5' in piano2

    def test_export_includes_full_column_set(
        self, writer_client, plan, planned_item,
    ):
        resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        piano_header = zf.read('fustaia.csv').decode('utf-8').splitlines()[0]
        for col in [S.COL_ID, S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL,
                    S.COL_STATE, S.COL_NOTE, S.COL_PARCEL_AREA_HA,
                    S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED,
                    S.COL_VOLUME_ACTUAL, S.COL_EXTRA_NOTE]:
            assert col in piano_header, f'missing column {col}'

    def test_fustaia_section_export_matches_zip_member(
        self, writer_client, plan, planned_item,
    ):
        zip_resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))
        expected = zf.read(S.CSV_FILE_HIGHFOREST)

        resp = writer_client.get(
            f'/api/piano-di-taglio/plan/export/{plan.id}/fustaia/',
        )

        assert resp.status_code == 200
        assert resp['Content-Type'] == 'text/csv; charset=utf-8'
        assert 'interventi-fustaia.csv' in resp['Content-Disposition']
        assert resp.content == expected
        rows = _csv_rows(resp.content)
        assert rows[0][0] == S.COL_ID
        assert rows[1][0] == str(planned_item.id)
        area_idx = rows[0].index(S.COL_PARCEL_AREA_HA)
        assert float(rows[1][area_idx].replace(',', '.')) == pytest.approx(
            10.5, abs=0.0001,
        )

    def test_ceduo_section_export_matches_zip_member(
        self, writer_client, plan, regions, eclasses,
    ):
        parcel = Parcel.objects.create(
            name='C1', region=regions[0], eclass=eclasses[2],
            area_ha=Decimal('4.25'),
        )
        detail = HarvestDetail.objects.create(
            description='Turno 20a', interval=20,
        )
        ParcelPlanDetail.objects.create(
            harvest_plan=plan, parcel=parcel, harvest_detail=detail,
        )
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcel, year_planned=2028,
            intervention_area_ha=Decimal('2.5'), note='Ceduo note',
            state=HarvestPlanItemState.PLANNED,
        )
        zip_resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))
        expected = zf.read(S.CSV_FILE_COPPICE)

        resp = writer_client.get(
            f'/api/piano-di-taglio/plan/export/{plan.id}/ceduo/',
        )

        assert resp.status_code == 200
        assert resp['Content-Type'] == 'text/csv; charset=utf-8'
        assert 'interventi-ceduo.csv' in resp['Content-Disposition']
        assert resp.content == expected
        rows = _csv_rows(resp.content)
        assert rows[0][0] == S.COL_ID
        assert rows[1][0] == str(item.id)

    def test_round_trip_whole_region_item(
        self, writer_client, plan, regions, parcels,
    ):
        # A whole-region item exists → exports to fustaia.csv with
        # Particella='X', Note=flag string → re-import preserves the row.
        HarvestPlanItem.objects.create(
            harvest_plan=plan, region=regions[0], parcel=None,
            year_planned=2033, damaged=True,
            state=HarvestPlanItemState.PLANNED,
        )
        resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        fustaia_bytes = zf.read(S.CSV_FILE_HIGHFOREST)
        text = fustaia_bytes.decode('utf-8')
        rows = _csv_rows(fustaia_bytes)
        parcel_idx = rows[0].index(S.COL_PARCEL)
        area_idx = rows[0].index(S.COL_PARCEL_AREA_HA)
        region_row = next(
            row for row in rows
            if row[parcel_idx] == S.PARCEL_WHOLE_REGION_MARK
        )
        assert float(region_row[area_idx].replace(',', '.')) == pytest.approx(
            15.5, abs=0.0001,
        )
        assert S.FLAG_DAMAGED in text

        before = HarvestPlanItem.objects.filter(
            harvest_plan=plan, parcel__isnull=True,
        ).count()
        reup = _post_plan_csv_import(
            writer_client,
            harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(fustaia_bytes),
        )
        assert reup.status_code == 200, reup.json()
        after = HarvestPlanItem.objects.filter(
            harvest_plan=plan, parcel__isnull=True,
        ).count()
        assert after == before  # idempotent

    def test_round_trip_export_then_reimport(
        self, writer_client, plan, planned_item, parcels, species,
    ):
        # Export a plan with one fustaia item, then re-import the same
        # CSV (Italian locale + display names) back into the same plan —
        # should be idempotent and preserve the volume.
        resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        fustaia_bytes = zf.read('fustaia.csv')

        reup = _post_plan_csv_import(
            writer_client,
            harvest_plan_id=plan.id,
            fustaia_file=io.BytesIO(fustaia_bytes),
        )
        assert reup.status_code == 200, reup.json()
        items = HarvestPlanItem.objects.filter(harvest_plan=plan)
        assert items.count() == 1  # idempotent
        assert items[0].id == planned_item.id
        assert float(items[0].volume_planned_m3) == float(planned_item.volume_planned_m3)

    def test_round_trip_duplicate_region_items_and_notes(
        self, writer_client, plan, regions,
    ):
        items = [
            (regions[0], True, False, 'Serra catastrofato'),
            (regions[0], False, True, 'Serra fitosanitario'),
            (regions[1], True, False, 'Capistrano catastrofato'),
            (regions[1], False, True, 'Capistrano fitosanitario'),
        ]
        for region, damaged, unhealthy, note in items:
            HarvestPlanItem.objects.create(
                harvest_plan=plan, region=region, parcel=None,
                year_planned=2026, damaged=damaged, unhealthy=unhealthy,
                note=note, state=HarvestPlanItemState.PLANNED,
            )

        resp = writer_client.get(f'/api/piano-di-taglio/plan/export/{plan.id}/')
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        fustaia_bytes = zf.read(S.CSV_FILE_HIGHFOREST)
        text = fustaia_bytes.decode('utf-8')
        assert S.COL_ID in text.splitlines()[0]
        assert S.COL_EXTRA_NOTE in text.splitlines()[0]
        assert 'Serra catastrofato' in text
        assert 'Capistrano fitosanitario' in text

        copy = HarvestPlan.objects.create(
            name='Copied plan', year_start=2026, year_end=2026,
        )
        for _ in range(2):
            reup = _post_plan_csv_import(
                writer_client,
                harvest_plan_id=copy.id,
                fustaia_file=io.BytesIO(fustaia_bytes),
            )
            assert reup.status_code == 200, reup.json()

        copied = HarvestPlanItem.objects.filter(
            harvest_plan=copy, parcel__isnull=True,
        )
        assert copied.count() == 4
        assert set(copied.values_list('note', flat=True)) == {
            'Serra catastrofato',
            'Serra fitosanitario',
            'Capistrano catastrofato',
            'Capistrano fitosanitario',
        }
        assert copied.filter(region=regions[0], year_planned=2026).count() == 2
        assert copied.filter(region=regions[1], year_planned=2026).count() == 2


# ---------------------------------------------------------------------------
# Item CRUD
# ---------------------------------------------------------------------------

class TestItemCRUD:
    def _save(self, client, payload):
        return client.post(
            '/api/piano-di-taglio/item/save/',
            data=json.dumps(payload), content_type='application/json',
        )

    def test_reader_form_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/piano-di-taglio/item/form/')
        assert resp.status_code == 403

    def test_add_form_renders_for_plan(self, writer_client, plan):
        resp = writer_client.get(f'/api/piano-di-taglio/item/form/?plan={plan.id}')
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert 'id="item-form"' in html
        assert f'name="harvest_plan_id" value="{plan.id}"' in html

    def test_create_fustaia_item(self, writer_client, plan, parcels):
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_PARCEL_ID: parcels[0].id,
            FIELD_YEAR_PLANNED: 2027,
            FIELD_VOLUME_PLANNED_M3: '150',
            FIELD_NOTE: '',
        })
        assert resp.status_code == 200
        data = resp.json()
        item = HarvestPlanItem.objects.get(id=data[ROW_ID])
        assert item.year_planned == 2027
        assert float(item.volume_planned_m3) == 150.0

    def test_create_requires_compresa(self, writer_client, plan):
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_YEAR_PLANNED: 2027,
        })
        assert resp.status_code == 400
        assert S.ERR_PLAN_ITEM_COMPRESA_REQUIRED in resp.json()[MESSAGE]

    def test_create_with_both_region_and_parcel(
        self, writer_client, plan, parcels,
    ):
        # The form cascade always submits both region_id and parcel_id;
        # the server normalises to parcel-scoped and ignores the
        # redundant region_id (storage is region XOR parcel).
        parcel = parcels[0]
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_REGION_ID: parcel.region_id,
            FIELD_PARCEL_ID: parcel.id,
            FIELD_YEAR_PLANNED: 2027,
            FIELD_VOLUME_PLANNED_M3: '50',
        })
        assert resp.status_code == 200
        item = HarvestPlanItem.objects.get(id=resp.json()[ROW_ID])
        assert item.parcel_id == parcel.id
        assert item.region_id is None

    def test_create_region_wide_requires_flag(self, writer_client, plan, regions):
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_REGION_ID: regions[0].id,
            FIELD_YEAR_PLANNED: 2027,
        })
        assert resp.status_code == 400
        assert (S.ERR_PLAN_ITEM_REGION_REQUIRES_FLAG
                in resp.json()[MESSAGE])

    def test_create_region_wide_with_flag(self, writer_client, plan, regions):
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_REGION_ID: regions[0].id,
            FIELD_YEAR_PLANNED: 2027,
            FIELD_DAMAGED: True,
        })
        assert resp.status_code == 200

    def test_create_flattens_model_validation_errors(
        self, writer_client, plan, parcels, monkeypatch,
    ):
        def fail_clean(self):
            raise ValidationError(['Errore uno', 'Errore due'])

        monkeypatch.setattr(HarvestPlanItem, 'clean', fail_clean)
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_PARCEL_ID: parcels[0].id,
            FIELD_YEAR_PLANNED: 2027,
            FIELD_VOLUME_PLANNED_M3: '150',
        })

        assert resp.status_code == 400
        message = resp.json()[MESSAGE]
        assert 'Errore uno' in message
        assert 'Errore due' in message
        assert '[' not in message

    def test_create_hides_raw_integrity_trigger_text(
        self, writer_client, plan, parcels, monkeypatch,
    ):
        def fail_save(self, *args, **kwargs):
            raise IntegrityError('harvest_plan_item: exactly one of region or parcel must be set')

        monkeypatch.setattr(HarvestPlanItem, 'save', fail_save)
        resp = self._save(writer_client, {
            FIELD_HARVEST_PLAN_ID: plan.id,
            FIELD_PARCEL_ID: parcels[0].id,
            FIELD_YEAR_PLANNED: 2027,
            FIELD_VOLUME_PLANNED_M3: '150',
        })

        assert resp.status_code == 400
        assert resp.json()[MESSAGE] == S.ERROR_GENERIC

    def test_update_missing_version_conflicts(self, writer_client, planned_item):
        resp = self._save(writer_client, {
            ROW_ID: planned_item.id,
            FIELD_PARCEL_ID: planned_item.parcel_id,
            FIELD_YEAR_PLANNED: planned_item.year_planned + 1,
            FIELD_VOLUME_PLANNED_M3: str(planned_item.volume_planned_m3),
            FIELD_NOTE: 'missing version update',
        })

        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        planned_item.refresh_from_db()
        assert planned_item.note != 'missing version update'

    def test_update_rejects_target_change_when_linked_harvest_exists(
        self, writer_client, planned_item, parcels, products, crews,
    ):
        Harvest.objects.create(
            date=date_type(2025, 6, 1), product=products[0], crew=crews[0],
            parcel=planned_item.parcel, harvest_plan_item=planned_item,
            mass_q=Decimal('10.00'), volume_m3=Decimal('1.000'),
            damaged=planned_item.damaged, unhealthy=planned_item.unhealthy,
            psr=planned_item.psr,
        )

        resp = self._save(writer_client, {
            ROW_ID: planned_item.id,
            VERSION: planned_item.version,
            FIELD_PARCEL_ID: parcels[1].id,
            FIELD_YEAR_PLANNED: planned_item.year_planned,
            FIELD_VOLUME_PLANNED_M3: str(planned_item.volume_planned_m3),
            FIELD_NOTE: 'target changed',
        })

        assert resp.status_code == 400
        assert S.ERR_PLAN_ITEM_LINKED_HARVESTS_INVARIANT in resp.json()[MESSAGE]
        planned_item.refresh_from_db()
        assert planned_item.parcel_id == parcels[0].id
        assert planned_item.note != 'target changed'

    def test_delete_planned(self, writer_client, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/item/delete/{planned_item.id}/',
            data=json.dumps({VERSION: planned_item.version}),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert not HarvestPlanItem.objects.filter(id=planned_item.id).exists()

    def test_delete_saves_nonce(self, writer_client, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/item/delete/{planned_item.id}/',
            data=json.dumps({
                VERSION: planned_item.version,
                FIELD_NONCE: 'item-delete-nonce',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='item-delete-nonce').exists()

    def test_delete_stale_version_conflicts(self, writer_client, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/item/delete/{planned_item.id}/',
            data=json.dumps({VERSION: planned_item.version + 1}),
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert HarvestPlanItem.objects.filter(id=planned_item.id).exists()

    def test_delete_blocked_when_not_planned(self, writer_client, planned_item):
        planned_item.state = HarvestPlanItemState.OPEN
        planned_item.version += 1
        planned_item.save()
        resp = writer_client.post(
            f'/api/piano-di-taglio/item/delete/{planned_item.id}/',
            data=json.dumps({VERSION: planned_item.version}),
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert (S.ERR_PLAN_ITEM_STATE_NOT_PLANNED
                in resp.json()[MESSAGE])

    def test_item_data_view(self, writer_client, planned_item):
        resp = writer_client.get(
            f'/api/piano-di-taglio/item/data/{planned_item.id}/',
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data[ROW_ID] == planned_item.id
        assert TRANSITION_RECORDS in data


# ---------------------------------------------------------------------------
# Transition save (Apri / Chiudi cantiere)
# ---------------------------------------------------------------------------

class TestTransitionSave:
    def _post(self, client, payload):
        return client.post(
            '/api/piano-di-taglio/transition/save/',
            data=json.dumps(payload), content_type='application/json',
        )

    def test_apri_cantiere_from_planned(self, writer_client, planned_item):
        resp = self._post(writer_client, {
            FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id,
            FIELD_OPEN: True,
            FIELD_DATE: '2024-09-01',
            FIELD_NOTE: 'permit 42/2024',
        })
        assert resp.status_code == 200
        planned_item.refresh_from_db()
        assert planned_item.state == HarvestPlanItemState.OPEN
        assert planned_item.date_actual.isoformat() == '2024-09-01'
        assert HarvestTransition.objects.filter(
            harvest_plan_item=planned_item, open=True,
        ).exists()

    def test_chiudi_cantiere_from_open(self, writer_client, planned_item):
        planned_item.state = HarvestPlanItemState.OPEN
        planned_item.save()
        resp = self._post(writer_client, {
            FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id,
            FIELD_OPEN: False,
            FIELD_DATE: '2024-12-01',
        })
        assert resp.status_code == 200
        planned_item.refresh_from_db()
        assert planned_item.state == HarvestPlanItemState.CLOSED

    def test_invalid_transition_rejected(self, writer_client, planned_item):
        """Chiudi from PLANNED is not in the allowed transitions table."""
        resp = self._post(writer_client, {
            FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id,
            FIELD_OPEN: False,
            FIELD_DATE: '2024-09-01',
        })
        assert resp.status_code == 400
        assert (S.ERR_TRANSITION_INVALID_STATE
                in resp.json()[MESSAGE])

    def test_malformed_slash_date_rejected(self, writer_client, planned_item):
        resp = self._post(writer_client, {
            FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id,
            FIELD_OPEN: True,
            FIELD_DATE: '09/2024',
        })

        assert resp.status_code == 400
        assert S.ERR_DATE_INVALID in resp.json()[MESSAGE]


# ---------------------------------------------------------------------------
# Per-item Esporta
# ---------------------------------------------------------------------------

class TestItemExport:
    def test_item_export_shape(self, writer_client, planned_item):
        resp = writer_client.get(
            f'/api/piano-di-taglio/item/export/{planned_item.id}/',
        )
        assert resp.status_code == 200
        assert resp['Content-Type'] == 'application/zip'
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = set(zf.namelist())
        assert f'martellate_{planned_item.id}.csv' in names
        assert f'prelievi_{planned_item.id}.csv' in names

    def test_item_export_uses_prefixed_percent_headers(
        self, writer_client, planned_item, crews, products, tractors, species,
    ):
        tractor = tractors[0]
        tractor.name = 'T1'
        tractor.save(update_fields=['name'])
        harvest = Harvest.objects.create(
            date=date_type(2025, 8, 1), product=products[0],
            parcel=planned_item.parcel, crew=crews[0], mass_q=Decimal('10'),
            harvest_plan_item=planned_item,
        )
        HarvestTractor.objects.create(harvest=harvest, tractor=tractor, percent=100)

        resp = writer_client.get(
            f'/api/piano-di-taglio/item/export/{planned_item.id}/',
        )
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        text = zf.read(f'prelievi_{planned_item.id}.csv').decode('utf-8-sig')
        delimiter, _ = csv_io.export_format()
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        assert f'{S.CSV_COL_SPECIES_PREFIX}{species[0].common_name}' in rows[0]
        assert f'{S.CSV_COL_TRACTOR_PREFIX}T1' in rows[0]
        assert 'Fiat 110-90' not in rows[0]

    def test_export_numero_is_mark_number_not_id(
        self, writer_client, planned_item, species,
    ):
        """The martellate CSV 'Numero' column must carry TreeMark.number,
        not the DB id (which just starts at 1)."""
        tree = Tree.objects.create(
            species=species[0], parcel=planned_item.parcel,
            lat=38.5, lon=16.3, acc_m=5)
        tm = TreeMark.objects.create(
            harvest_plan_item=planned_item, tree=tree, number=1440,
            date=date_type(2025, 6, 1), d_cm=30, h_m=Decimal('20.0'),
            h_measured=False, volume_m3=Decimal('0.7'), mass_q=Decimal('0.5'),
            lat=38.5, lon=16.3, acc_m=5, operator='Mario')
        assert tm.number != tm.id  # the bug only manifests when they differ

        resp = writer_client.get(
            f'/api/piano-di-taglio/item/export/{planned_item.id}/')
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        text = zf.read(f'martellate_{planned_item.id}.csv').decode('utf-8-sig')
        delimiter, _ = csv_io.export_format()
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        numero = rows[1][rows[0].index(S.CSV_COL_NUMBER)]
        assert numero == '1440', f'expected mark number 1440, got {numero!r}'


# ---------------------------------------------------------------------------
# Auth gates
# ---------------------------------------------------------------------------

class TestAuth:
    def test_reader_cannot_save_plan(self, reader_client, db):
        resp = reader_client.post(
            '/api/piano-di-taglio/plan/save/',
            data=json.dumps({
                FIELD_NAME: 'x',
                FIELD_YEAR_START: 2024, FIELD_YEAR_END: 2030,
            }),
            content_type='application/json',
        )
        assert resp.status_code == 403

    def test_reader_can_read_plans(self, reader_client, db, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        from apps.base.digests import generate_harvest_plans
        generate_harvest_plans()
        resp = reader_client.get('/api/piano-di-taglio/plans/data/')
        assert resp.status_code == 200


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


class TestDigestInvalidation:
    """Regression tests: transition saves must update the state in the
    harvest_plan_items digest."""

    @staticmethod
    def _items_state(client, item_id, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = client.get('/api/piano-di-taglio/items/data/')
        d = _read_gzip_json(resp)
        row = next(r for r in d[ROWS]
                   if r[d[COLUMNS].index(ROW_ID)] == item_id)
        return row[d[COLUMNS].index(S.COL_STATE)]

    def test_transition_invalidates_items_digest(
        self, writer_client, planned_item, tmp_path, settings,
    ):
        state_before = self._items_state(
            writer_client, planned_item.id, tmp_path, settings,
        )
        assert state_before == S.STATE_PLANNED
        writer_client.post(
            '/api/piano-di-taglio/transition/save/',
            data=json.dumps({
                FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id,
                FIELD_OPEN: True,
                FIELD_DATE: '2024-09-01',
                FIELD_NOTE: '',
            }),
            content_type='application/json',
        )
        state_after = self._items_state(
            writer_client, planned_item.id, tmp_path, settings,
        )
        assert state_after == S.STATE_OPEN, (
            f'harvest_plan_items digest should reflect state change '
            f'(was {state_before}, now {state_after})'
        )

    def test_digest_is_not_browser_cacheable(
        self, writer_client, planned_item, tmp_path, settings,
    ):
        """Digests carry `no-store`.  The app caches them in-memory and
        revalidates itself; a heuristically-cached browser copy would serve
        a stale table after a write (the `Last-Modified`-without-
        `Cache-Control` trap)."""
        settings.DIGEST_DIR = tmp_path
        resp = writer_client.get('/api/piano-di-taglio/items/data/')
        assert 'no-store' in resp['Cache-Control']

    def test_unchanged_digest_still_304s(
        self, writer_client, planned_item, tmp_path, settings,
    ):
        """`no-store` must not cost bandwidth: an unchanged digest still
        answers 304 to a conditional GET (cache.js sends If-Modified-Since)."""
        settings.DIGEST_DIR = tmp_path
        url = '/api/piano-di-taglio/items/data/'
        r1 = writer_client.get(url)
        r2 = writer_client.get(url, HTTP_IF_MODIFIED_SINCE=r1['Last-Modified'])
        assert r2.status_code == 304

    def test_mark_edit_and_delete_reach_the_digest(
        self, writer_client, planned_item, species, tmp_path, settings,
    ):
        """Edits and deletes must be reflected by the per-item mark digest
        once it is re-served (lazy regeneration on read)."""
        settings.DIGEST_DIR = tmp_path
        sp = species[0]
        url = f'/api/piano-di-taglio/mark-trees/{planned_item.id}/'
        save, delete = '/api/piano-di-taglio/mark/save/', '/api/piano-di-taglio/mark/delete/'
        body = {
            FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id, FIELD_SPECIES_ID: sp.id,
            FIELD_D_CM: 30, FIELD_H_M: '20.0', FIELD_H_MEASURED: False,
            FIELD_VOLUME_M3: '0.7022', FIELD_MASS_Q: '0.56',
            FIELD_LAT: 38.5, FIELD_LON: 16.3, FIELD_OPERATOR: 'Mario',
            FIELD_DATE: '2025-06-01', FIELD_NONCE: 'c1',
        }
        post = lambda u, b: writer_client.post(
            u, data=json.dumps(b), content_type='application/json')

        tm_id = post(save, body).json()[ROW_ID]
        d = _read_gzip_json(writer_client.get(url))
        dcol = d[COLUMNS].index(S.COL_D_CM)
        assert d[ROWS][0][dcol] == 30

        post(save, {**body, ROW_ID: tm_id, FIELD_D_CM: 40, VERSION: 1, FIELD_NONCE: 'e1'})
        d = _read_gzip_json(writer_client.get(url))
        assert d[ROWS][0][dcol] == 40, 'digest did not reflect the edit'

        post(delete, {ROW_ID: tm_id, VERSION: 2, FIELD_NONCE: 'd1'})
        d = _read_gzip_json(writer_client.get(url))
        assert len(d[ROWS]) == 0, 'digest did not reflect the delete'


# ---------------------------------------------------------------------------
# Tree-mark CRUD
# ---------------------------------------------------------------------------

class TestMarkSave:
    SAVE_URL = '/api/piano-di-taglio/mark/save/'
    DELETE_URL = '/api/piano-di-taglio/mark/delete/'

    def test_reader_form_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/piano-di-taglio/mark/form/')
        assert resp.status_code == 403

    def _mark_body(self, item, species, **overrides):
        body = {
            FIELD_HARVEST_PLAN_ITEM_ID: item.id,
            FIELD_SPECIES_ID: species.id,
            FIELD_D_CM: 30,
            FIELD_H_M: '20.0',
            FIELD_H_MEASURED: False,
            FIELD_VOLUME_M3: '0.7022',
            FIELD_MASS_Q: '0.56',
            FIELD_LAT: 38.5,
            FIELD_LON: 16.3,
            FIELD_OPERATOR: 'Mario',
            FIELD_DATE: '2025-06-01',
            FIELD_NONCE: 'test-nonce-1',
        }
        body.update(overrides)
        return body

    def test_create_mark(self, writer_client, planned_item, species):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp, **{FIELD_NUMBER: 12},
            )),
            content_type='application/json',
        )
        assert resp.status_code == 200
        data = resp.json()
        assert any(p[DATA_ID].startswith('mark_trees_') for p in data[PATCHES])
        assert any(p[DATA_ID] == 'harvest_plan_items' for p in data[PATCHES])
        assert TreeMark.objects.count() == 1
        tm = TreeMark.objects.first()
        assert tm.number == 12
        assert tm.d_cm == 30
        assert tm.operator == 'Mario'

    def test_create_mark_allows_blank_number(
        self, writer_client, planned_item, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, species[0])),
            content_type='application/json',
        )

        assert resp.status_code == 200
        assert TreeMark.objects.get().number is None

    def test_rejects_zero_diameter_or_height(
        self, writer_client, planned_item, species,
    ):
        """A mark needs D and h > 0."""
        for i, override in enumerate(({FIELD_D_CM: 0}, {FIELD_H_M: '0'})):
            resp = writer_client.post(
                self.SAVE_URL,
                data=json.dumps(self._mark_body(
                    planned_item, species[0],
                    **{**override, FIELD_NONCE: f'zero-{i}'})),
                content_type='application/json',
            )
            assert resp.status_code == 400, (override, resp.content)
        assert TreeMark.objects.count() == 0

    def test_h_measured_string_zero_is_false(
        self, writer_client, planned_item, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, species[0],
                **{FIELD_H_MEASURED: '0', FIELD_NONCE: 'hmeas-0'},
            )),
            content_type='application/json',
        )
        assert resp.status_code == 200
        tm = TreeMark.objects.first()
        assert tm.h_measured is False

    def test_h_measured_string_one_is_true(
        self, writer_client, planned_item, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, species[0],
                **{FIELD_H_MEASURED: '1', FIELD_NONCE: 'hmeas-1'},
            )),
            content_type='application/json',
        )
        assert resp.status_code == 200
        tm = TreeMark.objects.first()
        assert tm.h_measured is True

    def test_create_auto_advances_to_marked(
        self, writer_client, planned_item, species,
    ):
        assert planned_item.state == HarvestPlanItemState.PLANNED
        writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, species[0])),
            content_type='application/json',
        )
        planned_item.refresh_from_db()
        assert planned_item.state == HarvestPlanItemState.MARKED

    def test_create_with_null_volume(
        self, writer_client, planned_item, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, species[0],
                **{FIELD_VOLUME_M3: '', FIELD_MASS_Q: '',
                   FIELD_NONCE: 'null-vol'},
            )),
            content_type='application/json',
        )
        assert resp.status_code == 200
        tm = TreeMark.objects.first()
        assert tm.volume_m3 is None
        assert tm.mass_q is None

    def test_create_rejects_negative_volume_and_mass(
        self, writer_client, planned_item, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, species[0],
                **{FIELD_VOLUME_M3: '-1', FIELD_MASS_Q: '-2'},
            )),
            content_type='application/json',
        )

        assert resp.status_code == 400
        assert S.ERR_MARK_VOLUME_NEGATIVE in resp.json()[MESSAGE]
        assert S.ERR_MARK_MASS_NEGATIVE in resp.json()[MESSAGE]

    def test_create_materializes_volume(
        self, writer_client, planned_item, species,
    ):
        writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, species[0],
                **{FIELD_VOLUME_M3: '1.500'},
            )),
            content_type='application/json',
        )
        planned_item.refresh_from_db()
        assert planned_item.volume_marked_m3 == Decimal('1.500')

    def test_parcel_scoped_mark_rejects_other_submitted_parcel(
        self, writer_client, planned_item, parcels, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, species[0],
                **{FIELD_PARCEL_ID: parcels[1].id},
            )),
            content_type='application/json',
        )

        assert resp.status_code == 400
        assert S.ERR_MARK_PARCEL_NOT_IN_TARGET in resp.json()[MESSAGE]
        assert TreeMark.objects.count() == 0

    def test_update_mark(self, writer_client, planned_item, species):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, sp)),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        resp2 = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp,
                **{ROW_ID: tm_id, VERSION: version, FIELD_D_CM: 40,
                   FIELD_NONCE: 'test-nonce-2'},
            )),
            content_type='application/json',
        )
        assert resp2.status_code == 200
        tm = TreeMark.objects.get(id=tm_id)
        assert tm.d_cm == 40

    def test_update_mark_stale_version_conflicts(
        self, writer_client, planned_item, species,
    ):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, sp)),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        stale_version = next(p for p in data[PATCHES]
                             if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        tm = TreeMark.objects.get(id=tm_id)
        tm.d_cm = 35
        tm.version += 1
        tm.save(update_fields=[FIELD_D_CM, VERSION])

        conflict = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp,
                **{ROW_ID: tm_id, VERSION: stale_version, FIELD_D_CM: 40,
                   FIELD_NONCE: 'mark-stale-update'},
            )),
            content_type='application/json',
        )

        assert conflict.status_code == 400
        payload = conflict.json()
        assert payload[STATUS] == STATUS_CONFLICT
        patch = payload[PATCHES][0]
        tm.refresh_from_db()
        assert tm.d_cm == 35
        assert patch == {
            DATA_ID: f'mark_trees_{planned_item.id}',
            ROW_ID: tm.id,
            RECORD: build_tree_mark_record(tm),
        }

    def test_update_mark_rejects_submitted_item_mismatch(
        self, writer_client, plan, planned_item, parcels, species,
    ):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, sp)),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        other_item = HarvestPlanItem.objects.create(
            harvest_plan=plan,
            parcel=parcels[1],
            year_planned=2025,
            volume_planned_m3=Decimal('100.0'),
            state=HarvestPlanItemState.PLANNED,
        )

        resp2 = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                other_item, sp,
                **{ROW_ID: tm_id, VERSION: version, FIELD_D_CM: 40,
                   FIELD_PARCEL_ID: parcels[1].id,
                   FIELD_NONCE: 'wrong-item-update'},
            )),
            content_type='application/json',
        )

        assert resp2.status_code == 400
        assert S.ERR_MARK_ITEM_MISMATCH in resp2.json()[MESSAGE]
        tm = TreeMark.objects.get(id=tm_id)
        assert tm.harvest_plan_item_id == planned_item.id
        assert tm.d_cm == 30
        planned_item.refresh_from_db()
        other_item.refresh_from_db()
        assert planned_item.volume_marked_m3 == Decimal('0.702')
        assert other_item.volume_marked_m3 is None

    def test_update_region_wide_mark_can_change_parcel(
        self, writer_client, plan, parcels, species,
    ):
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, region=parcels[0].region, parcel=None,
            damaged=True, year_planned=2025,
            volume_planned_m3=Decimal('100.0'),
            state=HarvestPlanItemState.PLANNED,
        )
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                item, sp, **{FIELD_PARCEL_ID: parcels[0].id},
            )),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]

        resp2 = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                item, sp,
                **{ROW_ID: tm_id, VERSION: version,
                   FIELD_PARCEL_ID: parcels[1].id,
                   FIELD_NONCE: 'change-mark-parcel'},
            )),
            content_type='application/json',
        )

        assert resp2.status_code == 200
        tm = TreeMark.objects.select_related('tree').get(id=tm_id)
        assert tm.tree.parcel_id == parcels[1].id

    def test_create_mark_rejects_duplicate_number(
        self, writer_client, planned_item, species,
    ):
        sp = species[0]
        first = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp, **{FIELD_NUMBER: 7},
            )),
            content_type='application/json',
        )
        assert first.status_code == 200

        second = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp,
                **{FIELD_NUMBER: 7, FIELD_NONCE: 'duplicate-mark-number'},
            )),
            content_type='application/json',
        )

        assert second.status_code == 400
        assert S.ERR_MARK_NUMBER_DUPLICATE.format(7) in second.json()[MESSAGE]
        assert TreeMark.objects.count() == 1

    def test_update_mark_rejects_invalid_number_and_preserves_existing(
        self, writer_client, planned_item, species,
    ):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp, **{FIELD_NUMBER: 7},
            )),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]

        resp2 = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp,
                **{ROW_ID: tm_id, VERSION: version, FIELD_NUMBER: 'abc',
                   FIELD_NONCE: 'bad-number'},
            )),
            content_type='application/json',
        )

        assert resp2.status_code == 400
        assert resp2.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert S.ERR_MARK_NUMBER_INVALID in resp2.json()[MESSAGE]
        assert TreeMark.objects.get(id=tm_id).number == 7

    def test_update_mark_missing_number_field_preserves_existing(
        self, writer_client, planned_item, species,
    ):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp, **{FIELD_NUMBER: 7},
            )),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        body = self._mark_body(
            planned_item, sp,
            **{ROW_ID: tm_id, VERSION: version, FIELD_NONCE: 'missing-number'},
        )
        assert FIELD_NUMBER not in body

        resp2 = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(body),
            content_type='application/json',
        )

        assert resp2.status_code == 200
        assert TreeMark.objects.get(id=tm_id).number == 7

    def test_mark_form_renders_blank_nullable_number(
        self, writer_client, planned_item, species,
    ):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, sp, **{FIELD_NUMBER: ''})),
            content_type='application/json',
        )
        tm_id = resp.json()[ROW_ID]

        form = writer_client.get(
            f'/api/piano-di-taglio/mark/form/{tm_id}/',
        )

        assert form.status_code == 200
        html = form.json()[HTML]
        assert 'value="None"' not in html
        assert 'id="tf-number" name="number" min="1"' in html
        assert 'value=""' in html

    def test_update_mark_can_clear_number(self, writer_client, planned_item, species):
        sp = species[0]
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp, **{FIELD_NUMBER: 7},
            )),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]

        resp2 = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(
                planned_item, sp,
                **{ROW_ID: tm_id, VERSION: version, FIELD_NUMBER: '',
                   FIELD_NONCE: 'clear-number'},
            )),
            content_type='application/json',
        )

        assert resp2.status_code == 200
        assert TreeMark.objects.get(id=tm_id).number is None

    def test_delete_mark(self, writer_client, planned_item, species):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, species[0])),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        resp2 = writer_client.post(
            self.DELETE_URL,
            data=json.dumps({ROW_ID: tm_id, VERSION: version}),
            content_type='application/json',
        )
        assert resp2.status_code == 200
        assert TreeMark.objects.count() == 0
        assert Tree.objects.count() == 0

    def test_delete_mark_stale_version_conflicts(
        self, writer_client, planned_item, species,
    ):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, species[0])),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        stale_version = next(p for p in data[PATCHES]
                             if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        tm = TreeMark.objects.get(id=tm_id)
        tm.version += 1
        tm.save(update_fields=[VERSION])

        conflict = writer_client.post(
            self.DELETE_URL,
            data=json.dumps({ROW_ID: tm_id, VERSION: stale_version}),
            content_type='application/json',
        )

        assert conflict.status_code == 400
        payload = conflict.json()
        assert payload[STATUS] == STATUS_CONFLICT
        tm.refresh_from_db()
        assert TreeMark.objects.filter(id=tm_id).exists()
        assert Tree.objects.filter(id=tm.tree_id).exists()
        assert payload[PATCHES][0] == {
            DATA_ID: f'mark_trees_{planned_item.id}',
            ROW_ID: tm.id,
            RECORD: build_tree_mark_record(tm),
        }

    def test_delete_mark_rejects_malformed_row_id(self, writer_client):
        resp = writer_client.post(
            self.DELETE_URL,
            data=json.dumps({ROW_ID: 'not-an-id', VERSION: '1'}),
            content_type='application/json',
        )

        assert resp.status_code == 400
        assert S.ERR_ROW_ID_INVALID in resp.json()[MESSAGE]

    def test_delete_closed_rejected(self, writer_client, planned_item, species):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, species[0])),
            content_type='application/json',
        )
        data = resp.json()
        tm_id = data[ROW_ID]
        version = next(p for p in data[PATCHES]
                       if p[DATA_ID].startswith('mark_trees_'))[RECORD][1]
        planned_item.state = HarvestPlanItemState.CLOSED
        planned_item.version += 1
        planned_item.save()
        resp2 = writer_client.post(
            self.DELETE_URL,
            data=json.dumps({ROW_ID: tm_id, VERSION: version}),
            content_type='application/json',
        )
        assert resp2.status_code == 400

    def test_create_closed_rejected(
        self, writer_client, planned_item, species,
    ):
        planned_item.state = HarvestPlanItemState.CLOSED
        planned_item.version += 1
        planned_item.save()
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps(self._mark_body(planned_item, species[0])),
            content_type='application/json',
        )
        assert resp.status_code == 400

    def test_validation_errors(self, writer_client, planned_item, species):
        resp = writer_client.post(
            self.SAVE_URL,
            data=json.dumps({
                FIELD_HARVEST_PLAN_ITEM_ID: planned_item.id,
                FIELD_NONCE: 'x',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data[STATUS] == STATUS_VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Tree-mark CSV import
# ---------------------------------------------------------------------------

class TestMarkCSVImport:
    IMPORT_URL = '/api/piano-di-taglio/mark/import-csv/'

    @staticmethod
    def _csv_content(rows):
        """Build an ipso-format CSV (semicolon-separated, UTF-8 BOM)."""
        header = 'Data;Compresa;Particella;Catastrofata;Numero;Genere;D_cm;H_m;H_measured;Lat;Lon;Acc_m;Operatore'
        lines = [header]
        for r in rows:
            lines.append(';'.join(str(x) for x in r))
        return ('﻿' + '\r\n'.join(lines) + '\r\n').encode('utf-8')

    @staticmethod
    def _post(client, item, csv_bytes, nonce=None):
        body = {
            FIELD_HARVEST_PLAN_ITEM_ID: item.id,
            FIELD_FILE: _csv_b64(csv_bytes),
        }
        if nonce:
            body[FIELD_NONCE] = nonce
        return client.post(
            TestMarkCSVImport.IMPORT_URL,
            data=json.dumps(body),
            content_type='application/json',
        )

    def test_v2_fingerprint_covers_every_canonical_field(self):
        base = {
            'source_row': 1,
            'date': date_type(2025, 6, 1),
            'parcel_id': 10,
            'species_id': 20,
            'number': 30,
            'd_cm': 40,
            'h_m': Decimal('21.50'),
            'h_measured': False,
            'lat': 38.5,
            'lon': 16.3,
            'acc_m': 5,
            'operator': 'Mario',
        }
        variants = {
            'source_row': 2,
            'date': date_type(2025, 6, 2),
            'parcel_id': 11,
            'species_id': 21,
            'number': 31,
            'd_cm': 41,
            'h_m': Decimal('21.51'),
            'h_measured': True,
            'lat': 38.6,
            'lon': 16.4,
            'acc_m': 6,
            'operator': 'Luigi',
        }
        fingerprint = csv_mark_fingerprint(**base)

        assert fingerprint.startswith('v2:')
        for field, value in variants.items():
            assert csv_mark_fingerprint(**(base | {field: value})) != fingerprint

    def test_import_basic(self, writer_client, planned_item, species, parcels):
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', '1',
             sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ])
        resp = self._post(writer_client, planned_item, csv_bytes)
        assert resp.status_code == 200
        data = resp.json()
        assert data['imported'] == 1
        assert TreeMark.objects.count() == 1
        tm = TreeMark.objects.first()
        assert tm.import_fingerprint is not None
        assert tm.number == 1
        assert tm.d_cm == 30

    def test_import_rejects_missing_lon_header(
        self, writer_client, planned_item, species, parcels,
    ):
        csv_bytes = self._csv_content([
            ['01/06/2025', parcels[0].region.name, parcels[0].name, '0', '1',
             species[0].common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ]).replace(b'Lon', b'Lng')

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 400
        assert resp.json()[MESSAGE] == S.ERR_CSV_MISSING_COLS.format(S.CSV_COL_LON)
        assert TreeMark.objects.count() == 0

    def test_import_rejects_other_parcel_in_same_region(
        self, writer_client, planned_item, species, parcels,
    ):
        other_parcel = parcels[1]
        assert other_parcel.region_id == planned_item.parcel.region_id
        csv_bytes = self._csv_content([
            ['01/06/2025', other_parcel.region.name, other_parcel.name, '0', '1',
             species[0].common_name, '30', '20,0', '0', '', '', '', 'Mario'],
        ])

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 400
        assert S.ERR_MARK_ROW_TARGET_MISMATCH.format(1) in resp.json()[MESSAGE]
        assert TreeMark.objects.count() == 0

    def test_import_accepts_legacy_ipso_species_header(
        self, writer_client, planned_item, species, parcels,
    ):
        csv_bytes = self._csv_content([
            ['01/06/2025', parcels[0].region.name, parcels[0].name, '0', '1',
             species[0].common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ]).replace(b'Genere', b'Specie')

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 200
        assert resp.json()['imported'] == 1
        assert TreeMark.objects.get().tree.species == species[0]

    def test_import_rejects_invalid_or_non_positive_mark_numbers(
        self, writer_client, planned_item, species, parcels,
    ):
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', 'abc',
             sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
            ['01/06/2025', parcel.region.name, parcel.name, '0', '0',
             sp.common_name, '31', '21,0', '0', '38.6', '16.4', '5', 'Mario'],
        ])

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert TreeMark.objects.count() == 0


    def test_import_rejects_duplicate_mark_numbers(
        self, writer_client, planned_item, species, parcels,
    ):
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', '7',
             sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
            ['01/06/2025', parcel.region.name, parcel.name, '0', '7',
             sp.common_name, '31', '21,0', '0', '38.6', '16.4', '5', 'Mario'],
        ])

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 400
        assert S.ERR_MARK_NUMBER_DUPLICATE.format(7) in resp.json()[MESSAGE]
        assert TreeMark.objects.count() == 0

    def test_import_preserves_blank_csv_mark_numbers(
        self, writer_client, planned_item, species, parcels,
    ):
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', '',
             sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
            ['01/06/2025', parcel.region.name, parcel.name, '0', '',
             sp.common_name, '31', '21,0', '0', '38.6', '16.4', '5', 'Mario'],
        ])

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 200
        assert list(
            TreeMark.objects.order_by('id').values_list('number', flat=True)
        ) == [None, None]

    def test_import_saves_nonce(self, writer_client, planned_item, species, parcels):
        csv_bytes = self._csv_content([
            ['01/06/2025', parcels[0].region.name, parcels[0].name, '0', '1',
             species[0].common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ])
        resp = self._post(
            writer_client, planned_item, csv_bytes, nonce='mark-import-nonce',
        )
        assert resp.status_code == 200
        assert UsedNonce.objects.filter(nonce='mark-import-nonce').exists()

    def test_import_rejects_fractional_diameter(
        self, writer_client, planned_item, species, parcels,
    ):
        """A fractional D_cm is flagged, not silently truncated to int."""
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', '1',
             sp.common_name, '30,5', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ])
        resp = self._post(writer_client, planned_item, csv_bytes)
        assert resp.status_code == 400
        assert TreeMark.objects.count() == 0

    @pytest.mark.parametrize(('index', 'value'), [
        (6, '0'),       # D_cm
        (7, '0'),       # H_m
        (8, 'maybe'),   # H_measured
        (9, 'bad'),     # Lat
        (10, 'bad'),    # Lon
        (11, 'bad'),    # Acc_m
    ])
    def test_import_rejects_invalid_measurements_and_optionals(
        self, writer_client, planned_item, species, parcels, index, value,
    ):
        sp = species[0]
        parcel = parcels[0]
        row = [
            '01/06/2025', parcel.region.name, parcel.name, '0', '1',
            sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario',
        ]
        row[index] = value
        csv_bytes = self._csv_content([row])

        resp = self._post(writer_client, planned_item, csv_bytes)

        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert TreeMark.objects.count() == 0

    def test_import_dedup(self, writer_client, planned_item, species, parcels):
        sp = species[0]
        parcel = parcels[0]
        row = ['01/06/2025', parcel.region.name, parcel.name, '0', '1',
               sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario']
        csv_bytes = self._csv_content([row])
        for _ in range(2):
            self._post(writer_client, planned_item, csv_bytes)
        assert TreeMark.objects.count() == 1

    def test_import_keeps_identical_unnumbered_source_rows(
        self, writer_client, planned_item, species, parcels,
    ):
        parcel = parcels[0]
        row = ['01/06/2025', parcel.region.name, parcel.name, '0', '',
               species[0].common_name, '30', '20,0', '0', '', '', '', 'Mario']
        csv_bytes = self._csv_content([row, row])

        first = self._post(writer_client, planned_item, csv_bytes)
        second = self._post(writer_client, planned_item, csv_bytes)

        assert first.status_code == 200
        assert first.json()['imported'] == 2
        assert second.status_code == 200
        assert second.json()['skipped_duplicates'] == 2
        assert TreeMark.objects.count() == 2

    def test_import_recognizes_legacy_fingerprint(
        self, writer_client, planned_item, species, parcels,
    ):
        parcel = parcels[0]
        species_name = species[0].common_name
        row = ['01/06/2025', parcel.region.name, parcel.name, '0', '1',
               species_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario']
        csv_bytes = self._csv_content([row])
        assert self._post(writer_client, planned_item, csv_bytes).status_code == 200
        TreeMark.objects.update(import_fingerprint=legacy_csv_mark_fingerprint(
            date=date_type(2025, 6, 1), species_name=species_name,
            d_cm=30, h_m=Decimal('20.0'), lat=38.5, lon=16.3,
            operator='Mario',
        ))

        retry = self._post(writer_client, planned_item, csv_bytes)

        assert retry.status_code == 200
        assert retry.json()['imported'] == 0
        assert retry.json()['skipped_duplicates'] == 1
        assert TreeMark.objects.count() == 1

    def test_import_auto_advances_state(
        self, writer_client, planned_item, species, parcels,
    ):
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', '1',
             sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ])
        self._post(writer_client, planned_item, csv_bytes)
        planned_item.refresh_from_db()
        assert planned_item.state == HarvestPlanItemState.MARKED

    def test_import_materializes_volume(
        self, writer_client, planned_item, species, parcels,
    ):
        sp = species[0]
        parcel = parcels[0]
        csv_bytes = self._csv_content([
            ['01/06/2025', parcel.region.name, parcel.name, '0', '1',
             sp.common_name, '30', '20,0', '0', '38.5', '16.3', '5', 'Mario'],
        ])
        self._post(writer_client, planned_item, csv_bytes)
        planned_item.refresh_from_db()
        assert planned_item.volume_marked_m3 is not None
        assert planned_item.volume_marked_m3 > 0

    def test_import_closed_rejected(
        self, writer_client, planned_item, species, parcels,
    ):
        planned_item.state = HarvestPlanItemState.CLOSED
        planned_item.version += 1
        planned_item.save()
        csv_bytes = self._csv_content([
            ['01/06/2025', parcels[0].region.name, parcels[0].name, '0', '1',
             species[0].common_name, '30', '20,0', '0', '', '', '', 'Mario'],
        ])
        resp = self._post(writer_client, planned_item, csv_bytes)
        assert resp.status_code == 400
