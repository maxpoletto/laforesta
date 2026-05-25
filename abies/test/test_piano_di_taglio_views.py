"""Tests for the Piano di taglio backend endpoints.

Covers plan CRUD, plan CSV import, plan-level Esporta CSV, plan-item
CRUD (including the state-gated delete), per-item Esporta CSV, and the
cantiere transition save view.  All write paths share the digest-stale
contract and the nonce-idempotency contract.
"""

import gzip
import io
import json
import zipfile
from decimal import Decimal

import pytest
from django.test import Client

from apps.base.models import (
    DigestStatus, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    HarvestTransition, ParcelPlanDetail, TreeHeightRegression,
)
from config import strings as S
from config.constants import (
    COLUMNS, DATA_ID, FIELD_CEDUO_FILE, FIELD_CREW_ID, FIELD_DAMAGED,
    FIELD_DATE, FIELD_DESCRIPTION, FIELD_FUSTAIA_FILE,
    FIELD_HARVEST_PLAN_ID, FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_INTERVENTION_AREA_HA, FIELD_MASS_Q, FIELD_NAME, FIELD_NONCE,
    FIELD_NOTE, FIELD_OPEN, FIELD_PARCEL_ID, FIELD_PRODUCT_ID,
    FIELD_PSR, FIELD_REGION_ID, FIELD_REGRESSION_FILE, FIELD_UNHEALTHY,
    FIELD_VOLUME_PLANNED_M3, FIELD_YEAR_END, FIELD_YEAR_PLANNED,
    FIELD_YEAR_START, HTML, MESSAGE, RECORD, ROW_ID, ROWS,
    STATUS, STATUS_CONFLICT, STATUS_VALIDATION_ERROR,
    TRANSITION_RECORDS, VERSION,
)


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
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert not HarvestPlan.objects.filter(id=plan.id).exists()

    def test_delete_blocked_when_active_item(self, writer_client, plan, planned_item):
        planned_item.state = HarvestPlanItemState.OPEN
        planned_item.save()
        resp = writer_client.post(
            f'/api/piano-di-taglio/plan/delete/{plan.id}/',
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
REGRESSION_CSV = (
    'Compresa,Genere,funzione,a,b,r2,n\r\n'
    'Capistrano,Abete,ln,12.0,3.5,0.85,42\r\n'
)


class TestPlanCSVImport:
    def _upload(self, client, **kwargs):
        return client.post(
            '/api/piano-di-taglio/plan/import-csv/',
            data=kwargs,
        )

    def test_import_three_files(self, writer_client, parcels, species):
        f = self._upload(
            writer_client,
            name='CSV plan',
            description='From CSV.',
            fustaia_file=io.BytesIO(FUSTAIA_CSV.encode('utf-8')),
            ceduo_file=io.BytesIO(CEDUO_CSV.encode('utf-8')),
            regression_file=io.BytesIO(REGRESSION_CSV.encode('utf-8')),
        )
        assert f.status_code == 200, f.json()
        data = f.json()
        assert data['n_items'] == 2
        assert data['n_regressions'] == 1
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

    def test_import_whole_region_fustaia(
        self, writer_client, plan, regions,
    ):
        # Particella = 'X' marks a whole-region item.  Note column must
        # contain "Catastrofato" or "Fitosanitario".
        csv_in = (
            f'Compresa;Particella;Anno;Volume previsto;Note\r\n'
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
            f'Compresa;Particella;Anno;Volume previsto;Note\r\n'
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
            f'Compresa;Particella;Anno;Volume previsto;Note\r\n'
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

    def test_import_into_existing_upserts_regression(
        self, writer_client, plan, parcels, species,
    ):
        self._upload(
            writer_client, harvest_plan_id=plan.id,
            regression_file=io.BytesIO(REGRESSION_CSV.encode('utf-8')),
        )
        regs = TreeHeightRegression.objects.filter(harvest_plan=plan)
        assert regs.count() == 1
        assert float(regs[0].a) == 12.0

        revised = (
            'Compresa,Genere,funzione,a,b,r2,n\r\n'
            'Capistrano,Abete,ln,15.0,4.0,0.90,55\r\n'
        )
        self._upload(
            writer_client, harvest_plan_id=plan.id,
            regression_file=io.BytesIO(revised.encode('utf-8')),
        )
        regs = TreeHeightRegression.objects.filter(harvest_plan=plan)
        assert regs.count() == 1
        assert float(regs[0].a) == 15.0

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
        assert set(zf.namelist()) == {
            'fustaia.csv', 'ceduo.csv', 'equazioni_ipsometro.csv',
        }
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
        for col in [S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL, S.COL_STATE,
                    S.COL_NOTE, S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED,
                    S.COL_VOLUME_ACTUAL]:
            assert col in piano_header, f'missing column {col}'

    def test_round_trip_whole_region_item(
        self, writer_client, plan, regions,
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
        fustaia_bytes = zf.read(S.CSV_FILE_FUSTAIA)
        text = fustaia_bytes.decode('utf-8')
        assert f';{S.PARCEL_WHOLE_REGION_MARK};' in text
        assert S.FLAG_DAMAGED in text

        before = HarvestPlanItem.objects.filter(
            harvest_plan=plan, parcel__isnull=True,
        ).count()
        reup = writer_client.post(
            '/api/piano-di-taglio/plan/import-csv/',
            data={
                'harvest_plan_id': plan.id,
                'fustaia_file': io.BytesIO(fustaia_bytes),
            },
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

        reup = writer_client.post(
            '/api/piano-di-taglio/plan/import-csv/',
            data={
                'harvest_plan_id': plan.id,
                'fustaia_file': io.BytesIO(fustaia_bytes),
            },
        )
        assert reup.status_code == 200, reup.json()
        items = HarvestPlanItem.objects.filter(harvest_plan=plan)
        assert items.count() == 1  # idempotent
        assert items[0].id == planned_item.id
        assert float(items[0].volume_planned_m3) == float(planned_item.volume_planned_m3)


# ---------------------------------------------------------------------------
# Item CRUD
# ---------------------------------------------------------------------------

class TestItemCRUD:
    def _save(self, client, payload):
        return client.post(
            '/api/piano-di-taglio/item/save/',
            data=json.dumps(payload), content_type='application/json',
        )

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

    def test_delete_planned(self, writer_client, planned_item):
        resp = writer_client.post(
            f'/api/piano-di-taglio/item/delete/{planned_item.id}/',
            content_type='application/json',
        )
        assert resp.status_code == 200
        assert not HarvestPlanItem.objects.filter(id=planned_item.id).exists()

    def test_delete_blocked_when_not_planned(self, writer_client, planned_item):
        planned_item.state = HarvestPlanItemState.OPEN
        planned_item.save()
        resp = writer_client.post(
            f'/api/piano-di-taglio/item/delete/{planned_item.id}/',
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


# ---------------------------------------------------------------------------
# Per-item Esporta CSV
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
