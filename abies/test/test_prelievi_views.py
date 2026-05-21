"""Tests for Prelievi API views: data, form, save, delete."""

import gzip
import json

import pytest
from django.test import Client

from apps.base.digests import generate_prelievi, mark_stale
from apps.base.models import (
    DigestStatus, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor


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
def harvest_plan(db):
    return HarvestPlan.objects.create(
        name='Test plan', year_start=2024, year_end=2034,
    )


@pytest.fixture
def open_item(harvest_plan, parcels):
    """A HarvestPlanItem in state OPEN (the Cantiere of the test harvest)."""
    return HarvestPlanItem.objects.create(
        harvest_plan=harvest_plan, parcel=parcels[0], year_planned=2024,
        state=HarvestPlanItemState.OPEN,
    )


@pytest.fixture
def region_wide_open_item(harvest_plan, regions):
    """Region-wide HarvestPlanItem (no parcel; damaged flag set)."""
    return HarvestPlanItem.objects.create(
        harvest_plan=harvest_plan, region=regions[0], year_planned=2024,
        state=HarvestPlanItemState.OPEN, damaged=True,
    )


@pytest.fixture
def harvest_fixtures(regions, eclasses, species, tractors, crews, products,
                     parcels, harvest_plan, open_item):
    """Return a dict of all reference fixtures needed for harvest operations."""
    return {
        'regions': regions, 'eclasses': eclasses, 'species': species,
        'tractors': tractors, 'crews': crews, 'products': products,
        'parcels': parcels,
        'harvest_plan': harvest_plan, 'open_item': open_item,
    }


@pytest.fixture
def sample_op(harvest_fixtures):
    """A saved Harvest for edit/delete tests."""
    f = harvest_fixtures
    op = Harvest.objects.create(
        date='2024-06-15', parcel=f['parcels'][0], crew=f['crews'][0],
        product=f['products'][0], mass_q=50, record1=999,
        harvest_plan_item=f['open_item'],
    )
    HarvestSpecies.objects.create(harvest=op, species=f['species'][0], percent=100)
    HarvestTractor.objects.create(harvest=op, tractor=f['tractors'][0], percent=100)
    return op


# ---------------------------------------------------------------------------
# Data endpoint
# ---------------------------------------------------------------------------

class TestDataView:
    def test_serves_gzip_json(self, writer_client, harvest_fixtures, sample_op, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()

        resp = writer_client.get('/api/prelievi/data/')
        assert resp.status_code == 200
        assert resp['Content-Type'] == 'application/json'
        assert resp['Content-Encoding'] == 'gzip'
        data = json.loads(gzip.decompress(resp.getvalue()))
        assert COLUMNS in data
        assert len(data[ROWS]) >= 1

    def test_304_on_not_modified(self, writer_client, harvest_fixtures, sample_op, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()

        resp1 = writer_client.get('/api/prelievi/data/')
        lm = resp1['Last-Modified']

        resp2 = writer_client.get('/api/prelievi/data/', HTTP_IF_MODIFIED_SINCE=lm)
        assert resp2.status_code == 304

    def test_requires_auth(self, db, harvest_fixtures):
        resp = Client().get('/api/prelievi/data/')
        assert resp.status_code == 302  # redirect to login


# ---------------------------------------------------------------------------
# Form endpoint
# ---------------------------------------------------------------------------

class TestFormView:
    def test_add_form_returns_html(self, writer_client, harvest_fixtures):
        resp = writer_client.get('/api/prelievi/form/')
        assert resp.status_code == 200
        data = resp.json()
        assert '<form' in data[HTML]
        assert 'id_date' in data[HTML]

    def test_edit_form_prepopulated(self, writer_client, harvest_fixtures, sample_op):
        resp = writer_client.get(f'/api/prelievi/form/{sample_op.id}/')
        assert resp.status_code == 200
        html = resp.json()[HTML]
        assert '2024-06-15' in html
        assert 'selected' in html

    def test_form_contains_species_and_tractors(self, writer_client, harvest_fixtures):
        resp = writer_client.get('/api/prelievi/form/')
        html = resp.json()[HTML]
        assert 'sp_' in html
        assert 'tr_' in html
        assert '100%' in html

    def test_edit_form_shows_percentages(self, writer_client, harvest_fixtures, sample_op):
        resp = writer_client.get(f'/api/prelievi/form/{sample_op.id}/')
        html = resp.json()[HTML]
        # The first species has 100% on sample_op
        assert 'value="100"' in html


# ---------------------------------------------------------------------------
# Save endpoint
# ---------------------------------------------------------------------------

class TestSaveView:
    def _post(self, client, data):
        return client.post(
            '/api/prelievi/save/',
            data=json.dumps(data),
            content_type='application/json',
        )

    def _base_payload(self, f, **overrides):
        """Canonical Create payload — Cantiere driven."""
        payload = {
            FIELD_DATE: '2024-07-01',
            FIELD_HARVEST_PLAN_ITEM_ID: str(f['open_item'].id),
            FIELD_CREW_ID: str(f['crews'][0].id),
            FIELD_PRODUCT_ID: str(f['products'][0].id),
            FIELD_MASS_Q: '10',
            FIELD_NOTE: '',
            'record1': '', 'record2': '',
            f'sp_{f["species"][0].id}': '100',
            f'tr_{f["tractors"][0].id}': '100',
        }
        payload.update(overrides)
        return payload

    def test_create_success(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{
                FIELD_MASS_Q: '30', FIELD_NOTE: 'test note',
                FIELD_NONCE: 'create-nonce-1',
                f'sp_{f["species"][0].id}': '60',
                f'sp_{f["species"][1].id}': '40',
            },
        ))
        assert resp.status_code == 200
        data = resp.json()
        assert data[DATA_ID] == 'prelievi'
        assert data[ROW_ID] > 0
        assert data[RECORD][2] == '2024-07-01'  # date is third column (after row_id, version)

        op = Harvest.objects.get(id=data[ROW_ID])
        assert float(op.mass_q) == 30.0
        assert op.note == 'test note'
        assert op.harvest_plan_item_id == f['open_item'].id
        assert HarvestSpecies.objects.filter(harvest=op).count() == 2
        assert HarvestTractor.objects.filter(harvest=op).count() == 1

    def test_create_marks_digest_stale(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        self._post(writer_client, self._base_payload(f))
        assert DigestStatus.objects.get(name='prelievi').stale is True
        assert DigestStatus.objects.get(name='parcel_year_production').stale is True
        assert DigestStatus.objects.get(name='harvest_plan_items').stale is True

    def test_create_auto_advances_state(self, writer_client, harvest_fixtures):
        """First Harvest on an OPEN item advances state to HARVESTING."""
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(f))
        assert resp.status_code == 200
        f['open_item'].refresh_from_db()
        assert f['open_item'].state == HarvestPlanItemState.HARVESTING

    def test_update_success(self, writer_client, harvest_fixtures, sample_op):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{
                ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
                FIELD_DATE: '2024-06-20', FIELD_MASS_Q: '60',
                'record1': '999',
            },
        ))
        assert resp.status_code == 200
        sample_op.refresh_from_db()
        assert str(sample_op.date) == '2024-06-20'
        assert float(sample_op.mass_q) == 60.0
        assert sample_op.version == 2

    def test_update_conflict(self, writer_client, harvest_fixtures, sample_op):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{
                ROW_ID: str(sample_op.id), VERSION: '999',
                FIELD_DATE: '2024-06-20', FIELD_MASS_Q: '60',
            },
        ))
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert RECORD in resp.json()

    def test_validation_error_missing_date(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_DATE: ''},
        ))
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert S.ERR_DATE_REQUIRED in resp.json()[MESSAGE]

    def test_validation_error_missing_cantiere(self, writer_client, harvest_fixtures):
        """New harvest without Cantiere → rejected with ERR_CANTIERE_REQUIRED."""
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_HARVEST_PLAN_ITEM_ID: ''},
        ))
        assert resp.status_code == 400
        assert S.ERR_CANTIERE_REQUIRED in resp.json()[MESSAGE]

    def test_validation_error_species_sum(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{f'sp_{f["species"][0].id}': '50'},
        ))
        assert resp.status_code == 400
        assert 'specie' in resp.json()[MESSAGE].lower()

    def test_vdp_duplicate(self, writer_client, harvest_fixtures, sample_op):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{'record1': '999'},
        ))
        assert resp.status_code == 400
        assert 'VDP' in resp.json()[MESSAGE]

    def test_validation_error_bad_quintals(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_MASS_Q: '-5'},
        ))
        assert resp.status_code == 400
        assert S.ERR_QUINTALS_POSITIVE in resp.json()[MESSAGE]

    def test_validation_error_non_numeric_quintals(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_MASS_Q: 'abc'},
        ))
        assert resp.status_code == 400
        assert S.ERR_QUINTALS_POSITIVE in resp.json()[MESSAGE]

    def test_validation_error_tractor_sum(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{f'tr_{f["tractors"][0].id}': '50'},
        ))
        assert resp.status_code == 400
        assert 'trattori' in resp.json()[MESSAGE].lower()

    def test_reader_forbidden(self, reader_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(reader_client, self._base_payload(f))
        assert resp.status_code == 403

    def test_region_wide_requires_parcel(self, writer_client,
                                         harvest_fixtures,
                                         region_wide_open_item):
        """Region-wide Cantiere → parcel_id is mandatory in the body."""
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_HARVEST_PLAN_ITEM_ID: str(region_wide_open_item.id)},
        ))
        assert resp.status_code == 400
        assert (S.ERR_PARCEL_REQUIRED_FOR_REGION_WIDE
                in resp.json()[MESSAGE])

    def test_region_wide_parcel_must_be_in_region(self, writer_client,
                                                  harvest_fixtures,
                                                  region_wide_open_item):
        """A parcel outside the item's region is rejected."""
        f = harvest_fixtures
        # parcels[2] is in regions[1]; region_wide_open_item is on regions[0].
        outsider = f['parcels'][2]
        assert outsider.region_id != region_wide_open_item.region_id
        resp = self._post(writer_client, self._base_payload(
            f, **{
                FIELD_HARVEST_PLAN_ITEM_ID: str(region_wide_open_item.id),
                'parcel_id': str(outsider.id),
            },
        ))
        assert resp.status_code == 400
        assert S.ERR_PARCEL_NOT_IN_REGION in resp.json()[MESSAGE]

    def test_region_wide_success(self, writer_client, harvest_fixtures,
                                 region_wide_open_item):
        """Region-wide Cantiere + parcel in the same region succeeds."""
        f = harvest_fixtures
        # parcels[0] and region_wide_open_item are both on regions[0].
        in_region = f['parcels'][0]
        assert in_region.region_id == region_wide_open_item.region_id
        resp = self._post(writer_client, self._base_payload(
            f, **{
                FIELD_HARVEST_PLAN_ITEM_ID: str(region_wide_open_item.id),
                'parcel_id': str(in_region.id),
            },
        ))
        assert resp.status_code == 200
        op = Harvest.objects.get(id=resp.json()[ROW_ID])
        assert op.parcel_id == in_region.id
        assert op.harvest_plan_item_id == region_wide_open_item.id
        # Damaged flag propagated from the item.
        assert op.damaged is True

    def test_parcel_scoped_ignores_submitted_parcel(self, writer_client,
                                                    harvest_fixtures):
        """For a parcel-scoped Cantiere, a stray parcel_id in the body is
        ignored — the item's parcel is authoritative."""
        f = harvest_fixtures
        # Try to mis-attribute to parcels[2] (Fabrizia).  The cantiere is
        # parcels[0] (Capistrano); the saved harvest must land on
        # parcels[0] regardless of what we submitted.
        resp = self._post(writer_client, self._base_payload(
            f, **{'parcel_id': str(f['parcels'][2].id)},
        ))
        assert resp.status_code == 200
        op = Harvest.objects.get(id=resp.json()[ROW_ID])
        assert op.parcel_id == f['parcels'][0].id

    def test_nonce_idempotency(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        payload = self._base_payload(f, **{FIELD_NONCE: 'idempotent-1'})
        resp1 = self._post(writer_client, payload)
        assert resp1.status_code == 200
        row_id_1 = resp1.json()[ROW_ID]

        # Replay with same nonce → same response, no new row.
        resp2 = self._post(writer_client, payload)
        assert resp2.status_code == 200
        assert resp2.json()[ROW_ID] == row_id_1
        assert Harvest.objects.count() == 1


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------

class TestDeleteView:
    def _post(self, client, data):
        return client.post(
            '/api/prelievi/delete/',
            data=json.dumps(data),
            content_type='application/json',
        )

    def test_delete_success(self, writer_client, harvest_fixtures, sample_op):
        resp = self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
            FIELD_NONCE: 'del-1',
        })
        assert resp.status_code == 200
        assert resp.json()[ROW_ID] == sample_op.id
        assert not Harvest.objects.filter(id=sample_op.id).exists()

    def test_delete_cascades_junctions(self, writer_client, harvest_fixtures, sample_op):
        sp_count_before = HarvestSpecies.objects.count()
        self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
        })
        assert HarvestSpecies.objects.count() < sp_count_before

    def test_delete_conflict(self, writer_client, harvest_fixtures, sample_op):
        resp = self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: '999',
        })
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_CONFLICT
        assert Harvest.objects.filter(id=sample_op.id).exists()

    def test_delete_not_found(self, writer_client, harvest_fixtures):
        resp = self._post(writer_client, {ROW_ID: '99999', VERSION: '1'})
        assert resp.status_code == 404

    def test_reader_forbidden(self, reader_client, harvest_fixtures, sample_op):
        resp = self._post(reader_client, {
            ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
        })
        assert resp.status_code == 403


# Import S for assertion comparisons
from config import strings as S  # noqa: E402
from config.constants import (
    COLUMNS, DATA_ID, FIELD_CREW_ID, FIELD_DATE, FIELD_HARVEST_PLAN_ITEM_ID,
    FIELD_MASS_Q, FIELD_NONCE, FIELD_NOTE, FIELD_PRODUCT_ID,
    HTML, MESSAGE, RECORD, ROWS, ROW_ID,
    STATUS, STATUS_CONFLICT, STATUS_VALIDATION_ERROR, VERSION,
)
