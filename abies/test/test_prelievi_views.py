"""Tests for Prelievi API views: data, form, save, delete, CSV import."""

import base64
import gzip
import json
import re
from datetime import date, timedelta

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

    def test_digest_aggregates_minor_species(self, writer_client, harvest_fixtures, tmp_path, settings):
        """Minor species quintals and percentages fold into Altro."""
        settings.DIGEST_DIR = tmp_path
        f = harvest_fixtures
        minor_sp = next(s for s in f['species'] if s.minor)
        other_sp = next(s for s in f['species'] if s.common_name == S.SPECIES_OTHER)
        op = Harvest.objects.create(
            date='2024-08-01', parcel=f['parcels'][0], crew=f['crews'][0],
            product=f['products'][0], mass_q=100, record1=600,
            harvest_plan_item=f['open_item'],
        )
        HarvestSpecies.objects.create(harvest=op, species=minor_sp, percent=100)
        HarvestTractor.objects.create(harvest=op, tractor=f['tractors'][0], percent=100)
        generate_prelievi()

        resp = writer_client.get('/api/prelievi/data/')
        data = json.loads(gzip.decompress(resp.getvalue()))
        cols = data[COLUMNS]
        # Minor species must NOT appear as a column.
        assert minor_sp.common_name not in cols
        # Altro must appear and carry the full 100 quintals.
        other_idx = cols.index(other_sp.common_name)
        row = next(r for r in data[ROWS] if r[0] == op.id)
        assert row[other_idx] == 100.0
        # Percentage column for Altro must be 100.
        pct_idx = cols.index(f'{other_sp.common_name} %')
        assert row[pct_idx] == 100

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

    def test_reader_forbidden(self, reader_client, db):
        resp = reader_client.get('/api/prelievi/form/')
        assert resp.status_code == 403

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

    def test_form_uses_tractor_name(self, writer_client, harvest_fixtures):
        tractor = harvest_fixtures['tractors'][0]
        tractor.name = 'T1'
        tractor.save(update_fields=['name'])
        resp = writer_client.get('/api/prelievi/form/')
        html = resp.json()[HTML]
        assert 'T1' in html
        assert 'Fiat 110-90' not in html

    def test_form_excludes_minor_species(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        minor = [s for s in f['species'] if s.minor]
        major = [s for s in f['species'] if not s.minor]
        assert minor, 'fixture must include at least one minor species'
        resp = writer_client.get('/api/prelievi/form/')
        html = resp.json()[HTML]
        for s in minor:
            assert f'sp_{s.id}' not in html
        for s in major:
            assert f'sp_{s.id}' in html

    def test_edit_form_aggregates_minor_pcts(self, writer_client, harvest_fixtures):
        """Editing a harvest with a minor species shows its % under Altro."""
        f = harvest_fixtures
        minor_sp = next(s for s in f['species'] if s.minor)
        other_sp = next(s for s in f['species'] if s.common_name == S.SPECIES_OTHER)
        op = Harvest.objects.create(
            date='2024-08-01', parcel=f['parcels'][0], crew=f['crews'][0],
            product=f['products'][0], mass_q=20, record1=500,
            harvest_plan_item=f['open_item'],
        )
        HarvestSpecies.objects.create(harvest=op, species=f['species'][0], percent=60)
        HarvestSpecies.objects.create(harvest=op, species=minor_sp, percent=40)
        HarvestTractor.objects.create(harvest=op, tractor=f['tractors'][0], percent=100)
        resp = writer_client.get(f'/api/prelievi/form/{op.id}/')
        html = resp.json()[HTML]
        # Altro input should carry the aggregated 40%.
        assert f'name="sp_{other_sp.id}"' in html
        assert f'value="40"' in html

    def test_edit_form_shows_percentages(self, writer_client, harvest_fixtures, sample_op):
        resp = writer_client.get(f'/api/prelievi/form/{sample_op.id}/')
        html = resp.json()[HTML]
        # The first species has 100% on sample_op
        assert 'value="100"' in html

    def _vdp_value(self, html):
        """Extract the value rendered into the VDP (record1) input."""
        m = re.search(rf'name="{FIELD_RECORD1}"\s+value="([^"]*)"', html)
        assert m, 'VDP input not found in form HTML'
        return m.group(1)

    def test_add_form_defaults_vdp_to_max_plus_one(
            self, writer_client, harvest_fixtures, sample_op):
        """A fresh add form pre-fills VDP with max(existing VDP) + 1."""
        # sample_op carries record1=999, so the next VDP defaults to 1000.
        html = writer_client.get('/api/prelievi/form/').json()[HTML]
        assert self._vdp_value(html) == '1000'

    def test_add_form_defaults_vdp_to_one_when_no_records(
            self, writer_client, harvest_fixtures):
        """With no existing VDP values, the add form defaults VDP to 1."""
        html = writer_client.get('/api/prelievi/form/').json()[HTML]
        assert self._vdp_value(html) == '1'

    def test_edit_form_does_not_inject_vdp_default(
            self, writer_client, harvest_fixtures):
        """Editing a legacy harvest with NULL VDP keeps the field blank,
        rather than injecting the add-form max+1 default."""
        f = harvest_fixtures
        # A separate harvest sets a non-zero max VDP (the edit form would show
        # '778' here if the default were wrongly applied to edits).
        Harvest.objects.create(
            date='2024-05-01', parcel=f['parcels'][0], crew=f['crews'][0],
            product=f['products'][0], mass_q=10, record1=777,
            harvest_plan_item=f['open_item'],
        )
        legacy = Harvest.objects.create(
            date='2024-05-02', parcel=f['parcels'][0], crew=f['crews'][0],
            product=f['products'][0], mass_q=10, record1=None,
        )
        html = writer_client.get(f'/api/prelievi/form/{legacy.id}/').json()[HTML]
        assert self._vdp_value(html) == ''


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
        primary_patch = data[PATCHES][0]
        assert primary_patch[DATA_ID] == 'prelievi'
        assert primary_patch[ROW_ID] == data[ROW_ID]
        # row_id, version, region_id, parcel_id, date...
        assert primary_patch[RECORD][4] == '2024-07-01'
        item_patch = next(p for p in data[PATCHES]
                          if p[DATA_ID] == 'harvest_plan_items')
        assert item_patch[ROW_ID] == f['open_item'].id

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
        data = resp.json()
        assert data[STATUS] == STATUS_CONFLICT
        assert RECORD in data
        assert data[PATCHES] == [{
            DATA_ID: 'prelievi', ROW_ID: sample_op.id, RECORD: data[RECORD],
        }]

    def test_validation_error_missing_date(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_DATE: ''},
        ))
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert S.ERR_DATE_REQUIRED in resp.json()[MESSAGE]

    def test_validation_error_future_date(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        future_date = (date.today() + timedelta(days=1)).isoformat()

        resp = self._post(writer_client, self._base_payload(
            f, **{FIELD_DATE: future_date},
        ))

        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert S.ERR_DATE_FUTURE in resp.json()[MESSAGE]
        assert Harvest.objects.count() == 0

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

    def test_update_allows_unchanged_historical_duplicate_vdp(
        self, writer_client, harvest_fixtures, sample_op,
    ):
        f = harvest_fixtures
        Harvest.objects.create(
            date='2024-06-16', parcel=f['parcels'][0], crew=f['crews'][0],
            product=f['products'][0], mass_q=10, record1=sample_op.record1,
            harvest_plan_item=f['open_item'],
        )

        resp = self._post(writer_client, self._base_payload(
            f, **{
                ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
                FIELD_NOTE: 'edited', FIELD_RECORD1: str(sample_op.record1),
            },
        ))

        assert resp.status_code == 200
        sample_op.refresh_from_db()
        assert sample_op.note == 'edited'
        assert sample_op.record1 == 999

    def test_update_rejects_changed_duplicate_vdp(
        self, writer_client, harvest_fixtures, sample_op,
    ):
        f = harvest_fixtures
        Harvest.objects.create(
            date='2024-06-16', parcel=f['parcels'][0], crew=f['crews'][0],
            product=f['products'][0], mass_q=10, record1=123,
            harvest_plan_item=f['open_item'],
        )

        resp = self._post(writer_client, self._base_payload(
            f, **{
                ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
                FIELD_RECORD1: '123',
            },
        ))

        assert resp.status_code == 400
        assert S.ERR_VDP_DUPLICATE.format(123) in resp.json()[MESSAGE]
        sample_op.refresh_from_db()
        assert sample_op.record1 == 999

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

    def test_validation_error_malformed_integer_fields(self, writer_client,
                                                       harvest_fixtures):
        f = harvest_fixtures
        cases = [
            (FIELD_CREW_ID, S.COL_CREW),
            (FIELD_PRODUCT_ID, S.COL_PRODUCT),
            (FIELD_RECORD1, S.COL_VDP),
            (FIELD_RECORD2, S.COL_PROT),
            (FIELD_HARVEST_PLAN_ITEM_ID, S.COL_WORKSITE),
            (ROW_ID, ROW_ID),
        ]
        for field, label in cases:
            resp = self._post(writer_client, self._base_payload(
                f, **{field: 'abc'},
            ))
            assert resp.status_code == 400, field
            assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR, field
            assert (S.ERR_BOSCO_INTEGER_REQUIRED.format(label)
                    in resp.json()[MESSAGE]), field

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
        data = resp.json()
        assert data[ROW_ID] == sample_op.id
        assert data[DELETES] == [{DATA_ID: 'prelievi', ROW_ID: sample_op.id}]
        assert len(data[PATCHES]) == 1
        assert data[PATCHES][0][DATA_ID] == 'harvest_plan_items'
        assert data[PATCHES][0][ROW_ID] == harvest_fixtures['open_item'].id
        assert not Harvest.objects.filter(id=sample_op.id).exists()

    def test_delete_cascades_junctions(self, writer_client, harvest_fixtures, sample_op):
        sp_count_before = HarvestSpecies.objects.count()
        self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
        })
        assert HarvestSpecies.objects.count() < sp_count_before

    def test_delete_marks_digest_stale(self, writer_client, harvest_fixtures, sample_op):
        for name in ('prelievi', 'harvest_plan_items'):
            DigestStatus.objects.update_or_create(name=name, defaults={'stale': False})
        self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
        })
        assert DigestStatus.objects.get(name='prelievi').stale is True
        assert DigestStatus.objects.get(name='harvest_plan_items').stale is True

    def test_delete_conflict(self, writer_client, harvest_fixtures, sample_op):
        resp = self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: '999',
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data[STATUS] == STATUS_CONFLICT
        assert data[PATCHES] == [{
            DATA_ID: 'prelievi', ROW_ID: sample_op.id, RECORD: data[RECORD],
        }]
        assert Harvest.objects.filter(id=sample_op.id).exists()

    def test_delete_non_numeric_version_conflicts(self, writer_client, harvest_fixtures, sample_op):
        resp = self._post(writer_client, {
            ROW_ID: str(sample_op.id), VERSION: 'not-a-number',
        })
        assert resp.status_code == 400
        data = resp.json()
        assert data[STATUS] == STATUS_CONFLICT
        assert data[PATCHES] == [{
            DATA_ID: 'prelievi', ROW_ID: sample_op.id, RECORD: data[RECORD],
        }]
        assert Harvest.objects.filter(id=sample_op.id).exists()

    def test_delete_not_found(self, writer_client, harvest_fixtures):
        resp = self._post(writer_client, {ROW_ID: '99999', VERSION: '1'})
        assert resp.status_code == 404

    def test_delete_non_object_json_returns_validation_error(
            self, writer_client, harvest_fixtures):
        resp = writer_client.post(
            '/api/prelievi/delete/',
            data='[]',
            content_type='application/json',
        )
        assert resp.status_code == 400
        assert resp.json()[STATUS] == STATUS_VALIDATION_ERROR
        assert resp.json()[MESSAGE] == S.ERR_JSON_INVALID

    def test_reader_forbidden(self, reader_client, harvest_fixtures, sample_op):
        resp = self._post(reader_client, {
            ROW_ID: str(sample_op.id), VERSION: str(sample_op.version),
        })
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# CSV import endpoint
# ---------------------------------------------------------------------------

class TestCsvImportView:
    def _post(self, client, data):
        return client.post(
            '/api/prelievi/import-csv/',
            data=json.dumps(data),
            content_type='application/json',
        )

    def _csv_b64(self, text):
        raw = text.encode('utf-8-sig')
        return base64.b64encode(raw).decode('ascii')

    def _csv_text(self, f):
        tractor = f['tractors'][0]
        tractor.name = 'Fiat 110-90'
        tractor.save(update_fields=['name'])
        species_name = f['species'][0].common_name
        header = (
            f'{S.CSV_COL_REGION},{S.CSV_COL_PARCEL},{S.CSV_COL_DATA},'
            f'{S.CSV_COL_CREW},{S.CSV_COL_PRODUCT},{S.CSV_COL_QUINTALS},'
            f'{S.CSV_COL_VDP},{S.CSV_COL_PROT},'
            f'{S.CSV_COL_HARVEST_DAMAGED},{S.CSV_COL_HARVEST_UNHEALTHY},'
            f'{S.CSV_COL_HARVEST_PSR},{S.CSV_COL_EXTRA_NOTE},'
            f'{S.CSV_COL_SPECIES_PREFIX}{species_name},'
            f'{S.CSV_COL_TRACTOR_PREFIX}{tractor.name}'
        )
        row = (
            f'{f["parcels"][0].region.name},{f["parcels"][0].name},2024-08-20,'
            f'{f["crews"][0].name},{f["products"][0].name},25,123,,'
            f'false,false,false,,100,100'
        )
        return f'{header}\n{row}\n'

    def test_import_success(self, writer_client, harvest_fixtures):
        resp = self._post(writer_client, {
            FIELD_FILE: self._csv_b64(self._csv_text(harvest_fixtures)),
            FIELD_NONCE: 'prelievi-csv-1',
        })
        assert resp.status_code == 200
        op = Harvest.objects.get(record1=123)
        assert op.harvest_plan_item_id is None
        assert op.parcel == harvest_fixtures['parcels'][0]
        assert float(op.mass_q) == 25.0
        assert HarvestSpecies.objects.filter(harvest=op, percent=100).exists()
        assert HarvestTractor.objects.filter(harvest=op, percent=100).exists()
        assert DigestStatus.objects.get(name='prelievi').stale is True

    def test_missing_file_validation_error(self, writer_client, harvest_fixtures):
        resp = self._post(writer_client, {})
        assert resp.status_code == 400
        data = resp.json()
        assert data[STATUS] == STATUS_VALIDATION_ERROR
        assert S.ERR_CSV_FILE_REQUIRED in data[FIELD_ERRORS]

    def test_reader_forbidden(self, reader_client, harvest_fixtures):
        resp = self._post(reader_client, {
            FIELD_FILE: self._csv_b64(self._csv_text(harvest_fixtures)),
        })
        assert resp.status_code == 403


# Import S for assertion comparisons
from config import strings as S  # noqa: E402
from config.constants import (
    COLUMNS, DATA_ID, FIELD_ACTIVE, FIELD_COMMON_NAME, FIELD_CREW_ID,
    FIELD_DATE, FIELD_DENSITY, FIELD_ERRORS, FIELD_FILE,
    FIELD_HARVEST_PLAN_ITEM_ID, FIELD_MANUFACTURER, FIELD_MASS_Q,
    FIELD_MINOR, FIELD_MODEL, FIELD_NONCE, FIELD_NOTE, FIELD_PRODUCT_ID,
    FIELD_RECORD1, FIELD_RECORD2,
    DELETES, HTML, MESSAGE, PATCHES, RECORD, ROWS, ROW_ID,
    STATUS, STATUS_CONFLICT, STATUS_VALIDATION_ERROR, VERSION,
)


def _read_gzip_json(resp):
    return json.loads(gzip.decompress(resp.getvalue()))


class TestDigestInvalidation:
    """Regression tests: harvest writes must update the materialized
    volume_actual_m3 on the linked HarvestPlanItem and mark the
    harvest_plan_items digest stale so it regenerates correctly."""

    @staticmethod
    def _items_volume_actual(client, item_id, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        resp = client.get('/api/piano-di-taglio/items/data/')
        d = _read_gzip_json(resp)
        row = next(r for r in d[ROWS]
                   if r[d[COLUMNS].index(ROW_ID)] == item_id)
        return row[d[COLUMNS].index(S.COL_VOLUME_ACTUAL)]

    def _post_harvest(self, client, f, **overrides):
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
        return client.post(
            '/api/prelievi/save/',
            data=json.dumps(payload),
            content_type='application/json',
        )

    def test_harvest_save_updates_volume_actual(
        self, writer_client, harvest_fixtures, tmp_path, settings,
    ):
        f = harvest_fixtures
        vol_before = self._items_volume_actual(
            writer_client, f['open_item'].id, tmp_path, settings,
        )
        self._post_harvest(writer_client, f, **{FIELD_MASS_Q: '20'})
        vol_after = self._items_volume_actual(
            writer_client, f['open_item'].id, tmp_path, settings,
        )
        assert vol_after > vol_before, (
            f'harvest_plan_items.volume_actual_m3 should increase after '
            f'harvest save (was {vol_before}, now {vol_after})'
        )

    def test_harvest_delete_updates_volume_actual(
        self, writer_client, harvest_fixtures, tmp_path, settings,
    ):
        f = harvest_fixtures
        resp = self._post_harvest(writer_client, f, **{FIELD_MASS_Q: '20'})
        assert resp.status_code == 200
        row_id = resp.json()[ROW_ID]
        version = Harvest.objects.get(id=row_id).version

        vol_before = self._items_volume_actual(
            writer_client, f['open_item'].id, tmp_path, settings,
        )
        assert vol_before > 0

        writer_client.post(
            '/api/prelievi/delete/',
            data=json.dumps({ROW_ID: str(row_id), VERSION: str(version)}),
            content_type='application/json',
        )
        vol_after = self._items_volume_actual(
            writer_client, f['open_item'].id, tmp_path, settings,
        )
        assert vol_after < vol_before, (
            f'harvest_plan_items.volume_actual_m3 should decrease after '
            f'harvest delete (was {vol_before}, now {vol_after})'
        )

    @staticmethod
    def _prelievi_columns(client, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        return _read_gzip_json(client.get('/api/prelievi/data/'))[COLUMNS]

    def test_species_minor_toggle_invalidates_prelievi(
        self, writer_client, harvest_fixtures, tmp_path, settings,
    ):
        """Flagging a species minor changes the prelievi column set, so the
        species write must mark the prelievi digest stale — without needing
        a subsequent harvest write to do it."""
        f = harvest_fixtures
        castagno = next(s for s in f['species'] if s.common_name == 'Castagno')

        cols = self._prelievi_columns(writer_client, tmp_path, settings)
        assert 'Castagno' in cols, 'major species should have its own column'

        resp = writer_client.post(
            '/api/impostazioni/species/save/',
            data=json.dumps({
                ROW_ID: str(castagno.id), VERSION: str(castagno.version),
                FIELD_COMMON_NAME: castagno.common_name, FIELD_DENSITY: '9.0',
                FIELD_ACTIVE: 'true', FIELD_MINOR: 'true',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 200, resp.content

        cols = self._prelievi_columns(writer_client, tmp_path, settings)
        assert 'Castagno' not in cols, (
            'species minor toggle did not invalidate the prelievi digest'
        )

    def test_tractor_rename_invalidates_prelievi(
        self, writer_client, harvest_fixtures, tmp_path, settings,
    ):
        """Tractor labels are prelievi columns, so a tractor rename must
        mark the prelievi digest stale."""
        f = harvest_fixtures
        fiat = next(t for t in f['tractors'] if t.manufacturer == 'Fiat')

        cols = self._prelievi_columns(writer_client, tmp_path, settings)
        assert 'Fiat 110-90' in cols

        resp = writer_client.post(
            '/api/impostazioni/tractors/save/',
            data=json.dumps({
                ROW_ID: str(fiat.id), VERSION: str(fiat.version),
                FIELD_MANUFACTURER: 'Fiat', FIELD_MODEL: '110-90X',
                FIELD_ACTIVE: 'true',
            }),
            content_type='application/json',
        )
        assert resp.status_code == 200, resp.content

        cols = self._prelievi_columns(writer_client, tmp_path, settings)
        assert 'Fiat 110-90X' in cols and 'Fiat 110-90' not in cols, (
            'tractor rename did not invalidate the prelievi digest'
        )
