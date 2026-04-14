"""Tests for Prelievi API views: data, form, save, delete."""

import gzip
import json

import pytest
from django.test import Client

from apps.base.digests import generate_prelievi, mark_stale
from apps.base.models import DigestStatus
from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor


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
def harvest_fixtures(regions, eclasses, species, tractors, crews, optypes, notes, parcels):
    """Return a dict of all reference fixtures needed for harvest operations."""
    return {
        'regions': regions, 'eclasses': eclasses, 'species': species,
        'tractors': tractors, 'crews': crews, 'optypes': optypes,
        'notes': notes, 'parcels': parcels,
    }


@pytest.fixture
def sample_op(harvest_fixtures):
    """A saved HarvestOp for edit/delete tests."""
    f = harvest_fixtures
    op = HarvestOp.objects.create(
        date='2024-06-15', parcel=f['parcels'][0], crew=f['crews'][0],
        optype=f['optypes'][0], quintals=50, record1=999,
    )
    HarvestSpecies.objects.create(harvest_op=op, species=f['species'][0], percent=100)
    HarvestTractor.objects.create(harvest_op=op, tractor=f['tractors'][0], percent=100)
    return op


# ---------------------------------------------------------------------------
# Data endpoint
# ---------------------------------------------------------------------------

class TestDataView:
    def test_serves_gzip_json(self, writer_client, harvest_fixtures, sample_op, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()

        resp = writer_client.get('/abies/api/prelievi/data/')
        assert resp.status_code == 200
        assert resp['Content-Type'] == 'application/json'
        assert resp['Content-Encoding'] == 'gzip'
        data = json.loads(gzip.decompress(resp.getvalue()))
        assert 'columns' in data
        assert len(data['rows']) >= 1

    def test_304_on_not_modified(self, writer_client, harvest_fixtures, sample_op, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()

        resp1 = writer_client.get('/abies/api/prelievi/data/')
        lm = resp1['Last-Modified']

        resp2 = writer_client.get('/abies/api/prelievi/data/', HTTP_IF_MODIFIED_SINCE=lm)
        assert resp2.status_code == 304

    def test_requires_auth(self, db, harvest_fixtures):
        resp = Client().get('/abies/api/prelievi/data/')
        assert resp.status_code == 302  # redirect to login


# ---------------------------------------------------------------------------
# Form endpoint
# ---------------------------------------------------------------------------

class TestFormView:
    def test_add_form_returns_html(self, writer_client, harvest_fixtures):
        resp = writer_client.get('/abies/api/prelievi/form/')
        assert resp.status_code == 200
        data = resp.json()
        assert '<form' in data['html']
        assert 'id_date' in data['html']

    def test_edit_form_prepopulated(self, writer_client, harvest_fixtures, sample_op):
        resp = writer_client.get(f'/abies/api/prelievi/form/{sample_op.id}/')
        assert resp.status_code == 200
        html = resp.json()['html']
        assert '2024-06-15' in html
        assert 'selected' in html

    def test_form_contains_species_and_tractors(self, writer_client, harvest_fixtures):
        resp = writer_client.get('/abies/api/prelievi/form/')
        html = resp.json()['html']
        assert 'sp_' in html
        assert 'tr_' in html
        assert '100%' in html

    def test_edit_form_shows_percentages(self, writer_client, harvest_fixtures, sample_op):
        resp = writer_client.get(f'/abies/api/prelievi/form/{sample_op.id}/')
        html = resp.json()['html']
        # The first species has 100% on sample_op
        assert 'value="100"' in html


# ---------------------------------------------------------------------------
# Save endpoint
# ---------------------------------------------------------------------------

class TestSaveView:
    def _post(self, client, data):
        return client.post(
            '/abies/api/prelievi/save/',
            data=json.dumps(data),
            content_type='application/json',
        )

    def test_create_success(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '30', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': 'test note',
            f'sp_{f["species"][0].id}': '60',
            f'sp_{f["species"][1].id}': '40',
            f'tr_{f["tractors"][0].id}': '100',
            'nonce': 'create-nonce-1',
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data['data_id'] == 'prelievi'
        assert data['row_id'] > 0
        assert data['record'][2] == '2024-07-01'  # date is third column (after row_id, version)

        # Verify DB
        op = HarvestOp.objects.get(id=data['row_id'])
        assert float(op.quintals) == 30.0
        assert op.extra_note == 'test note'
        assert HarvestSpecies.objects.filter(harvest_op=op).count() == 2
        assert HarvestTractor.objects.filter(harvest_op=op).count() == 1

    def test_create_marks_digest_stale(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
            f'sp_{f["species"][0].id}': '100',
            f'tr_{f["tractors"][0].id}': '100',
        })
        assert DigestStatus.objects.get(name='prelievi').stale is True
        assert DigestStatus.objects.get(name='parcel_year_production').stale is True

    def test_update_success(self, writer_client, harvest_fixtures, sample_op):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'row_id': str(sample_op.id), 'version': str(sample_op.version),
            'date': '2024-06-20', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '60', 'note_id': '', 'record1': '999', 'record2': '',
            'extra_note': '',
            f'sp_{f["species"][0].id}': '100',
            f'tr_{f["tractors"][0].id}': '100',
        })
        assert resp.status_code == 200
        sample_op.refresh_from_db()
        assert str(sample_op.date) == '2024-06-20'
        assert float(sample_op.quintals) == 60.0
        assert sample_op.version == 2

    def test_update_conflict(self, writer_client, harvest_fixtures, sample_op):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'row_id': str(sample_op.id), 'version': '999',  # wrong version
            'date': '2024-06-20', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '60', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'conflict'
        assert 'record' in resp.json()

    def test_validation_error_missing_date(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'validation_error'
        assert S.ERR_DATE_REQUIRED in resp.json()['message']

    def test_validation_error_species_sum(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
            f'sp_{f["species"][0].id}': '50',  # doesn't sum to 100
        })
        assert resp.status_code == 400
        assert 'specie' in resp.json()['message'].lower()

    def test_vdp_duplicate(self, writer_client, harvest_fixtures, sample_op):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '999', 'record2': '',
            'extra_note': '',
        })
        assert resp.status_code == 400
        assert 'VDP' in resp.json()['message']

    def test_validation_error_bad_quintals(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '-5', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
        })
        assert resp.status_code == 400
        assert S.ERR_QUINTALS_POSITIVE in resp.json()['message']

    def test_validation_error_non_numeric_quintals(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': 'abc', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
        })
        assert resp.status_code == 400
        assert S.ERR_QUINTALS_POSITIVE in resp.json()['message']

    def test_validation_error_tractor_sum(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(writer_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
            f'sp_{f["species"][0].id}': '100',
            f'tr_{f["tractors"][0].id}': '50',  # doesn't sum to 100
        })
        assert resp.status_code == 400
        assert 'trattori' in resp.json()['message'].lower()

    def test_reader_forbidden(self, reader_client, harvest_fixtures):
        f = harvest_fixtures
        resp = self._post(reader_client, {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '',
        })
        assert resp.status_code == 403

    def test_nonce_idempotency(self, writer_client, harvest_fixtures):
        f = harvest_fixtures
        payload = {
            'date': '2024-07-01', 'parcel_id': str(f['parcels'][0].id),
            'crew_id': str(f['crews'][0].id), 'optype_id': str(f['optypes'][0].id),
            'quintals': '10', 'note_id': '', 'record1': '', 'record2': '',
            'extra_note': '', 'nonce': 'idempotent-1',
            f'sp_{f["species"][0].id}': '100',
            f'tr_{f["tractors"][0].id}': '100',
        }
        resp1 = self._post(writer_client, payload)
        assert resp1.status_code == 200
        row_id_1 = resp1.json()['row_id']

        # Replay with same nonce — should get same response, no new row
        resp2 = self._post(writer_client, payload)
        assert resp2.status_code == 200
        assert resp2.json()['row_id'] == row_id_1
        assert HarvestOp.objects.count() == 1


# ---------------------------------------------------------------------------
# Delete endpoint
# ---------------------------------------------------------------------------

class TestDeleteView:
    def _post(self, client, data):
        return client.post(
            '/abies/api/prelievi/delete/',
            data=json.dumps(data),
            content_type='application/json',
        )

    def test_delete_success(self, writer_client, harvest_fixtures, sample_op):
        resp = self._post(writer_client, {
            'row_id': str(sample_op.id), 'version': str(sample_op.version),
            'nonce': 'del-1',
        })
        assert resp.status_code == 200
        assert resp.json()['row_id'] == sample_op.id
        assert not HarvestOp.objects.filter(id=sample_op.id).exists()

    def test_delete_cascades_junctions(self, writer_client, harvest_fixtures, sample_op):
        sp_count_before = HarvestSpecies.objects.count()
        self._post(writer_client, {
            'row_id': str(sample_op.id), 'version': str(sample_op.version),
        })
        assert HarvestSpecies.objects.count() < sp_count_before

    def test_delete_conflict(self, writer_client, harvest_fixtures, sample_op):
        resp = self._post(writer_client, {
            'row_id': str(sample_op.id), 'version': '999',
        })
        assert resp.status_code == 400
        assert resp.json()['status'] == 'conflict'
        assert HarvestOp.objects.filter(id=sample_op.id).exists()

    def test_delete_not_found(self, writer_client, harvest_fixtures):
        resp = self._post(writer_client, {'row_id': '99999', 'version': '1'})
        assert resp.status_code == 404

    def test_reader_forbidden(self, reader_client, harvest_fixtures, sample_op):
        resp = self._post(reader_client, {
            'row_id': str(sample_op.id), 'version': str(sample_op.version),
        })
        assert resp.status_code == 403


# Import S for assertion comparisons
from config import strings as S  # noqa: E402
