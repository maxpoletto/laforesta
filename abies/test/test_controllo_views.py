"""Tests for Controllo (audit) views and digest generation."""

import gzip
import json

import pytest
from django.test import Client

from apps.base.digests import generate_audit
from apps.base.models import Crew
from apps.prelievi.models import HarvestOp
from config import strings as S


@pytest.fixture
def writer_client(writer_user):
    c = Client()
    c.force_login(writer_user)
    return c


@pytest.fixture
def audit_fixtures(regions, eclasses, species, tractors, crews, optypes, notes, parcels):
    """Create reference fixtures whose history records feed the audit digest."""
    return {
        'regions': regions, 'eclasses': eclasses, 'species': species,
        'tractors': tractors, 'crews': crews, 'optypes': optypes,
        'notes': notes, 'parcels': parcels,
    }


# ---------------------------------------------------------------------------
# Data endpoint
# ---------------------------------------------------------------------------

class TestDataView:
    def test_serves_gzip_json(self, writer_client, audit_fixtures, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_audit()

        resp = writer_client.get('/abies/api/controllo/data/')
        assert resp.status_code == 200
        assert resp['Content-Type'] == 'application/json'
        assert resp['Content-Encoding'] == 'gzip'
        data = json.loads(gzip.decompress(resp.getvalue()))
        assert data['columns'][0] == 'row_id'
        assert S.COL_TIMESTAMP in data['columns']
        assert len(data['rows']) >= 1

    def test_requires_auth(self, db):
        resp = Client().get('/abies/api/controllo/data/')
        assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Audit digest generation
# ---------------------------------------------------------------------------

class TestAuditDigest:
    def test_insert_records_appear(self, audit_fixtures, tmp_path, settings):
        """Creating objects via ORM produces insert audit rows."""
        settings.DIGEST_DIR = tmp_path
        generate_audit()
        data = _load_digest(tmp_path / 'audit.json.gz')

        # audit_fixtures created crews, species, tractors — each should have
        # at least one 'Inserimento' row.
        actions = {row[4] for row in data['rows']}
        assert S.ACTION_INSERT in actions

    def test_update_shows_diff(self, audit_fixtures, tmp_path, settings):
        """Editing an object produces a row with before/after values."""
        settings.DIGEST_DIR = tmp_path
        crew = audit_fixtures['crews'][0]
        crew.name = 'Gamma'
        crew.save()

        generate_audit()
        data = _load_digest(tmp_path / 'audit.json.gz')

        update_rows = [r for r in data['rows']
                       if r[3] == S.TABLE_CREW and r[4] == S.ACTION_UPDATE]
        assert len(update_rows) >= 1
        row = update_rows[0]
        assert 'Alfa' in row[5]   # old value
        assert 'Gamma' in row[6]  # new value

    def test_delete_shows_old_values(self, audit_fixtures, tmp_path, settings):
        """Deleting an object produces a row with the old values."""
        settings.DIGEST_DIR = tmp_path
        f = audit_fixtures
        op = HarvestOp.objects.create(
            date='2024-06-15', parcel=f['parcels'][0], crew=f['crews'][0],
            optype=f['optypes'][0], quintals=50,
        )
        op.delete()

        generate_audit()
        data = _load_digest(tmp_path / 'audit.json.gz')

        delete_rows = [r for r in data['rows']
                       if r[3] == S.TABLE_HARVEST_OP and r[4] == S.ACTION_DELETE]
        assert len(delete_rows) >= 1
        assert 'Q.li: 50' in delete_rows[0][5]  # old value contains quintals

    def test_user_history_tracked(self, writer_user, tmp_path, settings):
        """User model changes appear in audit."""
        settings.DIGEST_DIR = tmp_path
        generate_audit()
        data = _load_digest(tmp_path / 'audit.json.gz')

        user_rows = [r for r in data['rows'] if r[3] == S.TABLE_USER]
        assert len(user_rows) >= 1

    def test_rows_sorted_desc(self, audit_fixtures, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_audit()
        data = _load_digest(tmp_path / 'audit.json.gz')
        timestamps = [row[1] for row in data['rows']]
        assert timestamps == sorted(timestamps, reverse=True)


def _load_digest(path):
    return json.loads(gzip.open(path, 'rt').read())
