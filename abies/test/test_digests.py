"""Tests for digest generation and staleness management."""

import gzip
import json
import pytest
from decimal import Decimal
from pathlib import Path

from django.conf import settings

from apps.base.digests import (
    generate_prelievi, generate_parcels, generate_crews,
    generate_parcel_year_production, generate_audit, generate_all,
    mark_stale, regenerate_if_stale, _write_gzip_json,
)
from apps.base.models import Crew, DigestStatus, Role, User
from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor


# ---------------------------------------------------------------------------
# _write_gzip_json
# ---------------------------------------------------------------------------

class TestWriteGzipJSON:
    def test_creates_valid_gzip_json(self, tmp_path):
        dest = tmp_path / 'test.json.gz'
        data = {'columns': ['a', 'b'], 'rows': [[1, 2]]}
        _write_gzip_json(data, dest)
        assert dest.exists()
        with gzip.open(dest, 'rt') as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_no_partial_on_error(self, tmp_path):
        dest = tmp_path / 'test.json.gz'
        # Write an initial valid file.
        _write_gzip_json({'ok': True}, dest)
        # Now write something that will fail during json.dump.
        class BadObj:
            pass
        with pytest.raises(TypeError):
            _write_gzip_json(BadObj(), dest)
        # Original file should still be intact.
        with gzip.open(dest, 'rt') as f:
            assert json.load(f) == {'ok': True}


# ---------------------------------------------------------------------------
# mark_stale / regenerate_if_stale
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_mark_stale_creates_record(self, db):
        mark_stale('prelievi')
        ds = DigestStatus.objects.get(name='prelievi')
        assert ds.stale is True

    def test_mark_stale_idempotent(self, db):
        mark_stale('prelievi')
        mark_stale('prelievi')
        assert DigestStatus.objects.filter(name='prelievi').count() == 1

    def test_regenerate_clears_stale_flag(self, db, parcels, crews, species,
                                          tractors, optypes):
        mark_stale('prelievi')
        regenerate_if_stale('prelievi')
        ds = DigestStatus.objects.get(name='prelievi')
        assert ds.stale is False

    def test_regenerate_creates_file(self, db, parcels, crews, species,
                                     tractors, optypes):
        mark_stale('prelievi')
        path = regenerate_if_stale('prelievi')
        assert path.exists()
        assert path.suffix == '.gz'


# ---------------------------------------------------------------------------
# generate_prelievi
# ---------------------------------------------------------------------------

class TestGeneratePrelievi:
    @pytest.fixture
    def harvest_data(self, parcels, crews, optypes, species, tractors):
        op = HarvestOp.objects.create(
            date='2024-06-01', optype=optypes[0], parcel=parcels[0],
            crew=crews[0], quintals=Decimal('200'),
        )
        HarvestSpecies.objects.create(harvest_op=op, species=species[0], percent=60)
        HarvestSpecies.objects.create(harvest_op=op, species=species[1], percent=40)
        HarvestTractor.objects.create(harvest_op=op, tractor=tractors[0], percent=100)
        return op

    def test_output_shape(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert 'columns' in data
        assert 'rows' in data
        assert data['columns'][0] == 'row_id'
        assert len(data['rows']) == 1

    def test_species_quintal_columns(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        cols = data['columns']
        row = data['rows'][0]
        # Abete at 60% of 200 = 120.0
        abete_idx = cols.index('Abete')
        assert row[abete_idx] == 120.0
        # Castagno at 40% of 200 = 80.0
        castagno_idx = cols.index('Castagno')
        assert row[castagno_idx] == 80.0
        # Altro at 0%
        altro_idx = cols.index('Altro')
        assert row[altro_idx] == 0.0

    def test_percentage_columns_present(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        cols = data['columns']
        assert 'Abete %' in cols
        row = data['rows'][0]
        assert row[cols.index('Abete %')] == 60

    def test_empty_table(self, db, species, tractors):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert data['rows'] == []
        assert len(data['columns']) > 0


# ---------------------------------------------------------------------------
# generate_parcels
# ---------------------------------------------------------------------------

class TestGenerateParcels:
    def test_output(self, parcels):
        generate_parcels()
        path = settings.DIGEST_DIR / 'parcels.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert len(data['rows']) == 3
        assert data['columns'][0] == 'row_id'


# ---------------------------------------------------------------------------
# generate_crews
# ---------------------------------------------------------------------------

class TestGenerateCrews:
    def test_output(self, crews):
        generate_crews()
        path = settings.DIGEST_DIR / 'crews.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert len(data['rows']) == 2
        names = [r[1] for r in data['rows']]
        assert 'Alfa' in names


# ---------------------------------------------------------------------------
# generate_parcel_year_production
# ---------------------------------------------------------------------------

class TestGenerateParcelYearProduction:
    def test_aggregation(self, parcels, crews, optypes):
        HarvestOp.objects.create(
            date='2024-01-10', optype=optypes[0], parcel=parcels[0],
            crew=crews[0], quintals=Decimal('50'),
        )
        HarvestOp.objects.create(
            date='2024-06-15', optype=optypes[0], parcel=parcels[0],
            crew=crews[0], quintals=Decimal('30'),
        )
        generate_parcel_year_production()
        path = settings.DIGEST_DIR / 'parcel_year_production.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert len(data['rows']) == 1  # same parcel, same year
        assert data['rows'][0][3] == 80.0  # 50 + 30


# ---------------------------------------------------------------------------
# generate_audit — user display name
# ---------------------------------------------------------------------------

class TestGenerateAudit:
    def test_user_with_name(self, db):
        user = User.objects.create_user(
            username='jdoe', password='testpass123!',
            first_name='John', last_name='Doe', role=Role.WRITER,
        )
        c = Crew(name='TestCrew', active=True)
        c._history_user = user
        c.save()

        generate_audit()
        path = settings.DIGEST_DIR / 'audit.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        # Find the crew insert row — user column should show "John Doe"
        crew_rows = [r for r in data['rows'] if 'TestCrew' in (r[6] or '')]
        assert len(crew_rows) >= 1
        assert crew_rows[0][2] == 'John Doe'


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_generates_all_digests(self, db, parcels, crews, species, tractors, optypes,
                                   tmp_path, settings):
        # Redirect to tmp_path so the test doesn't clobber dev digests.
        settings.DIGEST_DIR = tmp_path
        generate_all()
        for name in ('prelievi', 'parcels', 'crews', 'parcel_year_production', 'audit'):
            path = settings.DIGEST_DIR / f'{name}.json.gz'
            assert path.exists(), f'{name}.json.gz not generated'
