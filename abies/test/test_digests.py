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
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S
from config.constants import (
    COLUMNS, ROWS, ROW_ID,
)


# ---------------------------------------------------------------------------
# _write_gzip_json
# ---------------------------------------------------------------------------

class TestWriteGzipJSON:
    def test_creates_valid_gzip_json(self, tmp_path):
        dest = tmp_path / 'test.json.gz'
        data = {COLUMNS: ['a', 'b'], ROWS: [[1, 2]]}
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
                                          tractors, products):
        mark_stale('prelievi')
        regenerate_if_stale('prelievi')
        ds = DigestStatus.objects.get(name='prelievi')
        assert ds.stale is False

    def test_regenerate_creates_file(self, db, parcels, crews, species,
                                     tractors, products):
        mark_stale('prelievi')
        path = regenerate_if_stale('prelievi')
        assert path.exists()
        assert path.suffix == '.gz'

    def test_concurrent_mark_during_regeneration_survives(
        self, db, settings, tmp_path, monkeypatch,
    ):
        """A staleness mark that lands *while* a digest is generating must
        not be cleared by that regeneration.

        Otherwise the on-disk digest reflects the pre-write data yet reads
        as fresh forever, and the only recovery is deleting the file by
        hand.  This is the intermittent empty/stale prelievi digest: a
        regeneration whose data snapshot fell inside a destructive
        re-import window wrote an empty file, then swallowed the import's
        own staleness mark.
        """
        from apps.base import digests

        settings.DIGEST_DIR = tmp_path
        name = 'prelievi'
        mark_stale(name)

        def gen_with_concurrent_write():
            # The generator reads its snapshot and writes the file; a
            # concurrent writer then commits and re-marks the digest
            # stale, between this write and the post-generation clear.
            digests._write_gzip_json(
                {COLUMNS: [ROW_ID], ROWS: []}, digests._dest(name))
            mark_stale(name)

        monkeypatch.setitem(digests._GENERATORS, name, gen_with_concurrent_write)
        regenerate_if_stale(name)

        assert DigestStatus.objects.get(name=name).stale is True, (
            'regeneration cleared a staleness mark raised during generation'
        )


# ---------------------------------------------------------------------------
# generate_prelievi
# ---------------------------------------------------------------------------

class TestGeneratePrelievi:
    @pytest.fixture
    def harvest_data(self, parcels, crews, products, species, tractors):
        op = Harvest.objects.create(
            date='2024-06-01', product=products[0], parcel=parcels[0],
            crew=crews[0], mass_q=Decimal('200'),
        )
        HarvestSpecies.objects.create(harvest=op, species=species[0], percent=60)
        HarvestSpecies.objects.create(harvest=op, species=species[1], percent=40)
        HarvestTractor.objects.create(harvest=op, tractor=tractors[0], percent=100)
        return op

    def test_output_shape(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert COLUMNS in data
        assert ROWS in data
        assert data[COLUMNS][0] == ROW_ID
        assert len(data[ROWS]) == 1

    def test_species_quintal_columns(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        cols = data[COLUMNS]
        row = data[ROWS][0]
        # Abete at 60% of 200 = 120.0
        abete_idx = cols.index('Abete')
        assert row[abete_idx] == 120.0
        # Castagno at 40% of 200 = 80.0
        castagno_idx = cols.index('Castagno')
        assert row[castagno_idx] == 80.0
        # Altro at 0%
        other_idx = cols.index('Altro')
        assert row[other_idx] == 0.0

    def test_percentage_columns_present(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        cols = data[COLUMNS]
        assert 'Abete %' in cols
        row = data[ROWS][0]
        assert row[cols.index('Abete %')] == 60

    def test_empty_table(self, db, species, tractors):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert data[ROWS] == []
        assert len(data[COLUMNS]) > 0


# ---------------------------------------------------------------------------
# generate_parcels
# ---------------------------------------------------------------------------

class TestGenerateParcels:
    def test_output(self, parcels):
        generate_parcels()
        path = settings.DIGEST_DIR / 'parcels.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert len(data[ROWS]) == 3
        assert data[COLUMNS][0] == ROW_ID


# ---------------------------------------------------------------------------
# generate_crews
# ---------------------------------------------------------------------------

class TestGenerateCrews:
    def test_output(self, crews):
        generate_crews()
        path = settings.DIGEST_DIR / 'crews.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert len(data[ROWS]) == 2
        names = [r[1] for r in data[ROWS]]
        assert 'Alfa' in names


# ---------------------------------------------------------------------------
# generate_parcel_year_production
# ---------------------------------------------------------------------------

class TestGenerateParcelYearProduction:
    def test_aggregation(self, parcels, crews, products):
        Harvest.objects.create(
            date='2024-01-10', product=products[0], parcel=parcels[0],
            crew=crews[0], mass_q=Decimal('50'),
        )
        Harvest.objects.create(
            date='2024-06-15', product=products[0], parcel=parcels[0],
            crew=crews[0], mass_q=Decimal('30'),
        )
        generate_parcel_year_production()
        path = settings.DIGEST_DIR / 'parcel_year_production.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert len(data[ROWS]) == 1  # same parcel, same year
        assert data[ROWS][0][3] == 80.0  # 50 + 30


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
        crew_rows = [r for r in data[ROWS] if 'TestCrew' in (r[6] or '')]
        assert len(crew_rows) >= 1
        assert crew_rows[0][2] == 'John Doe'


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_generates_all_digests(self, db, parcels, crews, species, tractors, products,
                                   tmp_path, settings):
        # Redirect to tmp_path so the test doesn't clobber dev digests.
        settings.DIGEST_DIR = tmp_path
        generate_all()
        for name in ('prelievi', 'parcels', 'crews', 'parcel_year_production', 'audit'):
            path = settings.DIGEST_DIR / f'{name}.json.gz'
            assert path.exists(), f'{name}.json.gz not generated'
