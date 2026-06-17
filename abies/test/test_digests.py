"""Tests for digest generation and staleness management."""

import gzip
import json
import math
import pytest
from decimal import Decimal
from pathlib import Path

from django.conf import settings

from apps.base.digests import (
    aggregate_sp_pcts, annual_increment_pct, basal_area_m2,
    build_harvest_record, diameter_class_cm, generate_future_production,
    generate_parcel_dendrometry, generate_parcel_dendrometry_points,
    generate_prelievi, generate_preserved_trees, generate_parcels,
    generate_crews, generate_audit, generate_all, mark_stale,
    regenerate_if_stale, prelievi_species_cols, _write_gzip_json,
    _audit_configs, _tracked_models,
)
from apps.base.models import (
    Crew, DigestStatus, HarvestPlan, HarvestPlanItem, HypsoParamSet,
    HypsoParamSource, Parcel, Role, Sample, SampleArea, SampleGrid, Survey,
    Tree, TreeMark, TreePreserved, TreeSample, User,
)
from apps.mannesi.models import LicensePlate, ProductionCredit, WorkHour
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor
from config import strings as S
from config.constants import (
    COLUMNS, COL_COPPICE, COL_PARCEL_ID, COL_REGION_ID, COL_SPECIES_ID,
    COL_SURVEY_ID, COL_TREE_ID, DIGEST_FUTURE_PRODUCTION,
    DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES, ROWS, ROW_ID,
    VERSION,
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

    def test_includes_stable_region_and_parcel_ids(self, harvest_data):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        cols = data[COLUMNS]
        row = data[ROWS][0]
        assert row[cols.index(COL_REGION_ID)] == harvest_data.parcel.region_id
        assert row[cols.index(COL_PARCEL_ID)] == harvest_data.parcel_id

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

    def test_tractor_columns_use_canonical_name(self, harvest_data, tractors):
        tractors[0].name = 'T1'
        tractors[0].save(update_fields=['name'])
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        cols = data[COLUMNS]
        row = data[ROWS][0]
        assert 'T1' in cols
        assert 'T1 %' in cols
        assert 'Fiat 110-90' not in cols
        assert row[cols.index('T1')] == 200.0
        assert row[cols.index('T1 %')] == 100

    def test_empty_table(self, db, species, tractors):
        generate_prelievi()
        path = settings.DIGEST_DIR / 'prelievi.json.gz'
        with gzip.open(path, 'rt') as f:
            data = json.load(f)
        assert data[ROWS] == []
        assert len(data[COLUMNS]) > 0

    def test_build_record_with_precomputed_maps_matches_generator(
            self, harvest_data, species, tractors, tmp_path, settings,
            django_assert_num_queries,
    ):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()
        with gzip.open(tmp_path / 'prelievi.json.gz', 'rt') as f:
            data = json.load(f)

        full = Harvest.objects.select_related(
            'parcel__region', 'crew', 'product',
        ).get(pk=harvest_data.pk)
        species_ids, _, minor_ids, other_id = prelievi_species_cols()
        tractor_ids = [t.id for t in tractors]
        sp_pcts = aggregate_sp_pcts(
            {species[0].id: 60, species[1].id: 40}, minor_ids, other_id,
        )
        tr_pcts = {tractors[0].id: 100}

        with django_assert_num_queries(0):
            built = build_harvest_record(
                full, species_ids=species_ids, tractor_ids=tractor_ids,
                sp_pcts=sp_pcts, tr_pcts=tr_pcts,
            )
        assert built == data[ROWS][0]

    def test_generator_uses_bulk_harvest_percentage_maps(
            self, harvest_data, tmp_path, settings, django_assert_num_queries,
    ):
        settings.DIGEST_DIR = tmp_path

        with django_assert_num_queries(5):
            generate_prelievi()


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
# Bosco digests
# ---------------------------------------------------------------------------

class TestGenerateBoscoDigests:
    def _read(self, tmp_path, name):
        with gzip.open(tmp_path / f'{name}.json.gz', 'rt') as f:
            return json.load(f)

    def test_parcels_include_bosco_metadata(self, parcels, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        p = parcels[0]
        p.ave_age = 72
        p.location_name = 'Mulu'
        p.altitude_min_m = 840
        p.altitude_max_m = 895
        p.aspect = 'S-E'
        p.grade_pct = 7
        p.desc_veg = 'Descrizione vegetazione'
        p.desc_geo = 'Descrizione geologia'
        p.save()

        generate_parcels()
        data = self._read(tmp_path, 'parcels')
        cols = data[COLUMNS]
        row = next(r for r in data[ROWS] if r[0] == p.id)

        for col in (VERSION, COL_REGION_ID, COL_COPPICE, S.COL_AREA_CAD_HA,
                    S.COL_TYPE, S.COL_DESC_VEG, S.COL_DESC_GEO):
            assert col in cols
        assert row[cols.index(VERSION)] == 1
        assert row[cols.index(COL_REGION_ID)] == p.region_id
        assert row[cols.index(COL_COPPICE)] is False
        assert row[cols.index(S.COL_TYPE)] == S.TYPE_HIGHFOREST
        assert row[cols.index(S.COL_DESC_VEG)] == 'Descrizione vegetazione'
        assert row[cols.index(S.COL_DESC_GEO)] == 'Descrizione geologia'

    def test_preserved_trees_digest(self, parcels, species, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        kept = Tree.objects.create(
            species=species[0], parcel=parcels[0], preserved=True,
            estimated_birth_year=1920, lat=38.1, lon=16.2,
        )
        pai = TreePreserved.objects.create(
            tree=kept, parcel=parcels[0], number=7, date='2024-09-15',
            d_cm=42, h_m=Decimal('18.50'), h_measured=True,
            lat=38.1, lon=16.2, note='nota',
        )
        Tree.objects.create(species=species[1], parcel=parcels[0], preserved=False)

        generate_preserved_trees()
        data = self._read(tmp_path, DIGEST_PRESERVED_TREES)
        cols = data[COLUMNS]
        assert data[ROWS] == [[
            pai.id, pai.version, kept.id, parcels[0].id, species[0].id,
            parcels[0].region.name, parcels[0].name, species[0].common_name,
            7, '2024-09-15', 1920, 42, 18.5, True, 38.1, 16.2, 'nota',
        ]]
        assert cols == [
            ROW_ID, VERSION, COL_TREE_ID, COL_PARCEL_ID, COL_SPECIES_ID,
            S.COL_REGION, S.COL_PARCEL, S.COL_SPECIES, S.COL_NUMBER,
            S.COL_DATE, S.COL_ESTIMATED_BIRTH_YEAR, S.COL_D_CM, S.COL_H_M,
            S.COL_H_MEASURED, S.COL_LAT, S.COL_LON, S.COL_NOTE,
        ]

    def test_future_production_active_highforest_only(
            self, parcels, regions, eclasses, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        plan = HarvestPlan.objects.create(
            name='Piano attivo', year_start=2026, year_end=2040, active=True,
        )
        highforest = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2028,
            volume_planned_m3=Decimal('12.500'),
        )
        coppice_parcel = Parcel.objects.create(
            name='C1', region=regions[0], eclass=eclasses[2],
            area_ha=Decimal('3.00'),
        )
        HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=coppice_parcel, year_planned=2029,
            volume_planned_m3=Decimal('99.000'),
        )
        HarvestPlanItem.objects.create(
            harvest_plan=plan, region=regions[0], year_planned=2030,
            volume_planned_m3=Decimal('88.000'), damaged=True,
        )

        generate_future_production()
        data = self._read(tmp_path, DIGEST_FUTURE_PRODUCTION)
        cols = data[COLUMNS]
        assert data[ROWS] == [[
            highforest.id, highforest.version, plan.id, parcels[0].id,
            parcels[0].region.name, parcels[0].name, 2028, 12.5,
        ]]
        assert cols == [
            ROW_ID, VERSION, S.COL_HARVEST_PLAN, COL_PARCEL_ID,
            S.COL_REGION, S.COL_PARCEL, S.COL_YEAR_PLANNED,
            S.COL_VOLUME_PLANNED,
        ]

    def test_diameter_class_and_increment_helpers(self):
        assert diameter_class_cm(17) == 15
        assert diameter_class_cm(18) == 20
        assert diameter_class_cm(22) == 20
        assert diameter_class_cm(23) == 25
        assert annual_increment_pct(18, 9, 2) == 2.0
        assert annual_increment_pct(30, 15, 2) == 2.0
        assert annual_increment_pct(18, 0, 2) is None

    def test_parcel_dendrometry_uses_active_surveys(
            self, parcels, species, tmp_path, settings,
    ):
        settings.DIGEST_DIR = tmp_path
        grid = SampleGrid.objects.create(name='Bosco grid')
        area = SampleArea.objects.create(
            sample_grid=grid, parcel=parcels[0], number='1', lat=38.0, lon=16.0,
        )
        active = Survey.objects.create(name='Active survey', sample_grid=grid, active=True)
        inactive = Survey.objects.create(name='Inactive survey', sample_grid=grid)
        sample = Sample.objects.create(sample_area=area, survey=active, date='2026-06-01')
        inactive_sample = Sample.objects.create(
            sample_area=area, survey=inactive, date='2026-06-02',
        )
        tree1 = Tree.objects.create(species=species[0], parcel=parcels[0])
        tree2 = Tree.objects.create(species=species[0], parcel=parcels[0])
        tree3 = Tree.objects.create(species=species[1], parcel=parcels[0])
        ts1 = TreeSample.objects.create(
            sample=sample, tree=tree1, number=1, d_cm=18,
            h_m=Decimal('20.00'), l10_mm=9,
            volume_m3=Decimal('0.1000'), mass_q=Decimal('1.000'),
        )
        ts2 = TreeSample.objects.create(
            sample=sample, tree=tree2, number=2, d_cm=22,
            h_m=Decimal('22.00'), l10_mm=0,
            volume_m3=Decimal('0.2000'), mass_q=Decimal('2.000'),
        )
        ts3 = TreeSample.objects.create(
            sample=inactive_sample, tree=tree3, number=1, d_cm=40,
            h_m=Decimal('30.00'), l10_mm=20,
            volume_m3=Decimal('9.0000'), mass_q=Decimal('9.000'),
        )

        generate_parcel_dendrometry()
        data = self._read(tmp_path, DIGEST_PARCEL_DENDROMETRY)
        cols = data[COLUMNS]
        assert cols == [
            ROW_ID, COL_PARCEL_ID, COL_SURVEY_ID, COL_SPECIES_ID,
            S.COL_REGION, S.COL_PARCEL, S.COL_SURVEY, S.COL_SAMPLE_AREA_HA, S.COL_SPECIES,
            S.COL_DIAM_CLASS_CM, S.COL_N_TREES, S.COL_VOLUME_M3,
            S.COL_BASAL_AREA_M2, S.COL_AVG_H_M, S.COL_INCREMENT_PCT,
        ]
        assert len(data[ROWS]) == 1
        row = data[ROWS][0]
        assert row[cols.index(COL_PARCEL_ID)] == parcels[0].id
        assert row[cols.index(COL_SURVEY_ID)] == active.id
        assert row[cols.index(COL_SPECIES_ID)] == species[0].id
        assert row[cols.index(S.COL_SAMPLE_AREA_HA)] == round(math.pi * area.r_m ** 2 / 10000, 6)
        assert row[cols.index(S.COL_DIAM_CLASS_CM)] == 20
        assert row[cols.index(S.COL_N_TREES)] == 2
        assert row[cols.index(S.COL_VOLUME_M3)] == 0.3
        expected_basal = round(basal_area_m2(18) + basal_area_m2(22), 6)
        assert row[cols.index(S.COL_BASAL_AREA_M2)] == expected_basal
        assert row[cols.index(S.COL_AVG_H_M)] == 21.0
        assert row[cols.index(S.COL_INCREMENT_PCT)] == 2.0

        generate_parcel_dendrometry_points()
        points = self._read(tmp_path, DIGEST_PARCEL_DENDROMETRY_POINTS)
        assert [r[0] for r in points[ROWS]] == [ts1.id, ts2.id]
        assert points[COLUMNS] == [
            ROW_ID, COL_PARCEL_ID, COL_SURVEY_ID, COL_TREE_ID,
            COL_SPECIES_ID, S.COL_REGION, S.COL_PARCEL, S.COL_SURVEY,
            S.COL_SPECIES, S.COL_D_CM, S.COL_H_M,
        ]
        point_cols = points[COLUMNS]
        assert points[ROWS][0][point_cols.index(COL_SPECIES_ID)] == species[0].id

        hypso_set = HypsoParamSet.objects.create(
            source=HypsoParamSource.COMPUTED, min_n=5,
            use_for_height_plots=True,
        )
        hypso_set.surveys.set([inactive])

        generate_parcel_dendrometry()
        data = self._read(tmp_path, DIGEST_PARCEL_DENDROMETRY)
        assert data[ROWS][0][cols.index(COL_SURVEY_ID)] == active.id

        generate_parcel_dendrometry_points()
        points = self._read(tmp_path, DIGEST_PARCEL_DENDROMETRY_POINTS)
        assert [r[0] for r in points[ROWS]] == [ts3.id]
        assert points[ROWS][0][point_cols.index(COL_SURVEY_ID)] == inactive.id


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

    def test_covers_all_tracked_models(self, db):
        """Every history-tracked model is wired into the audit config.

        Locks the contract that broke the Controllo log: a model gains
        HistoricalRecords() but is never added to _audit_configs(), so its
        writes are silently absent from the audit.  To stop auditing a
        model, remove its HistoricalRecords() — don't drop it from here.
        """
        configured = {model for model, _, _ in _audit_configs()}
        assert _tracked_models() == configured

    def test_bulk_tree_models_not_tracked(self, db):
        """Tree / TreeSample / TreeMark are excluded from history (and thus
        the audit) so bulk CSV imports don't swamp the log."""
        tracked = _tracked_models()
        assert Tree not in tracked
        assert TreeSample not in tracked
        assert TreeMark not in tracked

    def test_parcel_metadata_update_appears(self, parcels, settings, tmp_path):
        """Parcel metadata edits are audited for the Controllo page."""
        settings.DIGEST_DIR = tmp_path
        parcel = parcels[0]
        parcel.desc_veg = 'Vegetazione audit'
        parcel.save()

        generate_audit()
        with gzip.open(tmp_path / 'audit.json.gz', 'rt') as f:
            data = json.load(f)

        rows = [r for r in data[ROWS]
                if r[3] == S.TABLE_PARCEL and r[4] == S.AUDIT_UPDATE]
        assert any(f'{S.COL_DESC_VEG}: Vegetazione audit' in (r[6] or '')
                   for r in rows)

    def test_domain_model_inserts_appear(self, db, settings, tmp_path):
        """Inserts into formerly-missing domain models surface in the audit."""
        settings.DIGEST_DIR = tmp_path
        plan = HarvestPlan.objects.create(
            name='Piano 2026', year_start=2026, year_end=2030,
        )
        grid = SampleGrid.objects.create(name='Griglia A')
        Survey.objects.create(name='Rilievo 1', sample_grid=grid)
        HypsoParamSet.objects.create(source=HypsoParamSource.COMPUTED, min_n=5)
        LicensePlate.objects.create(value='AB123CD')
        crew = Crew.objects.create(name='Mannesi audit crew')
        WorkHour.objects.create(date='2026-01-01', crew=crew, hours=Decimal('3'))
        ProductionCredit.objects.create(date='2026-01-01', crew=crew, mass_q=Decimal('4'))

        generate_audit()
        with gzip.open(tmp_path / 'audit.json.gz', 'rt') as f:
            data = json.load(f)

        tables = {row[3] for row in data[ROWS] if row[4] == S.AUDIT_INSERT}
        for table in (S.TABLE_HARVEST_PLAN, S.TABLE_SAMPLE_GRID,
                      S.TABLE_SURVEY, S.TABLE_HYPSO_PARAM_SET,
                      S.TABLE_MANNESI_LICENSE_PLATE, S.TABLE_MANNESI_HOURS,
                      S.TABLE_MANNESI_CREDIT):
            assert table in tables, f'{table} insert missing from audit'

        plan_rows = [r for r in data[ROWS] if r[3] == S.TABLE_HARVEST_PLAN]
        assert any('Piano 2026' in (r[6] or '') for r in plan_rows)


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_generates_all_digests(self, db, parcels, crews, species, tractors, products,
                                   tmp_path, settings):
        # Redirect to tmp_path so the test doesn't clobber dev digests.
        settings.DIGEST_DIR = tmp_path
        generate_all()
        for name in ('prelievi', 'parcels', 'crews', 'audit'):
            path = settings.DIGEST_DIR / f'{name}.json.gz'
            assert path.exists(), f'{name}.json.gz not generated'
