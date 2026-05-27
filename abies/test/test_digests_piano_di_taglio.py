"""Tests for the Piano di taglio digests (PT-30..35).

Covers:
- generate_harvest_plans / build_harvest_plan_record (PT-30)
- generate_harvest_plan_items / build_harvest_plan_item_record (PT-31)
- generate_tree_height_regressions / build_tree_height_regression_record (PT-32)
- generate_mark_trees_for_item / build_tree_mark_record (PT-33)
- generate_prelievi Cantiere column (PT-34)
- build_<digest>_record == generator row shape contract (PT-35)

Tests that depend on a write view marking digests stale (the full
write→stale→regenerate path in B5) land in Phase 4 alongside the views.
"""

import gzip
import json
from decimal import Decimal

import pytest

from apps.base.digests import (
    _resolve_generator,
    build_harvest_plan_item_record,
    build_harvest_plan_record,
    build_harvest_record,
    build_tree_height_regression_record,
    build_tree_mark_record,
    generate_harvest_plan_items,
    generate_harvest_plans,
    generate_mark_trees_for_item,
    generate_prelievi,
    generate_tree_height_regressions,
)
from apps.base.models import (
    Eclass, HarvestPlan, HarvestPlanItem, HarvestPlanItemState,
    Parcel, Tree, TreeHeightRegression, TreeMark,
)
from apps.prelievi.models import Harvest
from config import strings as S
from config.constants import COLUMNS, ROWS, ROW_ID, VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plan(db):
    return HarvestPlan.objects.create(
        name='Piano 2020-2030', description='Test plan.',
        year_start=2020, year_end=2030,
    )


@pytest.fixture
def coppice_eclass(db):
    return Eclass.objects.create(name='C', coppice=True)


@pytest.fixture
def coppice_parcel(regions, coppice_eclass):
    return Parcel.objects.create(
        name='42', region=regions[0], eclass=coppice_eclass,
        area_ha=Decimal('3.5'),
    )


@pytest.fixture
def fustaia_item(plan, parcels):
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=parcels[0], year_planned=2025,
        volume_planned_m3=Decimal('100.0'),
        volume_marked_m3=Decimal('95.5'),
        volume_actual_m3=Decimal('80.0'),
        state=HarvestPlanItemState.HARVESTING,
        damaged=False, unhealthy=False, psr=False,
        note='regular cut',
    )


@pytest.fixture
def ceduo_item(plan, coppice_parcel):
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, parcel=coppice_parcel, year_planned=2026,
        intervention_area_ha=Decimal('1.2'),
        state=HarvestPlanItemState.PLANNED,
        damaged=False, unhealthy=False, psr=False,
    )


@pytest.fixture
def region_wide_item(plan, regions):
    return HarvestPlanItem.objects.create(
        harvest_plan=plan, region=regions[0], year_planned=2025,
        volume_planned_m3=Decimal('30.0'),
        state=HarvestPlanItemState.OPEN,
        damaged=True, unhealthy=False, psr=False,
        note='post-storm cleanup',
    )


def _load(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PT-30: harvest_plans
# ---------------------------------------------------------------------------

class TestGenerateHarvestPlans:
    def test_output_shape(self, plan, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plans()
        data = _load(tmp_path / 'harvest_plans.json.gz')

        assert data[COLUMNS][0] == ROW_ID
        assert data[COLUMNS][1] == VERSION
        assert S.COL_NAME in data[COLUMNS]
        assert S.COL_YEAR_START in data[COLUMNS]
        assert S.COL_YEAR_END in data[COLUMNS]
        assert len(data[ROWS]) == 1

    def test_row_values(self, plan, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plans()
        data = _load(tmp_path / 'harvest_plans.json.gz')

        cols = data[COLUMNS]
        row = data[ROWS][0]
        assert row[cols.index(ROW_ID)] == plan.id
        assert row[cols.index(VERSION)] == plan.version
        assert row[cols.index(S.COL_NAME)] == 'Piano 2020-2030'
        assert row[cols.index(S.COL_DESCRIPTION)] == 'Test plan.'
        assert row[cols.index(S.COL_YEAR_START)] == 2020
        assert row[cols.index(S.COL_YEAR_END)] == 2030

    def test_build_record_matches_generator(self, plan, tmp_path, settings):
        """Shape contract: build_harvest_plan_record produces a row
        identical to what generate_harvest_plans writes for the same plan.
        """
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plans()
        data = _load(tmp_path / 'harvest_plans.json.gz')
        row_from_gen = data[ROWS][0]
        row_from_build = build_harvest_plan_record(plan)
        assert row_from_build == row_from_gen

    def test_sorted_by_year_start_desc(self, db, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        HarvestPlan.objects.create(name='Old', year_start=2010, year_end=2020)
        HarvestPlan.objects.create(name='New', year_start=2025, year_end=2035)
        HarvestPlan.objects.create(name='Mid', year_start=2020, year_end=2030)
        generate_harvest_plans()
        data = _load(tmp_path / 'harvest_plans.json.gz')
        names = [r[data[COLUMNS].index(S.COL_NAME)] for r in data[ROWS]]
        assert names == ['New', 'Mid', 'Old']


# ---------------------------------------------------------------------------
# PT-31: harvest_plan_items
# ---------------------------------------------------------------------------

class TestGenerateHarvestPlanItems:
    def test_fustaia_row(self, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plan_items()
        data = _load(tmp_path / 'harvest_plan_items.json.gz')

        cols = data[COLUMNS]
        row = next(r for r in data[ROWS] if r[cols.index(ROW_ID)] == fustaia_item.id)
        assert row[cols.index(S.COL_HARVEST_PLAN)] == fustaia_item.harvest_plan_id
        assert row[cols.index(S.COL_YEAR_PLANNED)] == 2025
        assert row[cols.index(S.COL_COMPRESA)] == 'Capistrano'
        assert row[cols.index(S.COL_PARCEL)] == '1'
        assert row[cols.index(S.COL_TYPE)] == S.TYPE_FUSTAIA
        assert row[cols.index(S.COL_STATE)] == S.STATE_HARVESTING
        assert row[cols.index(S.COL_VOLUME_PLANNED)] == 100.0
        assert row[cols.index(S.COL_VOLUME_MARKED)] == 95.5
        assert row[cols.index(S.COL_VOLUME_ACTUAL)] == 80.0
        assert row[cols.index(S.COL_PARCEL_AREA_HA)] == 10.5
        assert row[cols.index(S.COL_NOTE)] == ''  # no flags set
        assert row[cols.index(S.COL_EXTRA_NOTE)] == 'regular cut'

    def test_ceduo_row(self, ceduo_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plan_items()
        data = _load(tmp_path / 'harvest_plan_items.json.gz')

        cols = data[COLUMNS]
        row = next(r for r in data[ROWS] if r[cols.index(ROW_ID)] == ceduo_item.id)
        assert row[cols.index(S.COL_TYPE)] == S.TYPE_CEDUO
        assert row[cols.index(S.COL_INTERVENTION_AREA_HA)] == 1.2
        assert row[cols.index(S.COL_STATE)] == S.STATE_PLANNED
        # Coppice: volume_planned_m3 is NULL → renders as ''.
        assert row[cols.index(S.COL_VOLUME_PLANNED)] == ''
        assert row[cols.index(S.COL_VOLUME_MARKED)] == ''
        assert row[cols.index(S.COL_VOLUME_ACTUAL)] == 0.0

    def test_region_wide_row(self, region_wide_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plan_items()
        data = _load(tmp_path / 'harvest_plan_items.json.gz')

        cols = data[COLUMNS]
        row = next(r for r in data[ROWS] if r[cols.index(ROW_ID)] == region_wide_item.id)
        assert row[cols.index(S.COL_COMPRESA)] == 'Capistrano'
        assert row[cols.index(S.COL_PARCEL)] == S.PARCEL_WHOLE_REGION_MARK
        # Region-wide items have no Eclass → Tipo is empty.
        assert row[cols.index(S.COL_TYPE)] == ''
        # Parcel-area cross-check column is empty.
        assert row[cols.index(S.COL_PARCEL_AREA_HA)] == ''
        # Flag rendering: damaged=True only → S.FLAG_DAMAGED.
        assert row[cols.index(S.COL_NOTE)] == S.FLAG_DAMAGED
        assert row[cols.index(S.COL_STATE)] == S.STATE_OPEN

    def test_multiple_flags_comma_joined(self, plan, parcels, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        item = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[0], year_planned=2025,
            damaged=True, unhealthy=True, psr=False,
        )
        generate_harvest_plan_items()
        data = _load(tmp_path / 'harvest_plan_items.json.gz')
        cols = data[COLUMNS]
        row = next(r for r in data[ROWS] if r[cols.index(ROW_ID)] == item.id)
        # render_flag_note iterates D, U, P → "Catastrofato, Fitosanitario".
        assert row[cols.index(S.COL_NOTE)] == f'{S.FLAG_DAMAGED}, {S.FLAG_UNHEALTHY}'

    def test_year_actual_blank_when_planned(self, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        # fustaia_item has date_actual = None.
        generate_harvest_plan_items()
        data = _load(tmp_path / 'harvest_plan_items.json.gz')
        cols = data[COLUMNS]
        row = next(r for r in data[ROWS] if r[cols.index(ROW_ID)] == fustaia_item.id)
        assert row[cols.index(S.COL_YEAR_ACTUAL)] == ''

    def test_build_record_matches_generator(self, fustaia_item, ceduo_item,
                                            region_wide_item, tmp_path, settings):
        """B5 shape contract — build_<digest>_record helper produces the
        same row the generator writes for the same item.
        """
        settings.DIGEST_DIR = tmp_path
        generate_harvest_plan_items()
        data = _load(tmp_path / 'harvest_plan_items.json.gz')
        for item in (fustaia_item, ceduo_item, region_wide_item):
            # Re-fetch with select_related to mirror generator's prefetch.
            full = (HarvestPlanItem.objects
                    .select_related('parcel__region', 'parcel__eclass', 'region')
                    .get(pk=item.pk))
            built = build_harvest_plan_item_record(full)
            gen_row = next(
                r for r in data[ROWS]
                if r[data[COLUMNS].index(ROW_ID)] == item.id
            )
            assert built == gen_row


# ---------------------------------------------------------------------------
# PT-32: tree_height_regressions
# ---------------------------------------------------------------------------

class TestGenerateTreeHeightRegressions:
    def test_output(self, plan, regions, species, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        thr = TreeHeightRegression.objects.create(
            harvest_plan=plan, region=regions[0], species=species[0],
            function='ln', a=Decimal('5.1234'), b=Decimal('-3.4500'),
            r2=Decimal('0.8500'), n=42,
        )
        generate_tree_height_regressions()
        data = _load(tmp_path / 'tree_height_regressions.json.gz')

        cols = data[COLUMNS]
        assert S.COL_FUNCTION in cols
        assert S.COL_A in cols
        assert S.COL_N_REGRESSION in cols
        row = data[ROWS][0]
        assert row[cols.index(ROW_ID)] == thr.id
        assert row[cols.index(S.COL_HARVEST_PLAN)] == plan.id
        assert row[cols.index(S.COL_COMPRESA)] == 'Capistrano'
        assert row[cols.index(S.COL_SPECIES)] == 'Abete'
        assert row[cols.index(S.COL_FUNCTION)] == 'ln'
        assert row[cols.index(S.COL_A)] == 5.1234
        assert row[cols.index(S.COL_B)] == -3.45
        assert row[cols.index(S.COL_N_REGRESSION)] == 42

    def test_build_record_matches_generator(self, plan, regions, species,
                                            tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        thr = TreeHeightRegression.objects.create(
            harvest_plan=plan, region=regions[1], species=species[1],
            function='ln', a=Decimal('4.0'), b=Decimal('-2.0'),
            r2=Decimal('0.9'), n=20,
        )
        generate_tree_height_regressions()
        data = _load(tmp_path / 'tree_height_regressions.json.gz')
        full = (TreeHeightRegression.objects
                .select_related('region', 'species').get(pk=thr.pk))
        assert build_tree_height_regression_record(full) == data[ROWS][0]


# ---------------------------------------------------------------------------
# PT-33: mark_trees_<id>
# ---------------------------------------------------------------------------

class TestGenerateMarkTreesForItem:
    @pytest.fixture
    def trees(self, species, parcels):
        return [
            Tree.objects.create(species=species[0], parcel=parcels[0]),
            Tree.objects.create(species=species[1], parcel=parcels[0]),
            Tree.objects.create(species=species[0], parcel=parcels[0]),
        ]

    @pytest.fixture
    def marks(self, fustaia_item, trees):
        return [
            TreeMark.objects.create(
                harvest_plan_item=fustaia_item, tree=trees[0], number=1,
                date='2025-06-15', d_cm=40, h_m=Decimal('22.5'),
                h_measured=True,
                volume_m3=Decimal('1.5'), mass_q=Decimal('13.5'),
                lat=38.4, lon=16.1, operator='Mario',
            ),
            TreeMark.objects.create(
                harvest_plan_item=fustaia_item, tree=trees[1], number=2,
                date='2025-06-10', d_cm=35, h_m=Decimal('20.0'),
                h_measured=False,
                volume_m3=Decimal('1.0'), mass_q=Decimal('9.2'),
                lat=38.41, lon=16.11, operator='Luigi',
            ),
            TreeMark.objects.create(
                harvest_plan_item=fustaia_item, tree=trees[2], number=3,
                date='2025-06-15', d_cm=50, h_m=Decimal('25.0'),
                h_measured=True,
                volume_m3=Decimal('2.2'), mass_q=Decimal('19.8'),
                lat=38.42, lon=16.12, operator='Mario',
            ),
        ]

    def test_output_columns(self, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_mark_trees_for_item(fustaia_item.id)
        data = _load(tmp_path / f'mark_trees_{fustaia_item.id}.json.gz')
        cols = data[COLUMNS]
        for c in (ROW_ID, 'version', S.COL_DATE, S.COL_NUMERO, S.COL_SPECIES,
                  S.COL_D_CM, S.COL_H_M, S.COL_H_MEASURED, S.COL_V_M3,
                  S.COL_MASS_Q, S.COL_LAT, S.COL_LON, S.COL_OPERATOR):
            assert c in cols

    def test_empty_when_no_marks(self, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_mark_trees_for_item(fustaia_item.id)
        data = _load(tmp_path / f'mark_trees_{fustaia_item.id}.json.gz')
        assert data[ROWS] == []

    def test_rows_sorted_by_number(self, marks, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_mark_trees_for_item(fustaia_item.id)
        data = _load(tmp_path / f'mark_trees_{fustaia_item.id}.json.gz')
        nums = [r[data[COLUMNS].index(S.COL_NUMERO)] for r in data[ROWS]]
        assert nums == sorted(nums)

    def test_numero_is_1_based_sequential(self, marks, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_mark_trees_for_item(fustaia_item.id)
        data = _load(tmp_path / f'mark_trees_{fustaia_item.id}.json.gz')
        numeros = [r[data[COLUMNS].index(S.COL_NUMERO)] for r in data[ROWS]]
        assert numeros == [1, 2, 3]

    def test_isolated_per_item(self, marks, plan, parcels, trees, tmp_path, settings):
        """A mark under a different plan item must not appear in this
        item's digest.
        """
        settings.DIGEST_DIR = tmp_path
        other = HarvestPlanItem.objects.create(
            harvest_plan=plan, parcel=parcels[1], year_planned=2025,
        )
        # Need a fresh tree to satisfy UNIQUE(harvest_plan_item, tree).
        from apps.base.models import Tree
        other_tree = Tree.objects.create(species=trees[0].species, parcel=parcels[1])
        TreeMark.objects.create(
            harvest_plan_item=other, tree=other_tree, number=1,
            date='2025-07-01', d_cm=30, h_m=Decimal('18.0'),
            volume_m3=Decimal('0.8'), mass_q=Decimal('7.2'),
            lat=38.5, lon=16.2, operator='X',
        )
        generate_mark_trees_for_item(marks[0].harvest_plan_item_id)
        data = _load(tmp_path / f'mark_trees_{marks[0].harvest_plan_item_id}.json.gz')
        assert len(data[ROWS]) == 3  # only the three marks from `marks`

    def test_build_record_matches_generator(self, marks, fustaia_item,
                                            tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_mark_trees_for_item(fustaia_item.id)
        data = _load(tmp_path / f'mark_trees_{fustaia_item.id}.json.gz')
        for row in data[ROWS]:
            tm_id = row[data[COLUMNS].index(ROW_ID)]
            tm = (TreeMark.objects
                  .select_related('tree__species').get(pk=tm_id))
            assert build_tree_mark_record(tm) == row


# ---------------------------------------------------------------------------
# Dynamic resolver
# ---------------------------------------------------------------------------

class TestDynamicResolver:
    def test_mark_trees_pattern_resolves(self, fustaia_item, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        gen = _resolve_generator(f'mark_trees_{fustaia_item.id}')
        assert gen is not None
        gen()
        assert (tmp_path / f'mark_trees_{fustaia_item.id}.json.gz').exists()

    def test_mark_trees_garbage_returns_none(self):
        assert _resolve_generator('mark_trees_xyz') is None

    def test_unknown_name_returns_none(self):
        assert _resolve_generator('nope') is None


# ---------------------------------------------------------------------------
# PT-34: prelievi Cantiere column
# ---------------------------------------------------------------------------

class TestPrelieviCantiere:
    @pytest.fixture
    def linked_harvest(self, fustaia_item, parcels, crews, products):
        # The Harvest must hit parcels[0] (the same one fustaia_item is
        # bound to) and damaged/unhealthy/psr must match (all False here).
        return Harvest.objects.create(
            date='2025-09-01', parcel=parcels[0], crew=crews[0],
            product=products[0], mass_q=Decimal('50'),
            harvest_plan_item=fustaia_item,
        )

    @pytest.fixture
    def unlinked_harvest(self, parcels, crews, products):
        return Harvest.objects.create(
            date='2024-09-01', parcel=parcels[1], crew=crews[0],
            product=products[0], mass_q=Decimal('30'),
        )

    def test_cantiere_column_present(self, db, species, tractors,
                                     tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()
        data = _load(tmp_path / 'prelievi.json.gz')
        assert S.COL_CANTIERE in data[COLUMNS]

    def test_cantiere_value_when_linked(self, linked_harvest, species,
                                        tractors, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()
        data = _load(tmp_path / 'prelievi.json.gz')
        cols = data[COLUMNS]
        row = next(r for r in data[ROWS]
                   if r[cols.index(ROW_ID)] == linked_harvest.id)
        assert row[cols.index(S.COL_CANTIERE)] == linked_harvest.harvest_plan_item_id

    def test_cantiere_blank_when_unlinked(self, unlinked_harvest, species,
                                          tractors, tmp_path, settings):
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()
        data = _load(tmp_path / 'prelievi.json.gz')
        cols = data[COLUMNS]
        row = next(r for r in data[ROWS]
                   if r[cols.index(ROW_ID)] == unlinked_harvest.id)
        assert row[cols.index(S.COL_CANTIERE)] == ''

    def test_build_harvest_record_matches_generator(self, linked_harvest,
                                                    species, tractors,
                                                    tmp_path, settings):
        """B5 shape contract for the existing Harvest digest."""
        settings.DIGEST_DIR = tmp_path
        generate_prelievi()
        data = _load(tmp_path / 'prelievi.json.gz')
        full = (Harvest.objects
                .select_related('parcel__region', 'crew', 'product')
                .get(pk=linked_harvest.pk))
        built = build_harvest_record(full)
        gen_row = next(r for r in data[ROWS]
                       if r[data[COLUMNS].index(ROW_ID)] == linked_harvest.id)
        assert built == gen_row
