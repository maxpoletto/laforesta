"""
Tests for acc.py forest analysis module.

Test categories:
(a) Aggregation consistency - total vs per-particella breakdown
(b) Cross-query consistency - @@tsv totals match @@tcd sums, @@tsv/@@tpt consistency
(c) Correct scaling with different sample areas
(d) Edge cases
(f) Confidence interval sanity
(g) Small tree (D <= 20cm) exclusion
(h) Harvest calculation (volume and basal area rules)

Test data (in test/data/):
Mature parcels (age >= 60, use volume rules):
- Parcel A: 10 ha, 1 sample area, age=60  -> sampled_frac = 0.0125, scale = 80
- Parcel B: 10 ha, 2 sample areas, age=70 -> sampled_frac = 0.025,  scale = 40
- Parcel C: 10 ha, 5 sample areas, age=80 -> sampled_frac = 0.0625, scale = 16

Young parcels (age < 60, use basal area rules):
- Parcel D: 10 ha, 1 sample area, age=20  -> PP_max = 15% (age range 0-29)
- Parcel E: 10 ha, 1 sample area, age=45  -> PP_max = 20% (age range 30-59)

Species: Faggio, Cerro (both have 2-coefficient Tabacchi equations)
Each parcel includes both small trees (D <= 20cm) and mature trees (D > 20cm).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import acc
from acc import (COL_COMPRESA, COL_DIAMETER_CM, COL_GENERE, COL_HEIGHT_M,
                 COL_PARTICELLA, COL_V_M3)

# Fixtures are defined in conftest.py


# =============================================================================
# (a) AGGREGATION CONSISTENCY
# =============================================================================

class TestAggregationConsistency:
    """Test that aggregated totals match sum of per-particella breakdowns."""

    def test_tsv_volume_aggregation(self, data_all):
        """@@tsv total volume should equal sum of per-particella volumes."""
        # Total (no per_particella breakdown)
        df_total = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        total_volume = df_total['volume'].sum()

        # Per-particella breakdown
        df_per_parcel = acc.calculate_tsv_table(
            data_all, group_cols=[COL_PARTICELLA], calc_margin=False, calc_total=True
        )
        sum_per_parcel = df_per_parcel['volume'].sum()

        assert np.isclose(total_volume, sum_per_parcel, rtol=1e-9), \
            f"Total {total_volume} != sum of per-parcel {sum_per_parcel}"

    def test_tsv_tree_count_aggregation(self, data_all):
        """@@tsv total tree count should equal sum of per-particella counts."""
        df_total = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        total_trees = df_total['n_trees'].sum()

        df_per_parcel = acc.calculate_tsv_table(
            data_all, group_cols=[COL_PARTICELLA], calc_margin=False, calc_total=True
        )
        sum_per_parcel = df_per_parcel['n_trees'].sum()

        assert np.isclose(total_trees, sum_per_parcel, rtol=1e-9), \
            f"Total {total_trees} != sum of per-parcel {sum_per_parcel}"

    def test_tsv_volume_by_species_aggregation(self, data_all):
        """@@tsv per-genere volumes should sum to total."""
        df_total = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        total_volume = df_total['volume'].sum()

        df_per_species = acc.calculate_tsv_table(
            data_all, group_cols=[COL_GENERE], calc_margin=False, calc_total=True
        )
        sum_per_species = df_per_species['volume'].sum()

        assert np.isclose(total_volume, sum_per_species, rtol=1e-9), \
            f"Total {total_volume} != sum by species {sum_per_species}"

    def test_tsv_volume_parcel_species_aggregation(self, data_all):
        """@@tsv per-particella-per-genere should sum to total."""
        df_total = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        total_volume = df_total['volume'].sum()

        df_detailed = acc.calculate_tsv_table(
            data_all, group_cols=[COL_PARTICELLA, COL_GENERE],
            calc_margin=False, calc_total=True
        )
        sum_detailed = df_detailed['volume'].sum()

        assert np.isclose(total_volume, sum_detailed, rtol=1e-9), \
            f"Total {total_volume} != sum of detailed {sum_detailed}"

    def test_tip_incremento_corrente_aggregation(self, data_all):
        """@@tip incremento_corrente should sum correctly across breakdowns.

        Note: ip_medio (percentage) does NOT sum - only incremento_corrente does.
        """
        # Total across all
        df_total = acc.calculate_ip_table(data_all, group_cols=[], stime_totali=True)
        total_ic = df_total['incremento_corrente'].sum()

        # Per-particella breakdown
        df_per_parcel = acc.calculate_ip_table(
            data_all, group_cols=[COL_PARTICELLA], stime_totali=True
        )
        sum_per_parcel = df_per_parcel['incremento_corrente'].sum()

        assert np.isclose(total_ic, sum_per_parcel, rtol=1e-6), \
            f"Total IC {total_ic} != sum of per-parcel IC {sum_per_parcel}"


# =============================================================================
# (b) CROSS-QUERY CONSISTENCY
# =============================================================================

class TestCrossQueryConsistency:
    """Test that different directives report consistent data."""

    def test_tsv_tcd_volume_consistency(self, data_all):
        """@@tsv total volume should match sum of @@tcd volume buckets."""
        # @@tsv total
        df_tsv = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        tsv_volume = df_tsv['volume'].sum()

        # @@tcd volume_tot (fine buckets, summed)
        tcd_df = acc.calculate_cd_data(
            data_all, metrica='volume_tot', stime_totali=True, fine=True
        )
        tcd_volume = tcd_df.sum().sum()

        assert np.isclose(tsv_volume, tcd_volume, rtol=1e-9), \
            f"@@tsv volume {tsv_volume} != @@tcd volume {tcd_volume}"

    def test_tsv_tcd_tree_count_consistency(self, data_all):
        """@@tsv tree count should match sum of @@tcd tree count buckets."""
        df_tsv = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        tsv_trees = df_tsv['n_trees'].sum()

        tcd_df = acc.calculate_cd_data(
            data_all, metrica='alberi_tot', stime_totali=True, fine=True
        )
        tcd_trees = tcd_df.sum().sum()

        assert np.isclose(tsv_trees, tcd_trees, rtol=1e-9), \
            f"@@tsv trees {tsv_trees} != @@tcd trees {tcd_trees}"

    def test_tcd_fine_coarse_consistency(self, data_all):
        """@@tcd fine buckets should sum to coarse buckets."""
        # Fine buckets (5, 10, 15, ...)
        fine_df = acc.calculate_cd_data(
            data_all, metrica='volume_tot', stime_totali=True, fine=True
        )

        # Coarse buckets (1-30, 31-50, 50+)
        coarse_df = acc.calculate_cd_data(
            data_all, metrica='volume_tot', stime_totali=True, fine=False
        )

        # Sum fine buckets into coarse ranges
        fine_total = fine_df.sum().sum()
        coarse_total = coarse_df.sum().sum()

        assert np.isclose(fine_total, coarse_total, rtol=1e-9), \
            f"Fine total {fine_total} != coarse total {coarse_total}"

    def test_tcd_basal_area_consistency(self, data_all):
        """@@tcd basal area computed two ways should match."""
        # Via tcd
        tcd_df = acc.calculate_cd_data(
            data_all, metrica='G_tot', stime_totali=True, fine=True
        )
        tcd_basal = tcd_df.sum().sum()

        # Manual calculation from trees
        trees = data_all.trees
        parcels = data_all.parcels
        manual_basal = 0.0
        for (region, parcel), ptrees in trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
            sf = parcels[(region, parcel)].sampled_frac
            basal = np.pi / 4 * ptrees[COL_DIAMETER_CM] ** 2 / 10000  # m²
            manual_basal += basal.sum() / sf

        assert np.isclose(tcd_basal, manual_basal, rtol=1e-9), \
            f"@@tcd G {tcd_basal} != manual G {manual_basal}"

    def test_tsv_tpt_volume_consistency(self, data_all, harvest_rules):
        """@@tsv and @@tpt should report the same total volume."""
        df_tsv = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True,
            calc_mature=True
        )
        df_tpt = acc.calculate_tpt_table(data_all, harvest_rules, group_cols=[])

        tsv_vol = df_tsv['volume'].sum()
        tpt_vol = df_tpt['volume'].sum()

        assert np.isclose(tsv_vol, tpt_vol, rtol=1e-9), \
            f"@@tsv volume {tsv_vol} != @@tpt volume {tpt_vol}"

    def test_tsv_tpt_volume_mature_consitency(self, data_all, harvest_rules):
        """@@tsv and @@tpt should report the same volume_mature."""
        df_tsv = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True,
            calc_mature=True
        )
        df_tpt = acc.calculate_tpt_table(data_all, harvest_rules, group_cols=[])

        tsv_vol_ss = df_tsv['volume_mature'].sum()
        tpt_vol_ss = df_tpt['volume_mature'].sum()

        assert np.isclose(tsv_vol_ss, tpt_vol_ss, rtol=1e-9), \
            f"@@tsv vol_ss {tsv_vol_ss} != @@tpt vol_ss {tpt_vol_ss}"


# =============================================================================
# (c) CORRECT SCALING WITH DIFFERENT SAMPLE AREAS
# =============================================================================

class TestSampleAreaScaling:
    """Test that trees scale correctly based on sample area density.

    Test data:
    - Parcel A: 10 ha, 1 sample area  -> sampled_frac = 0.0125, scale = 80
    - Parcel B: 10 ha, 2 sample areas -> sampled_frac = 0.025,  scale = 40
    - Parcel C: 10 ha, 5 sample areas -> sampled_frac = 0.0625, scale = 16
    - Parcel D: 10 ha, 1 sample area  -> sampled_frac = 0.0125, scale = 80
    - Parcel E: 10 ha, 1 sample area  -> sampled_frac = 0.0125, scale = 80
    """

    def test_sampled_frac_computation(self, data_all):
        """Verify sampled_frac is computed correctly."""
        parcels = data_all.parcels

        # Parcel A: 1 sample area, 10 ha -> 1 * 0.125 / 10 = 0.0125
        assert np.isclose(parcels[('Test', 'A')].sampled_frac, 0.0125), \
            f"Parcel A sampled_frac wrong: {parcels[('Test', 'A')].sampled_frac}"

        # Parcel B: 2 sample areas, 10 ha -> 2 * 0.125 / 10 = 0.025
        assert np.isclose(parcels[('Test', 'B')].sampled_frac, 0.025), \
            f"Parcel B sampled_frac wrong: {parcels[('Test', 'B')].sampled_frac}"

        # Parcel C: 5 sample areas, 10 ha -> 5 * 0.125 / 10 = 0.0625
        assert np.isclose(parcels[('Test', 'C')].sampled_frac, 0.0625), \
            f"Parcel C sampled_frac wrong: {parcels[('Test', 'C')].sampled_frac}"

    def test_tree_scaling_parcel_a(self, data_parcel_a):
        """Parcel A: 6 sampled trees should scale to 6 * 80 = 480 estimated trees."""
        df = acc.calculate_tsv_table(
            data_parcel_a, group_cols=[], calc_margin=False, calc_total=True
        )
        # 6 trees / 0.0125 = 480
        expected_trees = 6 / 0.0125
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Parcel A trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_tree_scaling_parcel_b(self, data_parcel_b):
        """Parcel B: 8 sampled trees should scale to 8 * 40 = 320 estimated trees."""
        df = acc.calculate_tsv_table(
            data_parcel_b, group_cols=[], calc_margin=False, calc_total=True
        )
        # 8 trees / 0.025 = 320
        expected_trees = 8 / 0.025
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Parcel B trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_tree_scaling_parcel_c(self, data_parcel_c):
        """Parcel C: 12 sampled trees should scale to 12 * 16 = 192 estimated trees."""
        df = acc.calculate_tsv_table(
            data_parcel_c, group_cols=[], calc_margin=False, calc_total=True
        )
        # 12 trees / 0.0625 = 192
        expected_trees = 12 / 0.0625
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Parcel C trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_total_scaled_trees(self, data_all):
        """Total scaled trees across all parcels."""
        df = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        # A: 6/0.0125=480, B: 8/0.025=320, C: 12/0.0625=192, D: 6/0.0125=480, E: 6/0.0125=480
        expected_trees = 480 + 320 + 192 + 480 + 480
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Total trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_volume_scaling_consistency(
            self, data_all, data_parcel_a, data_parcel_b, data_parcel_c,
            data_parcel_d, data_parcel_e):
        """Sum of per-parcel volumes should equal total volume."""
        df_all = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        df_a = acc.calculate_tsv_table(
            data_parcel_a, group_cols=[], calc_margin=False, calc_total=True
        )
        df_b = acc.calculate_tsv_table(
            data_parcel_b, group_cols=[], calc_margin=False, calc_total=True
        )
        df_c = acc.calculate_tsv_table(
            data_parcel_c, group_cols=[], calc_margin=False, calc_total=True
        )
        df_d = acc.calculate_tsv_table(
            data_parcel_d, group_cols=[], calc_margin=False, calc_total=True
        )
        df_e = acc.calculate_tsv_table(
            data_parcel_e, group_cols=[], calc_margin=False, calc_total=True
        )

        total = df_all['volume'].sum()
        sum_parts = (df_a['volume'].sum() + df_b['volume'].sum() + df_c['volume'].sum() +
                     df_d['volume'].sum() + df_e['volume'].sum())

        assert np.isclose(total, sum_parts, rtol=1e-9), \
            f"Total volume {total} != sum of parts {sum_parts}"

    def test_per_ha_values(self, data_all):
        """Per-hectare values should be total divided by total area (50 ha)."""
        # Total volume
        tcd_tot = acc.calculate_cd_data(
            data_all, metrica='volume_tot', stime_totali=True, fine=True
        )
        total_volume = tcd_tot.sum().sum()

        # Per-hectare volume
        tcd_ha = acc.calculate_cd_data(
            data_all, metrica='volume_ha', stime_totali=True, fine=True
        )
        per_ha_volume = tcd_ha.sum().sum()

        # Total area = 10 + 10 + 10 + 10 + 10 = 50 ha
        expected_per_ha = total_volume / 50.0

        assert np.isclose(per_ha_volume, expected_per_ha, rtol=1e-9), \
            f"Per-ha volume {per_ha_volume} != expected {expected_per_ha}"


# =============================================================================
# (d) EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_parcel_single_species(self, trees_df, particelle_df):
        """Test with single parcel and single species."""
        data = acc.parcel_data(
            ["alberi.csv"], trees_df, particelle_df,
            regions=["Test"], parcels=["A"], species=["Faggio"]
        )

        df = acc.calculate_tsv_table(data, group_cols=[], calc_margin=False, calc_total=True)
        assert df['n_trees'].sum() > 0
        assert df['volume'].sum() > 0

    def test_empty_filter_raises(self, trees_df, particelle_df):
        """Empty filter results should raise ValueError."""
        with pytest.raises(ValueError, match="Nessun dato trovato"):
            acc.parcel_data(
                ["alberi.csv"], trees_df, particelle_df,
                regions=["Test"], parcels=["A"], species=["NonExistent"]
            )

    def test_particella_without_compresa_raises(self, trees_df, particelle_df):
        """Specifying particella without compresa should raise."""
        with pytest.raises(ValueError, match="compresa"):
            acc.parcel_data(
                ["alberi.csv"], trees_df, particelle_df,
                regions=[], parcels=["A"], species=[]
            )

    def test_zero_values_in_diameter_classes(self, data_parcel_a):
        """Species with no trees in a diameter class should have zero."""
        tcd_df = acc.calculate_cd_data(
            data_parcel_a, metrica='alberi_tot', stime_totali=True, fine=True
        )
        # All values should be >= 0 (no NaN or negative)
        assert (tcd_df >= 0).all().all()
        # Some values should be zero (not all diameter classes have trees)
        assert (tcd_df == 0).any().any()


# =============================================================================
# (f) CONFIDENCE INTERVAL SANITY
# =============================================================================

class TestConfidenceInterval:
    """Test confidence interval calculations."""

    def test_margin_is_positive(self, data_all):
        """Confidence interval margin should be positive."""
        trees = data_all.trees
        _, margin = acc.calculate_volume_confidence_interval(trees)
        assert margin > 0, f"Margin should be positive, got {margin}"

    def test_margin_increases_with_volume(self, trees_df, particelle_df):
        """Larger samples should have larger absolute margins."""
        # Single parcel
        data_a = acc.parcel_data(
            ["alberi.csv"], trees_df, particelle_df,
            regions=["Test"], parcels=["A"], species=[]
        )
        _, margin_a = acc.calculate_volume_confidence_interval(data_a.trees)

        # All parcels (more trees)
        data_all = acc.parcel_data(
            ["alberi.csv"], trees_df, particelle_df,
            regions=["Test"], parcels=[], species=[]
        )
        _, margin_all = acc.calculate_volume_confidence_interval(data_all.trees)

        # More trees -> larger absolute margin (not necessarily proportionally)
        assert margin_all > margin_a, \
            f"Margin for all parcels ({margin_all}) should exceed single parcel ({margin_a})"

    def test_ci_bounds_bracket_volume(self, data_all):
        """Confidence interval should bracket the point estimate."""
        df = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=True, calc_total=True
        )
        row = df.iloc[0]

        # vol_lo < volume < vol_hi
        assert row['vol_lo'] < row['volume'] < row['vol_hi'], \
            f"CI [{row['vol_lo']}, {row['vol_hi']}] should bracket {row['volume']}"

    def test_ci_per_species(self, data_all):
        """Each species should have valid CI bounds."""
        df = acc.calculate_tsv_table(
            data_all, group_cols=[COL_GENERE], calc_margin=True, calc_total=True
        )

        for _, row in df.iterrows():
            assert row['vol_lo'] < row['volume'] < row['vol_hi'], \
                f"Species {row[COL_GENERE]}: CI [{row['vol_lo']}, {row['vol_hi']}] " \
                f"should bracket {row['volume']}"


# =============================================================================
# (g) SMALL TREES (D <= 20cm EXCLUSION)
# =============================================================================

class TestMature:
    """Test that trees with D <= 20cm are correctly excluded from harvest calculations."""

    def test_volume_mature_excludes_small_trees(self, data_all):
        """volume_mature should exclude trees with D <= 20cm."""
        df = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True,
            calc_mature=True
        )

        total_vol = df['volume'].sum()
        vol_ss = df['volume_mature'].sum()

        # volume_mature should be less than total (we have small trees)
        assert vol_ss < total_vol, \
            f"Vol small ({vol_ss}) should be less than total ({total_vol})"

    def test_small_trees_count(self, data_all):
        """Verify the number of small trees in test data."""
        trees = data_all.trees
        n_small = (trees[COL_DIAMETER_CM] <= acc.MATURE_THRESHOLD).sum()
        n_mature = (trees[COL_DIAMETER_CM] > acc.MATURE_THRESHOLD).sum()

        # Each parcel (A,B,C,D,E) has 2 small trees: 10 total
        # A: 4 mature, B: 6 mature, C: 10 mature, D: 4 mature, E: 4 mature = 28 total
        assert n_small == 10, f"Expected 10 small trees, got {n_small}"
        assert n_mature == 28, f"Expected 28 mature trees, got {n_mature}"

    def test_volume_mature_consistency(self, data_all):
        """Manual calculation of volume_mature should match."""
        trees = data_all.trees
        parcels = data_all.parcels

        # Manual calculation
        manual_vol_mature = 0.0
        for (region, parcel), ptrees in trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
            sf = parcels[(region, parcel)].sampled_frac
            above = ptrees[ptrees[COL_DIAMETER_CM] > acc.MATURE_THRESHOLD]
            manual_vol_mature += above[COL_V_M3].sum() / sf

        # Via calculate_tsv_table
        df = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True,
            calc_mature=True
        )
        computed_vol_mature = df['volume_mature'].sum()

        assert np.isclose(manual_vol_mature, computed_vol_mature, rtol=1e-9), \
            f"Manual vol_mature {manual_vol_mature} != computed {computed_vol_mature}"


# =============================================================================
# (h) HARVEST CALCULATION
# =============================================================================

class TestHarvestCalculation:
    """Test harvest (prelievo) calculations."""

    def test_volume_harvest_excludes_sottomisura(self, data_all, harvest_rules):
        """Volume-based harvest should use volume_mature."""
        df = acc.calculate_tpt_table(data_all, harvest_rules, group_cols=[])

        vol_mature = df['volume_mature'].sum()
        harvest = df['harvest'].sum()

        assert harvest <= vol_mature, \
            f"Harvest ({harvest}) should not exceed vol_mature ({vol_mature})"

    def test_harvest_per_parcel(self, data_all, harvest_rules):
        """Harvest totals should equal sum of per-parcel harvests."""
        df_total = acc.calculate_tpt_table(data_all, harvest_rules, group_cols=[])
        df_parcels = acc.calculate_tpt_table(
            data_all, harvest_rules, group_cols=[COL_PARTICELLA]
        )

        total_harvest = df_total['harvest'].sum()
        sum_parcels = df_parcels['harvest'].sum()

        assert np.isclose(total_harvest, sum_parcels, rtol=1e-9), \
            f"Total harvest {total_harvest} != sum of parcels {sum_parcels}"

    def test_basal_area_harvest_parcel_d(self, data_parcel_d, harvest_rules):
        """Parcel D (age=20) should use 15% basal area harvest rule."""
        df = acc.calculate_tpt_table(data_parcel_d, harvest_rules, group_cols=[])

        assert df['harvest'].sum() > 0, "Should have some harvest"

        vol_mature = df['volume_mature'].sum()
        harvest = df['harvest'].sum()
        assert harvest < vol_mature, "Harvest should be less than total mature volume"

    def test_basal_area_harvest_parcel_e(self, data_parcel_e, harvest_rules):
        """Parcel E (age=45) should use 20% basal area harvest rule."""
        df = acc.calculate_tpt_table(data_parcel_e, harvest_rules, group_cols=[])

        assert df['harvest'].sum() > 0, "Should have some harvest"

        vol_mature = df['volume_mature'].sum()
        harvest = df['harvest'].sum()
        assert harvest < vol_mature, "Harvest should be less than total mature volume"

    def test_small_trees_excluded_from_basal_area_harvest(
            self, data_parcel_d, harvest_rules):
        """Small trees (D <= 20) should be excluded from basal area harvest calculation."""
        df = acc.calculate_tpt_table(data_parcel_d, harvest_rules, group_cols=[])

        trees = data_parcel_d.trees
        small_trees = trees[trees[COL_DIAMETER_CM] <= acc.MATURE_THRESHOLD]
        mature_trees = trees[trees[COL_DIAMETER_CM] > acc.MATURE_THRESHOLD]

        assert len(small_trees) == 2, "Should have 2 small trees"
        assert len(mature_trees) == 4, "Should have 4 mature trees"

        sf = data_parcel_d.parcels[('Test', 'D')].sampled_frac
        expected_vol_mature = mature_trees[COL_V_M3].sum() / sf
        actual_vol_mature = df['volume_mature'].sum()

        assert np.isclose(actual_vol_mature, expected_vol_mature, rtol=1e-9), \
            f"volume_mature {actual_vol_mature} != expected {expected_vol_mature}"


class TestPrelievoMassimo:
    """Test the max_harvest harvest rules function directly."""

    def test_ceduo_returns_zero(self):
        """Ceduo compartments should return (0, 0)."""
        from harvest_rules import max_harvest
        vol_limit, area_limit = max_harvest('F', 80, 400.0, 30.0)
        assert vol_limit == 0.0
        assert area_limit == 0.0

    def test_unknown_comparto_raises(self):
        """Unknown comparto should raise ValueError."""
        from harvest_rules import max_harvest
        with pytest.raises(ValueError, match="Comparto sconosciuto"):
            max_harvest('Z', 60, 300.0, 20.0)

    def test_volume_rules_high_stock(self):
        """Age >= 60, high stock -> 25% volume limit."""
        from harvest_rules import max_harvest
        # Comparto A: provv_min=350, volume=700 m3/ha
        # 700 > 180% * 350 = 630 -> pp_max = 25%
        vol_limit, area_limit = max_harvest('A', 80, 700.0, 40.0)
        assert np.isclose(vol_limit, 700.0 * 25 / 100)
        assert area_limit == math.inf

    def test_volume_rules_low_stock(self):
        """Age >= 60, low stock -> 0% (below all thresholds)."""
        from harvest_rules import max_harvest
        # Comparto A: provv_min=350, volume=100 m3/ha
        # 100 <= 120% * 350 = 420 -> pp_max = 0
        vol_limit, area_limit = max_harvest('A', 80, 100.0, 10.0)
        assert vol_limit == 0.0
        assert area_limit == math.inf

    def test_basal_area_rules_young(self):
        """Age 0-29 -> 15% of basal area."""
        from harvest_rules import max_harvest
        vol_limit, area_limit = max_harvest('A', 20, 300.0, 25.0)
        assert vol_limit == math.inf
        assert np.isclose(area_limit, 25.0 * 15 / 100)

    def test_basal_area_rules_middle(self):
        """Age 30-59 -> 20% of basal area."""
        from harvest_rules import max_harvest
        vol_limit, area_limit = max_harvest('A', 45, 300.0, 25.0)
        assert vol_limit == math.inf
        assert np.isclose(area_limit, 25.0 * 20 / 100)


class TestComputeHarvest:
    """Test the unified compute_harvest function."""

    def test_empty_harvestable(self):
        """No mature trees -> (0, 0)."""
        import pandas as pd
        trees = pd.DataFrame({
            acc.COL_DIAMETER_CM: [10.0, 15.0],
            acc.COL_V_M3: [0.1, 0.2],
        })
        vol_mature, harvest = acc.compute_harvest(trees, 0.0125, 1000, 1000)
        assert vol_mature == 0.0
        assert harvest == 0.0

    def test_volume_limit_stops_harvest(self):
        """Volume limit should stop harvest before all trees taken."""
        import pandas as pd
        trees = pd.DataFrame({
            acc.COL_DIAMETER_CM: [25.0, 30.0, 40.0, 50.0],
            acc.COL_V_M3: [0.3, 0.5, 1.0, 2.0],
        })
        sf = 1.0  # No scaling for simplicity
        # Volume limit = 1.0 -> should take D=25 (0.3) and D=30 (0.5) = 0.8, skip D=40 (0.3+0.5+1.0=1.8 > 1.0)
        vol_mature, harvest = acc.compute_harvest(trees, sf, 1.0, math.inf)
        assert np.isclose(vol_mature, 3.8)  # 0.3+0.5+1.0+2.0
        assert np.isclose(harvest, 0.8)     # 0.3+0.5

    def test_area_limit_stops_harvest(self):
        """Basal area limit should stop harvest before all trees taken."""
        import pandas as pd
        trees = pd.DataFrame({
            acc.COL_DIAMETER_CM: [25.0, 30.0, 40.0],
            acc.COL_V_M3: [0.3, 0.5, 1.0],
        })
        sf = 1.0
        # G for D=25: pi/4 * 0.25^2 = 0.04909 m2
        # G for D=30: pi/4 * 0.30^2 = 0.07069 m2
        # area_limit = 0.06 -> take only D=25
        vol_mature, harvest = acc.compute_harvest(trees, sf, math.inf, 0.06)
        assert np.isclose(harvest, 0.3)


# =============================================================================
# VOLUME CALCULATION SPOT CHECKS
# =============================================================================

class TestVolumeCalculation:
    """Basic sanity checks on volume calculations."""

    def test_volumes_are_positive(self, trees_df):
        """All calculated volumes should be positive."""
        assert (trees_df[COL_V_M3] > 0).all(), "All volumes should be positive"

    def test_volume_increases_with_size(self, trees_df):
        """Larger trees (D*h) should generally have larger volumes."""
        # Group by species to compare within species
        for genere, group in trees_df.groupby(COL_GENERE):
            if len(group) < 2:
                continue
            # Sort by D²*h
            group = group.copy()
            group['d2h'] = group[COL_DIAMETER_CM] ** 2 * group[COL_HEIGHT_M]
            group = group.sort_values('d2h')

            # Volumes should be monotonically increasing (within species)
            volumes = group[COL_V_M3].values
            assert all(volumes[i] <= volumes[i+1] for i in range(len(volumes)-1)), \
                f"Volumes for {genere} should increase with tree size"

    def test_faggio_volume_formula(self):
        """Spot check Faggio volume formula: V = (0.81151 + 0.038965 * D² * h) / 1000."""
        # D=30, h=20 -> V = (0.81151 + 0.038965 * 900 * 20) / 1000 = 0.702 m³
        df = pd.DataFrame({COL_DIAMETER_CM: [30.0], COL_HEIGHT_M: [20.0], COL_GENERE: ['Faggio']})
        result = acc.calculate_all_trees_volume(df)
        expected = (0.81151 + 0.038965 * 900 * 20) / 1000
        assert np.isclose(result[COL_V_M3].iloc[0], expected, rtol=1e-6), \
            f"Faggio volume {result[COL_V_M3].iloc[0]} != expected {expected}"

    def test_cerro_volume_formula(self):
        """Spot check Cerro volume formula: V = (-0.043221 + 0.038079 * D² * h) / 1000."""
        # D=30, h=20 -> V = (-0.043221 + 0.038079 * 900 * 20) / 1000 = 0.685 m³
        df = pd.DataFrame({COL_DIAMETER_CM: [30.0], COL_HEIGHT_M: [20.0], COL_GENERE: ['Cerro']})
        result = acc.calculate_all_trees_volume(df)
        expected = (-0.043221 + 0.038079 * 900 * 20) / 1000
        assert np.isclose(result[COL_V_M3].iloc[0], expected, rtol=1e-6), \
            f"Cerro volume {result[COL_V_M3].iloc[0]} != expected {expected}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
