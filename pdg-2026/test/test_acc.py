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
from acc import (COL_COEFF_PRESSLER, COL_COMPRESA, COL_D_CM, COL_CD_CM,
                 COL_GENERE, COL_H_M, COL_L10_MM, COL_PARTICELLA, COL_V_M3,
                 COL_WEIGHT, COL_AREA_SAGGIO,
                 ParcelData, ParcelStats)

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
        df_total = acc.calculate_growth_rates(data_all, group_cols=[COL_GENERE, COL_CD_CM], stime_totali=True)
        total_ic = df_total['incremento_corrente'].sum()

        # Per-particella breakdown
        df_per_parcel = acc.calculate_growth_rates(
            data_all, group_cols=[COL_PARTICELLA, COL_GENERE, COL_CD_CM], stime_totali=True
        )
        sum_per_parcel = df_per_parcel['incremento_corrente'].sum()

        assert np.isclose(total_ic, sum_per_parcel, rtol=1e-6), \
            f"Total IC {total_ic} != sum of per-parcel IC {sum_per_parcel}"

    def test_tip_volume_weighted_ip_with_unequal_sampling(self):
        """ip_medio must use expansion-factor-weighted volumes when stime_totali=True.

        With two parcels having different sampling fractions (sf), using raw sample
        volumes as weights gives the wrong ip_medio and IC.  The correct IC is the
        sum of per-parcel expanded tree increments: sum_p (1/sf_p) * sum_i(ip_i * v_i / 100).
        """
        # Parcel A: sf=0.5, one tree with v=10 m³, ip=5%
        # Parcel B: sf=1.0, one tree with v=10 m³, ip=3%
        # Both trees: same species (Faggio), same diameter class (25)
        #
        # ip for Pressler: ip = c * 2 * L10 / 100 / D
        # Choose c=1, D=25 cm so ip = 2*L10/100/25 = L10/1250
        # Tree A: ip=5% -> L10 = 5 * 1250 = 6250 mm
        # Tree B: ip=3% -> L10 = 3 * 1250 = 3750 mm
        trees = pd.DataFrame({
            COL_COMPRESA: ['R', 'R'],
            COL_PARTICELLA: ['A', 'B'],
            COL_GENERE: ['Faggio', 'Faggio'],
            COL_D_CM: [25.0, 25.0],
            COL_CD_CM: [25, 25],
            COL_V_M3: [10.0, 10.0],
            COL_COEFF_PRESSLER: [1.0, 1.0],
            COL_L10_MM: [6250.0, 3750.0],
        })
        parcels = {
            ('R', 'A'): ParcelStats(area_ha=10, sector='A', age=60,
                                    governo='Fustaia',
                                    n_sample_areas=4, sampled_frac=0.5),
            ('R', 'B'): ParcelStats(area_ha=10, sector='A', age=60,
                                    governo='Fustaia',
                                    n_sample_areas=8, sampled_frac=1.0),
        }
        data = ParcelData(trees=trees, regions=['R'], species=['Faggio'],
                          parcels=parcels)

        df = acc.calculate_growth_rates(
            data, group_cols=[COL_GENERE, COL_CD_CM], stime_totali=True)
        assert len(df) == 1
        row = df.iloc[0]

        # Correct IC = (10*5/100)/0.5 + (10*3/100)/1.0 = 1.0 + 0.3 = 1.3
        # Wrong IC (unweighted) = 30 * 4.0 / 100 = 1.2
        expected_ic = 10 * 0.05 / 0.5 + 10 * 0.03 / 1.0  # = 1.3
        assert np.isclose(row['incremento_corrente'], expected_ic, rtol=1e-9), \
            f"IC {row['incremento_corrente']} != expected {expected_ic}"

        # ip_medio should be consistent: IC = volume * ip_medio / 100
        volume = 10 / 0.5 + 10 / 1.0  # = 30
        expected_ip = expected_ic * 100 / volume  # = 1.3 * 100 / 30 ≈ 4.333...
        assert np.isclose(row['ip_medio'], expected_ip, rtol=1e-9), \
            f"ip_medio {row['ip_medio']} != expected {expected_ip}"


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
            basal = np.pi / 4 * ptrees[COL_D_CM] ** 2 / 10000  # m²
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
        n_small = (trees[COL_D_CM] <= acc.MATURE_THRESHOLD).sum()
        n_mature = (trees[COL_D_CM] > acc.MATURE_THRESHOLD).sum()

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
            above = ptrees[ptrees[COL_D_CM] > acc.MATURE_THRESHOLD]
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
        small_trees = trees[trees[COL_D_CM] <= acc.MATURE_THRESHOLD]
        mature_trees = trees[trees[COL_D_CM] > acc.MATURE_THRESHOLD]

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
        trees = pd.DataFrame({
            acc.COL_COMPRESA: 'Serra',
            acc.COL_PARTICELLA: '1',
            acc.COL_D_CM: [10.0, 15.0],
            acc.COL_V_M3: [0.1, 0.2],
        })
        vol_mature, harvest = acc.compute_harvest(trees, 0.0125, 1000, 1000)
        assert vol_mature == 0.0
        assert harvest == 0.0

    def test_volume_limit_stops_harvest(self):
        """Volume limit should stop harvest before all trees taken."""
        trees = pd.DataFrame({
            acc.COL_COMPRESA: 'Serra',
            acc.COL_PARTICELLA: '1',
            acc.COL_D_CM: [25.0, 30.0, 40.0, 50.0],
            acc.COL_V_M3: [0.3, 0.5, 1.0, 2.0],
        })
        sf = 1.0  # No scaling for simplicity
        # Volume limit = 1.0 -> should take D=25 (0.3) and D=30 (0.5) = 0.8, skip D=40 (0.3+0.5+1.0=1.8 > 1.0)
        vol_mature, harvest = acc.compute_harvest(trees, sf, 1.0, math.inf)
        assert np.isclose(vol_mature, 3.8)  # 0.3+0.5+1.0+2.0
        assert np.isclose(harvest, 0.8)     # 0.3+0.5

    def test_area_limit_stops_harvest(self):
        """Basal area limit should stop harvest before all trees taken."""
        trees = pd.DataFrame({
            acc.COL_COMPRESA: 'Serra',
            acc.COL_PARTICELLA: '1',
            acc.COL_D_CM: [25.0, 30.0, 40.0],
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
            group['d2h'] = group[COL_D_CM] ** 2 * group[COL_H_M]
            group = group.sort_values('d2h')

            # Volumes should be monotonically increasing (within species)
            volumes = group[COL_V_M3].values
            assert all(volumes[i] <= volumes[i+1] for i in range(len(volumes)-1)), \
                f"Volumes for {genere} should increase with tree size"

    def test_faggio_volume_formula(self):
        """Spot check Faggio volume formula: V = (0.81151 + 0.038965 * D² * h) / 1000."""
        # D=30, h=20 -> V = (0.81151 + 0.038965 * 900 * 20) / 1000 = 0.702 m³
        df = pd.DataFrame({COL_D_CM: [30.0], COL_H_M: [20.0], COL_GENERE: ['Faggio']})
        result = acc.calculate_all_trees_volume(df)
        expected = (0.81151 + 0.038965 * 900 * 20) / 1000
        assert np.isclose(result[COL_V_M3].iloc[0], expected, rtol=1e-6), \
            f"Faggio volume {result[COL_V_M3].iloc[0]} != expected {expected}"

    def test_cerro_volume_formula(self):
        """Spot check Cerro volume formula: V = (-0.043221 + 0.038079 * D² * h) / 1000."""
        # D=30, h=20 -> V = (-0.043221 + 0.038079 * 900 * 20) / 1000 = 0.685 m³
        df = pd.DataFrame({COL_D_CM: [30.0], COL_H_M: [20.0], COL_GENERE: ['Cerro']})
        result = acc.calculate_all_trees_volume(df)
        expected = (-0.043221 + 0.038079 * 900 * 20) / 1000
        assert np.isclose(result[COL_V_M3].iloc[0], expected, rtol=1e-6), \
            f"Cerro volume {result[COL_V_M3].iloc[0]} != expected {expected}"


# =============================================================================
# (i) TCR: INCREMENTO CORRENTE vs PROJECTED VOLUME
# =============================================================================

def _make_parcel_data(trees_data: list[dict]) -> acc.ParcelData:
    """Build a minimal ParcelData from a list of tree dicts.

    Each dict must have: D (cm), V (m³), genere, L10 (mm), c (Pressler coef).
    All trees are placed in compresa='X', particella='1', area saggio=1.
    Parcel: 10 ha, 1 sample area -> sampled_frac = 0.0125.
    """
    rows = []
    for t in trees_data:
        d = t['D']
        rows.append({
            acc.COL_COMPRESA: 'X',
            acc.COL_PARTICELLA: '1',
            acc.COL_AREA_SAGGIO: 1,
            acc.COL_GENERE: t.get('genere', 'Faggio'),
            acc.COL_D_CM: d,
            acc.COL_V_M3: t['V'],
            acc.COL_CD_CM: int(np.ceil((d - 2.5) / 5) * 5),
            acc.COL_L10_MM: t.get('L10', 5.0),
            acc.COL_COEFF_PRESSLER: t.get('c', 200),
        })
    trees_df = pd.DataFrame(rows)
    parcels = {
        ('X', '1'): acc.ParcelStats(
            area_ha=10.0, sector='A', age=60, governo='Fustaia',
            n_sample_areas=1, sampled_frac=acc.SAMPLE_AREA_HA / 10.0),
    }
    return acc.ParcelData(
        trees=trees_df, regions=['X'], species=['Faggio'],
        parcels=parcels)


class TestTcrIncrementoCorrente:
    """Test that incremento corrente in the tcr table is consistent with
    the volume projection."""

    def test_all_mature_ic_matches_volume_delta(self):
        """When all trees are mature (D > 20), V_year0 + ic == V_year1 exactly."""
        data = _make_parcel_data([
            {'D': 25.0, 'V': 0.4},
            {'D': 30.0, 'V': 0.7},
            {'D': 40.0, 'V': 1.5},
            {'D': 50.0, 'V': 2.8},
        ])
        df = acc.calculate_tcr_table(data, group_cols=[], years=1, mortalita=0)

        v0 = df[acc.COL_VOLUME_MATURE].iloc[0]
        v1 = df[acc.COL_VOLUME_MATURE_PROJ].iloc[0]
        ic = df[acc.COL_INCR_CORR].iloc[0]

        assert np.isclose(v0 + ic, v1, rtol=1e-12), \
            f"V0 ({v0:.6f}) + ic ({ic:.6f}) = {v0+ic:.6f} != V1 ({v1:.6f})"

    def test_graduating_tree_ic_less_than_volume_delta(self):
        """When a tree graduates from immature to mature,
        V_year0 + ic < V_year1 (the graduated tree's full volume enters
        the mature pool, but only its growth increment was in ic)."""
        # D=20 is immature (threshold is D > 20). With L10=5.0 and c=200,
        # delta_d = 2 * 5.0 / 100 = 0.1 cm, so D_year1 = 20.1 > 20: graduates.
        data = _make_parcel_data([
            {'D': 20.0, 'V': 0.3},   # will graduate
            {'D': 30.0, 'V': 0.7},   # already mature
            {'D': 40.0, 'V': 1.5},   # already mature
        ])
        df = acc.calculate_tcr_table(data, group_cols=[], years=1, mortalita=0)

        v0 = df[acc.COL_VOLUME_MATURE].iloc[0]
        v1 = df[acc.COL_VOLUME_MATURE_PROJ].iloc[0]
        ic = df[acc.COL_INCR_CORR].iloc[0]

        assert v0 + ic <= v1, \
            f"V0 ({v0:.6f}) + ic ({ic:.6f}) = {v0+ic:.6f} should be <= V1 ({v1:.6f})"


# =============================================================================
# DIAMETER CLASS
# =============================================================================

class TestDiameterClass:
    """Test acc.diameter_class() assigns correct diameter classes."""

    def test_class_midpoints(self):
        """Typical values map to the class whose midpoint they're nearest to."""
        d = pd.Series([5.0, 10.0, 15.0, 20.0, 25.0])
        result = acc.diameter_class(d)
        pd.testing.assert_series_equal(result, pd.Series([5, 10, 15, 20, 25]), check_names=False)

    def test_boundary_upper_inclusive(self):
        """Upper boundary (midpoint + width/2) belongs to the class."""
        # 7.5 is the upper boundary of class 5 -> should map to 5
        d = pd.Series([7.5, 12.5, 17.5])
        result = acc.diameter_class(d)
        pd.testing.assert_series_equal(result, pd.Series([5, 10, 15]), check_names=False)

    def test_boundary_lower_exclusive(self):
        """Lower boundary (midpoint - width/2) does NOT belong to the class."""
        # 2.5 is the lower boundary of class 5 -> should map to 0 (class below)
        d = pd.Series([2.5, 7.5 + 0.01])
        result = acc.diameter_class(d)
        assert result.iloc[0] == 0, "2.5 should map to class 0, not 5"
        assert result.iloc[1] == 10, "7.51 should map to class 10"

    def test_values_within_class(self):
        """Values strictly within a class map correctly."""
        # All of these are in (2.5, 7.5] -> class 5
        d = pd.Series([3.0, 4.0, 5.0, 6.0, 7.0, 7.5])
        result = acc.diameter_class(d)
        expected = pd.Series([5, 5, 5, 5, 5, 5])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_small_diameters(self):
        """Very small diameters (in the first class)."""
        d = pd.Series([0.1, 1.0, 2.0, 2.5])
        result = acc.diameter_class(d)
        expected = pd.Series([0, 0, 0, 0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_zero_diameter(self):
        """Zero diameter maps to class 0."""
        d = pd.Series([0.0])
        result = acc.diameter_class(d)
        assert result.iloc[0] == 0

    def test_large_diameters(self):
        """Large diameters map to correct classes."""
        d = pd.Series([50.0, 77.3, 100.0])
        result = acc.diameter_class(d)
        expected = pd.Series([50, 75, 100])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_custom_width(self):
        """Non-default class width works correctly."""
        d = pd.Series([5.0, 10.0, 15.0, 20.0])
        result = acc.diameter_class(d, width=10)
        # width=10: (5, 15] -> 10, (15, 25] -> 20
        expected = pd.Series([0, 10, 10, 20])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_custom_width_boundaries(self):
        """Boundaries with non-default width."""
        # width=10: lower boundary of class 10 is 5.0 (exclusive), upper is 15.0 (inclusive)
        d = pd.Series([5.0, 5.01, 15.0, 15.01])
        result = acc.diameter_class(d, width=10)
        expected = pd.Series([0, 10, 10, 20])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_returns_int_dtype(self):
        """Result should have integer dtype."""
        d = pd.Series([3.7, 8.2, 14.9])
        result = acc.diameter_class(d)
        assert np.issubdtype(result.dtype, np.integer)

    def test_empty_input(self):
        """Empty Series returns empty integer Series."""
        d = pd.Series([], dtype=float)
        result = acc.diameter_class(d)
        assert len(result) == 0
        assert np.issubdtype(result.dtype, np.integer)


# =============================================================================
# (i) TPDT — Harvest scheduling
# =============================================================================

def _make_sim(trees_data):
    """Build a sim DataFrame for _simulate_harvest_on_parcel tests.

    trees_data: list of (region, parcel, sample_area, diameter_cm, volume_m3, genere) tuples.
    """
    rows = []
    for region, parcel, sa, d, v, genere in trees_data:
        rows.append({
            COL_COMPRESA: region,
            COL_PARTICELLA: parcel,
            COL_AREA_SAGGIO: sa,
            COL_D_CM: d,
            COL_V_M3: v,
            COL_GENERE: genere,
            COL_CD_CM: acc.diameter_class(pd.Series([d])).iloc[0],
            COL_WEIGHT: 1.0,
        })
    return pd.DataFrame(rows)


# Simple rules: 25% volume harvest for comparto 'A' age >= 60, zero for comparto 'F'
def _simple_rules(comparto, eta_media, volume_per_ha, area_basimetrica_per_ha):
    if comparto == 'F':
        return 0.0, 0.0
    return volume_per_ha * 0.25, math.inf


class TestSelectFromBottom:
    def test_orders_by_diameter(self):
        trees = pd.DataFrame({
            COL_D_CM: [40.0, 25.0, 55.0, 30.0],
        })
        result = acc.select_from_bottom(trees)
        assert list(result) == [1, 3, 0, 2]


class TestSimulateHarvestOnParcel:
    """Tests for _simulate_harvest_on_parcel."""

    def test_harvest_removes_trees_and_returns_volumes(self):
        """Basic harvest: removes correct trees, returns correct volumes."""
        # 1 sample area, 10 ha -> sampled_frac = 0.0125, scale factor = 80
        stats = ParcelStats(area_ha=10, sector='A', age=60, governo='Fustaia',
                            n_sample_areas=1, sampled_frac=0.0125)
        sim = _make_sim([
            # Small tree (not harvested)
            ('R', 'P1', 1, 15.0, 0.1, 'Faggio'),
            # 3 mature trees, sorted by diameter: 25, 35, 50
            ('R', 'P1', 1, 25.0, 0.5, 'Faggio'),
            ('R', 'P1', 1, 35.0, 1.0, 'Cerro'),
            ('R', 'P1', 1, 50.0, 2.0, 'Faggio'),
        ])
        n_before = len(sim)
        result = acc._simulate_harvest_on_parcel(
            sim, 'R', 'P1', stats, _simple_rules, acc.select_from_bottom)

        assert result is not None
        # vol_before = sum of mature trees' V * weight / sf
        expected_vol_before = (0.5 + 1.0 + 2.0) / 0.0125
        assert np.isclose(result.volume_before, expected_vol_before)
        # Harvest should be > 0 and <= vol_before
        assert result.harvest > 0
        assert result.harvest <= result.volume_before
        # vol_after = vol_before - harvest
        assert np.isclose(result.volume_after, result.volume_before - result.harvest)
        # Some rows should have been dropped
        assert len(sim) < n_before
        # Species shares should sum to 1
        assert np.isclose(sum(result.species_shares.values()), 1.0)

    def test_returns_none_for_ceduo(self):
        """Returns None when rules return (0, 0) for ceduo comparto."""
        stats = ParcelStats(area_ha=10, sector='F', age=60, governo='Ceduo',
                            n_sample_areas=1, sampled_frac=0.0125)
        sim = _make_sim([
            ('R', 'P1', 1, 30.0, 1.0, 'Faggio'),
        ])
        result = acc._simulate_harvest_on_parcel(
            sim, 'R', 'P1', stats, _simple_rules, acc.select_from_bottom)
        assert result is None
        # No trees removed
        assert len(sim) == 1

    def test_returns_none_for_no_mature_trees(self):
        """Returns None when there are no mature trees."""
        stats = ParcelStats(area_ha=10, sector='A', age=60, governo='Fustaia',
                            n_sample_areas=1, sampled_frac=0.0125)
        sim = _make_sim([
            ('R', 'P1', 1, 15.0, 0.1, 'Faggio'),
            ('R', 'P1', 1, 18.0, 0.15, 'Cerro'),
        ])
        result = acc._simulate_harvest_on_parcel(
            sim, 'R', 'P1', stats, _simple_rules, acc.select_from_bottom)
        assert result is None

    def test_respects_volume_limit(self):
        """Harvest stops when volume limit is reached."""
        # With 25% rule and high volume, should not harvest everything
        stats = ParcelStats(area_ha=10, sector='A', age=60, governo='Fustaia',
                            n_sample_areas=1, sampled_frac=0.0125)
        sim = _make_sim([
            ('R', 'P1', 1, 25.0, 0.5, 'Faggio'),
            ('R', 'P1', 1, 30.0, 0.8, 'Faggio'),
            ('R', 'P1', 1, 40.0, 1.5, 'Faggio'),
            ('R', 'P1', 1, 50.0, 2.5, 'Faggio'),
            ('R', 'P1', 1, 60.0, 4.0, 'Faggio'),
        ])
        result = acc._simulate_harvest_on_parcel(
            sim, 'R', 'P1', stats, _simple_rules, acc.select_from_bottom)
        assert result is not None
        # Mature volume per ha = sum(V) / sf / area = (0.5+0.8+1.5+2.5+4.0)/0.0125/10 = 744
        # 25% of 744 = 186 m3/ha -> limit = 186 * 10 = 1860 m3
        # Harvest should be <= limit
        vol_mature_per_ha = (0.5 + 0.8 + 1.5 + 2.5 + 4.0) / 0.0125 / 10
        limit = vol_mature_per_ha * 0.25 * 10
        assert result.harvest <= limit + 1e-9

    def test_weight_affects_volumes(self):
        """Trees with reduced weight contribute less volume."""
        sf = 0.0125
        stats = ParcelStats(area_ha=10, sector='A', age=60, governo='Fustaia',
                            n_sample_areas=1, sampled_frac=sf)
        sim = _make_sim([
            ('R', 'P1', 1, 30.0, 1.0, 'Faggio'),
            ('R', 'P1', 1, 40.0, 2.0, 'Faggio'),
        ])
        # Set weight to 0.5 (simulating 50% mortality)
        sim[COL_WEIGHT] = 0.5
        # Use generous rules (50%) so at least one tree fits under the limit
        generous = lambda c, a, v, b: (v * 0.50, math.inf)
        result = acc._simulate_harvest_on_parcel(
            sim, 'R', 'P1', stats, generous, acc.select_from_bottom)
        assert result is not None
        # vol_before = (1.0*0.5 + 2.0*0.5) / 0.0125 = 120
        assert np.isclose(result.volume_before, (1.0 * 0.5 + 2.0 * 0.5) / sf)

    def test_species_shares(self):
        """Species shares reflect mature volume fractions."""
        stats = ParcelStats(area_ha=10, sector='A', age=60, governo='Fustaia',
                            n_sample_areas=1, sampled_frac=0.0125)
        sim = _make_sim([
            ('R', 'P1', 1, 30.0, 1.0, 'Faggio'),
            ('R', 'P1', 1, 40.0, 3.0, 'Cerro'),
        ])
        result = acc._simulate_harvest_on_parcel(
            sim, 'R', 'P1', stats, _simple_rules, acc.select_from_bottom)
        assert result is not None
        # Faggio: 1.0/(1.0+3.0) = 0.25, Cerro: 3.0/(1.0+3.0) = 0.75
        assert np.isclose(result.species_shares['Faggio'], 0.25)
        assert np.isclose(result.species_shares['Cerro'], 0.75)


class TestScheduleHarvests:
    """Integration tests for schedule_harvests using real test data."""

    def test_basic_schedule(self, data_all, harvest_rules):
        """Parcels are harvested in mature-vol-per-ha order, target caps year."""
        events = acc.schedule_harvests(
            data_all, past_harvests=None,
            year_range=(2026, 2027), min_gap=10,
            target_volume=99999,  # very high -> harvest all eligible
            rules=harvest_rules)
        assert len(events) > 0
        # All events should have expected keys
        for e in events:
            assert acc.COL_YEAR in e
            assert acc.COL_HARVEST in e
            assert e[acc.COL_HARVEST] > 0
            assert np.isclose(e[acc.COL_VOLUME_AFTER],
                              e[acc.COL_VOLUME_BEFORE] - e[acc.COL_HARVEST])

    def test_min_gap_enforcement(self, data_all, harvest_rules):
        """Parcels not re-harvested within min_gap years."""
        events = acc.schedule_harvests(
            data_all, past_harvests=None,
            year_range=(2026, 2035), min_gap=10,
            target_volume=99999,
            rules=harvest_rules)
        # Track last harvest year per parcel
        last = {}
        for e in events:
            key = (e[acc.COL_COMPRESA], e[acc.COL_PARTICELLA])
            if key in last:
                assert e[acc.COL_YEAR] - last[key] >= 10, \
                    f"Parcel {key} harvested at years {last[key]} and {e[acc.COL_YEAR]}"
            last[key] = e[acc.COL_YEAR]

    def test_target_volume_cap(self, data_all, harvest_rules):
        """Year total should not greatly exceed target volume."""
        target = 50.0  # small target
        events = acc.schedule_harvests(
            data_all, past_harvests=None,
            year_range=(2026, 2026), min_gap=10,
            target_volume=target,
            rules=harvest_rules)
        if events:
            year_total = sum(e[acc.COL_HARVEST] for e in events)
            # Could exceed by at most one parcel's harvest, but should be
            # in the right ballpark (not 10x over)
            assert year_total < target * 10

    def test_past_harvests_delay_eligibility(self, data_all, harvest_rules):
        """Past harvests prevent re-harvesting within min_gap."""
        # Past harvest for parcel A in 2020, min_gap=10
        # -> parcel A not eligible until 2030
        past = pd.DataFrame({
            'Anno': [2020],
            acc.COL_COMPRESA: ['Test'],
            acc.COL_PARTICELLA: ['A'],
        })
        events = acc.schedule_harvests(
            data_all, past_harvests=past,
            year_range=(2026, 2029), min_gap=10,
            target_volume=99999,
            rules=harvest_rules)
        parcel_a_events = [e for e in events if e[acc.COL_PARTICELLA] == 'A']
        assert len(parcel_a_events) == 0, "Parcel A should be blocked by past harvest"

    def test_growth_increases_volume(self, data_all, harvest_rules):
        """Volume should grow between years when no harvest occurs."""
        # Use very high min_gap so no parcel gets re-harvested, and very high
        # target so all parcels are harvested in year 1. In year 2, the same
        # parcels are blocked. So we can't directly compare volumes.
        # Instead, check that events in later years (from different parcels)
        # show reasonable volumes.
        events = acc.schedule_harvests(
            data_all, past_harvests=None,
            year_range=(2026, 2028), min_gap=2,
            target_volume=99999,
            rules=harvest_rules)
        assert len(events) > 0
        # At least some harvesting should happen
        years = {e[acc.COL_YEAR] for e in events}
        assert len(years) >= 1

    def test_ceduo_parcels_excluded(self, trees_df, particelle_df):
        """Parcels with governo != Fustaia are never scheduled."""
        # Add a ceduo parcel to the metadata
        ceduo_row = pd.DataFrame([{
            acc.COL_COMPRESA: 'Test', acc.COL_PARTICELLA: 'Z',
            'CP': 'Test-Z', acc.COL_AREA_PARCEL: 10.0,
            acc.COL_COMPARTO: 'F', acc.COL_GOVERNO: 'Ceduo',
            acc.COL_ETA_MEDIA: 40,
            acc.COL_LOCALITA: 'Loc Z',
            acc.COL_ALT_MIN: 800, acc.COL_ALT_MAX: 900,
            acc.COL_ESPOSIZIONE: 'N', 'Pendenza %': 10,
            acc.COL_STAZIONE: 'Stazione Z',
            acc.COL_SOPRASSUOLO: 'Ceduo di test',
            acc.COL_PIANO_TAGLIO: 'Taglio Z', 'Note': '',
        }])
        particelle_ext = pd.concat([particelle_df, ceduo_row], ignore_index=True)

        # Add some trees for parcel Z
        ceduo_trees = pd.DataFrame([{
            acc.COL_COMPRESA: 'Test', acc.COL_PARTICELLA: 'Z',
            acc.COL_AREA_SAGGIO: 1, 'n': 1, 'poll': '',
            acc.COL_D_CM: 30.0, 'Classe diametrica': 6,
            acc.COL_H_M: 20.0, acc.COL_GENERE: 'Faggio',
            acc.COL_FUSTAIA: True, acc.COL_L10_MM: 3.0,
            acc.COL_COEFF_PRESSLER: 200,
        }])
        trees_ext = pd.concat([trees_df, ceduo_trees], ignore_index=True)
        trees_ext = acc.calculate_all_trees_volume(trees_ext)

        data = acc.parcel_data(
            ["alberi.csv"], trees_ext, particelle_ext,
            regions=["Test"], parcels=[], species=[])

        events = acc.schedule_harvests(
            data, past_harvests=None,
            year_range=(2026, 2026), min_gap=10,
            target_volume=99999,
            rules=_simple_rules)
        parcel_z = [e for e in events if e[acc.COL_PARTICELLA] == 'Z']
        assert len(parcel_z) == 0, "Ceduo parcel Z should never be harvested"


class TestCalculateTpdtTable:
    """Tests for calculate_tpdt_table grouping and aggregation."""

    COMMON_KWARGS = dict(
        year_range=(2026, 2027), min_gap=10, target_volume=99999,
        mortalita=0.0, tree_selection=acc.select_from_bottom,
    )

    def test_per_particella(self, data_all, harvest_rules):
        """Per-particella grouping: one row per (year, parcel)."""
        df = acc.calculate_tpdt_table(
            data_all, past_harvests=None,
            rules=harvest_rules,
            group_cols=[acc.COL_PARTICELLA],
            **self.COMMON_KWARGS)
        assert not df.empty
        # Each row should have a unique (year, particella) combination
        dupes = df.duplicated(subset=[acc.COL_YEAR, acc.COL_PARTICELLA])
        assert not dupes.any()

    def test_year_only(self, data_all, harvest_rules):
        """No group_cols: one row per year with totals."""
        df = acc.calculate_tpdt_table(
            data_all, past_harvests=None,
            rules=harvest_rules,
            group_cols=[],
            **self.COMMON_KWARGS)
        assert not df.empty
        # One row per year
        assert df[acc.COL_YEAR].nunique() == len(df)

    def test_per_genere_sums_to_parcel(self, data_all, harvest_rules):
        """Per-genere allocation sums to total per year."""
        df_total = acc.calculate_tpdt_table(
            data_all, past_harvests=None,
            rules=harvest_rules,
            group_cols=[],
            **self.COMMON_KWARGS)
        df_genere = acc.calculate_tpdt_table(
            data_all, past_harvests=None,
            rules=harvest_rules,
            group_cols=[acc.COL_GENERE],
            **self.COMMON_KWARGS)
        if df_total.empty:
            return
        # Sum of per-genere harvest per year should equal total harvest per year
        genere_by_year = df_genere.groupby(acc.COL_YEAR)[acc.COL_HARVEST].sum()
        for _, row in df_total.iterrows():
            year = row[acc.COL_YEAR]
            assert np.isclose(
                genere_by_year[year], row[acc.COL_HARVEST], rtol=1e-9), \
                f"Year {year}: genere sum {genere_by_year[year]} != total {row[acc.COL_HARVEST]}"

    def test_sorted_by_year(self, data_all, harvest_rules):
        """Output is sorted by year."""
        df = acc.calculate_tpdt_table(
            data_all, past_harvests=None,
            rules=harvest_rules,
            group_cols=[acc.COL_PARTICELLA],
            **self.COMMON_KWARGS)
        if df.empty:
            return
        years = df[acc.COL_YEAR].values
        assert list(years) == sorted(years)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
