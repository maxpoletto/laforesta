"""
Tests for acc.py forest analysis module.

Test categories:
(a) Aggregation consistency - total vs per-particella breakdown
(b) Cross-query consistency - @@tsv totals match @@tcd sums
(c) Correct scaling with different sample areas
(d) Edge cases
(f) Confidence interval sanity

Test data (in test/data/):
- Parcel A: 10 ha, 1 sample area  -> sampled_frac = 0.0125, scale = 80
- Parcel B: 10 ha, 2 sample areas -> sampled_frac = 0.025,  scale = 40
- Parcel C: 10 ha, 5 sample areas -> sampled_frac = 0.0625, scale = 16
- Species: Faggio, Cerro (both have 2-coefficient Tabacchi equations)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import acc

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
            data_all, group_cols=['Particella'], calc_margin=False, calc_total=True
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
            data_all, group_cols=['Particella'], calc_margin=False, calc_total=True
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
            data_all, group_cols=['Genere'], calc_margin=False, calc_total=True
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
            data_all, group_cols=['Particella', 'Genere'],
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
            data_all, group_cols=['Particella'], stime_totali=True
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
        trees = data_all['trees']
        parcels = data_all['parcels']
        manual_basal = 0.0
        for (region, parcel), ptrees in trees.groupby(['Compresa', 'Particella']):
            sf = parcels[(region, parcel)]['sampled_frac']
            basal = np.pi / 4 * ptrees['D(cm)'] ** 2 / 10000  # m²
            manual_basal += basal.sum() / sf

        assert np.isclose(tcd_basal, manual_basal, rtol=1e-9), \
            f"@@tcd G {tcd_basal} != manual G {manual_basal}"


# =============================================================================
# (c) CORRECT SCALING WITH DIFFERENT SAMPLE AREAS
# =============================================================================

class TestSampleAreaScaling:
    """Test that trees scale correctly based on sample area density.

    Test data:
    - Parcel A: 10 ha, 1 sample area  -> sampled_frac = 0.0125, scale = 80
    - Parcel B: 10 ha, 2 sample areas -> sampled_frac = 0.025,  scale = 40
    - Parcel C: 10 ha, 5 sample areas -> sampled_frac = 0.0625, scale = 16
    """

    def test_sampled_frac_computation(self, data_all):
        """Verify sampled_frac is computed correctly."""
        parcels = data_all['parcels']

        # Parcel A: 1 sample area, 10 ha -> 1 * 0.125 / 10 = 0.0125
        assert np.isclose(parcels[('Test', 'A')]['sampled_frac'], 0.0125), \
            f"Parcel A sampled_frac wrong: {parcels[('Test', 'A')]['sampled_frac']}"

        # Parcel B: 2 sample areas, 10 ha -> 2 * 0.125 / 10 = 0.025
        assert np.isclose(parcels[('Test', 'B')]['sampled_frac'], 0.025), \
            f"Parcel B sampled_frac wrong: {parcels[('Test', 'B')]['sampled_frac']}"

        # Parcel C: 5 sample areas, 10 ha -> 5 * 0.125 / 10 = 0.0625
        assert np.isclose(parcels[('Test', 'C')]['sampled_frac'], 0.0625), \
            f"Parcel C sampled_frac wrong: {parcels[('Test', 'C')]['sampled_frac']}"

    def test_tree_scaling_parcel_a(self, data_parcel_a):
        """Parcel A: 4 sampled trees should scale to 4 * 80 = 320 estimated trees."""
        df = acc.calculate_tsv_table(
            data_parcel_a, group_cols=[], calc_margin=False, calc_total=True
        )
        # 4 trees / 0.0125 = 320
        expected_trees = 4 / 0.0125
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Parcel A trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_tree_scaling_parcel_b(self, data_parcel_b):
        """Parcel B: 6 sampled trees should scale to 6 * 40 = 240 estimated trees."""
        df = acc.calculate_tsv_table(
            data_parcel_b, group_cols=[], calc_margin=False, calc_total=True
        )
        # 6 trees / 0.025 = 240
        expected_trees = 6 / 0.025
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Parcel B trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_tree_scaling_parcel_c(self, data_parcel_c):
        """Parcel C: 10 sampled trees should scale to 10 * 16 = 160 estimated trees."""
        df = acc.calculate_tsv_table(
            data_parcel_c, group_cols=[], calc_margin=False, calc_total=True
        )
        # 10 trees / 0.0625 = 160
        expected_trees = 10 / 0.0625
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Parcel C trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_total_scaled_trees(self, data_all):
        """Total scaled trees across all parcels."""
        df = acc.calculate_tsv_table(
            data_all, group_cols=[], calc_margin=False, calc_total=True
        )
        # 4/0.0125 + 6/0.025 + 10/0.0625 = 320 + 240 + 160 = 720
        expected_trees = 320 + 240 + 160
        assert np.isclose(df['n_trees'].sum(), expected_trees, rtol=1e-9), \
            f"Total trees {df['n_trees'].sum()} != expected {expected_trees}"

    def test_volume_scaling_consistency(self, data_all, data_parcel_a, data_parcel_b, data_parcel_c):
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

        total = df_all['volume'].sum()
        sum_parts = df_a['volume'].sum() + df_b['volume'].sum() + df_c['volume'].sum()

        assert np.isclose(total, sum_parts, rtol=1e-9), \
            f"Total volume {total} != sum of parts {sum_parts}"

    def test_per_ha_values(self, data_all):
        """Per-hectare values should be total divided by total area (30 ha)."""
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

        # Total area = 10 + 10 + 10 = 30 ha
        expected_per_ha = total_volume / 30.0

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
        trees = data_all['trees']
        _, margin = acc.calculate_volume_confidence_interval(trees)
        assert margin > 0, f"Margin should be positive, got {margin}"

    def test_margin_increases_with_volume(self, trees_df, particelle_df):
        """Larger samples should have larger absolute margins."""
        # Single parcel
        data_a = acc.parcel_data(
            ["alberi.csv"], trees_df, particelle_df,
            regions=["Test"], parcels=["A"], species=[]
        )
        _, margin_a = acc.calculate_volume_confidence_interval(data_a['trees'])

        # All parcels (more trees)
        data_all = acc.parcel_data(
            ["alberi.csv"], trees_df, particelle_df,
            regions=["Test"], parcels=[], species=[]
        )
        _, margin_all = acc.calculate_volume_confidence_interval(data_all['trees'])

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
            data_all, group_cols=['Genere'], calc_margin=True, calc_total=True
        )

        for _, row in df.iterrows():
            assert row['vol_lo'] < row['volume'] < row['vol_hi'], \
                f"Species {row['Genere']}: CI [{row['vol_lo']}, {row['vol_hi']}] " \
                f"should bracket {row['volume']}"


# =============================================================================
# VOLUME CALCULATION SPOT CHECKS
# =============================================================================

class TestVolumeCalculation:
    """Basic sanity checks on volume calculations (not full Tabacchi verification)."""

    def test_volumes_are_positive(self, trees_df):
        """All calculated volumes should be positive."""
        assert (trees_df['V(m3)'] > 0).all(), "All volumes should be positive"

    def test_volume_increases_with_size(self, trees_df):
        """Larger trees (D*h) should generally have larger volumes."""
        # Group by species to compare within species
        for genere, group in trees_df.groupby('Genere'):
            if len(group) < 2:
                continue
            # Sort by D²*h
            group = group.copy()
            group['d2h'] = group['D(cm)'] ** 2 * group['h(m)']
            group = group.sort_values('d2h')

            # Volumes should be monotonically increasing (within species)
            volumes = group['V(m3)'].values
            assert all(volumes[i] <= volumes[i+1] for i in range(len(volumes)-1)), \
                f"Volumes for {genere} should increase with tree size"

    def test_faggio_volume_formula(self):
        """Spot check Faggio volume formula: V = (0.81151 + 0.038965 * D² * h) / 1000."""
        # D=30, h=20 -> V = (0.81151 + 0.038965 * 900 * 20) / 1000 = 0.702 m³
        volume = acc.calculate_one_tree_volume(30, 20, 'Faggio')
        expected = (0.81151 + 0.038965 * 900 * 20) / 1000
        assert np.isclose(volume, expected, rtol=1e-6), \
            f"Faggio volume {volume} != expected {expected}"

    def test_cerro_volume_formula(self):
        """Spot check Cerro volume formula: V = (-0.043221 + 0.038079 * D² * h) / 1000."""
        # D=30, h=20 -> V = (-0.043221 + 0.038079 * 900 * 20) / 1000 = 0.685 m³
        volume = acc.calculate_one_tree_volume(30, 20, 'Cerro')
        expected = (-0.043221 + 0.038079 * 900 * 20) / 1000
        assert np.isclose(volume, expected, rtol=1e-6), \
            f"Cerro volume {volume} != expected {expected}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
