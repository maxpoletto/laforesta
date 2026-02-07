#!/usr/bin/env python3
"""
Forest Growth Projection Model

A size-structured (transition matrix) approach to projecting forest volume growth.

Inputs:
- Tree table: region, parcel, species, diameter_cm, height_m, volume_m3, ipr_mm
  (ipr = thickness of last 10 growth rings in mm)
- Parcel table: region, parcel, average_age

The model:
1. Calculates percentage increment (p_v) from ipr using Pressler's formula
2. Groups trees into 5-cm diameter classes
3. Projects forward by:
   - Estimating diameter growth rate from ipr
   - Determining class transitions (trees moving to larger classes)
   - Applying volume increment within each class
   - Applying mortality (optional)
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


# --- Height-Diameter curves ---

class HeightCurves:
    """
    Manages height-diameter curves by (region, species).

    Each curve is of the form: h = a * ln(d) + b
    where d is diameter in cm and h is height in m.
    """

    def __init__(self):
        self.curves: dict[tuple[str, str], tuple[float, float]] = {}
        self._default_curve: Optional[tuple[float, float]] = None

    def add_curve(self, region: str, species: str, a: float, b: float):
        """Add a curve for a specific (region, species) combination."""
        self.curves[(region, species)] = (a, b)

    def set_default(self, a: float, b: float):
        """Set a default curve for unknown (region, species) combinations."""
        self._default_curve = (a, b)

    def get_height(self, diameter_cm: float, region: str, species: str) -> Optional[float]:
        """
        Estimate height from diameter using the appropriate curve.

        Returns None if no curve is available and no default is set.
        """
        if diameter_cm <= 0:
            return None

        key = (region, species)
        if key in self.curves:
            a, b = self.curves[key]
        elif self._default_curve is not None:
            a, b = self._default_curve
        else:
            return None

        # h = a * ln(d) + b
        return a * np.log(diameter_cm) + b

    def load_from_dataframe(self, df: pd.DataFrame,
                            region_col: str = 'region',
                            species_col: str = 'species',
                            a_col: str = 'a',
                            b_col: str = 'b'):
        """Load curves from a DataFrame."""
        for _, row in df.iterrows():
            self.add_curve(row[region_col], row[species_col], row[a_col], row[b_col])

    def __repr__(self):
        return f"HeightCurves({len(self.curves)} curves)"


# --- Volume estimation ---

def estimate_volume(diameter_cm: float, height_m: float,
                    form_factor: float = 0.45) -> float:
    """
    Estimate tree volume using the standard formula.

    V = (π/4) * d² * h * f

    where:
    - d = diameter at breast height (m)
    - h = height (m)
    - f = form factor (typically 0.4-0.5 for conifers)

    Returns volume in m³.
    """
    if diameter_cm <= 0 or height_m <= 0:
        return 0.0

    d_m = diameter_cm / 100.0  # Convert cm to m
    return (np.pi / 4) * (d_m ** 2) * height_m * form_factor


# --- Diameter class utilities ---

def diameter_class(d_cm: float, class_width: float = 5.0) -> int:
    """
    Assign a diameter to its class.
    Classes are labeled by their lower bound: 0-5 -> 0, 5-10 -> 5, etc.
    """
    return int((d_cm // class_width) * class_width)


def class_midpoint(class_lower: int, class_width: float = 5.0) -> float:
    """Midpoint of a diameter class."""
    return class_lower + class_width / 2


# --- Pressler's formula for percentage increment ---

def pressler_pv(d_cm: float, ipr_mm: float, n_rings: int = 10) -> float:
    """
    Pressler's formula for percentage volume increment.

    p_v ≈ (400 / n) * (Δr / d)

    where:
    - n = number of years (rings)
    - Δr = radial increment over n years (in same units as d)
    - d = current diameter

    The factor 400 comes from: 200 * 2, where 200 converts radius to diameter
    and accounts for the approximate relationship between radial and volume growth.

    Note: This is a simplification. The true relationship depends on form factor
    and height growth, but Pressler's formula is widely used for quick estimates.
    """
    if d_cm <= 0 or ipr_mm <= 0:
        return 0.0

    # Convert ipr from mm to cm to match diameter units
    delta_r_cm = ipr_mm / 10.0

    # Pressler's formula: p_v = (400 / n) * (Δr / d)
    # This gives annual percentage increment
    p_v = (400.0 / n_rings) * (delta_r_cm / d_cm)

    return p_v


def annual_diameter_increment(ipr_mm: float, n_rings: int = 10) -> float:
    """
    Estimate annual diameter increment from ipr.

    ipr is radial increment over n years, so annual diameter increment is:
    Δd = 2 * (ipr / n)
    """
    return 2.0 * (ipr_mm / 10.0) / n_rings  # Convert mm to cm, radial to diameter


# --- Mortality model (simple, can be refined) ---

def annual_mortality_rate(diameter_cm: float) -> float:
    """
    Simple mortality model based on diameter class.

    This is a placeholder - real mortality rates should come from:
    - Regional yield tables
    - Species-specific survival curves
    - Site-specific data

    General patterns:
    - Higher mortality in small trees (suppression, competition)
    - Lower mortality in medium trees
    - Slightly higher in very large/old trees (senescence)
    """
    # Placeholder U-shaped mortality curve
    if diameter_cm < 15:
        return 0.02  # 2% annual mortality for small trees
    elif diameter_cm < 40:
        return 0.01  # 1% for medium trees
    else:
        return 0.015  # 1.5% for large trees


# --- Core projection logic ---

@dataclass
class ProjectionResult:
    """Results of a growth projection."""
    year: int
    trees: pd.DataFrame
    volume_by_class: pd.DataFrame
    total_volume: float
    n_trees: float  # Can be fractional due to mortality


def calculate_tree_growth_params(trees: pd.DataFrame, n_rings: int = 10) -> pd.DataFrame:
    """
    Add growth parameters to tree table.

    Adds columns:
    - diameter_class: 5-cm class assignment
    - p_v: percentage volume increment (annual)
    - delta_d: annual diameter increment (cm/year)
    """
    df = trees.copy()

    df['diameter_class'] = df['diameter_cm'].apply(diameter_class)
    df['p_v'] = df.apply(
        lambda row: pressler_pv(row['diameter_cm'], row['ipr_mm'], n_rings),
        axis=1
    )
    df['delta_d'] = df['ipr_mm'].apply(
        lambda ipr: annual_diameter_increment(ipr, n_rings)
    )

    return df


def project_one_year(
    trees: pd.DataFrame,
    apply_mortality: bool = True,
    height_curves: Optional[HeightCurves] = None,
    recalculate_volume: bool = True,
    form_factor: float = 0.45
) -> pd.DataFrame:
    """
    Project tree table forward by one year.

    For each tree:
    1. Apply diameter growth
    2. Update height from height-diameter curves (if provided)
    3. Either recalculate volume from new dimensions, or apply percentage increment
    4. Update diameter class
    5. Apply mortality (reduces tree count or weight)

    Parameters:
    - trees: DataFrame with tree data and growth parameters
    - apply_mortality: Whether to apply mortality
    - height_curves: HeightCurves object for updating heights (optional)
    - recalculate_volume: If True and height_curves provided, recalculate volume
                          from diameter and height. If False, use p_v increment.
    - form_factor: Form factor for volume calculation (default 0.45)
    """
    df = trees.copy()

    # If no 'weight' column exists, each row represents 1 tree
    if 'weight' not in df.columns:
        df['weight'] = 1.0

    # Apply diameter growth
    df['diameter_cm'] = df['diameter_cm'] + df['delta_d']

    # Update height from curves if available
    if height_curves is not None:
        df['height_m'] = df.apply(
            lambda row: height_curves.get_height(
                row['diameter_cm'], row['region'], row['species']
            ), # type: ignore[arg-type]
            axis=1
        )

    # Update volume
    if height_curves is not None and recalculate_volume:
        # Recalculate volume from new diameter and height
        df['volume_m3'] = df.apply(
            lambda row: estimate_volume(
                row['diameter_cm'], row['height_m'], form_factor
            ),
            axis=1
        )
    else:
        # Apply volume growth using percentage increment
        # v_new = v_old * (1 + p_v/100)
        df['volume_m3'] = df['volume_m3'] * (1 + df['p_v'] / 100)

    # Update diameter class
    df['diameter_class'] = df['diameter_cm'].apply(diameter_class)

    # Apply mortality (reduces weight)
    if apply_mortality:
        mortality_rates = df['diameter_cm'].apply(annual_mortality_rate)
        df['weight'] = df['weight'] * (1 - mortality_rates)

    # Note: we don't update p_v or delta_d here - that would require
    # re-measuring ipr, which we can't do for projections.
    # In reality, growth rates decline with age, so this is optimistic.
    # A more sophisticated model would decay p_v over time.

    return df


def project_growth(
    trees: pd.DataFrame,
    years: int,
    apply_mortality: bool = True,
    decay_growth_rate: bool = False,
    growth_decay_factor: float = 0.98,
    height_curves: Optional[HeightCurves] = None,
    recalculate_volume: bool = True,
    form_factor: float = 0.45
) -> list[ProjectionResult]:
    """
    Project forest growth over multiple years.

    Parameters:
    - trees: Tree table with growth parameters already calculated
    - years: Number of years to project
    - apply_mortality: Whether to apply mortality
    - decay_growth_rate: Whether to reduce p_v over time (more realistic)
    - growth_decay_factor: If decaying, multiply p_v by this each year
    - height_curves: HeightCurves object for h-d relationships (optional)
    - recalculate_volume: If True, recalculate volume from d and h each year
    - form_factor: Form factor for volume calculation

    Returns list of ProjectionResult for year 0 (current) through year n.
    """
    results = []
    current_trees = trees.copy()

    if 'weight' not in current_trees.columns:
        current_trees['weight'] = 1.0

    # Record initial state (year 0)
    results.append(_make_result(current_trees, year=0))

    for year in range(1, years + 1):
        current_trees = project_one_year(
            current_trees,
            apply_mortality,
            height_curves=height_curves,
            recalculate_volume=recalculate_volume,
            form_factor=form_factor
        )

        if decay_growth_rate:
            current_trees['p_v'] = current_trees['p_v'] * growth_decay_factor
            current_trees['delta_d'] = current_trees['delta_d'] * growth_decay_factor

        results.append(_make_result(current_trees, year=year))

    return results


def _make_result(trees: pd.DataFrame, year: int) -> ProjectionResult:
    """Create a ProjectionResult from current tree state."""

    # Volume by diameter class (weighted)
    vol_by_class = trees.groupby('diameter_class').apply(
        lambda g: pd.Series({
            'n_trees': g['weight'].sum(),
            'total_volume': (g['volume_m3'] * g['weight']).sum(),
            'mean_diameter': (g['diameter_cm'] * g['weight']).sum() / g['weight'].sum()
                if g['weight'].sum() > 0 else 0
        }),
        include_groups=False # type: ignore[arg-type]
    ).reset_index()

    return ProjectionResult(
        year=year,
        trees=trees.copy(),
        volume_by_class=vol_by_class,
        total_volume=(trees['volume_m3'] * trees['weight']).sum(),
        n_trees=trees['weight'].sum()
    )


# --- Aggregation by parcel/region ---

def summarize_by_parcel(
    results: list[ProjectionResult],
    group_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Summarize projection results by parcel (or other grouping).

    Returns a table with one row per (group, year) combination.
    """
    if group_cols is None:
        group_cols = ['region', 'parcel']
    summaries = []

    for result in results:
        df = result.trees
        year = result.year
        grouped = df.groupby(group_cols).apply(
            lambda g: pd.Series({
                'year': year, # pylint: disable=cell-var-from-loop
                'n_trees': g['weight'].sum(),
                'total_volume_m3': (g['volume_m3'] * g['weight']).sum(),
                'mean_p_v': (g['p_v'] * g['weight']).sum() / g['weight'].sum()
                    if g['weight'].sum() > 0 else 0,
            }),
            include_groups=False # type: ignore[arg-type]
        ).reset_index()
        summaries.append(grouped)

    return pd.concat(summaries, ignore_index=True)


# --- Example usage ---

def create_sample_data(curves: HeightCurves) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample tree and parcel data for testing."""

    # Sample tree data
    np.random.seed(42)
    n_trees = 100

    trees = pd.DataFrame({
        'region': ['Lombardia'] * n_trees,
        'parcel': np.random.choice(['A1', 'A2', 'B1'], n_trees),
        'species': np.random.choice(['Picea abies', 'Fagus sylvatica'], n_trees),
        'diameter_cm': np.random.uniform(10, 50, n_trees),
        'ipr_mm': np.random.uniform(5, 25, n_trees),  # Last 10 rings in mm
    })

    trees['height_m'] = trees.apply(
        lambda row: curves.get_height(row['diameter_cm'],
                                      row['region'], row['species']), # type: ignore[arg-type]
        axis=1)

    trees['volume_m3'] = trees.apply(
        lambda row: estimate_volume(row['diameter_cm'], row['height_m']),
        axis=1)

    # Parcel metadata
    parcels = pd.DataFrame({
        'region': ['Lombardia'] * 3,
        'parcel': ['A1', 'A2', 'B1'],
        'average_age': [60, 45, 80],
    })

    return trees, parcels


def create_sample_height_curves() -> HeightCurves:
    """
    Create sample height-diameter curves.

    These are illustrative - real curves should be fitted to local data.
    The form is: h = a * ln(d) + b

    Typical values for temperate forests:
    - a (slope): 8-12 for conifers, 6-10 for broadleaves
    - b (intercept): -15 to -5, depending on site quality
    """
    curves = HeightCurves()

    # Picea abies (Norway spruce) - tends to be taller
    # At d=30cm: h = 10*ln(30) - 10 = 10*3.4 - 10 = 24m
    curves.add_curve('Lombardia', 'Picea abies', a=10.0, b=-10.0)

    # Fagus sylvatica (European beech) - slightly shorter for same diameter
    # At d=30cm: h = 8*ln(30) - 5 = 8*3.4 - 5 = 22.2m
    curves.add_curve('Lombardia', 'Fagus sylvatica', a=8.0, b=-5.0)

    # Set a default for unknown combinations
    curves.set_default(a=9.0, b=-8.0)

    return curves


def main():    # Demo
    """Entry point."""
    height_curves = create_sample_height_curves()
    trees, parcels = create_sample_data(height_curves)

    print("=== Input Data ===")
    print(f"Trees: {len(trees)} records")
    print(trees.head(10))
    print("\nParcels:")
    print(parcels)
    print(f"\nHeight curves: {height_curves}")

    # Add growth parameters
    trees_with_params = calculate_tree_growth_params(trees, n_rings=10)
    print("\n=== Trees with Growth Parameters ===")
    print(trees_with_params[['species', 'diameter_cm', 'diameter_class',
                             'height_m', 'ipr_mm', 'p_v', 'delta_d']].head(10))

    # Show what height curves predict vs measured
    print("\n=== Height Curve Validation ===")
    sample = trees_with_params.head(10).copy()
    sample['h_predicted'] = sample.apply(
        lambda row: height_curves.get_height(row['diameter_cm'],
                    row['region'], row['species']),  # type: ignore[arg-type]
        axis=1
    )
    print(sample[['species', 'diameter_cm', 'height_m', 'h_predicted']].round(1))

    # Project 20 years WITH height curves (volume recalculated from d and h)
    print("\n=== Projection WITH Height Curves (volume recalculated) ===")
    results_with_curves = project_growth(
        trees_with_params,
        years=20,
        apply_mortality=True,
        decay_growth_rate=True,
        growth_decay_factor=0.99,
        height_curves=height_curves,
        recalculate_volume=True
    )

    for r in results_with_curves[::5]:  # Every 5 years
        print(f"Year {r.year:2d}: {r.n_trees:6.1f} trees, {r.total_volume:8.1f} m³")

    # Project 20 years WITHOUT height curves (volume from p_v only)
    print("\n=== Projection WITHOUT Height Curves (p_v increment) ===")
    results_no_curves = project_growth(
        trees_with_params,
        years=20,
        apply_mortality=True,
        decay_growth_rate=True,
        growth_decay_factor=0.99,
        height_curves=None,
        recalculate_volume=False
    )

    for r in results_no_curves[::5]:
        print(f"Year {r.year:2d}: {r.n_trees:6.1f} trees, {r.total_volume:8.1f} m³")

    # Compare the two approaches
    print("\n=== Comparison: With vs Without Height Curves ===")
    print("Year  | With Curves | Without Curves | Difference")
    print("-" * 55)
    for r1, r2 in zip(results_with_curves[::5], results_no_curves[::5]):
        diff = r1.total_volume - r2.total_volume
        diff_pct = 100 * diff / r2.total_volume if r2.total_volume > 0 else 0
        print(f"{r1.year:4d}  | {r1.total_volume:8.1f} m³ | " +
              f"{r2.total_volume:10.1f} m³ | {diff:+6.1f} m³ ({diff_pct:+.1f}%)")

    # Summarize by parcel
    parcel_summary = summarize_by_parcel(results_with_curves)
    print("\n=== Volume by Parcel Over Time (with height curves) ===")
    pivot = parcel_summary.pivot_table(
        index='parcel',
        columns='year',
        values='total_volume_m3'
    )
    print(pivot[[0, 5, 10, 15, 20]].round(1))

if __name__ == "__main__":
    main()
