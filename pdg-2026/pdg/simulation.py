"""Harvest simulation engine: harvest_parcel, year_step, schedule_harvests, growth tables."""

import bisect
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from natsort import natsort_keygen

from pdg.harvest_rules import HarvestRulesFunc, max_harvest
from pdg.computation import (
    COL_COMPRESA, COL_PARTICELLA, COL_AREA_SAGGIO,
    COL_GENERE, COL_D_CM, COL_V_M3, COL_CD_CM, COL_SCALE,
    COL_PRESSLER, COL_L10_MM,
    GOV_FUSTAIA, MATURE_FILTER,
    ParcelData, ParcelStats,
    basal_area_m2, calculate_area_and_volume, diameter_class,
)


@dataclass
class GrowthTables:
    """Growth rate lookup tables built from parcel data."""
    by_group: dict          # (compresa, genere, diametro) -> (inc_pct, delta_d)
    available_diams: dict   # (compresa, genere) -> sorted list of diameter classes
    groupby_cols: list[str]

@dataclass
class HarvestResult:
    """Result of harvesting one parcel."""
    volume_before: float    # mature volume before harvest
    harvest: float          # volume harvested
    species_shares: dict[str, float]  # fraction of mature volume per species
    harvested_indices: list  # DataFrame indices of harvested trees
from pdg.formatters import fmt_num

# Internal column names for simulation state
COL_WEIGHT = '_weight'
COL_DIAM_GROWTH = '_diam_growth'

# Internal DataFrame column names for growth tables
COL_IP_MEDIO = 'ip_medio'
COL_INCR_CORR = 'incremento_corrente'
COL_DELTA_D = 'delta_d'

# Parcel ordering strategies for schedule_harvests
ORDINE_VOL_HA = 'vol_ha'    # highest mature volume per hectare first (default)
ORDINE_VOL_TOT = 'vol_tot'  # highest total mature volume first
ORDINE_DATA = 'data'        # oldest last-harvest date first, ties broken by vol/ha

# Internal DataFrame column names for harvest plan events (shared with core.py)
COL_YEAR = 'year'
COL_HARVEST = 'harvest'
COL_VOLUME_BEFORE = 'volume_before'
COL_VOLUME_AFTER = 'volume_after'
COL_SPECIES_SHARES = '_species_shares'

# Type: takes mature harvestable trees, returns their indices in harvest order
TreeSelectionFunc = Callable[[pd.DataFrame], pd.Index]


def select_from_bottom(trees: pd.DataFrame) -> pd.Index:
    """Thinning from below: smallest mature trees first."""
    return trees.sort_values(COL_D_CM).index


# =============================================================================
# GROWTH RATE CALCULATION
# =============================================================================

def growth_per_group(trees: pd.DataFrame, group_cols: list[str],
                     stime_totali: bool) -> pd.DataFrame:
    """Compute per-group growth metrics from tree data.

    group_cols must include COL_GENERE and COL_CD_CM.  Computes per group:
      - ip_medio: volume-weighted mean Pressler percentage increment
      - delta_d: mean annual diameter increment (cm)
      - incremento_corrente: volume * ip/100
    When stime_totali is True, volumes are scaled by COL_SCALE per tree.
    """
    assert COL_GENERE in group_cols and COL_CD_CM in group_cols

    rows = []
    for group_key, group_trees in trees.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_key))  # type: ignore[reportGeneralTypeIssues]
        ip_per_tree = (group_trees[COL_PRESSLER] * 2 * group_trees[COL_L10_MM]
                       / 100 / group_trees[COL_D_CM])
        delta_d = (2 * group_trees[COL_L10_MM] / 100).mean()

        # Volume-weighted mean ip; scale weights by 1/sampled_frac when
        # extrapolating to totals (parcels have different sampling densities).
        scale = group_trees[COL_SCALE] if stime_totali else 1
        v = group_trees[COL_V_M3]
        volume = (v * scale).sum()
        ip_medio = (ip_per_tree * v * scale).sum() / volume

        row_dict[COL_IP_MEDIO] = ip_medio  # type: ignore[reportGeneralTypeIssues]
        row_dict[COL_DELTA_D] = delta_d  # type: ignore[reportGeneralTypeIssues]
        row_dict[COL_INCR_CORR] = volume * ip_medio / 100  # type: ignore[reportGeneralTypeIssues]
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def year_step(sim: pd.DataFrame, weight: np.ndarray,
              growth: GrowthTables, mortality: float,
              diam_growth: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Advance the tree growth simulation by one year.

    If diam_growth is provided, applies diameter growth from the previous
    year before looking up new growth rates.  In this way, volume growth uses
    the year's starting diameter class, and trees that cross the maturity threshold
    bring their volume into the mature pool only in the following year ("ingrowth").

    Updates sim (volumes, diameters) and weight (mortality) in place.
    Returns (delta_d, fallback_count) — delta_d is diam_growth for the next year
    """

    def find_nearest_diameter(diams: list[int], target: int) -> int:
        """Return the element of diams closest to target."""
        idx = bisect.bisect_left(diams, target)
        if idx == 0:
            return diams[0]
        if idx >= len(diams):
            return diams[-1]
        if target - diams[idx - 1] <= diams[idx] - target:
            return diams[idx - 1]
        return diams[idx]

    def lookup_growth(sim: pd.DataFrame,
                      growth: GrowthTables) -> tuple[np.ndarray, np.ndarray, int]:
        """Look up percent growth and diameter change each tree row from the growth table.

        Returns (inc_pct, delta_d, fallback_count).  When a tree's exact diameter bucket
        is not in the growth table, falls back to the nearest available bucket.
        """
        prefix_cols = [c for c in growth.groupby_cols if c != COL_CD_CM]

        keys = list(zip(*(sim[c] for c in growth.groupby_cols)))
        rates = [growth.by_group.get(k) for k in keys]
        inc_pct = np.array([r[0] if r else np.nan for r in rates])
        delta_d = np.array([r[1] if r else np.nan for r in rates])

        missing = np.isnan(inc_pct)
        fallbacks = int(missing.sum())
        for i in np.where(missing)[0]:
            row = sim.iloc[i]
            prefix = tuple(row[c] for c in prefix_cols)
            diams = growth.available_diams.get(prefix)
            if not diams:
                inc_pct[i], delta_d[i] = 0.0, 0.0
                continue
            nearest = find_nearest_diameter(diams, int(row[COL_CD_CM]))
            inc_pct[i], delta_d[i] = growth.by_group.get(prefix + (nearest,), (0.0, 0.0))

        return inc_pct, delta_d, fallbacks

    if diam_growth is not None:
        sim[COL_D_CM] = sim[COL_D_CM].values + diam_growth
        sim[COL_CD_CM] = diameter_class(sim[COL_D_CM])  # type: ignore[reportGeneralTypeIssues]

    inc_pct, delta_d, fallbacks = lookup_growth(sim, growth)
    sim[COL_V_M3] = sim[COL_V_M3].values * (1 + inc_pct / 100)  # type: ignore[reportGeneralTypeIssues]
    weight *= (1 - mortality / 100)

    return delta_d, fallbacks


def growth_tables(data: ParcelData) -> GrowthTables:
    """Build growth rate lookup tables from parcel data."""
    groupby_cols = [COL_COMPRESA, COL_GENERE, COL_CD_CM]
    growth_df = growth_per_group(data.trees, groupby_cols, stime_totali=True)
    by_group = {}
    available_diams = defaultdict(list)
    for _, row in growth_df.iterrows():
        key = tuple(row[c] for c in groupby_cols)
        by_group[key] = (row[COL_IP_MEDIO], row[COL_DELTA_D])
        prefix = key[:-1]
        available_diams[prefix].append(int(row[COL_CD_CM]))  # type: ignore[reportGeneralTypeIssues]
    for prefix in available_diams:
        available_diams[prefix] = sorted(set(available_diams[prefix]))
    return GrowthTables(by_group, available_diams, groupby_cols)


# =============================================================================
# HARVEST
# =============================================================================

def snapshot_volumes(sim: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Compute total effective volume per parcel from simulation state."""
    vol = sim[COL_V_M3] * sim[COL_WEIGHT] * sim[COL_SCALE]
    return vol.groupby([sim[COL_COMPRESA], sim[COL_PARTICELLA]]).sum().to_dict()


def write_volume_log(volume_log: dict[int, dict[tuple[str, str], float]],
                     filepath: str | Path) -> None:
    """Write per-parcel volume log from simulation to CSV.

    Rows: (Compresa, Particella) sorted by compresa then particella (natural order).
    Columns: years.
    Cells: total volume (m³) at the beginning of each year.
    """
    years = sorted(volume_log.keys())
    all_keys: set[tuple[str, str]] = set()
    for year_data in volume_log.values():
        all_keys.update(year_data.keys())

    natsort_key = natsort_keygen()
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], natsort_key(k[1])))

    rows = []
    for compresa, particella in sorted_keys:
        row: dict[str, str | float] = {COL_COMPRESA: compresa, COL_PARTICELLA: particella}
        for year in years:
            row[str(year)] = volume_log[year].get((compresa, particella), 0.0)
        rows.append(row)

    pd.DataFrame(rows).to_csv(filepath, index=False, float_format="%.1f")


def _mature_vol_per_ha(sim: pd.DataFrame, region: str, parcel: str,
                       stats: ParcelStats) -> float:
    """Compute mature effective volume per hectare for a parcel in the simulation."""
    parcel_trees = sim[(sim[COL_COMPRESA] == region) & (sim[COL_PARTICELLA] == parcel)]
    vol, _ = calculate_area_and_volume(parcel_trees, MATURE_FILTER,  # type: ignore[reportGeneralTypeIssues]
                                       weight=parcel_trees[COL_WEIGHT])  # type: ignore[reportGeneralTypeIssues]
    return vol / stats.area_ha


def harvest_parcel(trees: pd.DataFrame, stats: ParcelStats,
                   rules: HarvestRulesFunc,
                   selection_fn: TreeSelectionFunc,
                   weight: pd.Series | None = None,
                   prudence: float = 100.0,
                   ) -> HarvestResult | None:
    """Compute harvest for one parcel's trees.

    Filters mature trees, applies harvest rules to determine limits, then
    selects trees (via selection_fn) up to those limits.

    Does not mutate trees — caller should drop result.harvested_indices if needed.

    Args:
        trees: Trees for this parcel (must have COL_SCALE)
        stats: ParcelStats for this parcel
        rules: returns (vol_limit_ha, area_limit_ha) given sector/age/stats
        selection_fn: orders mature trees for harvest (e.g., smallest-first)
        weight: per-tree mortality factor for simulation (default: 1)
    """
    vol_mature, basal = calculate_area_and_volume(trees, MATURE_FILTER, weight=weight)
    if vol_mature == 0:
        return None

    vol_limit_ha, area_limit_ha = rules(
        stats.sector, stats.age, vol_mature / stats.area_ha, basal / stats.area_ha)  # type: ignore[reportGeneralTypeIssues]
    if vol_limit_ha == 0 and area_limit_ha == 0:
        return None

    # Per-tree scaled values for the selection loop
    mature = trees[MATURE_FILTER(trees)]
    w = weight[mature.index] if weight is not None else 1
    scale = mature[COL_SCALE]
    tree_vol = mature[COL_V_M3] * w * scale
    tree_basal = basal_area_m2(mature[COL_D_CM]) * w * scale

    # Select trees in harvest order, accumulate until limits
    vol_limit = vol_limit_ha * stats.area_ha * prudence / 100
    area_limit = area_limit_ha * stats.area_ha * prudence / 100
    ordered_idx = selection_fn(mature)  # type: ignore[reportGeneralTypeIssues]
    harvested = []
    cum_vol, cum_area = 0.0, 0.0
    for idx in ordered_idx:
        tv, ta = tree_vol[idx], tree_basal[idx]
        if cum_vol + tv > vol_limit or cum_area + ta > area_limit:
            break
        cum_vol += tv
        cum_area += ta
        harvested.append(idx)

    # Species shares of mature volume (for pro-rata allocation)
    species_shares = {}
    for genere, g_trees in mature.groupby(COL_GENERE):
        species_shares[genere] = tree_vol[g_trees.index].sum() / vol_mature

    return HarvestResult(vol_mature, cum_vol, species_shares, harvested)  # type: ignore[reportGeneralTypeIssues]


def schedule_harvests(
    data: ParcelData,
    past_harvests: pd.DataFrame | None,
    year_range: tuple[int, int],
    min_gap: int,
    target_volume: float,
    mortality: float = 0.0,
    rules: HarvestRulesFunc = max_harvest,
    tree_selection: TreeSelectionFunc = select_from_bottom,
    volume_log: dict[int, dict[tuple[str, str], float]] | None = None,
    prudence: float = 100.0,
    ordine: str = ORDINE_VOL_HA,
    particelle_min: int = 0,
    gap_overrides: dict[int, int] | None = None,
) -> list[dict]:
    """Schedule harvests using a greedy algorithm with year-by-year growth simulation.

    Only considers parcels where governo == Fustaia.

    Args:
        ordine: Parcel priority within each year. 'vol_ha' = highest mature
            volume/ha first (default); 'vol_tot' = highest total mature volume
            first; 'data' = oldest last-harvest date first, ties by vol/ha.

    Returns list of dicts, one per (year, parcel) harvest event, with keys:
        year, Compresa, Particella, harvest, volume_before, volume_after, _species_shares
    """
    trees = data.trees
    parcels = data.parcels

    # Identify fustaia parcels only
    fustaia_keys = [(r, p) for (r, p), s in parcels.items()
                    if s.governo == GOV_FUSTAIA]
    if not fustaia_keys:
        return []

    # Mutable copies of parcel stats so we can increment age year by year
    sim_parcels = {k: copy(parcels[k]) for k in fustaia_keys}

    # Build growth lookup
    growth = growth_tables(data)

    # Build sim DataFrame with weight and diam_growth columns
    sim = trees[[COL_COMPRESA, COL_PARTICELLA, COL_AREA_SAGGIO,
                 COL_GENERE, COL_D_CM, COL_V_M3, COL_CD_CM, COL_SCALE]].copy()
    sim[COL_WEIGHT] = 1.0
    sim[COL_DIAM_GROWTH] = 0.0

    # Filter sim to fustaia parcel trees only
    fustaia_set = set(fustaia_keys)
    sim_parcel_keys = list(zip(sim[COL_COMPRESA], sim[COL_PARTICELLA]))
    sim = sim[[k in fustaia_set for k in sim_parcel_keys]].copy()

    # Build last_harvest dict from past harvests
    last_harvest: dict[tuple[str, str], int] = {}
    if past_harvests is not None and not past_harvests.empty:
        for _, row in past_harvests.iterrows():
            key = (row[COL_COMPRESA], row[COL_PARTICELLA])
            if key in fustaia_set:
                last_harvest[key] = max(last_harvest.get(key, 0), int(row['Anno']))  # type: ignore[reportGeneralTypeIssues]

    first_year, last_year = year_range
    events = []
    diam_growth_arr = None

    for y in range(first_year, last_year + 1):
        if volume_log is not None:
            volume_log[y] = snapshot_volumes(sim)

        # Compute mature vol/ha for each fustaia parcel and build priority list.
        parcel_vols = {}
        for region, parcel in fustaia_keys:
            vol_ha = _mature_vol_per_ha(sim, region, parcel, sim_parcels[(region, parcel)])  # type: ignore[reportGeneralTypeIssues]
            parcel_vols[(region, parcel)] = vol_ha

        if ordine == ORDINE_VOL_TOT:
            parcel_priority = [
                (-vol_ha * sim_parcels[(r, p)].area_ha, r, p)
                for (r, p), vol_ha in parcel_vols.items()]
        elif ordine == ORDINE_DATA:
            parcel_priority = [
                (last_harvest.get((r, p), 0), -vol_ha, r, p)
                for (r, p), vol_ha in parcel_vols.items()]
        else:  # ORDINE_VOL_HA (default)
            parcel_priority = [
                (-vol_ha, r, p)
                for (r, p), vol_ha in parcel_vols.items()]
        parcel_priority.sort()

        year_total = 0.0
        year_parcels = 0
        n_gap_skip = 0
        n_no_harvest = 0
        for *_, region, parcel in parcel_priority:
            # Min-gap check (per-year override takes precedence)
            effective_gap = gap_overrides.get(y, min_gap) if gap_overrides else min_gap
            if last_harvest.get((region, parcel), 0) > y - effective_gap:
                n_gap_skip += 1
                continue

            parcel_mask = (sim[COL_COMPRESA] == region) & (sim[COL_PARTICELLA] == parcel)
            parcel_trees = sim[parcel_mask]
            result = harvest_parcel(
                parcel_trees, sim_parcels[(region, parcel)],  # type: ignore[reportGeneralTypeIssues]
                rules, tree_selection, weight=parcel_trees[COL_WEIGHT],  # type: ignore[reportGeneralTypeIssues]
                prudence=prudence)
            if result is None or result.harvest == 0:
                n_no_harvest += 1
                continue

            sim.drop(result.harvested_indices, inplace=True)  # type: ignore[reportGeneralTypeIssues]

            events.append({
                COL_YEAR: y,
                COL_COMPRESA: region,
                COL_PARTICELLA: parcel,
                COL_HARVEST: result.harvest,
                COL_VOLUME_BEFORE: result.volume_before,
                COL_VOLUME_AFTER: result.volume_before - result.harvest,
                COL_SPECIES_SHARES: result.species_shares,
            })
            last_harvest[(region, parcel)] = y
            year_total += result.harvest
            year_parcels += 1
            if year_total >= target_volume and year_parcels >= particelle_min:
                break

        n_harvested = sum(1 for e in events if e[COL_YEAR] == y)
        n_total = len(parcel_priority)
        if year_total < target_volume:
            print(f"  @@piano_di_taglio anno {y}: obiettivo {fmt_num(target_volume, 0)} m³, "
                  f"raggiunto {fmt_num(year_total, 0)} m³ "
                  f"({n_harvested} tagliate, {n_gap_skip} in pausa, "
                  f"{n_no_harvest} non idonee, {n_total} totali)")

        # Growth step
        weight = sim[COL_WEIGHT].values.copy()  # type: ignore[reportGeneralTypeIssues]
        diam_growth_arr = sim[COL_DIAM_GROWTH].values if y > first_year else None  # type: ignore[reportGeneralTypeIssues]
        diam_growth_arr, _ = year_step(
            sim, weight, growth, mortality, diam_growth_arr)  # type: ignore[reportGeneralTypeIssues]
        sim[COL_WEIGHT] = weight
        sim[COL_DIAM_GROWTH] = diam_growth_arr

        # Age progression: each parcel ages one year
        for p in sim_parcels.values():
            p.age += 1

    if volume_log is not None:
        volume_log[last_year + 1] = snapshot_volumes(sim)

    return events
