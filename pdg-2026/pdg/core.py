"""Calculate/render functions and rendering helpers."""

from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsort_keygen

from pdg.harvest_rules import HarvestRulesFunc, max_harvest
from pdg.computation import (
    SAMPLE_AREA_HA, MATURE_THRESHOLD, MATURE_FILTER,
    COL_D_CM, COL_H_M, COL_V_M3, COL_GENERE, COL_COMPRESA, COL_PARTICELLA,
    COL_CD_CM, COL_SCALE, COL_AREA_SAGGIO, COL_COEFF_PRESSLER, COL_L10_MM,
    COL_AREA_PARCEL, COL_COMPARTO, COL_GOVERNO, GOV_FUSTAIA,
    COL_ESPOSIZIONE, COL_STAZIONE, COL_SOPRASSUOLO, COL_PIANO_TAGLIO,
    COL_ALT_MIN, COL_ALT_MAX, COL_LOCALITA, COL_ETA_MEDIA,
    GROUP_COLS_ALIGN,
    ParcelData, ParcelStats,
    basal_area_m2, calculate_area_and_volume, calculate_volume_confidence_interval,
    diameter_class,
    SP_ABETE, SP_ACERO, SP_CASTAGNO, SP_CERRO, SP_CILIEGIO, SP_DOUGLAS,
    SP_FAGGIO, SP_LECCIO, SP_ONTANO, SP_PINO, SP_PINO_LARICIO,
    SP_PINO_MARITTIMO, SP_PINO_NERO, SP_SORBO,
)
from pdg.io import load_csv, load_trees, read_past_harvests, file_cache
from pdg.formatters import (
    OPT_STILE, CurveInfo,
    fmt_num, RenderResult, ColSpec,
    SnippetFormatter,
)
from pdg.simulation import (
    COL_IP_MEDIO, COL_INCR_CORR, COL_DELTA_D,
    COL_YEAR, COL_HARVEST, COL_VOLUME_BEFORE, COL_VOLUME_AFTER, COL_SPECIES_SHARES,
    COL_WEIGHT,
    TreeSelectionFunc, select_from_bottom,
    growth_per_group, harvest_parcel, schedule_harvests,
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

skip_graphs = False  # pylint: disable=invalid-name

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'upper left'
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

# =============================================================================
# INTERNAL DATAFRAME COLUMN NAMES (for calculate/render pairs)
# =============================================================================

# Common columns (used across tsv, tpt)
COL_VOLUME = 'volume'
COL_VOLUME_MATURE = 'volume_mature'
COL_AREA_HA = 'area_ha'
# tsv-specific
COL_N_TREES = 'n_trees'
COL_VOL_LO = 'vol_lo'
COL_VOL_HI = 'vol_hi'
# tpt-specific
COL_SECTOR = 'sector'
COL_AGE = 'age'
COL_PP_MAX = 'pp_max'
ROW_TOTAL = 'Totale'

# =============================================================================
# OPTION KEYS (shared between process_directive in pdg.py and render_* functions)
# =============================================================================

# Common options (used across multiple directives)
OPT_PER_COMPRESA = 'per_compresa'
OPT_PER_PARTICELLA = 'per_particella'
OPT_PER_GENERE = 'per_genere'
OPT_STIME_TOTALI = 'stime_totali'
OPT_TOTALI = 'totali'
OPT_METRICA = 'metrica'
# tsv-specific
OPT_INTERV_FIDUC = 'intervallo_fiduciario'
OPT_SOLO_MATURE = 'solo_mature'
# tpt column toggles
OPT_COL_COMPARTO = 'col_comparto'
OPT_COL_ETA = 'col_eta'
OPT_COL_AREA_HA = 'col_area_ha'
OPT_COL_VOLUME = 'col_volume'
OPT_COL_VOLUME_HA = 'col_volume_ha'
OPT_COL_VOLUME_MATURE = 'col_volume_mature'
OPT_COL_VOLUME_MATURE_HA = 'col_volume_mature_ha'
OPT_COL_PP_MAX = 'col_pp_max'
OPT_COL_PRELIEVO = 'col_prelievo'
OPT_COL_PRELIEVO_HA = 'col_prelievo_ha'
OPT_COL_INCR_CORR = 'col_incr_corr'
# Graph axis limits
OPT_X_MAX = 'x_max'
OPT_Y_MAX = 'y_max'
# tpdt-specific
OPT_ANNO_INIZIO = 'anno_inizio'
OPT_ANNO_FINE = 'anno_fine'
OPT_INTERVALLO = 'intervallo'
OPT_MORTALITA = 'mortalita'
OPT_VOLUME_OBIETTIVO = 'volume_obiettivo'
OPT_CALENDARIO = 'calendario'
# tpdt column toggles
OPT_COL_PRIMA_DOPO = 'col_prima_dopo'
# Required file parameters (used as option keys for validation)
OPT_EQUAZIONI = 'equazioni'


region_cache = {}

def parcel_data(tree_files: list[str], tree_df: pd.DataFrame, parcel_df: pd.DataFrame,
                regions: list[str], parcels: list[str], species: list[str]) -> ParcelData:
    """
    Compute parcel data.

    Args:
        tree_files: List of tree data files (used for cache key)
        tree_df: Tree data
        parcel_df: Parcel metadata
        regions: List of regions ("compresa") names
        parcels: List of parcels ("particella") names
        species: List of species ("genere") names

        Empty lists mean all regions/parcels/species.

    Returns:
        ParcelData with filtered trees, regions, species, and per-parcel stats.

    Raises:
        ValueError: for various invalid conditions
    """
    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    def _filter_df(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
        """Return new DataFrame with rows of df where df[column] is in values."""
        if not values:
            return df
        return df[df[column].isin(values)]  # type: ignore[reportGeneralTypeIssues]

    if parcels and not regions:
        raise ValueError("Se si specifica la particella occorre specificare la compresa")
    key = (
        tuple(sorted(tree_files)),
        tuple(sorted(regions)),
        tuple(sorted(parcels)),
        tuple(sorted(species)),
    )
    if key in region_cache:
        return region_cache[key]

    trees_region = tree_df.copy()
    trees_region = _filter_df(trees_region, COL_COMPRESA, regions)
    trees_region = _filter_df(trees_region, COL_PARTICELLA, parcels)
    trees_region_species = _filter_df(trees_region, COL_GENERE, species).copy()
    if len(trees_region_species) == 0:
        raise ValueError(f"Nessun dato trovato per comprese '{regions}' " +
                         f"particelle '{parcels}' generi '{species}'")

    parcel_stats = {}
    for (region, parcel), trees in trees_region.groupby([COL_COMPRESA, COL_PARTICELLA]):  # type: ignore[reportGeneralTypeIssues]
        md_row = parcel_df[
            (parcel_df[COL_COMPRESA] == region) &
            (parcel_df[COL_PARTICELLA] == parcel)
        ]
        if len(md_row) != 1:
            raise ValueError(f"Nessun metadato per particella {region}/{parcel}")

        md = md_row.iloc[0]
        area_ha = md[COL_AREA_PARCEL]
        n_sample_areas = trees.drop_duplicates(
            subset=[COL_COMPRESA, COL_PARTICELLA, COL_AREA_SAGGIO]).shape[0]
        if n_sample_areas == 0:
            raise ValueError(f"Nessuna area di saggio per particella {region}/{parcel}")
        sampled_frac = n_sample_areas * SAMPLE_AREA_HA / area_ha

        parcel_stats[(region, parcel)] = ParcelStats(
            area_ha=area_ha,
            sector=md[COL_COMPARTO],
            age=md[COL_ETA_MEDIA],
            governo=md[COL_GOVERNO],
            n_sample_areas=n_sample_areas,
            sampled_frac=sampled_frac,
        )

    trees_region_species[COL_CD_CM] = diameter_class(trees_region_species[COL_D_CM])  # type: ignore[reportGeneralTypeIssues]

    data = ParcelData(
        trees=trees_region_species,
        regions=sorted(trees_region[COL_COMPRESA].unique()),
        species=sorted(trees_region_species[COL_GENERE].unique()),
        parcels=parcel_stats,
    )
    region_cache[key] = data
    return data


# =============================================================================
# COLOR MAP
# =============================================================================

def get_color_map() -> dict:
    """
    Create consistent color mapping for species.

    Returns:
        Dict mapping species -> matplotlib hex color string
    """
    # Manual color mapping for the 14 Tabacchi species
    # Organized by type: deciduous (yellows/greens), conifers (blues), special (pinks/reds)
    return {
        # Deciduous broadleaves (yellow-green spectrum)
        SP_FAGGIO:        '#F4F269',  # canary-yellow
        SP_CASTAGNO:      '#CEE26B',  # lime-cream
        SP_ACERO:         '#A8D26D',  # willow-green
        SP_CERRO:         '#82C26E',  # moss-green
        SP_ONTANO:        '#5CB270',  # emerald
        SP_LECCIO:        '#4DA368',  # emerald-dark (adjusted for distinction)

        # Conifers (blue-aqua spectrum)
        SP_ABETE:         '#0c63e7',  # royal-blue - firs
        SP_DOUGLAS:       '#07c8f9',  # sky-aqua
        SP_PINO:          '#09a6f3',  # fresh-sky - common pines
        SP_PINO_NERO:     '#0a85ed',  # brilliant-azure
        SP_PINO_LARICIO:  '#0a85ed',  # brilliant-azure

        # Rare (coral/pink spectrum)
        SP_PINO_MARITTIMO: '#FB6363', # vibrant-coral
        SP_CILIEGIO:      '#DC4E5E',  # lobster-pink - cherry
        SP_SORBO:         '#BE385A',  # rosewood - rowan
    }


# =============================================================================
# RENDERING HELPERS
# =============================================================================

def dedup_total_area(df: pd.DataFrame, identity_cols: list[str]) -> float:
    """Sum area_ha, deduplicating rows created by expansion based on species.

    identity_cols: all grouping columns (e.g. group_cols, or
    [COL_YEAR] + group_cols for tables with a year dimension).
    COL_GENERE, if present, is excluded from deduplication so that
    each spatial unit's area is counted once.
    """
    dedup_cols = [c for c in identity_cols if c != COL_GENERE]
    if dedup_cols and len(dedup_cols) < len(identity_cols):
        df = df.drop_duplicates(subset=dedup_cols)
    return df[COL_AREA_HA].sum()  # type: ignore[reportGeneralTypeIssues]

def render_table(df: pd.DataFrame, group_cols: list[str],
                 col_specs: list[ColSpec], formatter: SnippetFormatter,
                 add_totals: bool) -> RenderResult:
    """Generic table renderer from a DataFrame and column specifications.

    Args:
        df: Data to render.
        group_cols: Columns used for grouping (appear first as left-aligned headers).
        col_specs: List of column specifications
        formatter: Output format (HTML/LaTeX/CSV).
        add_totals: Whether to append a totals row
    """
    col_specs = [c for c in col_specs if c.enabled]
    headers = [(col, GROUP_COLS_ALIGN[col]) for col in group_cols]
    headers += [(c.title, c.align) for c in col_specs]

    display_rows = []
    for _, row in df.iterrows():
        display_row = [str(row[col]) for col in group_cols]
        for c in col_specs:
            if isinstance(c.format, str):
                display_row.append(fmt_num(row[c.format], 1))  # type: ignore[reportGeneralTypeIssues]
            else:
                display_row.append(c.format(row))
        display_rows.append(display_row)

    if add_totals:
        total_row = [ROW_TOTAL]
        # We need to "eat" one column to make room for the totals label.
        if group_cols:
            total_row += [''] * (len(group_cols) - 1)
        else:
            if col_specs[0].total is not None:
                raise ValueError("La prima colonna della tabella non può essere aggregabile se non ci sono colonne di raggruppamento")
            else:
                col_specs = col_specs[1:]  # Skip first column
        for c in col_specs:
            if c.total is None:
                total_row.append('')
            elif isinstance(c.total, str):
                total_row.append(fmt_num(df[c.total].sum(), 1))  # type: ignore[reportGeneralTypeIssues]
            else:
                total_row.append(c.total(df))
        display_rows.append(total_row)

    return RenderResult(snippet=formatter.format_table(headers, display_rows))


# =============================================================================
# CURVE IPSOMETRICHE
# =============================================================================

def render_hypsometric_graph(data: ParcelData, equations_df: pd.DataFrame,
                     output_path: Path, formatter: SnippetFormatter,
                     color_map: dict, **options) -> RenderResult:
    """
    Generate height-diameter graphs.

    Args:
        data: Output from region_data
        equations_df: Pre-computed equations from CSV
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        RenderResult with snippet and filepath.
    """
    #pylint: disable=too-many-locals
    trees = data.trees
    species = data.species
    regions = data.regions

    figsize = (4, 3)
    fig, ax = plt.subplots(figsize=figsize)

    # First pass: scatter points (once per species)
    for sp in species:
        sp_data = trees[trees[COL_GENERE] == sp]
        x = sp_data[COL_D_CM].values  # type: ignore[reportGeneralTypeIssues]
        y = sp_data[COL_H_M].values  # type: ignore[reportGeneralTypeIssues]
        ax.scatter(x, y, color=color_map[sp], label=sp, alpha=0.7, linewidth=2, s=1)  # type: ignore[reportGeneralTypeIssues]

    # Second pass: regression curves (per compresa/genere pair)
    curve_info = []
    for region in regions:
        for sp in species:
            sp_data = trees[trees[COL_GENERE] == sp]
            x = sp_data[COL_D_CM].values  # type: ignore[reportGeneralTypeIssues]

            # Look up pre-computed equation from equations.csv, if any
            eq_row = equations_df[
                (equations_df['compresa'] == region) &
                (equations_df['genere'] == sp)
            ]

            if len(eq_row) > 0:
                eq = eq_row.iloc[0]
                x_smooth = np.linspace(1, x.max(), 100)  # type: ignore[reportGeneralTypeIssues]
                if eq['funzione'] == 'ln':
                    y_smooth = eq['a'] * np.log(x_smooth) + eq['b']
                    eq_str = f"y = {fmt_num(eq['a'], 2)}*ln(x) + {fmt_num(eq['b'], 2)}"
                else:  # 'lin'
                    y_smooth = eq['a'] * x_smooth + eq['b']
                    eq_str = f"y = {fmt_num(eq['a'], 2)}*x + {fmt_num(eq['b'], 2)}"

                ax.plot(x_smooth, y_smooth, color=color_map[sp],
                    linestyle='--', alpha=0.8, linewidth=1.5)

                curve_info.append(CurveInfo(
                    genere=sp,
                    equation=eq_str,
                    r_squared=eq['r2'],
                    n_points=int(eq['n']),
                ))

    if not skip_graphs:
        x_max = max(options[OPT_X_MAX], trees[COL_D_CM].max() + 3)
        y_max = max(options[OPT_Y_MAX], (trees[COL_H_M].max() + 6) // 5 * 5)
        ax.set_xlabel('Diametro (cm)')
        ax.set_ylabel('Altezza (m)')
        ax.set_xlim(-0.5, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(0, x_max, 1+x_max//10))
        td = min(ax.get_ylim()[1] // 5, 4)
        y_ticks = np.arange(0, ax.get_ylim()[1] + 1, td)
        ax.set_yticks(y_ticks)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')

    plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data, curve_info=curve_info)

    return RenderResult(filepath=output_path, snippet=snippet)


# =============================================================================
# CLASSI DIAMETRICHE
# =============================================================================

GCD_Y_LABELS = {
    'alberi_ha': 'Stima alberi / ha',
    'alberi_tot': 'Stima alberi totali',
    'volume_ha': 'Stima volume / ha (m³/ha)',
    'volume_tot': 'Stima volume totale (m³)',
    'G_ha': 'Area basimetrica (m²/ha)',
    'G_tot': 'Area basimetrica totale (m²)',
    'altezza': 'Altezza media (m)',
}

# Coarse bins for @@tabella_classi_diametriche table
COARSE_BIN0 = "1-30 cm"
COARSE_BIN1 = "31-50 cm"
COARSE_BIN2 = "50+ cm"
COARSE_BINS = [COARSE_BIN0, COARSE_BIN1, COARSE_BIN2]

def calculate_diameter_class_data(data: ParcelData, metrica: str, stime_totali: bool,
                       fine: bool = True) -> pd.DataFrame:
    """Calculate diameter class data for @@grafico_classi_diametriche/@@tabella_classi_diametriche directives.

    Args:
        data: Output from region_data
        metrica: alberi_*|volume_*|G_*|altezza
        stime_totali: scale by 1/sampled_frac if True (ignored for altezza)
        fine: True for 5cm buckets (5, 10, 15...), False for 3 coarse buckets

    Returns DataFrame indexed by bucket with species as columns.
    """
    trees = data.trees
    parcels = data.parcels
    species = data.species
    per_ha = metrica.endswith('_ha')

    if fine:
        bucket_key = COL_CD_CM
    else:
        def coarse_bin(d):
            return COARSE_BIN0 if d <= 30 else COARSE_BIN1 if d <= 50 else COARSE_BIN2
        bucket_key = trees[COL_CD_CM].apply(coarse_bin)

    bucket_vals = trees[COL_CD_CM] if fine else bucket_key

    # For height, compute mean directly (no scaling)
    if metrica == 'altezza':
        combined = trees.groupby([bucket_vals, COL_GENERE])[COL_H_M].mean().unstack(fill_value=0)  # type: ignore[reportGeneralTypeIssues]
        return combined.reindex(columns=species, fill_value=0).sort_index()  # type: ignore[reportGeneralTypeIssues]

    # Per-tree value: volume, basal area, or 1 (for counting)
    if metrica.startswith('volume'):
        raw = trees[COL_V_M3]
    elif metrica.startswith('G'):
        raw = basal_area_m2(trees[COL_D_CM])
    else:
        raw = pd.Series(1.0, index=trees.index)

    # Aggregate per (diameter_bucket, species) -> pivot to species columns.
    # When stime_totali, each tree's value is weighted by 1/sampled_frac
    # so that per-parcel sampling densities are accounted for.
    values = (raw * trees[COL_SCALE]) if stime_totali else raw
    combined = values.groupby([bucket_vals, trees[COL_GENERE]]).sum().unstack(fill_value=0)  # type: ignore[reportGeneralTypeIssues]

    if per_ha:
        parcel_keys = set(zip(trees[COL_COMPRESA], trees[COL_PARTICELLA]))
        combined = combined / sum(parcels[k].area_ha for k in parcel_keys)
    return combined.reindex(columns=species, fill_value=0).sort_index()  # type: ignore[reportGeneralTypeIssues]


def render_diameter_class_graph(data: ParcelData, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict, **options) -> RenderResult:
    """
    Generate diameter class histograms.

    Args:
        data: Output from region_data
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping
        options: metrica (alberi_*|volume_*|G_*), stime_totali, x_max, y_max

    Returns:
        RenderResult with snippet and filepath.
    """
    if not skip_graphs:
        species = data.species
        metrica = options[OPT_METRICA]
        stime_totali = options[OPT_STIME_TOTALI]

        values_df = calculate_diameter_class_data(data, metrica, stime_totali, fine=True)
        use_lines = metrica == 'altezza'

        figsize = (4, 3.75)
        fig, ax = plt.subplots(figsize=figsize)

        bottom = np.zeros(len(values_df.index))
        for genere in species:
            if genere not in values_df.columns:
                continue
            series = values_df[genere]
            if use_lines:
                nonzero = series[series > 0]
                ax.plot(nonzero.index, nonzero.values, marker='o', markersize=3, linewidth=1.5,  # type: ignore[reportGeneralTypeIssues]
                        color=color_map[genere], label=genere, alpha=0.85)
            else:
                ax.bar(series.index, series.values, bottom=bottom, width=4,  # type: ignore[reportGeneralTypeIssues]
                    label=genere, color=color_map[genere],
                    alpha=0.8, edgecolor='white', linewidth=0.5)
                bottom += series

        # x_max in cm (fine buckets are 5, 10, 15...)
        max_bucket = values_df.index.max() if len(values_df) > 0 else 50
        x_max = options[OPT_X_MAX] if options[OPT_X_MAX] > 0 else max_bucket + 5  # type: ignore[reportGeneralTypeIssues]
        y_max_auto = values_df.max().max() if use_lines else values_df.sum(axis=1).max()
        y_max = options[OPT_Y_MAX] if options[OPT_Y_MAX] > 0 else y_max_auto * 1.1

        ax.set_xlabel('Diametro (cm)')
        ax.set_ylabel(GCD_Y_LABELS[metrica])
        ax.set_xlim(0, x_max)  # type: ignore[reportGeneralTypeIssues]
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(0, x_max + 1, 10))  # type: ignore[reportGeneralTypeIssues]
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        handles, labels = ax.get_legend_handles_labels()
        if not use_lines:
            # Reverse legend order to match visual stacking order (top-to-bottom)
            handles, labels = reversed(handles), reversed(labels)
        ax.legend(handles, labels, title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data)

    return RenderResult(filepath=output_path, snippet=snippet)


def render_diameter_class_table(data: ParcelData, formatter: SnippetFormatter, **options) -> RenderResult:
    """
    Generate diameter class table (@@tabella_classi_diametriche directive).

    Creates a table with rows for each species and columns for diameter ranges:
    (0,30], (30,50], (50,max].
    """
    species = data.species
    metrica = options[OPT_METRICA]
    stime_totali = options[OPT_STIME_TOTALI]
    totals = metrica != 'altezza'
    use_decimals = metrica.startswith('volume') or metrica.startswith('G') or metrica == 'altezza'

    values_df = calculate_diameter_class_data(data, metrica, stime_totali, fine=False)

    headers = [(COL_GENERE, 'l')] + [(b, 'r') for b in COARSE_BINS]
    if totals:
        headers += [('Totale', 'r')]
    dec = 1 if use_decimals else 0

    rows = []
    col_totals = {b: 0.0 for b in COARSE_BINS}
    for genere in species:
        row = [genere]
        row_total = 0.0
        for b in COARSE_BINS:
            val = cast(float, values_df.at[b, genere]) if b in values_df.index else 0.0
            row.append(fmt_num(val, dec))
            if totals:
                row_total += val
                col_totals[b] += val
        if totals:
            row.append(fmt_num(row_total, dec))
        rows.append(row)

    # Add totals row
    if totals:
        total_row = ['Totale'] + [fmt_num(col_totals[b], dec) for b in COARSE_BINS]
        total_row.append(fmt_num(sum(col_totals.values()), dec))
        rows.append(total_row)

    return RenderResult(snippet=formatter.format_table(headers, rows))


# =============================================================================
# PARCEL PROPERTIES
# =============================================================================

def render_prop(particelle_df: pd.DataFrame, compresa: str, particella: str,
                formatter: SnippetFormatter) -> RenderResult:
    """Render parcel properties (@@prop directive)."""
    row = particelle_df[(particelle_df[COL_COMPRESA] == compresa) &
                        (particelle_df[COL_PARTICELLA] == particella)]
    if row.empty:
        raise ValueError(f"Particella '{particella}' non trovata in compresa '{compresa}'")
    row = row.iloc[0]

    area = f"{fmt_num(row[COL_AREA_PARCEL], 2)} ha"
    altitudine = f"{int(row[COL_ALT_MIN])}-{int(row[COL_ALT_MAX])} m"

    short_fields = [
        ('Area', area),
        (COL_LOCALITA, row[COL_LOCALITA]),
        (COL_ETA_MEDIA, f"{int(row[COL_ETA_MEDIA])} anni"),
        (COL_GOVERNO, row[COL_GOVERNO]),
        ('Altitudine', altitudine),
        (COL_ESPOSIZIONE, row[COL_ESPOSIZIONE] or ''),
    ]
    paragraph_fields = [
        (COL_STAZIONE, row[COL_STAZIONE]),
        (COL_SOPRASSUOLO, row[COL_SOPRASSUOLO]),
        (COL_PIANO_TAGLIO, row[COL_PIANO_TAGLIO]),
    ]

    return RenderResult(snippet=formatter.format_prop(short_fields, paragraph_fields))


# =============================================================================
# STIMA VOLUMI
# =============================================================================

def calculate_volume_table(data: ParcelData, group_cols: list[str],
                        calc_margin: bool, calc_total: bool,
                        calc_mature: bool = False) -> pd.DataFrame:
    """Calculate the table rows for the @@volumi directive. Returns a DataFrame.

    Args:
        data: Output from parcel_data
        group_cols: Grouping columns (Compresa, Particella, Genere)
        calc_margin: If True, include confidence interval margin columns
        calc_total: If True, scale by 1/sampled_frac to estimate totals
        calc_mature: If True, include volume_mature column (trees with D > 20cm only)
    """
    #pylint: disable=too-many-locals
    trees = data.trees
    if COL_V_M3 not in trees.columns:
        raise ValueError("@@volumi richiede dati con volumi (manca la colonna COL_V_M3). "
                         "Esegui --calcola-altezze-volumi per calcolarli.")
    parcels = data.parcels

    if not group_cols:
        trees = trees.copy()
        trees['_'] = 'Totale'
        group_cols = ['_']  # Pseudo-column for single aggregation

    rows = []
    for group_key, group_trees in trees.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_key))  # type: ignore[reportGeneralTypeIssues]

        # When calc_total, weight each tree by 1/sampled_frac to extrapolate
        # from sample plots to full parcel estimates.
        scale = group_trees[COL_SCALE] if calc_total else 1
        n_trees = group_trees[COL_SCALE].sum() if calc_total else float(len(group_trees))
        volume = (group_trees[COL_V_M3] * scale).sum()
        vol_mature = 0.0
        if calc_mature:
            mature_mask = group_trees[COL_D_CM] > MATURE_THRESHOLD
            vol_mature = (group_trees.loc[mature_mask, COL_V_M3] * (
                group_trees.loc[mature_mask, COL_SCALE] if calc_total else 1)).sum()
        # CI must be computed per parcel (non-linear function of tree data)
        margin = 0.0
        if calc_margin:
            if calc_total:
                for (region, parcel), ptrees in group_trees.groupby(  # type: ignore[reportGeneralTypeIssues]
                        [COL_COMPRESA, COL_PARTICELLA]):
                    _, pmargin = calculate_volume_confidence_interval(ptrees)
                    margin += pmargin / parcels[(region, parcel)].sampled_frac  # type: ignore[reportGeneralTypeIssues]
            else:
                _, margin = calculate_volume_confidence_interval(group_trees)

        row_dict[COL_N_TREES] = n_trees  # type: ignore[reportGeneralTypeIssues]
        row_dict[COL_VOLUME] = volume  # type: ignore[reportGeneralTypeIssues]
        if calc_mature:
            row_dict[COL_VOLUME_MATURE] = vol_mature  # type: ignore[reportGeneralTypeIssues]
        if calc_margin:
            row_dict[COL_VOL_LO] = volume - margin  # type: ignore[reportGeneralTypeIssues]
            row_dict[COL_VOL_HI] = volume + margin  # type: ignore[reportGeneralTypeIssues]
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if '_' in group_cols:
        group_cols.remove('_')
        df = df.drop(columns=['_'])
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)


def render_volume_table(data: ParcelData, formatter: SnippetFormatter, **options) -> RenderResult:
    """Generate volume summary table (@@volumi directive)."""
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    if options[OPT_PER_GENERE]:
        group_cols.append(COL_GENERE)

    df = calculate_volume_table(data, group_cols,
        options[OPT_INTERV_FIDUC], options[OPT_STIME_TOTALI],
        options[OPT_SOLO_MATURE])

    has_ci = options[OPT_INTERV_FIDUC]
    col_specs = [
        ColSpec('N. Alberi', 'r',
                lambda r: fmt_num(r[COL_N_TREES], 0),
                lambda d: fmt_num(d[COL_N_TREES].sum(), 0), True),
        ColSpec('Volume (m³)', 'r', COL_VOLUME, COL_VOLUME, True),
        ColSpec('Vol. mature (m³)', 'r', COL_VOLUME_MATURE, COL_VOLUME_MATURE,
                options.get(OPT_SOLO_MATURE, False)),
        ColSpec('IF inf (m³)', 'r', COL_VOL_LO, COL_VOL_LO, has_ci),
        ColSpec('IF sup (m³)', 'r', COL_VOL_HI, COL_VOL_HI, has_ci),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


# =============================================================================
# INCREMENTO PERCENTUALE
# =============================================================================

_GROWTH_REQUIRED_COLS = [COL_COEFF_PRESSLER, COL_L10_MM, COL_D_CM, COL_V_M3]

def check_growth_columns(trees: pd.DataFrame) -> None:
    """Validate that trees has the columns needed for growth computation."""
    for col in _GROWTH_REQUIRED_COLS:
        if col not in trees.columns:
            raise ValueError(f"Direttiva richiede la colonna '{col}'. "
                             "Esegui --calcola-incrementi e --calcola-altezze-volumi.")

def render_pct_growth_table(data: ParcelData, formatter: SnippetFormatter, **options) -> RenderResult:
    """Generate IP summary table (@@tabella_incremento_percentuale directive)."""
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    group_cols += [COL_GENERE, COL_CD_CM]

    check_growth_columns(data.trees)
    df = growth_per_group(data.trees, group_cols, options[OPT_STIME_TOTALI])

    col_specs = [
        ColSpec('Incr. pct.', 'r', COL_IP_MEDIO, None, True),
        ColSpec('Incr. corr. (m³)', 'r', COL_INCR_CORR, COL_INCR_CORR, True),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


def render_pct_growth_graph(data: ParcelData, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict,
                     **options) -> RenderResult:
    """Generate IP line graph (@@grafico_incremento_percentuale directive)."""
    if not skip_graphs:
        group_cols = []
        if options[OPT_PER_COMPRESA]:
            group_cols.append(COL_COMPRESA)
        if options[OPT_PER_PARTICELLA]:
            group_cols.append(COL_PARTICELLA)
        group_cols += [COL_GENERE, COL_CD_CM]

        check_growth_columns(data.trees)
        df = growth_per_group(data.trees, group_cols, options[OPT_STIME_TOTALI])

        metrica = options[OPT_METRICA]
        if metrica == 'ip':
            y_col, y_label = COL_IP_MEDIO, 'Incremento % medio'
        else:
            y_col, y_label = COL_INCR_CORR, 'Incremento corrente (m³)'

        # Each curve is a unique (optional compresa, optional particella, genere) tuple
        curve_cols = [c for c in group_cols if c != COL_CD_CM]

        fig, ax = plt.subplots(figsize=(5, 3.5))

        for curve_key, curve_df in df.groupby(curve_cols):
            if isinstance(curve_key, str):
                curve_key = (curve_key,)
            label = ' / '.join(str(k) for k in curve_key)  # type: ignore[reportGeneralTypeIssues]
            genere = curve_key[-1]  # last element is always Genere  # type: ignore[reportGeneralTypeIssues]
            curve_df = curve_df.sort_values(COL_CD_CM)
            ax.plot(curve_df[COL_CD_CM], curve_df[y_col],
                    marker='o', markersize=3, linewidth=1.5,
                    color=color_map.get(genere, '#0c63e7'),
                    label=label, alpha=0.85)

        ax.set_xlabel('Diametro (cm)')
        ax.set_ylabel(y_label)
        x_max = df[COL_CD_CM].max() + 5
        ax.set_xticks(range(0, x_max + 1, 10))
        ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data)
    return RenderResult(filepath=output_path, snippet=snippet)


# =============================================================================
# RIPRESA (HARVEST TABLE)
# =============================================================================

def calculate_harvest_table(data: ParcelData, rules: HarvestRulesFunc,
                        group_cols: list[str]) -> pd.DataFrame:
    """
    Calculate harvest (prelievo totale) table data for the @@prelievi directive.

    Calls harvest_parcel() per parcel to compute harvest volumes, then aggregates
    by group_cols. When Genere is in group_cols, harvest is allocated pro-rata
    using species shares from harvest_parcel().
    """
    #pylint: disable=too-many-locals
    trees = data.trees
    if COL_V_M3 not in trees.columns:
        raise ValueError("@@prelievi richiede dati con volumi (colonna V(m3) mancante). "
                         "Esegui --calcola-altezze-volumi per calcolarli.")
    parcels = data.parcels

    added_dummy = False
    if not group_cols:
        trees = trees.copy()
        trees['_'] = 'Totale'
        group_cols = ['_']
        added_dummy = True

    per_parcel = COL_PARTICELLA in group_cols or len(parcels) == 1

    # First pass: compute harvest for each particella
    parcel_info = {}
    for (region, parcel), part_trees in trees.groupby([COL_COMPRESA, COL_PARTICELLA]):  # type: ignore[reportGeneralTypeIssues]
        try:
            p = parcels[(region, parcel)]  # type: ignore[reportGeneralTypeIssues]
        except KeyError as e:
            raise ValueError(f"Particella {region}/{parcel} non trovata") from e

        vol_total, _ = calculate_area_and_volume(part_trees)
        result = harvest_parcel(part_trees, p, rules, select_from_bottom)
        if result is None:
            parcel_info[(region, parcel)] = None
            continue

        pp_max = result.harvest / result.volume_before * 100 if result.volume_before > 0 else 0

        parcel_info[(region, parcel)] = {
            COL_SECTOR: p.sector, COL_AGE: p.age, COL_AREA_HA: p.area_ha,
            COL_VOLUME: vol_total,
            COL_VOLUME_MATURE: result.volume_before,
            COL_PP_MAX: pp_max,
            COL_HARVEST: result.harvest,
            COL_SPECIES_SHARES: result.species_shares,
        }

    # Second pass: aggregate by group_cols
    rows = []
    for group_key, group_trees in trees.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row_dict = dict(zip(group_cols, group_key))

        volume, vol_mature, harvest, area_ha = 0.0, 0.0, 0.0, 0.0
        any_tree = False
        last_pp_max, last_sector, last_age = 0.0, '', 0

        for (region, parcel), part_trees in group_trees.groupby([COL_COMPRESA, COL_PARTICELLA]):  # type: ignore[reportGeneralTypeIssues]
            info = parcel_info[(region, parcel)]
            if info is None:
                continue

            any_tree = True

            if COL_GENERE in group_cols:
                # Pro-rata harvest allocation using pre-computed species shares
                genere = part_trees[COL_GENERE].iloc[0]
                frac = info[COL_SPECIES_SHARES].get(genere, 0)
                group_vol, _ = calculate_area_and_volume(part_trees)
                group_vol_mature, _ = calculate_area_and_volume(part_trees, MATURE_FILTER)
                volume += group_vol
                vol_mature += group_vol_mature
                harvest += info[COL_HARVEST] * frac
            else:
                volume += info[COL_VOLUME]
                vol_mature += info[COL_VOLUME_MATURE]
                harvest += info[COL_HARVEST]

            area_ha += info[COL_AREA_HA]
            last_pp_max = info[COL_PP_MAX]
            last_sector = info[COL_SECTOR]
            last_age = info[COL_AGE]

        if not any_tree:
            continue

        if per_parcel:
            row_dict[COL_SECTOR] = last_sector
            row_dict[COL_AGE] = last_age
            row_dict[COL_PP_MAX] = last_pp_max
        row_dict[COL_AREA_HA] = area_ha
        row_dict[COL_VOLUME] = volume
        row_dict[COL_VOLUME_MATURE] = vol_mature
        row_dict[COL_HARVEST] = harvest
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if added_dummy:
        group_cols.remove('_')
        df = df.drop(columns=['_'])
    if not group_cols or df.empty:
        return df
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)


def render_harvest_table(data: ParcelData, rules: HarvestRulesFunc,
                     formatter: SnippetFormatter, **options) -> RenderResult:
    """Render harvest (prelievo totale) table (@@prelievi directive)."""
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    if options[OPT_PER_GENERE]:
        group_cols.append(COL_GENERE)

    df = calculate_harvest_table(data, rules, group_cols)
    if df.empty:
        return RenderResult(snippet='')

    per_parcel = COL_PARTICELLA in group_cols or len(data.parcels) == 1

    # When grouping only by species, area cannot be meaningfully assigned to
    # individual species in a mixed forest, so hide area and per-hectare columns.
    genere_only = group_cols == [COL_GENERE]
    total_area = dedup_total_area(df, group_cols)

    # total_fn lambdas close over total_area (computed above)
    col_specs = [
        ColSpec('Classe', 'l', lambda r: str(r[COL_SECTOR]), None,
                options[OPT_COL_COMPARTO] and per_parcel),
        ColSpec('Età', 'r', lambda r: fmt_num(r[COL_AGE], 0), None,
                options[OPT_COL_ETA] and per_parcel),
        ColSpec('Area (ha)', 'r', COL_AREA_HA, lambda _: fmt_num(total_area, 1),
         options[OPT_COL_AREA_HA] and not genere_only),
        ColSpec('Vol tot (m³)', 'r', COL_VOLUME, COL_VOLUME, options[OPT_COL_VOLUME]),
        ColSpec('Vol/ha (m³/ha)', 'r',
            lambda r: fmt_num(r[COL_VOLUME] / r[COL_AREA_HA], 1),
            lambda d: fmt_num(d[COL_VOLUME].sum() / total_area, 1),
            options[OPT_COL_VOLUME_HA] and not genere_only),
        ColSpec('Provv. (m³)', 'r', COL_VOLUME_MATURE, COL_VOLUME_MATURE,
                options[OPT_COL_VOLUME_MATURE]),
        ColSpec('Provv. (m³/ha)', 'r',
                lambda r: fmt_num(r[COL_VOLUME_MATURE] / r[COL_AREA_HA], 1),
                lambda d: fmt_num(d[COL_VOLUME_MATURE].sum() / total_area, 1),
                options[OPT_COL_VOLUME_MATURE_HA] and not genere_only),
        ColSpec('Prel \\%', 'r', lambda r: fmt_num(r[COL_PP_MAX], 0),
                None, options[OPT_COL_PP_MAX] and per_parcel),
        ColSpec('Prel tot (m³)', 'r', COL_HARVEST, COL_HARVEST, options[OPT_COL_PRELIEVO]),
        ColSpec('Prel/ha (m³/ha)', 'r',
                lambda r: fmt_num(r[COL_HARVEST] / r[COL_AREA_HA], 1),
                lambda d: fmt_num(d[COL_HARVEST].sum() / total_area, 1),
                options[OPT_COL_PRELIEVO_HA] and not genere_only),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


# =============================================================================
# PIANO DI TAGLIO (HARVEST PLAN)
# =============================================================================

def calculate_harvest_plan(
    data: ParcelData,
    past_harvests: pd.DataFrame | None,
    year_range: tuple[int, int],
    min_gap: int,
    target_volume: float,
    mortalita: float,
    rules: HarvestRulesFunc,
    tree_selection: TreeSelectionFunc,
    group_cols: list[str],
) -> pd.DataFrame:
    """Compute harvest schedule table grouped by year and optional columns.

    Calls schedule_harvests() then aggregates. When COL_GENERE is in group_cols,
    each event is expanded into per-species rows using pro-rata allocation.
    """
    events = schedule_harvests(
        data, past_harvests, year_range, min_gap, target_volume,
        mortalita, rules, tree_selection)
    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    numeric_cols = [COL_HARVEST, COL_VOLUME_BEFORE, COL_VOLUME_AFTER]

    df[COL_AREA_HA] = df.apply(
        lambda r: data.parcels[(r[COL_COMPRESA], r[COL_PARTICELLA])].area_ha,
        axis=1)

    # Expand per-species if requested
    if COL_GENERE in group_cols:
        expanded = []
        for _, row in df.iterrows():
            shares = row[COL_SPECIES_SHARES]
            for genere, frac in shares.items():  # type: ignore[reportGeneralTypeIssues]
                new_row = {c: row[c] for c in [COL_YEAR, COL_COMPRESA,
                                                COL_PARTICELLA, COL_AREA_HA]}
                new_row[COL_GENERE] = genere
                for c in numeric_cols:
                    new_row[c] = row[c] * frac
                expanded.append(new_row)
        df = pd.DataFrame(expanded)

    # Group and sum (area_ha sums alongside volumes).
    # Always include COL_COMPRESA in groupby when per-parcel, so the
    # (compresa, particella) key is available for sector/age lookup.
    agg_cols = [COL_YEAR] + group_cols
    if COL_PARTICELLA in group_cols and COL_COMPRESA not in agg_cols:
        agg_cols.append(COL_COMPRESA)
    agg_cols = [c for c in agg_cols if c in df.columns]
    df = df.groupby(agg_cols, as_index=False)[numeric_cols + [COL_AREA_HA]].sum()

    # Sort by year, then by group_cols (natsort for Particella)
    sort_cols = [COL_YEAR] + group_cols
    df = df.sort_values(  # type: ignore[reportGeneralTypeIssues]
        sort_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)

    # Derive sector and age when per-parcel rows are available
    if COL_PARTICELLA in group_cols:
        first_year = year_range[0]
        df[COL_SECTOR] = df.apply(
            lambda r: data.parcels[(r[COL_COMPRESA], r[COL_PARTICELLA])].sector,
            axis=1)
        df[COL_AGE] = df.apply(
            lambda r: data.parcels[(r[COL_COMPRESA], r[COL_PARTICELLA])].age
                      + (r[COL_YEAR] - first_year),
            axis=1)

    return df


def render_harvest_plan(data: ParcelData, past_harvests: pd.DataFrame | None,
                      rules: HarvestRulesFunc,
                      formatter: SnippetFormatter, **options) -> RenderResult:
    """Render harvest schedule table (@@piano_di_taglio directive)."""
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    if options[OPT_PER_GENERE]:
        group_cols.append(COL_GENERE)

    df = calculate_harvest_plan(
        data, past_harvests,
        year_range=(options[OPT_ANNO_INIZIO], options[OPT_ANNO_FINE]),
        min_gap=options[OPT_INTERVALLO],
        target_volume=options[OPT_VOLUME_OBIETTIVO],
        mortalita=options[OPT_MORTALITA],
        rules=rules,
        tree_selection=select_from_bottom,
        group_cols=group_cols)
    if df.empty:
        return RenderResult(snippet='')

    per_parcel = COL_PARTICELLA in group_cols or len(data.parcels) == 1

    col_specs = [
        ColSpec('Anno', 'l', lambda r: str(int(r[COL_YEAR])), None, True),
        ColSpec('Classe', 'l', lambda r: str(r[COL_SECTOR]), None,
                options[OPT_COL_COMPARTO] and per_parcel),
        ColSpec('Età', 'r', lambda r: fmt_num(r[COL_AGE], 0), None,
                options[OPT_COL_ETA] and per_parcel),
        ColSpec('Provv. prima\n(m³/ha)', 'r',
                lambda r: fmt_num(r[COL_VOLUME_BEFORE] / r[COL_AREA_HA], 1),
                None,
                options[OPT_COL_PRIMA_DOPO]),
        ColSpec('Prelievo (m³)', 'r', COL_HARVEST, COL_HARVEST, True),
        ColSpec('Prel \\%', 'r',
                lambda r: fmt_num(r[COL_HARVEST] / r[COL_VOLUME_BEFORE] * 100, 0)
                    if r[COL_VOLUME_BEFORE] > 0 else '—',
                None, options[OPT_COL_PP_MAX] and per_parcel),
        ColSpec('Provv. dopo\n(m³/ha)', 'r',
                lambda r: fmt_num(r[COL_VOLUME_AFTER] / r[COL_AREA_HA], 1),
                None,
                options[OPT_COL_PRIMA_DOPO]),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])
