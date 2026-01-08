#!/usr/bin/env python3
# pylint: disable=too-many-lines
# pylint: disable=singleton-comparison
"""
Forest Analysis: Accrescimenti Tool
Three-mode tool for equation generation, height calculation, and report generation.
"""

from abc import ABC, abstractmethod
import argparse
from pathlib import Path
import re
import subprocess
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Matplotlib configuration
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

SAMPLE_AREAS_PER_HA = 8
MIN_TREES_PER_HA = 0.5 # Ignore buckets less than this in classi diametriche graphs.

# =============================================================================
# REGRESSION / CURVE FITTING
# =============================================================================

class RegressionFunc(ABC):
    """Abstract base class for regression functions."""

    def __init__(self):
        self.a = None
        self.b = None
        self.r2 = None
        self.x_range = None
        self.n_points = None

    @abstractmethod
    def _clean_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clean and validate input data. Returns (x_clean, y_clean)."""

    @abstractmethod
    def _fit_params(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Fit parameters to data. Returns (a, b)."""

    @abstractmethod
    def _predict(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Predict y values from x using parameters a, b."""

    @abstractmethod
    def _create_lambda(self, a: float, b: float):
        """Create lambda function for prediction."""

    @abstractmethod
    def _format_equation(self, a: float, b: float) -> str:
        """Format equation as string."""

    def fit(self, x: np.ndarray, y: np.ndarray, min_points: int = 10) -> bool:
        """Fit the regression function to data. Returns True if successful."""
        x_clean, y_clean = self._clean_data(x, y)

        if len(x_clean) < min_points:
            return False

        self.a, self.b = self._fit_params(x_clean, y_clean)
        y_pred = self._predict(x_clean, self.a, self.b)

        # Calculate R²
        if np.var(y_clean) > 0:
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            self.r2 = 1 - (ss_res / ss_tot)
        else:
            self.r2 = 0.0

        self.x_range = (x_clean.min(), x_clean.max())
        self.n_points = len(x_clean)
        return True

    def get_lambda(self):
        """Return a lambda function for prediction."""
        if self.a is None or self.b is None:
            return None
        return self._create_lambda(self.a, self.b)

    def __str__(self) -> str:
        """Return string representation of the fitted function."""
        if self.a is None or self.b is None:
            return "Not fitted"
        return self._format_equation(self.a, self.b)


class LogarithmicRegression(RegressionFunc):
    """Logarithmic regression: y = a*ln(x) + b"""

    def _clean_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = (x > 0) & np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _fit_params(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        coeffs = np.polyfit(np.log(x), y, 1)
        return coeffs[0], coeffs[1]

    def _predict(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.log(x) + b

    def _create_lambda(self, a: float, b: float):
        return lambda x: a * np.log(np.maximum(x, 0.1)) + b

    def _format_equation(self, a: float, b: float) -> str:
        return f"y = {a:.2f}*ln(x) + {b:.2f}"


class LinearRegression(RegressionFunc):
    """Linear regression: y = a*x + b"""

    def _clean_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def _fit_params(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0], coeffs[1]

    def _predict(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    def _create_lambda(self, a: float, b: float):
        return lambda x: a * x + b

    def _format_equation(self, a: float, b: float) -> str:
        return f"y = {a:.2f}*x + {b:.2f}"


# =============================================================================
# SNIPPET FORMATTERS (for template substitution)
# =============================================================================

class SnippetFormatter(ABC):
    """Formats individual components (images, metadata) for template insertion."""

    @abstractmethod
    def format_image(self, filepath: Path) -> str:
        """Format image reference for this format."""

    @abstractmethod
    def format_metadata(self, stat: dict, curve_info: list = None) -> str:
        """Format metadata block for this format.

        Args:
            stats: Statistics about the region/species
            curve_info: List of dicts with {species, equation, r_squared, n_points}
                       from equations.csv
        """

    @abstractmethod
    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Format a data table for this format.

        Args:
            headers: Column headers
            rows: Data rows (each row is a list of strings)

        Returns:
            Formatted table snippet
        """


class HTMLSnippetFormatter(SnippetFormatter):
    """HTML snippet formatter."""

    def format_image(self, filepath: Path) -> str:
        return f'<img src="{filepath.name}" class="graph-image">'

    def format_metadata(self, stat: dict, curve_info: list = None) -> str:
        """Format metadata as HTML."""
        html = '<div class="graph-details">\n'
        html += f'<p><strong>Alberi campionati:</strong> {stat["sampled_trees"]:d}</p>\n'
        html += f'<p><strong>Stima totale:</strong> {stat["estimated_total"]:d}</p>\n'

        if "mean_height" in stat:
            html += f'<p><strong>Altezza media:</strong> {stat["mean_height"]:.1f} m</p>\n'
        if "mean_diameter_class" in stat:
            html += '<p><strong>Classe diametrica media:</strong> '
            html += f'{stat["mean_diameter_class"]:.0f}</p>\n'

        if curve_info:
            i = 'i' if len(curve_info) > 1 else 'e'
            html += f'<br><p><strong>Equazion{i} interpolant{i}:</strong></p>\n'
            for curve in curve_info:
                html += (f'<p>{curve["species"]}: {curve["equation"]} '
                        f'(R² = {curve["r_squared"]:.2f}, n = {curve["n_points"]})</p>\n')

        html += '</div>\n'
        return html

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Format table as HTML."""
        html = '<table class="volume-table">\n'
        html += '  <thead>\n    <tr>\n'
        for header in headers:
            html += f'      <th>{header}</th>\n'
        html += '    </tr>\n  </thead>\n'
        html += '  <tbody>\n'
        for row in rows:
            html += '    <tr>\n'
            for cell in row:
                html += f'      <td>{cell}</td>\n'
            html += '    </tr>\n'
        html += '  </tbody>\n</table>\n'
        return html


class LaTeXSnippetFormatter(SnippetFormatter):
    """LaTeX snippet formatter."""

    def format_image(self, filepath: Path) -> str:
        return (f'\\begin{{figure}}[H]\n'
                f'  \\centering\n'
                f'  \\includegraphics[width=0.9\\textwidth]{{{filepath.name}}}\n'
                f'\\end{{figure}}\n')

    def format_metadata(self, stat: dict, curve_info: list = None) -> str:
        """Format metadata as LaTeX."""
        latex = '\\begin{quote}\\small\n'
        latex += f"\\textbf{{Alberi campionati:}} {stat['sampled_trees']:d}\\\\\n"
        latex += f"\\textbf{{Stima totale:}} {stat['ntrees_total']:d}\\\\\n"

        if "mean_height" in stat:
            latex += f"\\textbf{{Altezza media:}} {stat['mean_height']:.1f} m\\\\\n"
        if "mean_diameter_class" in stat:
            latex += "\\textbf{{Classe diametrica media:}} "
            latex += f"{stat['mean_diameter_class']:.0f}\\\\\n"

        if curve_info:
            i = 'i' if len(curve_info) > 1 else 'e'
            latex += f'\\\\\n\\textbf{{Equazion{i} interpolant{i}:}}\\\\\n'
            for curve in curve_info:
                eq = curve['equation'].replace('*', r'\times ')
                eq = eq.replace('ln', r'\ln')
                latex += (f"{curve['species']}: ${eq}$ ($R^2$ = {curve['r_squared']:.2f}, "
                         f"$n$ = {curve['n_points']})\\\\\n")

        latex += '\\end{quote}\n'
        return latex

    def format_table(self, headers: list[str], rows: list[list[str]]) -> str:
        """Format table as LaTeX using longtable for page breaks."""
        n_cols = len(headers)
        # Use l for text columns, r for numeric columns
        # Assume first column is text, the rest are numeric (volumes)
        if n_cols > 1:
            col_spec = 'l'  + 'r' * (n_cols - 1)
        else:
            col_spec = 'l'

        # Use longtable instead of tabular to allow page breaks
        latex = f'\\begin{{longtable}}{{{col_spec}}}\n'
        latex += '\\hline\n'
        latex += ' & '.join(headers) + ' \\\\\n'
        latex += '\\hline\n'
        latex += '\\endfirsthead\n'  # End of first page header
        latex += '\\multicolumn{' + str(n_cols) + '}{c}{\\textit{(continua dalla pagina precedente)}} \\\\\n'
        latex += '\\hline\n'
        latex += ' & '.join(headers) + ' \\\\\n'
        latex += '\\hline\n'
        latex += '\\endhead\n'  # Header for subsequent pages
        latex += '\\hline\n'
        latex += '\\multicolumn{' + str(n_cols) + '}{r}{\\textit{(continua alla pagina successiva)}} \\\\\n'
        latex += '\\endfoot\n'  # Footer for all pages except last
        latex += '\\hline\n'
        latex += '\\endlastfoot\n'  # Footer for last page
        for row in rows:
            latex += ' & '.join(row) + ' \\\\\n'
        latex += '\\end{longtable}\n'
        return latex


# =============================================================================
# VOLUME CALCULATION (Tabacchi equations)
# =============================================================================

# Lower-triangular covariance matrices for volume equation coefficients (by species).
TABACCHI_COV = {
    'Abete': np.array([
        [  4.9584,     0,         0 ],
        [  1.1274e-3,  7.6175e-7, 0 ],
        [ -7.1820e-1, -2.0243e-4, 1.1287e-1 ],
    ]),
    'Acero': np.array([
        [  9.8852e-1,  0],
        [ -4.7366e-4,  8.4075e-7]
    ]),
    'Castagno': np.array([
        [  6.5052,     0,         0 ],
        [  2.0090e-3,  1.2430e-6, 0 ],
        [ -1.0771,    -3.9067e-4, 1.9110e-1 ]
    ]),
    'Cerro': np.array([
        [  4.5573e-1, 0 ],
        [ -1.8540e-4, 3.6935e-7 ]
    ]),
    'Ciliegio': np.array([
        [  1.5377e1,   0,         0 ],
        [  8.9101e-3,  9.7080e-6, 0 ],
        [ -2.6997,    -1.8132e-3, 4.9690e-1 ]
    ]),
    'Douglas': np.array([
        [  6.2135e1,   0,         0 ],
        [  6.9406e-3,  1.2592e-6, 0 ],
        [ -6.8517,    -8.2763e-4, 7.7268e-1 ]
    ]),
    'Faggio': np.array([
        [  1.2573,    0 ],
        [ -3.2331e-4, 6.4872e-7 ]
    ]),
    'Leccio': np.array([
        [  8.9968,     0,         0 ],
        [  4.6303e-3,  3.9302e-6, 0 ],
        [ -1.6058,    -9.3376e-4, 3.0078e-1 ]
    ]),
    'Ontano': np.array([
        [  4.9867e1,   0,         0 ],
        [  1.3116e-2,  5.3498e-6, 0 ],
        [ -7.1964,    -2.0513e-3, 1.0716 ]
    ]),
    'Pino': np.array([
        [  3.2482,    0 ],
        [ -7.5710e-4, 3.0428e-7 ]
    ]),
    'Pino Laricio': np.array([
        [  3.2482,    0 ],
        [ -7.5710e-4, 3.0428e-7 ]
    ]),
    'Pino Marittimo': np.array([
        [  2.6524e-1, 0 ],
        [ -1.2270e-4, 5.9640e-7 ]
    ]),
    'Pino Nero': np.array([
        [  2.9797e1,   0,         0 ],
        [  4.5880e-3,  1.3001e-6, 0 ],
        [ -3.0604,    -5.4676e-4, 3.3202e-1 ]
    ]),
    'Sorbo': np.array([
        [  1.5377e1,   0,         0 ],
        [  8.9101e-3,  9.7080e-6, 0 ],
        [ -2.6997,    -1.8132e-3, 4.9690e-1 ]
    ]),
}

# Volume equation coefficients (as row vectors).
TABACCHI_B = {
    'Abete':          np.array([ -1.8381,    3.7836e-2, 3.9934e-1 ]),
    'Acero':          np.array([  1.6905,    3.7082e-2 ]),
    'Castagno':       np.array([ -2,         3.6524e-2, 7.4466e-1 ]),
    'Cerro':          np.array([ -4.3221e-2, 3.8079e-2 ]),
    'Ciliegio':       np.array([  2.3118,    3.1278e-2, 3.7159e-1 ]),
    'Douglas':        np.array([ -7.9946,    3.3343e-2, 1.2186 ]),
    'Faggio':         np.array([  8.1151e-1, 3.8965e-2 ]),
    'Leccio':         np.array([ -2.2219,    3.9685e-2, 6.2762e-1 ]),
    'Ontano':         np.array([ -2.2932e1,  3.2641e-2, 2.991 ]),
    'Pino':           np.array([  6.4383,    3.8594e-2 ]),
    'Pino Laricio':   np.array([  6.4383,    3.8594e-2 ]),
    'Pino Marittimo': np.array([  2.9963,    3.8302e-2 ]),
    'Pino Nero':      np.array([ -2.1480e1,  3.3448e-2, 2.9088]),
    'Sorbo':          np.array([  2.3118,    3.1278e-2, 3.7159e-1 ]),
}

# Degrees of freedom for t-statistic (n - 2)
TABACCHI_NS = {
    'Abete': 46,
    'Acero': 37,
    'Castagno': 85,
    'Cerro': 88,
    'Ciliegio': 22,
    'Douglas': 35,
    'Faggio': 91,
    'Leccio': 83,
    'Ontano': 35,
    'Pino': 50,
    'Pino Laricio': 50,
    'Pino Marittimo': 26,
    'Pino Nero': 63,
    'Sorbo': 22,
}

# Residual variance (s²)
TABACCHI_S2 = {
    'Abete': 1.5284e-5,
    'Acero': 2.2710e-5,
    'Castagno': 3.0491e-5,
    'Cerro': 2.5866e-5,
    'Ciliegio': 4.0506e-5,
    'Douglas': 9.0103e-6,
    'Faggio': 5.1468e-5,
    'Leccio': 6.0915e-5,
    'Ontano': 3.9958e-5,
    'Pino': 6.3906e-6,
    'Pino Laricio': 6.3906e-6,
    'Pino Marittimo': 1.4031e-5,
    'Pino Nero': 1.7090e-5,
    'Sorbo': 4.0506e-5,
}

# Make covariance matrices symmetric
def make_symmetric():
    """Make covariance matrices symmetric."""
    # pylint: disable=consider-using-dict-items
    for genere in TABACCHI_COV:
        m = TABACCHI_COV[genere]
        TABACCHI_COV[genere] = m + m.T - np.diag(np.diag(m))
    # pylint: enable=consider-using-dict-items
make_symmetric()


def calculate_tree_volume(diameter: float, height: float, genere: str) -> float:
    """
    Calculate volume for a single tree using Tabacchi equations.

    Args:
        diameter: Diameter in cm (D)
        height: Height in m (h)
        genere: Species name

    Returns:
        Volume in m³

    Raises:
        ValueError: If genere is not in Tabacchi tables
    """
    if genere not in TABACCHI_B:
        raise ValueError(f"Genere '{genere}' non trovato nelle tavole di Tabacchi")

    b = TABACCHI_B[genere]

    # Volume equation: V = b0 + b1 * D² * h [+ b2 * D]
    d2h = (diameter ** 2) * height
    if len(b) == 2:
        volume = (b[0] + b[1] * d2h) / 1000  # Convert to m³
    else:  # len(b) == 3
        volume = (b[0] + b[1] * d2h + b[2] * diameter) / 1000

    return volume


def calculate_volume_confidence_interval(trees_df: pd.DataFrame) -> tuple[float, float]:
    """
    Calculate confidence interval margin for a group of trees using Tabacchi equations.

    Conservative approach: assumes perfect correlation between species.
    - For each species: margin_i = t_i * sqrt(v0_i + v1_i)
    - Total margin = sum(margin_i)

    Where:
    - V0: variance from coefficient uncertainty (D1 @ cov @ D1.T)
    - V1: residual variance (sum of s² * (D²h)²)

    Args:
        trees_df: DataFrame with columns D(cm), h(m), Genere, V(m3)

    Returns:
        Tuple of (total_volume, margin_of_error) in m³

    Raises:
        ValueError: If any genere not in Tabacchi tables
    """
    total_volume = 0.0
    total_margin = 0.0  # Sum margins (conservative: assumes perfect correlation)

    for genere, group in trees_df.groupby('Genere'):
        if genere not in TABACCHI_B:
            raise ValueError(f"Genere '{genere}' non presente in Tabacchi")

        n_trees = len(group)
        b = TABACCHI_B[genere]
        cov = TABACCHI_COV[genere]
        s2 = TABACCHI_S2[genere]
        df = TABACCHI_NS[genere] - 2

        # Build D0 matrix (n_trees x n_coefficients)
        d0 = np.zeros((n_trees, len(b)))
        d0[:, 0] = 1
        d0[:, 1] = (group['D(cm)'].values ** 2) * group['h(m)'].values
        if len(b) == 3:
            d0[:, 2] = group['D(cm)'].values

        # D1 = sum of rows of D0 (1 x n_coefficients)
        d1 = np.sum(d0, axis=0).reshape(1, -1)

        # V0: Coefficient uncertainty variance
        v0 = (d1 @ cov @ d1.T)[0, 0]

        # V1: Residual variance
        d2h = d0[:, 1]  # D²h values for each tree
        v1 = np.sum(d2h ** 2) * s2

        # Species-specific t-statistic
        t_stat = stats.t.ppf(1 - 0.05/2, df)

        # Species-specific margin
        margin_species = t_stat * np.sqrt(v0 + v1) / 1000  # Convert to m³

        # Accumulate
        total_volume += group['V(m3)'].sum()
        total_margin += margin_species

    return total_volume, total_margin


def apply_volume_equations(alberi_file: str, output_file: str) -> None:
    """
    Calculate volumes for all trees in the input file.

    Args:
        alberi_file: Input CSV with tree data (requires D(cm), h(m), Genere columns)
        output_file: Output CSV with added V(m3) column
    """
    print("Calcolo volumi con equazioni di Tabacchi")
    print(f"Input: {alberi_file}")
    print(f"Output: {output_file}")

    trees_df = pd.read_csv(alberi_file)
    required = ['D(cm)', 'h(m)', 'Genere']
    missing = [col for col in required if col not in trees_df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel file di input: {missing}")

    trees_df['V(m3)'] = 0.0

    # Calculate volumes for each tree
    calculated = 0

    for idx, row in trees_df.iterrows():
        genere = row['Genere']
        diameter = row['D(cm)']
        height = row['h(m)']

        if pd.isna(diameter) or pd.isna(height):
            raise ValueError(f"Dati mancanti per riga {idx}: Diametro={diameter}, Altezza={height}")
        if genere not in TABACCHI_B:
            raise ValueError(f"Genere '{genere}' non trovato nelle tavole di Tabacchi")

        volume = calculate_tree_volume(diameter, height, genere)
        trees_df.at[idx, 'V(m3)'] = volume
        calculated += 1

    trees_df.to_csv(output_file, index=False, float_format="%.6f")

    print("\nRiepilogo:")
    print(f"  Alberi totali: {len(trees_df)}")
    print(f"  Volumi calcolati: {calculated}")

    total_volume = trees_df['V(m3)'].sum()
    print(f"\n  Volume totale campionato: {total_volume:.2f} m³")


# =============================================================================
# DATA PREPARATION (pure data, no rendering)
# =============================================================================

def region_data(trees_df: pd.DataFrame, particelle_df: pd.DataFrame,
                compresa: Optional[str] = None,
                particella: Optional[str] = None,
                genere: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
    """
    Compute region data (shared across all directives).

    Args:
        trees_df: Full tree database
        particelle_df: Parcel metadata
        compresa: Optional compresa name (None means all comprese)
        particella: Optional particella name (requires compresa)
        genere: Optional species name (None means all species)

    Returns:
        Tuple of (filtered_trees, metrics ... ) where metrics include:
            - 'area_ha': Total area in hectares
            - 'sample_areas': Number of sample areas

    Raises:
        ValueError: If particella specified without compresa, or no data found
    """
    if particella and not compresa:
        raise ValueError("particella richiede compresa")

    trees = trees_df.copy()

    if compresa:
        trees = trees[trees['Compresa'] == compresa]
    if particella:
        trees = trees[trees['Particella'] == particella]
    if genere:
        trees = trees[trees['Genere'] == genere]
    if len(trees) == 0:
        raise ValueError(f"Nessun dato trovato per compresa '{compresa}' " +
                         f"particella '{particella}' genere '{genere}'")

    parcels = particelle_df.copy()
    if compresa:
        parcels = parcels[parcels['Compresa'] == compresa]
    if particella:
        parcels = parcels[parcels['Particella'] == particella]

    area_ha = parcels['Area (ha)'].sum()
    # "Area saggio" labels are not unique at least across comprese.
    n_sample_areas = trees.drop_duplicates(
        subset=['Compresa', 'Particella', 'Area saggio']).shape[0]

    n_trees_sampled = len(trees)
    n_trees_per_ha_by_diameter = n_trees_sampled * SAMPLE_AREAS_PER_HA / n_sample_areas
    n_trees_total = round(n_trees_per_ha_by_diameter * area_ha)

    # Calculate max significant diameter class (for x-axis scaling)
    max_significant_diameter_class = 0
    if len(trees) > 0 and n_sample_areas > 0:
        n_trees_by_diameter = trees.groupby('Classe diametrica').size()
        n_trees_per_ha_by_diameter = n_trees_by_diameter * SAMPLE_AREAS_PER_HA / n_sample_areas
        common_diameters = n_trees_per_ha_by_diameter[n_trees_per_ha_by_diameter >= MIN_TREES_PER_HA]
        if len(common_diameters) > 0:
            max_significant_diameter_class = int(common_diameters.index.max())

    return {
        'compresa': compresa,
        'particella': particella,
        'trees': trees,
        'species_list': sorted(trees['Genere'].unique()),
        'area_ha': area_ha,
        'sample_areas': n_sample_areas,
        'sampled_trees': n_trees_sampled,
        'ntrees_total': n_trees_total,
        'ntrees_per_ha': n_trees_per_ha_by_diameter, # Not used
        'mean_diameter_class': trees['Classe diametrica'].mean(),
        'mean_height': trees['h(m)'].mean(),
        'max_significant_diameter_class': max_significant_diameter_class,
    }


# =============================================================================
# COMPUTATION LAYER (equation generation and application)
# =============================================================================

def fit_curves_grouped(groups: Iterable[tuple[tuple[str, str], pd.DataFrame]],
                       funzione: str, min_points: int = 10) -> pd.DataFrame:
    """
    Fit regression curves to grouped data.

    Args:
        groups: Iterable of ((compresa, genere), DataFrame) tuples
        funzione: 'log' or 'lin'
        min_points: Minimum number of points required for fitting

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    RegressionClass = LogarithmicRegression if funzione == 'log' else LinearRegression
    func_name = 'ln' if funzione == 'log' else 'lin'

    results = []
    for (compresa, genere), group_df in groups:
        regr = RegressionClass()
        if regr.fit(group_df['x'].values, group_df['y'].values, min_points=min_points):
            results.append({
                'compresa': compresa,
                'genere': genere,
                'funzione': func_name,
                'a': regr.a,
                'b': regr.b,
                'r2': regr.r2,
                'n': regr.n_points
            })
            print(f"  {compresa} - {genere}: {regr} (R² = {regr.r2:.2f}, n = {regr.n_points})")
        else:
            print(f"  {compresa} - {genere}: dati insufficienti (n < {min_points})")

    return pd.DataFrame(results)


def fit_curves_from_ipsometro(ipsometro_file: str, funzione: str = 'log') -> pd.DataFrame:
    """
    Generate equations from ipsometer field measurements.

    Args:
        ipsometro_file: CSV with columns [Compresa, Genere, Diametro, Altezza]
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    df = pd.read_csv(ipsometro_file)
    df['x'] = df['Diametro']
    df['y'] = df['Altezza']
    groups = []

    for (compresa, specie), group in df.groupby(['Compresa', 'Specie']):
        groups.append(((compresa, specie), group))
    return fit_curves_grouped(groups, funzione)


def fit_curves_from_originali(alberi_file: str, funzione: str = 'log') -> pd.DataFrame:
    """
    Generate equations from original tree database heights.

    Args:
        alberi_file: CSV with tree data
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    df = pd.read_csv(alberi_file)
    df = df[df['Fustaia'] == True].copy()
    df['x'] = df['D(cm)']
    df['y'] = df['h(m)']

    groups = [(key, group) for key, group in df.groupby(['Compresa', 'Genere'])]
    return fit_curves_grouped(groups, funzione)


def fit_curves_from_tabelle(tabelle_file: str, particelle_file: str,
                            funzione: str = 'log') -> pd.DataFrame:
    """
    Generate equations from alsometric tables, replicated for each compresa.

    Args:
        tabelle_file: CSV with alsometric data [Genere, Diam 130cm, Altezza indicativa]
        particelle_file: CSV to discover which comprese exist
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    df_particelle = pd.read_csv(particelle_file)
    comprese = sorted(df_particelle['Compresa'].dropna().unique())

    df_als = pd.read_csv(tabelle_file)
    df_als['Diam 130cm'] = pd.to_numeric(df_als['Diam 130cm'], errors='coerce')
    df_als['Altezza indicativa'] = pd.to_numeric(df_als['Altezza indicativa'], errors='coerce')
    df_als['x'] = df_als['Diam 130cm']
    df_als['y'] = df_als['Altezza indicativa']

    groups = []
    for compresa in comprese:
        for genere, group in df_als.groupby('Genere'):
            groups.append(((compresa, genere), group))
    return fit_curves_grouped(groups, funzione)

#pylint: disable=too-many-locals
def apply_height_equations(alberi_file: str, equations_file: str,
                           output_file: str) -> None:
    """
    Apply height equations to tree database, updating heights.

    Args:
        alberi_file: Input tree CSV
        equations_file: CSV with equations [compresa, genere, funzione, a, b, r2, n]
        output_file: Output tree CSV with updated heights
    """
    trees_df = pd.read_csv(alberi_file)
    equations_df = pd.read_csv(equations_file)

    total_trees = len(trees_df)
    trees_updated = 0
    trees_unchanged = 0

    print(f"Applicazione equazioni a {total_trees} alberi...")

    output_df = trees_df.copy()
    output_df['h(m)'] = output_df['h(m)'].astype(float)

    for (compresa, genere), group in trees_df.groupby(['Compresa', 'Genere']):
        eq_row = equations_df[
            (equations_df['compresa'] == compresa) &
            (equations_df['genere'] == genere)
        ]

        if len(eq_row) == 0:
            print(f"  {compresa} - {genere}: nessuna equazione trovata; altezze immutate")
            trees_unchanged += len(group)
            continue

        eq = eq_row.iloc[0]

        indices = group.index
        diameters = trees_df.loc[indices, 'D(cm)'].astype(float)

        if eq['funzione'] == 'ln':
            new_heights = eq['a'] * np.log(np.maximum(diameters, 0.1)) + eq['b']
        else:  # 'lin'
            new_heights = eq['a'] * diameters + eq['b']

        output_df.loc[indices, 'h(m)'] = new_heights.astype(float)
        trees_updated += len(group)

        print(f"  {compresa} - {genere}: {len(group)} alberi aggiornati")

    output_df.to_csv(output_file, index=False, float_format="%.3f")

    print("\nRiepilogo:")
    print(f"  Totale alberi: {total_trees}")
    print(f"  Alberi aggiornati: {trees_updated}")
    print(f"  Alberi non modificati: {trees_unchanged}")
    print(f"File salvato: {output_file}")
#pylint: enable=too-many-locals

# =============================================================================
# RENDERING AND TEMPLATE PROCESSING
# =============================================================================

def render_ci_graph(data: dict, max_diameter: int, equations_df: pd.DataFrame,
                    output_path: Path, formatter: SnippetFormatter,
                    color_map: dict) -> dict:
    """
    Generate curve ipsometriche (height-diameter) graph.

    Args:
        data: Output from region_data()
        equations_df: Pre-computed equations from CSV
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    trees = data['trees']
    species_list = data['species_list']
    compresa = data['compresa']

    fig, ax = plt.subplots(figsize=(4, 3))
    curve_info = []
    ymax = 0

    for species in species_list:
        sp_data = trees[trees['Genere'] == species]
        x = sp_data['D(cm)'].values
        y = sp_data['h(m)'].values
        ymax = max(ymax, y.max())

        ax.scatter(x, y, color=color_map[species], label=species, alpha=0.7, linewidth=2, s=1)

        # Look up pre-computed equation from equations.csv
        eq_row = equations_df[
            (equations_df['compresa'] == compresa) &
            (equations_df['genere'] == species)
        ]

        if len(eq_row) > 0:
            eq = eq_row.iloc[0]

            x_min, x_max = 1, x.max()
            x_smooth = np.linspace(x_min, x_max, 100)

            if eq['funzione'] == 'ln':
                y_smooth = eq['a'] * np.log(x_smooth) + eq['b']
                eq_str = f"y = {eq['a']:.2f}*ln(x) + {eq['b']:.2f}"
            else:  # 'lin'
                y_smooth = eq['a'] * x_smooth + eq['b']
                eq_str = f"y = {eq['a']:.2f}*x + {eq['b']:.2f}"

            ax.plot(x_smooth, y_smooth, color=color_map[species],
                   linestyle='--', alpha=0.8, linewidth=1.5)

            # Save curve info for metadata display
            curve_info.append({
                'species': species,
                'equation': eq_str,
                'r_squared': eq['r2'],
                'n_points': int(eq['n'])
            })

    ax.set_xlabel('Diametro (cm)', fontweight='bold')
    ax.set_ylabel('Altezza (m)', fontweight='bold')
    ax.set_xlim(-0.5, max_diameter + 3)
    ax.set_xticks(range(0, max_diameter + 1, (max_diameter + 1)//10))
    ax.set_ylim(0, (ymax + 6)//5*5)
    td = min(ax.get_ylim()[1] // 5, 4)
    y_ticks = np.arange(0, ax.get_ylim()[1] + 1, td)
    ax.set_yticks(y_ticks)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    snippet = formatter.format_image(output_path)
    snippet += '\n' + formatter.format_metadata(data, curve_info=curve_info)

    return {
        'filepath': output_path,
        'snippet': snippet
    }


def render_cd_graph(data: dict, max_diameter_class: int, output_path: Path,
                    formatter: SnippetFormatter, color_map: dict) -> dict:
    """
    Generate classi diametriche (diameter class histogram) graph.

    Args:
        data: Output from region_data()
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    trees = data['trees']
    species_list = data['species_list']
    sample_areas = data['sample_areas']

    counts = (trees.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
              * SAMPLE_AREAS_PER_HA / sample_areas)

    fig, ax = plt.subplots(figsize=(4, 3.75))

    bottom = np.zeros(len(counts.index))
    for species in species_list:
        if species not in counts.columns:
            continue
        values = counts[species].values
        ax.bar(counts.index, values, bottom=bottom,
               label=species, color=color_map[species],
               alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Stima alberi / ha', fontweight='bold')
    ax.set_xlim(-0.5, max_diameter_class + 0.5)
    ax.set_xticks(range(0, max_diameter_class + 1, 2))
    ax.set_ylim(0, counts.sum(axis=1).max() * 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Reverse legend order to match visual stacking order (top-to-bottom)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),
              title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    snippet = formatter.format_image(output_path)
    snippet += '\n' + formatter.format_metadata(data)

    return {
        'filepath': output_path,
        'snippet': snippet
    }


PART_PATTERN = re.compile(r'^(\d+)([a-zA-Z]*)$')
def natural_sort_key(value: str) -> tuple:
    """
    Generate a sort key for natural (alphanumeric) sorting.

    Handles particelle like "2a", "10b", etc. so that:
    - Numeric parts sort numerically (9 < 10)
    - Alphabetic suffixes sort alphabetically (2a < 2b)

    Args:
        value: String to generate sort key for

    Returns:
        Tuple of (numeric_part, alphabetic_part) for sorting
    """
    match = PART_PATTERN.match(value)
    if match:
        numeric = int(match.group(1))
        alpha = match.group(2)
        return (numeric, alpha)
    return (float('inf'), str(value))


def render_tsv_table(data: dict, particelle_df: pd.DataFrame,
                     formatter: SnippetFormatter,
                     **options: dict) -> dict:
    """
    Generate volume summary table (@@tsv directive).

    Args:
        data: Output from region_data()
        particelle_df: Parcel metadata
        formatter: HTML or LaTeX snippet formatter
        options: Dictionary of options:
            - per_compresa: If True, show rows per compresa; if False, aggregate (default: si)
            - per_particella: If True, show rows per particella; if False, aggregate (default: si)
            - per_genere: If True, show rows per genere; if False, aggregate (default: si)
            - stime_totali: If True, scale to total volume; if False, show sampled volume
            - intervallo_fiduciario: If True, include confidence interval columns
            - totali: If True, add totals row at bottom

    Returns:
        dict with 'snippet' key containing formatted table
    """
    trees = data['trees']
    compresa = data['compresa']
    particella = data['particella']
    genere = data.get('genere')  # May not be in data dict if not filtered

    # Determine grouping columns based on per_* flags and filters
    # If a dimension is already filtered (specific value provided), don't group by it
    # Otherwise, use the per_* flag to decide
    group_cols = []

    if not compresa and options.get('per_compresa', True):
        group_cols.append('Compresa')
    if not particella and options.get('per_particella', True):
        group_cols.append('Particella')
    if not genere and options.get('per_genere', True):
        group_cols.append('Genere')

    # If no grouping columns, aggregate everything into one row
    if not group_cols:
        group_cols = ['_dummy']  # Dummy column for single aggregation
        trees = trees.copy()
        trees['_dummy'] = 'Totale'

    table_rows = []
    for group_key, group_df in trees.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_key))

        n_trees = len(group_df)
        sampled_volume = group_df['V(m3)'].sum()

        if options['stime_totali']:
            group_compresa = row_dict.get('Compresa', compresa)
            group_particella = row_dict.get('Particella', particella)

            if group_particella:
                region_particelle = particelle_df[
                    (particelle_df['Compresa'] == group_compresa) &
                    (particelle_df['Particella'] == group_particella)
                ]
            else:
                region_particelle = particelle_df[particelle_df['Compresa'] == group_compresa]

            area_ha = region_particelle['Area (ha)'].sum()
            sample_areas = len(group_df['Area saggio'].unique())

            volume = sampled_volume * area_ha * SAMPLE_AREAS_PER_HA / sample_areas
            n_trees_display = int(n_trees * area_ha * SAMPLE_AREAS_PER_HA / sample_areas)
        else:
            volume = sampled_volume
            n_trees_display = n_trees

        row_dict['N_alberi'] = n_trees_display
        row_dict['Volume'] = volume

        # Compute confidence intervals if requested
        if options['intervallo_fiduciario']:
            # Use full confidence interval calculation (V0 + V1 + t-statistic)
            _, margin_sampled = calculate_volume_confidence_interval(group_df)

            # Scale margin if stime_totali
            if options['stime_totali']:
                margin = margin_sampled * area_ha * SAMPLE_AREAS_PER_HA / sample_areas
            else:
                margin = margin_sampled

            row_dict['IF_inf'] = volume - margin
            row_dict['IF_sup'] = volume + margin

        table_rows.append(row_dict)

    # Sort rows
    if len(table_rows) == 0:
        raise ValueError("@@tsv: Nessun dato da visualizzare")

    def sort_key(row):
        """Generate sort key with natural sorting for Particella column."""
        key_parts = []
        for col in group_cols:
            value = row.get(col, '')
            if col == 'Particella':
                key_parts.append(natural_sort_key(value))
            else:
                key_parts.append((0, str(value)))  # Regular string sort for other columns
        return tuple(key_parts)

    table_rows = sorted(table_rows, key=sort_key)

    # Build table headers
    headers = []
    show_compresa = 'Compresa' in group_cols
    show_particella = 'Particella' in group_cols
    show_genere = 'Genere' in group_cols
    show_dummy = '_dummy' in group_cols

    if show_compresa:
        headers.append('Compresa')
    if show_particella:
        headers.append('Particella')
    if show_genere:
        headers.append('Genere')

    # Always show value columns
    headers.append('N. Alberi')
    headers.append('Volume (m³)')
    if options['intervallo_fiduciario']:
        headers.append('IF inf (m³)')
        headers.append('IF sup (m³)')

    # Build table data rows
    data_rows = []
    total_trees = 0
    total_volume = 0.0
    total_if_inf = 0.0
    total_if_sup = 0.0

    for row_dict in table_rows:
        row = []
        if show_compresa:
            row.append(str(row_dict.get('Compresa', '')))
        if show_particella:
            row.append(str(row_dict.get('Particella', '')))
        if show_genere:
            row.append(str(row_dict.get('Genere', '')))
        # Skip _dummy column in output
        row.append(f"{row_dict['N_alberi']}")
        row.append(f"{row_dict['Volume']:.2f}")
        if options['intervallo_fiduciario']:
            row.append(f"{row_dict['IF_inf']:.2f}")
            row.append(f"{row_dict['IF_sup']:.2f}")

        data_rows.append(row)

        # Accumulate totals
        total_trees += row_dict['N_alberi']
        total_volume += row_dict['Volume']
        if options['intervallo_fiduciario']:
            total_if_inf += row_dict['IF_inf']
            total_if_sup += row_dict['IF_sup']

    # Add totals row if requested (skip if dummy aggregation, which is already a total)
    if options['totali'] and not show_dummy:
        total_row = []
        # Fill in blank cells for grouping columns (except the last one where we put 'Totale')
        n_group_cols = ((1 if show_compresa else 0) +
                       (1 if show_particella else 0) +
                       (1 if show_genere else 0))

        # Add blank cells for all but the last grouping column
        for _ in range(n_group_cols - 1):
            total_row.append('')
        # Put 'Totale' in the last grouping column (or first column if no grouping)
        if n_group_cols > 0:
            total_row.append('Totale')
        total_row.append(f"{total_trees}")
        total_row.append(f"{total_volume:.2f}")
        if options['intervallo_fiduciario']:
            total_row.append(f"{total_if_inf:.2f}")
            total_row.append(f"{total_if_sup:.2f}")
        data_rows.append(total_row)

    # Format table
    snippet = formatter.format_table(headers, data_rows)

    return {'snippet': snippet}


def parse_template_directive(line: str) -> Optional[dict]:
    """
    Parse a template directive like @@ci(compresa=Serra, genere=Abete).

    Returns:
        dict with keys: 'keyword', 'params', 'full_text'
        or None if not a valid directive
    """
    # Match pattern: @@keyword(param=value, param=value, ...)
    pattern = r'@@(\w+)\((.*?)\)'
    match = re.search(pattern, line)

    if not match:
        return None

    keyword = match.group(1)
    params_str = match.group(2)
    full_text = match.group(0)

    params = {}
    if params_str.strip():
        # Split by comma and parse key=value pairs
        for param in params_str.split(','):
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()

    return {
        'keyword': keyword,
        'params': params,
        'full_text': full_text
    }

DIRECTIVE_PATTERN = re.compile(r'@@(\w+)\((.*?)\)')
def process_template(template_text: str, trees_df: pd.DataFrame,
                     particelle_df: pd.DataFrame, equations_df: pd.DataFrame,
                     output_dir: Path, format_type: str) -> str:
    """
    Process template by substituting @@directives with generated content.

    Args:
        template_text: Input template
        trees_df: Tree database
        particelle_df: Parcel metadata
        equations_df: Pre-computed equations
        output_dir: Where to save generated graphs
        format_type: 'html' or 'latex'

    Returns:
        Processed template text
    """
    # Track filenames to make duplicates unique
    filename_counts = {}

    def build_graph_filename(compresa: Optional[str], particella: Optional[str],
                             genere: Optional[str], keyword: str) -> str:
        """Build a filename for a graph based on the parameters."""
        parts = []
        if compresa:
            parts.append(compresa)
        else:
            parts.append('tutte')
        if particella:
            parts.append(particella)
        if genere:
            parts.append(genere)
        parts.append(keyword)

        base_name = '_'.join(parts)

        # Make duplicates unique with a counter
        if base_name in filename_counts:
            filename_counts[base_name] += 1
            return f"{base_name}_{filename_counts[base_name]}.png"
        else:
            filename_counts[base_name] = 1
            return base_name + '.png'

    def process_directive(match):
        directive = parse_template_directive(match.group(0))
        if not directive:
            return match.group(0)  # Return unchanged if parsing fails

        keyword = directive['keyword']
        params = directive['params']
        compresa = params.get('compresa', None)
        particella = params.get('particella', None)
        genere = params.get('genere', None)

        try:
            cache_key = (compresa, particella, genere)
            data = data_cache[cache_key]

            match keyword:
                case 'tsv':
                    options = {
                        'per_compresa': params.get('per_compresa', 'si').lower() == 'si',
                        'per_particella': params.get('per_particella', 'si').lower() == 'si',
                        'per_genere': params.get('per_genere', 'si').lower() == 'si',
                        'stime_totali': params.get('stime_totali', 'no').lower() == 'si',
                        'intervallo_fiduciario':
                            params.get('intervallo_fiduciario', 'no').lower() == 'si',
                        'totali': params.get('totali', 'no').lower() == 'si'
                    }
                    result = render_tsv_table(data, particelle_df, formatter, **options)
                case 'cd':
                    filename = build_graph_filename(compresa, particella, genere, keyword)
                    result = render_cd_graph(data, max_diameter_class,
                                             output_dir / filename, formatter, color_map)
                case 'ci':
                    filename = build_graph_filename(compresa, particella, genere, keyword)
                    result = render_ci_graph(data, max_diameter, equations_df,
                                             output_dir / filename, formatter, color_map)
                case _:
                    raise ValueError(f"Comando sconosciuto: {keyword}")

            return result['snippet']

        except Exception as e:
            raise ValueError(f"ERRORE nella generazione di {directive['full_text']}: {e}") from e

    formatter = HTMLSnippetFormatter() if format_type == 'html' else LaTeXSnippetFormatter()
    color_map = get_color_map()
    max_diameter = trees_df['D(cm)'].max()

    # First pass: parse all directives and pre-compute and cache region_data
    data_cache = {}  # Cache region_data results by (compresa, particella, genere)

    for match in DIRECTIVE_PATTERN.finditer(template_text):
        keyword, params_str = match.groups()
        params = dict(re.findall(r'(\w+)=([^,]+)', params_str))
        compresa = params.get('compresa')
        particella = params.get('particella')
        genere = params.get('genere')

        cache_key = (compresa, particella, genere)
        if cache_key not in data_cache:
            data_cache[cache_key] = region_data(trees_df, particelle_df,
                                                compresa, particella, genere)

    # Compute global max diameter class from all cached datasets
    max_diameter_class = 0
    for data in data_cache.values():
        max_diameter_class = max(max_diameter_class, data['max_significant_diameter_class'])
    if max_diameter_class == 0:
        max_diameter_class = int(trees_df['Classe diametrica'].max())

    # Find and replace all directives
    processed = re.sub(DIRECTIVE_PATTERN, process_directive, template_text)

    return processed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_parcels(particelle_file: str) -> None:
    """
    List all (compresa, particella) pairs from particelle file.

    Args:
        particelle_file: CSV with parcel data
    """
    df = pd.read_csv(particelle_file)

    # Filter out rows with missing Compresa or Particella
    df = df.dropna(subset=['Compresa', 'Particella'])

    # Group by Compresa and list particelle
    for compresa in sorted(df['Compresa'].unique()):
        compresa_data = df[df['Compresa'] == compresa]
        particelle = sorted(compresa_data['Particella'].astype(str).unique(),
                          key=natural_sort_key)
        for particella in particelle:
            print(f"  {compresa},{particella}")


def get_color_map() -> dict:
    """
    Create consistent color mapping for species.

    Returns:
        Dict mapping species -> matplotlib color (as RGBA tuple)
    """
    # Manual color mapping for the 14 Tabacchi species
    # Organized by type: deciduous (yellows/greens), conifers (blues), special (pinks/reds)
    color_palette = {
        # Deciduous broadleaves (yellow-green spectrum)
        'Faggio': '#F4F269',          # canary-yellow
        'Castagno': '#CEE26B',        # lime-cream
        'Acero': '#A8D26D',           # willow-green
        'Cerro': '#82C26E',           # moss-green
        'Ontano': '#5CB270',          # emerald
        'Leccio': '#4DA368',          # emerald-dark (adjusted for distinction)

        # Conifers (blue-aqua spectrum)
        'Abete': '#0c63e7',           # royal-blue - firs
        'Douglas': '#07c8f9',         # sky-aqua
        'Pino': '#09a6f3',            # fresh-sky - common pines
        'Pino Nero': '#0a85ed',       # brilliant-azure
        'Pino Laricio': '#0a85ed',    # brilliant-azure

        # Special cases (coral/pink spectrum)
        'Pino Marittimo': '#FB6363',  # vibrant-coral
        'Ciliegio': '#DC4E5E',        # lobster-pink - cherry
        'Sorbo': '#BE385A',           # rosewood - rowan
    }

    # Convert hex to RGBA tuples for matplotlib
    def hex_to_rgba(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 8:  # RRGGBBAA
            r, g, b, a = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4, 6)]
        else:  # RRGGBB
            r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
            a = 1.0
        return (r, g, b, a)

    return {species: hex_to_rgba(color) for species, color in color_palette.items()}


# =============================================================================
# MAIN AND ARGUMENT PARSING
# =============================================================================

def run_genera_equazioni(args):
    """Execute Mode 1: Generate equations."""
    print(f"Generazione equazioni da fonte: {args.fonte_altezze}")
    print(f"Funzione: {args.funzione}")

    if args.fonte_altezze == 'ipsometro':
        equations_df = fit_curves_from_ipsometro(args.input, args.funzione)
    elif args.fonte_altezze == 'originali':
        equations_df = fit_curves_from_originali(args.input, args.funzione)
    elif args.fonte_altezze == 'tabelle':
        equations_df = fit_curves_from_tabelle(args.input, args.particelle, args.funzione)
    else:
        raise ValueError(f"Fonte altezze non supportata: {args.fonte_altezze}")

    if equations_df is not None:
        equations_df.to_csv(args.output, index=False, float_format="%.4f")
        print(f"Equazioni salvate in: {args.output}")
        print(f"Totale equazioni generate: {len(equations_df)}")
    else:
        print("ERRORE: Nessuna equazione generata (funzioni stub non implementate)")


def run_calcola_altezze(args):
    """Calculate heights."""
    print(f"Calcolo altezze usando equazioni da: {args.equazioni}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    apply_height_equations(args.input, args.equazioni, args.output)
    print("Altezze calcolate con successo")


def run_calcola_volumi(args):
    """Calculate volumes."""
    print("Calcolo volumi usando tavole di Tabacchi")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    apply_volume_equations(args.input, args.output)


def run_report(args):
    """Generate report from template."""
    format_type = args.formato
    print(f"Generazione report formato: {format_type}")
    print(f"Template: {args.input}")
    print(f"Output directory: {args.output_dir}")

    trees_df = pd.read_csv(args.alberi)
    trees_df = trees_df[trees_df['Fustaia'] == True].copy()  # Filter to fustaia only
    particelle_df = pd.read_csv(args.particelle)
    equations_df = pd.read_csv(args.equazioni)

    with open(args.input, 'r', encoding='utf-8') as f:
        template_text = f.read()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = process_template(template_text, trees_df, particelle_df,
                                 equations_df, output_dir, format_type)
    output_file = output_dir / Path(args.input).name
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed)
    print(f"Report generato: {output_file}")
    if format_type == 'pdf':
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', output_file.name],
            cwd=output_dir,
            capture_output=True,
            check=True
        )
        print(f"Report generato: {output_file.with_suffix('.pdf')}")

def run_lista_particelle(args):
    """List land parcels."""
    print("Particelle disponibili:")
    list_parcels(args.particelle)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Analisi Accrescimenti - Tool unificato',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modalità di utilizzo:

1. GENERA EQUAZIONI:
   ./acc.py --genera-equazioni --funzione=log --fonte-altezze=ipsometro \\
            --input altezze.csv --output equations.csv

   ./acc.py --genera-equazioni --funzione=log --fonte-altezze=tabelle \\
            --input alsometrie.csv --particelle particelle.csv --output equations.csv

2. CALCOLA ALTEZZE:
   ./acc.py --calcola-altezze --equazioni equations.csv \\
            --input alberi.csv --output alberi-calcolati.csv

3. CALCOLA VOLUMI:
   ./acc.py --calcola-volumi --input alberi-calcolati.csv \\
            --output alberi-con-volumi.csv

4. GENERA REPORT:
   ./acc.py --report --formato=html --equazioni equations.csv \\
            --alberi alberi-con-volumi.csv --particelle particelle.csv \\
            --input template.html --output-dir report/

5. LISTA PARTICELLE:
   ./acc.py --lista-particelle --particelle particelle.csv
"""
    )

    # Mode selection (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--genera-equazioni', action='store_true',
                           help='Genera equazioni di interpolazione')
    run_group.add_argument('--calcola-altezze', action='store_true',
                           help='Calcola altezze usando equazioni')
    run_group.add_argument('--calcola-volumi', action='store_true',
                           help='Calcola volumi usando tavole di Tabacchi')
    run_group.add_argument('--report', action='store_true',
                           help='Genera report da template')
    run_group.add_argument('--lista-particelle', action='store_true',
                           help='Lista particelle (compresa, particella)')

    # Common file arguments
    files_group = parser.add_argument_group('File di input/output')
    files_group.add_argument('--input',
                            help='File di input')
    files_group.add_argument('--output',
                            help='File di output')
    files_group.add_argument('--output-dir',
                            help='Directory di output (per report)')
    files_group.add_argument('--equazioni',
                            help='File CSV con equazioni')
    files_group.add_argument('--alberi',
                            help='File CSV con dati alberi')
    files_group.add_argument('--particelle',
                            help='File CSV con dati particelle')

    # Specific options for --genera-equazioni
    eq_group = parser.add_argument_group('Opzioni per --genera-equazioni')
    eq_group.add_argument('--funzione', choices=['log', 'lin'], default='log',
                         help='Tipo di funzione (default: log)')
    eq_group.add_argument('--fonte-altezze',
                         choices=['ipsometro', 'originali', 'tabelle'],
                         help='Fonte dei dati di altezza')

    # Specific options for --report
    report_group = parser.add_argument_group('Opzioni per --report')
    report_group.add_argument('--formato', choices=['html', 'latex', 'pdf'], default='pdf',
                             help='Formato output (default: pdf)')
    report_group.add_argument('--ometti-generi-sconosciuti', action='store_true',
                             help='Ometti dai grafici generi per cui non abbiamo equazioni')

    args = parser.parse_args()

    if args.genera_equazioni:
        if not args.fonte_altezze:
            parser.error('--genera-equazioni richiede --fonte-altezze')
        if not args.input:
            parser.error('--genera-equazioni richiede --input')
        if not args.output:
            parser.error('--genera-equazioni richiede --output')
        if args.fonte_altezze == 'tabelle' and not args.particelle:
            parser.error('--fonte-altezze=tabelle richiede --particelle')
        run_genera_equazioni(args)

    elif args.calcola_altezze:
        if not args.equazioni:
            parser.error('--calcola-altezze richiede --equazioni')
        if not args.input:
            parser.error('--calcola-altezze richiede --input')
        if not args.output:
            parser.error('--calcola-altezze richiede --output')
        run_calcola_altezze(args)

    elif args.calcola_volumi:
        if not args.input:
            parser.error('--calcola-volumi richiede --input')
        if not args.output:
            parser.error('--calcola-volumi richiede --output')
        run_calcola_volumi(args)

    elif args.report:
        if not args.equazioni:
            parser.error('--report richiede --equazioni')
        if not args.alberi:
            parser.error('--report richiede --alberi')
        if not args.particelle:
            parser.error('--report richiede --particelle')
        if not args.input:
            parser.error('--report richiede --input')
        if not args.output_dir:
            parser.error('--report richiede --output-dir')
        run_report(args)

    elif args.lista_particelle:
        if not args.particelle:
            parser.error('--lista-particelle richiede --particelle')
        run_lista_particelle(args)


if __name__ == "__main__":
    main()
