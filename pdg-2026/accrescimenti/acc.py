#!/usr/bin/env python3
# pylint: disable=too-many-lines
# pylint: disable=singleton-comparison
"""
Forest Analysis: estiamtion of forest characteristics and growth ("accrescimenti").
"""

from abc import ABC, abstractmethod
import argparse
from collections import defaultdict
import csv
import io
from pathlib import Path
import re
import subprocess
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsort_keygen
from scipy import stats

SAMPLE_AREA_HA = 0.125
MIN_TREES_PER_HA = 0.5 # Ignore buckets less than this in classi diametriche graphs.

# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

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

class SnippetFormatter(ABC):
    """Formats individual components (images, metadata) for template insertion."""

    @abstractmethod
    def format_image(self, filepath: Path) -> str:
        """Format image reference for this format."""

    @abstractmethod
    def format_metadata(self, data: dict, curve_info: list = None) -> str:
        """Format metadata block for this format.

        Args:
            stats: Statistics about the region/species
            curve_info: List of dicts with {species, equation, r_squared, n_points}
                       from equations.csv
        """

    @abstractmethod
    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]], small: bool = False) -> str:
        """Format a data table for this format.

        Args:
            headers: Column headers
            rows: Data rows (each row is a list of strings)
            small: If True, use smaller font size
        Returns:
            Formatted table snippet
        """

    @abstractmethod
    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        """Format parcel properties for this format.

        Args:
            short_fields: List of (label, value) tuples for inline display
            paragraph_fields: List of (label, value) tuples for paragraph display
        Returns:
            Formatted properties snippet
        """


class HTMLSnippetFormatter(SnippetFormatter):
    """HTML snippet formatter."""

    def format_image(self, filepath: Path) -> str:
        return f'<img src="{filepath.name}" class="graph-image">'

    def format_metadata(self, data: dict, curve_info: list = None) -> str:
        """Format metadata as HTML."""
        html = '<div class="graph-details">\n'
        html += f'<p><strong>Comprese:</strong> {data["regions"]}</p>\n'
        html += f'<p><strong>Generi:</strong> {data["species"]}</p>\n'
        html += f'<p><strong>Alberi campionati:</strong> {data["trees"].shape[0]:d}</p>\n'
        if curve_info:
            i = 'i' if len(curve_info) > 1 else 'e'
            html += f'<br><p><strong>Equazion{i} interpolant{i}:</strong></p>\n'
            for curve in curve_info:
                html += (f'<p>{curve["genere"]}: {curve["equation"]} '
                         f'(R² = {curve["r_squared"]:.2f}, n = {curve["n_points"]})</p>\n')
        html += '</div>\n'
        return html

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]],
        small: bool = False) -> str:
        """Format table as HTML.
        Headers is a list of tuples (header, justification).
        Justification is 'l' for left, 'r' for right, 'c' for center.
        """
        justify_style = {'l': 'col_left', 'r': 'col_right', 'c': 'col_center'}
        justify = [justify_style[h[1]] for h in headers]
        html = '<table class="volume-table">\n'
        html += '  <thead>\n    <tr>\n'
        for header, j in zip([h[0] for h in headers], justify):
            html += f'      <th class="{j}">{header}</th>\n'
        html += '    </tr>\n  </thead>\n'
        html += '  <tbody>\n'
        for row in rows:
            html += '    <tr>\n'
            for cell, j in zip(row, justify):
                html += f'      <td class="{j}">{cell}</td>\n'
            html += '    </tr>\n'
        html += '  </tbody>\n</table>\n'
        return html

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        """Format parcel properties as HTML."""
        html = '<div class="parcel-props">\n'
        html += '<p class="props-inline">'
        html += ' · '.join(f'<strong>{label}:</strong> {value}' for label, value in short_fields)
        html += '</p>\n'
        for label, value in paragraph_fields:
            html += f'<p><strong>{label}:</strong> {value}</p>\n'
        html += '</div>\n'
        return html


class LaTeXSnippetFormatter(SnippetFormatter):
    """LaTeX snippet formatter."""

    def format_image(self, filepath: Path) -> str:
        return (f'\\begin{{figure}}[H]\n'
                f'  \\centering\n'
                f'  \\includegraphics[width=0.9\\textwidth]{{{filepath.name}}}\n'
                f'\\end{{figure}}\n')

    def format_metadata(self, data: dict, curve_info: list = None) -> str:
        """Format metadata as LaTeX."""
        latex = '\\begin{quote}\\small\n'
        latex += f"\\textbf{{Comprese:}} {data['regions']}\\\\\n"
        latex += f"\\textbf{{Generi:}} {data['species']}\\\\\n"
        latex += f"\\textbf{{Alberi campionati:}} {data['trees'].shape[0]:d}\\\\\n"
        if curve_info:
            i = 'i' if len(curve_info) > 1 else 'e'
            latex += f'\n\\textbf{{Equazion{i} interpolant{i}:}}\\\\\n'
            for curve in curve_info:
                eq = curve['equation'].replace('*', r'\times ')
                eq = eq.replace('ln', r'\ln')
                latex += (f"{curve['genere']}: ${eq}$ ($R^2$ = {curve['r_squared']:.2f}, "
                         f"$n$ = {curve['n_points']})\\\\\n")

        latex += '\\end{quote}\n'
        return latex

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]], small: bool = False) -> str:
        """Format table as LaTeX using longtable for page breaks.
           Headers is a list of tuples (header, justification).
           Justification is 'l' for left, 'r' for right, 'c' for center.
        """
        col_specs = [h[1] for h in headers]
        # Use longtable instead of tabular to allow page breaks
        latex = '\\begin{small}\n' if small else ''
        latex += f'\\begin{{longtable}}{{ {"".join(col_specs)} }}\n'
        latex += '\\hline\n'
        latex += ' & '.join([h[0] for h in headers]) + ' \\\\\n'
        latex += '\\hline\n'
        latex += '\\endfirsthead\n'  # End of first page header
        latex += '\\multicolumn{' + str(len(headers)) + '}{c}'
        latex += '{\\textit{(continua dalla pagina precedente)}} \\\\\n'
        latex += '\\hline\n'
        latex += ' & '.join([h[0] for h in headers]) + ' \\\\\n'
        latex += '\\hline\n'
        latex += '\\endhead\n'  # Header for subsequent pages
        latex += '\\hline\n'
        latex += '\\multicolumn{' + str(len(headers)) + '}{r}'
        latex += '{\\textit{(continua alla pagina successiva)}} \\\\\n'
        latex += '\\endfoot\n'  # Footer for all pages except last
        latex += '\\hline\n'
        latex += '\\endlastfoot\n'  # Footer for last page
        for row in rows:
            latex += ' & '.join(row) + ' \\\\\n'
        latex += '\\end{longtable}\n'
        if small:
            latex += '\\end{small}\n'
        return latex

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        """Format parcel properties as LaTeX."""
        formatted = [f'\\textbf{{{label}:}} {value}' for label, value in short_fields]
        lines = [' $\\cdot$ '.join(formatted[i:i+2]) for i in range(0, len(formatted), 2)]
        latex = '\\noindent ' + ' \\\\\n'.join(lines) + '\n\n'
        for label, value in paragraph_fields:
            latex += f'\\noindent\\textbf{{{label}:}} {value}\n\n'
        return latex


class CSVSnippetFormatter(SnippetFormatter):
    """CSV snippet formatter for table-only output."""

    def format_image(self, filepath: Path) -> str:
        raise NotImplementedError("Formato CSV non supporta immagini (direttive @@g*)")

    def format_metadata(self, data: dict, curve_info: list = None) -> str:
        raise NotImplementedError("Formato CSV non supporta metadati")

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]],
                     small: bool = False) -> str:
        """Format table as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([h[0] for h in headers])
        writer.writerows(rows)
        return output.getvalue()

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        raise NotImplementedError("Formato CSV non supporta @@prop")


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


def calculate_one_tree_volume(diameter: float, height: float, genere: str) -> float:
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


def calculate_all_trees_volume(trees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volumes for all trees using Tabacchi equations.

    Args:
        trees_df: DataFrame with D(cm), h(m), Genere columns

    Returns:
        DataFrame with added/updated V(m3) column
    """
    required = ['D(cm)', 'h(m)', 'Genere']
    missing = [col for col in required if col not in trees_df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    result_df = trees_df.copy()
    result_df['V(m3)'] = 0.0

    for idx, row in result_df.iterrows():
        genere = row['Genere']
        diameter = row['D(cm)']
        height = row['h(m)']

        if pd.isna(diameter) or pd.isna(height):
            raise ValueError(f"Dati mancanti per riga {idx}: D={diameter}, h={height}")
        if genere not in TABACCHI_B:
            raise ValueError(f"Genere '{genere}' non trovato nelle tavole di Tabacchi")

        result_df.at[idx, 'V(m3)'] = calculate_one_tree_volume(diameter, height, genere)

    return result_df


# =============================================================================
# DATA PREPARATION
# =============================================================================

region_cache = {}
def parcel_data(tree_files: list[str], tree_df: pd.DataFrame, parcel_df: pd.DataFrame,
                regions: list[str], parcels: list[str], species: list[str]) -> dict:
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
        Dictionary with the following keys:
        - 'trees': DataFrame with set of tree filtered by regions, parcels and species
        - 'regions': List of regions in trees data
        - 'species': List of species in trees data
        - 'parcels': Dictionary with metadata for each parcel

    Raises:
        ValueError: for various invalid conditions
    """
    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    def _filter_df(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
        """Return new DataFrame with rows of df where df[column] is in values."""
        if not values:
            return df
        return df[df[column].isin(values)]

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
    trees_region = _filter_df(trees_region, 'Compresa', regions)
    trees_region = _filter_df(trees_region, 'Particella', parcels)
    trees_region_species = _filter_df(trees_region, 'Genere', species)
    if len(trees_region_species) == 0:
        raise ValueError(f"Nessun dato trovato per comprese '{regions}' " +
                         f"particelle '{parcels}' generi '{species}'")

    parcel_stats = {}
    for (region, parcel), trees in trees_region.groupby(['Compresa', 'Particella']):
        md_row = parcel_df[
            (parcel_df['Compresa'] == region) &
            (parcel_df['Particella'] == parcel)
        ]
        if len(md_row) != 1:
            raise ValueError(f"Nessun metadato per particella {region}/{parcel}")

        md = md_row.iloc[0]
        area_ha = md['Area (ha)']
        n_sample_areas = trees.drop_duplicates(
            subset=['Compresa', 'Particella', 'Area saggio']).shape[0]
        if n_sample_areas == 0:
            raise ValueError(f"Nessuna area di saggio per particella {region}/{parcel}")
        sampled_frac = n_sample_areas * SAMPLE_AREA_HA / area_ha

        parcel_stats[(region, parcel)] = {
            'area_ha': area_ha,
            'sector': md['Comparto'],
            'age': md['Età media'],
            'n_sample_areas': n_sample_areas,
            'sampled_frac': sampled_frac,
        }

    data = {
        'trees': trees_region_species,
        'regions': sorted(trees_region['Compresa'].unique()),
        'species': sorted(trees_region_species['Genere'].unique()),
        'parcels': parcel_stats,
    }
    region_cache[key] = data
    return data

file_cache = {}
def load_csv(filenames: list[str] | str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load CSV file(s), skipping comment lines starting with #."""
    if isinstance(filenames, str):
        filenames = [filenames]
    key = (data_dir, tuple(sorted(filenames)))
    if key not in file_cache:
        files = [ data_dir / filename if data_dir else Path(filename) for filename in filenames]
        file_cache[key] = pd.concat(
            [pd.read_csv(file, comment='#') for file in files],
            ignore_index=True)
    return file_cache[key]

def load_trees(filenames: list[str] | str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load trees from CSV file(s), skipping comment lines starting with #."""
    df = load_csv(filenames, data_dir)
    df.drop(df[df['Fustaia']==False].index,inplace=True)
    df['Particella'] = df['Particella'].astype(str)
    return df


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
        ipsometro_file: CSV with columns [Compresa, Genere, D(cm), h(m)]
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    df = load_csv(ipsometro_file, None)
    df['x'] = df['D(cm)']
    df['y'] = df['h(m)']
    groups = []

    for (compresa, genere), group in df.groupby(['Compresa', 'Genere']):
        groups.append(((compresa, genere), group))
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
    df = load_trees(alberi_file)
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
    df_particelle = load_csv(particelle_file)
    comprese = sorted(df_particelle['Compresa'].dropna().unique())

    df_als = load_csv(tabelle_file)
    df_als['Diam 130cm'] = pd.to_numeric(df_als['Diam 130cm'], errors='coerce')
    df_als['Altezza indicativa'] = pd.to_numeric(df_als['Altezza indicativa'], errors='coerce')
    df_als['x'] = df_als['Diam 130cm']
    df_als['y'] = df_als['Altezza indicativa']

    groups = []
    for compresa in comprese:
        for genere, group in df_als.groupby('Genere'):
            groups.append(((compresa, genere), group))
    return fit_curves_grouped(groups, funzione)

def compute_heights(trees_df: pd.DataFrame, equations_df: pd.DataFrame,
                    verbose: bool = False) -> tuple[pd.DataFrame, int, int]:
    """
    Apply height equations to tree database, updating heights.

    Args:
        trees_df: DataFrame with Compresa, Genere, D(cm) columns
        equations_df: DataFrame with compresa, genere, funzione, a, b columns
        verbose: If True, print progress messages

    Returns:
        Tuple of (updated DataFrame, trees_updated count, trees_unchanged count)
    """
    trees_updated = 0
    trees_unchanged = 0

    result_df = trees_df.copy()
    result_df['h(m)'] = result_df['h(m)'].astype(float)

    for (compresa, genere), group in trees_df.groupby(['Compresa', 'Genere']):
        eq_row = equations_df[
            (equations_df['compresa'] == compresa) &
            (equations_df['genere'] == genere)
        ]

        if len(eq_row) == 0:
            if verbose:
                print(f"  {compresa} - {genere}: nessuna equazione; altezze immutate")
            trees_unchanged += len(group)
            continue

        eq = eq_row.iloc[0]
        indices = group.index
        diameters = trees_df.loc[indices, 'D(cm)'].astype(float)

        if eq['funzione'] == 'ln':
            new_heights = eq['a'] * np.log(np.maximum(diameters, 0.1)) + eq['b']
        else:  # 'lin'
            new_heights = eq['a'] * diameters + eq['b']

        result_df.loc[indices, 'h(m)'] = new_heights.astype(float)
        trees_updated += len(group)

        if verbose:
            print(f"  {compresa} - {genere}: {len(group)} alberi aggiornati")

    return result_df, trees_updated, trees_unchanged


# =============================================================================
# RENDERING OF INDIVIDUAL DIRECTIVES
# =============================================================================

# CURVE IPSOMETRICHE ==========================================================

def render_gci_graph(data: dict, equations_df: pd.DataFrame,
                     output_path: Path, formatter: SnippetFormatter,
                     color_map: dict, **options) -> dict:
    """
    Generate height-diameter graphs.

    Args:
        data: Output from region_data
        equations_df: Pre-computed equations from CSV
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        Dict with keys:
        - 'filepath': Path to generated PNG
        - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    #pylint: disable=too-many-locals
    trees = data['trees']
    species = data['species']
    regions = data['regions']

    fig, ax = plt.subplots(figsize=(4, 3))

    # First pass: scatter points (once per species)
    for sp in species:
        sp_data = trees[trees['Genere'] == sp]
        x = sp_data['D(cm)'].values
        y = sp_data['h(m)'].values
        ax.scatter(x, y, color=color_map[sp], label=sp, alpha=0.7, linewidth=2, s=1)

    # Second pass: regression curves (per compresa/genere pair)
    curve_info = []
    for region in regions:
        for sp in species:
            sp_data = trees[trees['Genere'] == sp]
            x = sp_data['D(cm)'].values

            # Look up pre-computed equation from equations.csv, if any
            eq_row = equations_df[
                (equations_df['compresa'] == region) &
                (equations_df['genere'] == sp)
            ]

            if len(eq_row) > 0:
                eq = eq_row.iloc[0]
                x_smooth = np.linspace(1, x.max(), 100)
                if eq['funzione'] == 'ln':
                    y_smooth = eq['a'] * np.log(x_smooth) + eq['b']
                    eq_str = f"y = {eq['a']:.2f}*ln(x) + {eq['b']:.2f}"
                else:  # 'lin'
                    y_smooth = eq['a'] * x_smooth + eq['b']
                    eq_str = f"y = {eq['a']:.2f}*x + {eq['b']:.2f}"

                ax.plot(x_smooth, y_smooth, color=color_map[sp],
                    linestyle='--', alpha=0.8, linewidth=1.5)

                curve_info.append({
                    'genere': sp,
                    'equation': eq_str,
                    'r_squared': eq['r2'],
                    'n_points': int(eq['n'])
                })

    x_max = max(options.get('x_max', 0), trees['D(cm)'].max() + 3)
    y_max = max(options.get('y_max', 0), (trees['h(m)'].max() + 6) // 5 * 5)
    ax.set_xlabel('Diametro (cm)', fontweight='bold')
    ax.set_ylabel('Altezza (m)', fontweight='bold')
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

    snippet = formatter.format_image(output_path)
    snippet += '\n' + formatter.format_metadata(data, curve_info=curve_info)

    return {
        'filepath': output_path,
        'snippet': snippet
    }


# CLASSI DIAMETRICHE ==========================================================

def render_gcd_graph(data: dict, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict, **options) -> dict:
    """
    Generate diameter class histograms.

    Args:
        data: Output from region_data
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    trees = data['trees']
    species = data['species']
    parcels = data['parcels']

    counts, area_ha = {}, 0
    for (region, parcel), ptrees in trees.groupby(['Compresa', 'Particella']):
        p = parcels[(region, parcel)]
        counts[(region, parcel)] = (
            ptrees.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
            / p['sampled_frac'])
        area_ha += p['area_ha']
    counts = pd.concat(counts.values()).groupby(level=0).sum()/area_ha
    counts = counts[species].fillna(0).sort_index()

    fig, ax = plt.subplots(figsize=(4, 3.75))

    bottom = np.zeros(len(counts.index))
    for genere in species:
        if genere not in counts.columns:
            continue
        values = counts[genere].values
        ax.bar(counts.index, values, bottom=bottom,
               label=genere, color=color_map[genere],
               alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values

    x_max = (options.get('x_max', 0)
        if options.get('x_max', 0) > 0 else trees['Classe diametrica'].max()+2)
    y_max = (options.get('y_max', 0)
        if options.get('y_max', 0) > 0 else counts.sum(axis=1).max() * 1.1)

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Stima alberi / ha', fontweight='bold')
    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, x_max, 2))
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


BIN0 = "0-19 cm"
BIN1 = "20-39 cm"
BIN2 = "40+ cm"
def render_tcd_table(data: dict, formatter: SnippetFormatter, **options) -> dict:
    """
    Generate diameter class table (@@tcd directive).

    Creates a table with rows for each species and columns for diameter ranges:
    [0-20) cm, [20-40) cm, and ≥40 cm. Values are estimated trees per hectare.

    Args:
        data: Output from parcel_data
        formatter: HTML or LaTeX snippet formatter

    Returns:
        dict with 'snippet' key containing formatted table
    """
    trees = data['trees']
    species = data['species']
    parcels = data['parcels']

    # Assign diameter range bins
    def diameter_bin(d):
        if d < 20:
            return BIN0
        elif d < 40:
            return BIN1
        return BIN2

    trees = trees.copy()
    trees['Classe'] = trees['D(cm)'].apply(diameter_bin)

    # Count trees per hectare, grouped by parcel then aggregated
    counts, area_ha = {}, 0
    for (region, parcel), ptrees in trees.groupby(['Compresa', 'Particella']):
        p = parcels[(region, parcel)]
        counts[(region, parcel)] = (
            ptrees.groupby(['Classe', 'Genere']).size().unstack(fill_value=0)
            / p['sampled_frac'])
        area_ha += p['area_ha']
    counts = pd.concat(counts.values()).groupby(level=0).sum() / area_ha

    # Build table: rows = species, columns = diameter ranges
    bin_order = [BIN0, BIN1, BIN2]
    headers = [('Genere', 'l')] + [(b, 'r') for b in bin_order] + [('Totale', 'r')]

    rows = []
    col_totals = {b: 0.0 for b in bin_order}
    for genere in species:
        row = [genere]
        row_total = 0.0
        for b in bin_order:
            val = counts.loc[b, genere] if b in counts.index and genere in counts.columns else 0
            row.append(f"{val:.1f}")
            row_total += val
            col_totals[b] += val
        row.append(f"{row_total:.1f}")
        rows.append(row)

    # Add totals row
    total_row = ['Totale'] + [f"{col_totals[b]:.1f}" for b in bin_order]
    total_row.append(f"{sum(col_totals.values()):.1f}")
    rows.append(total_row)

    return {'snippet': formatter.format_table(headers, rows)}


def render_prop(particelle_df: pd.DataFrame, compresa: str, particella: str,
                formatter: SnippetFormatter) -> dict:
    """
    Render parcel properties (@@prop directive).

    Args:
        particelle_df: DataFrame with parcel metadata
        compresa: Compresa name
        particella: Particella identifier
        formatter: HTML or LaTeX snippet formatter

    Returns:
        dict with 'snippet' key containing formatted properties
    """
    row = particelle_df[(particelle_df['Compresa'] == compresa) &
                        (particelle_df['Particella'] == particella)]
    if row.empty:
        raise ValueError(f"Particella '{particella}' non trovata in compresa '{compresa}'")
    row = row.iloc[0]

    area = f"{row['Area (ha)']:.2f} ha"
    altitudine = f"{int(row['Altitudine min'])}-{int(row['Altitudine max'])} m"

    short_fields = [
        ('Area', area),
        ('Località', row['Località']),
        ('Età media', f"{int(row['Età media'])} anni"),
        ('Governo', row['Governo']),
        ('Altitudine', altitudine),
        ('Esposizione', row['Esposizione'] or ''),
    ]
    paragraph_fields = [
        ('Stazione', row['Stazione']),
        ('Soprassuolo', row['Soprassuolo']),
    ]

    return {'snippet': formatter.format_prop(short_fields, paragraph_fields)}


# STIMA VOLUMI ================================================================

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


def calculate_tsv_table(data: dict, group_cols: list[str],
                        calc_margin: bool, calc_total: bool) -> pd.DataFrame:
    """Calculate the table rows for the @@tsv directive. Returns a DataFrame."""
    #pylint: disable=too-many-locals
    trees = data['trees']
    if 'V(m3)' not in trees.columns:
        raise ValueError("@@tsv richiede dati con volumi (manca la colonna 'V(m3)'). "
                         "Esegui --calcola-altezze-volumi per calcolarli.")
    parcels = data['parcels']

    if not group_cols:
        trees = trees.copy()
        trees['_'] = 'Totale'
        group_cols = ['_']  # Pseudo-column for single aggregation

    rows = []
    for group_key, group_trees in trees.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_key))

        if calc_total:
            # First scale per-parcel, then aggregate (sampling density varies across parcels)
            n_trees, volume, margin = 0.0, 0.0, 0.0

            for (region, parcel), ptrees in group_trees.groupby(['Compresa', 'Particella']):
                try:
                    p = parcels[(region, parcel)]
                except KeyError as e:
                    raise ValueError(f"Particella {region}/{parcel} non trovata") from e
                sf = p['sampled_frac']
                n_trees += len(ptrees) / sf
                volume += ptrees['V(m3)'].sum() / sf
                if calc_margin:
                    _, pmargin = calculate_volume_confidence_interval(ptrees)
                    margin += pmargin / sf
        else:
            n_trees = len(group_trees)
            volume = group_trees['V(m3)'].sum()
            if calc_margin:
                _, margin = calculate_volume_confidence_interval(group_trees)

        row_dict['n_trees'] = n_trees
        row_dict['volume'] = volume
        if calc_margin:
            row_dict['vol_lo'] = volume - margin
            row_dict['vol_hi'] = volume + margin
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if '_' in group_cols:
        group_cols.remove('_')
        df = df.drop(columns=['_'])
    return df.sort_values(group_cols,
        key=lambda col: natsort_keygen()(col) if col.name == 'Particella' else col)


def render_tsv_table(data: dict, formatter: SnippetFormatter,
                     **options: dict) -> dict:
    """
    Generate volume summary table (@@tsv directive).

    Args:
        data: Output from region_data
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
    group_cols = []
    if options['per_compresa']:
        group_cols.append('Compresa')
    if options['per_particella']:
        group_cols.append('Particella')
    if options['per_genere']:
        group_cols.append('Genere')

    df = calculate_tsv_table(data, group_cols,
        options['intervallo_fiduciario'], options['stime_totali'])

    headers = []
    show_region = 'Compresa' in group_cols
    show_parcel = 'Particella' in group_cols
    show_species = 'Genere' in group_cols

    if show_region:
        headers.append(('Compresa', 'l'))
    if show_parcel:
        headers.append(('Particella', 'l'))
    if show_species:
        headers.append(('Genere', 'l'))
    headers.append(('N. Alberi', 'r'))
    headers.append(('Volume (m³)', 'r'))
    if options['intervallo_fiduciario']:
        headers.append(('IF inf (m³)', 'r'))
        headers.append(('IF sup (m³)', 'r'))

    # Format numeric columns as strings.
    df_display = df.copy()
    df_display['n_trees'] = df_display['n_trees'].apply(lambda x: f"{x:.0f}")
    for col in ['volume', 'vol_lo', 'vol_hi']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")

    # Add totals row. Note that if there are no grouping columns,
    # the result already includes (just) the total row.
    display_rows = df_display.values.tolist()
    if options['totali'] and group_cols:
        total_row = ['Totale'] + [''] * (len(group_cols) - 1)
        total_row.append(f"{df['n_trees'].sum():.0f}")
        total_row.append(f"{df['volume'].sum():.2f}")
        if 'vol_lo' in df.columns:
            total_row.append(f"{df['vol_lo'].sum():.2f}")
            total_row.append(f"{df['vol_hi'].sum():.2f}")
        display_rows.append(total_row)

    return {'snippet': formatter.format_table(headers, display_rows)}


def render_gsv_graph(data: dict, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict,
                     **options) -> dict:
    """
    Generate volume summary horizontal bar graph (@@gsv directive).

    Args:
        data: Output from parcel_data
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping
        options: per_compresa, per_particella, per_genere flags

    Returns:
        dict with 'filepath' and 'snippet' keys
    """
    group_cols = []
    if options.get('per_compresa', True):
        group_cols.append('Compresa')
    if options.get('per_particella', True):
        group_cols.append('Particella')

    # For stacking, we need per-genere data even if displaying by compresa/particella
    stacked = options.get('per_genere', False) and group_cols
    if stacked:
        base_cols = group_cols.copy()
        group_cols.append('Genere')

    df = calculate_tsv_table(data, group_cols, calc_margin=False, calc_total=True)
    if df.empty:
        return {'snippet': '', 'filepath': None}

    if stacked:
        # Pivot to get genere as columns for stacking
        pivot_df = df.pivot_table(index=base_cols, columns='Genere',
                                  values='volume', fill_value=0)
        labels = ['/'.join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
                  for idx in pivot_df.index]
        generi = pivot_df.columns.tolist()

        # Sort by compresa then particella (natural sort for particella)
        if 'Particella' in base_cols:
            sort_keys = [pivot_df.index.get_level_values(c) for c in base_cols]
            sort_idx = sorted(range(len(labels)), key=lambda i: tuple(
                natural_sort_key(str(sort_keys[j][i])) if base_cols[j] == 'Particella'
                else (0, str(sort_keys[j][i])) for j in range(len(base_cols))))
            labels = [labels[i] for i in sort_idx]
            pivot_df = pivot_df.iloc[sort_idx]

        # Calculate spacing for compresa groups
        spacing = []
        if 'Compresa' in base_cols and 'Particella' in base_cols:
            comprese = pivot_df.index.get_level_values('Compresa')
            for i, c in enumerate(comprese):
                spacing.append(0.3 if i > 0 and c != comprese[i-1] else 0)
        else:
            spacing = [0] * len(labels)

        # Calculate y positions with spacing
        y_positions = []
        cumulative = 0
        for s in spacing:
            cumulative += s
            y_positions.append(cumulative)
            cumulative += 1

        # Figure height: ~0.35 inches per bar, minimum 3
        fig_height = max(3, len(labels) * 0.35 + sum(spacing) * 0.35)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        # Draw stacked horizontal bars
        left = np.zeros(len(labels))
        for genere in generi:
            values = pivot_df[genere].values
            ax.barh(y_positions, values, left=left, label=genere,
                    color=color_map.get(genere, '#0c63e7'), height=0.8,
                    edgecolor='white', linewidth=0.5)
            left += values

        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        handles, lbl = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(lbl),
                  title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')
    else:
        # Simple bars (no stacking)
        if options.get('per_genere', False):
            # Per-genere only: one bar per genere
            labels = df['Genere'].tolist()
            values = df['volume'].values
        elif group_cols:
            labels = ['/'.join(str(row[c]) for c in group_cols) for _, row in df.iterrows()]
            values = df['volume'].values
        else:
            labels = ['Totale']
            values = df['volume'].values

        # Calculate spacing for compresa groups
        spacing = []
        if 'Compresa' in group_cols and 'Particella' in group_cols:
            comprese = df['Compresa'].tolist()
            for i, c in enumerate(comprese):
                spacing.append(0.3 if i > 0 and c != comprese[i-1] else 0)
        else:
            spacing = [0] * len(labels)

        y_positions = []
        cumulative = 0
        for s in spacing:
            cumulative += s
            y_positions.append(cumulative)
            cumulative += 1

        fig_height = max(3, len(labels) * 0.35 + sum(spacing) * 0.35)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        ax.barh(y_positions, values, color='#0c63e7', height=0.8,
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

    ax.set_xlabel('Volume (m³)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    ax.set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    snippet = formatter.format_image(output_path)
    snippet += '\n' + formatter.format_metadata(data)

    return {'filepath': output_path, 'snippet': snippet}


# RIPRESA =====================================================================

def compute_pp_max(volume_per_ha: float, eta_media: float,
                   provvigione_minima: float,
                   provv_vol_df: pd.DataFrame,
                   provv_eta_df: pd.DataFrame) -> float:
    """
    Compute maximum harvest percentage (PP_max) based on volume and age rules.

    Args:
        volume_per_ha: Total volume per hectare in the particella
        eta_media: Mean age of trees in the particella
        provvigione_minima: Minimum stock for this comparto (from comparti table)
        provv_vol_df: Volume-based harvest rules (PPM, PP_max columns)
        provv_eta_df: Age-based harvest rules (Anni, PP_max columns)

    Returns:
        Maximum harvest percentage (0-100)
    """
    # Step 1: Find PP_max from volume rules
    # Rules are in descending order of PPM, take first match where v > PPM * pm / 100
    pp_max_vol = 0
    for _, row in provv_vol_df.iterrows():
        threshold = row['PPM'] * provvigione_minima / 100
        if volume_per_ha > threshold:
            pp_max_vol = row['PP_max']
            break

    # Step 2: Find PP_max limit from age rules
    # Rules are in descending order of Anni, take first match where età > Anni
    pp_max_eta = 100  # Default: no age limit
    for _, row in provv_eta_df.iterrows():
        if eta_media > row['Anni']:
            pp_max_eta = row['PP_max']
            break

    # Return the more restrictive of the two
    return min(pp_max_vol, pp_max_eta)


def calculate_tpt_table(data: dict, comparti_df: pd.DataFrame,
                        provv_vol_df: pd.DataFrame, provv_eta_df: pd.DataFrame,
                        group_cols: list[str]) -> pd.DataFrame:
    """
    Calculate harvest (prelievo totale) table data for the @@tpt directive.

    Algorithm per particella:
    1. Filter to Fustaia only (ceduo is excluded via Provvigione minima = -1)
    2. Calculate scaled total volume V_total (all species in particella)
    3. v = V_total / area (volume per hectare)
    4. Look up pm = provvigione_minima[comparto]
    5. Find PP_max from provv_vol (first row where v > PPM * pm / 100)
    6. Cap PP_max using provv_eta (first row where età > Anni)
    7. harvestable = V_total_scaled * PP_max / 100

    Note: per_genere grouping is not supported because PP_max is a parcel-level
    constraint. The harvest percentage can be applied to any subset of species.

    Args:
        data: Output from parcel_data
        comparti_df: Comparto -> Provvigione minima mapping
        provv_vol_df: Volume-based harvest rules
        provv_eta_df: Age-based harvest rules
        group_cols: List of grouping columns (Compresa, Particella only)

    Returns:
        DataFrame with columns depending on group_cols, plus:
        comparto, volume, pp_max, harvest_ha, harvest_total, n_trees
    """
    #pylint: disable=too-many-locals
    trees = data['trees']
    if 'V(m3)' not in trees.columns:
        raise ValueError("@@tpt richiede dati con volumi (colonna V(m3) mancante). "
                         "Esegui --calcola-altezze-volumi per calcolarli.")
    parcels = data['parcels']

    if not group_cols:
        trees = trees.copy()
        trees['_'] = 'Totale'
        group_cols = ['_']

    per_parcel = 'Particella' in group_cols or len(parcels) == 1
    rows = []
    sector_pm = dict(zip(comparti_df['Comparto'], comparti_df['Provvigione minima']))
    for group_key, group_trees in trees.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_key))
        volume, pp_max, harvest, area_ha = 0.0, 0.0, 0.0, 0.0
        any_tree = False
        for (region, parcel), part_trees in group_trees.groupby(['Compresa', 'Particella']):
            try:
                p = parcels[(region, parcel)]
            except KeyError as e:
                raise ValueError(f"Particella {region}/{parcel} non trovata") from e

            sector, p_area, sf, age = p['sector'], p['area_ha'], p['sampled_frac'], p['age']
            try:
                provv_min = sector_pm[sector]
            except KeyError as e:
                raise ValueError(f"Comparto {p['sector']} non trovato") from e
            if provv_min < 0:  # Skip ceduo
                continue

            any_tree = True
            p_vol = part_trees['V(m3)'].sum() / sf
            pp_max = compute_pp_max(p_vol / p_area, age, provv_min, provv_vol_df, provv_eta_df)
            volume += p_vol
            harvest += p_vol * pp_max / 100
            area_ha += p_area

        if not any_tree: # All ceduo in this grouping
            continue

        if per_parcel:
            row_dict['sector'] = sector
            row_dict['age'] = age
        row_dict['area_ha'] = area_ha
        row_dict['volume'] = volume
        if per_parcel:
            row_dict['pp_max'] = pp_max
        row_dict['harvest'] = harvest
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if '_' in group_cols:
        group_cols.remove('_')
        df = df.drop(columns=['_'])
    return df.sort_values(group_cols,
        key=lambda col: natsort_keygen()(col) if col.name == 'Particella' else col)


def render_tpt_table(data: dict, comparti_df: pd.DataFrame,
                     provv_vol_df: pd.DataFrame, provv_eta_df: pd.DataFrame,
                     formatter: SnippetFormatter,
                     **options: dict) -> dict:
    """
    Render harvest (prelievo totale) table (@@tpt directive).

    Args:
        data: Output from parcel_data
        comparti_df: Comparto -> Provvigione minima mapping
        provv_vol_df: Volume-based harvest rules
        provv_eta_df: Age-based harvest rules
        formatter: HTML or LaTeX snippet formatter
        options: Dictionary of options:
            - per_compresa, per_particella: Grouping flags
            - col_comparto: Show comparto column (default si)
            - col_volume: Show total volume column (default si)
            - col_volume_ha: Show total volume per hectare column (default si)
            - col_pp_max: Show provvigione percentuale massima column (default si)
            - col_prelievo: Show harvest column (default si)
            - totali: Add totals row

        comparto and pp_max only appear if per_particella is True.

    Returns:
        dict with 'snippet' key containing formatted table
    """
    # Determine grouping columns (per_genere not supported for tpt)
    group_cols = []
    if options['per_compresa']:
        group_cols.append('Compresa')
    if options['per_particella']:
        group_cols.append('Particella')

    df = calculate_tpt_table(data, comparti_df, provv_vol_df, provv_eta_df, group_cols)
    if df.empty:
        return {'snippet': ''}

    per_parcel = 'Particella' in group_cols or len(data['parcels']) == 1

    headers = []
    for col in group_cols:
        headers.append((col, 'l'))
    if options['col_comparto'] and per_parcel:
        headers.append(('Comparto', 'l'))
    if options['col_eta'] and per_parcel:
        headers.append(('Età (anni)', 'r'))
    if options['col_area_ha']:
        headers.append(('Area (ha)', 'r'))
    if options['col_volume']:
        headers.append(('Vol tot (m³)', 'r'))
    if options['col_volume_ha']:
        headers.append(('Vol/ha (m³/ha)', 'r'))
    if options['col_pp_max'] and per_parcel:
        headers.append(('Prelievo \\%', 'r'))
    if options['col_prelievo']:
        headers.append(('Prelievo (m³)', 'r'))

    # Build display rows
    display_rows = []
    for _, row in df.iterrows():
        display_row = []
        for col in group_cols:
            display_row.append(str(row[col]))
        if options['col_comparto'] and per_parcel:
            display_row.append(str(row['sector']))
        if options['col_eta'] and per_parcel:
            display_row.append(f"{row['age']:.0f}")
        if options['col_area_ha']:
            display_row.append(f"{row['area_ha']:.2f}")
        if options['col_volume']:
            display_row.append(f"{row['volume']:.2f}")
        if options['col_volume_ha']:
            display_row.append(f"{row['volume'] / row['area_ha']:.2f}")
        if options['col_pp_max'] and per_parcel:
            display_row.append(f"{row['pp_max']:.0f}")
        if options['col_prelievo']:
            display_row.append(f"{row['harvest']:.2f}")
        display_rows.append(display_row)

    # Add totals row if requested (and there are grouping columns)
    if options.get('totali', False) and group_cols:
        total_row = ['Totale'] + [''] * (len(group_cols) - 1)
        if options['col_comparto'] and per_parcel:
            total_row.append('')
        if options['col_eta'] and per_parcel:
            total_row.append('')
        if options['col_area_ha']:
            total_row.append(f"{df['area_ha'].sum():.2f}")
        if options['col_volume']:
            total_row.append(f"{df['volume'].sum():.2f}")
        if options['col_volume_ha']:
            total_row.append(f"{df['volume'].sum() / df['area_ha'].sum():.2f}")
        if options['col_pp_max'] and per_parcel:
            total_row.append('')  # PP_max doesn't aggregate meaningfully
        if options['col_prelievo']:
            total_row.append(f"{df['harvest'].sum():.2f}")
        display_rows.append(total_row)

    return {'snippet': formatter.format_table(headers, display_rows, small=True)}


def render_gpt_graph(data: dict, comparti_df: pd.DataFrame,
                     provv_vol_df: pd.DataFrame, provv_eta_df: pd.DataFrame,
                     output_path: Path, formatter: SnippetFormatter,
                     **options) -> dict:
    """
    Generate harvest horizontal bar graph (@@gpt directive).

    Args:
        data: Output from parcel_data
        comparti_df: Comparto -> Provvigione minima mapping
        provv_vol_df: Volume-based harvest rules
        provv_eta_df: Age-based harvest rules
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        options: per_compresa, per_particella flags (per_genere not supported)

    Returns:
        dict with 'filepath' and 'snippet' keys
    """
    group_cols = []
    if options.get('per_compresa', True):
        group_cols.append('Compresa')
    if options.get('per_particella', True):
        group_cols.append('Particella')

    df = calculate_tpt_table(data, comparti_df, provv_vol_df, provv_eta_df, group_cols)
    if df.empty:
        return {'snippet': '', 'filepath': None}

    if group_cols:
        labels = ['/'.join(str(row[c]) for c in group_cols) for _, row in df.iterrows()]
    else:
        labels = ['Totale']
    values = df['harvest'].values

    # Calculate spacing for compresa groups
    spacing = []
    if 'Compresa' in group_cols and 'Particella' in group_cols:
        comprese = df['Compresa'].tolist()
        for i, c in enumerate(comprese):
            spacing.append(0.3 if i > 0 and c != comprese[i-1] else 0)
    else:
        spacing = [0] * len(labels)

    y_positions = []
    cumulative = 0
    for s in spacing:
        cumulative += s
        y_positions.append(cumulative)
        cumulative += 1

    fig_height = max(3, len(labels) * 0.35 + sum(spacing) * 0.35)
    fig, ax = plt.subplots(figsize=(5, fig_height))

    ax.barh(y_positions, values, color='#0c63e7', height=0.8,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel('Prelievo (m³)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    ax.set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    snippet = formatter.format_image(output_path)
    snippet += '\n' + formatter.format_metadata(data)

    return {'filepath': output_path, 'snippet': snippet}


# =============================================================================
# TEMPLATE PROCESSING
# =============================================================================

def parse_template_directive(line: str) -> Optional[dict]:
    """
    Parse a template directive like @@gci(compresa=Serra, genere=Abete).

    Filter keys (compresa, particella, genere) are always lists (even single values):
        @@gcd(compresa=Serra) -> {'compresa': ['Serra']}
        @@gcd(compresa=Serra, compresa=Fabrizia) -> {'compresa': ['Serra', 'Fabrizia']}

    Other keys remain scalar values.

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

    # Keys that should always be lists (filter parameters + file parameters)
    list_keys = {'compresa', 'particella', 'genere', 'alberi', 'equazioni',
                 'comparti', 'provv_vol', 'provv_eta'}

    params = {}
    if params_str.strip():
        # Split by comma and parse key=value pairs
        for param in params_str.split(','):
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key in list_keys:
                    # Always accumulate into lists
                    if key not in params:
                        params[key] = []
                    params[key].append(value)
                else:
                    # Scalar value (last one wins if repeated)
                    params[key] = value

    return {
        'keyword': keyword,
        'params': params,
        'full_text': full_text
    }

DIRECTIVE_PATTERN = re.compile(r'@@(\w+)\((.*?)\)')
def process_template(template_text: str, data_dir: Path,
                     parcel_file: str,
                     output_dir: Path,
                     format_type: str) -> str:
    """
    Process template by substituting @@directives with generated content.

    Args:
        template_text: Input template
        data_dir: Base directory for data files (alberi, equazioni)
        tree_files: List of tree data files
        equation_files: List of equation data files
        parcel_file: Parcel metadata file
        output_dir: Where to save generated graphs
        format_type: 'html' or 'latex'

    Returns:
        Processed template text
    """
    # Track filenames to make duplicates unique
    filename_counts = defaultdict(int)
    def _build_graph_filename(comprese: list[str], particelle: list[str],
                              generi: list[str], keyword: str) -> str:
        """Build a filename for a graph based on the parameters (all lists)."""
        parts = [keyword]
        if comprese:
            parts.append('-'.join(sorted(comprese)))
        else:
            parts.append('tutte')
        parts.append('-'.join(sorted(particelle)))
        parts.append('-'.join(sorted(generi)))
        base_name = '_'.join(parts)
        filename_counts[base_name] += 1
        return f'{base_name}_{filename_counts[base_name]:02d}.png'

    def process_directive(match):
        directive = parse_template_directive(match.group(0))
        if not directive:
            return match.group(0)  # Return unchanged if parsing fails

        try:
            keyword = directive['keyword']
            params = directive['params']

            if format_type == 'csv' and (keyword.startswith('g') or keyword == 'prop'):
                raise ValueError(f"@@{keyword}: il formato CSV supporta solo direttive @@t* (tabelle)")

            alberi_files = params.get('alberi')
            equazioni_files = params.get('equazioni')

            if not alberi_files and keyword != 'prop':
                raise ValueError(f"@@{keyword} richiede alberi=FILE")
            if keyword == 'gci' and not equazioni_files:
                raise ValueError("@@gci richiede equazioni=FILE")
            elif keyword == 'tpt':
                if not params.get('comparti'):
                    raise ValueError("@@tpt richiede comparti=FILE")
                if not params.get('provv_vol'):
                    raise ValueError("@@tpt richiede provv_vol=FILE")
                if not params.get('provv_eta'):
                    raise ValueError("@@tpt richiede provv_eta=FILE")
            elif keyword == 'gpt':
                if not params.get('comparti'):
                    raise ValueError("@@gpt richiede comparti=FILE")
                if not params.get('provv_vol'):
                    raise ValueError("@@gpt richiede provv_vol=FILE")
                if not params.get('provv_eta'):
                    raise ValueError("@@gpt richiede provv_eta=FILE")

            comprese = params.get('compresa', [])
            particelle = params.get('particella', [])
            generi = params.get('genere', [])

            if keyword == 'prop':
                if len(comprese) != 1 or len(particelle) != 1 or len(params) != 2:
                    raise ValueError("@@prop richiede esattamente compresa=X e particella=Y")
                result = render_prop(particelle_df, comprese[0], particelle[0], formatter)
                return result['snippet']

            trees_df = load_trees(alberi_files, data_dir)
            data = parcel_data(alberi_files, trees_df, particelle_df, comprese, particelle, generi)

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
                    result = render_tsv_table(data, formatter, **options)
                case 'tpt':
                    if params.get('per_genere', None) is not None:
                        raise ValueError("@@tpt non supporta il parametro 'per_genere'")
                    comparti_df = load_csv(params['comparti'], data_dir)
                    provv_vol_df = load_csv(params['provv_vol'], data_dir)
                    provv_eta_df = load_csv(params['provv_eta'], data_dir)
                    options = {
                        'per_compresa': params.get('per_compresa', 'si').lower() == 'si',
                        'per_particella': params.get('per_particella', 'si').lower() == 'si',
                        'col_comparto': params.get('col_comparto', 'si').lower() == 'si',
                        'col_eta': params.get('col_eta', 'si').lower() == 'si',
                        'col_area_ha': params.get('col_area_ha', 'si').lower() == 'si',
                        'col_volume': params.get('col_volume', 'si').lower() == 'si',
                        'col_volume_ha': params.get('col_volume_ha', 'si').lower() == 'si',
                        'col_pp_max': params.get('col_pp_max', 'si').lower() == 'si',
                        'col_prelievo': params.get('col_prelievo', 'si').lower() == 'si',
                        'totali': params.get('totali', 'no').lower() == 'si',
                    }
                    result = render_tpt_table(data, comparti_df, provv_vol_df,
                                              provv_eta_df, formatter, **options)
                case 'gcd':
                    options = {
                        'x_max': int(params.get('x_max', 0)),
                        'y_max': int(params.get('y_max', 0)),
                    }
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gcd_graph(data, output_dir / filename,
                                             formatter, color_map, **options)
                case 'tcd':
                    result = render_tcd_table(data, formatter)
                case 'gci':
                    options = {
                        'x_max': int(params.get('x_max', 0)),
                        'y_max': int(params.get('y_max', 0)),
                    }
                    equations_df = load_csv(equazioni_files, data_dir)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gci_graph(data, equations_df, output_dir / filename,
                                             formatter, color_map, **options)
                case 'gsv':
                    options = {
                        'per_compresa': params.get('per_compresa', 'si').lower() == 'si',
                        'per_particella': params.get('per_particella', 'si').lower() == 'si',
                        'per_genere': params.get('per_genere', 'no').lower() == 'si',
                    }
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gsv_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case 'gpt':
                    if params.get('per_genere', None) is not None:
                        raise ValueError("@@gpt non supporta il parametro 'per_genere'")
                    comparti_df = load_csv(params['comparti'], data_dir)
                    provv_vol_df = load_csv(params['provv_vol'], data_dir)
                    provv_eta_df = load_csv(params['provv_eta'], data_dir)
                    options = {
                        'per_compresa': params.get('per_compresa', 'si').lower() == 'si',
                        'per_particella': params.get('per_particella', 'si').lower() == 'si',
                    }
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gpt_graph(data, comparti_df, provv_vol_df, provv_eta_df,
                                              output_dir / filename, formatter, **options)
                case _:
                    raise ValueError(f"Comando sconosciuto: {keyword}")

            return result['snippet']

        except Exception as e:
            raise ValueError(f"ERRORE nella generazione di {directive['full_text']}: {e}") from e

    match format_type:
        case 'html':
            formatter = HTMLSnippetFormatter()
        case 'csv':
            formatter = CSVSnippetFormatter()
        case 'latex' | 'pdf':
            formatter = LaTeXSnippetFormatter()
        case _:
            raise ValueError(f"Formato non supportato: {format_type}")
    color_map = get_color_map()
    particelle_df = load_csv(parcel_file)

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
    df = load_csv(particelle_file)

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

        # Rare (coral/pink spectrum)
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
    """Generate equations."""
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


def run_calcola_altezze_volumi(args):
    """Calculate heights and volumes in one pass."""
    print(f"Calcolo altezze con equazioni da: {args.equazioni}")
    print("Calcolo volumi con tavole del Tabacchi")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    trees_df = load_trees(args.input)
    equations_df = load_csv(args.equazioni)

    trees_df, updated, unchanged = compute_heights(trees_df, equations_df, verbose=True)
    trees_df = calculate_all_trees_volume(trees_df)
    print(f"\nCalcolo altezze e volumi: {updated} alberi aggiornati, {unchanged} immutati")

    trees_df.sort_values(by=['Compresa', 'Particella', 'Area saggio', 'n'],
        key=lambda col: natsort_keygen()(col) if col.name == 'Particella' else col,
        inplace=True)
    trees_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nFile salvato: {args.output}")


def run_report(args):
    """Generate report from template."""
    print(f"Generazione report formato: {args.formato}")
    print(f"Input: {args.input}")
    print(f"Cartella dati: {args.dati}")
    print(f"Cartella output: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.input, 'r', encoding='utf-8') as f:
        template_text = f.read()

    processed = process_template(template_text, Path(args.dati), args.particelle,
                                 output_dir, args.formato)
    output_file = output_dir / Path(args.input).name
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed)
    if args.formato == 'pdf':
        for _ in range(2):
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', output_file.name],
                cwd=output_dir,
                capture_output=True,
                check=True
            )

        print(f"Report generato: {output_file.with_suffix('.pdf')}")
    else:
        print(f"Report generato: {output_file}")


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

2. CALCOLA ALTEZZE E VOLUMI:
   ./acc.py --calcola-altezze-volumi --equazioni equations.csv \\
            --input alberi.csv --output alberi-calcolati.csv

3. GENERA REPORT:
   ./acc.py --report --formato=html --dati csv/ --particelle particelle.csv \\
            --input template.html --output-dir report/
   (Directives specify alberi=file.csv and equazioni=file.csv)

4. LISTA PARTICELLE:
   ./acc.py --lista-particelle --particelle particelle.csv
"""
    )

    # Mode selection (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--genera-equazioni', action='store_true',
                           help='Genera equazioni di interpolazione')
    run_group.add_argument('--calcola-altezze-volumi', action='store_true',
                           help='Calcola altezze (equazioni) e volumi (Tabacchi)')
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
                            help='File CSV con equazioni (per --calcola-altezze-volumi)')
    files_group.add_argument('--dati',
                            help='Directory base per file dati (per --report)')
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
    report_group.add_argument('--formato', choices=['csv', 'html', 'latex', 'pdf'], default='pdf',
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

    elif args.calcola_altezze_volumi:
        if not args.equazioni:
            parser.error('--calcola-altezze-volumi richiede --equazioni')
        if not args.input:
            parser.error('--calcola-altezze-volumi richiede --input')
        if not args.output:
            parser.error('--calcola-altezze-volumi richiede --output')
        run_calcola_altezze_volumi(args)

    elif args.report:
        if not args.dati:
            parser.error('--report richiede --dati')
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
