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
from dataclasses import dataclass
import io
from pathlib import Path
import re
import subprocess
from typing import Callable, Iterable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsort_keygen
from scipy import stats

SAMPLE_AREA_HA = 0.125
MATURE_THRESHOLD = 20 # Diameter (cm) threshold for "mature" trees (smaller are not harvested)
MIN_TREES_PER_HA = 0.5 # Ignore buckets less than this in classi diametriche graphs.

skip_graphs = False  # Global flag to skip graph generation pylint: disable=invalid-name

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
    def format_image(self, filepath: Path, options: Optional[dict] = None) -> str:
        """Format image reference for this format."""

    @abstractmethod
    def format_metadata(self, data: dict, curve_info: Optional[list] = None) -> str:
        """Format metadata block for this format.

        Args:
            stats: Statistics about the region/species
            curve_info: List of dicts with {species, equation, r_squared, n_points}
                       from equations.csv
        """

    @abstractmethod
    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
        """Format a data table for this format.

        Args:
            headers: Column headers
            rows: Data rows (each row is a list of strings)
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

    def format_image(self, filepath: Path, options: Optional[dict] = None) -> str:
        cls = options['stile'] if options and options['stile'] else 'graph-image'
        return f'<img src="{filepath.name}" class="{cls}">'

    def format_metadata(self, data: dict, curve_info: Optional[list] = None) -> str:
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

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
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

    def format_image(self, filepath: Path, options: Optional[dict] = None) -> str:
        fmt = options['stile'] if options and options['stile'] else 'width=0.5\\textwidth'
        latex = '\\begin{center}\n'
        latex += f'  \\includegraphics[{fmt}]{{{filepath.name}}}\n'
        latex += '\\end{center}\n'
        return latex

    def format_metadata(self, data: dict, curve_info: Optional[list] = None) -> str:
        """Format metadata as LaTeX."""
        if not curve_info:
            return ""
        latex = '\\begin{quote}\\small\n'
        i = 'i' if len(curve_info) > 1 else 'e'
        latex += f'\n\\textbf{{Equazion{i} interpolant{i}:}}\\\\\n'
        for curve in curve_info:
            eq = curve['equation'].replace('*', r'\times ')
            eq = eq.replace('ln', r'\ln')
            latex += (f"{curve['genere']}: ${eq}$ ($R^2$ = {curve['r_squared']:.2f}, "
                        f"$n$ = {curve['n_points']})\\\\\n")
        latex += '\\end{quote}\n'
        return latex

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
        """Format table as LaTeX using longtable for page breaks.
           Headers is a list of tuples (header, justification).
           Justification is 'l' for left, 'r' for right, 'c' for center.
        """
        col_specs = [h[1] for h in headers]
        # Use longtable instead of tabular to allow page breaks
        latex = f'\\begin{{longtable}}{{ {"".join(col_specs)} }}\n'
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

    def format_image(self, filepath: Path, options: Optional[dict] = None) -> str:
        raise NotImplementedError("Formato CSV non supporta immagini (direttive @@g*)")

    def format_metadata(self, data: dict, curve_info: Optional[list] = None) -> str:
        raise NotImplementedError("Formato CSV non supporta metadati")

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
        """Format table as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([h[0] for h in headers])
        writer.writerows(rows)
        return output.getvalue()

    def format_prop(self, short_fields: list[tuple[str, str]],
                    paragraph_fields: list[tuple[str, str]]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([label for label, _ in short_fields + paragraph_fields])
        writer.writerow([value for _, value in short_fields + paragraph_fields])
        return output.getvalue()


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
    def _create_lambda(self, a: float, b: float) -> Callable:
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

    def _create_lambda(self, a: float, b: float) -> Callable:
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

    def _create_lambda(self, a: float, b: float) -> Callable:
        return lambda x: a * x + b

    def _format_equation(self, a: float, b: float) -> str:
        return f"y = {a:.2f}*x + {b:.2f}"


# =============================================================================
# VOLUME CALCULATION (Tabacchi equations)
# =============================================================================

# Species name constants (the known set of Tabacchi species).
SP_ABETE = 'Abete'
SP_ACERO = 'Acero'
SP_CASTAGNO = 'Castagno'
SP_CERRO = 'Cerro'
SP_CILIEGIO = 'Ciliegio'
SP_DOUGLAS = 'Douglas'
SP_FAGGIO = 'Faggio'
SP_LECCIO = 'Leccio'
SP_ONTANO = 'Ontano'
SP_PINO = 'Pino'
SP_PINO_LARICIO = 'Pino Laricio'
SP_PINO_MARITTIMO = 'Pino Marittimo'
SP_PINO_NERO = 'Pino Nero'
SP_SORBO = 'Sorbo'


@dataclass
class TabacchiParams:
    """Volume equation parameters for a single species (from Tabacchi tables)."""
    b: np.ndarray       # coefficient vector
    cov: np.ndarray      # covariance matrix (stored lower-triangular, made symmetric at init)
    n: int               # degrees of freedom (n - 2)
    s2: float            # residual variance

def _make_symmetric(m: np.ndarray) -> np.ndarray:
    """Convert lower-triangular matrix to full symmetric."""
    return m + m.T - np.diag(np.diag(m))

# Single dict of Tabacchi volume equation parameters, keyed by species.
# Adding a species = adding one entry here instead of updating 4 separate dicts.
TABACCHI: dict[str, TabacchiParams] = {
    SP_ABETE: TabacchiParams(
        b   = np.array([ -1.8381,    3.7836e-2, 3.9934e-1 ]),
        cov = _make_symmetric(np.array([
            [  4.9584,     0,         0 ],
            [  1.1274e-3,  7.6175e-7, 0 ],
            [ -7.1820e-1, -2.0243e-4, 1.1287e-1 ]])),
        n   = 46,
        s2  = 1.5284e-5),
    SP_ACERO: TabacchiParams(
        b   = np.array([  1.6905,    3.7082e-2 ]),
        cov = _make_symmetric(np.array([
            [  9.8852e-1,  0],
            [ -4.7366e-4,  8.4075e-7]])),
        n   = 37,
        s2  = 2.2710e-5),
    SP_CASTAGNO: TabacchiParams(
        b   = np.array([ -2,         3.6524e-2, 7.4466e-1 ]),
        cov = _make_symmetric(np.array([
            [  6.5052,     0,         0 ],
            [  2.0090e-3,  1.2430e-6, 0 ],
            [ -1.0771,    -3.9067e-4, 1.9110e-1 ]])),
        n   = 85,
        s2  = 3.0491e-5),
    SP_CERRO: TabacchiParams(
        b   = np.array([ -4.3221e-2, 3.8079e-2 ]),
        cov = _make_symmetric(np.array([
            [  4.5573e-1, 0 ],
            [ -1.8540e-4, 3.6935e-7 ]])),
        n   = 88,
        s2  = 2.5866e-5),
    SP_CILIEGIO: TabacchiParams(
        b   = np.array([  2.3118,    3.1278e-2, 3.7159e-1 ]),
        cov = _make_symmetric(np.array([
            [  1.5377e1,   0,         0 ],
            [  8.9101e-3,  9.7080e-6, 0 ],
            [ -2.6997,    -1.8132e-3, 4.9690e-1 ]])),
        n   = 22,
        s2  = 4.0506e-5),
    SP_DOUGLAS: TabacchiParams(
        b   = np.array([ -7.9946,    3.3343e-2, 1.2186 ]),
        cov = _make_symmetric(np.array([
            [  6.2135e1,   0,         0 ],
            [  6.9406e-3,  1.2592e-6, 0 ],
            [ -6.8517,    -8.2763e-4, 7.7268e-1 ]])),
        n   = 35,
        s2  = 9.0103e-6),
    SP_FAGGIO: TabacchiParams(
        b   = np.array([  8.1151e-1, 3.8965e-2 ]),
        cov = _make_symmetric(np.array([
            [  1.2573,    0 ],
            [ -3.2331e-4, 6.4872e-7 ]])),
        n   = 91,
        s2  = 5.1468e-5),
    SP_LECCIO: TabacchiParams(
        b   = np.array([ -2.2219,    3.9685e-2, 6.2762e-1 ]),
        cov = _make_symmetric(np.array([
            [  8.9968,     0,         0 ],
            [  4.6303e-3,  3.9302e-6, 0 ],
            [ -1.6058,    -9.3376e-4, 3.0078e-1 ]])),
        n   = 83,
        s2  = 6.0915e-5),
    SP_ONTANO: TabacchiParams(
        b   = np.array([ -2.2932e1,  3.2641e-2, 2.991 ]),
        cov = _make_symmetric(np.array([
            [  4.9867e1,   0,         0 ],
            [  1.3116e-2,  5.3498e-6, 0 ],
            [ -7.1964,    -2.0513e-3, 1.0716 ]])),
        n   = 35,
        s2  = 3.9958e-5),
    SP_PINO: TabacchiParams(
        b   = np.array([  6.4383,    3.8594e-2 ]),
        cov = _make_symmetric(np.array([
            [  3.2482,    0 ],
            [ -7.5710e-4, 3.0428e-7 ]])),
        n   = 50,
        s2  = 6.3906e-6),
    SP_PINO_LARICIO: TabacchiParams(
        b   = np.array([  6.4383,    3.8594e-2 ]),
        cov = _make_symmetric(np.array([
            [  3.2482,    0 ],
            [ -7.5710e-4, 3.0428e-7 ]])),
        n   = 50,
        s2  = 6.3906e-6),
    SP_PINO_MARITTIMO: TabacchiParams(
        b   = np.array([  2.9963,    3.8302e-2 ]),
        cov = _make_symmetric(np.array([
            [  2.6524e-1, 0 ],
            [ -1.2270e-4, 5.9640e-7 ]])),
        n   = 26,
        s2  = 1.4031e-5),
    SP_PINO_NERO: TabacchiParams(
        b   = np.array([ -2.1480e1,  3.3448e-2, 2.9088]),
        cov = _make_symmetric(np.array([
            [  2.9797e1,   0,         0 ],
            [  4.5880e-3,  1.3001e-6, 0 ],
            [ -3.0604,    -5.4676e-4, 3.3202e-1 ]])),
        n   = 63,
        s2  = 1.7090e-5),
    SP_SORBO: TabacchiParams(
        b   = np.array([  2.3118,    3.1278e-2, 3.7159e-1 ]),
        cov = _make_symmetric(np.array([
            [  1.5377e1,   0,         0 ],
            [  8.9101e-3,  9.7080e-6, 0 ],
            [ -2.6997,    -1.8132e-3, 4.9690e-1 ]])),
        n   = 22,
        s2  = 4.0506e-5),
}


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
    if genere not in TABACCHI:
        raise ValueError(f"Genere '{genere}' non trovato nelle tavole di Tabacchi")

    b = TABACCHI[genere].b

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
        genere = cast(str, genere)
        group = cast(pd.DataFrame, group)
        if genere not in TABACCHI:
            raise ValueError(f"Genere '{genere}' non presente in Tabacchi")

        n_trees = len(group)
        tp = TABACCHI[genere]
        b, cov, s2 = tp.b, tp.cov, tp.s2
        df = tp.n - 2

        # Build D0 matrix (n_trees x n_coefficients)
        d0 = np.zeros((n_trees, len(b)))
        d_values = cast(np.ndarray, group['D(cm)'].values)
        h_values = cast(np.ndarray, group['h(m)'].values)
        d0[:, 0] = 1
        d0[:, 1] = (d_values ** 2) * h_values
        if len(b) == 3:
            d0[:, 2] = d_values

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
        idx = cast(int, idx)
        row = cast(pd.Series, row)
        genere = row['Genere']
        diameter = row['D(cm)']
        height = row['h(m)']

        if pd.isna(diameter) or pd.isna(height):
            raise ValueError(f"Dati mancanti per riga {idx}: D={diameter}, h={height}")
        if genere not in TABACCHI:
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
    trees_region_species = _filter_df(trees_region, 'Genere', species).copy()
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

    # Compute diameter bucket: D in (0,5] -> 5, D in (5,10] -> 10, etc.
    trees_region_species['Diametro'] = (
        (np.floor((trees_region_species['D(cm)'] - 1) / 5) + 1) * 5).astype(int)

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
        x_values = cast(np.ndarray, group_df['x'].values)
        y_values = cast(np.ndarray, group_df['y'].values)
        regr = RegressionClass()
        if regr.fit(x_values, y_values, min_points=min_points):
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

    groups = list(df.groupby(['Compresa', 'Genere']))
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

    figsize = (4, 3)
    fig, ax = plt.subplots(figsize=figsize)

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

    if not skip_graphs:
        x_max = max(options['x_max'], trees['D(cm)'].max() + 3)
        y_max = max(options['y_max'], (trees['h(m)'].max() + 6) // 5 * 5)
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

    return {
        'filepath': output_path,
        'snippet': snippet
    }


# CLASSI DIAMETRICHE ==========================================================

GCD_Y_LABELS = {
    'alberi_ha': 'Stima alberi / ha',
    'alberi_tot': 'Stima alberi totali',
    'volume_ha': 'Stima volume / ha (m³/ha)',
    'volume_tot': 'Stima volume totale (m³)',
    'G_ha': 'Area basimetrica (m²/ha)',
    'G_tot': 'Area basimetrica totale (m²)',
    'altezza': 'Altezza media (m)',
}

# Coarse bins for @@tcd table
COARSE_BIN0 = "1-30 cm"
COARSE_BIN1 = "31-50 cm"
COARSE_BIN2 = "50+ cm"
COARSE_BINS = [COARSE_BIN0, COARSE_BIN1, COARSE_BIN2]

# Aggregation types for calculate_cd_data
AGG_COUNT, AGG_VOLUME, AGG_BASAL, AGG_HEIGHT = 0, 1, 2, 3


def calculate_cd_data(data: dict, metrica: str, stime_totali: bool,
                       fine: bool = True) -> pd.DataFrame:
    """Calculate diameter class data for @@gcd/@@tcd directives.

    Args:
        data: Output from region_data
        metrica: alberi_*|volume_*|G_*|altezza
        stime_totali: scale by 1/sampled_frac if True (ignored for altezza)
        fine: True for 5cm buckets (5, 10, 15...), False for 3 coarse buckets

    Returns DataFrame indexed by bucket with species as columns.
    """
    trees = data['trees']
    parcels = data['parcels']
    species = data['species']

    if metrica.startswith('volume'):
        agg_type = AGG_VOLUME
    elif metrica.startswith('G'):
        agg_type = AGG_BASAL
    elif metrica == 'altezza':
        agg_type = AGG_HEIGHT
    else:
        agg_type = AGG_COUNT
    per_ha = metrica.endswith('_ha')

    if fine:
        bucket_key = 'Diametro'
    else:
        def coarse_bin(d):
            return COARSE_BIN0 if d <= 30 else COARSE_BIN1 if d <= 50 else COARSE_BIN2
        bucket_key = trees['Diametro'].apply(coarse_bin)

    # For height, compute mean directly (no per-parcel scaling)
    if agg_type == AGG_HEIGHT:
        bucket_vals = trees['Diametro'] if fine else bucket_key
        combined = trees.groupby([bucket_vals, 'Genere'])['h(m)'].mean().unstack(fill_value=0)
        return combined.reindex(columns=species, fill_value=0).sort_index()

    results, area_ha = {}, 0
    for (region, parcel), ptrees in trees.groupby(['Compresa', 'Particella']):
        p = parcels[(region, parcel)]
        area_ha += p['area_ha']
        bucket_vals = ptrees['Diametro'] if fine else cast(pd.Series, bucket_key).loc[ptrees.index]
        if agg_type == AGG_VOLUME:
            agg = ptrees.groupby([bucket_vals, 'Genere'])['V(m3)'].sum().unstack(fill_value=0)
        elif agg_type == AGG_BASAL:
            # Basal area: π/4 * D² in cm², convert to m²
            basal = np.pi / 4 * ptrees['D(cm)'] ** 2 / 10000
            agg = basal.groupby([bucket_vals, ptrees['Genere']]).sum().unstack(fill_value=0)
        else:
            agg = ptrees.groupby([bucket_vals, 'Genere']).size().unstack(fill_value=0)
        if stime_totali:
            agg = agg / p['sampled_frac']
        results[(region, parcel)] = agg

    combined = pd.concat(results.values()).groupby(level=0).sum()
    if per_ha:
        combined = combined / area_ha
    return combined.reindex(columns=species, fill_value=0).sort_index()


def render_gcd_graph(data: dict, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict, **options) -> dict:
    """
    Generate diameter class histograms.

    Args:
        data: Output from region_data
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping
        options: metrica (alberi_*|volume_*|G_*), stime_totali, x_max, y_max

    Returns:
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    if not skip_graphs:
        species = data['species']
        metrica = options['metrica']
        stime_totali = options['stime_totali']

        values_df = calculate_cd_data(data, metrica, stime_totali, fine=True)
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
                ax.plot(nonzero.index, nonzero.values, marker='o', markersize=3, linewidth=1.5,
                        color=color_map[genere], label=genere, alpha=0.85)
            else:
                ax.bar(series.index, series.values, bottom=bottom, width=4,
                    label=genere, color=color_map[genere],
                    alpha=0.8, edgecolor='white', linewidth=0.5)
                bottom += series

        # x_max in cm (fine buckets are 5, 10, 15...)
        max_bucket = values_df.index.max() if len(values_df) > 0 else 50
        x_max = options['x_max'] if options['x_max'] > 0 else max_bucket + 5
        y_max_auto = values_df.max().max() if use_lines else values_df.sum(axis=1).max()
        y_max = options['y_max'] if options['y_max'] > 0 else y_max_auto * 1.1

        ax.set_xlabel('Diametro (cm)')
        ax.set_ylabel(GCD_Y_LABELS[metrica])
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(0, x_max + 1, 10))
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

    return {
        'filepath': output_path,
        'snippet': snippet
    }


def render_tcd_table(data: dict, formatter: SnippetFormatter, **options) -> dict:
    """
    Generate diameter class table (@@tcd directive).

    Creates a table with rows for each species and columns for diameter ranges:
    (0,30], (30,50], (50,max].

    Args:
        data: Output from parcel_data
        formatter: HTML or LaTeX snippet formatter
        options: metrica (alberi_*|volume_*|G_*|altezza), stime_totali

    Returns:
        dict with 'snippet' key containing formatted table
    """
    species = data['species']
    metrica = options['metrica']
    stime_totali = options['stime_totali']
    use_decimals = metrica.startswith('volume') or metrica.startswith('G') or metrica == 'altezza'

    values_df = calculate_cd_data(data, metrica, stime_totali, fine=False)

    headers = [('Genere', 'l')] + [(b, 'r') for b in COARSE_BINS] + [('Totale', 'r')]
    fmt = "{:.2f}" if use_decimals else "{:.0f}"

    rows = []
    col_totals = {b: 0.0 for b in COARSE_BINS}
    for genere in species:
        row = [genere]
        row_total = 0.0
        for b in COARSE_BINS:
            val = cast(float, values_df.at[b, genere]) if b in values_df.index else 0.0
            row.append(fmt.format(val))
            row_total += val
            col_totals[b] += val
        row.append(fmt.format(row_total))
        rows.append(row)

    # Add totals row
    total_row = ['Totale'] + [fmt.format(col_totals[b]) for b in COARSE_BINS]
    total_row.append(fmt.format(sum(col_totals.values())))
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
        ('Piano del taglio', row['Piano del taglio']),
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
                        calc_margin: bool, calc_total: bool,
                        calc_mature: bool = False) -> pd.DataFrame:
    """Calculate the table rows for the @@tsv directive. Returns a DataFrame.

    Args:
        data: Output from parcel_data
        group_cols: Grouping columns (Compresa, Particella, Genere)
        calc_margin: If True, include confidence interval margin columns
        calc_total: If True, scale by 1/sampled_frac to estimate totals
        calc_mature: If True, include volume_mature column (trees with D > 20cm only)
    """
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

        n_trees, volume, vol_mature, margin = 0.0, 0.0, 0.0, 0.0
        if calc_total:
            # First scale per-parcel, then aggregate (sampling density varies across parcels)
            for (region, parcel), ptrees in group_trees.groupby(['Compresa', 'Particella']):
                try:
                    p = parcels[(region, parcel)]
                except KeyError as e:
                    raise ValueError(f"Particella {region}/{parcel} non trovata") from e
                sf = p['sampled_frac']
                n_trees += len(ptrees) / sf
                volume += ptrees['V(m3)'].sum() / sf
                if calc_mature:
                    above_thresh = ptrees[ptrees['D(cm)'] > MATURE_THRESHOLD]
                    vol_mature += above_thresh['V(m3)'].sum() / sf
                if calc_margin:
                    _, pmargin = calculate_volume_confidence_interval(ptrees)
                    margin += pmargin / sf
        else:
            n_trees = len(group_trees)
            volume = group_trees['V(m3)'].sum()
            if calc_mature:
                above_thresh = group_trees[group_trees['D(cm)'] > MATURE_THRESHOLD]
                vol_mature = above_thresh['V(m3)'].sum()
            if calc_margin:
                _, margin = calculate_volume_confidence_interval(group_trees)

        row_dict['n_trees'] = n_trees
        row_dict[COL_VOLUME] = volume
        if calc_mature:
            row_dict[COL_VOLUME_MATURE] = vol_mature
        if calc_margin:
            row_dict['vol_lo'] = volume - margin
            row_dict['vol_hi'] = volume + margin
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if '_' in group_cols:
        group_cols.remove('_')
        df = df.drop(columns=['_'])
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == 'Particella' else col)


def render_tsv_table(data: dict, formatter: SnippetFormatter, **options) -> dict:
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
            - solo_mature: If True, include volume excluding D<=20cm trees
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
        options['intervallo_fiduciario'], options['stime_totali'],
        options['solo_mature'])

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
    if options.get('solo_mature', False):
        headers.append(('Vol. mature (m³)', 'r'))
    if options['intervallo_fiduciario']:
        headers.append(('IF inf (m³)', 'r'))
        headers.append(('IF sup (m³)', 'r'))

    # Format numeric columns as strings.
    df_display = df.copy()
    df_display['n_trees'] = df_display['n_trees'].apply(lambda x: f"{x:.0f}")
    for col in [COL_VOLUME, COL_VOLUME_MATURE, 'vol_lo', 'vol_hi']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")

    # Add totals row. Note that if there are no grouping columns,
    # the result already includes (just) the total row.
    display_rows = df_display.values.tolist()
    if options['totali'] and group_cols:
        total_row = ['Totale'] + [''] * (len(group_cols) - 1)
        total_row.append(f"{df['n_trees'].sum():.0f}")
        total_row.append(f"{df[COL_VOLUME].sum():.2f}")
        if COL_VOLUME_MATURE in df.columns:
            total_row.append(f"{df[COL_VOLUME_MATURE].sum():.2f}")
        if 'vol_lo' in df.columns:
            total_row.append(f"{df['vol_lo'].sum():.2f}")
            total_row.append(f"{df['vol_hi'].sum():.2f}")
        display_rows.append(total_row)

    return {'snippet': formatter.format_table(headers, display_rows)}


def calculate_ip_table(data: dict, group_cols: list[str],
                       stime_totali: bool) -> pd.DataFrame:
    """Calculate the table rows for the @@tip/@@gip directives. Returns a DataFrame.

    For each group (always includes Genere and Diametro bucket), computes:
      - ip_medio: mean of IP across trees in the group
      - incremento_corrente: sum(V(m3)) * mean(IP) / 100
    When stime_totali is True, volumes are scaled by 1/sampled_frac per parcel.
    """
    trees = data['trees']
    parcels = data['parcels']
    for col in (group_cols + ['c(1/a)', 'Ipr(mm)', 'D(cm)', 'V(m3)']):
        if col not in trees.columns:
            raise ValueError(f"@@tip/@@gip richiede la colonna '{col}'. "
                             "Esegui --calcola-incrementi e --calcola-altezze-volumi.")

    all_cols = group_cols + ['Genere', 'Diametro']

    rows = []
    for group_key, group_trees in trees.groupby(all_cols):
        row_dict = dict(zip(all_cols, group_key))
        ip_medio = (group_trees['c(1/a)'] * 2 * group_trees['Ipr(mm)']
                    / 100 / group_trees['D(cm)']).mean()

        if stime_totali:
            volume = 0.0
            for (region, parcel), ptrees in group_trees.groupby(['Compresa', 'Particella']):
                sf = parcels[(region, parcel)]['sampled_frac']
                volume += ptrees['V(m3)'].sum() / sf
        else:
            volume = group_trees['V(m3)'].sum()

        row_dict['ip_medio'] = ip_medio
        row_dict['incremento_corrente'] = volume * ((1 + ip_medio / 100)**2 - 1)
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df.sort_values(
        all_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == 'Particella' else col)


def render_tip_table(data: dict, formatter: SnippetFormatter, **options) -> dict:
    """Generate IP summary table (@@tip directive)."""
    group_cols = []
    if options['per_compresa']:
        group_cols.append('Compresa')
    if options['per_particella']:
        group_cols.append('Particella')

    df = calculate_ip_table(data, group_cols, options['stime_totali'])

    headers = []
    if options['per_compresa']:
        headers.append(('Compresa', 'l'))
    if options['per_particella']:
        headers.append(('Particella', 'l'))
    headers.append(('Genere', 'l'))
    headers.append(('Diametro', 'r'))
    headers.append(('Incr. pct.', 'r'))
    headers.append(('Incr. corr. (m³)', 'r'))

    df_display = df.copy()
    # Convert bucket (5, 10, 15...) to range label ("1-5", "6-10", "11-15"...)
    df_display['Diametro'] = df_display['Diametro'].apply(lambda n: f"{n-4}-{n}")
    df_display['ip_medio'] = df_display['ip_medio'].apply(lambda x: f"{x:.2f}")
    df_display['incremento_corrente'] = df_display['incremento_corrente'].apply(
        lambda x: f"{x:.2f}")

    # Keep only display columns in order
    col_order = [h[0] for h in headers]
    rename = {'ip_medio': 'Incr. pct.',
              'incremento_corrente': 'Incr. corr. (m³)'}
    df_display = df_display.rename(columns=rename)
    df_display = df_display[col_order]
    display_rows = df_display.values.tolist()
    if options['totali']:
        total_row = ['Totale'] + [''] * (len(headers) - 2)
        total_row.append(f"{df['incremento_corrente'].sum():.2f}")
        display_rows.append(total_row)
    rows = [list(map(str, row)) for row in display_rows]
    return {'snippet': formatter.format_table(headers, rows)}


def render_gip_graph(data: dict, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict,
                     **options) -> dict:
    """Generate IP line graph (@@gip directive)."""
    if not skip_graphs:
        group_cols = []
        if options['per_compresa']:
            group_cols.append('Compresa')
        if options['per_particella']:
            group_cols.append('Particella')

        df = calculate_ip_table(data, group_cols, options['stime_totali'])

        metrica = options['metrica']
        if metrica == 'ip':
            y_col, y_label = 'ip_medio', 'Incremento % medio'
        else:
            y_col, y_label = 'incremento_corrente', 'Incremento corrente (m³)'

        # Each curve is a unique (optional compresa, optional particella, genere) tuple
        curve_cols = group_cols + ['Genere']

        fig, ax = plt.subplots(figsize=(5, 3.5))

        for curve_key, curve_df in df.groupby(curve_cols):
            if isinstance(curve_key, str):
                curve_key = (curve_key,)
            label = ' / '.join(str(k) for k in curve_key)
            genere = curve_key[-1]  # last element is always Genere
            curve_df = curve_df.sort_values('Diametro')
            ax.plot(curve_df['Diametro'], curve_df[y_col],
                    marker='o', markersize=3, linewidth=1.5,
                    color=color_map.get(genere, '#0c63e7'),
                    label=label, alpha=0.85)

        ax.set_xlabel('Diametro (cm)')
        ax.set_ylabel(y_label)
        x_max = df['Diametro'].max() + 5
        ax.set_xticks(range(0, x_max + 1, 10))
        ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data)
    return {'filepath': output_path, 'snippet': snippet}


BARH_GROUP_SPACING = 0.3   # Extra vertical gap between compresa groups in bar charts
BARH_INCHES_PER_BAR = 0.35 # Approximate figure height per bar
BARH_MIN_HEIGHT = 3        # Minimum figure height in inches

def _barh_layout(n_bars: int, group_values: Iterable | None = None
                ) -> tuple[list[float], float]:
    """Compute y_positions and fig_height for grouped horizontal bar charts.

    Args:
        n_bars: Number of bars to draw.
        group_values: If provided, insert spacing between consecutive groups
            (e.g. compresa values). Adjacent bars with different group values
            get extra vertical spacing.

    Returns:
        (y_positions, fig_height)
    """
    if group_values is not None:
        groups = list(group_values)
        spacing = [BARH_GROUP_SPACING if i > 0 and groups[i] != groups[i-1] else 0
                   for i in range(len(groups))]
    else:
        spacing = [0] * n_bars

    y_positions = []
    cumulative = 0.0
    for s in spacing:
        cumulative += s
        y_positions.append(cumulative)
        cumulative += 1

    fig_height = max(BARH_MIN_HEIGHT,
                     n_bars * BARH_INCHES_PER_BAR + sum(spacing) * BARH_INCHES_PER_BAR)
    return y_positions, fig_height


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
    if not skip_graphs:
        group_cols = []
        if options['per_compresa']:
            group_cols.append('Compresa')
        if options['per_particella']:
            group_cols.append('Particella')

        # For stacking, we need per-genere data even if displaying by compresa/particella
        stacked = options['per_genere'] and group_cols
        base_cols: list[str] = []
        if stacked:
            base_cols = group_cols.copy()
            group_cols.append('Genere')

        df = calculate_tsv_table(data, group_cols, calc_margin=False, calc_total=True)
        if df.empty:
            return {'snippet': '', 'filepath': None}

        if stacked:
            # Pivot to get genere as columns for stacking
            pivot_df = df.pivot_table(index=base_cols, columns='Genere',
                                    values=COL_VOLUME, fill_value=0)
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

            comprese = (pivot_df.index.get_level_values('Compresa')
                        if 'Compresa' in base_cols and 'Particella' in base_cols
                        else None)
            y_positions, fig_height = _barh_layout(len(labels), comprese)
            fig, ax = plt.subplots(figsize=(5, fig_height))

            # Draw stacked horizontal bars
            left = np.zeros(len(labels))
            for genere in generi:
                values = cast(np.ndarray, pivot_df[genere].values)
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
            if options['per_genere']:
                # Per-genere only: one bar per genere
                labels = df['Genere'].tolist()
            elif group_cols:
                labels = ['/'.join(str(row[c]) for c in group_cols) for _, row in df.iterrows()]
            else:
                labels = ['Totale']
            values = cast(np.ndarray, df[COL_VOLUME].values)

            comprese = (df['Compresa'].tolist()
                        if 'Compresa' in group_cols and 'Particella' in group_cols
                        else None)
            y_positions, fig_height = _barh_layout(len(labels), comprese)
            fig, ax = plt.subplots(figsize=(5, fig_height))

            ax.barh(y_positions, values, color='#0c63e7', height=0.8,
                    edgecolor='white', linewidth=0.5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()

        ax.set_xlabel('Volume (m³)')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data)

    return {'filepath': output_path, 'snippet': snippet}


# RIPRESA =====================================================================

# Column names for tpt (prelievo totale) DataFrames, shared between calculate_ and render_.
COL_SECTOR = 'sector'
COL_AGE = 'age'
COL_AREA_HA = 'area_ha'
COL_VOLUME = 'volume'
COL_VOLUME_MATURE = 'volume_mature'
COL_PP_MAX = 'pp_max'
COL_HARVEST = 'harvest'

def get_age_rule(eta_media: float, provv_eta_df: pd.DataFrame) -> tuple[float, bool]:
    """
    Get the harvest rule for a given age.

    Args:
        eta_media: Mean age of trees in the particella
        provv_eta_df: Age-based harvest rules (Anni, PP_max columns)

    Returns:
        Tuple of (pp_max_percentage, use_volume_rules).
        If use_volume_rules is True, pp_max applies to volume.
        If False, pp_max applies to basal area.
    """
    # Rules are in descending order of Anni, take first match where età >= Anni
    for _, row in provv_eta_df.iterrows():
        if eta_media >= row['Anni']:
            pp_max = row['PP_max']
            # If PP_max is 100%, use volume rules instead of basal area
            use_volume_rules = pp_max >= 100
            return pp_max, use_volume_rules
    return 0, False


def compute_pp_max_volume(volume_mature_per_ha: float,
                          provvigione_minima: float,
                          provv_vol_df: pd.DataFrame) -> float:
    """
    Compute maximum harvest percentage based on volume rules.

    Args:
        volume_mature_per_ha: Volume per hectare excluding D <= 20cm trees
        provvigione_minima: Minimum stock for this comparto (from comparti table)
        provv_vol_df: Volume-based harvest rules (PPM, PP_max columns)

    Returns:
        Maximum harvest percentage (0-100)
    """
    # Rules are in descending order of PPM, take first match where v > PPM * pm / 100
    for _, row in provv_vol_df.iterrows():
        threshold = row['PPM'] * provvigione_minima / 100
        if volume_mature_per_ha > threshold:
            return row['PP_max']
    return 0


def compute_harvest_by_basal_area(trees_df: pd.DataFrame, pp_max_basal: float,
                                   sampled_frac: float) -> tuple[float, float]:
    """
    Compute harvest volume by selecting trees up to a basal area limit.

    Trees are harvested starting from smallest diameter (D > 20cm) until
    the cumulative basal area reaches pp_max_basal% of total basal area.

    Args:
        trees_df: DataFrame with D(cm), V(m3) columns for a single parcel
        pp_max_basal: Maximum percentage of basal area that can be harvested
        sampled_frac: Sampling fraction for this parcel

    Returns:
        Tuple of (volume_mature, harvest_volume), both scaled.
    """
    # Filter to trees above sottomisura threshold
    harvestable = trees_df[trees_df['D(cm)'] > MATURE_THRESHOLD].copy()

    if harvestable.empty:
        return 0.0, 0.0

    # Compute basal area for each tree (m²)
    harvestable['G'] = np.pi / 4 * harvestable['D(cm)'] ** 2 / 10000

    # Total volume and basal area (scaled)
    total_volume = harvestable['V(m3)'].sum() / sampled_frac
    total_basal = harvestable['G'].sum() / sampled_frac

    # Target basal area to harvest
    target_basal = total_basal * pp_max_basal / 100

    # Sort by diameter (smallest first) and accumulate
    harvestable = harvestable.sort_values('D(cm)')

    cumulative_basal = 0.0
    cumulative_volume = 0.0

    for _, tree in harvestable.iterrows():
        tree_basal = tree['G'] / sampled_frac
        tree_volume = tree['V(m3)'] / sampled_frac

        if cumulative_basal + tree_basal > target_basal:
            # This tree would exceed the limit - stop here
            break

        cumulative_basal += tree_basal
        cumulative_volume += tree_volume

    return total_volume, cumulative_volume


def calculate_tpt_table(data: dict, comparti_df: pd.DataFrame,
                        provv_vol_df: pd.DataFrame, provv_eta_df: pd.DataFrame,
                        group_cols: list[str]) -> pd.DataFrame:
    """
    Calculate harvest (prelievo totale) table data for the @@tpt directive.

    Algorithm per particella:
    1. Filter to Fustaia only (ceduo is excluded via Provvigione minima = -1)
    2. Get age-based rule: if age >= 60, use volume rules; else use basal area rules
    3. For volume rules (age >= 60):
       - Compute volume_mature (trees with D > 20cm)
       - Find PP_max from provv_vol based on volume_mature/ha
       - harvest = volume_mature * PP_max / 100
    4. For basal area rules (age < 60):
       - Find PP_max from provv_eta (applies to basal area)
       - Harvest trees D > 20cm from smallest to largest until cumulative
         basal area reaches PP_max% of total basal area

    pp_max is always computed at the particella level. If Genere is in group_cols,
    the harvest breakdown by species uses pro-rata allocation.

    Args:
        data: Output from parcel_data
        comparti_df: Comparto -> Provvigione minima mapping
        provv_vol_df: Volume-based harvest rules
        provv_eta_df: Age-based harvest rules
        group_cols: List of grouping columns (Compresa, Particella, Genere)

    Returns:
        DataFrame with columns depending on group_cols, plus:
        sector, age, area_ha, volume, volume_mature, pp_max, harvest
    """
    #pylint: disable=too-many-locals
    trees = data['trees']
    if 'V(m3)' not in trees.columns:
        raise ValueError("@@tpt richiede dati con volumi (colonna V(m3) mancante). "
                         "Esegui --calcola-altezze-volumi per calcolarli.")
    parcels = data['parcels']

    added_dummy = False
    if not group_cols:
        trees = trees.copy()
        trees['_'] = 'Totale'
        group_cols = ['_']
        added_dummy = True

    per_parcel = 'Particella' in group_cols or len(parcels) == 1
    sector_pm = dict(zip(comparti_df['Comparto'], comparti_df['Provvigione minima']))

    # First pass: compute harvest for each particella
    parcel_info = {}
    for (region, parcel), part_trees in trees.groupby(['Compresa', 'Particella']):
        try:
            p = parcels[(region, parcel)]
        except KeyError as e:
            raise ValueError(f"Particella {region}/{parcel} non trovata") from e

        sector, p_area, sf, age = p['sector'], p['area_ha'], p['sampled_frac'], p['age']
        try:
            provv_min = sector_pm[sector]
        except KeyError as e:
            raise ValueError(f"Comparto {p['sector']} non trovato") from e

        if provv_min < 0:  # Ceduo
            parcel_info[(region, parcel)] = {'skip': True}
            continue

        # Total volume (all trees)
        total_volume = part_trees['V(m3)'].sum() / sf

        # Get age-based rule
        pp_max_age, use_volume_rules = get_age_rule(age, provv_eta_df)

        if use_volume_rules:
            # Age >= threshold: use volume-based rules
            # Compute volume excluding sottomisura
            above_threshold = part_trees[part_trees['D(cm)'] > MATURE_THRESHOLD]
            vol_mature = above_threshold['V(m3)'].sum() / sf

            # Get PP_max from volume rules
            pp_max = compute_pp_max_volume(vol_mature / p_area, provv_min, provv_vol_df)
            harvest = vol_mature * pp_max / 100
        else:
            # Age < threshold: use basal area rules
            pp_max = pp_max_age
            vol_mature, harvest = compute_harvest_by_basal_area(part_trees, pp_max, sf)

        parcel_info[(region, parcel)] = {
            COL_SECTOR: sector, COL_AGE: age, COL_AREA_HA: p_area, 'sf': sf,
            COL_VOLUME: total_volume,
            COL_VOLUME_MATURE: vol_mature,
            COL_PP_MAX: pp_max,
            COL_HARVEST: harvest,
            'skip': False
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

        for (region, parcel), part_trees in group_trees.groupby(['Compresa', 'Particella']):
            info = parcel_info[(region, parcel)]
            if info['skip']:
                continue

            any_tree = True

            # For per-genere breakdown, compute this group's share of the parcel
            if 'Genere' in group_cols:
                # Pro-rata allocation based on volume_mature
                above_thresh = part_trees[part_trees['D(cm)'] > MATURE_THRESHOLD]
                group_vol = part_trees['V(m3)'].sum() / info['sf']
                group_vol_senza = above_thresh['V(m3)'].sum() / info['sf']

                # Fraction of parcel's volume_mature
                if info[COL_VOLUME_MATURE] > 0:
                    frac = group_vol_senza / info[COL_VOLUME_MATURE]
                else:
                    frac = 0

                volume += group_vol
                vol_mature += group_vol_senza
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
    if not group_cols:
        return df
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == 'Particella' else col)


def render_tpt_table(data: dict, comparti_df: pd.DataFrame,
                     provv_vol_df: pd.DataFrame, provv_eta_df: pd.DataFrame,
                     formatter: SnippetFormatter, **options) -> dict:
    """
    Render harvest (prelievo totale) table (@@tpt directive).

    Args:
        data: Output from parcel_data
        comparti_df: Comparto -> Provvigione minima mapping
        provv_vol_df: Volume-based harvest rules
        provv_eta_df: Age-based harvest rules
        formatter: HTML or LaTeX snippet formatter
        options: Dictionary of options:
            - per_compresa, per_particella, per_genere: Grouping flags
            - col_comparto: Show comparto column (default si)
            - col_volume: Show total volume column (default no)
            - col_volume_mature: Show volume excluding D<=20cm (default no)
            - col_volume_ha: Show total volume per hectare column (default si)
            - col_volume_mature_ha: Show volume_mature per hectare column (default si)
            - col_pp_max: Show provvigione percentuale massima column (default si)
            - col_prelievo: Show harvest column (default si)
            - col_prelievo_ha: Show harvest per hectare column (default si)
            - totali: Add totals row

        comparto and pp_max only appear if per_particella is True.
        Note: pp_max is computed from volume_mature (D > 20cm).

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

    df = calculate_tpt_table(data, comparti_df, provv_vol_df, provv_eta_df, group_cols)
    if df.empty:
        return {'snippet': ''}

    per_parcel = 'Particella' in group_cols or len(data['parcels']) == 1

    # When per_genere, area_ha is duplicated across species; dedupe for totals
    if 'Genere' in group_cols:
        parcel_cols = [c for c in group_cols if c != 'Genere'] or [COL_AREA_HA]
        total_area = df.drop_duplicates(subset=parcel_cols)[COL_AREA_HA].sum()
    else:
        total_area = df[COL_AREA_HA].sum()

    # Column spec: (header, align, format_fn, total_fn, condition)
    # format_fn(row) -> str: format one data cell
    # total_fn: column name to sum, callable(df, total_area) -> str, or None (empty cell)
    col_specs = [
        ('Comp.', 'l',
         lambda r: str(r[COL_SECTOR]),
         None,
         options['col_comparto'] and per_parcel),
        ('Età (aa)', 'r',
         lambda r: f"{r[COL_AGE]:.0f}",
         None,
         options['col_eta'] and per_parcel),
        ('Area (ha)', 'r',
         lambda r: f"{r[COL_AREA_HA]:.2f}",
         lambda _df, ta: f"{ta:.2f}",
         options['col_area_ha']),
        ('Vol tot (m³)', 'r',
         lambda r: f"{r[COL_VOLUME]:.2f}",
         COL_VOLUME,
         options['col_volume']),
        ('Vol/ha (m³/ha)', 'r',
         lambda r: f"{r[COL_VOLUME] / r[COL_AREA_HA]:.2f}",
         lambda _df, ta: f"{_df[COL_VOLUME].sum() / ta:.2f}",
         options['col_volume_ha']),
        ('Vol mature (m³)', 'r',
         lambda r: f"{r[COL_VOLUME_MATURE]:.2f}",
         COL_VOLUME_MATURE,
         options['col_volume_mature']),
        ('Vol mature/ha (m³/ha)', 'r',
         lambda r: f"{r[COL_VOLUME_MATURE] / r[COL_AREA_HA]:.2f}",
         lambda _df, ta: f"{_df[COL_VOLUME_MATURE].sum() / ta:.2f}",
         options['col_volume_mature_ha']),
        ('Prelievo \\%', 'r',
         lambda r: f"{r[COL_PP_MAX]:.0f}",
         None,
         options['col_pp_max'] and per_parcel),
        ('Prel tot (m³)', 'r',
         lambda r: f"{r[COL_HARVEST]:.2f}",
         COL_HARVEST,
         options['col_prelievo']),
        ('Prel/ha (m³/ha)', 'r',
         lambda r: f"{r[COL_HARVEST] / r[COL_AREA_HA]:.2f}",
         lambda _df, ta: f"{_df[COL_HARVEST].sum() / ta:.2f}",
         options['col_prelievo_ha']),
    ]
    active_specs = [(h, a, fmt, tot) for h, a, fmt, tot, cond in col_specs if cond]

    # Build headers
    headers = [(col, 'l') for col in group_cols]
    headers += [(h, a) for h, a, _, _ in active_specs]

    # Build display rows
    display_rows = []
    for _, row in df.iterrows():
        display_row = [str(row[col]) for col in group_cols]
        display_row += [fmt(row) for _, _, fmt, _ in active_specs]
        display_rows.append(display_row)

    # Add totals row if requested (and there are grouping columns)
    if options['totali'] and group_cols:
        total_row = ['Totale'] + [''] * (len(group_cols) - 1)
        for _, _, _, tot_fn in active_specs:
            if tot_fn is None:
                total_row.append('')
            elif isinstance(tot_fn, str):
                total_row.append(f"{df[tot_fn].sum():.2f}")
            else:
                total_row.append(tot_fn(df, total_area))
        display_rows.append(total_row)

    return {'snippet': formatter.format_table(headers, display_rows)}


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
    if not skip_graphs:
        group_cols = []
        if options['per_compresa']:
            group_cols.append('Compresa')
        if options['per_particella']:
            group_cols.append('Particella')

        df = calculate_tpt_table(data, comparti_df, provv_vol_df, provv_eta_df, group_cols)
        if df.empty:
            return {'snippet': '', 'filepath': None}

        if group_cols:
            labels = ['/'.join(str(row[c]) for c in group_cols) for _, row in df.iterrows()]
        else:
            labels = ['Totale']
        values = cast(np.ndarray, df[COL_HARVEST].values)

        comprese = (df['Compresa'].tolist()
                    if 'Compresa' in group_cols and 'Particella' in group_cols
                    else None)
        y_positions, fig_height = _barh_layout(len(labels), comprese)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        ax.barh(y_positions, values, color='#0c63e7', height=0.8,
                edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        ax.set_xlabel('Prelievo (m³)')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data)

    return {'filepath': output_path, 'snippet': snippet}


# =============================================================================
# TEMPLATE PROCESSING
# =============================================================================


def check_allowed_params(directive: str, params: dict, options: dict):
    """Check that all keys in params are in options, raise ValueError if not."""
    bad_keys = []
    for key in params.keys():
        if key not in options and key not in ['alberi', 'compresa', 'particella', 'genere']:
            bad_keys.append(key)
    if bad_keys:
        raise ValueError(f"Parametri non validi '{bad_keys}' in @@{directive}")


def check_required_params(directive: str, params: dict, required_keys: list[str]):
    """Check that all keys in required_keys are present in params, raise ValueError if not."""
    missing_keys = []
    for key in required_keys:
        if key not in params:
            missing_keys.append(key)
    if missing_keys:
        raise ValueError(f"Parametri obbligatori mancanti '{missing_keys}' in @@{directive}")


def check_param_values(options: dict, key: str, valid_values: list[str], directive: str):
    """Check that options[key] is in valid_values, raise ValueError if not."""
    value = options.get(key)
    if value not in valid_values:
        raise ValueError(f"Valore non valido per '{key}' in @@{directive}: '{value}'. "
                         f"Valori validi: {', '.join(valid_values)}")


def _bool_opt(params: dict, key: str, enabled: bool = True) -> bool:
    """Parse a si/no boolean option from template directive params."""
    return params.get(key, 'si' if enabled else 'no').lower() == 'si'


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
                     format_type: str,
                     template_dir: Optional[Path] = None) -> str:
    """
    Process template by substituting @@directives with generated content.

    Args:
        template_text: Input template
        data_dir: Base directory for data files (alberi, equazioni)
        parcel_file: Parcel metadata file
        output_dir: Where to save generated graphs
        format_type: 'html' or 'tex'
        template_dir: Directory containing template files (for @@particelle modello)

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

    def render_particelle(comprese: list[str], particelle: list[str],
                          particelle_df: pd.DataFrame, params: dict):
        """
        Render information about all parcels in compresa by filling in a model template.
        """
        if len(comprese) != 1:
            raise ValueError("@@particelle richiede esattamente compresa=X")
        modello = params.get('modello')
        if not modello:
            raise ValueError("@@particelle richiede modello=BASENAME")
        if not template_dir:
            raise ValueError("@@particelle richiede --input per trovare il modello")

        match format_type:
            case 'html': ext = '.html'
            case 'tex' | 'pdf': ext = '.tex'
            case 'csv': ext = '.csv'
            case _: raise ValueError(f"Formato non supportato per @@particelle: {format_type}")
        modello_path = template_dir / (modello + ext)
        if not modello_path.exists():
            raise ValueError(f"Modello non trovato: {modello_path}")
        with open(modello_path, 'r', encoding='utf-8') as f:
            modello_text = f.read()

        compresa = comprese[0]
        parcel_rows = particelle_df[(particelle_df['Compresa'] == compresa) &
                                    (particelle_df['Governo'] == 'Fustaia')]
        parcel_list = sorted(parcel_rows['Particella'].unique(), key=natural_sort_key)
        if particelle:
            parcel_list = [p for p in parcel_list if p in particelle]

        output_parts = []
        for particella in parcel_list:
            expanded = modello_text.replace('@@compresa', compresa)
            expanded = expanded.replace('@@particella', str(particella))
            processed_part = re.sub(DIRECTIVE_PATTERN, process_directive, expanded)
            output_parts.append(processed_part)
        return '\n'.join(output_parts)

    def process_directive(match):
        directive = parse_template_directive(match.group(0))
        if not directive:
            return match.group(0)  # Return unchanged if parsing fails

        try:
            keyword = directive['keyword']
            params = directive['params']

            csv_unsupported = keyword.startswith('g')
            if format_type == 'csv' and csv_unsupported:
                raise ValueError(
                    f"@@{keyword}: il formato CSV non supporta direttive grafiche (@@g*)")

            alberi_files = params.get('alberi')
            equazioni_files = params.get('equazioni')

            if not alberi_files and keyword not in ('prop', 'particelle'):
                raise ValueError(f"@@{keyword} richiede alberi=FILE")

            comprese = params.get('compresa', [])
            particelle = params.get('particella', [])
            generi = params.get('genere', [])

            if keyword == 'prop':
                if len(comprese) != 1 or len(particelle) != 1 or len(params) != 2:
                    raise ValueError("@@prop richiede esattamente compresa=X e particella=Y")
                result = render_prop(particelle_df, comprese[0], particelle[0], formatter)
                return result['snippet']

            if keyword == 'particelle':
                return render_particelle(comprese, particelle, particelle_df, params)

            trees_df = load_trees(alberi_files, data_dir)
            data = parcel_data(alberi_files, trees_df, particelle_df, comprese, particelle, generi)

            match keyword:
                case 'tsv':
                    options = {
                        'per_compresa': _bool_opt(params, 'per_compresa'),
                        'per_particella': _bool_opt(params, 'per_particella'),
                        'per_genere': _bool_opt(params, 'per_genere'),
                        'stime_totali': _bool_opt(params, 'stime_totali'),
                        'intervallo_fiduciario': _bool_opt(params, 'intervallo_fiduciario', False),
                        'solo_mature': _bool_opt(params, 'solo_mature', False),
                        'totali': _bool_opt(params, 'totali', False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_tsv_table(data, formatter, **options)
                case 'tpt':
                    if 'genere' in params:
                        raise ValueError("@@tpt non supporta il parametro 'genere' "
                                         "(usa 'per_genere=si' per raggruppare per specie)")
                    options = {
                        'comparti': True,
                        'provv_vol': True,
                        'provv_eta': True,
                        'per_compresa': _bool_opt(params, 'per_compresa'),
                        'per_particella': _bool_opt(params, 'per_particella'),
                        'per_genere': _bool_opt(params, 'per_genere', False),
                        'col_comparto': _bool_opt(params, 'col_comparto'),
                        'col_eta': _bool_opt(params, 'col_eta'),
                        'col_area_ha': _bool_opt(params, 'col_area_ha'),
                        'col_volume': _bool_opt(params, 'col_volume', False),
                        'col_volume_ha': _bool_opt(params, 'col_volume_ha', False),
                        'col_volume_mature': _bool_opt(params, 'col_volume_mature'),
                        'col_volume_mature_ha': _bool_opt(params, 'col_volume_mature_ha'),
                        'col_pp_max': _bool_opt(params, 'col_pp_max'),
                        'col_prelievo': _bool_opt(params, 'col_prelievo'),
                        'col_prelievo_ha': _bool_opt(params, 'col_prelievo_ha'),
                        'totali': _bool_opt(params, 'totali', False),
                    }
                    check_allowed_params(keyword, params, options)
                    check_required_params(keyword, params, ['comparti', 'provv_vol', 'provv_eta'])
                    comparti_df = load_csv(params['comparti'], data_dir)
                    provv_vol_df = load_csv(params['provv_vol'], data_dir)
                    provv_eta_df = load_csv(params['provv_eta'], data_dir)
                    result = render_tpt_table(data, comparti_df, provv_vol_df,
                                              provv_eta_df, formatter, **options)
                case 'tip':
                    options = {
                        'per_compresa': _bool_opt(params, 'per_compresa', False),
                        'per_particella': _bool_opt(params, 'per_particella', False),
                        'stime_totali': _bool_opt(params, 'stime_totali'),
                        'totali': _bool_opt(params, 'totali', False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_tip_table(data, formatter, **options)
                case 'gip':
                    options = {
                        'per_compresa': _bool_opt(params, 'per_compresa', False),
                        'per_particella': _bool_opt(params, 'per_particella', False),
                        'stime_totali': _bool_opt(params, 'stime_totali'),
                        'metrica': params.get('metrica', 'ip'),
                        'stile': params.get('stile'),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, 'metrica', ['ip', 'ic'], '@@gip')
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gip_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case 'gcd':
                    options = {
                        'x_max': int(params.get('x_max', 0)),
                        'y_max': int(params.get('y_max', 0)),
                        'stile': params.get('stile'),
                        'metrica': params.get('metrica', 'alberi_ha'),
                        'stime_totali': _bool_opt(params, 'stime_totali'),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, 'metrica',
                        ['alberi_ha', 'G_ha', 'volume_ha',
                         'alberi_tot', 'G_tot', 'volume_tot', 'altezza'],
                        '@@gcd')
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gcd_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case 'tcd':
                    options = {
                        'metrica': params.get('metrica', 'alberi_ha'),
                        'stime_totali': _bool_opt(params, 'stime_totali'),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, 'metrica',
                        ['alberi_ha', 'G_ha', 'volume_ha',
                         'alberi_tot', 'G_tot', 'volume_tot', 'altezza'],
                        '@@tcd')
                    result = render_tcd_table(data, formatter, **options)
                case 'gci':
                    options = {
                        'equazioni': True,
                        'x_max': int(params.get('x_max', 0)),
                        'y_max': int(params.get('y_max', 0)),
                        'stile': params.get('stile'),
                    }
                    check_allowed_params(keyword, params, options)
                    check_required_params(keyword, params, ['equazioni'])
                    equations_df = load_csv(equazioni_files, data_dir)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gci_graph(data, equations_df, output_dir / filename,
                                              formatter, color_map, **options)
                case 'gsv':
                    options = {
                        'per_compresa': _bool_opt(params, 'per_compresa'),
                        'per_particella': _bool_opt(params, 'per_particella'),
                        'per_genere': _bool_opt(params, 'per_genere', False),
                        'stile': params.get('stile'),
                    }
                    check_allowed_params(keyword, params, options)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gsv_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case 'gpt':
                    if 'per_genere' in params:
                        raise ValueError("@@gpt non supporta il parametro 'per_genere'")
                    options = {
                        'comparti': True,
                        'provv_vol': True,
                        'provv_eta': True,
                        'per_compresa': _bool_opt(params, 'per_compresa'),
                        'per_particella': _bool_opt(params, 'per_particella'),
                        'stile': params.get('stile'),
                    }
                    check_allowed_params(keyword, params, options)
                    check_required_params(keyword, params, ['comparti', 'provv_vol', 'provv_eta'])
                    comparti_df = load_csv(params['comparti'], data_dir)
                    provv_vol_df = load_csv(params['provv_vol'], data_dir)
                    provv_eta_df = load_csv(params['provv_eta'], data_dir)
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
        case 'tex' | 'pdf':
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


def run_calcola_incrementi(args):
    """Calculate IP (incremento percentuale) for each tree."""
    print("Calcolo incrementi percentuali")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    trees_df = load_trees(args.input)
    trees_df['IP'] = trees_df['c(1/a)'] * 2 * trees_df['Ipr(mm)'] / 100 / trees_df['D(cm)']
    trees_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nFile salvato: {args.output}")


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

    trees_df.sort_values(
        by=['Compresa', 'Particella', 'Area saggio', 'n'],
        key=lambda col: col.map(natsort_keygen()) if col.name == 'Particella' else col,
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
                                 output_dir, args.formato, Path(args.input).parent)
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
    run_group.add_argument('--calcola-incrementi', action='store_true',
                           help='Calcola incrementi percentuali (IP)')
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
    report_group.add_argument('--formato', choices=['csv', 'html', 'tex', 'pdf'], default='pdf',
                             help='Formato output (default: pdf)')
    report_group.add_argument('--ometti-generi-sconosciuti', action='store_true',
                             help='Ometti dai grafici generi per cui non abbiamo equazioni')

    # Other options
    opt_group = parser.add_argument_group('Altre opzioni')
    opt_group.add_argument('--non-rigenerare-grafici', action='store_true', default=False,
                           help='Non rigenerare grafici esistenti (per --report)')

    args = parser.parse_args()

    if args.non_rigenerare_grafici:
        #pylint: disable=global-statement
        global skip_graphs
        skip_graphs = True

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

    elif args.calcola_incrementi:
        if not args.input:
            parser.error('--calcola-incrementi richiede --input')
        if not args.output:
            parser.error('--calcola-incrementi richiede --output')
        run_calcola_incrementi(args)

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
