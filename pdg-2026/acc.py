#!/usr/bin/env python3
# pylint: disable=too-many-lines
# pylint: disable=singleton-comparison
"""
Forest Analysis: estimation of forest characteristics and growth ("accrescimenti").
"""

from abc import ABC, abstractmethod
import argparse
import bisect
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

from harvest_rules import max_harvest, HarvestRulesFunc

SAMPLE_AREA_HA = 0.125
MATURE_THRESHOLD = 20 # Diameter (cm) threshold for "mature" trees (smaller are not harvested)
MIN_TREES_PER_HA = 0.5 # Ignore buckets less than this in classi diametriche graphs.

skip_graphs = False  # Global flag to skip graph generation pylint: disable=invalid-name

# Input DataFrame column names (from CSV files).
# Trees data
COL_DIAMETER_CM = 'D(cm)'
COL_HEIGHT_M = 'h(m)'
COL_V_M3 = 'V(m3)'
COL_GENERE = 'Genere'
COL_COMPRESA = 'Compresa'
COL_PARTICELLA = 'Particella'
COL_DIAMETRO = 'Diametro'       # Computed: diameter bucket (5 cm classes)
COL_FUSTAIA = 'Fustaia'         # Boolean column in trees data
COL_AREA_SAGGIO = 'Area saggio'
COL_COEFF_PRESSLER = 'c'
COL_L10_MM = 'L10(mm)'
# Parcel metadata (particelle_df)
COL_AREA_PARCEL = 'Area (ha)'
COL_COMPARTO = 'Comparto'
COL_GOVERNO = 'Governo'
GOV_FUSTAIA = 'Fustaia'         # Value of COL_GOVERNO (not a column name)
COL_ESPOSIZIONE = 'Esposizione'
COL_STAZIONE = 'Stazione'
COL_SOPRASSUOLO = 'Soprassuolo'
COL_PIANO_TAGLIO = 'Piano del taglio'
COL_ALT_MIN = 'Altitudine min'
COL_ALT_MAX = 'Altitudine max'
COL_LOCALITA = 'Località'
COL_ETA_MEDIA = 'Età media'
# Alsometric (ALS) curve data
COL_DIAM_130 = 'Diam 130cm'
COL_ALT_INDICATIVA = 'Altezza indicativa'

@dataclass
class RenderResult:
    """Result of a render_* function: snippet for template insertion, optional file path."""
    snippet: str
    filepath: Path | None = None

@dataclass
class CurveInfo:
    """Metadata for one regression curve (used in @@gci graph legends)."""
    genere: str
    equation: str
    r_squared: float
    n_points: int

@dataclass
class Directive:
    """A parsed @@keyword(params) template directive."""
    keyword: str
    params: dict
    full_text: str

@dataclass
class ParcelStats:
    """Per-parcel metadata computed from tree data and parcel metadata."""
    area_ha: float
    sector: str
    age: float
    n_sample_areas: int
    sampled_frac: float

@dataclass
class ParcelData:
    """Filtered tree data and associated parcel statistics."""
    trees: pd.DataFrame
    regions: list[str]
    species: list[str]
    parcels: dict[tuple[str, str], ParcelStats]

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
    def format_metadata(self, data: 'ParcelData', curve_info: Optional[list] = None) -> str:
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
        cls = options[OPT_STILE] if options and options[OPT_STILE] else 'graph-image'
        return f'<img src="{filepath.name}" class="{cls}">'

    def format_metadata(self, data: 'ParcelData', curve_info: Optional[list] = None) -> str:
        """Format metadata as HTML."""
        html = '<div class="graph-details">\n'
        html += f'<p><strong>Comprese:</strong> {data.regions}</p>\n'
        html += f'<p><strong>Generi:</strong> {data.species}</p>\n'
        html += f'<p><strong>Alberi campionati:</strong> {data.trees.shape[0]:d}</p>\n'
        if curve_info:
            i = 'i' if len(curve_info) > 1 else 'e'
            html += f'<br><p><strong>Equazion{i} interpolant{i}:</strong></p>\n'
            for curve in curve_info:
                html += (f'<p>{curve.genere}: {curve.equation} '
                         f'(R² = {curve.r_squared:.2f}, n = {curve.n_points})</p>\n')
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
        fmt = options[OPT_STILE] if options and options[OPT_STILE] else 'width=0.5\\textwidth'
        latex = '\\begin{center}\n'
        latex += f'  \\includegraphics[{fmt}]{{{filepath.name}}}\n'
        latex += '\\end{center}\n'
        return latex

    def format_metadata(self, data: 'ParcelData', curve_info: Optional[list] = None) -> str:
        """Format metadata as LaTeX."""
        if not curve_info:
            return ""
        latex = '\\begin{quote}\\small\n'
        i = 'i' if len(curve_info) > 1 else 'e'
        latex += f'\n\\textbf{{Equazion{i} interpolant{i}:}}\\\\\n'
        for curve in curve_info:
            eq = curve.equation.replace('*', r'\times ')
            eq = eq.replace('ln', r'\ln')
            latex += (f"{curve.genere}: ${eq}$ ($R^2$ = {curve.r_squared:.2f}, "
                        f"$n$ = {curve.n_points})\\\\\n")
        latex += '\\end{quote}\n'
        return latex

    def format_table(self, headers: list[tuple[str, str]], rows: list[list[str]]) -> str:
        """Format table as LaTeX using longtable for page breaks.
           Headers is a list of tuples (header, justification).
           Justification is 'l' for left, 'r' for right, 'c' for center.
        """
        just = [h[1] for h in headers]
        # Use longtable instead of tabular to allow page breaks
        latex = f'\\begin{{longtable}}{{ {"".join(just)} }}\n'
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

    def format_metadata(self, data: 'ParcelData', curve_info: Optional[list] = None) -> str:
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

    for genere, group in trees_df.groupby(COL_GENERE):
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
        d_values = cast(np.ndarray, group[COL_DIAMETER_CM].values)
        h_values = cast(np.ndarray, group[COL_HEIGHT_M].values)
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
        total_volume += group[COL_V_M3].sum()
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
    required = [COL_DIAMETER_CM, COL_HEIGHT_M, COL_GENERE]
    missing = [col for col in required if col not in trees_df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    # Validate: no missing data
    na_mask = trees_df[COL_DIAMETER_CM].isna() | trees_df[COL_HEIGHT_M].isna()
    if na_mask.any():
        idx = na_mask.idxmax()
        raise ValueError(f"Dati mancanti per riga {idx}: "
                         f"D={trees_df.at[idx, COL_DIAMETER_CM]}, "
                         f"h={trees_df.at[idx, COL_HEIGHT_M]}")

    result_df = trees_df.copy()
    result_df[COL_V_M3] = 0.0

    for genere, group in result_df.groupby(COL_GENERE):
        if genere not in TABACCHI:
            raise ValueError(f"Genere '{genere}' non trovato nelle tavole di Tabacchi")
        b = TABACCHI[genere].b
        d = group[COL_DIAMETER_CM]
        h = group[COL_HEIGHT_M]
        d2h = d ** 2 * h
        if len(b) == 2:
            vol = (b[0] + b[1] * d2h) / 1000
        else:
            vol = (b[0] + b[1] * d2h + b[2] * d) / 1000
        result_df.loc[group.index, COL_V_M3] = vol

    return result_df


def diameter_class(d: pd.DataFrame, width: int = 5) -> pd.DataFrame:
    """
    Returns a dataframe containing the width-cm diameter classes corresponding to
    the diameters (in cm) in d.

    For example, with width = 5, D in (2.5, 7.5] -> 5, D in (7.5, 12.5] -> 10.
    """
    return (np.ceil((d - (width/2)) / width) * width).astype(int)


# =============================================================================
# DATA PREPARATION
# =============================================================================

region_cache = {}
def parcel_data(tree_files: list[str], tree_df: pd.DataFrame, parcel_df: pd.DataFrame,
                regions: list[str], parcels: list[str], species: list[str]) -> 'ParcelData':
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
    trees_region = _filter_df(trees_region, COL_COMPRESA, regions)
    trees_region = _filter_df(trees_region, COL_PARTICELLA, parcels)
    trees_region_species = _filter_df(trees_region, COL_GENERE, species).copy()
    if len(trees_region_species) == 0:
        raise ValueError(f"Nessun dato trovato per comprese '{regions}' " +
                         f"particelle '{parcels}' generi '{species}'")

    parcel_stats = {}
    for (region, parcel), trees in trees_region.groupby([COL_COMPRESA, COL_PARTICELLA]):
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
            n_sample_areas=n_sample_areas,
            sampled_frac=sampled_frac,
        )

    trees_region_species[COL_DIAMETRO] = diameter_class(trees_region_species[COL_DIAMETER_CM])

    data = ParcelData(
        trees=trees_region_species,
        regions=sorted(trees_region[COL_COMPRESA].unique()),
        species=sorted(trees_region_species[COL_GENERE].unique()),
        parcels=parcel_stats,
    )
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
    df.drop(df[df[COL_FUSTAIA]==False].index,inplace=True)
    df[COL_PARTICELLA] = df[COL_PARTICELLA].astype(str)
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
    df['x'] = df[COL_DIAMETER_CM]
    df['y'] = df[COL_HEIGHT_M]
    groups = []

    for (compresa, genere), group in df.groupby([COL_COMPRESA, COL_GENERE]):
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
    df['x'] = df[COL_DIAMETER_CM]
    df['y'] = df[COL_HEIGHT_M]

    groups = list(df.groupby([COL_COMPRESA, COL_GENERE]))
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
    comprese = sorted(df_particelle[COL_COMPRESA].dropna().unique())

    df_als = load_csv(tabelle_file)
    df_als[COL_DIAM_130] = pd.to_numeric(df_als[COL_DIAM_130], errors='coerce')
    df_als[COL_ALT_INDICATIVA] = pd.to_numeric(df_als[COL_ALT_INDICATIVA], errors='coerce')
    df_als['x'] = df_als[COL_DIAM_130]
    df_als['y'] = df_als[COL_ALT_INDICATIVA]

    groups = []
    for compresa in comprese:
        for genere, group in df_als.groupby(COL_GENERE):
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
    result_df[COL_HEIGHT_M] = result_df[COL_HEIGHT_M].astype(float)

    for (compresa, genere), group in trees_df.groupby([COL_COMPRESA, COL_GENERE]):
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
        diameters = trees_df.loc[indices, COL_DIAMETER_CM].astype(float)

        if eq['funzione'] == 'ln':
            new_heights = eq['a'] * np.log(np.maximum(diameters, 0.1)) + eq['b']
        else:  # 'lin'
            new_heights = eq['a'] * diameters + eq['b']

        result_df.loc[indices, COL_HEIGHT_M] = new_heights.astype(float)
        trees_updated += len(group)

        if verbose:
            print(f"  {compresa} - {genere}: {len(group)} alberi aggiornati")

    return result_df, trees_updated, trees_unchanged


# =============================================================================
# RENDERING OF INDIVIDUAL DIRECTIVES
# =============================================================================

# CURVE IPSOMETRICHE ==========================================================


def render_gci_graph(data: ParcelData, equations_df: pd.DataFrame,
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
        Dict with keys:
        - 'filepath': Path to generated PNG
        - 'snippet': Formatted HTML/LaTeX snippet for template substitution
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
        x = sp_data[COL_DIAMETER_CM].values
        y = sp_data[COL_HEIGHT_M].values
        ax.scatter(x, y, color=color_map[sp], label=sp, alpha=0.7, linewidth=2, s=1)

    # Second pass: regression curves (per compresa/genere pair)
    curve_info = []
    for region in regions:
        for sp in species:
            sp_data = trees[trees[COL_GENERE] == sp]
            x = sp_data[COL_DIAMETER_CM].values

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

                curve_info.append(CurveInfo(
                    genere=sp,
                    equation=eq_str,
                    r_squared=eq['r2'],
                    n_points=int(eq['n']),
                ))

    if not skip_graphs:
        x_max = max(options[OPT_X_MAX], trees[COL_DIAMETER_CM].max() + 3)
        y_max = max(options[OPT_Y_MAX], (trees[COL_HEIGHT_M].max() + 6) // 5 * 5)
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


def calculate_cd_data(data: ParcelData, metrica: str, stime_totali: bool,
                       fine: bool = True) -> pd.DataFrame:
    """Calculate diameter class data for @@gcd/@@tcd directives.

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
        bucket_key = COL_DIAMETRO
    else:
        def coarse_bin(d):
            return COARSE_BIN0 if d <= 30 else COARSE_BIN1 if d <= 50 else COARSE_BIN2
        bucket_key = trees[COL_DIAMETRO].apply(coarse_bin)

    # For height, compute mean directly (no per-parcel scaling)
    if agg_type == AGG_HEIGHT:
        bucket_vals = trees[COL_DIAMETRO] if fine else bucket_key
        combined = trees.groupby([bucket_vals, COL_GENERE])[COL_HEIGHT_M].mean().unstack(fill_value=0)
        return combined.reindex(columns=species, fill_value=0).sort_index()

    results, area_ha = {}, 0
    for (region, parcel), ptrees in trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
        p = parcels[(region, parcel)]
        area_ha += p.area_ha
        bucket_vals = ptrees[COL_DIAMETRO] if fine else cast(pd.Series, bucket_key).loc[ptrees.index]
        if agg_type == AGG_VOLUME:
            agg = ptrees.groupby([bucket_vals, COL_GENERE])[COL_V_M3].sum().unstack(fill_value=0)
        elif agg_type == AGG_BASAL:
            # Basal area: π/4 * D² in cm², convert to m²
            basal = np.pi / 4 * ptrees[COL_DIAMETER_CM] ** 2 / 10000
            agg = basal.groupby([bucket_vals, ptrees[COL_GENERE]]).sum().unstack(fill_value=0)
        else:
            agg = ptrees.groupby([bucket_vals, COL_GENERE]).size().unstack(fill_value=0)
        if stime_totali:
            agg = agg / p.sampled_frac
        results[(region, parcel)] = agg

    combined = pd.concat(results.values()).groupby(level=0).sum()
    if per_ha:
        combined = combined / area_ha
    return combined.reindex(columns=species, fill_value=0).sort_index()


def render_gcd_graph(data: ParcelData, output_path: Path,
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
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    if not skip_graphs:
        species = data.species
        metrica = options[OPT_METRICA]
        stime_totali = options[OPT_STIME_TOTALI]

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
        x_max = options[OPT_X_MAX] if options[OPT_X_MAX] > 0 else max_bucket + 5
        y_max_auto = values_df.max().max() if use_lines else values_df.sum(axis=1).max()
        y_max = options[OPT_Y_MAX] if options[OPT_Y_MAX] > 0 else y_max_auto * 1.1

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

    return RenderResult(filepath=output_path, snippet=snippet)


def render_tcd_table(data: ParcelData, formatter: SnippetFormatter, **options) -> RenderResult:
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
    species = data.species
    metrica = options[OPT_METRICA]
    stime_totali = options[OPT_STIME_TOTALI]
    use_decimals = metrica.startswith('volume') or metrica.startswith('G') or metrica == 'altezza'

    values_df = calculate_cd_data(data, metrica, stime_totali, fine=False)

    headers = [(COL_GENERE, 'l')] + [(b, 'r') for b in COARSE_BINS] + [('Totale', 'r')]
    fmt = "{:.1f}" if use_decimals else "{:.0f}"

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

    return RenderResult(snippet=formatter.format_table(headers, rows))


def render_prop(particelle_df: pd.DataFrame, compresa: str, particella: str,
                formatter: SnippetFormatter) -> RenderResult:
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
    row = particelle_df[(particelle_df[COL_COMPRESA] == compresa) &
                        (particelle_df[COL_PARTICELLA] == particella)]
    if row.empty:
        raise ValueError(f"Particella '{particella}' non trovata in compresa '{compresa}'")
    row = row.iloc[0]

    area = f"{row[COL_AREA_PARCEL]:.2f} ha"
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


# STIMA VOLUMI ================================================================

# DataFrame column names shared between calculate_* and render_* functions.
# Common columns (used across tsv, tpt, gsv, gpt)
COL_VOLUME = 'volume'
COL_VOLUME_MATURE = 'volume_mature'
COL_AREA_HA = 'area_ha'
# tsv-specific
COL_N_TREES = 'n_trees'
COL_VOL_LO = 'vol_lo'
COL_VOL_HI = 'vol_hi'
# tip-specific
COL_IP_MEDIO = 'ip_medio'
COL_INCR_CORR = 'incremento_corrente'
COL_DELTA_D = 'delta_d'
# tcr-specific
COL_VOLUME_PROJ = 'volume_proj'
COL_VOLUME_MATURE_PROJ = 'volume_mature_proj'
# tpt-specific
COL_SECTOR = 'sector'
COL_AGE = 'age'
COL_PP_MAX = 'pp_max'
COL_HARVEST = 'harvest'
ROW_TOTAL = 'Totale'

# Option keys shared between process_directive and render_*/calculate_* functions.
# Common options (used across multiple directives)
OPT_PER_COMPRESA = 'per_compresa'
OPT_PER_PARTICELLA = 'per_particella'
OPT_PER_GENERE = 'per_genere'
OPT_STIME_TOTALI = 'stime_totali'
OPT_TOTALI = 'totali'
OPT_STILE = 'stile'
OPT_METRICA = 'metrica'
# tsv-specific
OPT_INTERVALLO_FIDUCIARIO = 'intervallo_fiduciario'
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
# tcr-specific
OPT_ANNI = 'anni'
OPT_MORTALITA = 'mortalita'
# Required file parameters (used as option keys for validation)
OPT_EQUAZIONI = 'equazioni'

# Output format types
class Fmt:
    HTML = 'html'
    TEX  = 'tex'
    PDF  = 'pdf'
    CSV  = 'csv'

# Template directive keywords
class Dir:
    GCI  = 'gci'
    GCD  = 'gcd'
    TCD  = 'tcd'
    TSV  = 'tsv'
    GSV  = 'gsv'
    TPT  = 'tpt'
    GPT  = 'gpt'
    TIP  = 'tip'
    GIP  = 'gip'
    TCR  = 'tcr'
    PROP = 'prop'
    PARTICELLE = 'particelle'

# Column spec for table rendering
@dataclass
class ColSpec:
    # Column title
    title: str
    # Alignment: 'l', 'r', or 'c'
    align: str
    # Format spec: column name to display as .1f, or custom format function
    format: str | Callable
    # Total spec: column name to sum, or callable(df) -> str, or None for no total
    total: str | Callable | None
    # True if column should be rendered
    enabled: bool

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
    headers = [(col, 'l') for col in group_cols]
    headers += [(c.title, c.align) for c in col_specs]

    display_rows = []
    for _, row in df.iterrows():
        display_row = [str(row[col]) for col in group_cols]
        for c in col_specs:
            if isinstance(c.format, str):
                display_row.append(f"{row[c.format]:.1f}")
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
                total_row.append(f"{df[c.total].sum():.1f}")
            else:
                total_row.append(c.total(df))
        display_rows.append(total_row)

    return RenderResult(snippet=formatter.format_table(headers, display_rows))


def calculate_tsv_table(data: ParcelData, group_cols: list[str],
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
    trees = data.trees
    if COL_V_M3 not in trees.columns:
        raise ValueError("@@tsv richiede dati con volumi (manca la colonna COL_V_M3). "
                         "Esegui --calcola-altezze-volumi per calcolarli.")
    parcels = data.parcels

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
            for (region, parcel), ptrees in group_trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
                try:
                    p = parcels[(region, parcel)]
                except KeyError as e:
                    raise ValueError(f"Particella {region}/{parcel} non trovata") from e
                sf = p.sampled_frac
                n_trees += len(ptrees) / sf
                volume += ptrees[COL_V_M3].sum() / sf
                if calc_mature:
                    above_thresh = ptrees[ptrees[COL_DIAMETER_CM] > MATURE_THRESHOLD]
                    vol_mature += above_thresh[COL_V_M3].sum() / sf
                if calc_margin:
                    _, pmargin = calculate_volume_confidence_interval(ptrees)
                    margin += pmargin / sf
        else:
            n_trees = len(group_trees)
            volume = group_trees[COL_V_M3].sum()
            if calc_mature:
                above_thresh = group_trees[group_trees[COL_DIAMETER_CM] > MATURE_THRESHOLD]
                vol_mature = above_thresh[COL_V_M3].sum()
            if calc_margin:
                _, margin = calculate_volume_confidence_interval(group_trees)

        row_dict[COL_N_TREES] = n_trees
        row_dict[COL_VOLUME] = volume
        if calc_mature:
            row_dict[COL_VOLUME_MATURE] = vol_mature
        if calc_margin:
            row_dict[COL_VOL_LO] = volume - margin
            row_dict[COL_VOL_HI] = volume + margin
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if '_' in group_cols:
        group_cols.remove('_')
        df = df.drop(columns=['_'])
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)


def render_tsv_table(data: ParcelData, formatter: SnippetFormatter, **options) -> RenderResult:
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
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    if options[OPT_PER_GENERE]:
        group_cols.append(COL_GENERE)

    df = calculate_tsv_table(data, group_cols,
        options[OPT_INTERVALLO_FIDUCIARIO], options[OPT_STIME_TOTALI],
        options[OPT_SOLO_MATURE])

    has_ci = options[OPT_INTERVALLO_FIDUCIARIO]
    col_specs = [
        ColSpec('N. Alberi', 'r',
                lambda r: f"{r[COL_N_TREES]:.0f}",
                lambda d: f"{d[COL_N_TREES].sum():.0f}", True),
        ColSpec('Volume (m³)', 'r', COL_VOLUME, COL_VOLUME, True),
        ColSpec('Vol. mature (m³)', 'r', COL_VOLUME_MATURE, COL_VOLUME_MATURE,
                options.get(OPT_SOLO_MATURE, False)),
        ColSpec('IF inf (m³)', 'r', COL_VOL_LO, COL_VOL_LO, has_ci),
        ColSpec('IF sup (m³)', 'r', COL_VOL_HI, COL_VOL_HI, has_ci),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


def calculate_growth_rates(data: ParcelData, group_cols: list[str],
                           stime_totali: bool) -> pd.DataFrame:
    """Calculate the table rows for the @@tip/@@gip directives. Returns a DataFrame.

    group_cols must include COL_GENERE and COL_DIAMETRO.  Computes per group:
      - ip_medio: mean Pressler percentage increment
      - delta_d: mean annual diameter increment (cm)
      - incremento_corrente: volume * ip/100
    When stime_totali is True, volumes are scaled by 1/sampled_frac per parcel.
    """
    trees = data.trees
    parcels = data.parcels
    for col in (COL_GENERE, COL_DIAMETRO):
        if col not in group_cols:
            raise ValueError(f"group_cols deve includere '{col}'")
    for col in (group_cols + [COL_COEFF_PRESSLER, COL_L10_MM, COL_DIAMETER_CM, COL_V_M3]):
        if col not in trees.columns:
            raise ValueError(f"Direttiva richiede la colonna '{col}'. "
                             "Esegui --calcola-incrementi e --calcola-altezze-volumi.")

    rows = []
    for group_key, group_trees in trees.groupby(group_cols):
        row_dict = dict(zip(group_cols, group_key))
        ip_medio = (group_trees[COL_COEFF_PRESSLER] * 2 * group_trees[COL_L10_MM]
                    / 100 / group_trees[COL_DIAMETER_CM]).mean()
        delta_d = (2 * group_trees[COL_L10_MM] / 100).mean()

        if stime_totali:
            volume = 0.0
            for (region, parcel), ptrees in group_trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
                sf = parcels[(region, parcel)].sampled_frac
                volume += ptrees[COL_V_M3].sum() / sf
        else:
            volume = group_trees[COL_V_M3].sum()

        row_dict[COL_IP_MEDIO] = ip_medio
        row_dict[COL_DELTA_D] = delta_d
        row_dict[COL_INCR_CORR] = volume * ip_medio / 100
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df.sort_values(
        group_cols,
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)


def render_tip_table(data: ParcelData, formatter: SnippetFormatter, **options) -> RenderResult:
    """Generate IP summary table (@@tip directive)."""
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    group_cols += [COL_GENERE, COL_DIAMETRO]

    df = calculate_growth_rates(data, group_cols, options[OPT_STIME_TOTALI])

    col_specs = [
        ColSpec('Genere', 'l', lambda r: str(r[COL_GENERE]), None, True),
        ColSpec('D (cm)', 'r', lambda r: f"{r[COL_DIAMETRO]}", None, True),
        ColSpec('Incr. pct.', 'r', COL_IP_MEDIO, None, True),
        ColSpec('Incr. corr. (m³)', 'r', COL_INCR_CORR, COL_INCR_CORR, True),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


def render_gip_graph(data: ParcelData, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict,
                     **options) -> RenderResult:
    """Generate IP line graph (@@gip directive)."""
    if not skip_graphs:
        group_cols = []
        if options[OPT_PER_COMPRESA]:
            group_cols.append(COL_COMPRESA)
        if options[OPT_PER_PARTICELLA]:
            group_cols.append(COL_PARTICELLA)
        group_cols += [COL_GENERE, COL_DIAMETRO]

        df = calculate_growth_rates(data, group_cols, options[OPT_STIME_TOTALI])

        metrica = options[OPT_METRICA]
        if metrica == 'ip':
            y_col, y_label = COL_IP_MEDIO, 'Incremento % medio'
        else:
            y_col, y_label = COL_INCR_CORR, 'Incremento corrente (m³)'

        # Each curve is a unique (optional compresa, optional particella, genere) tuple
        curve_cols = [c for c in group_cols if c != COL_DIAMETRO]

        fig, ax = plt.subplots(figsize=(5, 3.5))

        for curve_key, curve_df in df.groupby(curve_cols):
            if isinstance(curve_key, str):
                curve_key = (curve_key,)
            label = ' / '.join(str(k) for k in curve_key)
            genere = curve_key[-1]  # last element is always Genere
            curve_df = curve_df.sort_values(COL_DIAMETRO)
            ax.plot(curve_df[COL_DIAMETRO], curve_df[y_col],
                    marker='o', markersize=3, linewidth=1.5,
                    color=color_map.get(genere, '#0c63e7'),
                    label=label, alpha=0.85)

        ax.set_xlabel('Diametro (cm)')
        ax.set_ylabel(y_label)
        x_max = df[COL_DIAMETRO].max() + 5
        ax.set_xticks(range(0, x_max + 1, 10))
        ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    snippet = formatter.format_image(output_path, options)
    snippet += '\n' + formatter.format_metadata(data)
    return RenderResult(filepath=output_path, snippet=snippet)


def year_step(sim: pd.DataFrame, weight: np.ndarray,
              growth_by_group: dict, available_diams: dict,
              groupby_cols: list[str], mortalita: float,
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

    def lookup_growth(sim: pd.DataFrame, growth_by_group: dict,
                      available_diams: dict,
                      groupby_cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        """Look up percent growth and diameter change each tree row from the growth table.

        Returns (inc_pct, delta_d, fallback_count).  When a tree's exact diameter bucket
        is not in the growth table, falls back to the nearest available bucket.
        """
        prefix_cols = [c for c in groupby_cols if c != COL_DIAMETRO]

        keys = list(zip(*(sim[c] for c in groupby_cols)))
        growth = [growth_by_group.get(k) for k in keys]
        inc_pct = np.array([r[0] if r else np.nan for r in growth])
        delta_d = np.array([r[1] if r else np.nan for r in growth])

        missing = np.isnan(inc_pct)
        fallbacks = int(missing.sum())
        for i in np.where(missing)[0]:
            row = sim.iloc[i]
            prefix = tuple(row[c] for c in prefix_cols)
            diams = available_diams.get(prefix)
            if not diams:
                inc_pct[i], delta_d[i] = 0.0, 0.0
                continue
            nearest = find_nearest_diameter(diams, int(row[COL_DIAMETRO]))
            inc_pct[i], delta_d[i] = growth_by_group.get(prefix + (nearest,), (0.0, 0.0))

        return inc_pct, delta_d, fallbacks

    if diam_growth is not None:
        sim[COL_DIAMETER_CM] = sim[COL_DIAMETER_CM].values + diam_growth
        sim[COL_DIAMETRO] = diameter_class(sim[COL_DIAMETER_CM])

    inc_pct, delta_d, fallbacks = lookup_growth(
        sim, growth_by_group, available_diams, groupby_cols)
    sim[COL_V_M3] = sim[COL_V_M3].values * (1 + inc_pct / 100)
    weight *= (1 - mortalita / 100)

    return delta_d, fallbacks


def calculate_tcr_table(data: ParcelData, group_cols: list[str],
                        years: int, mortalita: float) -> pd.DataFrame:
    """Compute growth projection table: year-0 vs year-N volumes per group.

    Runs a per-tree simulation for `years` years using Pressler growth rates,
    with optional mortality and height re-estimation.
    """
    #pylint: disable=too-many-locals
    trees = data.trees
    parcels = data.parcels

    # Year-0 volumes
    df0 = calculate_tsv_table(data, list(group_cols),
                              calc_margin=False, calc_total=True, calc_mature=True)

    # Growth rates by (compresa, [particella,] genere, diametro)
    groupby_cols = [COL_COMPRESA]
    groupby_cols = groupby_cols + [COL_GENERE, COL_DIAMETRO]
    growth_df = calculate_growth_rates(data, groupby_cols, stime_totali=True)

    # Build lookup dict and available-diameters index for fallback
    growth_by_group = {}
    available_diams = defaultdict(list)
    for _, row in growth_df.iterrows():
        key = tuple(row[c] for c in groupby_cols)
        growth_by_group[key] = (row[COL_IP_MEDIO], row[COL_DELTA_D])
        prefix = key[:-1]
        available_diams[prefix].append(int(row[COL_DIAMETRO]))
    for prefix in available_diams:
        available_diams[prefix] = sorted(set(available_diams[prefix]))

    # Copy tree records for simulation
    sim = trees[[COL_COMPRESA, COL_PARTICELLA, COL_AREA_SAGGIO,
                 COL_GENERE, COL_DIAMETER_CM, COL_V_M3, COL_DIAMETRO]].copy()
    weight = np.ones(len(sim))

    fallback_count = 0
    diam_growth = None
    for _ in range(years):
        diam_growth, fallbacks = year_step(
            sim, weight, growth_by_group, available_diams,
            groupby_cols, mortalita, diam_growth)
        fallback_count += fallbacks

    # Bake mortality into volume
    sim[COL_V_M3] = sim[COL_V_M3].values * weight

    # Year-N volumes via same aggregation
    sim_data = ParcelData(trees=sim, regions=data.regions,
                          species=data.species, parcels=data.parcels)
    dfN = calculate_tsv_table(sim_data, list(group_cols),
                              calc_margin=False, calc_total=True, calc_mature=True)

    # Merge year-0 and year-N
    dfN = dfN.rename(columns={
        COL_VOLUME_MATURE: COL_VOLUME_MATURE_PROJ,
    }).drop(columns=[COL_N_TREES])
    if group_cols:
        result = df0.merge(dfN, on=group_cols)
    else:
        result = pd.concat([df0.reset_index(drop=True),
                           dfN.reset_index(drop=True)], axis=1)

    # Per-tree ic using compresa-level growth rates, restricted to mature
    # trees (D > threshold) to match the volume columns.
    small_trees = trees[COL_DIAMETER_CM] <= MATURE_THRESHOLD
    tree_keys = list(zip(*(trees[c] for c in groupby_cols)))
    tree_ip = np.array([growth_by_group.get(k, (0, 0))[0] for k in tree_keys])
    tree_ic = pd.Series(trees[COL_V_M3].values * tree_ip / 100, index=trees.index)
    tree_ic[small_trees] = 0.0

    # Add area_ha and incremento corrente per group
    meta_rows = []
    for group_key, group_trees in (
            trees.groupby(group_cols) if group_cols else [('_', trees)]):
        row = {}
        if group_cols:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = dict(zip(group_cols, group_key))
        area, ic_sum = 0.0, 0.0
        for (region, parcel), ptrees in group_trees.groupby(
                [COL_COMPRESA, COL_PARTICELLA]):
            p = parcels[(region, parcel)]
            area += p.area_ha
            ic_sum += tree_ic[ptrees.index].sum() / p.sampled_frac
        row[COL_AREA_HA] = area
        row[COL_INCR_CORR] = ic_sum
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows)
    if group_cols:
        result = result.merge(meta_df, on=group_cols)
    else:
        result[COL_AREA_HA] = meta_df[COL_AREA_HA].iloc[0]
        result[COL_INCR_CORR] = meta_df[COL_INCR_CORR].iloc[0]

    if fallback_count:
        total_lookups = len(sim) * years
        pct = 100 * fallback_count / total_lookups
        print(f"  @@tcr: {pct:.2f}% ricerche con fallback al diametro più vicino"
              f" ({fallback_count}/{total_lookups})")

    return result


def render_tcr_table(data: ParcelData, formatter: SnippetFormatter,
                     **options) -> RenderResult:
    """Render growth projection table (@@tcr directive)."""
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    if options[OPT_PER_GENERE]:
        group_cols.append(COL_GENERE)

    years = options[OPT_ANNI]
    df = calculate_tcr_table(data, group_cols, years, options[OPT_MORTALITA])

    # When grouping only by species, area cannot be meaningfully assigned to
    # individual species in a mixed forest, so hide area and per-hectare columns.
    genere_only = group_cols == [COL_GENERE]

    # Area deduplication for totals: genere creates duplicate spatial rows
    total_area = 0
    if not genere_only:
        if COL_GENERE in group_cols:
            parcel_cols = [c for c in group_cols if c != COL_GENERE]
            total_area = df.drop_duplicates(subset=parcel_cols)[COL_AREA_HA].sum()
        else:
            total_area = df[COL_AREA_HA].sum()

    n = f"+{years}aa"
    col_specs = [
        ColSpec('Area (ha)', 'r', COL_AREA_HA, lambda _: f"{total_area:.1f}", not genere_only),
        ColSpec('Volume (m³)', 'r', COL_VOLUME_MATURE, COL_VOLUME_MATURE,
                options[OPT_COL_VOLUME_MATURE]),
        ColSpec(f'Volume {n} (m³)', 'r', COL_VOLUME_MATURE_PROJ, COL_VOLUME_MATURE_PROJ,
                options[OPT_COL_VOLUME_MATURE]),
        ColSpec(f'Incr. corr. (m³)', 'r', COL_INCR_CORR, COL_INCR_CORR,
                options[OPT_COL_INCR_CORR]),
        ColSpec('Volume/ha', 'r',
                lambda r: f"{r[COL_VOLUME_MATURE] / r[COL_AREA_HA]:.1f}",
                lambda d: f"{d[COL_VOLUME_MATURE].sum() / total_area:.1f}",
                options[OPT_COL_VOLUME_MATURE_HA] and not genere_only),
        ColSpec(f'Volume {n}/ha', 'r',
                lambda r: f"{r[COL_VOLUME_MATURE_PROJ] / r[COL_AREA_HA]:.1f}",
                lambda d: f"{d[COL_VOLUME_MATURE_PROJ].sum() / total_area:.1f}",
                options[OPT_COL_VOLUME_MATURE_HA] and not genere_only),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


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


def render_gsv_graph(data: ParcelData, output_path: Path,
                     formatter: SnippetFormatter, color_map: dict,
                     **options) -> RenderResult:
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
        if options[OPT_PER_COMPRESA]:
            group_cols.append(COL_COMPRESA)
        if options[OPT_PER_PARTICELLA]:
            group_cols.append(COL_PARTICELLA)

        # For stacking, we need per-genere data even if displaying by compresa/particella
        stacked = options[OPT_PER_GENERE] and group_cols
        base_cols: list[str] = []
        if stacked:
            base_cols = group_cols.copy()
            group_cols.append(COL_GENERE)

        df = calculate_tsv_table(data, group_cols, calc_margin=False, calc_total=True)
        if df.empty:
            return RenderResult(snippet='')

        if stacked:
            # Pivot to get genere as columns for stacking
            pivot_df = df.pivot_table(index=base_cols, columns=COL_GENERE,
                                    values=COL_VOLUME, fill_value=0)
            labels = ['/'.join(str(x) for x in idx) if isinstance(idx, tuple) else str(idx)
                    for idx in pivot_df.index]
            generi = pivot_df.columns.tolist()

            # Sort by compresa then particella (natural sort for particella)
            if COL_PARTICELLA in base_cols:
                sort_keys = [pivot_df.index.get_level_values(c) for c in base_cols]
                natsort = natsort_keygen()
                sort_idx = sorted(range(len(labels)), key=lambda i: tuple(
                    natsort(str(sort_keys[j][i])) if base_cols[j] == COL_PARTICELLA
                    else (0, str(sort_keys[j][i])) for j in range(len(base_cols))))
                labels = [labels[i] for i in sort_idx]
                pivot_df = pivot_df.iloc[sort_idx]

            comprese = (pivot_df.index.get_level_values(COL_COMPRESA)
                        if COL_COMPRESA in base_cols and COL_PARTICELLA in base_cols
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
            if options[OPT_PER_GENERE]:
                # Per-genere only: one bar per genere
                labels = df[COL_GENERE].tolist()
            elif group_cols:
                labels = ['/'.join(str(row[c]) for c in group_cols) for _, row in df.iterrows()]
            else:
                labels = ['Totale']
            values = cast(np.ndarray, df[COL_VOLUME].values)

            comprese = (df[COL_COMPRESA].tolist()
                        if COL_COMPRESA in group_cols and COL_PARTICELLA in group_cols
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

    return RenderResult(filepath=output_path, snippet=snippet)


# RIPRESA =====================================================================


def compute_harvest(trees_df: pd.DataFrame, sampled_frac: float,
                     volume_limit: float, area_limit: float) -> tuple[float, float]:
    """Harvest trees smallest-first until either limit is reached.

    Args:
        trees_df: Parcel trees with D(cm), V(m3) columns
        sampled_frac: Sampling fraction
        volume_limit: Max harvest volume (m3, absolute for parcel)
        area_limit: Max harvest basal area (m2, absolute for parcel)

    Returns:
        (volume_mature, harvest_volume), both scaled.
    """
    assert trees_df[COL_COMPRESA].nunique() == 1 and trees_df[COL_PARTICELLA].nunique() == 1
    harvestable = trees_df[trees_df[COL_DIAMETER_CM] > MATURE_THRESHOLD].copy()
    if harvestable.empty:
        return 0.0, 0.0

    # G in m2 per tree: D is in cm, D2 in cm2, /10000 converts cm2 -> m2
    harvestable['G'] = np.pi / 4 * harvestable[COL_DIAMETER_CM] ** 2 / 10000
    total_volume = harvestable[COL_V_M3].sum() / sampled_frac

    harvestable = harvestable.sort_values(COL_DIAMETER_CM)
    cum_vol, cum_area = 0.0, 0.0

    for _, tree in harvestable.iterrows():
        tv = tree[COL_V_M3] / sampled_frac
        ta = tree['G'] / sampled_frac
        if cum_vol + tv > volume_limit or cum_area + ta > area_limit:
            break
        cum_vol += tv
        cum_area += ta

    return total_volume, cum_vol


def calculate_tpt_table(data: ParcelData, rules: HarvestRulesFunc,
                        group_cols: list[str]) -> pd.DataFrame:
    """
    Calculate harvest (prelievo totale) table data for the @@tpt directive.

    Algorithm per particella:
    1. Compute volume and basal area of mature trees (D > 20cm) per hectare
    2. Call rules() to get (volume_limit_ha, area_limit_ha)
    3. Scale limits to parcel area and call compute_harvest() which iterates
       trees smallest-to-largest, stopping when either limit is reached

    pp_max is computed as effective harvest percentage (harvest / vol_mature * 100).
    If Genere is in group_cols, the harvest breakdown by species uses pro-rata allocation.

    Args:
        data: Output from parcel_data
        rules: HarvestRulesFunc
        group_cols: List of grouping columns (Compresa, Particella, Genere)

    Returns:
        DataFrame with columns depending on group_cols, plus:
        sector, age, area_ha, volume, volume_mature, pp_max, harvest
    """
    #pylint: disable=too-many-locals
    trees = data.trees
    if COL_V_M3 not in trees.columns:
        raise ValueError("@@tpt richiede dati con volumi (colonna V(m3) mancante). "
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
    for (region, parcel), part_trees in trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
        try:
            p = parcels[(region, parcel)]
        except KeyError as e:
            raise ValueError(f"Particella {region}/{parcel} non trovata") from e

        sector, p_area, sf, age = p.sector, p.area_ha, p.sampled_frac, p.age

        # Compute per-ha stats of mature trees for rules lookup
        mature = part_trees[part_trees[COL_DIAMETER_CM] > MATURE_THRESHOLD]
        vol_mature_per_ha = mature[COL_V_M3].sum() / sf / p_area
        basal_per_ha = (np.pi / 4 * mature[COL_DIAMETER_CM] ** 2 / 10000).sum() / sf / p_area

        vol_limit_ha, area_limit_ha = rules(sector, age, vol_mature_per_ha, basal_per_ha)
        if vol_limit_ha == 0 and area_limit_ha == 0:
            parcel_info[(region, parcel)] = None
            continue

        # Scale limits to absolute parcel values
        vol_limit = vol_limit_ha * p_area
        area_limit = area_limit_ha * p_area

        # Total volume (all trees)
        total_volume = part_trees[COL_V_M3].sum() / sf

        vol_mature, harvest = compute_harvest(part_trees, sf, vol_limit, area_limit)

        # Effective harvest percentage
        pp_max = harvest / vol_mature * 100 if vol_mature > 0 else 0

        parcel_info[(region, parcel)] = {
            COL_SECTOR: sector, COL_AGE: age, COL_AREA_HA: p_area, 'sf': sf,
            COL_VOLUME: total_volume,
            COL_VOLUME_MATURE: vol_mature,
            COL_PP_MAX: pp_max,
            COL_HARVEST: harvest,
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

        for (region, parcel), part_trees in group_trees.groupby([COL_COMPRESA, COL_PARTICELLA]):
            info = parcel_info[(region, parcel)]
            if info is None:
                continue

            any_tree = True

            # For per-genere breakdown, compute this group's share of the parcel
            if COL_GENERE in group_cols:
                # Pro-rata allocation based on volume_mature
                above_thresh = part_trees[part_trees[COL_DIAMETER_CM] > MATURE_THRESHOLD]
                group_vol = part_trees[COL_V_M3].sum() / info['sf']
                group_vol_senza = above_thresh[COL_V_M3].sum() / info['sf']

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
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col)


def render_tpt_table(data: ParcelData, rules: HarvestRulesFunc,
                     formatter: SnippetFormatter, **options) -> RenderResult:
    """
    Render harvest (prelievo totale) table (@@tpt directive).

    Args:
        data: Output from parcel_data
        rules: HarvestRulesFunc
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
        Note: pp_max is the effective harvest percentage (harvest / vol_mature * 100).

    Returns:
        dict with 'snippet' key containing formatted table
    """
    group_cols = []
    if options[OPT_PER_COMPRESA]:
        group_cols.append(COL_COMPRESA)
    if options[OPT_PER_PARTICELLA]:
        group_cols.append(COL_PARTICELLA)
    if options[OPT_PER_GENERE]:
        group_cols.append(COL_GENERE)

    df = calculate_tpt_table(data, rules, group_cols)
    if df.empty:
        return RenderResult(snippet='')

    per_parcel = COL_PARTICELLA in group_cols or len(data.parcels) == 1

    # When grouping only by species, area cannot be meaningfully assigned to
    # individual species in a mixed forest, so hide area and per-hectare columns.
    genere_only = group_cols == [COL_GENERE]

    # Area deduplication for totals: genere creates duplicate spatial rows
    if not genere_only and COL_GENERE in group_cols:
        parcel_cols = [c for c in group_cols if c != COL_GENERE]
        total_area = df.drop_duplicates(subset=parcel_cols)[COL_AREA_HA].sum()
    else:
        total_area = df[COL_AREA_HA].sum()

    # total_fn lambdas close over total_area (computed above)
    col_specs = [
        ColSpec('Comp.', 'l', lambda r: str(r[COL_SECTOR]), None,
                options[OPT_COL_COMPARTO] and per_parcel),
        ColSpec('Età (aa)', 'r', lambda r: f"{r[COL_AGE]:.0f}", None,
                options[OPT_COL_ETA] and per_parcel),
        ColSpec('Area (ha)', 'r', COL_AREA_HA, lambda _: f"{total_area:.1f}",
         options[OPT_COL_AREA_HA] and not genere_only),
        ColSpec('Vol tot (m³)', 'r', COL_VOLUME, COL_VOLUME, options[OPT_COL_VOLUME]),
        ColSpec('Vol/ha (m³/ha)', 'r',
            lambda r: f"{r[COL_VOLUME] / r[COL_AREA_HA]:.1f}",
            lambda d: f"{d[COL_VOLUME].sum() / total_area:.1f}",
            options[OPT_COL_VOLUME_HA] and not genere_only),
        ColSpec('Vol mature (m³)', 'r', COL_VOLUME_MATURE, COL_VOLUME_MATURE,
                options[OPT_COL_VOLUME_MATURE]),
        ColSpec('Vol mature/ha (m³/ha)', 'r',
                lambda r: f"{r[COL_VOLUME_MATURE] / r[COL_AREA_HA]:.1f}",
                lambda d: f"{d[COL_VOLUME_MATURE].sum() / total_area:.1f}",
                options[OPT_COL_VOLUME_MATURE_HA] and not genere_only),
        ColSpec('Prelievo \\%', 'r', lambda r: f"{r[COL_PP_MAX]:.0f}",
                None, options[OPT_COL_PP_MAX] and per_parcel),
        ColSpec('Prel tot (m³)', 'r', COL_HARVEST, COL_HARVEST, options[OPT_COL_PRELIEVO]),
        ColSpec('Prel/ha (m³/ha)', 'r',
                lambda r: f"{r[COL_HARVEST] / r[COL_AREA_HA]:.1f}",
                lambda d: f"{d[COL_HARVEST].sum() / total_area:.1f}",
                options[OPT_COL_PRELIEVO_HA] and not genere_only),
    ]
    return render_table(df, group_cols, col_specs, formatter, options[OPT_TOTALI])


def render_gpt_graph(data: ParcelData, rules: HarvestRulesFunc,
                     output_path: Path, formatter: SnippetFormatter,
                     **options) -> RenderResult:
    """
    Generate harvest horizontal bar graph (@@gpt directive).

    Args:
        data: Output from parcel_data
        rules: HarvestRulesFunc
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        options: per_compresa, per_particella flags (per_genere not supported)

    Returns:
        dict with 'filepath' and 'snippet' keys
    """
    if not skip_graphs:
        group_cols = []
        if options[OPT_PER_COMPRESA]:
            group_cols.append(COL_COMPRESA)
        if options[OPT_PER_PARTICELLA]:
            group_cols.append(COL_PARTICELLA)

        df = calculate_tpt_table(data, rules, group_cols)
        if df.empty:
            return RenderResult(snippet='')

        if group_cols:
            labels = ['/'.join(str(row[c]) for c in group_cols) for _, row in df.iterrows()]
        else:
            labels = ['Totale']
        values = cast(np.ndarray, df[COL_HARVEST].values)

        comprese = (df[COL_COMPRESA].tolist()
                    if COL_COMPRESA in group_cols and COL_PARTICELLA in group_cols
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

    return RenderResult(filepath=output_path, snippet=snippet)


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


def parse_template_directive(line: str) -> Optional[Directive]:
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
    list_keys = {'compresa', 'particella', 'genere', 'alberi', OPT_EQUAZIONI}

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

    return Directive(keyword=keyword, params=params, full_text=full_text)

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
            case Fmt.HTML: ext = '.html'
            case Fmt.TEX | Fmt.PDF: ext = '.tex'
            case Fmt.CSV: ext = '.csv'
            case _: raise ValueError(f"Formato non supportato per @@particelle: {format_type}")
        modello_path = template_dir / (modello + ext)
        if not modello_path.exists():
            raise ValueError(f"Modello non trovato: {modello_path}")
        with open(modello_path, 'r', encoding='utf-8') as f:
            modello_text = f.read()

        compresa = comprese[0]
        parcel_rows = particelle_df[(particelle_df[COL_COMPRESA] == compresa) &
                                    (particelle_df[COL_GOVERNO] == GOV_FUSTAIA)]
        parcel_list = sorted(parcel_rows[COL_PARTICELLA].unique(), key=natsort_keygen())
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
            keyword = directive.keyword
            params = directive.params

            csv_unsupported = keyword.startswith('g')
            if format_type == Fmt.CSV and csv_unsupported:
                raise ValueError(
                    f"@@{keyword}: il formato CSV non supporta direttive grafiche (@@g*)")

            alberi_files = params.get('alberi')
            equazioni_files = params.get(OPT_EQUAZIONI)

            if not alberi_files and keyword not in (Dir.PROP, Dir.PARTICELLE):
                raise ValueError(f"@@{keyword} richiede alberi=FILE")

            comprese = params.get('compresa', [])
            particelle = params.get('particella', [])
            generi = params.get('genere', [])

            if keyword == Dir.PROP:
                if len(comprese) != 1 or len(particelle) != 1 or len(params) != 2:
                    raise ValueError("@@prop richiede esattamente compresa=X e particella=Y")
                result = render_prop(particelle_df, comprese[0], particelle[0], formatter)
                return result.snippet

            if keyword == Dir.PARTICELLE:
                return render_particelle(comprese, particelle, particelle_df, params)

            trees_df = load_trees(alberi_files, data_dir)
            data = parcel_data(alberi_files, trees_df, particelle_df, comprese, particelle, generi)

            match keyword:
                case Dir.TSV:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                        OPT_INTERVALLO_FIDUCIARIO: _bool_opt(params, OPT_INTERVALLO_FIDUCIARIO, False),
                        OPT_SOLO_MATURE: _bool_opt(params, OPT_SOLO_MATURE, False),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_tsv_table(data, formatter, **options)
                case Dir.TPT:
                    if 'genere' in params:
                        raise ValueError("@@tpt non supporta il parametro 'genere' "
                                         "(usa 'per_genere=si' per raggruppare per specie)")
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE, False),
                        OPT_COL_COMPARTO: _bool_opt(params, OPT_COL_COMPARTO),
                        OPT_COL_ETA: _bool_opt(params, OPT_COL_ETA),
                        OPT_COL_AREA_HA: _bool_opt(params, OPT_COL_AREA_HA),
                        OPT_COL_VOLUME: _bool_opt(params, OPT_COL_VOLUME, False),
                        OPT_COL_VOLUME_HA: _bool_opt(params, OPT_COL_VOLUME_HA, False),
                        OPT_COL_VOLUME_MATURE: _bool_opt(params, OPT_COL_VOLUME_MATURE),
                        OPT_COL_VOLUME_MATURE_HA: _bool_opt(params, OPT_COL_VOLUME_MATURE_HA),
                        OPT_COL_PP_MAX: _bool_opt(params, OPT_COL_PP_MAX),
                        OPT_COL_PRELIEVO: _bool_opt(params, OPT_COL_PRELIEVO),
                        OPT_COL_PRELIEVO_HA: _bool_opt(params, OPT_COL_PRELIEVO_HA),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_tpt_table(data, max_harvest,
                                              formatter, **options)
                case Dir.TIP:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA, False),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA, False),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    result = render_tip_table(data, formatter, **options)
                case Dir.GIP:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA, False),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA, False),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                        OPT_METRICA: params.get(OPT_METRICA, 'ip'),
                        OPT_STILE: params.get(OPT_STILE),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, OPT_METRICA, ['ip', 'ic'], '@@gip')
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gip_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case Dir.TCR:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE, False),
                        OPT_ANNI: int(params.get(OPT_ANNI, '0')),
                        OPT_MORTALITA: float(params.get(OPT_MORTALITA, '0')),
                        OPT_COL_VOLUME_MATURE: _bool_opt(params, OPT_COL_VOLUME_MATURE),
                        OPT_COL_VOLUME_MATURE_HA: _bool_opt(
                            params, OPT_COL_VOLUME_MATURE_HA),
                        OPT_COL_INCR_CORR: _bool_opt(params, OPT_COL_INCR_CORR, True),
                        OPT_TOTALI: _bool_opt(params, OPT_TOTALI, False),
                    }
                    check_allowed_params(keyword, params, options)
                    check_required_params(keyword, params, [OPT_ANNI])
                    if options[OPT_ANNI] <= 0:
                        raise ValueError("@@tcr richiede anni=N con N > 0")
                    result = render_tcr_table(data, formatter, **options)
                case Dir.GCD:
                    options = {
                        OPT_X_MAX: int(params.get(OPT_X_MAX, 0)),
                        OPT_Y_MAX: int(params.get(OPT_Y_MAX, 0)),
                        OPT_STILE: params.get(OPT_STILE),
                        OPT_METRICA: params.get(OPT_METRICA, 'alberi_ha'),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, OPT_METRICA,
                        ['alberi_ha', 'G_ha', 'volume_ha',
                         'alberi_tot', 'G_tot', 'volume_tot', 'altezza'],
                        '@@gcd')
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gcd_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case Dir.TCD:
                    options = {
                        OPT_METRICA: params.get(OPT_METRICA, 'alberi_ha'),
                        OPT_STIME_TOTALI: _bool_opt(params, OPT_STIME_TOTALI),
                    }
                    check_allowed_params(keyword, params, options)
                    check_param_values(options, OPT_METRICA,
                        ['alberi_ha', 'G_ha', 'volume_ha',
                         'alberi_tot', 'G_tot', 'volume_tot', 'altezza'],
                        '@@tcd')
                    result = render_tcd_table(data, formatter, **options)
                case Dir.GCI:
                    options = {
                        OPT_EQUAZIONI: True,
                        OPT_X_MAX: int(params.get(OPT_X_MAX, 0)),
                        OPT_Y_MAX: int(params.get(OPT_Y_MAX, 0)),
                        OPT_STILE: params.get(OPT_STILE),
                    }
                    check_allowed_params(keyword, params, options)
                    check_required_params(keyword, params, [OPT_EQUAZIONI])
                    equations_df = load_csv(equazioni_files, data_dir)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gci_graph(data, equations_df, output_dir / filename,
                                              formatter, color_map, **options)
                case Dir.GSV:
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_PER_GENERE: _bool_opt(params, OPT_PER_GENERE, False),
                        OPT_STILE: params.get(OPT_STILE),
                    }
                    check_allowed_params(keyword, params, options)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gsv_graph(data, output_dir / filename,
                                              formatter, color_map, **options)
                case Dir.GPT:
                    if OPT_PER_GENERE in params:
                        raise ValueError("@@gpt non supporta il parametro 'per_genere'")
                    options = {
                        OPT_PER_COMPRESA: _bool_opt(params, OPT_PER_COMPRESA),
                        OPT_PER_PARTICELLA: _bool_opt(params, OPT_PER_PARTICELLA),
                        OPT_STILE: params.get(OPT_STILE),
                    }
                    check_allowed_params(keyword, params, options)
                    filename = _build_graph_filename(comprese, particelle, generi, keyword)
                    result = render_gpt_graph(data, max_harvest,
                                              output_dir / filename, formatter, **options)
                case _:
                    raise ValueError(f"Comando sconosciuto: {keyword}")

            return result.snippet

        except Exception as e:
            raise ValueError(f"ERRORE nella generazione di {directive.full_text}: {e}") from e

    match format_type:
        case Fmt.HTML:
            formatter = HTMLSnippetFormatter()
        case Fmt.CSV:
            formatter = CSVSnippetFormatter()
        case Fmt.TEX | Fmt.PDF:
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
    df = df.dropna(subset=[COL_COMPRESA, COL_PARTICELLA])

    # Group by Compresa and list particelle
    for compresa in sorted(df[COL_COMPRESA].unique()):
        compresa_data = df[df[COL_COMPRESA] == compresa]
        particelle = sorted(compresa_data[COL_PARTICELLA].astype(str).unique(),
                          key=natsort_keygen())
        for particella in particelle:
            print(f"  {compresa},{particella}")


def get_color_map() -> dict:
    """
    Create consistent color mapping for species.

    Returns:
        Dict mapping species -> matplotlib hex color string
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

    return color_palette


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
    trees_df['IP'] = trees_df[COL_COEFF_PRESSLER] * 2 * trees_df[COL_L10_MM] / 100 / trees_df[COL_DIAMETER_CM]
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
        by=[COL_COMPRESA, COL_PARTICELLA, COL_AREA_SAGGIO, 'n'],
        key=lambda col: col.map(natsort_keygen()) if col.name == COL_PARTICELLA else col,
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
    if args.formato == Fmt.PDF:
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
    report_group.add_argument('--formato', choices=[Fmt.CSV, Fmt.HTML, Fmt.TEX, Fmt.PDF], default=Fmt.PDF,
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
