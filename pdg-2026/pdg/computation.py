"""Pure computation: constants, dataclasses, Tabacchi equations, regression, primitives."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from pdg.formatters import fmt_num

# =============================================================================
# DOMAIN CONSTANTS
# =============================================================================

SAMPLE_AREA_HA = 0.125
MATURE_THRESHOLD = 20  # Diameter (cm) threshold for "mature" trees (smaller are not harvested)
MIN_TREES_PER_HA = 0.5  # Ignore buckets less than this in classi diametriche graphs.

# =============================================================================
# INPUT DATAFRAME COLUMN NAMES (from CSV files)
# =============================================================================

# Trees data
COL_D_CM = 'D(cm)'
COL_H_M = 'h(m)'
COL_V_M3 = 'V(m3)'
COL_GENERE = 'Genere'
COL_COMPRESA = 'Compresa'
COL_PARTICELLA = 'Particella'
COL_CD_CM = 'Classe diam.'      # Computed: diameter bucket (5 cm classes)
COL_SCALE = '_scale'            # Computed: 1/sampled_frac (scaling factor per tree)
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

GROUP_COLS_ALIGN = {
    COL_COMPRESA: 'l',
    COL_PARTICELLA: 'l',
    COL_GENERE: 'l',
    COL_CD_CM: 'r'
}

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CurveInfo:
    """Metadata for one regression curve (used in @@grafico_classi_ipsometriche graph legends)."""
    genere: str
    equation: str
    r_squared: float
    n_points: int

@dataclass
class ParcelStats:
    """Per-parcel metadata computed from tree data and parcel metadata."""
    area_ha: float
    sector: str
    age: float
    governo: str
    n_sample_areas: int
    sampled_frac: float

@dataclass
class ParcelData:
    """Filtered tree data and associated parcel statistics."""
    trees: pd.DataFrame
    regions: list[str]
    species: list[str]
    parcels: dict[tuple[str, str], ParcelStats]

    def __post_init__(self):
        if COL_SCALE not in self.trees.columns:
            sf_map = {k: 1 / ps.sampled_frac for k, ps in self.parcels.items()}
            self.trees[COL_SCALE] = [
                sf_map[(r, p)]
                for r, p in zip(self.trees[COL_COMPRESA], self.trees[COL_PARTICELLA])]

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
        return f"y = {fmt_num(a, 2)}*ln(x) + {fmt_num(b, 2)}"


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
        return f"y = {fmt_num(a, 2)}*x + {fmt_num(b, 2)}"


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
    from typing import cast
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
        d_values = cast(np.ndarray, group[COL_D_CM].values)
        h_values = cast(np.ndarray, group[COL_H_M].values)
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
    required = [COL_D_CM, COL_H_M, COL_GENERE]
    missing = [col for col in required if col not in trees_df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    # Validate: no missing data
    na_mask = trees_df[COL_D_CM].isna() | trees_df[COL_H_M].isna()
    if na_mask.any():
        idx = na_mask.idxmax()
        raise ValueError(f"Dati mancanti per riga {idx}: "
                         f"D={trees_df.at[idx, COL_D_CM]}, "
                         f"h={trees_df.at[idx, COL_H_M]}")

    result_df = trees_df.copy()
    result_df[COL_V_M3] = 0.0

    for genere, group in result_df.groupby(COL_GENERE):
        if genere not in TABACCHI:
            raise ValueError(f"Genere '{genere}' non trovato nelle tavole di Tabacchi")
        b = TABACCHI[genere].b  # type: ignore[reportGeneralTypeIssues]
        d = group[COL_D_CM]
        h = group[COL_H_M]
        d2h = d ** 2 * h
        if len(b) == 2:
            vol = (b[0] + b[1] * d2h) / 1000
        else:
            vol = (b[0] + b[1] * d2h + b[2] * d) / 1000
        result_df.loc[group.index, COL_V_M3] = vol

    return result_df


def diameter_class(d: pd.Series, width: int = 5) -> pd.Series:
    """
    Returns a dataframe containing the width-cm diameter classes corresponding to
    the diameters (in cm) in d.

    For example, with width = 5, D in (2.5, 7.5] -> 5, D in (7.5, 12.5] -> 10.
    """
    return (np.ceil((d - (width/2)) / width) * width).astype(int)  # type: ignore[reportReturnType]


def basal_area_m2(d_cm):
    """Basal area in m² from diameter in cm: π/4 * (D/100)²."""
    return np.pi / 4 * d_cm ** 2 / 10000


MATURE_FILTER = lambda t: t[COL_D_CM] > MATURE_THRESHOLD


def calculate_area_and_volume(trees: pd.DataFrame,
                              filter_fn: Callable | None = None,
                              weight: pd.Series | None = None,
                              ) -> tuple[float, float]:
    """Scaled volume (m³) and basal area (m²) for a set of trees.

    Args:
        trees: Tree DataFrame with COL_V_M3, COL_D_CM, COL_SCALE columns
        filter_fn: predicate(trees) -> boolean mask (e.g., MATURE_FILTER)
        weight: optional per-tree factor (e.g., mortality weight in simulation)

    Returns:
        (volume, basal_area), both scaled by COL_SCALE.
    """
    if filter_fn is not None:
        mask = filter_fn(trees)
        trees = trees[mask]  # type: ignore[reportGeneralTypeIssues]
        if weight is not None:
            weight = weight[mask]  # type: ignore[reportGeneralTypeIssues]
    if trees.empty:
        return 0.0, 0.0
    scale = trees[COL_SCALE]
    w = weight if weight is not None else 1
    volume = (trees[COL_V_M3] * w * scale).sum()
    basal = (basal_area_m2(trees[COL_D_CM]) * w * scale).sum()
    return volume, basal


# =============================================================================
# CURVE FITTING (equation generation and application)
# =============================================================================

def fit_curves_grouped(groups, funzione: str, min_points: int = 10) -> pd.DataFrame:
    """
    Fit regression curves to grouped data.

    Args:
        groups: Iterable of ((compresa, genere), DataFrame) tuples
        funzione: 'log' or 'lin'
        min_points: Minimum number of points required for fitting

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    from typing import cast
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
            print(f"  {compresa} - {genere}: {regr} (R² = {fmt_num(regr.r2, 2)}, n = {regr.n_points})")  # type: ignore[reportGeneralTypeIssues]
        else:
            print(f"  {compresa} - {genere}: dati insufficienti (n < {min_points})")

    return pd.DataFrame(results)


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
    result_df[COL_H_M] = result_df[COL_H_M].astype(float)

    for (compresa, genere), group in trees_df.groupby([COL_COMPRESA, COL_GENERE]):  # type: ignore[reportGeneralTypeIssues]
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
        diameters = trees_df.loc[indices, COL_D_CM].astype(float)

        if eq['funzione'] == 'ln':
            new_heights = eq['a'] * np.log(np.maximum(diameters, 0.1)) + eq['b']
        else:  # 'lin'
            new_heights = eq['a'] * diameters + eq['b']

        result_df.loc[indices, COL_H_M] = new_heights.astype(float)
        trees_updated += len(group)

        if verbose:
            print(f"  {compresa} - {genere}: {len(group)} alberi aggiornati")

    return result_df, trees_updated, trees_unchanged


def fit_curves_from_ipsometro(ipsometro_file: str, funzione: str = 'log') -> pd.DataFrame:
    """
    Generate equations from ipsometer field measurements.

    Args:
        ipsometro_file: CSV with columns [Compresa, Genere, D(cm), h(m)]
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    from pdg.io import load_csv
    df = load_csv(ipsometro_file, None)
    df['x'] = df[COL_D_CM]
    df['y'] = df[COL_H_M]
    groups = []

    for (compresa, genere), group in df.groupby([COL_COMPRESA, COL_GENERE]):  # type: ignore[reportGeneralTypeIssues]
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
    from pdg.io import load_trees
    df = load_trees(alberi_file)
    df['x'] = df[COL_D_CM]
    df['y'] = df[COL_H_M]

    groups = list(df.groupby([COL_COMPRESA, COL_GENERE]))
    return fit_curves_grouped(groups, funzione)  # type: ignore[reportGeneralTypeIssues]


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
    from pdg.io import load_csv
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
