#!/usr/bin/env python3
"""
Forest Analysis: Accrescimenti Tool
Three-mode tool for equation generation, height calculation, and report generation.
"""

from abc import ABC, abstractmethod
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        pass

    @abstractmethod
    def _fit_params(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Fit parameters to data. Returns (a, b)."""
        pass

    @abstractmethod
    def _predict(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Predict y values from x using parameters a, b."""
        pass

    @abstractmethod
    def _create_lambda(self, a: float, b: float):
        """Create lambda function for prediction."""
        pass

    @abstractmethod
    def _format_equation(self, a: float, b: float) -> str:
        """Format equation as string."""
        pass

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
        pass

    @abstractmethod
    def format_metadata(self, metadata: dict, curve_info: list = None) -> str:
        """Format metadata block for this format.

        Args:
            metadata: Statistics about the region/species
            curve_info: List of dicts with {species, equation, r_squared, n_points}
                       from equations.csv
        """
        pass


class HTMLSnippetFormatter(SnippetFormatter):
    """HTML snippet formatter."""

    def format_image(self, filepath: Path) -> str:
        return f'<img src="{filepath.name}" class="graph-image">'

    def format_metadata(self, metadata: dict, curve_info: list = None) -> str:
        """Format metadata as HTML."""
        html = '<div class="metadata">\n'
        html += f'<p><strong>Alberi campionati:</strong> {metadata["sampled_trees"]}</p>\n'
        html += f'<p><strong>Stima totale:</strong> {metadata["estimated_total"]}</p>\n'
        html += f'<p><strong>Area:</strong> {metadata["area_ha"]} ha</p>\n'

        if "mean_height" in metadata:
            html += f'<p><strong>Altezza media:</strong> {metadata["mean_height"]:.1f} m</p>\n'
        if "mean_diameter_class" in metadata:
            html += f'<p><strong>Classe diametrica media:</strong> {metadata["mean_diameter_class"]:.0f}</p>\n'

        if curve_info:
            html += '<br><p><strong>Funzioni di interpolazione:</strong></p>\n'
            for curve in curve_info:
                html += (f'<p>{curve["species"]}: {curve["equation"]} '
                        f'(R² = {curve["r_squared"]:.2f}, n = {curve["n_points"]})</p>\n')

        html += '</div>\n'
        return html


class LaTeXSnippetFormatter(SnippetFormatter):
    """LaTeX snippet formatter."""

    def format_image(self, filepath: Path) -> str:
        return (f'\\begin{{figure}}[H]\n'
                f'  \\centering\n'
                f'  \\includegraphics[width=0.9\\textwidth]{{{filepath.name}}}\n'
                f'\\end{{figure}}\n')

    def format_metadata(self, metadata: dict, curve_info: list = None) -> str:
        """Format metadata as LaTeX."""
        latex = '\\begin{quote}\\small\n'
        latex += f"\\textbf{{Alberi campionati:}} {metadata['sampled_trees']}\\\\\n"
        latex += f"\\textbf{{Stima totale:}} {metadata['estimated_total']}\\\\\n"
        latex += f"\\textbf{{Area:}} {metadata['area_ha']} ha\\\\\n"

        if "mean_height" in metadata:
            latex += f"\\textbf{{Altezza media:}} {metadata['mean_height']:.1f} m\\\\\n"
        if "mean_diameter_class" in metadata:
            latex += f"\\textbf{{Classe diametrica media:}} {metadata['mean_diameter_class']:.0f}\\\\\n"

        if curve_info:
            latex += '\\\\\n\\textbf{Funzioni di interpolazione:}\\\\\n'
            for curve in curve_info:
                eq = curve['equation'].replace('*', r'\times ')
                eq = eq.replace('ln', r'\ln')
                latex += (f"{curve['species']}: ${eq}$ ($R^2$ = {curve['r_squared']:.2f}, "
                         f"$n$ = {curve['n_points']})\\\\\n")

        latex += '\\end{quote}\n'
        return latex


# =============================================================================
# DATA PREPARATION LAYER (pure data, no rendering)
# =============================================================================

def prepare_region_data(trees_df: pd.DataFrame, particelle_df: pd.DataFrame,
                       compresa: str, particella: Optional[str] = None,
                       genere: Optional[str] = None) -> dict:
    """
    Filter and aggregate tree data based on parameters.

    Args:
        trees_df: Full tree database
        particelle_df: Parcel metadata
        compresa: Required compresa name
        particella: Optional particella name
        genere: Optional species name (None means all species)

    Returns:
        dict with keys:
            - 'trees': filtered DataFrame
            - 'metadata': computed statistics
            - 'species_list': list of species in this dataset
            - 'compresa': compresa name
            - 'particella': particella name or None
    """
    # TODO: Implement data filtering and aggregation
    raise NotImplementedError("prepare_region_data not yet implemented")


# =============================================================================
# COMPUTATION LAYER (equation generation and application)
# =============================================================================

def fit_curves_from_ipsometro(ipsometro_file: str, funzione: str = 'log') -> pd.DataFrame:
    """
    Generate equations from ipsometer field measurements.

    Args:
        ipsometro_file: CSV with columns [Compresa, Specie, Diametro, Altezza]
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    # TODO: Implement curve fitting from ipsometer data
    raise NotImplementedError("fit_curves_from_ipsometro not yet implemented")


def fit_curves_from_originali(alberi_file: str, particelle_file: str,
                              funzione: str = 'log') -> pd.DataFrame:
    """
    Generate equations from original tree database heights.

    Args:
        alberi_file: CSV with tree data
        particelle_file: CSV with parcel metadata
        funzione: 'log' or 'lin'

    Returns:
        DataFrame with columns [compresa, genere, funzione, a, b, r2, n]
    """
    # TODO: Implement curve fitting from tree database
    raise NotImplementedError("fit_curves_from_originali not yet implemented")


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
    # TODO: Implement curve fitting from alsometric tables
    raise NotImplementedError("fit_curves_from_tabelle not yet implemented")


def apply_height_equations(alberi_file: str, equations_file: str,
                           output_file: str) -> None:
    """
    Apply height equations to tree database, updating heights.

    Args:
        alberi_file: Input tree CSV
        equations_file: CSV with equations [compresa, genere, funzione, a, b, r2, n]
        output_file: Output tree CSV with updated heights
    """
    # TODO: Implement height calculation using equations
    raise NotImplementedError("apply_height_equations not yet implemented")


# =============================================================================
# RENDERING LAYER (graph generation with format-specific snippets)
# =============================================================================

def render_ci_graph(prepared_data: dict, equations_df: pd.DataFrame,
                   output_path: Path, formatter: SnippetFormatter,
                   color_map: dict) -> dict:
    """
    Generate curve ipsometriche (height-diameter) graph.

    Args:
        prepared_data: Output from prepare_region_data()
        equations_df: Pre-computed equations from CSV
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    # TODO: Implement curve ipsometriche graph generation
    raise NotImplementedError("render_ci_graph not yet implemented")


def render_cd_graph(prepared_data: dict, output_path: Path,
                   formatter: SnippetFormatter, color_map: dict) -> dict:
    """
    Generate classi diametriche (diameter class histogram) graph.

    Args:
        prepared_data: Output from prepare_region_data()
        output_path: Where to save the PNG
        formatter: HTML or LaTeX snippet formatter
        color_map: Species -> color mapping

    Returns:
        dict with keys:
            - 'filepath': Path to generated PNG
            - 'snippet': Formatted HTML/LaTeX snippet for template substitution
    """
    # TODO: Implement classi diametriche graph generation
    raise NotImplementedError("render_cd_graph not yet implemented")


# =============================================================================
# TEMPLATE PROCESSING
# =============================================================================

def parse_template_directive(line: str) -> Optional[dict]:
    """
    Parse a template directive like @@ci(compresa=Serra, genere=Abete).

    Returns:
        dict with keys: 'keyword', 'params', 'full_text'
        or None if not a valid directive
    """
    # TODO: Implement template directive parsing
    raise NotImplementedError("parse_template_directive not yet implemented")


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
    # TODO: Implement template processing
    raise NotImplementedError("process_template not yet implemented")


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
        particelle = sorted(compresa_data['Particella'].astype(str).unique())
        for particella in particelle:
            print(f"  {compresa},{particella}")


def get_color_map(species_list: list) -> dict:
    """
    Create consistent color mapping for species.

    Args:
        species_list: List of unique species names

    Returns:
        Dict mapping species -> matplotlib color
    """
    colors = plt.cm.Set3(np.linspace(0, 1, len(species_list)))  # type: ignore
    return dict(zip(sorted(species_list), colors))


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
        equations_df = fit_curves_from_originali(args.input, args.particelle, args.funzione)
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
    """Execute Mode 2: Calculate heights."""
    print(f"Calcolo altezze usando equazioni da: {args.equazioni}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    apply_height_equations(args.input, args.equazioni, args.output)
    print("Altezze calcolate con successo")


def run_report(args):
    """Execute Mode 3: Generate report from template."""
    format_type = args.formato
    print(f"Generazione report formato: {format_type}")
    print(f"Template: {args.input}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    trees_df = pd.read_csv(args.alberi)
    particelle_df = pd.read_csv(args.particelle)
    equations_df = pd.read_csv(args.equazioni)

    # Read template
    with open(args.input, 'r', encoding='utf-8') as f:
        template_text = f.read()

    # Process template
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = process_template(template_text, trees_df, particelle_df,
                                equations_df, output_dir, format_type)

    # Write output
    input_path = Path(args.input)
    output_file = output_dir / input_path.name
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed)

    print(f"Report generato: {output_file}")


def run_lista_particelle(args):
    """Execute utility mode: List particelle."""
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

3. GENERA REPORT:
   ./acc.py --report --formato=html --equazioni equations.csv \\
            --alberi alberi-calcolati.csv --particelle particelle.csv \\
            --input template.html --output-dir report/

4. LISTA PARTICELLE:
   ./acc.py --lista-particelle --particelle particelle.csv
"""
    )

    # Mode selection (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument('--genera-equazioni', action='store_true',
                           help='Genera equazioni di interpolazione')
    run_group.add_argument('--calcola-altezze', action='store_true',
                           help='Calcola altezze usando equazioni')
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
            parser.error(f'--fonte-altezze={args.fonte_altezze} richiede --particelle')
        run_genera_equazioni(args)

    elif args.calcola_altezze:
        if not args.equazioni:
            parser.error('--calcola-altezze richiede --equazioni')
        if not args.input:
            parser.error('--calcola-altezze richiede --input')
        if not args.output:
            parser.error('--calcola-altezze richiede --output')
        run_calcola_altezze(args)

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
