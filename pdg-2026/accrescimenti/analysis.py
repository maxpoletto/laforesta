#!/usr/bin/env python3
"""
Forest Parcel Analysis: Generate histograms of tree distribution by diameter class
and scatter plots with height-diameter relationships for each parcel
"""

from abc import ABC, abstractmethod
import argparse
from contextlib import contextmanager
import locale
import os
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'upper left'
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['legend.title_fontsize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

SAMPLE_AREAS_PER_HA = 8

#
# Output formatters
#

@contextmanager
def italian_locale():
    old_locale = locale.getlocale()
    try:
        locale.setlocale(locale.LC_ALL, 'it_IT.UTF-8')
        yield
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'it_IT')
            yield
        except locale.Error:
            yield
    finally:
        locale.setlocale(locale.LC_ALL, old_locale)

class OutputFormatter(ABC):
    """Abstract base class for output formatters"""

    @abstractmethod
    def generate_index_cd(self, files: list, output_dir: str) -> None:
        """Generate index for diameter class histograms"""
        pass

    @abstractmethod
    def generate_index_ci(self, files: list, output_dir: str) -> None:
        """Generate index for height-diameter curves"""
        pass

    @abstractmethod
    def finalize(self, output_dir: str) -> None:
        """Finalize output (e.g., compile PDF for LaTeX)"""
        pass

class HTMLFormatter(OutputFormatter):
    """HTML output formatter"""

    style = """
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .explanation {
            font-size: 16px;
            font-style: italic;
            text-align: left;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 5px 20px;
            background-color: #fafafa;
        }
        .histogram-item {
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #fafafa;
        }
        .histogram-title {
            font-size: 18px;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
        }
        .histogram-image {
            width: 100%;
            max-width: 800px;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
"""

    def generate_index_cd(self, files: list, output_dir: str) -> None:
        """Generate an HTML index file for the generated histograms"""
        files_sorted = sorted(files, key=lambda x: x[2])

        with open(Path(output_dir) / 'index.html', 'w', encoding='utf-8') as f:
            f.write(f'''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soc. Agr. La Foresta: distribuzione piante per classe diametrica</title>
    <style>{self.style}    </style>
</head>
<body>
    <div class="container">
        <h1>Soc. Agr. La Foresta: distribuzione piante per classe diametrica</h1>
''')

            for compresa, particella, filepath in files_sorted:
                title = f"{compresa} - Particella {particella}" if particella else compresa
                f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{title}</div>
            <img src="{filepath.name}" alt="Istogramma classi diametriche {title}" class="histogram-image">
        </div>
''')

            f.write('''    </div>
</body>
</html>''')

    def generate_index_ci(self, files: list, output_dir: str) -> None:
        """Generate an HTML index file for the generated scatter plots"""
        files_sorted = sorted(files, key=lambda x: x[2])

        with open(Path(output_dir) / 'index.html', 'w', encoding='utf-8') as f:
            f.write(f'''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soc. Agr. La Foresta: curve ipsometriche</title>
    <style>{self.style}    </style>
</head>
<body>
    <div class="container">
        <h1>Soc. Agr. La Foresta: curve ipsometriche</h1>
        <div class="explanation">
            <p>
                Le curve ipsometriche sono state calcolate con un modello di regressione logaritmico, per
                ogni combinazione di compresa, particella e genere. Vengono visualizzate solo per
                combinazioni con almeno 10 punti (n ≥ 10).
            </p>
        </div>
''')

            for compresa, particella, filepath in files_sorted:
                title = f"{compresa} - Particella {particella}" if particella else compresa
                f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{title}</div>
            <img src="{filepath.name}" alt="Curva ipsometrica {title}" class="histogram-image">
        </div>
''')

            f.write('''    </div>
</body>
</html>''')

    def finalize(self, output_dir: str) -> None:
        """No finalization needed for HTML"""
        pass


class LaTeXFormatter(OutputFormatter):
    """LaTeX output formatter"""

    HEADER = r'''\documentclass[a4paper,11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[italian]{babel}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{float}
\geometry{margin=2cm}
\usepackage{times}
'''
    FOOTER = r'''\end{document}
'''

    def __init__(self, compile_pdf: bool = False):
        self.compile_pdf = compile_pdf
        self.latex_files = []

    def generate_index_cd(self, files: list, output_dir: str) -> None:
        """Generate a LaTeX document for diameter class histograms"""
        files_sorted = sorted(files, key=lambda x: x[2])
        latex_file = Path(output_dir) / 'classi-diametriche.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(self.HEADER)
            f.write(r'''\title{Distribuzione piante per classe diametrica}
\date{}
\author{Società Agricola La Foresta}
\begin{document}
\maketitle''')

            for compresa, particella, filepath in files_sorted:
                title = f"{compresa} - Particella {particella}" if particella else compresa
                # Escape underscores in title for LaTeX
                title_escaped = title.replace('_', r'\_')
                f.write(f'''\\section*{{{title_escaped}}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{{filepath.name}}}
\\end{{figure}}

''')
            f.write(self.FOOTER)
        self.latex_files.append(latex_file)
        print(f"Generato file LaTeX: {latex_file}")

    def generate_index_ci(self, files: list, output_dir: str) -> None:
        """Generate a LaTeX document for height-diameter curves"""
        files_sorted = sorted(files, key=lambda x: x[2])
        latex_file = Path(output_dir) / 'curve-ipsometriche.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(self.HEADER)
            f.write(r'''\title{Curve ipsometriche}
\date{}
\author{Società Agricola La Foresta}
\begin{document}
\maketitle

\begin{quote}
\itshape
Le curve ipsometriche sono state calcolate con un modello di regressione logaritmico, per
ogni combinazione di compresa, particella e genere. Vengono visualizzate solo per
combinazioni con almeno 10 punti ($n \ge 10$).
\end{quote}

''')

            for compresa, particella, filepath in files_sorted:
                title = f"{compresa} - Particella {particella}" if particella else compresa
                # Escape underscores in title for LaTeX
                title_escaped = title.replace('_', r'\_')
                f.write(f'''\\section*{{{title_escaped}}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{{filepath.name}}}
\\end{{figure}}

''')

            f.write(self.FOOTER)

        self.latex_files.append(latex_file)
        print(f"Generato file LaTeX: {latex_file}")

    def finalize(self, output_dir: str) -> None:
        """Compile LaTeX files to PDF if requested"""
        if not self.compile_pdf:
            return
        for latex_file in self.latex_files:
            pdf_file = latex_file.with_suffix('.pdf')
            if pdf_file.exists():
                pdf_file.unlink()
            try:
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', latex_file.name],
                    cwd=latex_file.parent,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Warning: pdflatex returned non-zero exit code ({result.stderr})")

                if pdf_file.exists():
                    print(f"Generato file PDF: {pdf_file}")
                    # Clean up auxiliary files
                    for ext in ['.aux', '.log', '.out']:
                        aux_file = latex_file.with_suffix(ext)
                        if aux_file.exists():
                            aux_file.unlink()
                else:
                    print(f"Attenzione: file PDF non creato")
            except Exception as e:
                print(f"Errore nella compilazione LaTeX: {e}")

#
# Interpolation and curve fitting
#

class RegressionFunc(ABC):

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
        """Fit the regression function to data.

        Returns True if successful, False otherwise.
        """
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
        return f"y = {a:.4f}*ln(x) + {b:.4f}"


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
        return f"y = {a:.4f}*x + {b:.4f}"

def hif_alsometrie(alsometrie_file: str, interpolation_func: str,
                   alsometry_calc: bool, alsometrie_calcolate_file: str) -> dict | None:
    """Create height interpolation functions from alsometrie data.
    Returns a dictionary: hfuncs[species] = lambda function
    """
    try:
        df = pd.read_csv(alsometrie_file)
    except FileNotFoundError:
        print(f"Attenzione: {alsometrie_file} non trovato, uso i valori originali h(m)")
        return None

    # Convert numeric columns
    numeric_cols = ['Diam base', 'Diam 130cm', 'Volume dendrometrico', 'Volume cormometrico', 'Altezza indicativa']
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne {missing_cols} non trovate in {alsometrie_file}")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    hfuncs = {}
    interp_method = 'quadratic' if interpolation_func == 'quadratica' else 'linear'
    df['Altezza indicativa'] = (df.groupby('Genere')['Altezza indicativa']
                                .transform(lambda x: x.interpolate(method=interp_method)))
    for species in df['Genere'].unique():
        data = (df[df['Genere'] == species].dropna(subset=['Diam 130cm', 'Altezza indicativa'])
                .sort_values('Diam 130cm'))
        hfuncs[species] = (lambda x,
                            xvec=data['Diam 130cm'].values, yvec=data['Altezza indicativa'].values:
                            np.interp(x, xvec, yvec)) # type: ignore

    if alsometry_calc:
        df['Diam base'] = df['Diam base'].astype('Int64')
        df['Altezza indicativa'] = df.apply(
            lambda row: hfuncs[row['Genere']](row['Diam 130cm']) if hfuncs and row['Genere'] in hfuncs else row['Altezza indicativa'],
            axis=1)
        df.to_csv(alsometrie_calcolate_file, index=False, float_format="%.3f")
        print(f"File '{alsometrie_calcolate_file}' salvato")

    return hfuncs

def hif_altezze(altezze_file: str, regression_func: str) -> dict | None:
    """Create height interpolation functions from measured height data (ipsometro).

    Returns a nested dictionary: hfuncs[compresa][species] = lambda function
    """
    try:
        df = pd.read_csv(altezze_file)
    except FileNotFoundError:
        print(f"Attenzione: {altezze_file} non trovato, uso i valori originali h(m)")
        return None

    required_cols = ['Compresa', 'Specie', 'Diametro', 'Altezza']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne {missing_cols} non trovate in {altezze_file}")

    # Select regression class
    RegressionClass = LogarithmicRegression if regression_func == 'logaritmica' else LinearRegression

    hfuncs = {}
    for compresa in sorted(df['Compresa'].unique()):
        compresa_data = df[df['Compresa'] == compresa]
        hfuncs[compresa] = {}

        for species in sorted(compresa_data['Specie'].unique()):
            species_data = compresa_data[compresa_data['Specie'] == species]
            x = species_data['Diametro'].values
            y = species_data['Altezza'].values

            regr = RegressionClass()
            if regr.fit(x, y, min_points=10):
                hfuncs[compresa][species] = regr.get_lambda()
                print(f"  {species:15}: {regr}")
            else:
                print(f"  {species:15}: Troppi pochi punti ({len(x)}) per la regressione")

    return hfuncs

def get_height_function(hfuncs: dict, compresa: str, species: str):
    """Get height interpolation function from hfuncs dictionary.

    Handles both flat structure (alsometrie: hfuncs[species])
    and nested structure (ipsometro: hfuncs[compresa][species]).
    """
    if not hfuncs:
        return None
    if compresa in hfuncs and isinstance(hfuncs[compresa], dict):
        return hfuncs[compresa].get(species)
    return hfuncs.get(species)

def height_interpolation_functions(height_source: str, alsometrie_file: str, ipsometro_file: str,
                                   interpolation_func: str, regression_func: str,
                                   alsometry_calc: bool, alsometrie_calcolate_file: str) -> dict | None:
    """Create height interpolation functions."""
    assert height_source in ['alsometrie', 'ipsometro']
    assert interpolation_func in ['quadratica', 'lineare']
    assert regression_func in ['logaritmica', 'lineare']

    if height_source == 'alsometrie':
        return hif_alsometrie(alsometrie_file, interpolation_func, alsometry_calc, alsometrie_calcolate_file)
    elif height_source == 'ipsometro':
        return hif_altezze(ipsometro_file, regression_func)

#
# Create histograms for diameter class distribution
#

def create_cd(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str, set_title: bool) -> list:
    """
    Create a histogram for a specific region (parcel or compresa) showing tree distribution by diameter class
    with stacked bars for different species, scaled to estimated totals per hectare
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)

    if particella is None:
        region_data = trees[trees['Compresa'] == compresa]
        title = f'Distribuzione alberi per classe diametrica - {compresa}' if set_title else None
        print_name = compresa
    else:
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        title = f'Distribuzione alberi per classe diametrica - {compresa} Particella {particella}' if set_title else None
        print_name = f"{compresa}-{particella}"
    filename = f"{region['sort_key']}_classi-diametriche.png"

    assert len(region_data) > 0, f"Nessun dato per {print_name}"
    print(f"Generazione istogramma per {print_name}...")

    counts = (region_data.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
              * SAMPLE_AREAS_PER_HA / region['sample_areas'])

    fig, ax = plt.subplots(figsize=(4, 2.5))
    species_list = counts.columns.tolist()

    bottom = np.zeros(len(counts.index))
    for species in species_list:
        values = counts[species].values
        ax.bar(counts.index, values, bottom=bottom,
               label=species, color=color_map[species],
               alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Stima alberi / ha', fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=20)

    max_class = trees['Classe diametrica'].max()
    ax.set_xlim(-0.5, max_class + 0.5)
    ax.set_xticks(range(0, max_class + 1, 2))
    ax.set_ylim(0, counts.sum(axis=1).max() * 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02),
              alignment='left')

    with italian_locale():
        stats_text = f"""
Area: {locale.format_string('%.2f', region["area_ha"])} ha
Alberi campionati: {len(region_data)}
N. aree saggio: {region["sample_areas"]}
Stima totale alberi: {region["estimated_total"]:n}
Stima alberi / ha: {round(region["estimated_total"]/region["area_ha"]):n}
Specie prevalente: {region_data["Genere"].mode().iloc[0]}
Classe diametrica media: {round(region_data["Classe diametrica"].mean()):n}""".strip()

    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))
    plt.tight_layout()

    filepath = Path(output_dir) / filename
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Salvato istogramma per {print_name} in {filepath}")
    plt.close(fig)
    return [compresa, particella, filepath]

#
# Create scatter plots for height vs diameter class relationship
#

def create_ci(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str,
              hfuncs: dict, omit_unknown: bool, set_title: bool) -> list:
    """
    Create a scatter plot for a specific region (parcel or compresa) showing height vs diameter class relationship
    with logarithmic fit for each species
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)

    if particella is None:
        region_data = trees[trees['Compresa'] == compresa]
        title = f'Curve ipsometriche - {compresa}' if set_title else None
        print_name = compresa
    else:
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        title = f'Curve ipsometriche - {compresa} Particella {particella}' if set_title else None
        print_name = f"{compresa}-{particella}"
    filename = f"{region['sort_key']}_curve-ipsometriche.png"

    assert len(region_data) > 0, f"Nessun dato per {print_name}"
    print(f"Generazione grafico ipsometrico per {print_name}...")

    fig, ax = plt.subplots(figsize=(4, 3))

    species_list = sorted(region_data['Genere'].unique())
    polynomial_info = []

    ymax = 0
    for species in species_list:
        species_data = region_data[region_data['Genere'] == species]
        x = species_data['Classe diametrica'].values

        height_func = get_height_function(hfuncs, compresa, species)

        # Skip species not present in height functions if requested (generate only "clean" curves)
        if hfuncs and height_func is None and omit_unknown:
            print(f"Genere {species} non presente nelle funzioni di altezza, omesso dalle curve ipsometriche")
            continue

        # Use interpolated heights if available, otherwise use original h(m)
        if height_func is not None:
            # Get D(cm) values and apply interpolation function
            d_cm = species_data['D(cm)'].values
            y = np.array([height_func(d) for d in d_cm])
        else:
            y = species_data['h(m)'].values
        ymax = max(ymax, y.max())

        # Plot scatter points
        ax.scatter(x, y, color=color_map[species], label=species, alpha=0.7, s=20)

        # Fit logarithmic curve for visualization
        regr = LogarithmicRegression()
        if not regr.fit(x, y, min_points=10):
            continue

        x_min, x_max = regr.x_range
        x_smooth = np.linspace(x_min, x_max, 100)
        y_smooth = regr.get_lambda()(x_smooth)
        ax.plot(x_smooth, y_smooth, color=color_map[species], linestyle='--', alpha=0.8, linewidth=1.5)

        polynomial_info.append({
            'species': species,
            'equation': str(regr),
            'r_squared': regr.r2,
            'n_points': regr.n_points
        })

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Altezza (m)', fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', pad=20)

    max_class = trees['Classe diametrica'].max()
    ax.set_xlim(-0.5, max_class + 0.5)
    ax.set_xticks(range(0, max_class + 1, 2))

    ax.set_ylim(0, (ymax + 6)//5*5)
    td = min(ax.get_ylim()[1] // 5, 4)
    y_ticks = np.arange(0, ax.get_ylim()[1] + 1, td)
    ax.set_yticks(y_ticks)

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

    with italian_locale():
        stats_text = f"""
Area: {locale.format_string('%.2f', region["area_ha"])} ha
Alberi campionati: {len(region_data)}
N. aree saggio: {region["sample_areas"]}
Stima totale alberi: {region["estimated_total"]:n}
Stima alberi / ha: {round(region["estimated_total"]/region["area_ha"]):n}
Specie prevalente: {region_data["Genere"].mode().iloc[0]}
Altezza media: {locale.format_string('%.1f', region_data["h(m)"].mean())} m""".strip()

    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))

    if polynomial_info:
        poly_text = ""
        for i in polynomial_info:
            poly_text += f"{i['species']}: {i['equation']} (R² = {i['r_squared']:.3f}, n = {i['n_points']})\n"

        ax.text(0.01, -0.25, poly_text.strip(), transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=5, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, linewidth=0.2))

    plt.tight_layout()

    filepath = Path(output_dir) / filename
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Salvato grafico ipsometrico per {print_name} in {filepath}")
    plt.close(fig)
    return [compresa, particella, filepath]

def regions_dataframe(alberi_fustaia: pd.DataFrame, particelle: pd.DataFrame, per_particella: bool) -> pd.DataFrame:
    """Create a dataframe with region data. Region may be "compresa" or "particella". """
    groupby = ['Compresa', 'Particella'] if per_particella else ['Compresa']
    trees = (alberi_fustaia.groupby(groupby)
             .agg(sampled_trees=('Area saggio', 'size'),
                  sample_areas=('Area saggio', 'nunique'))
             .reset_index())
    areas = particelle.groupby(groupby)['Area (ha)'].sum().reset_index()
    df = (trees.merge(areas, on=groupby, how='left').rename(columns={'Area (ha)': 'area_ha'}))
    df['estimated_total'] = round((df['sampled_trees'] / df['sample_areas'])
                                  * SAMPLE_AREAS_PER_HA
                                  * df['area_ha']).astype(int)

    if per_particella:
        df['sort_key'] = df['Particella'].apply(
            lambda x: (f"{x}=" if str(x)[-1].isdigit() else str(x)).zfill(3))
        df['sort_key'] = df['Compresa'] + '-' + df['sort_key']
        print(f"Modalità per particella: {len(df)} particelle")
    else:
        df['sort_key'] = df['Compresa']
        print(f"Modalità per compresa: {len(df)} comprese")

    return df

def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(description='Analisi accrescimenti', add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Mostra questo messaggio e esci')
    g1 = parser.add_argument_group('Tipo e granularità dei risultati')
    g1.add_argument('--genera-classi-diametriche', action='store_true',
                    help='Genera istogrammi classi diametriche')
    g1.add_argument('--genera-curve-ipsometriche', action='store_true',
                    help='Genera curve ipsometriche')
    g1.add_argument('--genera-alberi-altezze-calcolate', action='store_true',
                    help='Genera tabella alberi campionati con altezze calcolate in base alla fonte delle altezze')
    g1.add_argument('--genera-alsometrie-calcolate', action='store_true',
                    help='Genera file con le altezze calcolate dalle curve ipsometriche')
    g1.add_argument('--fonte-altezze', type=str,
                    choices=['alsometrie', 'ipsometro', 'originali'],
                    default='originali',
                    help=('Fonte delle altezze per le curve ipsometriche (default: originali). ' +
                    '"Alsometrie" = tavole alsometriche (Tabacchi et al.); i valori intermedi verrano interpolati. ' +
                    '"Ipsometro" = dati misurati con ipsometro; i valori intermedi verranno calcolati per regressione. ' +
                    '"Originali" = dati originali.'))
    g1.add_argument('--per-particella', action='store_true',
                    help='Genera risultati per ogni coppia (compresa, particella) (default: solo per compresa)')
    g1.add_argument('--formato-output', type=str,
                    choices=['html', 'latex', 'pdf'],
                    default='html',
                    help='Formato di output per i documenti (default: html)')
    g2 = parser.add_argument_group('Altezze per curve ipsometriche')
    g2.add_argument('--funzione-interpolazione', type=str,
                    choices=['quadratica', 'lineare'],
                    default='quadratica',
                    help='Funzione interpolante per altezze da tabelle alsometriche (default: quadratica)')
    g2.add_argument('--funzione-regressione', type=str,
                    choices=['logaritmica', 'lineare'],
                    default='logaritmica',
                    help='Funzione di regressione per altezze da dati ipsometrici (default: logaritmica)')
    g2.add_argument('--ometti-generi-sconosciuti', action='store_true',
                    default=True,
                    help='Omette generi non presenti nel file alsometrie dalle curve ipsometriche')
    g3 = parser.add_argument_group('Nomi dei file di input')
    g3.add_argument('--input-dir', type=str, default='.',
                    help='Directory contenente i file di input'),
    g3.add_argument('--file-alsometrie', type=str, default='alsometrie.csv',
                    help='File con le altezze da tabelle alsometriche')
    g3.add_argument('--file-ipsometro', type=str, default='altezze.csv',
                    help='File con i dati delle altezze misurate con ipsometro')
    g3.add_argument('--file-alberi', type=str, default='alberi.csv',
                    help='File con i dati degli alberi campionati')
    g3.add_argument('--file-particelle', type=str, default='particelle.csv',
                    help='File con i dati delle particelle')
    g4 = parser.add_argument_group('Nomi dei file di output')
    g4.add_argument('--file-alberi-calcolati', type=str, default='alberi-calcolati.csv',
                    help='File con i dati degli alberi campionati con altezze calcolate')
    g4.add_argument('--file-alsometrie-calcolate', type=str, default='alsometrie-calcolate.csv',
                    help='File con dati alsometrici calcolati')
    g4.add_argument('--prefisso-output', type=str, default='./',
                    help='Prefisso per i file di output')

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        return

    args.file_alberi = Path(args.input_dir) / args.file_alberi
    args.file_particelle = Path(args.input_dir) / args.file_particelle
    args.file_alsometrie = Path(args.input_dir) / args.file_alsometrie
    args.file_ipsometro = Path(args.input_dir) / args.file_ipsometro

    alberi = pd.read_csv(args.file_alberi)
    particelle = pd.read_csv(args.file_particelle)

    alberi_fustaia = alberi[alberi['Fustaia'] == True].copy()
    print(f"Dati filtrati: {len(alberi_fustaia)} campioni di alberi a fustaia (su {len(alberi)} totali)")

    # Create consistent color mapping for all species
    all_species = sorted(alberi_fustaia['Genere'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_species)))
    color_map = dict(zip(all_species, colors))

    # Create region data (either per particella or per compresa)
    regions = regions_dataframe(alberi_fustaia, particelle, args.per_particella)

    # Create output formatter based on command-line argument
    if args.formato_output == 'html':
        formatter = HTMLFormatter()
    else:  # latex
        formatter = LaTeXFormatter(compile_pdf=args.formato_output == 'pdf')

    hfuncs = None
    if args.fonte_altezze != 'originali':
        hfuncs = height_interpolation_functions(
            height_source=args.fonte_altezze,
            alsometrie_file=str(args.file_alsometrie),
            ipsometro_file=str(args.file_ipsometro),
            interpolation_func=args.funzione_interpolazione,
            regression_func=args.funzione_regressione,
            alsometry_calc=args.genera_alsometrie_calcolate,
            alsometrie_calcolate_file=str(args.file_alsometrie_calcolate)
        )

    if args.genera_classi_diametriche:
        output_dir = Path(args.prefisso_output) / 'classi-diametriche'
        os.makedirs(output_dir, exist_ok=True)

        files = []
        for _, row in regions.sort_values(['sort_key']).iterrows():
            files.append(create_cd(trees=alberi_fustaia, region=row,
                                   color_map=color_map, output_dir=output_dir,
                                   set_title=args.formato_output == 'html'))

        formatter.generate_index_cd(files, output_dir)
        print(f"Istogrammi classi diametriche salvati in '{output_dir}'")

    if args.genera_curve_ipsometriche:
        output_dir = Path(args.prefisso_output) / 'curve-ipsometriche'
        os.makedirs(output_dir, exist_ok=True)

        files = []
        for _, row in regions.sort_values(['sort_key']).iterrows():
            files.append(create_ci(trees=alberi_fustaia, region=row,
                                   color_map=color_map, output_dir=output_dir,
                                   hfuncs=hfuncs, omit_unknown=args.ometti_generi_sconosciuti,
                                   set_title=args.formato_output == 'html'))

        formatter.generate_index_ci(files, output_dir)
        print(f"Curve ipsometriche salvate in '{output_dir}'")

    if args.genera_alberi_altezze_calcolate:
        alberi_calcolati = alberi.copy()
        def calc_height(row):
            height_func = get_height_function(hfuncs, row['Compresa'], row['Genere'])
            return height_func(row['D(cm)']) if height_func is not None else row['h(m)']
        alberi_calcolati['h(m)'] = alberi_calcolati.apply(calc_height, axis=1)
        alberi_calcolati.to_csv(args.file_alberi_calcolati, index=False, float_format="%.3f")
        print(f"File '{args.file_alberi_calcolati}' salvato")

    # Finalize output (compile PDF if LaTeX was chosen)
    if args.genera_classi_diametriche or args.genera_curve_ipsometriche:
        formatter.finalize(output_dir)

    with italian_locale():
        print(f"\nAnalisi completata.")
        print("\nRiepilogo:")

        for _, row in regions.sort_values(['sort_key']).iterrows():
            trees_per_ha = row['estimated_total'] / row['area_ha']
            label = f"{row['Compresa']} - Particella {row['Particella']}" if args.per_particella else row['Compresa']
            print(f"  {label}: "
                f"{row['sampled_trees']} alberi campionati, "
                f"{row['sample_areas']} aree saggio, "
                f"{locale.format_string('%.2f', row['area_ha'])} ha → "
                f"Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")

if __name__ == "__main__":
    main()
