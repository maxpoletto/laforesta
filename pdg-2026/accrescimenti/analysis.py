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
    def generate_index_cd(self, files: list, output_dir: str, one_species_per_graph: bool = False,
                          external_legends: bool = False, region_legends: dict = None) -> None:
        """Generate index for diameter class histograms"""
        pass

    @abstractmethod
    def generate_index_ci(self, files: list, output_dir: str, one_species_per_graph: bool = False,
                          external_legends: bool = False, region_legends: dict = None) -> None:
        """Generate index for height-diameter curves"""
        pass

    @abstractmethod
    def finalize(self, output_dir: str, base_name: str = None) -> None:
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

    def _format_legend_html(self, result: dict, one_species_per_graph: bool, for_cd: bool) -> str:
        """Format legend data as HTML."""
        legend = result['legend_data']
        region_stats = legend['region_stats']
        species_stats = legend['species_stats']

        html = '<div style="margin-top: 10px; padding: 10px; background-color: #f0f0f0; border-radius: 4px;">'

        # Common region stats (displayed once when one_species_per_graph is True)
        if one_species_per_graph and species_stats:
            # Show species-specific stats
            html += f'''<p style="margin: 2px 0;"><strong>Alberi campionati:</strong> {species_stats['sampled_trees']}</p>
<p style="margin: 2px 0;"><strong>Stima totale alberi:</strong> {species_stats['estimated_total']}</p>'''
            if for_cd:
                html += f'<p style="margin: 2px 0;"><strong>Classe diametrica media:</strong> {species_stats["mean_diameter_class"]}</p>'
            else:
                html += f'<p style="margin: 2px 0;"><strong>Altezza media:</strong> {species_stats["mean_height"]} m</p>'
        else:
            # Show region-level stats
            html += f'''<p style="margin: 2px 0;"><strong>Area:</strong> {region_stats['area_ha']} ha</p>
<p style="margin: 2px 0;"><strong>Alberi campionati:</strong> {region_stats['sampled_trees']}</p>
<p style="margin: 2px 0;"><strong>N. aree saggio:</strong> {region_stats['sample_areas']}</p>
<p style="margin: 2px 0;"><strong>Stima totale alberi:</strong> {region_stats['estimated_total']}</p>
<p style="margin: 2px 0;"><strong>Stima alberi / ha:</strong> {region_stats['estimated_per_ha']}</p>
<p style="margin: 2px 0;"><strong>Specie prevalente:</strong> {region_stats['dominant_species']}</p>'''
            if for_cd:
                html += f'<p style="margin: 2px 0;"><strong>Classe diametrica media:</strong> {region_stats["mean_diameter_class"]}</p>'
            else:
                html += f'<p style="margin: 2px 0;"><strong>Altezza media:</strong> {region_stats["mean_height"]} m</p>'

        # Add polynomial info for curves
        if not for_cd and 'polynomial_info' in legend and legend['polynomial_info']:
            html += '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ccc;">'
            html += '<p style="margin: 5px 0;"><strong>Curve di regressione:</strong></p>'
            for poly in legend['polynomial_info']:
                html += f'<p style="margin: 2px 0; font-size: 12px;">{poly["species"]}: {poly["equation"]} (R² = {poly["r_squared"]:.3f}, n = {poly["n_points"]})</p>'
            html += '</div>'

        html += '</div>'
        return html

    def _write_region_header_html(self, region_title: str, reg_stats: dict = None) -> str:
        """Generate HTML for a region header with optional stats."""
        html = f'''        <div style="margin: 20px 0; padding: 15px; background-color: #e8f4f8; border-radius: 8px;">
            <h2 style="margin-top: 0;">{region_title}</h2>
'''
        if reg_stats:
            html += f'''            <p style="margin: 2px 0;"><strong>Area:</strong> {reg_stats['area_ha']} ha</p>
            <p style="margin: 2px 0;"><strong>N. aree saggio:</strong> {reg_stats['sample_areas']}</p>
            <p style="margin: 2px 0;"><strong>Specie prevalente:</strong> {reg_stats['dominant_species']}</p>
'''
        html += '        </div>\n'
        return html

    def _write_graph_item_html(self, title: str, filepath_name: str, legend_html: str = None) -> str:
        """Generate HTML for a graph item with optional legend."""
        html = f'''        <div class="histogram-item">
            <div class="histogram-title">{title}</div>
            <img src="{filepath_name}" alt="Istogramma {title}" class="histogram-image">
'''
        if legend_html:
            html += legend_html
        html += '        </div>\n'
        return html

    def generate_index_cd(self, files: list, output_dir: str, one_species_per_graph: bool = False,
                          external_legends: bool = False, region_legends: dict = None) -> None:
        """Generate an HTML index file for the generated histograms"""
        files_sorted = sorted(files, key=lambda x: (x['compresa'], x.get('particella', ''), x.get('species', '')))

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

            # Group results by region when one_species_per_graph is True
            if one_species_per_graph:
                current_region = None
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    region_key = (compresa, particella)

                    # Write region header when entering new region
                    if region_key != current_region:
                        current_region = region_key
                        region_title = f"{compresa} - Particella {particella}" if particella else compresa

                        if external_legends and region_legends and region_key in region_legends:
                            f.write(self._write_region_header_html(region_title, region_legends[region_key]))

                    # Write species graph
                    legend_html = self._format_legend_html(result, one_species_per_graph, for_cd=True) if external_legends else None
                    f.write(self._write_graph_item_html(result['species'], result['filepath'].name, legend_html))
            else:
                # Original behavior: one graph per region
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    title = f"{compresa} - Particella {particella}" if particella else compresa

                    legend_html = self._format_legend_html(result, one_species_per_graph, for_cd=True) if external_legends else None
                    f.write(self._write_graph_item_html(title, result['filepath'].name, legend_html))

            f.write('''    </div>
</body>
</html>''')

    def generate_index_ci(self, files: list, output_dir: str, one_species_per_graph: bool = False,
                          external_legends: bool = False, region_legends: dict = None) -> None:
        """Generate an HTML index file for the generated scatter plots"""
        files_sorted = sorted(files, key=lambda x: (x['compresa'], x.get('particella', ''), x.get('species', '')))

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

            # Group results by region when one_species_per_graph is True
            if one_species_per_graph:
                current_region = None
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    region_key = (compresa, particella)

                    # Write region header when entering new region
                    if region_key != current_region:
                        current_region = region_key
                        region_title = f"{compresa} - Particella {particella}" if particella else compresa

                        if external_legends and region_legends and region_key in region_legends:
                            f.write(self._write_region_header_html(region_title, region_legends[region_key]))

                    # Write species graph
                    legend_html = self._format_legend_html(result, one_species_per_graph, for_cd=False) if external_legends else None
                    f.write(self._write_graph_item_html(result['species'], result['filepath'].name, legend_html))
            else:
                # Original behavior: one graph per region
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    title = f"{compresa} - Particella {particella}" if particella else compresa

                    legend_html = self._format_legend_html(result, one_species_per_graph, for_cd=False) if external_legends else None
                    f.write(self._write_graph_item_html(title, result['filepath'].name, legend_html))

            f.write('''    </div>
</body>
</html>''')

    def finalize(self, output_dir: str, base_name: str = None) -> None:
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
        self.fragments = []  # List of (name, title) tuples
        self.output_dir = None

    def _equation_to_latex(self, equation_str: str) -> str:
        """Convert equation string like 'y = 0.1234*ln(x) + 5.6789' to LaTeX math."""
        latex_eq = equation_str.replace('*', r'\times ')
        latex_eq = latex_eq.replace('ln', r'\ln')
        return latex_eq

    def _format_legend_latex(self, result: dict, one_species_per_graph: bool, for_cd: bool) -> str:
        """Format legend data as LaTeX."""
        legend = result['legend_data']
        region_stats = legend['region_stats']
        species_stats = legend['species_stats']

        latex = '\\begin{quote}\n\\small\n'

        # Common region stats (displayed once when one_species_per_graph is True)
        if one_species_per_graph and species_stats:
            # Show species-specific stats
            latex += f"\\textbf{{Alberi campionati:}} {species_stats['sampled_trees']}\\\\\n"
            latex += f"\\textbf{{Stima totale alberi:}} {species_stats['estimated_total']}\\\\\n"
            if for_cd:
                latex += f"\\textbf{{Classe diametrica media:}} {species_stats['mean_diameter_class']}\\\\\n"
            else:
                latex += f"\\textbf{{Altezza media:}} {species_stats['mean_height']} m\\\\\n"
        else:
            # Show region-level stats
            latex += f"\\textbf{{Area:}} {region_stats['area_ha']} ha\\\\\n"
            latex += f"\\textbf{{Alberi campionati:}} {region_stats['sampled_trees']}\\\\\n"
            latex += f"\\textbf{{N. aree saggio:}} {region_stats['sample_areas']}\\\\\n"
            latex += f"\\textbf{{Stima totale alberi:}} {region_stats['estimated_total']}\\\\\n"
            latex += f"\\textbf{{Stima alberi / ha:}} {region_stats['estimated_per_ha']}\\\\\n"
            latex += f"\\textbf{{Specie prevalente:}} {region_stats['dominant_species']}\\\\\n"
            if for_cd:
                latex += f"\\textbf{{Classe diametrica media:}} {region_stats['mean_diameter_class']}\\\\\n"
            else:
                latex += f"\\textbf{{Altezza media:}} {region_stats['mean_height']} m\\\\\n"

        # Add polynomial info for curves
        if not for_cd and 'polynomial_info' in legend and legend['polynomial_info']:
            latex += '\\vspace{0.5em}\n\\textbf{Curve di regressione:}\\\\\n'
            for poly in legend['polynomial_info']:
                eq = self._equation_to_latex(poly['equation'])
                # Use inline math mode to keep equation on same line with R² and n
                latex += f"{poly['species']}: ${eq}$ ($R^2$ = {poly['r_squared']:.3f}, $n$ = {poly['n_points']})\\\\\n"

        latex += '\\end{quote}\n'
        return latex

    def _write_section_header(self, f, title_escaped: str, first: bool) -> None:
        """Write section header with optional page break."""
        if not first:
            f.write('\\clearpage\n')
        f.write(f'\\section*{{{title_escaped}}}\n')

    def _write_figure(self, f, filepath: str) -> None:
        """Write a figure block."""
        f.write('\\begin{figure}[H]\n')
        f.write('    \\centering\n')
        f.write(f'    \\includegraphics[width=0.9\\textwidth]{{{filepath}}}\n')
        f.write('\\end{figure}\n')

    def _write_region_stats(self, f, reg_stats: dict) -> None:
        """Write region statistics block."""
        f.write('\\begin{quote}\n\\small\n')
        f.write(f"\\textbf{{Area:}} {reg_stats['area_ha']} ha\\\\\n")
        f.write(f"\\textbf{{N. aree saggio:}} {reg_stats['sample_areas']}\\\\\n")
        f.write(f"\\textbf{{Specie prevalente:}} {reg_stats['dominant_species']}\\\\\n")
        f.write('\\end{quote}\n\n')

    def _write_species_subsection(self, f, result: dict, one_species_per_graph: bool,
                                  for_cd: bool, external_legends: bool) -> None:
        """Write a species subsection with figure and optional legend."""
        species = result['species']
        filepath = result['filepath']
        species_escaped = species.replace('_', r'\_')

        f.write(f'\\subsection*{{{species_escaped}}}\n')
        self._write_figure(f, filepath.name)

        if external_legends:
            f.write(self._format_legend_latex(result, one_species_per_graph, for_cd))

    def generate_index_cd(self, files: list, output_dir: str, one_species_per_graph: bool = False,
                          external_legends: bool = False, region_legends: dict = None) -> None:
        """Generate a LaTeX fragment for diameter class histograms"""
        self.output_dir = Path(output_dir)
        files_sorted = sorted(files, key=lambda x: (x['compresa'], x.get('particella', ''), x.get('species', '')))
        latex_file = self.output_dir / 'classi-diametriche.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            first = True

            if one_species_per_graph:
                # One graph per species, grouped by region
                current_region = None
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    region_key = (compresa, particella)

                    # Write region header when entering new region
                    if region_key != current_region:
                        current_region = region_key
                        region_title = f"{compresa} - Particella {particella}" if particella else compresa
                        title_escaped = region_title.replace('_', r'\_')

                        self._write_section_header(f, title_escaped, first)
                        first = False

                        if external_legends and region_legends and region_key in region_legends:
                            self._write_region_stats(f, region_legends[region_key])

                    # Write species subsection
                    self._write_species_subsection(f, result, one_species_per_graph,
                                                   for_cd=True, external_legends=external_legends)
            else:
                # Original behavior: one graph per region, all species combined
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    title = f"{compresa} - Particella {particella}" if particella else compresa
                    title_escaped = title.replace('_', r'\_')

                    self._write_section_header(f, title_escaped, first)
                    first = False
                    self._write_figure(f, result['filepath'].name)

                    if external_legends:
                        f.write(self._format_legend_latex(result, one_species_per_graph, for_cd=True))
                    f.write('\n')

        self.fragments.append(('classi-diametriche.tex', 'Distribuzione piante per classe diametrica'))

    def generate_index_ci(self, files: list, output_dir: str, one_species_per_graph: bool = False,
                          external_legends: bool = False, region_legends: dict = None) -> None:
        """Generate a LaTeX fragment for height-diameter curves"""
        self.output_dir = Path(output_dir)
        files_sorted = sorted(files, key=lambda x: (x['compresa'], x.get('particella', ''), x.get('species', '')))
        latex_file = self.output_dir / 'curve-ipsometriche.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            # Write explanation first
            f.write(r'''
Le curve ipsometriche sono state calcolate con un modello di regressione logaritmico, per
ogni combinazione di compresa, particella e genere. Vengono visualizzate solo per
combinazioni con almeno 10 punti ($n \ge 10$).

''')

            # Group results by region when one_species_per_graph is True
            first = True

            if one_species_per_graph:
                # One graph per species, grouped by region
                current_region = None
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    region_key = (compresa, particella)

                    # Write region header when entering new region
                    if region_key != current_region:
                        current_region = region_key
                        region_title = f"{compresa} - Particella {particella}" if particella else compresa
                        title_escaped = region_title.replace('_', r'\_')

                        self._write_section_header(f, title_escaped, first)
                        first = False

                        if external_legends and region_legends and region_key in region_legends:
                            self._write_region_stats(f, region_legends[region_key])

                    # Write species subsection
                    self._write_species_subsection(f, result, one_species_per_graph,
                                                   for_cd=False, external_legends=external_legends)
            else:
                # Original behavior: one graph per region, all species combined
                for result in files_sorted:
                    compresa = result['compresa']
                    particella = result['particella']
                    title = f"{compresa} - Particella {particella}" if particella else compresa
                    title_escaped = title.replace('_', r'\_')

                    self._write_section_header(f, title_escaped, first)
                    first = False
                    self._write_figure(f, result['filepath'].name)

                    if external_legends:
                        f.write(self._format_legend_latex(result, one_species_per_graph, for_cd=False))
                    f.write('\n')

        self.fragments.append(('curve-ipsometriche.tex', 'Curve ipsometriche'))

    def finalize(self, output_dir: str, base_name: str) -> None:
        """Generate master document and optionally compile to PDF"""
        if not self.fragments:
            return

        output_dir = Path(output_dir)
        master_file = output_dir / f'{base_name}.tex'

        # Generate master document that includes fragments
        with open(master_file, 'w', encoding='utf-8') as f:
            f.write(self.HEADER)
            f.write(r'''\title{Società Agricola La Foresta\\Analisi Accrescimenti}
\date{}
\author{}
\begin{document}
\maketitle
''')
            for fragment_name, fragment_title in self.fragments:
                f.write(f'\\section{{{fragment_title}}}\n')
                f.write(f'\\input{{{fragment_name}}}\n')
                f.write('\\clearpage\n\n')

            f.write(self.FOOTER)

        if not self.compile_pdf:
            return

        pdf_file = master_file.with_suffix('.pdf')
        if pdf_file.exists():
            pdf_file.unlink()

        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', master_file.name],
                cwd=master_file.parent,
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise RuntimeError("pdflatex returned non-zero exit code")

            if pdf_file.exists():
                # Clean up auxiliary files
                for ext in ['.aux', '.log', '.out', '.toc']:
                    aux_file = master_file.with_suffix(ext)
                    if aux_file.exists():
                        aux_file.unlink()
            else:
                raise RuntimeError("File PDF non creato")
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

def format_region_stats(region: pd.Series, region_data: pd.DataFrame, for_cd: bool = True) -> dict:
    """Generate statistics for a region (shared across species when one_species_per_graph is enabled).

    Returns a dict with common stats that don't vary by species.
    """
    with italian_locale():
        stats = {
            'area_ha': locale.format_string('%.2f', region["area_ha"]),
            'sampled_trees': len(region_data),
            'sample_areas': region["sample_areas"],
            'estimated_total': locale.format_string('%d', region["estimated_total"]),
            'estimated_per_ha': locale.format_string('%d', round(region["estimated_total"]/region["area_ha"])),
            'dominant_species': region_data["Genere"].mode().iloc[0] if len(region_data) > 0 else 'N/A'
        }
        if for_cd:
            stats['mean_diameter_class'] = locale.format_string('%d', round(region_data["Classe diametrica"].mean()))
        else:
            stats['mean_height'] = locale.format_string('%.1f', region_data["h(m)"].mean())
    return stats

def format_species_stats(species_data: pd.DataFrame, region: pd.Series, for_cd: bool = True) -> dict:
    """Generate statistics specific to a species.

    Returns a dict with species-specific stats.
    """
    with italian_locale():
        stats = {
            'sampled_trees': len(species_data),
            'estimated_total': locale.format_string('%d', round(
                (len(species_data) / region['sample_areas']) * SAMPLE_AREAS_PER_HA * region['area_ha'])),
        }
        if for_cd:
            stats['mean_diameter_class'] = locale.format_string('%d', round(species_data["Classe diametrica"].mean()))
        else:
            stats['mean_height'] = locale.format_string('%.1f', species_data["h(m)"].mean())
    return stats

def create_cd(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str, set_title: bool,
              one_species_per_graph: bool = False, external_legends: bool = False) -> list:
    """
    Create histogram(s) for a specific region showing tree distribution by diameter class.

    Returns a list of result dictionaries, one per graph created.
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)

    if particella is None:
        region_data = trees[trees['Compresa'] == compresa]
        print_name = compresa
    else:
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        print_name = f"{compresa}-{particella}"

    assert len(region_data) > 0, f"Nessun dato per {print_name}"

    # Data traceability
    species_list = sorted(region_data['Genere'].unique())
    print(f"Generazione istogramma per {print_name}:")
    print(f"  Alberi campionati: {len(region_data)}")
    print(f"  Specie: {', '.join(species_list)}")
    print(f"  Stima totale: {region['estimated_total']} alberi")

    # Get region-level stats (shared across all species)
    region_stats = format_region_stats(region, region_data, for_cd=True)

    # Determine which species to process
    if one_species_per_graph:
        species_to_process = sorted(region_data['Genere'].unique())
    else:
        species_to_process = [None]  # None means all species in one graph

    results = []
    for species in species_to_process:
        if species is not None:
            # Single species mode
            species_data = region_data[region_data['Genere'] == species]
            title_suffix = f" - {species}"
            filename_suffix = f"_{species}"
            species_list = [species]
            species_stats = format_species_stats(species_data, region, for_cd=True)
        else:
            # All species mode
            species_data = region_data
            title_suffix = ""
            filename_suffix = ""
            species_list = sorted(region_data['Genere'].unique())
            species_stats = None

        if particella is None:
            title = f'Distribuzione alberi per classe diametrica - {compresa}{title_suffix}' if set_title else None
        else:
            title = f'Distribuzione alberi per classe diametrica - {compresa} Particella {particella}{title_suffix}' if set_title else None

        filename = f"{region['sort_key']}{filename_suffix}_classi-diametriche.png"

        # Calculate counts
        counts = (species_data.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
                  * SAMPLE_AREAS_PER_HA / region['sample_areas'])

        # Create figure
        fig, ax = plt.subplots(figsize=(4, 2.5))

        # Plot bars
        bottom = np.zeros(len(counts.index))
        for sp in species_list:
            if sp not in counts.columns:
                continue
            values = counts[sp].values
            ax.bar(counts.index, values, bottom=bottom,
                   label=sp, color=color_map[sp],
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

        # Add legend to graph if external_legends is False
        if not external_legends:
            ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

        # Build stats text - conditional based on one_species_per_graph
        if not external_legends:
            if one_species_per_graph and species_stats:
                # Species-specific stats
                stats_text = f"""
Alberi campionati: {species_stats['sampled_trees']}
Stima totale alberi: {species_stats['estimated_total']}
Classe diametrica media: {species_stats['mean_diameter_class']}""".strip()
            else:
                # Region-level stats
                stats_text = f"""
Area: {region_stats['area_ha']} ha
Alberi campionati: {region_stats['sampled_trees']}
N. aree saggio: {region_stats['sample_areas']}
Stima totale alberi: {region_stats['estimated_total']}
Stima alberi / ha: {region_stats['estimated_per_ha']}
Specie prevalente: {region_stats['dominant_species']}
Classe diametrica media: {region_stats['mean_diameter_class']}""".strip()

            ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))

        plt.tight_layout()

        filepath = Path(output_dir) / filename
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Salvato istogramma per {print_name}{title_suffix} in {filepath}")
        plt.close(fig)

        # Prepare legend data for external display
        legend_data = {
            'region_stats': region_stats,
            'species_stats': species_stats,
            'species_list': species_list,
        }

        results.append({
            'compresa': compresa,
            'particella': particella,
            'filepath': filepath,
            'species': species,
            'legend_data': legend_data
        })

    return results

#
# Create scatter plots for height vs diameter class relationship
#

def create_ci(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str,
              hfuncs: dict, omit_unknown: bool, set_title: bool,
              one_species_per_graph: bool = False, external_legends: bool = False) -> list:
    """
    Create scatter plot(s) for a specific region showing height vs diameter class relationship.

    Returns a list of result dictionaries, one per graph created.
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)

    if particella is None:
        region_data = trees[trees['Compresa'] == compresa]
        print_name = compresa
    else:
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        print_name = f"{compresa}-{particella}"

    assert len(region_data) > 0, f"Nessun dato per {print_name}"

    # Data traceability
    species_list = sorted(region_data['Genere'].unique())
    print(f"Generazione curve ipsometriche per {print_name}:")
    print(f"  Alberi campionati: {len(region_data)}")
    print(f"  Specie: {', '.join(species_list)}")
    print(f"  Altezza media: {region_data['h(m)'].mean():.1f} m")

    # Get region-level stats
    region_stats = format_region_stats(region, region_data, for_cd=False)

    # Determine which species to process
    if one_species_per_graph:
        species_to_process = sorted(region_data['Genere'].unique())
    else:
        species_to_process = [None]  # None means all species in one graph

    results = []
    for target_species in species_to_process:
        if target_species is not None:
            # Single species mode - check if we should skip this species
            height_func = get_height_function(hfuncs, compresa, target_species)
            if hfuncs and height_func is None and omit_unknown:
                print(f"Genere {target_species} non presente nelle funzioni di altezza, omesso dalle curve ipsometriche")
                continue

            species_data = region_data[region_data['Genere'] == target_species]
            title_suffix = f" - {target_species}"
            filename_suffix = f"_{target_species}"
            species_list = [target_species]
            species_stats = format_species_stats(species_data, region, for_cd=False)
        else:
            # All species mode
            species_data = region_data
            title_suffix = ""
            filename_suffix = ""
            species_list = sorted(region_data['Genere'].unique())
            species_stats = None

        if particella is None:
            title = f'Curve ipsometriche - {compresa}{title_suffix}' if set_title else None
        else:
            title = f'Curve ipsometriche - {compresa} Particella {particella}{title_suffix}' if set_title else None

        filename = f"{region['sort_key']}{filename_suffix}_curve-ipsometriche.png"

        # Create figure
        fig, ax = plt.subplots(figsize=(4, 3))

        polynomial_info = []
        ymax = 0

        for species in species_list:
            sp_data = species_data[species_data['Genere'] == species]
            x = sp_data['Classe diametrica'].values

            height_func = get_height_function(hfuncs, compresa, species)

            # Skip species not present in height functions if requested
            if hfuncs and height_func is None and omit_unknown:
                print(f"Genere {species} non presente nelle funzioni di altezza, omesso dalle curve ipsometriche")
                continue

            # Use interpolated heights if available, otherwise use original h(m)
            if height_func is not None:
                d_cm = sp_data['D(cm)'].values
                y = np.array([height_func(d) for d in d_cm])
            else:
                y = sp_data['h(m)'].values
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

        # Add legend to graph if external_legends is False
        if not external_legends:
            ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

        # Build stats text - conditional based on one_species_per_graph
        if not external_legends:
            if one_species_per_graph and species_stats:
                # Species-specific stats
                stats_text = f"""
Alberi campionati: {species_stats['sampled_trees']}
Stima totale alberi: {species_stats['estimated_total']}
Altezza media: {species_stats['mean_height']} m""".strip()
            else:
                # Region-level stats
                stats_text = f"""
Area: {region_stats['area_ha']} ha
Alberi campionati: {region_stats['sampled_trees']}
N. aree saggio: {region_stats['sample_areas']}
Stima totale alberi: {region_stats['estimated_total']}
Stima alberi / ha: {region_stats['estimated_per_ha']}
Specie prevalente: {region_stats['dominant_species']}
Altezza media: {region_stats['mean_height']} m""".strip()

            ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))

        # Add polynomial info to graph if external_legends is False
        if not external_legends and polynomial_info:
            poly_text = ""
            for i in polynomial_info:
                poly_text += f"{i['species']}: {i['equation']} (R² = {i['r_squared']:.3f}, n = {i['n_points']})\n"

            ax.text(0.01, -0.25, poly_text.strip(), transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    fontsize=5, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, linewidth=0.2))

        plt.tight_layout()

        filepath = Path(output_dir) / filename
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Salvato grafico ipsometrico per {print_name}{title_suffix} in {filepath}")
        plt.close(fig)

        # Prepare legend data for external display
        legend_data = {
            'region_stats': region_stats,
            'species_stats': species_stats,
            'species_list': species_list,
            'polynomial_info': polynomial_info,
        }

        results.append({
            'compresa': compresa,
            'particella': particella,
            'filepath': filepath,
            'species': target_species,
            'legend_data': legend_data
        })

    return results

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
    g1.add_argument('--un-genere-per-grafico', action='store_true',
                    default=False,
                    help='Genera un grafico separato per ogni genere (miglior visibilità)')
    g1.add_argument('--legenda-esterna', action='store_true',
                    default=True,
                    help='Mostra statistiche e equazioni interpolanti sotto il grafico invece che nel grafico')
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
                    help='Directory contenente i file di input')
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
    g4.add_argument('--nome-report', type=str, default='report',
                    help='Nome del file tex/pdf (senza suffisso)')
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
    all_species = sorted(alberi_fustaia['Genere'].unique())

    print("=" * 70)
    print("ANALISI ACCRESCIMENTI - CONFIGURAZIONE")
    print("=" * 70)
    print(f"Dati filtrati: {len(alberi_fustaia)} campioni di alberi a fustaia (su {len(alberi)} totali)")
    print(f"Specie presenti: {', '.join(all_species)}")
    print(f"Formato output: {args.formato_output.upper()}")
    print(f"Granularità: {'Per particella' if args.per_particella else 'Per compresa'}")
    print(f"Layout grafici: {'Un genere per grafico' if args.un_genere_per_grafico else 'Tutti i generi combinati'}")
    print(f"Legende: {'Esterne' if args.legenda_esterna else 'Nei grafici'}")
    print("=" * 70)
    print()

    # Create consistent color mapping for all species
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

    # For LaTeX/PDF, use single output directory; for HTML, use subdirectories
    if args.formato_output == 'html':
        cd_output_dir = Path(args.prefisso_output) / 'classi-diametriche'
        ci_output_dir = Path(args.prefisso_output) / 'curve-ipsometriche'
    else:
        cd_output_dir = Path(args.prefisso_output) / 'tex'
        ci_output_dir = Path(args.prefisso_output) / 'tex'

    if args.genera_classi_diametriche:
        os.makedirs(cd_output_dir, exist_ok=True)

        all_results = []
        region_legends = {}
        for _, row in regions.sort_values(['sort_key']).iterrows():
            results = create_cd(trees=alberi_fustaia, region=row,
                               color_map=color_map, output_dir=cd_output_dir,
                               set_title=args.formato_output == 'html',
                               one_species_per_graph=args.un_genere_per_grafico,
                               external_legends=args.legenda_esterna)
            all_results.extend(results)

            # Store region-level legends for external display
            if args.un_genere_per_grafico and args.legenda_esterna and results:
                region_key = (results[0]['compresa'], results[0]['particella'])
                region_legends[region_key] = results[0]['legend_data']['region_stats']

        formatter.generate_index_cd(all_results, cd_output_dir,
                                   one_species_per_graph=args.un_genere_per_grafico,
                                   external_legends=args.legenda_esterna,
                                   region_legends=region_legends)
        print(f"Istogrammi classi diametriche salvati in '{cd_output_dir}'")

    if args.genera_curve_ipsometriche:
        os.makedirs(ci_output_dir, exist_ok=True)

        all_results = []
        region_legends = {}
        for _, row in regions.sort_values(['sort_key']).iterrows():
            results = create_ci(trees=alberi_fustaia, region=row,
                               color_map=color_map, output_dir=ci_output_dir,
                               hfuncs=hfuncs, omit_unknown=args.ometti_generi_sconosciuti,
                               set_title=args.formato_output == 'html',
                               one_species_per_graph=args.un_genere_per_grafico,
                               external_legends=args.legenda_esterna)
            all_results.extend(results)

            # Store region-level legends for external display
            if args.un_genere_per_grafico and args.legenda_esterna and results:
                region_key = (results[0]['compresa'], results[0]['particella'])
                region_legends[region_key] = results[0]['legend_data']['region_stats']

        formatter.generate_index_ci(all_results, ci_output_dir,
                                   one_species_per_graph=args.un_genere_per_grafico,
                                   external_legends=args.legenda_esterna,
                                   region_legends=region_legends)
        print(f"Curve ipsometriche salvate in '{ci_output_dir}'")

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
        # Use whichever directory was set (they're the same for LaTeX mode)
        final_dir = cd_output_dir if args.genera_classi_diametriche else ci_output_dir
        formatter.finalize(final_dir, args.nome_report)

    with italian_locale():
        print()
        print("=" * 70)
        print("ANALISI COMPLETATA")
        print("=" * 70)
        print("\nRiepilogo dati per regione:")

        for _, row in regions.sort_values(['sort_key']).iterrows():
            trees_per_ha = row['estimated_total'] / row['area_ha']
            label = f"{row['Compresa']} - Particella {row['Particella']}" if args.per_particella else row['Compresa']
            print(f"  {label}:")
            print(f"    • {row['sampled_trees']} alberi campionati, {row['sample_areas']} aree saggio")
            print(f"    • {locale.format_string('%.2f', row['area_ha'])} ha")
            print(f"    • Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")
        print("=" * 70)

if __name__ == "__main__":
    main()
