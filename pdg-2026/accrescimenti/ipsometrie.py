#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse
import sys
import os

def load_data_from_csv(filename):
    """Load data from a 3-column CSV file."""
    try:
        df = pd.read_csv(filename, header=0, names=['Specie', 'Diametro', 'Altezza'])
        return df
    except FileNotFoundError:
        print(f"Errore: File '{filename}' non trovato")
        sys.exit(1)
    except Exception as e:
        print(f"Errore nel leggere il file: {e}")
        sys.exit(1)

def load_data_from_6column_csv(filename):
    """Load data from a 6-column CSV file with Compresa, ADS, Specie, Diametro, Altezza, Metodo."""
    try:
        df = pd.read_csv(filename)
        # Keep only relevant columns
        df = df[['Compresa', 'Particella', 'Specie', 'Diametro', 'Altezza']]
        return df
    except FileNotFoundError:
        print(f"Errore: File '{filename}' non trovato")
        sys.exit(1)
    except Exception as e:
        print(f"Errore nel leggere il file: {e}")
        sys.exit(1)

def log_func(x, a, b):
    return a * np.log(x) + b

def lin_func(x, a, b):
    return a * x + b

def analyze_and_plot(df, func, func_name, plot_file, fit_file=None, title_prefix=''):
    """
    Perform analysis and generate plot for a dataset.

    Args:
        df: DataFrame with columns Specie, Diametro, Altezza
        func: Fit function to use
        func_name: Name of the fit function for display
        plot_file: Path to save the plot
        fit_file: Optional path to save fit results (if None, print to stdout)
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(12, 7))

    species = sorted(df['Specie'].unique())

    # Colors for consistent plotting
    colors = plt.cm.tab10(np.linspace(0, 1, len(species)))

    # Prepare fit results output
    fit_results = []
    if title_prefix:
        fit_results.append(f"{title_prefix}")
    fit_results.append(f"Tipo di fit: {func_name}")
    fit_results.append(f"Dati: {len(df)} osservazioni")

    for i, sp in enumerate(species):
        species_data = df[df['Specie'] == sp]
        x = species_data['Diametro'].values
        y = species_data['Altezza'].values

        plt.scatter(x, y, label=sp, alpha=0.7, s=50, color=colors[i])

        if len(x) >= 3:
            popt, _ = curve_fit(func, x, y)
            a, b = popt

            y_pred = func(x, a, b)
            r2 = r2_score(y, y_pred)

            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = func(x_smooth, a, b)

            plt.plot(x_smooth, y_smooth, '--', color=colors[i],
                    alpha=0.6, linewidth=2,
                    label=f'{sp} fit (R²={r2:.3f})')

            fit_results.append(f"\n{sp}:")
            fit_results.append(f"  n = {len(x)}")
            if func_name == 'lineare':
                fit_results.append(f"  y = {a:.3f}·x + {b:.3f}")
            else:
                fit_results.append(f"  y = {a:.3f}·ln(x) + {b:.3f}")
            fit_results.append(f"  R² = {r2:.3f}")
        else:
            fit_results.append(f"\n{sp}: Troppi pochi punti ({len(x)}) per la regressione")

    # Extract output basename without extension for title
    prefix = f"{title_prefix}: " if title_prefix else ""
    plot_title = f'{prefix}Altezza vs Diametro per Specie (con fit {func_name})'

    plt.xlabel('Diametro a cm 150 (cm)', fontsize=11)
    plt.ylabel('Altezza (m)', fontsize=11)
    plt.title(plot_title, fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(plot_file, dpi=150, bbox_inches='tight')

    # Output fit results
    fit_text = '\n'.join(fit_results)
    if fit_file:
        with open(fit_file, 'w') as f:
            f.write(fit_text + '\n')
        print(f"Plot salvato in {plot_file}, fit salvato in {fit_file}")
    else:
        print(fit_text)
        print(f"\nPlot salvato in {plot_file} (fit {func_name})")

def main():
    parser = argparse.ArgumentParser(description='Analisi altezza-diametro per specie arboree')
    parser.add_argument('csv_file', help='File CSV con dati alberi')
    parser.add_argument('--format', choices=['3c', '6c'], default='6c',
                        help='Formato del CSV: 3 colonne (Specie,Diametro,Altezza) o 6 colonne (Compresa,ADS,Specie,Diametro,Altezza,Metodo)')
    parser.add_argument('--fit', choices=['lin', 'log'], default='log',
                        help='Tipo di fit: lineare (lin) o logaritmico (log) (default: log)')
    parser.add_argument('-o', '--output',
                        help='Prefisso per i file di output (default: ipsometrie)', default='ipsometrie')
    parser.add_argument('--per-particella', action='store_true', default=False,
                        help='Analisi per particella (usare solo con --format=6c)')

    args = parser.parse_args()

    # Choose fit function
    if args.fit == 'lin':
        func = lin_func
        func_name = 'lineare'
    elif args.fit == 'log':
        func = log_func
        func_name = 'logaritmico'
    else:
        print(f"Errore: Tipo di fit non valido: {args.fit}")
        sys.exit(1)

    if args.format == '3c':
        # Original 3-column format
        plot_file = f"{args.output}-plot.png"
        fit_file = f"{args.output}-fit.txt"
        df = load_data_from_csv(args.csv_file)
        print(f"Dati caricati da {args.csv_file}: {len(df)} osservazioni")
        analyze_and_plot(df, func, func_name, plot_file, fit_file)

    elif args.format == '6c':
        # New 6-column format with Compresa grouping
        prefix = f"{args.output}"

        df = load_data_from_6column_csv(args.csv_file)
        print(f"Dati caricati da {args.csv_file}: {len(df)} osservazioni")

        # Group by Compresa or Particella
        if args.per_particella:
            group_by = 'Particella'
        else:
            group_by = 'Compresa'
        regions = sorted(df[group_by].unique())
        plural = 'Particelle' if args.per_particella else 'Comprese'
        print(f"{plural} trovate: {', '.join(str(r) for r in regions)}\n")

        for region in regions:
            region_df = df[df[group_by] == region][['Specie', 'Diametro', 'Altezza']].copy()
            plot_file = f"{prefix}-{region}-plot.png"
            fit_file = f"{prefix}-{region}-fit.txt"
            analyze_and_plot(region_df, func, func_name, plot_file,
                           fit_file=fit_file, title_prefix=f"{group_by} {region}")

if __name__ == '__main__':
    main()
