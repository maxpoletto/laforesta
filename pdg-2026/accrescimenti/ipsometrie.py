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

def log_func(x, a, b):
    return a * np.log(x) + b

def lin_func(x, a, b):
    return a * x + b

def main():
    parser = argparse.ArgumentParser(description='Analisi altezza-diametro per specie arboree')
    parser.add_argument('csv_file', help='File CSV con 3 colonne: Specie, Diametro (cm), Altezza (m)')
    parser.add_argument('--fit', choices=['lineare', 'logaritmico'], default='logaritmico',
                        help='Tipo di fit: lineare o logaritmico (default: logaritmico)')
    parser.add_argument('-o', '--output', default='ipsometrie.png',
                        help='Nome del file di output (default: ipsometrie.png)')

    args = parser.parse_args()

    # Load data from CSV
    df = load_data_from_csv(args.csv_file)
    print(f"Dati caricati da {args.csv_file}: {len(df)} osservazioni")

    # Choose fit function
    if args.fit == 'lineare':
        func = lin_func
        func_name = 'lineare'
    elif args.fit == 'logaritmico':
        func = log_func
        func_name = 'logaritmico'
    else:
        print(f"Errore: Tipo di fit non valido: {args.fit}")
        sys.exit(1)

    plt.figure(figsize=(12, 7))
 
    species = sorted(df['Specie'].unique())
 
    # Colors for consistent plotting
    colors = plt.cm.tab10(np.linspace(0, 1, len(species)))
 
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

            print(f"\n{sp}:")
            print(f"  n = {len(x)}")
            if args.fit == 'linear':
                print(f"  y = {a:.3f}·x + {b:.3f}")
            else:
                print(f"  y = {a:.3f}·ln(x) + {b:.3f}")
            print(f"  R² = {r2:.3f}")
        else:
            print(f"\n{sp}: Troppi pochi punti ({len(x)}) per la regressione")

    # Extract output basename without extension for title
    output_basename = os.path.splitext(os.path.basename(args.output))[0]
    
    plt.xlabel('Diametro a cm 150 (cm)', fontsize=11)
    plt.ylabel('Altezza (m)', fontsize=11)
    plt.title(f'{output_basename}: Altezza vs Diametro per Specie (con fit {func_name})', fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot salvato in {args.output} (fit {func_name})")

if __name__ == '__main__':
    main()
