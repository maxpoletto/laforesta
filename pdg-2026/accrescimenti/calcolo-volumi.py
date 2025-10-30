#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy import stats
import argparse

#np.set_printoptions(suppress=True, precision=2)

cov = {
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
    'Pino': np.array([ # Messo uguale a pino laricio.
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
    'Sorbo': np.array([ # Come il ciliegio ("Altre latifoglie", Tabacchi p.400).
        [  1.5377e1,   0,         0 ],
        [  8.9101e-3,  9.7080e-6, 0 ],
        [ -2.6997,    -1.8132e-3, 4.9690e-1 ]
    ]),
}

bt = {
    'Abete':          np.array([ -1.8381,    3.7836e-2, 3.9934e-1 ]),
    'Acero':          np.array([  1.6905,    3.7082e-2 ]),
    'Castagno':       np.array([ -2,         3.6524e-2, 7.4466e-1 ]),
    'Cerro':          np.array([ -4.3221e-2, 3.8079e-2 ]),
    'Ciliegio':       np.array([  2.3118,    3.1278e-2, 3.7159e-1 ]),
    'Douglas':        np.array([ -7.9946,    3.3343e-2, 1.2186 ]),
    'Faggio':         np.array([  8.1151e-1, 3.8965e-2 ]),
    'Leccio':         np.array([ -2.2219,    3.9685e-2, 6.2762e-1 ]),
    'Ontano':         np.array([ -2.2932e1,  3.2641e-2, 2.991 ]),
    'Pino':           np.array([  6.4383,    3.8594e-2 ]), # Messo uguale a pino laricio.
    'Pino Laricio':   np.array([  6.4383,    3.8594e-2 ]),
    'Pino Marittimo': np.array([  2.9963,    3.8302e-2 ]),
    'Pino Nero':      np.array([ -2.1480e1,  3.3448e-2, 2.9088]),
    'Sorbo':          np.array([  2.3118,    3.1278e-2, 3.7159e-1 ]), # Come il ciliegio ("Altre latifoglie", Tabacchi p.400).
}

s2 = {
    'Abete': 1.5284e-5,
    'Acero': 2.2710e-5,
    'Castagno': 3.0491e-5,
    'Cerro': 2.5866e-5,
    'Ciliegio': 4.0506e-5,
    'Douglas': 9.0103e-6,
    'Faggio': 5.1468e-5,
    'Leccio': 6.0915e-5,
    'Ontano': 3.9958e-5,
    'Pino': 6.3906e-6, # Messo uguale a pino laricio.
    'Pino Laricio': 6.3906e-6,
    'Pino Marittimo': 1.4031e-5,
    'Pino Nero': 1.7090e-5,
    'Sorbo': 4.0506e-5, # Come il ciliegio ("Altre latifoglie", Tabacchi p.400).
}

# Make covariance matrices symmetric
for genere in cov:
    cov_matrix = cov[genere]
    cov[genere] = cov_matrix + cov_matrix.T - np.diag(np.diag(cov_matrix))

# Compute b arrays as transposes of bt (making them column vectors)
b = {}
for genere in bt:
    b[genere] = bt[genere].reshape(-1, 1)  # Convert to column vector (n x 1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate forest volume with confidence intervals')
parser.add_argument('csv_file', help='CSV file with tree data (DatiBase table)')
parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed calculations for each group')
args = parser.parse_args()

csv_file = args.csv_file
verbose = args.verbose

# Read CSV file
try:
    df_table = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(2)

# Print statistics about the df_table
if verbose:
    print("\nStatistiche sulla tabella 'Dati Base':")
    print(f"- Numero di righe: {df_table.shape[0]}")
    print(f"- Numero di colonne: {df_table.shape[1]}")
    print(f"- Nomi colonne: {list(df_table.columns)}")
    print("\nStatistiche descrittive (valori numerici):")
    print(df_table.describe(include='all'))

# Group data by Compresa, Particella, and Genere
grouped = df_table.groupby(['Compresa', 'Particella', 'Genere'])

if verbose:
    print(f"\nRaggruppamento per Compresa - Particella - Genere")
    print(f"Numero totale di gruppi: {len(grouped)}")

# Store results for summary table
results = []

# Process each group
for group_key, group_data in grouped:
    compresa, particella, genere = group_key
    n_g = len(group_data)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Gruppo: Compresa={compresa}, Particella={particella}, Genere={genere}")
        print(f"Numero di alberi (n_g): {n_g}")

    # Check if we have coefficients for this genus
    if genere not in b:
        print(f"\n⚠️  ATTENZIONE: Coefficienti non definiti per genere '{genere}'")
        exit(1)

    # Get coefficient vector b for this genus
    b_genere = b[genere]
    if verbose:
        print(f"\nVettore coefficienti b per {genere} ({b_genere.shape[0]} x {b_genere.shape[1]}):")
        print(b_genere)
    assert b_genere.shape[0] == 2 or b_genere.shape[0] == 3, f"Numero di coefficienti per {genere} non valido: {b_genere.shape[0]}"

    # Create D0 matrix
    # Column 1: all ones
    D0 = np.zeros((n_g, b_genere.shape[0]))
    D0[:, 0] = 1  # First column is all 1s

    # Column 2: D(cm)^2 * h(m) for each tree
    D_cm = group_data['D(cm)'].values
    h_m = group_data['h(m)'].values
    D0[:, 1] = (D_cm ** 2) * h_m

    # Column 3: D(cm) for each tree
    if b_genere.shape[0] == 3:
        D0[:, 2] = D_cm

    # Create D1 matrix - sum of all rows in D0
    D1 = np.sum(D0, axis=0).reshape(1, b_genere.shape[0])

    if verbose:
        print(f"\nMatrice D0 ({n_g} x {b_genere.shape[0]}):")
        print(D0)

        print(f"\nMatrice D1 (1 x {b_genere.shape[0]}) [somma delle righe di D0]:")
        print(D1)

    T0 = D1 @ b_genere / 1000 # Convert to m³

    if verbose:
        print(f"\nVolume atteso T0 = D1 × b:")
        print(T0)
        print(f"  T0 = {T0[0,0]:.4f} m³")

    # Get covariance matrix for this genus
    assert genere in cov and genere in s2, f"Genere '{genere}' non riconosciuto"

    cov_matrix = cov[genere]
    if verbose:
        print(f"\nMatrice di varianza-covarianza per {genere} ({cov_matrix.shape[0]}x{cov_matrix.shape[1]}):")
        print(cov_matrix)

    # Step 1: V0 = (D1 @ cov @ D1.T)
    V0 = (D1 @ cov_matrix @ D1.T)[0, 0]
    if verbose:
        print(f"\nV0 = varianza attesa")
        print(f"V0 = {V0:.6e}")

    # Step 2: V1 = sum of squares of second column of D0 * s2
    V1 = np.sum(D0[:, 1] ** 2) * s2[genere]
    if verbose:
        print(f"\nV1 = varianza residua")
        print(f"V1 = {V1:.6e}")

    # Step 3: Get t-statistic for 95% CI
    # T.INV.2T(0.05, n) in Excel = two-tailed t at alpha=0.05
    t_stat = stats.t.ppf(1 - 0.05/2, n_g)
    if verbose:
        print(f"\nt-statistic (α=0.05, df={n_g}):")
        print(f"t = {t_stat:.4f}")

    # Step 4: Confidence interval = T0 ± t * sqrt(V0 + V1)
    margin_of_error = t_stat * np.sqrt(V0 + V1) / 1000 # Convert to m³
    ci_lower = T0[0, 0] - margin_of_error
    ci_upper = T0[0, 0] + margin_of_error

    if verbose:
        print(f"\nMargine di errore = t * sqrt(V0 + V1) = {margin_of_error:.4f} m³")
        print("\nRiepilogo")
        print(f"  Numero alberi: {n_g}")
        print(f"  Volume totale atteso: {T0[0,0]:.4f} m³")
        print(f"  Volume medio per albero: {T0[0,0]/n_g:.4f} m³")
        print(f"  IF 95%: [{ci_lower:.4f}, {ci_upper:.4f}] m³")
        print(f"  Ampiezza IC: {ci_upper - ci_lower:.4f} m³")

    # Store results
    results.append({
        'Compresa': compresa,
        'Particella': particella,
        'Genere': genere,
        'N_alberi': n_g,
        'Volume_m3': T0[0, 0],
        'IC_low_m3': ci_lower,
        'IC_high_m3': ci_upper
    })

# Print summary table
if verbose:
    print("\n" + "="*80)

print("Volumi per particella e genere")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print(f"Volume totale: {results_df['Volume_m3'].sum():.4f} m³")
print(f"IC totale: [{results_df['IC_low_m3'].sum():.4f}, {results_df['IC_high_m3'].sum():.4f}] m³")
print("="*80)
