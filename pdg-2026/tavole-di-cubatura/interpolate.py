#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--volume-dendrometrico', action='store_true', help='Plot volume dendrometrico')
parser.add_argument('--volume-cormometrico', action='store_true', help='Plot volume cormometrico')
parser.add_argument('--interpolation', type=str, help='Interpolation method', default='quadratic')
parser.add_argument('--method', type=str, choices=['interpolate', 'fit'], default='fit',
                    help='Method to fill missing height values: interpolate or fit')
parser.add_argument('-i', '-input', type=str, help='Input CSV file', default='alsometrie.csv')
args = parser.parse_args()

def fit_logarithmic_curve(x, y) -> tuple[tuple[float, float], float]:
    """Fit logarithmic curve to data using np.polyfit and return parameters and R²"""
    x_clean = x[x > 0]
    y_clean = y[x > 0]

    if len(x_clean) < 10:
        return None, None

    # Fit logarithmic function: y = a*ln(x) + b
    # Using linear fit to log-transformed x values
    coeffs = np.polyfit(np.log(x_clean), y_clean, 1)
    a, b = coeffs

    # Calculate predictions and R²
    y_pred = a * np.log(x_clean) + b
    if np.var(y_clean) > 0:  # Avoid division by zero
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0  # No variance in y values

    return (a, b), r2

parameters = ['Altezza indicativa']
if args.volume_dendrometrico:
    parameters.append('Volume dendrometrico')
if args.volume_cormometrico:
    parameters.append('Volume cormometrico')

df = pd.read_csv(args.i)

# Convert numeric columns to float (handles string values properly)
numeric_columns = ['Diam base', 'Diam 130cm', 'Volume dendrometrico', 'Volume cormometrico', 'Altezza indicativa']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing height values based on chosen method
if args.method == 'interpolate':
    df['Altezza indicativa'] = df.groupby('Genere')['Altezza indicativa'].transform(lambda x: x.interpolate(method=args.interpolation))
elif args.method == 'fit':
    df_filled = df.copy()
    for species in df['Genere'].unique():
        species_mask = df['Genere'] == species
        species_data = df[species_mask]

        # Get non-missing data for fitting
        fit_mask = species_data['Altezza indicativa'].notna() & species_data['Diam 130cm'].notna()
        fit_data = species_data[fit_mask]

        x_fit = fit_data['Diam 130cm'].values
        y_fit = fit_data['Altezza indicativa'].values
        params, r2 = fit_logarithmic_curve(x_fit, y_fit)
        if params is None:
            print(f"{species:15}: Failed to fit logarithmic curve")
            continue

        a, b = params
        print(f"{species:15}: y = {a:.4f}*ln(x) + {b:.4f}, R² = {r2:.4f} ({len(fit_data)} points)")

        # Fill missing values using the fitted curve
        missing_mask = species_mask & df['Altezza indicativa'].isna() & df['Diam 130cm'].notna()
        missing_diameters = df.loc[missing_mask, 'Diam 130cm'].values

        if len(missing_diameters) > 0:
            predicted_heights = a * np.log(missing_diameters) + b
            df_filled.loc[missing_mask, 'Altezza indicativa'] = predicted_heights

    df = df_filled

# Save the interpolated data
df.to_csv('alsometrie_interpolated.csv', index=False, float_format='%g')

# Create a figure with subplots for each desired parameter
fig, axes = plt.subplots(1, len(parameters), figsize=(6 * len(parameters), 6))

# Define colors for each species
species_colors = {
    'Abete bianco': 'blue',
    'Faggio': 'green',
    'Pino laricio': 'red',
    'Castagno': 'purple',
    'Douglasia': 'orange',
}

if len(parameters) == 1:
    axes = [axes]
for i, parameter in enumerate(parameters):
    for species in df['Genere'].unique():
        species_data = df[df['Genere'] == species].dropna(subset=[parameter])
        if len(species_data) > 0:
            # Sort by diameter to ensure proper x-axis ordering
            species_data = species_data.sort_values('Diam 130cm').dropna(subset=[parameter])
            species_data.plot.scatter(x='Diam 130cm', y=parameter,
                                      ax=axes[i], label=species, alpha=0.7,
                                      color=species_colors[species], s=30)
    axes[i].set_title(f'{parameter} vs Diametro 130cm', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Diametro 130cm (cm)')
    unit = 'm' if parameter == 'Altezza indicativa' else 'mc'
    axes[i].set_ylabel(f'{parameter} ({unit})')

    axes[i].locator_params(axis='x', nbins=8)
    axes[i].locator_params(axis='y', nbins=10)

    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.tight_layout()
plt.savefig('alsometrie.png', dpi=200, bbox_inches='tight')
plt.show()
