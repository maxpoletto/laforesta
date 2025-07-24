#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--volume-dendrometrico', action='store_true', help='Plot volume dendrometrico')
parser.add_argument('--volume-cormometrico', action='store_true', help='Plot volume cormometrico')
parser.add_argument('--interpolation', type=str, help='Interpolation method', default='quadratic')
parser.add_argument('-i', '-input', type=str, help='Input CSV file', default='alsometrie.csv')
args = parser.parse_args()

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

# Interpolate missing height values for each species separately
df['Altezza indicativa'] = df.groupby('Genere')['Altezza indicativa'].transform(lambda x: x.interpolate(method=args.interpolation))

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
