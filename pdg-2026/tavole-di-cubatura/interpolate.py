#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the new alsometrie.csv file with species data
df = pd.read_csv('alsometrie.csv')

# Convert numeric columns to float (handles string values properly)
numeric_columns = ['Diam base', 'Diam 130cm', 'Volume dendrometrico', 'Volume cormometrico', 'Altezza indicativa']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Interpolate missing height values for each species separately
df['Altezza indicativa'] = df.groupby('Genere')['Altezza indicativa'].transform(lambda x: x.interpolate(method='cubic'))

# Save the interpolated data
df.to_csv('alsometrie_interpolated.csv', index=False, float_format='%g')

print("Unique species found:", df['Genere'].unique())
print("Data shape:", df.shape)

# Create a figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define colors for each species
species_colors = {
    'Abete bianco': 'blue',
    'Faggio': 'green', 
    'Pino laricio': 'red'
}

# Plot 1: Height vs Base Diameter
for i, parameter in enumerate(['Altezza indicativa', 'Volume dendrometrico', 'Volume cormometrico']):
    for species in df['Genere'].unique():
        species_data = df[df['Genere'] == species].dropna(subset=[parameter])
        if len(species_data) > 0:
            # Sort by diameter to ensure proper x-axis ordering
            species_data = species_data.sort_values('Diam base')
            species_data.plot.scatter(x='Diam base', y=parameter, 
                                      ax=axes[i], label=species, alpha=0.7, 
                                      color=species_colors[species], s=30)
    axes[i].set_title(f'{parameter} vs Diametro base', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Diametro base (cm)')
    unit = 'm' if parameter == 'Altezza indicativa' else 'mc'
    axes[i].set_ylabel(f'{parameter} ({unit})')

    axes[i].locator_params(axis='x', nbins=8)
    axes[i].locator_params(axis='y', nbins=10)
    
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.tight_layout()
plt.savefig('alsometrie.png', dpi=200, bbox_inches='tight')
plt.show()
