#!/usr/bin/env python3
"""
Forest Parcel Analysis: Generate histograms of tree distribution by diameter class for each parcel
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up matplotlib for high-quality plots suitable for reports
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

def create_parcel_histogram(data: pd.DataFrame, compresa: str, particella: str, global_color_map: dict, save_path: str = None) -> None:
    """
    Create a histogram for a specific parcel showing tree distribution by diameter class
    with stacked bars for different species
    """
    parcel_data = data[(data['Compresa'] == compresa) & (data['Particella'] == particella)]
    if len(parcel_data) == 0:
        print(f"Nessun dato per {compresa}-{particella}")
        return
    
    counts = parcel_data.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))    
    species_list = counts.columns.tolist()
    
    bottom = np.zeros(len(counts.index))
    for species in species_list:
        values = counts[species].values
        bars = ax.bar(counts.index, values, bottom=bottom, 
                     label=species, color=global_color_map[species], 
                     alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += values

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Numero di alberi', fontweight='bold')
    ax.set_title(f'Distribuzione alberi per classe diametrica - {compresa} Particella {particella}', 
                fontweight='bold', pad=20)
    
    max_diameter = max(40, data['Classe diametrica'].max())
    ax.set_xlim(-0.5, max_diameter + 0.5)
    ax.set_xticks(range(0, max_diameter + 1, 2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.05, 1), loc='upper left')

    stats_text = f'Totale alberi: {len(parcel_data)}\n'
    stats_text += f'Specie prevalente: {parcel_data["Genere"].mode().iloc[0]}\n'
    stats_text += f'Classe diametrica media: {parcel_data["Classe diametrica"].mean():.1f}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram for {compresa}-{particella} to {save_path}")
    plt.close(fig)

def main():
    """
    Main analysis function
    """
    alberi = pd.read_csv('alberi.csv')
    particelle = pd.read_csv('particelle.csv')
    
    alberi_fustaia = alberi[alberi['Fustaia'] == True].copy()
    print(f"Dati filtrati: {len(alberi_fustaia)} campioni di alberti a fustaia (su {len(alberi)} totali)")
    
    # Create consistent color mapping for all species
    all_species = sorted(alberi_fustaia['Genere'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_species)))
    global_color_map = dict(zip(all_species, colors))
    
    # Get unique parcel combinations
    parcels = alberi_fustaia.groupby(['Compresa', 'Particella']).size().reset_index(name='tree_count')
    print(f"Trovate {len(parcels)} particelle")
    output_dir = 'histograms'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for _, row in parcels.iterrows():
        compresa = row['Compresa']
        particella = row['Particella']

        filename = f"{compresa}_{particella}_histogram.png"
        filepath = os.path.join(output_dir, filename)
        print(f"Generazione istogramma per {compresa}-{particella}...")
        create_parcel_histogram(alberi_fustaia, compresa, particella, global_color_map, filepath)

    print(f"\nAnalisi completata. Tutti i grafici sono stati salvati in '{output_dir}'")
    print("\nRiepilogo:")
    parcel_summary = parcels.sort_values(['Compresa', 'Particella'])
    for _, row in parcel_summary.iterrows():
        print(f"  {row['Compresa']} - Particella {row['Particella']}: {row['tree_count']} alberi")

if __name__ == "__main__":
    main() 