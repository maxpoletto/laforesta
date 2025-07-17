#!/usr/bin/env python3
"""
Forest Parcel Analysis: Generate histograms of tree distribution by diameter class for each parcel
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import locale
from contextlib import contextmanager

# Set up matplotlib for high-quality plots suitable for reports
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
plt.rcParams['figure.figsize'] = (4, 2.5)

SAMPLE_AREAS_PER_HA = 8

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

def create_parcel_histogram(trees: pd.DataFrame, parcel: pd.Series, color_map: dict, output_dir: str) -> None:
    """
    Create a histogram for a specific parcel showing tree distribution by diameter class
    with stacked bars for different species, scaled to estimated totals per hectare
    """
    compresa = parcel['Compresa']
    particella = parcel['Particella']
    parcel_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
    assert len(parcel_data) > 0, f"Nessun dato per {compresa}-{particella}"

    padded_particella = str(particella).zfill(3)
    filename = f"{compresa}_{padded_particella}_histogram.png"
    filepath = os.path.join(output_dir, filename)
    print(f"Generazione istogramma per {compresa}-{particella}...")

    counts = (parcel_data.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
              * SAMPLE_AREAS_PER_HA / parcel['sample_areas'])

    fig, ax = plt.subplots()
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
    ax.set_title(f'Distribuzione alberi per classe diametrica - {compresa} Particella {particella}',
                fontweight='bold', pad=20)

    max_class = trees['Classe diametrica'].max()
    ax.set_xlim(-0.5, max_class + 0.5)
    ax.set_xticks(range(0, max_class + 1, 2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02),
              alignment='left')

    with italian_locale():
        stats_text = f"""
Area: {locale.format_string('%.2f', parcel["area_ha"])} ha
Alberi campionati: {len(parcel_data)}
N. aree saggio: {parcel["sample_areas"]}
Stima totale alberi: {parcel["estimated_total"]:n}
Stima alberi / ha: {round(parcel["estimated_total"]/parcel["area_ha"]):n}
Specie prevalente: {parcel_data["Genere"].mode().iloc[0]}
Classe diametrica media: {round(parcel_data["Classe diametrica"].mean()):n}""".strip()

    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))
    plt.tight_layout()

    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved histogram for {compresa}-{particella} to {filepath}")
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
    color_map = dict(zip(all_species, colors))

    parcel_stats = (alberi_fustaia.groupby(['Compresa', 'Particella'])
                   .agg(sampled_trees=('Area saggio', 'size'),
                        sample_areas=('Area saggio', 'nunique'))
                   .reset_index())

    parcels_df = (parcel_stats.merge(particelle[['Compresa', 'Particella', 'Area (ha)']],
                                     on=['Compresa', 'Particella'], how='left')
                  .rename(columns={'Area (ha)': 'area_ha'}))

    parcels_df['estimated_total'] = round((parcels_df['sampled_trees'] / parcels_df['sample_areas'])
                                          * SAMPLE_AREAS_PER_HA
                                          * parcels_df['area_ha']).astype(int)
    print(f"Trovate {len(parcels_df)} particelle")

    output_dir = 'histograms'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for _, row in parcels_df.iterrows():
        create_parcel_histogram(alberi_fustaia, row, color_map, output_dir)

    with italian_locale():
        print(f"\nAnalisi completata. Tutti i grafici sono stati salvati in '{output_dir}'")
        print("\nRiepilogo:")
        parcel_summary = parcels_df.sort_values(['Compresa', 'Particella'])
        for _, row in parcel_summary.iterrows():
            trees_per_ha = row['estimated_total'] / row['area_ha']
            print(f"  {row['Compresa']} - Particella {row['Particella']}: "
                f"{row['sampled_trees']} alberi campionati, "
                f"{row['sample_areas']} aree saggio, "
                f"{locale.format_string('%.2f', row['area_ha'])} ha → "
                f"Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")

if __name__ == "__main__":
    main()