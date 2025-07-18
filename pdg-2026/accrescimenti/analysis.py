#!/usr/bin/env python3
"""
Forest Parcel Analysis: Generate histograms of tree distribution by diameter class
and scatter plots with height-diameter relationships for each parcel
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import locale
import argparse
from contextlib import contextmanager
from scipy import stats

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

STYLE = """
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

def create_parcel_cd(trees: pd.DataFrame, parcel: pd.Series, color_map: dict, output_dir: str) -> list:
    """
    Create a histogram for a specific parcel showing tree distribution by diameter class
    with stacked bars for different species, scaled to estimated totals per hectare
    """
    compresa = parcel['Compresa']
    particella = parcel['Particella']
    parcel_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
    assert len(parcel_data) > 0, f"Nessun dato per {compresa}-{particella}"

    filename = f"{compresa}_{parcel['sort_key']}_classi-diametriche.png"
    filepath = os.path.join(output_dir, filename)
    print(f"Generazione istogramma per {compresa}-{particella}...")

    counts = (parcel_data.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
              * SAMPLE_AREAS_PER_HA / parcel['sample_areas'])

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
    return [compresa, particella, filepath]

def generate_html_index_cd(files: list, output_dir: str) -> None:
    """
    Generate an HTML index file for the generated histograms
    """
    files_sorted = sorted(files, key=lambda x: x[2])

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(f'''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soc. Agr. La Foresta: distribuzione piante per classe diametrica</title>
    <style>{STYLE}    </style>
</head>
<body>
    <div class="container">
        <h1>Soc. Agr. La Foresta: distribuzione piante per classe diametrica</h1>
''')

        for compresa, particella, filepath in files_sorted:
            # Get just the filename for the src attribute
            filename = os.path.basename(filepath)
            f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{compresa} - Particella {particella}</div>
            <img src="{filename}" alt="Istogramma {compresa}-{particella}" class="histogram-image">
        </div>
''')

        f.write('''    </div>
</body>
</html>''')

def create_parcel_ci(trees: pd.DataFrame, parcel: pd.Series, color_map: dict, output_dir: str) -> list:
    """
    Create a scatter plot for a specific parcel showing height vs diameter class relationship
    with logarithmic fit for each species
    """
    compresa = parcel['Compresa']
    particella = parcel['Particella']
    parcel_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
    assert len(parcel_data) > 0, f"Nessun dato per {compresa}-{particella}"

    filename = f"{compresa}_{parcel['sort_key']}_curve-ipsometriche.png"
    filepath = os.path.join(output_dir, filename)
    print(f"Generazione grafico ipsometrico per {compresa}-{particella}...")

    fig, ax = plt.subplots(figsize=(4, 3))

    species_list = sorted(parcel_data['Genere'].unique())
    polynomial_info = []

    for species in species_list:
        species_data = parcel_data[parcel_data['Genere'] == species]
        x = species_data['Classe diametrica'].values
        y = species_data['h(m)'].values

        # Plot scatter points
        ax.scatter(x, y, color=color_map[species], label=species, alpha=0.7, s=20)

        if len(species_data) >= 10:
            # Fit logarithmic function: y = a*ln(x) + b
            # Using x values > 0 only (diameter classes start from 1)
            x_log = x[x > 0]
            y_log = y[x > 0]

            if len(x_log) >= 10:
                coeffs = np.polyfit(np.log(x_log), y_log, 1)  # Linear fit to log(x)
                a, b = coeffs

                y_pred = a * np.log(x_log) + b
                if np.var(y_log) > 0:  # Avoid division by zero
                    ss_res = np.sum((y_log - y_pred) ** 2)
                    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                else:
                    r_squared = 0.0  # No variance in y values

                x_smooth = np.linspace(x_log.min(), x_log.max(), 100)
                y_smooth = a * np.log(x_smooth) + b
                ax.plot(x_smooth, y_smooth, color=color_map[species], linestyle='--', alpha=0.8, linewidth=1.5)

                polynomial_info.append({
                    'species': species,
                    'coeffs': (a, b),
                    'r_squared': r_squared,
                    'n_points': len(species_data)
                })

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Altezza (m)', fontweight='bold')
    ax.set_title(f'Curve ipsometriche - {compresa} Particella {particella}',
                fontweight='bold', pad=20)

    max_class = trees['Classe diametrica'].max()
    ax.set_xlim(-0.5, max_class + 0.5)
    ax.set_xticks(range(0, max_class + 1, 2))

    y_max = parcel_data['h(m)'].max().astype(int)
    ax.set_ylim(0, (y_max + 6)//5*5)
    td = min(ax.get_ylim()[1] // 5, 4)
    y_ticks = np.arange(0, ax.get_ylim()[1] + 1, td)
    ax.set_yticks(y_ticks)

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

    with italian_locale():
        stats_text = f"""
Area: {locale.format_string('%.2f', parcel["area_ha"])} ha
Alberi campionati: {len(parcel_data)}
N. aree saggio: {parcel["sample_areas"]}
Stima totale alberi: {parcel["estimated_total"]:n}
Stima alberi / ha: {round(parcel["estimated_total"]/parcel["area_ha"]):n}
Specie prevalente: {parcel_data["Genere"].mode().iloc[0]}
Altezza media: {locale.format_string('%.1f', parcel_data["h(m)"].mean())} m""".strip()

    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))

    if polynomial_info:
        poly_text = ""
        for info in polynomial_info:
            a, b = info['coeffs']
            r2 = info['r_squared']
            n = info['n_points']
            poly_text += f"{info['species']}: y = {a:.4f}ln(x) + {b:.4f} (R² = {r2:.3f}, n = {n})\n"

        ax.text(0.01, -0.25, poly_text.strip(), transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=5, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, linewidth=0.2))

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved scatter plot for {compresa}-{particella} to {filepath}")
    plt.close(fig)
    return [compresa, particella, filepath]

def generate_html_index_ci(files: list, output_dir: str) -> None:
    """
    Generate an HTML index file for the generated scatter plots
    """
    files_sorted = sorted(files, key=lambda x: x[2])

    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(f'''<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Soc. Agr. La Foresta: curve ipsometriche</title>
    <style>{STYLE}    </style>
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
            # Get just the filename for the src attribute
            filename = os.path.basename(filepath)
            f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{compresa} - Particella {particella}</div>
            <img src="{filename}" alt="Curva ipsometrica {compresa}-{particella}" class="histogram-image">
        </div>
''')

        f.write('''    </div>
</body>
</html>''')

def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(description='Forest Parcel Analysis')
    parser.add_argument('--solo-classi-diametriche', action='store_true',
                        help='Genera solo istogrammi classi diametriche')
    parser.add_argument('--solo-curve-ipsometriche', action='store_true',
                        help='Genera solo curve ipsometriche')

    args = parser.parse_args()

    classi_diametriche = True
    curve_ipsometriche = True

    if args.solo_classi_diametriche:
        curve_ipsometriche = False
    elif args.solo_curve_ipsometriche:
        classi_diametriche = False

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

    parcels_df['sort_key'] = parcels_df['Particella'].apply(
        lambda x: (f"{x}=" if str(x)[-1].isdigit() else str(x)).zfill(3))
    print(f"Trovate {len(parcels_df)} particelle")

    if classi_diametriche:
        output_dir = 'classi-diametriche'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = []
        for _, row in parcels_df.iterrows():
            files.append(create_parcel_cd(alberi_fustaia, row, color_map, output_dir))

        generate_html_index_cd(files, output_dir)
        print(f"Istogrammi classi diametriche salvati in '{output_dir}'")

    if curve_ipsometriche:
        output_dir = 'curve-ipsometriche'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = []
        for _, row in parcels_df.iterrows():
            files.append(create_parcel_ci(alberi_fustaia, row, color_map, output_dir))

        generate_html_index_ci(files, output_dir)
        print(f"Curve ipsometriche salvate in '{output_dir}'")

    with italian_locale():
        print(f"\nAnalisi completata.")
        print("\nRiepilogo:")
        for _, row in parcels_df.sort_values(['sort_key']).iterrows():
            trees_per_ha = row['estimated_total'] / row['area_ha']
            print(f"  {row['Compresa']} - Particella {row['Particella']}: "
                f"{row['sampled_trees']} alberi campionati, "
                f"{row['sample_areas']} aree saggio, "
                f"{locale.format_string('%.2f', row['area_ha'])} ha → "
                f"Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")

if __name__ == "__main__":
    main()