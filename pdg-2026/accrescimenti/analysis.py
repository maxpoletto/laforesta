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

def create_cd(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str) -> list:
    """
    Create a histogram for a specific region (parcel or compresa) showing tree distribution by diameter class
    with stacked bars for different species, scaled to estimated totals per hectare
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)
    
    if particella is not None:
        # Per-particella mode
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        filename = f"{compresa}_{region['sort_key']}_classi-diametriche.png"
        title = f'Distribuzione alberi per classe diametrica - {compresa} Particella {particella}'
        print_name = f"{compresa}-{particella}"
    else:
        # Per-compresa mode
        region_data = trees[trees['Compresa'] == compresa]
        filename = f"{compresa}_classi-diametriche.png"
        title = f'Distribuzione alberi per classe diametrica - {compresa}'
        print_name = compresa
    
    assert len(region_data) > 0, f"Nessun dato per {print_name}"
    
    filepath = os.path.join(output_dir, filename)
    print(f"Generazione istogramma per {print_name}...")

    counts = (region_data.groupby(['Classe diametrica', 'Genere']).size().unstack(fill_value=0)
              * SAMPLE_AREAS_PER_HA / region['sample_areas'])

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
    ax.set_title(title, fontweight='bold', pad=20)

    max_class = trees['Classe diametrica'].max()
    ax.set_xlim(-0.5, max_class + 0.5)
    ax.set_xticks(range(0, max_class + 1, 2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02),
              alignment='left')

    with italian_locale():
        stats_text = f"""
Area: {locale.format_string('%.2f', region["area_ha"])} ha
Alberi campionati: {len(region_data)}
N. aree saggio: {region["sample_areas"]}
Stima totale alberi: {region["estimated_total"]:n}
Stima alberi / ha: {round(region["estimated_total"]/region["area_ha"]):n}
Specie prevalente: {region_data["Genere"].mode().iloc[0]}
Classe diametrica media: {round(region_data["Classe diametrica"].mean()):n}""".strip()

    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#fbfbfb', alpha=1, linewidth=0.2))
    plt.tight_layout()

    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved histogram for {print_name} to {filepath}")
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
            filename = os.path.basename(filepath)
            f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{compresa} - Particella {particella}</div>
            <img src="{filename}" alt="Istogramma classi diametriche {compresa}-{particella}" class="histogram-image">
        </div>
''')

        f.write('''    </div>
</body>
</html>''')

def fit_logarithmic_curve(x, y) -> tuple[tuple[float, float], float, tuple[float, float]]:
    """Fit logarithmic curve to data using np.polyfit and return parameters and R²"""
    x_clean = x[x > 0]
    y_clean = y[x > 0]

    if len(x_clean) < 10:
        return None, None, None

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

    return (a, b), r2, (x_clean.min(), x_clean.max())

def create_height_interpolation_functions(alsometrie_file, method='fit', interpolation='quadratic'):
    """Create height interpolation functions from alsometrie data"""
    try:
        df = pd.read_csv(alsometrie_file)
    except FileNotFoundError:
        print(f"Warning: {alsometrie_file} not found, using original h(m) values")
        return {}

    # Convert numeric columns
    numeric_columns = ['Diam base', 'Diam 130cm', 'Volume dendrometrico', 'Volume cormometrico', 'Altezza indicativa']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    height_functions = {}

    if method == 'none':
        return None
    elif method == 'interpolate':
        df['Altezza indicativa'] = df.groupby('Genere')['Altezza indicativa'].transform(lambda x: x.interpolate(method=interpolation))
        for species in df['Genere'].unique():
            species_data = df[df['Genere'] == species].dropna(subset=['Diam 130cm', 'Altezza indicativa'])
            if len(species_data) >= 2:
                # Sort by diameter
                species_data = species_data.sort_values('Diam 130cm')
                x_data = species_data['Diam 130cm'].values
                y_data = species_data['Altezza indicativa'].values

                # Create interpolation function using numpy interp
                height_functions[species] = lambda x, x_data=x_data, y_data=y_data: np.interp(x, x_data, y_data)

    elif method == 'fit':
        # Use fitted logarithmic curves like interpolate.py does
        for species in df['Genere'].unique():
            species_data = df[df['Genere'] == species]

            # Get non-missing data for fitting
            fit_data = species_data.dropna(subset=['Altezza indicativa', 'Diam 130cm'])

            if len(fit_data) < 2:
                continue

            x_fit = fit_data['Diam 130cm'].values
            y_fit = fit_data['Altezza indicativa'].values

            params, r2, _ = fit_logarithmic_curve(x_fit, y_fit)
            if params is not None:
                a, b = params
                height_functions[species] = lambda x, a=a, b=b: a * np.log(np.maximum(x, 0.1)) + b
                print(f"{species:15}: y = {a:.4f}*ln(x) + {b:.4f}, R² = {r2:.4f} ({len(fit_data)} points)")

    return height_functions

def create_ci(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str,
              height_functions: dict = None) -> list:
    """
    Create a scatter plot for a specific region (parcel or compresa) showing height vs diameter class relationship
    with logarithmic fit for each species
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)
    
    if particella is not None:
        # Per-particella mode
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        filename = f"{compresa}_{region['sort_key']}_curve-ipsometriche.png"
        title = f'Curve ipsometriche - {compresa} Particella {particella}'
        print_name = f"{compresa}-{particella}"
    else:
        # Per-compresa mode
        region_data = trees[trees['Compresa'] == compresa]
        filename = f"{compresa}_curve-ipsometriche.png"
        title = f'Curve ipsometriche - {compresa}'
        print_name = compresa
    
    assert len(region_data) > 0, f"Nessun dato per {print_name}"
    
    filepath = os.path.join(output_dir, filename)
    print(f"Generazione grafico ipsometrico per {print_name}...")

    fig, ax = plt.subplots(figsize=(4, 3))

    species_list = sorted(region_data['Genere'].unique())
    polynomial_info = []

    for species in species_list:
        species_data = region_data[region_data['Genere'] == species]
        x = species_data['Classe diametrica'].values

        # Use interpolated heights if available, otherwise use original h(m)
        if height_functions and species in height_functions:
            # Get D(cm) values and apply interpolation function
            d_cm = species_data['D(cm)'].values
            y = np.array([height_functions[species](d) for d in d_cm])
        else:
            y = species_data['h(m)'].values

        # Plot scatter points
        ax.scatter(x, y, color=color_map[species], label=species, alpha=0.7, s=20)

        params, r2, limits = fit_logarithmic_curve(x, y)
        if params is None:
            continue

        a, b = params
        x_min, x_max = limits
        x_smooth = np.linspace(x_min, x_max, 100)
        y_smooth = a * np.log(x_smooth) + b
        ax.plot(x_smooth, y_smooth, color=color_map[species], linestyle='--', alpha=0.8, linewidth=1.5)

        polynomial_info.append({
            'species': species,
            'coeffs': (a, b),
            'r_squared': r2,
            'n_points': len(species_data)
        })

    ax.set_xlabel('Classe diametrica', fontweight='bold')
    ax.set_ylabel('Altezza (m)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)

    max_class = trees['Classe diametrica'].max()
    ax.set_xlim(-0.5, max_class + 0.5)
    ax.set_xticks(range(0, max_class + 1, 2))

    y_max = region_data['h(m)'].max().astype(int)
    ax.set_ylim(0, (y_max + 6)//5*5)
    td = min(ax.get_ylim()[1] // 5, 4)
    y_ticks = np.arange(0, ax.get_ylim()[1] + 1, td)
    ax.set_yticks(y_ticks)

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title='Specie', bbox_to_anchor=(1.01, 1.02), alignment='left')

    with italian_locale():
        stats_text = f"""
Area: {locale.format_string('%.2f', region["area_ha"])} ha
Alberi campionati: {len(region_data)}
N. aree saggio: {region["sample_areas"]}
Stima totale alberi: {region["estimated_total"]:n}
Stima alberi / ha: {round(region["estimated_total"]/region["area_ha"]):n}
Specie prevalente: {region_data["Genere"].mode().iloc[0]}
Altezza media: {locale.format_string('%.1f', region_data["h(m)"].mean())} m""".strip()

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
    print(f"Saved scatter plot for {print_name} to {filepath}")
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
    parser = argparse.ArgumentParser(description='Analisi accrescimenti')
    parser.add_argument('--solo-classi-diametriche', action='store_true',
                        help='Genera solo istogrammi classi diametriche')
    parser.add_argument('--solo-curve-ipsometriche', action='store_true',
                        help='Genera solo curve ipsometriche')
    parser.add_argument('--height-method', type=str, choices=['fit', 'interpolate', 'none'], default='none',
                        help='Method to compute height values: fit (logarithmic curve) or interpolate or none')
    parser.add_argument('--interpolation', type=str, default='quadratic',
                        help='Interpolation method when using interpolate method')
    parser.add_argument('--alsometrie-file', type=str, default='alsometrie.csv',
                        help='Path to alsometrie CSV file for height interpolation')
    parser.add_argument('--trees-file', type=str, default='alberi.csv',
                        help='Path to trees CSV file')
    parser.add_argument('--particelle-file', type=str, default='particelle.csv',
                        help='Path to particelle CSV file')
    parser.add_argument('--per-particella', action='store_true',
                        help='Generate graphs per Compresa-Particella pair (default: per Compresa only)')

    args = parser.parse_args()

    classi_diametriche = True
    curve_ipsometriche = True

    if args.solo_classi_diametriche:
        curve_ipsometriche = False
    elif args.solo_curve_ipsometriche:
        classi_diametriche = False

    alberi = pd.read_csv(args.trees_file)
    particelle = pd.read_csv(args.particelle_file)

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

    # Create region data (either per particella or per compresa)
    if args.per_particella:
        regions_df = parcels_df
        print(f"Modalità per particella: {len(regions_df)} particelle")
    else:
        # Aggregate by compresa
        compresa_stats = (alberi_fustaia.groupby(['Compresa'])
                         .agg(sampled_trees=('Area saggio', 'size'),
                              sample_areas=('Area saggio', 'nunique'))
                         .reset_index())
        
        compresa_areas = (particelle.groupby('Compresa')['Area (ha)'].sum().reset_index()
                         .rename(columns={'Area (ha)': 'area_ha'}))
        
        regions_df = (compresa_stats.merge(compresa_areas, on='Compresa', how='left'))
        regions_df['estimated_total'] = round((regions_df['sampled_trees'] / regions_df['sample_areas'])
                                              * SAMPLE_AREAS_PER_HA
                                              * regions_df['area_ha']).astype(int)
        print(f"Modalità per compresa: {len(regions_df)} comprese")

    # Create height interpolation functions if processing curve ipsometriche
    height_functions = {}
    if curve_ipsometriche:
        print(f"Creating height interpolation functions using method '{args.height_method}'...")
        height_functions = create_height_interpolation_functions(
            alsometrie_file=args.alsometrie_file,
            method=args.height_method,
            interpolation=args.interpolation
        )

    if classi_diametriche:
        output_dir = 'classi-diametriche'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = []
        for _, row in regions_df.iterrows():
            files.append(create_cd(alberi_fustaia, row, color_map, output_dir))

        generate_html_index_cd(files, output_dir)
        print(f"Istogrammi classi diametriche salvati in '{output_dir}'")

    if curve_ipsometriche:
        output_dir = 'curve-ipsometriche'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = []
        for _, row in regions_df.iterrows():
            files.append(create_ci(alberi_fustaia, row, color_map, output_dir, height_functions))

        generate_html_index_ci(files, output_dir)
        print(f"Curve ipsometriche salvate in '{output_dir}'")

    with italian_locale():
        print(f"\nAnalisi completata.")
        print("\nRiepilogo:")
        
        if args.per_particella:
            for _, row in regions_df.sort_values(['sort_key']).iterrows():
                trees_per_ha = row['estimated_total'] / row['area_ha']
                print(f"  {row['Compresa']} - Particella {row['Particella']}: "
                    f"{row['sampled_trees']} alberi campionati, "
                    f"{row['sample_areas']} aree saggio, "
                    f"{locale.format_string('%.2f', row['area_ha'])} ha → "
                    f"Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")
        else:
            for _, row in regions_df.iterrows():
                trees_per_ha = row['estimated_total'] / row['area_ha']
                print(f"  {row['Compresa']}: "
                    f"{row['sampled_trees']} alberi campionati, "
                    f"{row['sample_areas']} aree saggio, "
                    f"{locale.format_string('%.2f', row['area_ha'])} ha → "
                    f"Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")

if __name__ == "__main__":
    main()