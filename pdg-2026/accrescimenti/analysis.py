#!/usr/bin/env python3
"""
Forest Parcel Analysis: Generate histograms of tree distribution by diameter class
and scatter plots with height-diameter relationships for each parcel
"""

import argparse
from contextlib import contextmanager
import locale
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    if particella is None:
        region_data = trees[trees['Compresa'] == compresa]
        title = f'Distribuzione alberi per classe diametrica - {compresa}'
        print_name = compresa
    else:
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        title = f'Distribuzione alberi per classe diametrica - {compresa} Particella {particella}'
        print_name = f"{compresa}-{particella}"
    filename = f"{region['sort_key']}_classi-diametriche.png"

    assert len(region_data) > 0, f"Nessun dato per {print_name}"
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
    ax.set_ylim(0, counts.sum(axis=1).max() * 1.1)
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

    filepath = Path(output_dir) / filename
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved histogram for {print_name} to {filepath}")
    plt.close(fig)
    return [compresa, particella, filepath]

def generate_html_index_cd(files: list, output_dir: str) -> None:
    """
    Generate an HTML index file for the generated histograms
    """
    files_sorted = sorted(files, key=lambda x: x[2])

    with open(Path(output_dir) / 'index.html', 'w', encoding='utf-8') as f:
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
            title = f"{compresa} - Particella {particella}" if particella else compresa
            f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{title}</div>
            <img src="{filepath.name}" alt="Istogramma classi diametriche {title}" class="histogram-image">
        </div>
''')

        f.write('''    </div>
</body>
</html>''')

def fit_logarithmic_curve(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], float, tuple[float, float]]:
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

def height_interpolation_functions(alsometrie_file: str, height_method: str, interpolation_method: str,
                                   alsometry_calc: bool, alsometry_file: str) -> dict:
    """Create height interpolation functions from alsometrie data"""
    if height_method == 'originali':
        return None

    try:
        df = pd.read_csv(alsometrie_file)
    except FileNotFoundError:
        print(f"Warning: {alsometrie_file} not found, using original h(m) values")
        return None

    # Convert numeric columns
    numeric_cols = ['Diam base', 'Diam 130cm', 'Volume dendrometrico', 'Volume cormometrico', 'Altezza indicativa']
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne {missing_cols} non trovate in {alsometrie_file}")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    hfuncs = {}

    if height_method == 'interpolazione':
        df['Altezza indicativa'] = (df.groupby('Genere')['Altezza indicativa']
                                    .transform(lambda x: x.interpolate(method=interpolation_method)))
        for species in df['Genere'].unique():
            data = (df[df['Genere'] == species].dropna(subset=['Diam 130cm', 'Altezza indicativa'])
                    .sort_values('Diam 130cm'))
            hfuncs[species] = (lambda x,
                                xvec=data['Diam 130cm'].values, yvec=data['Altezza indicativa'].values:
                                np.interp(x, xvec, yvec)) # type: ignore
    elif height_method == 'regressione':
        for species in df['Genere'].unique():
            data = (df[df['Genere'] == species]
                    .dropna(subset=['Altezza indicativa', 'Diam 130cm']))
            params, r2, _ = fit_logarithmic_curve(data['Diam 130cm'].values,
                                                  data['Altezza indicativa'].values)
            if params is not None:
                a, b = params
                hfuncs[species] = lambda x, a=a, b=b: a * np.log(np.maximum(x, 0.1)) + b
                print(f"{species:15}: y = {a:.4f}*ln(x) + {b:.4f}, R² = {r2:.4f} ({len(data)} punti)")
    else:
        raise ValueError(f"Metodo di interpolazione altezze non valido: {height_method}")

    if alsometry_calc:
        df['Diam base'] = df['Diam base'].astype('Int64')
        df['Altezza indicativa'] = df.apply(
            lambda row: hfuncs[row['Genere']](row['Diam 130cm']) if hfuncs and row['Genere'] in hfuncs else row['Altezza indicativa'],
            axis=1)
        df.to_csv(alsometry_file, index=False, float_format="%.3f")
        print(f"File '{alsometry_file}' salvato")

    return hfuncs

def create_ci(trees: pd.DataFrame, region: pd.Series, color_map: dict, output_dir: str,
              hfuncs: dict, omit_unknown: bool) -> list:
    """
    Create a scatter plot for a specific region (parcel or compresa) showing height vs diameter class relationship
    with logarithmic fit for each species
    """
    compresa = region['Compresa']
    particella = region.get('Particella', None)

    if particella is None:
        region_data = trees[trees['Compresa'] == compresa]
        title = f'Curve ipsometriche - {compresa}'
        print_name = compresa
    else:
        region_data = trees[(trees['Compresa'] == compresa) & (trees['Particella'] == particella)]
        title = f'Curve ipsometriche - {compresa} Particella {particella}'
        print_name = f"{compresa}-{particella}"
    filename = f"{region['sort_key']}_curve-ipsometriche.png"

    assert len(region_data) > 0, f"Nessun dato per {print_name}"
    print(f"Generazione grafico ipsometrico per {print_name}...")

    fig, ax = plt.subplots(figsize=(4, 3))

    species_list = sorted(region_data['Genere'].unique())
    polynomial_info = []

    ymax = 0
    for species in species_list:
        species_data = region_data[region_data['Genere'] == species]
        x = species_data['Classe diametrica'].values

        # Skip species not present in alsometrie if requested (generate only "clean" curves)
        if hfuncs and species not in hfuncs and omit_unknown:
            print(f"Genere {species} non presente nel file alsometrie, omesso dalle curve ipsometriche")
            continue

        # Use interpolated heights if available, otherwise use original h(m)
        if hfuncs and species in hfuncs:
            # Get D(cm) values and apply interpolation function
            d_cm = species_data['D(cm)'].values
            y = np.array([hfuncs[species](d) for d in d_cm])
        else:
            y = species_data['h(m)'].values
        ymax = max(ymax, y.max())

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

    ax.set_ylim(0, (ymax + 6)//5*5)
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

    filepath = Path(output_dir) / filename
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved scatter plot for {print_name} to {filepath}")
    plt.close(fig)
    return [compresa, particella, filepath]

def generate_html_index_ci(files: list, output_dir: str) -> None:
    """
    Generate an HTML index file for the generated scatter plots
    """
    files_sorted = sorted(files, key=lambda x: x[2])

    with open(Path(output_dir) / 'index.html', 'w', encoding='utf-8') as f:
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
            title = f"{compresa} - Particella {particella}" if particella else compresa
            f.write(f'''        <div class="histogram-item">
            <div class="histogram-title">{title}</div>
            <img src="{filepath.name}" alt="Curva ipsometrica {title}" class="histogram-image">
        </div>
''')

        f.write('''    </div>
</body>
</html>''')

def regions_dataframe(alberi_fustaia: pd.DataFrame, particelle: pd.DataFrame, per_particella: bool) -> pd.DataFrame:
    """Create a dataframe with region data"""
    groupby = ['Compresa', 'Particella'] if per_particella else ['Compresa']
    trees = (alberi_fustaia.groupby(groupby)
             .agg(sampled_trees=('Area saggio', 'size'),
                  sample_areas=('Area saggio', 'nunique'))
             .reset_index())
    areas = particelle.groupby(groupby)['Area (ha)'].sum().reset_index()
    df = (trees.merge(areas, on=groupby, how='left').rename(columns={'Area (ha)': 'area_ha'}))
    df['estimated_total'] = round((df['sampled_trees'] / df['sample_areas'])
                                  * SAMPLE_AREAS_PER_HA
                                  * df['area_ha']).astype(int)

    if per_particella:
        df['sort_key'] = df['Particella'].apply(
            lambda x: (f"{x}=" if str(x)[-1].isdigit() else str(x)).zfill(3))
        df['sort_key'] = df['Compresa'] + '-' + df['sort_key']
        print(f"Modalità per particella: {len(df)} particelle")
    else:
        df['sort_key'] = df['Compresa']
        print(f"Modalità per compresa: {len(df)} comprese")

    return df

def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(description='Analisi accrescimenti', add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Mostra questo messaggio e esci')
    g1 = parser.add_argument_group('Tipo e granularità dei risultati')
    g1.add_argument('--genera-classi-diametriche', action='store_true',
                    help='Genera istogrammi classi diametriche')
    g1.add_argument('--genera-curve-ipsometriche', action='store_true',
                    help='Genera curve ipsometriche')
    g1.add_argument('--genera-alberi-altezze-calcolate', action='store_true',
                    help='Genera tabella alberi campionati con altezze calcolate in base ai dati alsometrici')
    g1.add_argument('--genera-alsometrie-calcolate', action='store_true',
                    help='Genera file con le altezze calcolate dalle curve ipsometriche')
    g1.add_argument('--per-particella', action='store_true',
                    help='Genera risultati per ogni coppia (compresa, particella) (default: solo per compresa)')
    g2 = parser.add_argument_group('Altezze per curve ipsometriche')
    g2.add_argument('--metodo-altezze', type=str, choices=['regressione', 'interpolazione', 'originali'], default='originali',
                    help='Metodo per calcolare le altezze: regressione (logaritmica) o interpolazione o usare i valori originali')
    g2.add_argument('--interpolazione', type=str, default='quadratic',
                    help='Metodo di interpolazione quando si usa il metodo di interpolazione ("cubic", "quadratic", "linear")')
    g2.add_argument('--ometti-generi-sconosciuti', action='store_true',
                    help='Omette generi non presenti nel file alsometrie dalle curve ipsometriche')
    g3 = parser.add_argument_group('Nomi dei file')
    g3.add_argument('--file-alsometrie', type=str, default='alsometrie.csv',
                    help='File con le altezze da tabelle alsometriche (input)')
    g3.add_argument('--file-alberi', type=str, default='alberi.csv',
                    help='File con i dati degli alberi campionati (input)')
    g3.add_argument('--file-particelle', type=str, default='particelle.csv',
                    help='File con i dati delle particelle (input)')
    g3.add_argument('--file-alberi-calcolati', type=str, default='alberi-calcolati.csv',
                    help='File con i dati degli alberi campionati con altezze calcolate (output)')
    g3.add_argument('--file-alsometrie-calcolate', type=str, default='alsometrie-calcolate.csv',
                    help='File con dati alsometrici calcolati (output)')
    g3.add_argument('--prefisso-output', type=str, default='',
                    help='Prefisso per i file di output')

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        return

    alberi = pd.read_csv(args.file_alberi)
    particelle = pd.read_csv(args.file_particelle)

    alberi_fustaia = alberi[alberi['Fustaia'] == True].copy()
    print(f"Dati filtrati: {len(alberi_fustaia)} campioni di alberi a fustaia (su {len(alberi)} totali)")

    # Create consistent color mapping for all species
    all_species = sorted(alberi_fustaia['Genere'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_species)))
    color_map = dict(zip(all_species, colors))

    # Create region data (either per particella or per compresa)
    regions = regions_dataframe(alberi_fustaia, particelle, args.per_particella)

    if args.genera_classi_diametriche:
        output_dir = Path(args.prefisso_output) / 'classi-diametriche'
        os.makedirs(output_dir, exist_ok=True)

        files = []
        for _, row in regions.sort_values(['sort_key']).iterrows():
            files.append(create_cd(trees=alberi_fustaia, region=row,
                                   color_map=color_map, output_dir=output_dir))

        generate_html_index_cd(files, output_dir)
        print(f"Istogrammi classi diametriche salvati in '{output_dir}'")

    hfuncs = {}
    if (args.genera_curve_ipsometriche
        or args.genera_alberi_altezze_calcolate
        or args.genera_alsometrie_calcolate):
        print(f"Creazione funzioni di interpolazione altezze usando metodo '{args.metodo_altezze}'...")
        hfuncs = height_interpolation_functions(
            alsometrie_file=args.file_alsometrie,
            height_method=args.metodo_altezze,
            interpolation_method=args.interpolazione,
            alsometry_calc=args.genera_alsometrie_calcolate,
            alsometry_file=args.file_alsometrie_calcolate,
        )

    if args.genera_curve_ipsometriche:
        output_dir = Path(args.prefisso_output) / 'curve-ipsometriche'
        os.makedirs(output_dir, exist_ok=True)

        files = []
        for _, row in regions.sort_values(['sort_key']).iterrows():
            files.append(create_ci(trees=alberi_fustaia, region=row,
                                   color_map=color_map, output_dir=output_dir,
                                   hfuncs=hfuncs, omit_unknown=args.ometti_generi_sconosciuti))

        generate_html_index_ci(files, output_dir)
        print(f"Curve ipsometriche salvate in '{output_dir}'")

    if args.genera_alberi_altezze_calcolate:
        alberi_calcolati = alberi.copy()
        alberi_calcolati['h(m)'] = alberi_calcolati.apply(
            lambda row: hfuncs[row['Genere']](row['D(cm)']) if (hfuncs and row['Genere'] in hfuncs) else row['h(m)'],
            axis=1)
        alberi_calcolati.to_csv(args.file_alberi_calcolati, index=False, float_format="%.3f")
        print(f"File '{args.file_alberi_calcolati}' salvato")

    with italian_locale():
        print(f"\nAnalisi completata.")
        print("\nRiepilogo:")

        for _, row in regions.sort_values(['sort_key']).iterrows():
            trees_per_ha = row['estimated_total'] / row['area_ha']
            label = f"{row['Compresa']} - Particella {row['Particella']}" if args.per_particella else row['Compresa']
            print(f"  {label}: "
                f"{row['sampled_trees']} alberi campionati, "
                f"{row['sample_areas']} aree saggio, "
                f"{locale.format_string('%.2f', row['area_ha'])} ha → "
                f"Stima totale: {row['estimated_total']:n} alberi ({round(trees_per_ha):n} alberi/ha)")

if __name__ == "__main__":
    main()