#!/usr/bin/env python3
"""
Generate golden reference CSVs for regression tests.

Run via: make regenerate-golden
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pdg.computation import COL_PARTICELLA, COL_COMPRESA, COL_GENERE, COL_CD_CM
from pdg.io import file_cache, load_csv, load_trees
from pdg.simulation import growth_per_group
from pdg.core import (
    region_cache, parcel_data,
    calculate_volume_table, calculate_harvest_table,
    calculate_diameter_class_data,
)
from pdg.harvest_rules import max_harvest

TEST_DIR = Path(__file__).parent / "data"


def main():
    file_cache.clear()
    region_cache.clear()
    trees_df = load_trees(['regression-alberi.csv'], TEST_DIR)
    particelle_df = load_csv('regression-particelle.csv', TEST_DIR)
    particelle_df[COL_PARTICELLA] = particelle_df[COL_PARTICELLA].astype(str)

    def pd_filtered(regions=None, parcels=None):
        region_cache.clear()
        return parcel_data(
            ['regression-alberi.csv'], trees_df, particelle_df,
            regions=regions or [], parcels=parcels or [], species=[])

    generated = []

    def save(df, name):
        fname = TEST_DIR / f'golden-{name}.csv'
        df.to_csv(fname, index=False, float_format='%.6f')
        generated.append(name)
        print(f'  {name}: {len(df)} rows')

    def save_indexed(df, name):
        fname = TEST_DIR / f'golden-{name}.csv'
        df.to_csv(fname, float_format='%.6f')
        generated.append(name)
        print(f'  {name}: {len(df)} rows')

    data_all = pd_filtered()
    data_serra = pd_filtered(regions=['Serra'])
    data_fab1 = pd_filtered(regions=['Fabrizia'], parcels=['1'])
    data_cap3 = pd_filtered(regions=['Capistrano'], parcels=['3'])

    # @@volumi — matches sec-volumi.tex and particella.tex invocations
    print('@@volumi:')
    save(calculate_volume_table(data_all,
        group_cols=[COL_COMPRESA],
        calc_margin=True, calc_total=True), 'tsv-per_compresa')
    save(calculate_volume_table(data_serra,
        group_cols=[COL_PARTICELLA],
        calc_margin=True, calc_total=True), 'tsv-serra-per_particella')
    save(calculate_volume_table(data_fab1,
        group_cols=[COL_GENERE],
        calc_margin=True, calc_total=True), 'tsv-fab1-per_genere')

    # @@prelievi — matches sec-ripresa.tex, particella.tex, relazione.tex
    print('@@prelievi:')
    save(calculate_harvest_table(data_all, max_harvest,
        group_cols=[COL_COMPRESA]), 'tpt-per_compresa')
    save(calculate_harvest_table(data_serra, max_harvest,
        group_cols=[COL_PARTICELLA]), 'tpt-serra-per_particella')
    save(calculate_harvest_table(data_cap3, max_harvest,
        group_cols=[COL_GENERE]), 'tpt-cap3-per_genere')
    save(calculate_harvest_table(data_serra, max_harvest,
        group_cols=[COL_PARTICELLA, COL_GENERE]),
        'tpt-serra-per_particella_genere')

    # @@tabella_incremento_percentuale — matches particella.tex
    print('@@tabella_incremento_percentuale:')
    save(growth_per_group(data_fab1.trees,
        group_cols=[COL_GENERE, COL_CD_CM],
        stime_totali=True), 'tip-fab1')
    save(growth_per_group(data_cap3.trees,
        group_cols=[COL_GENERE, COL_CD_CM],
        stime_totali=True), 'tip-cap3')

    # @@tabella_classi_diametriche — matches particella.tex (coarse bins, 4 metrics)
    print('@@tabella_classi_diametriche:')
    for metrica in ['alberi_ha', 'volume_ha', 'G_ha', 'altezza']:
        save_indexed(calculate_diameter_class_data(data_fab1,
            metrica=metrica, stime_totali=True, fine=False),
            f'tcd-fab1-{metrica}')
    # Fine volume_tot for cross-check
    save_indexed(calculate_diameter_class_data(data_all,
        metrica='volume_tot', stime_totali=True, fine=True),
        'tcd-all-volume_tot-fine')

    print(f'\nRegenerated {len(generated)} golden files.')


if __name__ == '__main__':
    main()
