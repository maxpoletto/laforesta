#!/usr/bin/env python3
"""
Generate golden reference CSVs for regression tests.

Run via: make regenerate-golden
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import acc
from harvest_rules import max_harvest

TEST_DIR = Path(__file__).parent / "data"


def main():
    acc.file_cache.clear()
    acc.region_cache.clear()
    trees_df = acc.load_trees(['regression-alberi.csv'], TEST_DIR)
    particelle_df = acc.load_csv('regression-particelle.csv', TEST_DIR)
    particelle_df[acc.COL_PARTICELLA] = particelle_df[acc.COL_PARTICELLA].astype(str)

    def pd_filtered(regions=None, parcels=None):
        acc.region_cache.clear()
        return acc.parcel_data(
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

    # @@tsv — matches sec-volumi.tex and particella.tex invocations
    print('@@tsv:')
    save(acc.calculate_tsv_table(data_all,
        group_cols=[acc.COL_COMPRESA],
        calc_margin=True, calc_total=True), 'tsv-per_compresa')
    save(acc.calculate_tsv_table(data_serra,
        group_cols=[acc.COL_PARTICELLA],
        calc_margin=True, calc_total=True), 'tsv-serra-per_particella')
    save(acc.calculate_tsv_table(data_fab1,
        group_cols=[acc.COL_GENERE],
        calc_margin=True, calc_total=True), 'tsv-fab1-per_genere')

    # @@tpt — matches sec-ripresa.tex, particella.tex, relazione.tex
    print('@@tpt:')
    save(acc.calculate_tpt_table(data_all, max_harvest,
        group_cols=[acc.COL_COMPRESA]), 'tpt-per_compresa')
    save(acc.calculate_tpt_table(data_serra, max_harvest,
        group_cols=[acc.COL_PARTICELLA]), 'tpt-serra-per_particella')
    save(acc.calculate_tpt_table(data_cap3, max_harvest,
        group_cols=[acc.COL_GENERE]), 'tpt-cap3-per_genere')
    save(acc.calculate_tpt_table(data_serra, max_harvest,
        group_cols=[acc.COL_PARTICELLA, acc.COL_GENERE]),
        'tpt-serra-per_particella_genere')

    # @@tip — matches particella.tex
    print('@@tip:')
    save(acc.calculate_growth_rates(data_fab1,
        group_cols=[acc.COL_GENERE, acc.COL_CD_CM],
        stime_totali=True), 'tip-fab1')
    save(acc.calculate_growth_rates(data_cap3,
        group_cols=[acc.COL_GENERE, acc.COL_CD_CM],
        stime_totali=True), 'tip-cap3')

    # @@tcd — matches particella.tex (coarse bins, 4 metrics)
    print('@@tcd:')
    for metrica in ['alberi_ha', 'volume_ha', 'G_ha', 'altezza']:
        save_indexed(acc.calculate_cd_data(data_fab1,
            metrica=metrica, stime_totali=True, fine=False),
            f'tcd-fab1-{metrica}')
    # Fine volume_tot for cross-check
    save_indexed(acc.calculate_cd_data(data_all,
        metrica='volume_tot', stime_totali=True, fine=True),
        'tcd-all-volume_tot-fine')

    print(f'\nRegenerated {len(generated)} golden files.')


if __name__ == '__main__':
    main()
