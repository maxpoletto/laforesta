"""Recorded early-year cuts must constrain inventory and future scheduling."""

import importlib.util
import math
from pathlib import Path
import sys

import pandas as pd
import pytest

from pdg.ceduo import CoppiceParcel, coppice_gantt_bars, schedule_coppice
from pdg.computation import (
    COL_AREA_SAGGIO, COL_CD_CM, COL_COMPRESA, COL_D_CM, COL_GENERE,
    COL_GOVERNO, COL_L10_MM, COL_PARTICELLA, COL_PRESSLER, COL_V_M3,
    GOV_CEDUO, GOV_FUSTAIA, ParcelData, ParcelStats,
)
from pdg.core import render_harvest_plan, render_harvest_table
from pdg.formatters import CSVSnippetFormatter
from pdg.io import COL_RECORDED_VOLUME, early_calendar, read_past_harvests
from pdg.simulation import (
    COL_HARVEST, COL_N_TREES, COL_RECORDED, COL_VOLUME_AFTER,
    COL_VOLUME_BEFORE, COL_YEAR, schedule_harvests, schedule_harvests_complete,
)


def calendar(*rows):
    return pd.DataFrame(rows, columns=[
        'Anno', COL_COMPRESA, COL_PARTICELLA, COL_GOVERNO, COL_RECORDED_VOLUME])


@pytest.fixture
def forest():
    # Unscaled rows make inventory balances explicit: each parcel has 60 m³.
    trees = pd.DataFrame([
        {COL_COMPRESA: 'X', COL_PARTICELLA: parcel, COL_AREA_SAGGIO: 1,
         COL_GENERE: 'Faggio', COL_D_CM: diameter, COL_CD_CM: diameter,
         COL_V_M3: volume, COL_PRESSLER: 200, COL_L10_MM: 0}
        for parcel in ('A', 'B')
        for diameter, volume in ((25, 10.0), (35, 20.0), (45, 30.0))
    ])
    parcels = {('X', p): ParcelStats(1.0, 'A', 60, GOV_FUSTAIA, 1, 1.0)
               for p in ('A', 'B')}
    return ParcelData(trees, ['X'], ['Faggio'], parcels)


def cut_all(sector, age, volume, basal):
    return volume, math.inf


def run(forest, past, **kwargs):
    options = dict(year_range=(2026, 2027), min_gap=1, target_volume=1e9,
                   rules=cut_all)
    options.update(kwargs)
    return schedule_harvests(forest, past, **options)


class TestCalendarValidation:
    def test_old_rows_need_no_volume_and_input_is_immutable(self):
        past = calendar((2015, 'X', 'A', GOV_FUSTAIA, None),
                        (2026, 'X', 'B', GOV_FUSTAIA, '15.5'))
        original = past.copy(deep=True)
        end, early = early_calendar(past, (2026, 2040))
        assert end == 2026
        assert early[COL_RECORDED_VOLUME].tolist() == [15.5]
        pd.testing.assert_frame_equal(past, original)

    @pytest.mark.parametrize('volume', [None, '', 'unknown', -1, 0, float('inf'), float('-inf')])
    def test_missing_or_invalid_early_volume_identifies_parcel(self, volume):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, volume))
        with pytest.raises(ValueError, match='Prelievo totale.*X/A.*2026'):
            early_calendar(past, (2026, 2040))

    def test_missing_volume_column_is_valid_only_without_overlap(self):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15)).drop(columns=COL_RECORDED_VOLUME)
        assert early_calendar(past, (2027, 2040))[1].empty
        with pytest.raises(ValueError, match='Prelievo totale'):
            early_calendar(past, (2026, 2040))

    def test_ceduo_needs_no_volume_column(self):
        past = calendar((2026, 'X', 'C', GOV_CEDUO, None)).drop(columns=COL_RECORDED_VOLUME)
        end, early = early_calendar(past, (2026, 2040))
        assert end == 2026
        assert early[COL_PARTICELLA].tolist() == ['C']

    def test_overlap_counts_empty_years_and_uses_entire_calendar(self, forest):
        # No X parcel or fustaia record in 2026: that still fixes the cutoff.
        past = calendar((2026, 'Other', 'C', GOV_CEDUO, 10))
        with pytest.raises(ValueError, match='2 anni sovrapposti'):
            run(forest, past, year_range=(2025, 2040))
        assert run(forest, past, year_range=(2026, 2026)) == []

    def test_duplicate_early_assignment_is_rejected(self):
        past = calendar(*[(2026, 'X', 'A', GOV_FUSTAIA, 15)] * 2)
        with pytest.raises(ValueError, match='duplicato.*X/A.*2026'):
            early_calendar(past, (2026, 2040))

    def test_semicolon_calendar_with_decimal_comma(self, tmp_path):
        path = tmp_path / 'calendar.csv'
        path.write_text('Anno;Compresa;Particella;Prelievo totale (m3)\n'
                        '2026;X;1;15,5\n')
        past = read_past_harvests(path)
        assert past.iloc[0][COL_PARTICELLA] == '1'
        assert early_calendar(past, (2026, 2040))[1].iloc[0][COL_RECORDED_VOLUME] == 15.5


class TestRecordedInventory:
    def test_blank_ceduo_volume_does_not_block_fustaia(self, forest):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15),
                        (2026, 'X', 'C', GOV_CEDUO, None))
        events = run(forest, past, year_range=(2026, 2026))
        assert [(e[COL_PARTICELLA], e[COL_HARVEST]) for e in events] == [('A', 15)]

    def test_only_recorded_cuts_then_future_uses_remaining_stock(self, forest, capsys):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15))
        log = {}
        events = run(forest, past, volume_log=log)
        early = [e for e in events if e[COL_YEAR] == 2026]
        assert len(early) == 1
        assert early[0][COL_PARTICELLA] == 'A'
        assert early[0][COL_HARVEST] == 15
        assert early[0][COL_N_TREES] == 1.25
        assert early[0][COL_VOLUME_BEFORE] == 60
        assert early[0][COL_VOLUME_AFTER] == 45
        assert early[0][COL_RECORDED]
        assert log[2026] == {('X', 'A'): 60, ('X', 'B'): 60}
        assert log[2027] == {('X', 'A'): 45, ('X', 'B'): 60}
        future = {e[COL_PARTICELLA]: e[COL_HARVEST]
                  for e in events if e[COL_YEAR] == 2027}
        assert future == {'A': 45, 'B': 60}
        assert 'anno 2026:' not in capsys.readouterr().out
        assert forest.trees[COL_V_M3].sum() == 120  # input remains untouched

    def test_same_selection_function_controls_recorded_cuts(self, forest):
        def largest_first(trees):
            return trees.sort_values(COL_D_CM, ascending=False).index

        # A 15 m³ cut leaves 15 of the largest row. With a future 10 m³
        # limit, the small row survives this selection and supplies that cut.
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15))
        events = run(forest, past, tree_selection=largest_first,
                     rules=lambda *args: (10, math.inf))
        early = events[0]
        assert early[COL_N_TREES] == 0.5
        future_a = next(e for e in events if e[COL_YEAR] == 2027
                        and e[COL_PARTICELLA] == 'A')
        assert future_a[COL_HARVEST] == 10

    def test_early_cuts_ignore_rules_targets_and_rest_period(self, forest):
        def forbidden(*args):
            pytest.fail('Planning rules must not run in an early year')

        past = calendar((2025, 'X', 'A', GOV_FUSTAIA, None),
                        (2026, 'X', 'A', GOV_FUSTAIA, 15))
        events = run(forest, past, year_range=(2026, 2026), rules=forbidden,
                     min_gap=10, target_volume=1, particelle_min=10, prudence=0,
                     forced={2026: [('X', 'B')]})
        assert len(events) == 1
        assert events[0][COL_HARVEST] == 15

    def test_future_rest_period_and_mortality_include_recorded_cut(self, forest):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15))
        log = {}
        events = run(forest, past, year_range=(2026, 2028), min_gap=2,
                     mortality=10, volume_log=log)
        a = [e for e in events if e[COL_PARTICELLA] == 'A']
        assert [e[COL_YEAR] for e in a] == [2026, 2028]
        assert log[2027][('X', 'A')] == pytest.approx(40.5)
        assert a[1][COL_HARVEST] == pytest.approx(36.45)

    def test_future_growth_applies_to_retained_inventory(self, forest):
        # Set rates so Pressler produces a 10% yearly volume increment.
        forest.trees[COL_L10_MM] = forest.trees[COL_D_CM] * 2.5
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15))
        log = {}
        run(forest, past, year_range=(2026, 2026), mortality=5, volume_log=log)
        assert log[2027][('X', 'A')] == pytest.approx(45 * 1.1 * .95)

    def test_recorded_volume_above_available_stock_is_an_error(self, forest):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 61))
        with pytest.raises(ValueError, match='X/A.*2026.*superiore.*disponibile'):
            run(forest, past)

    def test_full_coverage_never_adds_an_early_cut(self, forest):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15))
        events = schedule_harvests_complete(
            forest, past, (2026, 2027), 10, 1, rules=cut_all)
        assert [(e[COL_PARTICELLA], e[COL_HARVEST]) for e in events
                if e[COL_YEAR] == 2026] == [('A', 15)]
        assert any(e[COL_PARTICELLA] == 'B' and e[COL_YEAR] == 2027 for e in events)

    def test_calendar_without_current_parcel_still_freezes_early_year(self, forest):
        past = calendar((2026, 'Elsewhere', 'C', GOV_CEDUO, 10))
        assert run(forest, past, year_range=(2026, 2026)) == []


def render_options():
    return dict(anno_inizio=2026, anno_fine=2027, intervallo=1,
                volume_obiettivo=1e9, mortalita=0, riduzione=50,
                per_compresa=True, per_particella=True, per_genere=False,
                col_comparto=False, col_eta=False, col_pp_max=False,
                col_prima_dopo=False, col_area_ha=False, col_prelievo=True,
                col_prelievo_ha=False, totali=False)


class TestRecordedRendering:
    @pytest.mark.parametrize('grouping', [
        {},
        dict(per_compresa=False, per_particella=False),
        dict(per_particella=False, per_genere=True),
    ])
    def test_plan_and_prelievi_reduce_only_generated_volumes(self, forest, grouping):
        past = calendar((2026, 'X', 'A', GOV_FUSTAIA, 15))
        formatter = CSVSnippetFormatter()
        options = render_options()
        options.update(grouping)
        plan = render_harvest_plan(forest, past, cut_all, formatter, **options).snippet
        totals = render_harvest_table(forest, past, cut_all, formatter, **options).snippet
        assert '15' in plan
        # Inspect the numeric tables independently of CSV column order.
        from io import StringIO
        plan_df = pd.read_csv(StringIO(plan))
        totals_df = pd.read_csv(StringIO(totals))
        assert plan_df.loc[plan_df['Anno'] == 2026, 'Prelievo (m³)'].sum() == 15
        assert plan_df.loc[plan_df['Anno'] == 2027, 'Prelievo (m³)'].sum() == 52.5
        assert totals_df['Prel tot (m³)'].sum() == 67.5
        # A second rendering reuses the cache without reducing it again.
        assert render_harvest_table(forest, past, cut_all, formatter, **options).snippet == totals


def coppice(past, parcels, year_range=(2026, 2042), adjacencies=None):
    last = {}
    for _, row in past.iterrows():
        if row[COL_GOVERNO] == GOV_CEDUO:
            key = (row[COL_COMPRESA], row[COL_PARTICELLA])
            last[key] = max(last.get(key, 0), int(row['Anno']))
    return schedule_coppice(parcels, adjacencies or set(), last, year_range,
                            past_harvests=past)


class TestRecordedCoppice:
    def test_completed_record_covers_whole_parcel_until_next_rotation(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None))
        past['Intervento completo'] = 'si'
        parcels = [CoppiceParcel('X', 'A', 13.5, 18)]
        events = coppice(past, parcels, (2026, 2050))
        assert [(e.year, e.area_ha, e.cycle_start) for e in events] == [
            (2026, 13.5, 2026), (2044, 6.75, 2044), (2046, 6.75, 2044)]
        bars = coppice_gantt_bars(parcels, events)[0].bars
        assert [(b.start_year, b.cycle_idx, b.sub_idx, b.n_sub) for b in bars] == [
            (2026, 1, 1, 1), (2044, 2, 1, 2), (2046, 2, 2, 2)]

    @pytest.mark.parametrize('value', [None, '', 'no'])
    def test_blank_or_no_preserves_default_sub_interventions(self, value):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None))
        past['Intervento completo'] = value
        events = coppice(past, [CoppiceParcel('X', 'A', 13.5, 18)], (2026, 2030))
        assert [(e.year, e.area_ha) for e in events] == [(2026, 6.75), (2028, 6.75)]

    def test_invalid_completion_flag_identifies_record(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None))
        past['Intervento completo'] = 'maybe'
        with pytest.raises(ValueError, match='Intervento completo.*X/A.*2026'):
            coppice(past, [CoppiceParcel('X', 'A', 13.5, 18)])

    def test_completed_record_still_constrains_neighbors(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None))
        past['Intervento completo'] = 'si'
        parcels = [CoppiceParcel('X', 'A', 13.5, 18), CoppiceParcel('X', 'B', 8, 18)]
        events = coppice(past, parcels, (2026, 2030), {(('X', 'A'), ('X', 'B'))})
        assert [(e.particella, e.year, e.area_ha) for e in events] == [
            ('A', 2026, 13.5), ('B', 2028, 8)]

    @pytest.mark.parametrize('area, years', [(8, [2026, 2038]),
                                            (10, [2026, 2038]),
                                            (20, [2026, 2028, 2038, 2040]),
                                            (25, [2026, 2028, 2030, 2038, 2040, 2042])])
    def test_first_sub_intervention_continues_and_anchors_next_cycle(self, area, years):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None)).drop(columns=COL_RECORDED_VOLUME)
        parcels = [CoppiceParcel('X', 'A', area, 12)]
        events = coppice(past, parcels)
        assert [e.year for e in events] == years
        n_sub = math.ceil(area / 10)
        assert all(e.area_ha == area / n_sub for e in events)
        assert [e.cycle_start for e in events[:n_sub]] == [2026] * n_sub
        if n_sub > 1:
            bars = coppice_gantt_bars(parcels, events)[0].bars
            assert [b.sub_idx for b in bars[:n_sub]] == list(range(1, n_sub + 1))
            assert all(b.cycle_idx == 1 for b in bars[:n_sub])

    def test_unrecorded_parcel_waits_until_future_year(self):
        past = calendar((2026, 'Elsewhere', 'F', GOV_FUSTAIA, 10))
        events = coppice(past, [CoppiceParcel('X', 'A', 8, 12)], (2026, 2027))
        assert [e.year for e in events] == [2027]

    def test_recorded_cycle_is_truncated_by_requested_end(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, 10))
        events = coppice(past, [CoppiceParcel('X', 'A', 25, 12)], (2026, 2028))
        assert [e.year for e in events] == [2026, 2028]
        assert all(e.cycle_start == 2026 for e in events)

    @pytest.mark.parametrize('start', [2026, 2027])
    def test_new_cycle_is_scheduled_around_recorded_neighbor(self, start):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, 10))
        parcels = [CoppiceParcel('X', 'A', 8, 12), CoppiceParcel('X', 'B', 25, 12)]
        events = coppice(past, parcels, (start, 2030), {(('X', 'A'), ('X', 'B'))})
        b = [e for e in events if e.particella == 'B']
        assert [e.year for e in b] == [2028, 2030]
        assert all(e.cycle_start == 2028 for e in b)
        if start == 2026:
            assert [e.year for e in events if e.particella == 'A'] == [2026]

    def test_conflicting_fixed_records_cannot_be_rescheduled(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None),
                        (2026, 'X', 'B', GOV_CEDUO, None))
        parcels = [CoppiceParcel('X', 'A', 8, 12), CoppiceParcel('X', 'B', 25, 12)]
        with pytest.raises(ValueError, match='registrato X/A.*2026.*registrato X/B.*2026'):
            coppice(past, parcels, (2026, 2030), {(('X', 'A'), ('X', 'B'))})

    def test_new_cut_outside_end_year_is_omitted(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, None))
        parcels = [CoppiceParcel('X', 'A', 8, 12), CoppiceParcel('X', 'B', 8, 12)]
        events = coppice(past, parcels, (2026, 2027), {(('X', 'A'), ('X', 'B'))})
        assert [(e.particella, e.year) for e in events] == [('A', 2026)]

    def test_multiple_early_years_are_rejected(self):
        past = calendar((2026, 'Other', 'F', GOV_FUSTAIA, 10))
        with pytest.raises(ValueError, match='2 anni sovrapposti'):
            coppice(past, [CoppiceParcel('X', 'A', 8, 12)], (2025, 2040))

    def test_recorded_neighbor_outside_current_parcels_still_constrains_adjacency(self):
        past = calendar((2026, 'X', 'A', GOV_CEDUO, 10))
        events = coppice(past, [CoppiceParcel('X', 'B', 8, 12)], (2026, 2030),
                         {(('X', 'A'), ('X', 'B'))})
        assert [e.year for e in events] == [2028]


@pytest.fixture(scope='module')
def cli():
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')  # Render Gantt charts without a macOS UI session.
    spec = importlib.util.spec_from_file_location('pdg_cli_early', Path(__file__).parents[1] / 'pdg.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_both_ceduo_directives_use_recorded_cycle(cli, tmp_path):
    data_dir = Path(__file__).parent / 'data'
    # Neither directive consumes volumes, even in a mixed calendar.
    past = calendar((2026, 'X', 'A', GOV_CEDUO, None),
                    (2026, 'Other', 'F', GOV_FUSTAIA, None)).drop(columns=COL_RECORDED_VOLUME)
    calendar_path = tmp_path / 'calendar.csv'
    past.to_csv(calendar_path, index=False)
    adjacency_path = tmp_path / 'adjacencies.csv'
    adjacency_path.write_text('Compresa,A,B\n')
    params = (f'particelle=particelle-ceduo.csv,adiacenze={adjacency_path},'
              f'calendario={calendar_path},anno_inizio=2026,anno_fine=2040')
    result = cli.process_template(
        f'@@calendario_ceduo({params})\n@@tabella_ceduo({params})',
        data_dir, str(data_dir / 'particelle-ceduo.csv'), tmp_path, 'tex')
    assert '2026' in result
    assert '@@' not in result
    assert len(list(tmp_path.glob('tabella_ceduo_*.png'))) == 1


def test_plan_and_prelievi_directives_replay_exact_volume(cli, tmp_path, trees_df):
    from io import StringIO

    data_dir = Path(__file__).parent / 'data'
    calendar_path = tmp_path / 'calendar.csv'
    calendar((2026, 'Test', 'A', GOV_FUSTAIA, 15),
             (2026, 'Test', 'Ceduo', GOV_CEDUO, None)).to_csv(calendar_path, index=False)
    trees_df.to_csv(tmp_path / 'early-trees.csv', index=False)
    params = (f'alberi=early-trees.csv,calendario={calendar_path},anno_inizio=2026,'
              'anno_fine=2026,volume_obiettivo=1,prudenza=0,riduzione=50,'
              'prelievo_completo=si')
    for directive, extra, harvest_column in [
            ('piano_di_taglio', ',col_prima_dopo=no', 'Prelievo (m³)'),
            ('prelievi', '', 'Prel tot (m³)')]:
        result = cli.process_template(
            f'@@{directive}({params}{extra})', tmp_path,
            str(data_dir / 'particelle.csv'), tmp_path, 'csv')
        df = pd.read_csv(StringIO(result))
        assert df[COL_PARTICELLA].tolist() == ['A']
        assert df[harvest_column].tolist() == [15]


def test_cli_rejects_missing_early_volume(cli, tmp_path):
    data_dir = Path(__file__).parent / 'data'
    calendar_path = tmp_path / 'calendar.csv'
    calendar((2026, 'Test', 'A', GOV_FUSTAIA, None)).to_csv(calendar_path, index=False)
    directive = (f'@@piano_di_taglio(alberi=alberi.csv,calendario={calendar_path},'
                 'anno_inizio=2026,anno_fine=2027,volume_obiettivo=1000)')
    with pytest.raises(ValueError, match='Prelievo totale.*Test/A.*2026'):
        cli.process_template(directive, data_dir, str(data_dir / 'particelle.csv'),
                             tmp_path, 'csv')
