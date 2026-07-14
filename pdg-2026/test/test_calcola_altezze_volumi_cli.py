"""Regression tests for pdg.py --calcola-altezze-volumi."""

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


TREE_COLUMNS = [
    'Compresa', 'Particella', 'Area saggio', 'n', 'poll', 'D(cm)',
    'Classe diametrica', 'h(m)', 'Genere', 'Fustaia', 'L10(mm)', 'c',
]


def _write_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _run_calcola_altezze_volumi(tmp_path, tree_rows, equation_rows, *extra_args):
    input_path = tmp_path / 'alberi.csv'
    equations_path = tmp_path / 'equazioni.csv'
    output_path = tmp_path / 'alberi-calcolati.csv'
    _write_csv(input_path, tree_rows, TREE_COLUMNS)
    _write_csv(
        equations_path,
        equation_rows,
        ['compresa', 'genere', 'funzione', 'a', 'b', 'r2', 'n'],
    )

    subprocess.run(
        [
            sys.executable, str(ROOT / 'pdg.py'),
            '--calcola-altezze-volumi',
            '--equazioni', str(equations_path),
            '--input', str(input_path),
            '--output', str(output_path),
            *extra_args,
        ],
        cwd=ROOT,
        check=True,
    )
    return pd.read_csv(output_path)


def test_calcola_altezze_volumi_removes_duplicate_tree_occurrences(tmp_path):
    result = _run_calcola_altezze_volumi(
        tmp_path,
        [
            ['Serra', '14', '8', 9, '', 30, 6, 12.0, 'Acero', True, 11.0, 200],
            ['Serra', '14', '8', 9, '', 31, 6, 13.0, 'Acero', True, 22.0, 200],
            ['Serra', '14', '8', 10, '', 32, 6, 14.0, 'Acero', True, 33.0, 200],
        ],
        [
            ['Serra', 'Acero', 'ln', 10.0, 1.0, 0.9, 10],
        ],
    )

    assert len(result) == 2
    duplicate_key = (
        (result['Compresa'] == 'Serra') &
        (result['Particella'].astype(str) == '14') &
        (result['Area saggio'].astype(str) == '8') &
        (result['n'] == 9)
    )
    assert duplicate_key.sum() == 1
    assert result.loc[duplicate_key, 'L10(mm)'].iloc[0] == 11.0


def test_calcola_altezze_volumi_keeps_original_height_below_threshold(tmp_path):
    result = _run_calcola_altezze_volumi(
        tmp_path,
        [
            ['Test', '1', '1', 1, '', 1, 1, 9.0, 'Acero', True, 10.0, 200],
        ],
        [
            ['Test', 'Acero', 'ln', 1.0, 0.0, 0.9, 10],
        ],
    )

    assert result.loc[0, 'h(m)'] == 9.0
    assert result.loc[0, 'V(m3)'] > 0


def test_calcola_altezze_volumi_threshold_flag_is_configurable(tmp_path):
    result = _run_calcola_altezze_volumi(
        tmp_path,
        [
            ['Test', '1', '1', 1, '', 2, 1, 9.0, 'Acero', True, 10.0, 200],
        ],
        [
            ['Test', 'Acero', 'ln', 1.0, 0.0, 0.9, 10],
        ],
        '--altezza-minima-calcolata', '0',
    )

    assert 0 < result.loc[0, 'h(m)'] < 5
