"""Tests for row-group visual separation in format_table."""

import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdg.formatters import (
    LaTeXSnippetFormatter, HTMLSnippetFormatter, CSVSnippetFormatter,
    ColSpec,
)
from pdg.core import render_table

HEADERS = [('Anno', 'l'), ('Particella', 'l'), ('Prelievo (m3)', 'r')]

# Three years, with 2028 and 2030 having two rows each.
ROWS = [
    ['2028', 'P1', '100'],
    ['2028', 'P2', '200'],
    ['2029', 'P3', '150'],
    ['2030', 'P4', '120'],
    ['2030', 'P5', '180'],
]

# Group breaks: new year starts at indices 0, 2, 3.
GROUP_BREAKS = [0, 2, 3]


class TestLaTeXGroupBreaks:
    """LaTeX formatter should insert \\hline between year groups."""

    def test_hline_between_groups(self):
        fmt = LaTeXSnippetFormatter()
        result = fmt.format_table(HEADERS, ROWS, row_groups=GROUP_BREAKS)
        lines = result.split('\n')
        # Find the data lines (after \\endlastfoot, before \\end{longtable})
        data_lines = []
        in_data = False
        for line in lines:
            if '\\endlastfoot' in line:
                in_data = True
                continue
            if '\\end{longtable}' in line:
                break
            if in_data and line.strip():
                data_lines.append(line)
        # Between row index 1 (2028/P2) and 2 (2029/P3) there should be an \hline
        # Between row index 2 (2029/P3) and 3 (2030/P4) there should be an \hline
        # But NOT before the very first row (index 0), and NOT between rows of same year
        assert any('\\hline' in l for l in data_lines), \
            f"Expected \\hline in data section, got: {data_lines}"
        # Count hlines in data section: should be exactly 2 (between the 3 groups)
        hline_count = sum(1 for l in data_lines if '\\hline' in l)
        assert hline_count == 2, f"Expected 2 \\hlines, got {hline_count}"

    def test_no_groups_no_extra_hlines(self):
        """Without row_groups, no extra hlines in data section."""
        fmt = LaTeXSnippetFormatter()
        result_without = fmt.format_table(HEADERS, ROWS)
        result_with_none = fmt.format_table(HEADERS, ROWS, row_groups=None)
        lines_without = result_without.split('\n')
        lines_with_none = result_with_none.split('\n')
        # Data section should have no hlines
        for result_lines in [lines_without, lines_with_none]:
            in_data = False
            for line in result_lines:
                if '\\endlastfoot' in line:
                    in_data = True
                    continue
                if '\\end{longtable}' in line:
                    break
                if in_data and '\\hline' in line:
                    pytest.fail(f"Unexpected \\hline in data: {line}")


class TestHTMLGroupBreaks:
    """HTML formatter should add a CSS class on rows that start a new group."""

    def test_group_first_class(self):
        fmt = HTMLSnippetFormatter()
        result = fmt.format_table(HEADERS, ROWS, row_groups=GROUP_BREAKS)
        # Rows at indices 2 and 3 should have a distinguishing class.
        # Row at index 0 should NOT (it's the first row, no line needed above it).
        # Extract only <tbody> section to skip the header <tr>.
        tbody = result.split('<tbody>')[1].split('</tbody>')[0]
        tr_lines = [l.strip() for l in tbody.split('\n') if '<tr' in l]
        # Should have 5 data <tr> lines
        assert len(tr_lines) == 5, f"Expected 5 <tr>s, got {len(tr_lines)}: {tr_lines}"
        # Indices 0, 1: no group-first class (first group, no separator needed)
        assert 'group-first' not in tr_lines[0]
        assert 'group-first' not in tr_lines[1]
        # Index 2 (first row of 2029): should have group-first
        assert 'group-first' in tr_lines[2]
        # Index 3 (first row of 2030): should have group-first
        assert 'group-first' in tr_lines[3]
        # Index 4: same year as index 3, no class
        assert 'group-first' not in tr_lines[4]

    def test_no_groups_no_class(self):
        fmt = HTMLSnippetFormatter()
        result = fmt.format_table(HEADERS, ROWS)
        assert 'group-first' not in result


class TestCSVGroupBreaks:
    """CSV formatter should ignore row_groups."""

    def test_ignores_groups(self):
        fmt = CSVSnippetFormatter()
        result_without = fmt.format_table(HEADERS, ROWS)
        result_with = fmt.format_table(HEADERS, ROWS, row_groups=GROUP_BREAKS)
        assert result_without == result_with


class TestRenderTableGroupBy:
    """render_table should compute and pass group breaks when group_by_col is set."""

    def _make_df(self):
        return pd.DataFrame({
            'year': [2028, 2028, 2029, 2030, 2030],
            'harvest': [100.0, 200.0, 150.0, 120.0, 180.0],
        })

    def test_group_by_col_produces_hlines(self):
        df = self._make_df()
        col_specs = [
            ColSpec('Anno', 'l', lambda r: str(int(r['year'])), None, True),
            ColSpec('Prelievo', 'r', 'harvest', 'harvest', True),
        ]
        result = render_table(df, [], col_specs, LaTeXSnippetFormatter(),
                              add_totals=False, group_by_col='year')
        # Should have hlines between year groups in the data section
        lines = result.snippet.split('\n')
        in_data = False
        hline_count = 0
        for line in lines:
            if '\\endlastfoot' in line:
                in_data = True
                continue
            if '\\end{longtable}' in line:
                break
            if in_data and '\\hline' in line:
                hline_count += 1
        assert hline_count == 2

    def test_no_group_by_col_no_hlines(self):
        df = self._make_df()
        col_specs = [
            ColSpec('Anno', 'l', lambda r: str(int(r['year'])), None, True),
            ColSpec('Prelievo', 'r', 'harvest', 'harvest', True),
        ]
        result = render_table(df, [], col_specs, LaTeXSnippetFormatter(),
                              add_totals=False)
        lines = result.snippet.split('\n')
        in_data = False
        for line in lines:
            if '\\endlastfoot' in line:
                in_data = True
                continue
            if '\\end{longtable}' in line:
                break
            if in_data and '\\hline' in line:
                pytest.fail(f"Unexpected \\hline: {line}")
