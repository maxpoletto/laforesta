"""Tabacchi volume formula parity (Python ↔ JS).

Reads the JS source `apps/base/static/base/js/volume.js`, extracts
the TABACCHI_B table by regex, and confirms it matches the Python
table in `apps/base/tabacchi.py` (same coefficients, same shape).

Also verifies the Python volume function against a small fixture of
known D/h/species triples (sanity check, not parity).
"""

import re
from decimal import Decimal
from pathlib import Path

import pytest

from apps.base import tabacchi


JS_PATH = (Path(__file__).resolve().parent.parent
           / 'apps/base/static/base/js/volume.js')

# Numeric tuple in JS: `[ -1.8381, 3.7836e-2, 3.9934e-1 ]`.
_NUMBER = r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?'


def _parse_js_table() -> dict[str, tuple[float, ...]]:
    """Pull `[SP_X]: [ ... ]` rows out of TABACCHI_B."""
    text = JS_PATH.read_text(encoding='utf-8')
    # Map JS SP_* constants to their string values.
    consts = dict(re.findall(
        r"export const (SP_\w+) = '([^']+)';", text,
    ))
    rows = re.findall(
        r"\[(SP_\w+)\]:\s*\[([^\]]+)\]",
        text,
    )
    table: dict[str, tuple[float, ...]] = {}
    for const_name, body in rows:
        species = consts[const_name]
        coeffs = tuple(
            float(m.group(0))
            for m in re.finditer(_NUMBER, body)
        )
        table[species] = coeffs
    return table


def test_js_table_parses():
    """Sanity check: the regex extraction finds all 13 species."""
    js = _parse_js_table()
    assert len(js) == len(tabacchi.TABACCHI_B), \
        f'JS: {len(js)} species, Python: {len(tabacchi.TABACCHI_B)}'


def test_js_python_coefficient_parity():
    """Every species' b vector matches between JS and Python."""
    js = _parse_js_table()
    for name, py_b in tabacchi.TABACCHI_B.items():
        assert name in js, f'Missing in JS: {name}'
        js_b = js[name]
        assert len(js_b) == len(py_b), f'Length mismatch for {name}'
        for i, (a, b) in enumerate(zip(py_b, js_b)):
            assert abs(a - b) < 1e-9, \
                f'{name}[{i}]: py={a} != js={b}'


@pytest.mark.parametrize('d_cm,h_m,species,expected_m3', [
    # Faggio (length-2): V = (0.81151 + 0.038965 * D²h) / 1000
    #   D=30, h=20 → (0.81151 + 0.038965 * 18000) / 1000 ≈ 0.7022
    (30, 20, 'Faggio', Decimal('0.7022')),
    # Abete (length-3): V = (-1.8381 + 0.037836 * D²h + 0.39934 * D) / 1000
    #   D=40, h=25 → (-1.8381 + 0.037836 * 40000 + 0.39934*40)/1000 ≈ 1.5276
    (40, 25, 'Abete', Decimal('1.5276')),
])
def test_volume_known_values(d_cm, h_m, species, expected_m3):
    v = tabacchi.tabacchi_volume_m3(d_cm, h_m, species)
    # Allow ±0.001 m³ for rounding.
    assert abs(v - expected_m3) < Decimal('0.001'), f'V={v}, expected={expected_m3}'


def test_volume_unknown_species_raises():
    with pytest.raises(ValueError):
        tabacchi.tabacchi_volume_m3(20, 15, 'Altro')


def test_has_species():
    assert tabacchi.has_species('Faggio')
    assert not tabacchi.has_species('Altro')
