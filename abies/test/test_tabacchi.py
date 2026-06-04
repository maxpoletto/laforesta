"""Tabacchi volume formula parity (Python ↔ JS).

Reads the JS source `apps/base/static/base/js/volume.js`, extracts
the TABACCHI_B table by regex, and confirms it matches the Python
table in `apps/base/tabacchi.py` (same coefficients, same shape).

Also verifies the Python volume function against a small fixture of
known D/h/species triples (sanity check, not parity).

Finally, `test_js_python_stored_value_parity` drives the *real* volume.js
through node over a D/h/species/density grid and checks that the values an
interactive form would store match the CSV-import path within 1 ULP.  The
two cannot match exactly — JS computes in float64, Python in Decimal, so
half-way cases round differently — so the test asserts a 1-ULP bound rather
than equality.  It is skipped when node is unavailable.
"""

import json
import re
import shutil
import subprocess
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


# ---------------------------------------------------------------------------
# End-to-end stored-value parity (interactive JS vs CSV-import Python)
# ---------------------------------------------------------------------------

# 1 ULP at the stored precisions: volume DecimalField(.,4), mass (.,3).
_VOL_TOLERANCE = Decimal('0.0001')
_MASS_TOLERANCE = Decimal('0.001')
# Representative densities (q/m³): one whole, one fractional, to exercise the
# half-way rounding cases that drive the JS/Python residual.
_PARITY_DENSITIES = ('9.0', '7.5')

# Mirrors tree-form.js: store volume at 4 dp and derive mass from that stored
# volume.  `{volume_url}` / `{densities}` are filled in by _run_js_sweep.
_JS_DRIVER = '''
import {{ TABACCHI_B, tabacchiVolumeM3, massQ }} from {volume_url};
const DENSITIES = {densities};
const out = [];
for (const sp of Object.keys(TABACCHI_B)) {{
  for (let d = 5; d <= 80; d += 5) {{
    for (let h = 3; h <= 35; h += 2) {{
      const v = tabacchiVolumeM3(d, h, sp);
      if (v == null || v <= 0) continue;
      const vStored = Number(v.toFixed(4));
      for (const dens of DENSITIES) {{
        const m = massQ(vStored, dens);
        out.push([sp, d, h, dens, v.toFixed(4), m.toFixed(3)].join('|'));
      }}
    }}
  }}
}}
process.stdout.write(out.join('\\n'));
'''


def _run_js_sweep(tmp_path):
    """Drive the real volume.js via node; return rows of
    (species, d, h, density_str, vol_str, mass_str)."""
    driver = tmp_path / 'parity_driver.mjs'
    driver.write_text(_JS_DRIVER.format(
        volume_url=json.dumps(JS_PATH.as_uri()),
        densities='[' + ', '.join(_PARITY_DENSITIES) + ']',
    ), encoding='utf-8')
    proc = subprocess.run(
        ['node', str(driver)], capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f'node driver failed: {proc.stderr}'
    rows = []
    for line in proc.stdout.splitlines():
        sp, d, h, dens, vol, mass = line.split('|')
        rows.append((sp, int(d), int(h), dens, vol, mass))
    return rows


@pytest.mark.skipif(shutil.which('node') is None,
                    reason='node not available for JS parity check')
def test_js_python_stored_value_parity(tmp_path):
    """A tree entered via the interactive form (volume.js) stores the same
    volume and mass as the CSV-import path (tabacchi.py + tree_mass_q), within
    1 ULP.  Guards against the coefficient tables, the 4-dp volume rounding, or
    the mass-from-stored-volume rule drifting apart between JS and Python."""
    from apps.base.models import tree_mass_q

    rows = _run_js_sweep(tmp_path)
    assert rows, 'JS driver produced no rows'
    for species, d, h, dens_str, js_vol, js_mass in rows:
        py_vol = tabacchi.tabacchi_volume_m3(d, h, species)
        py_mass = tree_mass_q(py_vol, Decimal(dens_str))
        assert abs(Decimal(js_vol) - py_vol) <= _VOL_TOLERANCE, \
            f'{species} D={d} h={h}: vol js={js_vol} py={py_vol}'
        assert abs(Decimal(js_mass) - py_mass) <= _MASS_TOLERANCE, \
            f'{species} D={d} h={h} rho={dens_str}: mass js={js_mass} py={py_mass}'
