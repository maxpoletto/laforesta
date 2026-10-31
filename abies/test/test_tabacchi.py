"""Tabacchi volume formula parity (Python ↔ JS).

Supplies one coefficient fixture to the Python and JavaScript numerical cores.
The browser has no separate species catalogue; live previews receive database
configuration from the species form.

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
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from apps.base import tabacchi
from config import strings as S


JS_PATH = (Path(__file__).resolve().parent.parent
           / 'apps/base/static/base/js/volume.js')

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


@pytest.mark.parametrize('name,d,h', [
    ('Ontano', 5, 3), ('Pino Nero', 1, 1), ('Abete', 1, 1), ('Cerro', 1, 1),
])
def test_negative_estimates_are_zero_harvestable_volume(name, d, h):
    from apps.base.models import tree_mass_q

    volume = tabacchi.tabacchi_volume_m3(d, h, name)
    assert volume == Decimal('0.0000')
    assert not volume.is_signed()
    assert tree_mass_q(volume, Decimal('9')) == Decimal('0.000')


def test_small_positive_estimates_are_not_clamped():
    assert tabacchi.tabacchi_volume_m3(1, 1, 'Faggio') == Decimal('0.0009')


def test_has_species():
    assert tabacchi.has_species('Faggio')
    assert not tabacchi.has_species('Altro')


@pytest.mark.parametrize('name,proxy', [
    (S.SPECIES_OTHER_CONIFER, tabacchi.SP_ABETE),
    (S.SPECIES_OTHER_BROADLEAF, tabacchi.SP_FAGGIO),
])
def test_other_species_use_reference_equations(name, proxy):
    assert tabacchi.has_species(name)
    for d, h in ((5, 3), (30, 20), (40, 25), (80, 35)):
        assert tabacchi.tabacchi_volume_m3(d, h, name) == tabacchi.tabacchi_volume_m3(d, h, proxy)


# ---------------------------------------------------------------------------
# End-to-end stored-value parity (interactive JS vs CSV-import Python)
# ---------------------------------------------------------------------------

# 1 ULP at the stored precisions: volume DecimalField(.,4), mass (.,3).
_VOL_TOLERANCE = Decimal('0.0001')
_MASS_TOLERANCE = Decimal('0.001')
# Representative densities (q/m³): one whole, one fractional, to exercise the
# half-way rounding cases that drive the JS/Python residual.
_PARITY_DENSITIES = ('9.0', '7.5')
_PARITY_EQUATIONS = {**tabacchi.TABACCHI_B,
    **{name: tabacchi.TABACCHI_B[target] for name, target in tabacchi.TABACCHI_PROXIES.items()},
    'Configured equation': (123.45, .012345, .5),
}

# Mirrors tree-form.js: store volume at 4 dp and derive mass from that stored
# volume.  `{volume_url}` / `{densities}` are filled in by _run_js_sweep.
_JS_DRIVER = '''
import {{ volumeFromCoefficientsM3, massQ }} from {volume_url};
const COEFFICIENTS = {coefficients};
const DENSITIES = {densities};
const out = [];
if (volumeFromCoefficientsM3(30, 20, null) !== null) {{
  throw new Error('Legacy catch-all must remain unsupported');
}}
for (const sp of Object.keys(COEFFICIENTS)) {{
  for (let d = 1; d <= 80; d += 1) {{
    for (let h = 1; h <= 35; h += 2) {{
      const v = volumeFromCoefficientsM3(d, h, COEFFICIENTS[sp]);
      if (v == null) continue;
      if (v < 0 || Object.is(v, -0)) throw new Error('Negative harvestable volume');
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
        coefficients=json.dumps(_PARITY_EQUATIONS),
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
    1 ULP.  Guards against configured-coefficient evaluation, the 4-dp volume rounding, or
    the mass-from-stored-volume rule drifting apart between JS and Python."""
    from apps.base.models import tree_mass_q

    rows = _run_js_sweep(tmp_path)
    assert rows, 'JS driver produced no rows'
    assert {S.SPECIES_OTHER_CONIFER, S.SPECIES_OTHER_BROADLEAF} <= {row[0] for row in rows}
    for species, d, h, dens_str, js_vol, js_mass in rows:
        py_vol = tabacchi._volume_from_b(float(d), float(h), _PARITY_EQUATIONS[species])
        py_mass = tree_mass_q(py_vol, Decimal(dens_str))
        assert abs(Decimal(js_vol) - py_vol) <= _VOL_TOLERANCE, \
            f'{species} D={d} h={h}: vol js={js_vol} py={py_vol}'
        assert abs(Decimal(js_mass) - py_mass) <= _MASS_TOLERANCE, \
            f'{species} D={d} h={h} rho={dens_str}: mass js={js_mass} py={py_mass}'
