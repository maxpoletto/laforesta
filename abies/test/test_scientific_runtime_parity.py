"""Keep legitimate offline/live algorithms aligned through shared inputs."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from apps.base.dendrometry import basal_area_m2, diameter_class_cm


def test_diameter_area_and_offline_height_contracts():
    if shutil.which('node') is None:
        pytest.skip('node required')
    root = Path(__file__).resolve().parents[1]
    classes = [dict(d=d, mode=mode) for mode in ('centered', 'shifted_up', 'shifted_down')
               for d in (1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 99)]
    heights = [dict(equation=eq, d=d) for eq, d in (
        ({'a': 7, 'b': -4}, 30), ({'a': 0, 'b': 18.5}, 30),
        ({'a': 7, 'b': -4}, 0), ({'a': 7, 'b': -4}, -1),
        ({'a': 0, 'b': 0}, 30), (None, 30),
    )]
    script = f'''
      import {{diameterClassCm, basalAreaM2}} from {json.dumps((root / 'apps/base/static/base/js/dendrometry.js').as_uri())};
      import {{createRequire}} from 'node:module';
      import {{readFileSync}} from 'node:fs';
      const ipso = createRequire(import.meta.url)({json.dumps(str(root / 'apps/ipso/static/ipso/pwa/ipso.js'))});
      const cases = JSON.parse(readFileSync(0, 'utf8'));
      process.stdout.write(JSON.stringify({{
        classes: cases.classes.map(c => [diameterClassCm(c.d, c.mode), basalAreaM2(c.d)]),
        heights: cases.heights.map(c => ipso.computeH(c.equation, c.d)),
      }}));
    '''
    result = subprocess.run(['node', '--input-type=module', '-e', script],
        input=json.dumps(dict(classes=classes, heights=heights)), text=True,
        capture_output=True, check=True, timeout=10)
    actual = json.loads(result.stdout)
    for case, (diameter_class, area) in zip(classes, actual['classes']):
        assert diameter_class == diameter_class_cm(case['d'], case['mode'])
        assert math.isclose(area, basal_area_m2(case['d']), rel_tol=1e-14)
    # Ipso intentionally captures whole-meter automatic heights, including its
    # half-up rounding; desktop's editable preview has its separate precision.
    assert actual['heights'] == [20, 19, None, None, None, None]
