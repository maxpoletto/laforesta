"""Shared palette semantics; deliberate formatting input differences."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_palette_and_formatting_contracts_across_desktop_and_mobile():
    if shutil.which('node') is None:
        pytest.skip('node required')
    root = Path(__file__).resolve().parents[1]
    script = f'''
      import {{chartSeriesColor, speciesColorMap}} from {json.dumps((root / 'apps/base/static/base/js/charts.js').as_uri())};
      import {{fmtDecimal2}} from {json.dumps((root / 'apps/base/static/base/js/format.js').as_uri())};
      import {{createRequire}} from 'node:module';
      const require = createRequire(import.meta.url);
      const {{IpsoPalette}} = require({json.dumps(str(root / 'apps/ipso/static/ipso/pwa/palette.js'))});
      const {{IpsoFormat}} = require({json.dumps(str(root / 'apps/ipso/static/ipso/pwa/format.js'))});
      const indexes = [-1, 0, 1, 11, 12, 25];
      const names = [' Faggio ', 'Abete', 'Abete', ''];
      const all = ['Abete', 'Castagno', 'Faggio'];
      const values = [null, '', 12.5, '12.5', 'text'];
      process.stdout.write(JSON.stringify({{
        desktopColors: indexes.map(chartSeriesColor), mobileColors: indexes.map(IpsoPalette.chartSeriesColor),
        desktopNames: [...speciesColorMap(names, all)], mobileNames: [...IpsoPalette.speciesColorMap(names, all)],
        desktopNumbers: values.map(fmtDecimal2), mobileNumbers: values.map(IpsoFormat.fmtDecimal2),
      }}));
    '''
    result = subprocess.run(['node', '--input-type=module', '-e', script],
        text=True, capture_output=True, check=True, timeout=10)
    data = json.loads(result.stdout)
    assert data['desktopColors'] == data['mobileColors']
    assert data['desktopNames'] == data['mobileNames']
    assert data['desktopNumbers'] == ['', '', '12,50', '12.5', 'text']
    assert data['mobileNumbers'] == ['', '', '12,50', '12,50', 'text']
