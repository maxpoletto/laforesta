"""One GeoJSON name policy on the server and desktop/mobile geometry bridge."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from apps.base.parcel_geometry import parcel_key


def test_geometry_name_aliases_match_without_coercing_parcel_labels():
    if shutil.which('node') is None:
        pytest.skip('node required')
    props = [
        {'layer': 'R', 'name': 'R-01'},
        {'compresa': 'R', 'particella': '1b'},
        {'Compresa': 'R', 'Particella': 'R-10b'},
        {'layer': 'Preferred', 'compresa': 'Other', 'name': 'R-1', 'particella': '2'},
        {'layer': '', 'compresa': 'R', 'particella': '01'},
        {},
    ]
    features = [{'properties': p} for p in props]
    expected = [['R', '01'], ['R', '1b'], ['R', '10b'], ['Preferred', '1'], ['R', '01'], ['', '']]
    assert [list(parcel_key(f)) for f in features] == expected
    module = Path('apps/base/static/base/js/geo.js').resolve().as_uri()
    script = f'''
      import {{parcelNames}} from {json.dumps(module)};
      import {{readFileSync}} from 'node:fs';
      const rows = JSON.parse(readFileSync(0, 'utf8'));
      process.stdout.write(JSON.stringify(rows.map(f => {{
        const p = parcelNames(f); return [p.compresa, p.particella];
      }})));
    '''
    result = subprocess.run(['node', '--input-type=module', '-e', script],
        input=json.dumps(features), text=True, capture_output=True, check=True, timeout=10)
    assert json.loads(result.stdout) == expected
