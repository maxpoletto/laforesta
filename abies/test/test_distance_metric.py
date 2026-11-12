"""Desktop and mobile share WGS84 ellipsoidal distances and sample inference."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from apps.base.geodesy import distance_m


def test_shared_ellipsoidal_distance_and_sample_boundary():
    if shutil.which('node') is None:
        pytest.skip('node required')
    root = Path(__file__).resolve().parents[1]
    script = f'''
      import * as geo from {json.dumps((root / 'apps/base/static/base/js/geo.js').as_uri())};
      import {{createRequire}} from 'node:module';
      const upload = createRequire(import.meta.url)({json.dumps(str(root / 'apps/ipso/static/ipso/pwa/upload.js'))});
      const radius = 6378137; // Exact WGS84 equatorial metric for this fixture.
      const offsets = [11.999, 12, 12.001].map(m => m / radius * 180 / Math.PI);
      process.stdout.write(JSON.stringify({{
        desktop: geo.distanceMeters(0, 0, 0, 1),
        mobile: upload.distanceMeters(0, 0, 0, 1),
        desktopBoundary: offsets.map(lon => geo.distanceMeters(0, 0, 0, lon)),
        mobileBoundary: offsets.map(lon => upload.distanceMeters(0, 0, 0, lon)),
        antipode: geo.distanceMeters(0, 0, 0, 180),
        area: geo.geodesicArea([{{lat:0,lng:0}},{{lat:0,lng:1}},{{lat:1,lng:1}},{{lat:1,lng:0}}]),
      }}));
    '''
    result = subprocess.run(['node', '--input-type=module', '-e', script],
        check=True, text=True, capture_output=True, timeout=10)
    data = json.loads(result.stdout)
    expected = 6378137 * math.pi / 180
    assert data['desktop'] == pytest.approx(expected, abs=1e-8)
    assert data['mobile'] == pytest.approx(expected, abs=1e-8)
    assert data['desktopBoundary'] == pytest.approx([11.999, 12, 12.001], abs=1e-10)
    assert data['mobileBoundary'] == pytest.approx(data['desktopBoundary'], abs=1e-10)
    assert data['antipode'] == pytest.approx(20003931.458625447, abs=1e-6)
    assert data['area'] == pytest.approx(12308778361.469452, abs=.01)


def test_server_matches_shared_reference_vectors():
    references = json.loads((Path(__file__).parent / 'fixtures/wgs84-geodesy.json').read_text())
    for case in references['distances']:
        assert distance_m(*case['points']) == pytest.approx(case['meters'], abs=1e-6)
