"""The actual DB coefficients are the only species data the preview needs."""
import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from apps.base.equations import estimate_volume_and_mass

pytestmark = [pytest.mark.django_db, pytest.mark.skipif(shutil.which('node') is None, reason='node required')]


@pytest.mark.parametrize('proxy', [False, True])
def test_preview_and_server_receive_current_db_coefficients(species, proxy):
    root = species[0]
    equation = root.volume_equation
    equation.b0, equation.b1, equation.b2 = Decimal('1000'), Decimal('0'), None
    equation.save()
    root.common_name = 'User-renamed taxon'
    root.save()
    selected = species[4] if proxy else root
    selected.density = Decimal('7.25')
    selected.save()
    module = Path('apps/base/static/base/js/volume.js').resolve().as_uri()
    script = f'''
      import {{volumeFromCoefficientsM3, massQ}} from {json.dumps(module)};
      const volume = Number(volumeFromCoefficientsM3(30, 20, {selected.volume_coefficients_json}).toFixed(4));
      process.stdout.write(JSON.stringify([volume, massQ(volume, {float(selected.density)})]));
    '''
    result = subprocess.run(['node', '--input-type=module', '-e', script],
        check=True, text=True, capture_output=True, timeout=10)
    preview = tuple(Decimal(str(value)) for value in json.loads(result.stdout))
    assert preview == estimate_volume_and_mass(30, 20, selected) == (Decimal('1'), Decimal('7.25'))
