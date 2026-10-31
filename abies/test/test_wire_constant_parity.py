"""Value parity for the intentionally shared desktop/mobile/server wire subset."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from config import constants

# Explicit membership: UI labels, client preferences, and typed defaults are
# deliberately outside this wire-identifier contract.
SHARED_WIRE_NAMES = """
FIELD_ACC_M FIELD_CATEGORIES FIELD_CATEGORY_IDS FIELD_CLIENT_PHOTO_ID
FIELD_CONTENT_TYPE FIELD_COPPICE FIELD_CREATED_AT FIELD_DATE
FIELD_D_CM FIELD_ESTIMATED_BIRTH_YEAR FIELD_HEIGHT_PX FIELD_H_M
FIELD_ID FIELD_L10_MM FIELD_LAT FIELD_LON
FIELD_MAX_BYTES FIELD_MAX_TREE_NUMBER FIELD_MODE FIELD_NOTE
FIELD_NUMBER FIELD_OPERATOR FIELD_ORIGINAL_FILENAME FIELD_PARCEL_ID
FIELD_PHOTOS FIELD_PRESERVED FIELD_PRESSLER_COEFF FIELD_REFERENCE_VERSION
FIELD_REGION_ID FIELD_R_M FIELD_SAMPLE_AREA_ID FIELD_SAMPLE_GRID_ID
FIELD_SELECTABLE FIELD_SESSION_ID FIELD_SHOOT FIELD_SIZE_BYTES
FIELD_SPECIES FIELD_SPECIES_ID FIELD_STANDARD FIELD_SURVEY_ID
FIELD_TAKEN_AT FIELD_TEXT FIELD_TREES FIELD_WIDTH_PX
FIELD_YEAR IPSO_MODE_FREE_SURVEY IPSO_MODE_MARTELLATE IPSO_MODE_OBSERVATIONS
IPSO_MODE_SAMPLES IPSO_REF_HYPSOMETRY IPSO_REF_HYPSOMETRY_IDS IPSO_REF_OBSERVATION_CATEGORIES
IPSO_REF_PAI IPSO_REF_PARCELS IPSO_REF_SAMPLE_AREA_MAX_NUMBERS IPSO_REF_SAMPLING
IPSO_REF_SPECIES IPSO_REF_UPLOAD
""".split()


def test_shared_wire_values_match_three_runtimes():
    if shutil.which('node') is None:
        pytest.skip('node required')
    root = Path(__file__).resolve().parents[1]
    desktop = (root / 'apps/base/static/base/js/constants.js').as_uri()
    mobile = str(root / 'apps/ipso/static/ipso/pwa/constants.js')
    script = f"""
      import * as desktop from {json.dumps(desktop)};
      import {{createRequire}} from 'node:module';
      const mobile = createRequire(import.meta.url)({json.dumps(mobile)});
      process.stdout.write(JSON.stringify({{desktop, mobile}}));
    """
    result = subprocess.run(['node', '--input-type=module', '-e', script],
        check=True, capture_output=True, text=True, timeout=10)
    actual = json.loads(result.stdout)
    for name in SHARED_WIRE_NAMES:
        assert name in actual['desktop'] and name in actual['mobile'], name
        assert actual['desktop'][name] == actual['mobile'][name] == getattr(constants, name), name
