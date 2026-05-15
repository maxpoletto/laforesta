#!/usr/bin/env python3
"""Generate ipso/geo.js from abies/apps/base/static/base/js/geo.js.

Run from anywhere; paths are resolved relative to this script's location:

    laforesta/ipso/tools/vendor_geo.py

reads

    laforesta/abies/apps/base/static/base/js/geo.js

writes

    laforesta/ipso/geo.js

Transforms:
- Strip `import` lines (the abies file imports MapCommon for grid-planner
  helpers that ipso never calls; the references inside those functions
  are inert until call time, so leaving the function bodies is safe).
- Strip the `export` keyword from top-level declarations so ipso can load
  the file as a classic <script>.
- Append a CommonJS guard so `node tests.js` can require() the file.

abies remains the single source of truth for these geometry helpers.
"""

import re
import sys
from pathlib import Path

EXPORTS = [
    'pointInRing',
    'pointInPolygon',
    'findContainingParcel',
    'parcelLabel',
    'metersToDegLat',
    'metersToDegLng',
    'featureBbox',
    'buildBboxIndex',
    'distanceToBoundaryMeters',
]


def main():
    here = Path(__file__).resolve().parent
    src = here.parent.parent / 'abies' / 'apps' / 'base' / 'static' / 'base' / 'js' / 'geo.js'
    dst = here.parent / 'geo.js'

    text = src.read_text(encoding='utf-8')

    out_lines = []
    for line in text.splitlines():
        if re.match(r'^\s*import\s', line):
            continue
        out_lines.append(re.sub(r'^(\s*)export\s+', r'\1', line))

    body = '\n'.join(out_lines).rstrip() + '\n'

    guard = (
        '\n'
        '// CommonJS guard so node tests.js can require() this vendored file.\n'
        'if (typeof module !== "undefined") {\n'
        '  module.exports = {\n'
    )
    for name in EXPORTS:
        guard += f'    {name},\n'
    guard += '  };\n}\n'

    dst.write_text(body + guard, encoding='utf-8')
    print(f'vendored {src.name} → {dst} ({dst.stat().st_size} bytes)',
          file=sys.stderr)


if __name__ == '__main__':
    main()
