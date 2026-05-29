#!/usr/bin/env python3
"""Generate a test-flavored reference.json from ipso/test/test.geojson.

Used by `make deploy-test` to ship a reference.json whose `parcels`
list matches the polygons in test.geojson, so the operator can pick
the test compresa in the pre-session pulldown and the recording
screen can list the test parcels.

    laforesta/ipso/tools/build_test_reference.py <output-path>

Reads:
    laforesta/ipso/build/reference.json (for species + ipsometrica;
                                          built first by `make build`)
    laforesta/ipso/test/test.geojson    (for compresas + parcels)

Writes:
    <output-path>                       (the synthesized test reference)

Species and hypsometric regressions are preserved verbatim from the
real reference.json — the test compresa is unlikely to have its own
regression, so `recomputeAutoH` will show the "missing regression"
hint and the operator types h manually. That's fine for GPS testing.
"""

import json
import sys
from pathlib import Path


def particella_from_name(name: str) -> str:
    """Match the same split that geo.parcelLabel / ipso.particellaName use."""
    dash = name.find('-')
    return name[dash + 1:] if dash >= 0 else name


def main():
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <output-path>', file=sys.stderr)
        sys.exit(2)
    out_path = Path(sys.argv[1])
    here = Path(__file__).resolve().parent.parent  # ipso/

    with (here / 'build' / 'reference.json').open(encoding='utf-8') as f:
        ref = json.load(f)
    with (here / 'test' / 'test.geojson').open(encoding='utf-8') as f:
        gj = json.load(f)

    parcels = []
    seen = set()
    for feat in gj.get('features', []):
        props = feat.get('properties', {}) or {}
        compresa = props.get('layer', '') or ''
        particella = particella_from_name(props.get('name', '') or '')
        if not compresa or not particella:
            continue
        key = (compresa, particella)
        if key in seen:
            continue
        seen.add(key)
        parcels.append({'compresa': compresa, 'particella': particella})

    out = {
        'schema_version': ref.get('schema_version', 1),
        'species': ref.get('species', []),
        'parcels': parcels,
        'ipsometrica': ref.get('ipsometrica', {}),
    }
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'wrote {out_path} ({len(parcels)} parcels)', file=sys.stderr)


if __name__ == '__main__':
    main()
