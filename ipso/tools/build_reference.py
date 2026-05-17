#!/usr/bin/env python3
"""Generate a reference.json bundle from bosco/data CSVs + abies species list.

Usage:

    laforesta/ipso/tools/build_reference.py <output-path>

Sources:
- ../../bosco/data/particelle.csv         parcels (filtered: high-forest only)
- ../../bosco/data/equazioni_ipsometro.csv ipsometric regression coefficients
- abies SPECIES list (hardcoded below, mirror of
  abies/apps/base/management/commands/import_reference.py lines 35-43)

The Pino species is split into 'Pino Nero' and 'Pino Marittimo' so the auto-h
regression fires the right coefficients per region. The CSV output writes
these names verbatim.
"""

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1

# Mirrored from abies/apps/base/management/commands/import_reference.py,
# with Pino split into two species. (common, latin, sort_order, density q/m3)
SPECIES = [
    ('Abete',          'Abies alba',           10, 4.70),
    ('Castagno',       'Castanea sativa',      20, 5.50),
    ('Douglas',        'Pseudotsuga menziesii', 30, 5.00),
    ('Faggio',         'Fagus sylvatica',      40, 7.00),
    ('Ontano',         'Alnus cordata',        50, 4.60),
    ('Pino Nero',      'Pinus nigra',          60, 4.70),
    ('Pino Marittimo', 'Pinus pinaster',       61, 4.70),
    ('Altro',          '',                    999, 5.00),
]

COPPICE_ECLASS = 'F'      # Comparto F == coppice
HIGH_FOREST_GOVERNO = 'Fustaia'

# Italian-friendly natural sort for particella names like '2a', '10b', '12':
# split into chunks of digits and non-digits, compare numerically where digits.
_NATKEY_RE = re.compile(r'(\d+)')


def natural_key(s: str) -> list:
    parts = _NATKEY_RE.split(s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def load_parcels(path: Path) -> list[dict]:
    """Return high-forest parcels as [{compresa, particella}, ...]."""
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    parcels = []
    for row in rows:
        if row['Governo'] != HIGH_FOREST_GOVERNO:
            continue
        if row['Comparto'] == COPPICE_ECLASS:
            continue
        parcels.append({
            'compresa': row['Compresa'],
            'particella': row['Particella'],
        })
    parcels.sort(key=lambda p: (p['compresa'], natural_key(p['particella'])))
    return parcels


def load_ipsometrica(path: Path) -> dict:
    """Return {compresa: {specie: {a, b}}} from equazioni_ipsometro.csv."""
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    out: dict = {}
    for row in rows:
        if row['funzione'] != 'ln':
            raise SystemExit(
                f"equazioni_ipsometro.csv has unexpected funzione "
                f"{row['funzione']!r}; build_reference only supports 'ln'."
            )
        out.setdefault(row['compresa'], {})[row['genere']] = {
            'a': float(row['a']),
            'b': float(row['b']),
        }
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output-path>", file=sys.stderr)
        return 2
    out_path = Path(sys.argv[1])

    here = Path(__file__).resolve().parent
    ipso_root = here.parent
    repo_root = ipso_root.parent
    bosco_data = repo_root / 'bosco' / 'data'

    particelle = bosco_data / 'particelle.csv'
    equazioni = bosco_data / 'equazioni_ipsometro.csv'
    for p in (particelle, equazioni):
        if not p.is_file():
            print(f"missing source file: {p}", file=sys.stderr)
            return 1

    species = [
        {'common': c, 'latin': l, 'sort_order': s, 'density': d}
        for (c, l, s, d) in SPECIES
    ]
    parcels = load_parcels(particelle)
    ipsometrica = load_ipsometrica(equazioni)

    ref = {
        'schema_version': SCHEMA_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat(
            timespec='seconds'
        ).replace('+00:00', 'Z'),
        'species': species,
        'parcels': parcels,
        'ipsometrica': ipsometrica,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(ref, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write('\n')

    print(
        f"{out_path}: {len(species)} species, {len(parcels)} parcels, "
        f"{sum(len(v) for v in ipsometrica.values())} regression entries "
        f"across {len(ipsometrica)} regions",
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
