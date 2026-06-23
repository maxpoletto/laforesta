"""Shared reference data used by the canonical import path.

These tables and the in-repo species defaults have a single home so bootstrap
and the sampled-trees core share the same canonical aliases/defaults:

- ``GENERE_MAP``  — CSV ``Genere`` → ``Species.common_name`` aliasing (sampled
  trees), to absorb minor naming drift between the survey files and Abies.
- ``SPECIES_CSV`` / ``load_species()`` — the in-repo canonical species list
  (``apps/base/data/species.csv``), seeded as the default when a data dir omits
  ``species.csv``.
- ``PRODUCT_MAP`` — source product name → canonical name; its distinct values
  are the default product set.
"""

from decimal import Decimal
from pathlib import Path

from apps.base import csv_io
from config.constants import PRESSLER_DEFAULT, is_truthy

# --- species ----------------------------------------------------------------

# Canonical species list lives at apps/base/data/species.csv. It seeds fresh
# databases and feeds the Abies-served Ipso reference bundle. Density values
# are fresh-cut wood — the truck-scale weight basis for the harvest record.
# Admins refine via Impostazioni → Specie.
SPECIES_CSV = Path(__file__).resolve().parent / 'data' / 'species.csv'


def load_species() -> list[
    tuple[str, str, int | None, Decimal | None, Decimal | None, bool]
]:
    """Return the in-repo species list as ``(common, latin, sort_order,
    density, pressler_default, minor)`` tuples."""
    with SPECIES_CSV.open(encoding='utf-8') as f:
        reader = csv_io.read(f.read())
    return [
        (row['common'], row['latin'], reader.integer(row['sort_order']),
         reader.decimal(row['density_q_m3']),
         reader.decimal(row.get('pressler_default')) or PRESSLER_DEFAULT,
         is_truthy(row['minor']))
        for row in reader
    ]


# --- sampled-trees species aliasing -----------------------------------------

# CSV Genere → Species.common_name.  Most species match by name; a couple of
# synonyms are mapped explicitly so the import is robust to minor naming drift
# between the survey files (pdg-2026 lineage) and Abies.
GENERE_MAP = {
    'Abete': 'Abete',
    'Castagno': 'Castagno',
    'Douglas': 'Douglas',
    'Faggio': 'Faggio',
    'Ontano': 'Ontano',
    'Pino': 'Pino Laricio',
    'Pino Laricio': 'Pino Laricio',
    'Pino Marittimo': 'Pino Marittimo',
    'Pino Nero': 'Pino Nero',
}


# --- products ----------------------------------------------------------------

# Legacy CSV product name → canonical name.  The distinct values are the default
# product set seeded when a data dir omits products.csv.
PRODUCT_MAP = {
    'Tronchi': 'Tronchi',
    'Cippato': 'Cippato',
    'Ramaglia': 'Ramaglia',
    'Pertiche-puntelli-tronchi castagno venduti': 'Pertiche-Puntelli',
    'Pertiche-tronchi castagno': 'Pertiche-Tronchi',
}
