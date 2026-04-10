#!/usr/bin/env python3
"""Import reference entities (regions, eclasses, species, tractors, crews,
optypes, notes) into the database.

Idempotent: safe to re-run.  Uses get_or_create throughout.
"""

import csv
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Django bootstrap
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django  # noqa: E402
django.setup()

from apps.base.models import Crew, Eclass, Note, Optype, Region, Species, Tractor  # noqa: E402

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------

BOSCO_DATA = Path(__file__).resolve().parent.parent.parent / 'bosco' / 'data'
MANNESI_CSV = BOSCO_DATA / 'mannesi.csv'

# ---------------------------------------------------------------------------
# Static mappings
# ---------------------------------------------------------------------------

REGIONS = ['Capistrano', 'Fabrizia', 'Serra']

# Comparto → (name, coppice).  min_harvest_volume left at 0 for now.
ECLASSES = [
    ('A', False),
    ('B', False),
    ('C', False),
    ('D', False),
    ('E', False),
    ('F', True),
]

# (common_name, latin_name, sort_order) — 'Altro' sorts last.
SPECIES = [
    ('Abete', 'Abies alba', 10),
    ('Castagno', 'Castanea sativa', 20),
    ('Douglas', 'Pseudotsuga menziesii', 30),
    ('Faggio', 'Fagus sylvatica', 40),
    ('Ontano', 'Alnus cordata', 50),
    ('Pino', 'Pinus nigra', 60),
    ('Altro', '', 999),
]

# (manufacturer, model, year)
TRACTORS = [
    ('Equus', '175N UN', None),
    ('Fiat', '110-90', None),
    ('Fiat', '80-66', None),
    ('Landini', '135', None),
    ('New Holland', 'T5050', None),
]

# CSV optype name → canonical name
OPTYPE_MAP = {
    'Tronchi': 'Tronchi',
    'Cippato': 'Cippato',
    'Ramaglia': 'Ramaglia',
    'Pertiche-puntelli-tronchi castagno venduti': 'Pertiche-Puntelli',
    'Pertiche-tronchi castagno': 'Pertiche-Tronchi',
}

# CSV note name → canonical name
NOTE_MAP = {
    'PSR': 'PSR',
    'fitosanitario': 'Fitosanitario',
    'catastrofato': 'Catastrofate',
}

# ---------------------------------------------------------------------------
# Import functions
# ---------------------------------------------------------------------------

def import_regions():
    for name in REGIONS:
        Region.objects.get_or_create(name=name)
    print(f'Regions: {Region.objects.count()}')


def import_eclasses():
    for name, coppice in ECLASSES:
        Eclass.objects.get_or_create(name=name, defaults={'coppice': coppice})
    print(f'Eclasses: {Eclass.objects.count()}')


def import_species():
    for common, latin, order in SPECIES:
        obj, created = Species.objects.get_or_create(
            common_name=common,
            defaults={'latin_name': latin, 'sort_order': order},
        )
        if not created and obj.sort_order != order:
            obj.sort_order = order
            obj.save(update_fields=['sort_order'])
    print(f'Species: {Species.objects.count()}')


def import_tractors():
    for mfr, model, year in TRACTORS:
        Tractor.objects.get_or_create(
            manufacturer=mfr, model=model, defaults={'year': year},
        )
    print(f'Tractors: {Tractor.objects.count()}')


def import_crews():
    """Extract unique crew names from mannesi.csv."""
    with open(MANNESI_CSV, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        names = sorted({row['Squadra'] for row in reader if row['Squadra']})
    for name in names:
        Crew.objects.get_or_create(name=name)
    print(f'Crews: {Crew.objects.count()}')


def import_optypes():
    for canonical in OPTYPE_MAP.values():
        Optype.objects.get_or_create(name=canonical)
    print(f'Optypes: {Optype.objects.count()}')


def import_notes():
    for canonical in NOTE_MAP.values():
        Note.objects.get_or_create(name=canonical)
    print(f'Notes: {Note.objects.count()}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    import_regions()
    import_eclasses()
    import_species()
    import_tractors()
    import_crews()
    import_optypes()
    import_notes()
    print('Reference import complete.')


if __name__ == '__main__':
    run()
