"""Import reference entities (regions, eclasses, species, tractors, crews,
optypes, notes) from CSV files in <data_dir>.

Idempotent: safe to re-run. Uses get_or_create throughout. Crew names are
extracted from mannesi.csv (the only CSV that lists them), so this command
expects mannesi.csv to live in <data_dir>.
"""

import csv
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base.models import (
    Crew, Eclass, Note, Optype, Region, Species, Tractor,
)

# --- Static reference data --------------------------------------------------

REGIONS = ['Capistrano', 'Fabrizia', 'Serra']

# (name, coppice). min_harvest_volume left at 0 for now.
ECLASSES = [
    ('A', False),
    ('B', False),
    ('C', False),
    ('D', False),
    ('E', False),
    ('F', True),
]

# (common_name, latin_name, sort_order). 'Altro' sorts last.
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

# CSV optype name -> canonical name. Imported by import_mannesi.
OPTYPE_MAP = {
    'Tronchi': 'Tronchi',
    'Cippato': 'Cippato',
    'Ramaglia': 'Ramaglia',
    'Pertiche-puntelli-tronchi castagno venduti': 'Pertiche-Puntelli',
    'Pertiche-tronchi castagno': 'Pertiche-Tronchi',
}

# CSV note name -> canonical name. Imported by import_mannesi.
NOTE_MAP = {
    'PSR': 'PSR',
    'fitosanitario': 'Fitosanitario',
    'catastrofato': 'Catastrofate',
}


class Command(BaseCommand):
    help = "Import reference entities from CSV files in <data_dir>."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help="Directory containing mannesi.csv (for crew names).",
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        mannesi_csv = data_dir / 'mannesi.csv'
        if not mannesi_csv.is_file():
            raise CommandError(f'{mannesi_csv} not found')

        self._import_regions()
        self._import_eclasses()
        self._import_species()
        self._import_tractors()
        self._import_crews(mannesi_csv)
        self._import_optypes()
        self._import_notes()
        self.stdout.write('Reference import complete.')

    def _import_regions(self):
        for name in REGIONS:
            Region.objects.get_or_create(name=name)
        self.stdout.write(f'Regions: {Region.objects.count()}')

    def _import_eclasses(self):
        for name, coppice in ECLASSES:
            Eclass.objects.get_or_create(name=name, defaults={'coppice': coppice})
        self.stdout.write(f'Eclasses: {Eclass.objects.count()}')

    def _import_species(self):
        for common, latin, order in SPECIES:
            obj, created = Species.objects.get_or_create(
                common_name=common,
                defaults={'latin_name': latin, 'sort_order': order},
            )
            if not created and obj.sort_order != order:
                obj.sort_order = order
                obj.save(update_fields=['sort_order'])
        self.stdout.write(f'Species: {Species.objects.count()}')

    def _import_tractors(self):
        for mfr, model, year in TRACTORS:
            Tractor.objects.get_or_create(
                manufacturer=mfr, model=model, defaults={'year': year},
            )
        self.stdout.write(f'Tractors: {Tractor.objects.count()}')

    def _import_crews(self, mannesi_csv: Path):
        with open(mannesi_csv, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            names = sorted({row['Squadra'] for row in reader if row['Squadra']})
        for name in names:
            Crew.objects.get_or_create(name=name)
        self.stdout.write(f'Crews: {Crew.objects.count()}')

    def _import_optypes(self):
        for canonical in OPTYPE_MAP.values():
            Optype.objects.get_or_create(name=canonical)
        self.stdout.write(f'Optypes: {Optype.objects.count()}')

    def _import_notes(self):
        for canonical in NOTE_MAP.values():
            Note.objects.get_or_create(name=canonical)
        self.stdout.write(f'Notes: {Note.objects.count()}')
