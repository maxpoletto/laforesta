"""Import reference entities (regions, eclasses, species, tractors, crews,
products, notes) from CSV files in <data_dir>.

Idempotent: safe to re-run. Uses get_or_create throughout. Crew names are
extracted from mannesi.csv (the only CSV that lists them), so this command
expects mannesi.csv to live in <data_dir>.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from config import strings as S
from config.constants import is_truthy

from apps.base import csv_io
from apps.base.models import (
    Crew, Eclass, Product, Region, Species, Tractor,
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

# Canonical species list lives at apps/base/data/species.csv. Both abies
# (this command) and ipso (tools/build_reference.py) read it so the two
# apps stay in sync. Density values are fresh-cut wood — the truck-scale
# weight basis for the harvest record. Admins refine via Settings → Trees.
SPECIES_CSV = Path(__file__).resolve().parent.parent.parent / 'data' / 'species.csv'


def load_species():
    with SPECIES_CSV.open(encoding='utf-8') as f:
        reader = csv_io.read(f.read())
    return [
        (row['common'], row['latin'], reader.integer(row['sort_order']),
         reader.decimal(row['density_q_m3']), is_truthy(row['minor']))
        for row in reader
    ]


SPECIES = load_species()

# (manufacturer, model, year)
TRACTORS = [
    ('Equus', '175N UN', None),
    ('Fiat', '110-90', None),
    ('Fiat', '80-66', None),
    ('Landini', '135', None),
    ('New Holland', 'T5050', None),
]

# CSV product name -> canonical name. Imported by import_mannesi.
PRODUCT_MAP = {
    'Tronchi': 'Tronchi',
    'Cippato': 'Cippato',
    'Ramaglia': 'Ramaglia',
    'Pertiche-puntelli-tronchi castagno venduti': 'Pertiche-Puntelli',
    'Pertiche-tronchi castagno': 'Pertiche-Tronchi',
}

# CSV note value -> (damaged, unhealthy, psr) booleans on Harvest /
# HarvestPlanItem.  Imported by import_mannesi.  These three booleans
# replace the legacy `note` FK to the (now-removed) Note model.
NOTE_FLAG_MAP = {
    'PSR':            (False, False, True),
    'fitosanitario':  (False, True,  False),
    'catastrofato':   (True,  False, False),
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
        self._import_products()

        from apps.base.digests import mark_all_stale
        mark_all_stale()

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
        for common, latin, order, density, minor in SPECIES:
            obj, created = Species.objects.get_or_create(
                common_name=common,
                defaults={
                    'latin_name': latin, 'sort_order': order,
                    'density': density, 'minor': minor,
                },
            )
            if not created:
                update_fields = []
                if obj.sort_order != order:
                    obj.sort_order = order
                    update_fields.append('sort_order')
                if obj.density != density:
                    obj.density = density
                    update_fields.append('density')
                if obj.minor != minor:
                    obj.minor = minor
                    update_fields.append('minor')
                if update_fields:
                    obj.save(update_fields=update_fields)
        self.stdout.write(f'Species: {Species.objects.count()}')

    def _import_tractors(self):
        for mfr, model, year in TRACTORS:
            Tractor.objects.get_or_create(
                manufacturer=mfr, model=model, defaults={'year': year},
            )
        self.stdout.write(f'Tractors: {Tractor.objects.count()}')

    def _import_crews(self, mannesi_csv: Path):
        with open(mannesi_csv, encoding='utf-8-sig') as f:
            reader = csv_io.read(f.read())
            names = sorted({row[S.CSV_COL_CREW] for row in reader
                            if row[S.CSV_COL_CREW]})
        for name in names:
            Crew.objects.get_or_create(name=name)
        self.stdout.write(f'Crews: {Crew.objects.count()}')

    def _import_products(self):
        for canonical in PRODUCT_MAP.values():
            Product.objects.get_or_create(name=canonical)
        self.stdout.write(f'Products: {Product.objects.count()}')
