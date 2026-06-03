"""Import preserved trees (PAI) into the Tree table with preserved=true.

Reads bosco/data/piante-accrescimento-indefinito.csv (header:
Compresa,Particella,Numero,Genere,Diametro,Altezza,UTMLon,UTMLat,
GaussLon,GaussLat,Lon,Lat,Note).

PAI rows are individual specimens, not sample-area-related, so we
do not create TreeSample rows.  Each PAI gets one Tree row with
`preserved=True` and `coppice=False`.

Species mapping: the PAI file uses richer species names than our
M0 seed Species set; species that don't have a direct match fall
back to the catch-all 'Altro'.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.base import csv_io
from apps.base.numparse import coord_float
from apps.base.models import Parcel, Species, Tree
from config import strings as S

# PAI CSV Genere → Species.common_name.  Falls back to 'Altro' for
# species not in our seed table (less-common species in the PDG forest).
GENERE_MAP = {
    'Abete Bianco': 'Abete',
    'Abete Rosso': 'Altro',
    'Betulla Bianca': 'Altro',
    'Castagno': 'Castagno',
    'Ciliegio': 'Altro',
    'Douglas': 'Douglas',
    'Faggio': 'Faggio',
    'Farnia': 'Altro',
    'Larice': 'Altro',
    'Noce': 'Altro',
    'Ontano': 'Ontano',
    'Pino': 'Pino Laricio',
    'Pino Laricio': 'Pino Laricio',
    'Pino Marittimo': 'Pino Marittimo',
    'Pino Nero': 'Pino Nero',
    'Pino Strobo': 'Pino Strobo',
    'Pioppo Tremulo': 'Altro',
    'Tasso': 'Altro',
    'Tiglio': 'Altro',
}


class Command(BaseCommand):
    help = ("Import preserved trees (PAI) from "
            "piante-accrescimento-indefinito.csv.")

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help='Directory containing piante-accrescimento-indefinito.csv.',
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        csv_path = data_dir / 'piante-accrescimento-indefinito.csv'
        if not csv_path.is_file():
            raise CommandError(f'{csv_path} not found')

        species_cache = {s.common_name: s for s in Species.objects.all()}
        parcel_cache = {
            (p.region.name, p.name): p
            for p in Parcel.objects.select_related('region')
        }
        if not parcel_cache:
            raise CommandError(
                'Reference + parcel data must be loaded first.'
            )

        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv_io.read(f.read())

        with transaction.atomic():
            # Idempotent: clear existing PAI rows before reimporting.
            deleted, _ = Tree.objects.filter(preserved=True).delete()
            if deleted:
                self.stdout.write(f'Cleared {deleted} prior PAI trees')

            n_created = 0
            n_skipped = 0
            for i, row in enumerate(reader, 1):
                parcel = parcel_cache.get(
                    (row[S.CSV_COL_COMPRESA], row[S.CSV_COL_PARTICELLA])
                )
                if parcel is None:
                    n_skipped += 1
                    continue

                mapped = GENERE_MAP.get(row[S.CSV_COL_GENERE].strip(), 'Altro')
                species = species_cache.get(mapped)
                if species is None:
                    n_skipped += 1
                    continue

                Tree.objects.create(
                    species=species,
                    parcel=parcel,
                    lat=coord_float(reader.decimal(row[S.CSV_COL_LAT])),
                    lon=coord_float(reader.decimal(row[S.CSV_COL_LON])),
                    preserved=True,
                    coppice=False,
                )
                n_created += 1

        from apps.base.digests import mark_all_stale
        mark_all_stale()

        self.stdout.write(
            f'PAI Trees: {n_created} created, {n_skipped} skipped'
        )
