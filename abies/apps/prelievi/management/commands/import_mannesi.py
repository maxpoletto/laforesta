"""Import harvest operations from mannesi.csv.

Clears existing Harvest rows (cascading to junction tables) before
re-importing, so a re-run produces a deterministic result rather than
appending duplicates. Reference data must already be loaded.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base import csv_io
from apps.base.management.commands.import_reference import (
    NOTE_FLAG_MAP, PRODUCT_MAP,
)
from apps.base.models import (
    Crew, Parcel, Product, Species, Tractor,
)
from apps.prelievi.models import (
    Harvest, HarvestSpecies, HarvestTractor, harvest_volume_m3,
)
from config import strings as S

# CSV column prefix -> Species.common_name
SPECIES_COL_MAP = {
    'abete': 'Abete',
    'pino': 'Pino Laricio',
    'douglas': 'Douglas',
    'faggio': 'Faggio',
    'castagno': 'Castagno',
    'ontano': 'Ontano',
    'altro': 'Altro',
}

# CSV column prefix -> (Tractor.manufacturer, Tractor.model)
TRACTOR_COL_MAP = {
    'Equus': ('Equus', '175N UN'),
    'Fiat 110-90': ('Fiat', '110-90'),
    'Fiat 80-66': ('Fiat', '80-66'),
    'Landini 135': ('Landini', '135'),
    'New Holland T5050': ('New Holland', 'T5050'),
}

BATCH_SIZE = 500


def _pct(reader: csv_io.CsvReader, s: str) -> int:
    """Percentage cell → int; 0 for a blank/invalid column (i.e. 0%).  `percent`
    is an integer field, so a fractional value is rejected (→0), not truncated."""
    return reader.integer(s) or 0


class Command(BaseCommand):
    help = "Import harvest operations from mannesi.csv in <data_dir>."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help="Directory containing mannesi.csv.",
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        mannesi_csv = data_dir / 'mannesi.csv'
        if not mannesi_csv.is_file():
            raise CommandError(f'{mannesi_csv} not found')

        # Build FK lookup caches.
        crew_cache = {c.name: c for c in Crew.objects.all()}
        product_cache = {p.name: p for p in Product.objects.all()}
        species_cache = {s.common_name: s for s in Species.objects.all()}
        tractor_cache = {
            (t.manufacturer, t.model): t for t in Tractor.objects.all()
        }
        parcel_cache = {
            (p.region.name, p.name): p
            for p in Parcel.objects.select_related('region')
        }

        if not crew_cache or not parcel_cache:
            raise CommandError(
                'Reference and parcel data must be loaded first. '
                'Run import_reference and import_parcels before import_mannesi.'
            )

        with open(mannesi_csv, encoding='utf-8-sig') as f:
            reader = csv_io.read(f.read())

        ops_batch: list[Harvest] = []
        species_deferred: list[tuple[int, Species, int]] = []
        tractor_deferred: list[tuple[int, Tractor, int]] = []

        for i, row in enumerate(reader):
            region_name = row[S.CSV_COL_COMPRESA]
            parcel_name = row[S.CSV_COL_PARTICELLA]
            parcel = parcel_cache.get((region_name, parcel_name))
            if parcel is None:
                self.stdout.write(
                    f'Row {i + 1}: unknown parcel '
                    f'{region_name}-{parcel_name}, skipping'
                )
                continue

            crew = crew_cache.get(row[S.CSV_COL_CREW])
            if crew is None:
                self.stdout.write(
                    f'Row {i + 1}: unknown crew '
                    f'{row[S.CSV_COL_CREW]!r}, skipping'
                )
                continue

            product = product_cache.get(PRODUCT_MAP.get(row[S.CSV_COL_PRODUCT], ''))
            if product is None:
                self.stdout.write(
                    f'Row {i + 1}: unknown product '
                    f'{row[S.CSV_COL_PRODUCT]!r}, skipping'
                )
                continue

            damaged, unhealthy, psr = NOTE_FLAG_MAP.get(
                row.get(S.CSV_COL_NOTE, '').strip(),
                (False, False, False),
            )

            mass_q = reader.decimal(row[S.CSV_COL_QUINTALS])

            row_species_pcts: list[tuple[Species, int]] = []
            for col_prefix, common_name in SPECIES_COL_MAP.items():
                pct = _pct(reader, row.get(f'{col_prefix} %', ''))
                if pct > 0:
                    row_species_pcts.append((species_cache[common_name], pct))

            volume_m3 = harvest_volume_m3(
                mass_q, ((sp.density, pct) for sp, pct in row_species_pcts),
            )

            op = Harvest(
                date=row[S.CSV_COL_DATA],
                product=product,
                parcel=parcel,
                crew=crew,
                record1=reader.integer(row[S.CSV_COL_VDP]),
                record2=reader.integer(row.get(S.CSV_COL_PROT, '')),
                mass_q=mass_q,
                volume_m3=volume_m3,
                damaged=damaged,
                unhealthy=unhealthy,
                psr=psr,
                note=row.get(S.CSV_COL_EXTRA_NOTE, '').strip(),
            )
            op_idx = len(ops_batch)
            ops_batch.append(op)

            for sp, pct in row_species_pcts:
                species_deferred.append((op_idx, sp, pct))

            for col_prefix, (mfr, model) in TRACTOR_COL_MAP.items():
                pct = _pct(reader, row.get(f'{col_prefix} %', ''))
                if pct > 0:
                    tractor_deferred.append(
                        (op_idx, tractor_cache[(mfr, model)], pct),
                    )

        # Cascade-deletes the junction rows.
        deleted, _ = Harvest.objects.all().delete()
        if deleted:
            self.stdout.write(f'Cleared {deleted} existing records')

        Harvest.objects.bulk_create(ops_batch, batch_size=BATCH_SIZE)
        self.stdout.write(f'Harvests: {len(ops_batch)} created')

        species_records = [
            HarvestSpecies(harvest=ops_batch[idx], species=sp, percent=pct)
            for idx, sp, pct in species_deferred
        ]
        HarvestSpecies.objects.bulk_create(species_records, batch_size=BATCH_SIZE)
        self.stdout.write(f'HarvestSpecies: {len(species_records)} created')

        tractor_records = [
            HarvestTractor(harvest=ops_batch[idx], tractor=tr, percent=pct)
            for idx, tr, pct in tractor_deferred
        ]
        HarvestTractor.objects.bulk_create(tractor_records, batch_size=BATCH_SIZE)
        self.stdout.write(f'HarvestTractors: {len(tractor_records)} created')

        from apps.base.digests import mark_all_stale
        mark_all_stale()

        self.stdout.write('Mannesi import complete.')
