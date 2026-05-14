"""Import harvest operations from mannesi.csv.

Clears existing Harvest rows (cascading to junction tables) before
re-importing, so a re-run produces a deterministic result rather than
appending duplicates. Reference data must already be loaded.
"""

import csv
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base.management.commands.import_reference import (
    NOTE_MAP, PRODUCT_MAP,
)
from apps.base.models import (
    Crew, Note, Parcel, Product, Species, Tractor,
)
from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor

# CSV column prefix -> Species.common_name
SPECIES_COL_MAP = {
    'abete': 'Abete',
    'pino': 'Pino',
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


def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s or s == 'nd':
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _pct(s: str) -> int:
    """Parse a percentage field; treat blank/non-numeric as 0."""
    s = s.strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _harvest_volume_m3(quintals: Decimal,
                       species_pcts: list[tuple[Species, int]]) -> Decimal:
    """Compute SUM_over_species(quintals × pct/100 / species.density)."""
    total = Decimal('0')
    for species, pct in species_pcts:
        if species.density and species.density > 0:
            total += quintals * Decimal(pct) / Decimal(100) / species.density
    return total.quantize(Decimal('0.001'))


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
        note_cache = {n.name: n for n in Note.objects.all()}
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
            rows = list(csv.DictReader(f))

        ops_batch: list[Harvest] = []
        species_deferred: list[tuple[int, Species, int]] = []
        tractor_deferred: list[tuple[int, Tractor, int]] = []

        for i, row in enumerate(rows):
            region_name = row['Compresa']
            parcel_name = row['Particella']
            parcel = parcel_cache.get((region_name, parcel_name))
            if parcel is None:
                self.stdout.write(
                    f'Row {i + 1}: unknown parcel '
                    f'{region_name}-{parcel_name}, skipping'
                )
                continue

            crew = crew_cache.get(row['Squadra'])
            if crew is None:
                self.stdout.write(
                    f'Row {i + 1}: unknown crew {row["Squadra"]!r}, skipping'
                )
                continue

            product = product_cache.get(PRODUCT_MAP.get(row['Tipo'], ''))
            if product is None:
                self.stdout.write(
                    f'Row {i + 1}: unknown product {row["Tipo"]!r}, skipping'
                )
                continue

            note_name = NOTE_MAP.get(row.get('Note', '').strip(), '')
            note = note_cache.get(note_name) if note_name else None

            quintals = Decimal(row['Q.li'].strip())

            row_species_pcts: list[tuple[Species, int]] = []
            for col_prefix, common_name in SPECIES_COL_MAP.items():
                pct = _pct(row.get(f'{col_prefix} %', ''))
                if pct > 0:
                    row_species_pcts.append((species_cache[common_name], pct))

            volume_m3 = _harvest_volume_m3(quintals, row_species_pcts)

            op = Harvest(
                date=row['Data'],
                product=product,
                parcel=parcel,
                crew=crew,
                record1=_int_or_none(row['VDP']),
                record2=_int_or_none(row.get('Prot.', '')),
                quintals=quintals,
                volume_m3=volume_m3,
                note=note,
                extra_note=row.get('Altre note', '').strip(),
            )
            op_idx = len(ops_batch)
            ops_batch.append(op)

            for sp, pct in row_species_pcts:
                species_deferred.append((op_idx, sp, pct))

            for col_prefix, (mfr, model) in TRACTOR_COL_MAP.items():
                pct = _pct(row.get(f'{col_prefix} %', ''))
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
