"""Import sampled-tree measurements into Survey + Sample + Tree + TreeSample.

Reads bosco/data/alberi-calcolati.csv (header:
Compresa,Particella,Area saggio,n,poll,D(cm),Classe diametrica,h(m),
Genere,Fustaia,G(m2),V(m3),L10(mm),c,Pv(%/a),PvxV(m3/a),IP).

All 6106 rows in the current snapshot are Fustaia=True.  The file is
post-processed (V is already computed) but we recompute V via our
vendored Tabacchi tables to confirm parity and to support future CSVs
without a V column.

Behavior:
- Pre-creates one Survey named "Campagna 2024-2025" against the
  "Aree di saggio PDG 2026" grid (run import_sample_grid first).
- Default sample date is 2024-09-15 (placeholder; admins edit later
  via the campionamenti Section 3 inline date — see
  docs/pages/campionamenti.md).
- Groups rows by (Compresa, Particella, Area saggio) → one Sample.
- Each row creates one Tree (preserved=False, coppice=False) and one
  TreeSample.  V via Tabacchi, m = V × species.density.
- Skips rows whose Genere isn't in either Tabacchi or our Species
  table; logs counts.
- Idempotent: re-running truncates Sample/TreeSample/Tree (cascading
  via FK) for this Survey before reimporting.
"""

import csv
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.base.models import (
    Parcel, SampleArea, Sample, SampleGrid, Species, Survey, Tree, TreeSample,
)
from apps.base.tabacchi import has_species, tabacchi_volume_m3
from config import strings as S

SURVEY_NAME = 'Campagna 2024-2025'
SURVEY_DESC = ('Misure di alberi nelle aree di saggio PDG 2026, '
               'caricate dal file alberi-calcolati.csv.')
GRID_NAME = 'Aree di saggio PDG 2026'  # must match import_sample_grid
DEFAULT_SAMPLE_DATE = date(2024, 9, 15)

# CSV Genere → Species.common_name.  Most species match by name; a
# couple of synonyms are mapped explicitly so the ETL is robust to
# minor naming drift between pdg-2026 and Abies.
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


def _int_or_none(s: str) -> int | None:
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _bool(s: str) -> bool:
    return s.strip().lower() in ('true', '1', 'yes', 'si', 'sì')


class Command(BaseCommand):
    help = "Import tree samples from alberi-calcolati.csv."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help='Directory containing alberi-calcolati.csv.',
        )
        parser.add_argument(
            '--date', default=DEFAULT_SAMPLE_DATE.isoformat(),
            help=f'Sample date for all rows (default: {DEFAULT_SAMPLE_DATE}).',
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        csv_path = data_dir / 'alberi-calcolati.csv'
        if not csv_path.is_file():
            raise CommandError(f'{csv_path} not found')

        sample_date = date.fromisoformat(options['date'])

        try:
            grid = SampleGrid.objects.get(name=GRID_NAME)
        except SampleGrid.DoesNotExist:
            raise CommandError(
                f'Grid {GRID_NAME!r} not found. Run import_sample_grid first.'
            )

        species_cache = {s.common_name: s for s in Species.objects.all()}
        parcel_cache = {
            (p.region.name, p.name): p
            for p in Parcel.objects.select_related('region')
        }
        sample_area_cache = {
            (sa.parcel.region.name, sa.parcel.name, sa.number): sa
            for sa in SampleArea.objects.filter(sample_grid=grid)
                          .select_related('parcel__region')
        }

        with open(csv_path, encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))

        n_skipped_parcel = 0
        n_skipped_area = 0
        n_skipped_species = 0
        n_trees = 0
        n_samples = 0

        with transaction.atomic():
            survey, created = Survey.objects.get_or_create(
                name=SURVEY_NAME,
                defaults={
                    'sample_grid': grid,
                    'description': SURVEY_DESC,
                },
            )
            if not created:
                # Idempotent reimport: clear existing samples + their
                # children for this survey.  Trees that are otherwise
                # unreferenced are not auto-purged here; rare but
                # acceptable for a dev-only flow.
                deleted, _ = Sample.objects.filter(survey=survey).delete()
                if deleted:
                    self.stdout.write(
                        f'Cleared {deleted} prior samples for re-import'
                    )

            # First pass: group rows by (region, parcel, area) → Sample.
            samples_by_key: dict[tuple, Sample] = {}
            for row in rows:
                key = (row[S.CSV_COL_COMPRESA], row[S.CSV_COL_PARTICELLA],
                       row[S.CSV_COL_AREA_SAGGIO].strip())
                if key in samples_by_key:
                    continue
                sa = sample_area_cache.get(key)
                if sa is None:
                    continue  # logged later
                samples_by_key[key] = Sample.objects.create(
                    sample_area=sa, survey=survey, date=sample_date,
                )
                n_samples += 1

            # Second pass: create Tree + TreeSample rows.
            for i, row in enumerate(rows, 1):
                key = (row[S.CSV_COL_COMPRESA], row[S.CSV_COL_PARTICELLA],
                       row[S.CSV_COL_AREA_SAGGIO].strip())
                parcel = parcel_cache.get(
                    (row[S.CSV_COL_COMPRESA], row[S.CSV_COL_PARTICELLA]))
                if parcel is None:
                    n_skipped_parcel += 1
                    continue
                sample = samples_by_key.get(key)
                if sample is None:
                    n_skipped_area += 1
                    continue

                genere = row[S.CSV_COL_GENERE].strip()
                mapped = GENERE_MAP.get(genere)
                species = species_cache.get(mapped) if mapped else None
                if species is None:
                    n_skipped_species += 1
                    continue

                d_cm = int(float(row[S.CSV_COL_D_CM_LEGACY]))
                h_m = Decimal(row[S.CSV_COL_H_M_LEGACY]).quantize(
                    Decimal('0.01'), rounding=ROUND_HALF_UP)
                l10_mm = _int_or_none(row[S.CSV_COL_L10_MM_LEGACY]) or 0
                fustaia = _bool(row[S.CSV_COL_FUSTAIA])
                coppice = not fustaia

                # Compute V via Tabacchi when species is known to Tabacchi;
                # else leave NULL (e.g., 'Altro').  Coppice rows always NULL.
                if coppice or not has_species(mapped):
                    volume_m3 = None
                    mass_q = None
                else:
                    volume_m3 = tabacchi_volume_m3(d_cm, h_m, mapped)
                    mass_q = (volume_m3 * species.density).quantize(
                        Decimal('0.001'), rounding=ROUND_HALF_UP,
                    )

                tree = Tree.objects.create(
                    species=species, parcel=parcel,
                    preserved=False, coppice=coppice,
                )
                TreeSample.objects.create(
                    sample=sample, tree=tree, shoot=0, standard=False,
                    number=int(row[S.CSV_COL_N_LEGACY]), d_cm=d_cm, h_m=h_m,
                    l10_mm=l10_mm, volume_m3=volume_m3, mass_q=mass_q,
                )
                n_trees += 1

        from apps.base.digests import mark_all_stale
        mark_all_stale()

        self.stdout.write(
            f'Survey: {survey.name}\n'
            f'Samples: {n_samples}\n'
            f'TreeSamples: {n_trees} created\n'
            f'Skipped (no parcel): {n_skipped_parcel}\n'
            f'Skipped (no sample area): {n_skipped_area}\n'
            f'Skipped (unknown species): {n_skipped_species}'
        )
