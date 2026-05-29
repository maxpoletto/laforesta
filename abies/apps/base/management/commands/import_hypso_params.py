"""Install a hypsometric regression CSV as the active parameter set.

Reads `equazioni_ipsometro.csv` from <data_dir> and makes it the active
HypsoParamSet.  A missing file is a warning, not an error: a fresh install
simply starts with no parameters.
"""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.base import hypsometry
from apps.base.digests import mark_stale
from apps.base.models import HypsoParamSource
from config import strings as S
from config.constants import DIGEST_HYPSO_PARAMS


class Command(BaseCommand):
    help = "Install equazioni_ipsometro.csv as the active hypsometric parameter set."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help="Directory containing equazioni_ipsometro.csv.",
        )

    def handle(self, *args, data_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        csv_path = data_dir / S.CSV_FILE_REGRESSION
        if not csv_path.is_file():
            self.stdout.write(
                f'{csv_path} not found; no hypsometric parameters imported'
            )
            return

        rows, errors = hypsometry.parse_param_csv(
            csv_path.read_text(encoding='utf-8-sig')
        )
        if errors:
            for err in errors:
                self.stderr.write(err)
            raise CommandError(f'{csv_path} has errors; not imported')

        hypsometry.replace_active_set(
            rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[],
        )
        mark_stale(DIGEST_HYPSO_PARAMS)
        self.stdout.write(f'Hypsometry: imported {len(rows)} parameters')
