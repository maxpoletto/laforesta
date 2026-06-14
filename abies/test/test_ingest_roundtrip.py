"""Round-trip proof for the legacy→canonical converter (ingest/convert_laforesta).

Runs the converter on the *real* legacy data dir, then loads the result with the
strict ``bootstrap --check`` contract against an empty test DB and asserts it
validates clean (no ``CommandError``).  This is the key proof that the converter
emits exactly what bootstrap can consume today.

If the real legacy data dir is not present (e.g. CI without the data checkout),
the test is skipped rather than failed.
"""

import csv
import os
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from ingest.convert_laforesta import (
    OUT_REGIONS, OUT_SAMPLED_TREES, OUT_SURVEYS, main,
)

# Defaults to the local checkout; override with ABIES_LEGACY_DATA elsewhere.
LEGACY_DIR = Path(os.environ.get(
    'ABIES_LEGACY_DATA', '/Users/maxp/src/laforesta/abies-data'))

pytestmark = pytest.mark.skipif(
    not LEGACY_DIR.is_dir(),
    reason=f'legacy data dir {LEGACY_DIR} not present',
)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


@pytest.fixture
def converted(tmp_path):
    out_dir = tmp_path / 'canonical'
    counts = main(LEGACY_DIR, out_dir)
    return out_dir, counts


@pytest.mark.django_db
def test_bootstrap_check_validates_clean(converted, capsys):
    """The converted dir must pass ``bootstrap --check`` with no CommandError."""
    out_dir, _ = converted
    try:
        call_command('bootstrap', str(out_dir), '--check')
    except CommandError as exc:  # pragma: no cover - surfaces the report on failure
        report = capsys.readouterr().out
        pytest.fail(f'bootstrap --check failed: {exc}\n\n{report}')


def test_sanity_counts(converted):
    out_dir, counts = converted

    # 3 regions (Capistrano, Fabrizia, Serra).
    assert counts[OUT_REGIONS] == 3
    assert len(_rows(out_dir / OUT_REGIONS)) == 3

    # Parcels present.
    assert (out_dir / 'particelle.csv').is_file()
    assert len(_rows(out_dir / 'particelle.csv')) > 0

    # Exactly the two surveys.
    survey_rows = _rows(out_dir / OUT_SURVEYS)
    assert len(survey_rows) == 2
    assert {r['Rilevamento'] for r in survey_rows} == {
        'Campionamento calcolato', 'Campionamento altezze',
    }

    # Sampled trees are the union of both surveys.
    tree_rows = _rows(out_dir / OUT_SAMPLED_TREES)
    surveys_seen = {r['Rilevamento'] for r in tree_rows}
    assert surveys_seen == {'Campionamento calcolato', 'Campionamento altezze'}
    # The `poll == 'mat'` sentinel must still produce Matricina=True rows
    # (guards against a regression that silently drops the special case).
    assert any(r['Matricina'] == 'True' for r in tree_rows)


@pytest.mark.django_db
def test_bootstrap_actually_persists(converted):
    """A real (non --check) load persists the expected reference rows."""
    from apps.base.models import Region, Survey

    out_dir, _ = converted
    call_command('bootstrap', str(out_dir))
    assert Region.objects.count() == 3
    assert Survey.objects.count() == 2
