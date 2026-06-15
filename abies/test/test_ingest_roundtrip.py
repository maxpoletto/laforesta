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
    OUT_HARVESTS, OUT_HARVEST_PLAN_ITEMS, OUT_PRESERVED, OUT_REGIONS,
    OUT_SAMPLED_TREES, OUT_SURVEYS, OUT_TRACTORS, main,
)

# Defaults to the sibling data checkout; override with ABIES_LEGACY_DATA elsewhere.
_DEFAULT_LEGACY_DIR = Path(__file__).resolve().parents[2] / 'abies-data'
LEGACY_DIR = Path(os.environ.get('ABIES_LEGACY_DATA', _DEFAULT_LEGACY_DIR))

# Expected row counts (±10 tolerance where noted, exact otherwise).
EXPECTED_HARVESTS = 11941
EXPECTED_REGION_WIDE = 1468
EXPECTED_TRACTORS = 5

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
    survey_rows = _rows(out_dir / OUT_SURVEYS)
    active_surveys = [r['Rilevamento'] for r in survey_rows if r.get('Attivo') == 'True']
    assert active_surveys == ['Campionamento calcolato']
    # The `poll == 'mat'` sentinel must still produce Matricina=True rows
    # (guards against a regression that silently drops the special case).
    assert any(r['Matricina'] == 'True' for r in tree_rows)

    # Tractors: exactly 5 hard-coded La Foresta tractors.
    tractor_rows = _rows(out_dir / OUT_TRACTORS)
    assert len(tractor_rows) == EXPECTED_TRACTORS

    # Harvests: all ≈11941 mannesi.csv rows, ≈1468 with blank Particella (region-wide).
    harvest_rows = _rows(out_dir / OUT_HARVESTS)
    assert len(harvest_rows) == counts[OUT_HARVESTS]
    assert len(harvest_rows) == pytest.approx(EXPECTED_HARVESTS, abs=10)
    region_wide = [r for r in harvest_rows if not r['Particella']]
    assert len(region_wide) == pytest.approx(EXPECTED_REGION_WIDE, abs=10)

    # Harvest plan items: fustaia + ceduo combined, all non-zero.
    plan_rows = _rows(out_dir / OUT_HARVEST_PLAN_ITEMS)
    assert len(plan_rows) > 0
    assert len(plan_rows) == counts[OUT_HARVEST_PLAN_ITEMS]

    # Preserved trees: rows with valid Lon/Lat from PAI file.
    preserved_rows = _rows(out_dir / OUT_PRESERVED)
    assert len(preserved_rows) > 0
    assert len(preserved_rows) == counts[OUT_PRESERVED]


@pytest.mark.django_db
def test_bootstrap_actually_persists(converted):
    """A real (non --check) load persists the expected reference rows."""
    from apps.base.models import HarvestPlanItem, Region, Survey, Tractor, Tree
    from apps.prelievi.models import Harvest

    out_dir, counts = converted
    call_command('bootstrap', str(out_dir))
    assert Region.objects.count() == 3
    assert Survey.objects.count() == 2
    assert list(Survey.objects.filter(active=True).values_list('name', flat=True)) == [
        'Campionamento calcolato',
    ]

    # Tractors.
    assert Tractor.objects.count() == EXPECTED_TRACTORS

    # Harvests: ≈11941 total, ≈1468 region-wide (parcel IS NULL).
    assert Harvest.objects.count() == counts[OUT_HARVESTS]
    assert Harvest.objects.count() == pytest.approx(EXPECTED_HARVESTS, abs=10)
    assert Harvest.objects.filter(parcel__isnull=True).count() == pytest.approx(EXPECTED_REGION_WIDE, abs=10)

    # Harvest plan items.
    assert HarvestPlanItem.objects.count() > 0
    assert HarvestPlanItem.objects.count() == counts[OUT_HARVEST_PLAN_ITEMS]

    # Preserved trees (PAI).
    assert Tree.objects.filter(preserved=True).count() > 0
    assert Tree.objects.filter(preserved=True).count() == counts[OUT_PRESERVED]
