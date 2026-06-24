"""Round-trip proof for the legacy→canonical converter (abies-init/convert_laforesta).

Runs the converter on the *real* legacy data dir, then loads the result with the
strict ``bootstrap --check`` contract against an empty test DB and asserts it
validates clean (no ``CommandError``).  This is the key proof that the converter
emits exactly what bootstrap can consume today.

If the real legacy data dir is not present (e.g. CI without the data checkout),
the test is skipped rather than failed.
"""

import csv
import os
import sys
from pathlib import Path

ABIES_INIT_ROOT = Path(__file__).resolve().parents[1]
LAFORESTA_ROOT = ABIES_INIT_ROOT.parent
ABIES_ROOT = Path(os.environ.get('ABIES_ROOT', LAFORESTA_ROOT / 'abies'))
sys.path.insert(0, str(ABIES_INIT_ROOT))
sys.path.insert(0, str(ABIES_ROOT))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.test_settings')

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from convert_laforesta import (
    COL_ACTIVE, COL_CREW, COL_PARCEL, COL_TRACTOR_NAME, OUT_CREWS,
    OUT_HARVESTS, OUT_HARVEST_PLAN_ITEMS, OUT_MARKS_DIR, OUT_PRESERVED,
    OUT_REGIONS, OUT_SAMPLED_TREES, OUT_SPECIES, OUT_SURVEYS, OUT_TRACTORS,
    SRC_HARVESTS, SRC_MARTELLATE_DIR, SURVEY_LUCA, SURVEY_SABATINO, main,
)

# Defaults to the sibling data checkout; override with ABIES_LEGACY_DATA elsewhere.
_DEFAULT_LEGACY_DIR = LAFORESTA_ROOT / 'abies-data'
LEGACY_DIR = Path(os.environ.get('ABIES_LEGACY_DATA', _DEFAULT_LEGACY_DIR))

EXPECTED_TRACTORS = 6
EXPECTED_PAI_MINOR_SPECIES = {
    'Betulla Bianca',
    'Farnia',
    'Noce',
    'Pioppo Tremulo',
}
EXPECTED_ACTIVE_CREWS = {
    'Campese X2',
    'Manno',
    'Mix Max',
    'Ns Operai',
    'Zaffino 4x4',
    'Zaffino-Santaguida',
}

pytestmark = pytest.mark.skipif(
    not LEGACY_DIR.is_dir(),
    reason=f'legacy data dir {LEGACY_DIR} not present',
)


def _legacy_region_wide_harvests() -> int:
    return sum(
        1 for row in _rows(LEGACY_DIR / SRC_HARVESTS)
        if (row.get(COL_PARCEL) or '').strip() in ('', 'X')
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
    assert {r['Rilevamento'] for r in survey_rows} == {SURVEY_SABATINO, SURVEY_LUCA}

    # Sampled trees are the union of both surveys.
    tree_rows = _rows(out_dir / OUT_SAMPLED_TREES)
    surveys_seen = {r['Rilevamento'] for r in tree_rows}
    assert surveys_seen == {SURVEY_SABATINO, SURVEY_LUCA}
    survey_rows = _rows(out_dir / OUT_SURVEYS)
    active_surveys = [r['Rilevamento'] for r in survey_rows if r.get('Attivo') == 'True']
    assert active_surveys == [SURVEY_SABATINO]
    # The `poll == 'mat'` sentinel must still produce Matricina=True rows
    # (guards against a regression that silently drops the special case).
    assert any(r['Matricina'] == 'True' for r in tree_rows)
    assert {r['Pressler'] for r in tree_rows} == {'2'}
    assert 'Pino Laricio' not in {r['Genere'] for r in tree_rows}

    # Crews: only crews used in 2026 and the explicit new crew are active.
    crew_rows = _rows(out_dir / OUT_CREWS)
    assert {
        r[COL_CREW] for r in crew_rows if r[COL_ACTIVE] == 'true'
    } == EXPECTED_ACTIVE_CREWS

    # Tractors: exactly the hard-coded La Foresta tractors.
    tractor_rows = _rows(out_dir / OUT_TRACTORS)
    assert len(tractor_rows) == EXPECTED_TRACTORS
    assert 'Scania P380' in {r[COL_TRACTOR_NAME] for r in tractor_rows}

    # PAI-only species are canonical minor species, not squashed into Altro.
    species_rows = _rows(out_dir / OUT_SPECIES)
    species_by_name = {r['Genere']: r for r in species_rows}
    assert 'Pino Laricio' not in species_by_name
    assert {'Pino Nero', 'Pino Marittimo', 'Pino Strobo'} <= set(species_by_name)
    assert species_by_name['Pino Nero']['Minore'] == 'false'
    for name in EXPECTED_PAI_MINOR_SPECIES:
        assert species_by_name[name]['Minore'] == 'true'

    # Harvests: one canonical row for each current legacy mannesi.csv row.
    harvest_rows = _rows(out_dir / OUT_HARVESTS)
    assert len(harvest_rows) == counts[OUT_HARVESTS]
    assert len(harvest_rows) == len(_rows(LEGACY_DIR / SRC_HARVESTS))
    region_wide = [r for r in harvest_rows if not r[COL_PARCEL]]
    assert len(region_wide) == _legacy_region_wide_harvests()

    # Harvest plan items: fustaia + ceduo combined, all non-zero.
    plan_rows = _rows(out_dir / OUT_HARVEST_PLAN_ITEMS)
    assert len(plan_rows) > 0
    assert len(plan_rows) == counts[OUT_HARVEST_PLAN_ITEMS]

    # Mark upload CSVs: one normalized output file per staged source file.
    source_martellate = sorted((LEGACY_DIR / SRC_MARTELLATE_DIR).glob('*.csv'))
    output_marks = sorted((out_dir / OUT_MARKS_DIR).glob('*.csv'))
    assert len(output_marks) == len(source_martellate)
    assert counts[f'{OUT_MARKS_DIR}/*.csv'] == sum(
        len(_rows(path)) for path in output_marks
    )

    # Preserved trees: rows with valid Lon/Lat from PAI file.
    preserved_rows = _rows(out_dir / OUT_PRESERVED)
    assert len(preserved_rows) > 0
    assert len(preserved_rows) == counts[OUT_PRESERVED]
    preserved_species = {r['Genere'] for r in preserved_rows}
    assert 'Altro' not in preserved_species
    assert 'Pino Laricio' not in preserved_species
    assert {'Pino Nero', 'Pino Strobo'} <= preserved_species
    assert EXPECTED_PAI_MINOR_SPECIES <= preserved_species
    numbers_by_parcel = {}
    for row in preserved_rows:
        key = (row['Compresa'], row['Particella'])
        numbers_by_parcel.setdefault(key, []).append(int(row['Numero']))
    for numbers in numbers_by_parcel.values():
        assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.django_db
def test_bootstrap_actually_persists(converted):
    """A real (non --check) load persists the expected reference rows."""
    from apps.base.models import (
        HarvestPlanItem, Region, Survey, Tractor, Tree, TreePreserved,
    )
    from apps.prelievi.models import Harvest

    out_dir, counts = converted
    call_command('bootstrap', str(out_dir))
    assert Region.objects.count() == 3
    assert Survey.objects.count() == 2
    assert list(Survey.objects.filter(active=True).values_list('name', flat=True)) == [
        SURVEY_SABATINO,
    ]

    # Tractors.
    assert Tractor.objects.count() == EXPECTED_TRACTORS

    # Harvests: persisted rows match the converted canonical harvests.
    harvest_rows = _rows(out_dir / OUT_HARVESTS)
    assert Harvest.objects.count() == counts[OUT_HARVESTS]
    assert Harvest.objects.count() == len(harvest_rows)
    assert Harvest.objects.filter(parcel__isnull=True).count() == sum(
        1 for row in harvest_rows if not row[COL_PARCEL]
    )

    # Harvest plan items.
    assert HarvestPlanItem.objects.count() > 0
    assert HarvestPlanItem.objects.count() == counts[OUT_HARVEST_PLAN_ITEMS]

    # Preserved trees (PAI).
    assert Tree.objects.filter(preserved=True).count() > 0
    assert Tree.objects.filter(preserved=True).count() == counts[OUT_PRESERVED]
    assert TreePreserved.objects.count() == counts[OUT_PRESERVED]
