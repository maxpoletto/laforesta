from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.base.management.commands.check_release1_tree_observations import (
    PRESERVED_HISTORY_SURVEY_NAME,
)
from apps.base.models import (
    Sample, SampleArea, SampleGrid, Survey, Tree, TreePreserved, TreeSample,
)


def _call_check(phase: str) -> str:
    stdout = StringIO()
    call_command('check_release1_tree_observations', phase=phase, stdout=stdout)
    return stdout.getvalue()


def _tree(parcel, species, **kwargs) -> Tree:
    defaults = {
        'species': species[0],
        'parcel': parcel,
        'lat': 38.0,
        'lon': 16.0,
    }
    defaults.update(kwargs)
    return Tree.objects.create(**defaults)


def _structured_sample(parcel) -> Sample:
    grid = SampleGrid.objects.create(name='Grid')
    area = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel, number='1', lat=38.0, lon=16.0,
    )
    survey = Survey.objects.create(name='Structured survey', sample_grid=grid)
    return Sample.objects.create(
        sample_area=area, survey=survey, date=date(2026, 1, 15),
    )


def _free_sample(name: str, sample_date: date) -> Sample:
    survey = Survey.objects.create(name=name, sample_grid=None)
    return Sample.objects.create(
        sample_area=None, survey=survey, date=sample_date,
    )


def _legacy_preserved(parcel, species, *, number=7, tree=None) -> TreePreserved:
    tree = tree or _tree(
        parcel, species, preserved=True, lat=38.1, lon=16.1, acc_m=4,
    )
    return TreePreserved.objects.create(
        tree=tree,
        parcel=parcel,
        number=number,
        date=date(2025, 6, 1),
        d_cm=40,
        h_m=Decimal('18.50'),
        h_measured=True,
        volume_m3=Decimal('1.2500'),
        mass_q=Decimal('11.250'),
        lat=38.1,
        lon=16.1,
        acc_m=4,
        operator='Luca',
        note='storico',
    )


def _migrated_preserved_sample(parcel, legacy: TreePreserved) -> TreeSample:
    sample = _free_sample(PRESERVED_HISTORY_SURVEY_NAME, legacy.date)
    return TreeSample.objects.create(
        sample=sample,
        tree=legacy.tree,
        parcel=parcel,
        number=legacy.number,
        preserved_number=legacy.number,
        shoot=0,
        standard=False,
        d_cm=legacy.d_cm,
        h_m=legacy.h_m,
        h_measured=legacy.h_measured,
        volume_m3=legacy.volume_m3,
        mass_q=legacy.mass_q,
        lat=legacy.lat,
        lon=legacy.lon,
        acc_m=legacy.acc_m,
        operator=legacy.operator,
        note=legacy.note,
    )


def test_release1_preflight_passes_empty_db(db):
    assert 'Release 1 tree-observation preflight OK' in _call_check('pre')


def test_release1_preflight_rejects_incomplete_legacy_preserved_rows(parcels, species):
    tree = _tree(parcels[0], species)
    TreePreserved.objects.create(
        tree=tree, parcel=parcels[0], number=7, lat=38.1, lon=16.1,
    )

    with pytest.raises(CommandError, match='missing values required by migration'):
        _call_check('pre')


def test_release1_preflight_rejects_existing_free_samples(db):
    _free_sample('Unexpected free survey', date(2026, 1, 1))

    with pytest.raises(CommandError, match='Existing free Sample rows'):
        _call_check('pre')


def test_release1_postflight_accepts_migrated_legacy_preserved_rows(parcels, species):
    legacy = _legacy_preserved(parcels[0], species)
    _migrated_preserved_sample(parcels[0], legacy)

    assert 'Release 1 tree-observation postflight OK' in _call_check('post')


def test_release1_postflight_rejects_unmigrated_legacy_preserved_rows(parcels, species):
    _legacy_preserved(parcels[0], species)

    with pytest.raises(CommandError, match='without migrated TreeSample rows'):
        _call_check('post')


def test_release1_postflight_rejects_preserved_identity_split(parcels, species):
    sample_1 = _free_sample('Free 1', date(2026, 1, 1))
    sample_2 = _free_sample('Free 2', date(2026, 1, 2))
    tree_1 = _tree(parcels[0], species, preserved=True)
    tree_2 = _tree(parcels[0], species, preserved=True, lat=38.2, lon=16.2)
    for sample, tree, number in [(sample_1, tree_1, 1), (sample_2, tree_2, 2)]:
        TreeSample.objects.create(
            sample=sample,
            tree=tree,
            parcel=parcels[0],
            number=number,
            preserved_number=7,
            d_cm=40,
            h_m=Decimal('18.50'),
            h_measured=True,
        )

    with pytest.raises(CommandError, match='maps to more than one Tree'):
        _call_check('post')


def test_release1_postflight_rejects_structured_sample_parcel_mismatch(parcels, species):
    sample = _structured_sample(parcels[0])
    tree = _tree(parcels[1], species)
    TreeSample.objects.create(
        sample=sample,
        tree=tree,
        parcel=parcels[1],
        number=1,
        d_cm=30,
        h_m=Decimal('16.00'),
        h_measured=True,
    )

    with pytest.raises(CommandError, match='parcel different from SampleArea parcel'):
        _call_check('post')
