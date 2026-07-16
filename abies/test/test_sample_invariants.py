"""Schema-level invariants on Survey + Sample + SampleArea.

Confirms the SQLite trigger `sample_grid_match_*` actually fires on
INSERT/UPDATE of a Sample whose sample_area does not satisfy the survey mode:
structured surveys require a matching sample area, while unstructured surveys
require no sample area. Documented in `docs/database.md`.
"""

from datetime import date
from decimal import Decimal

import pytest
from django.db import DatabaseError, IntegrityError

from apps.base.models import (
    Parcel, Region, Eclass, Sample, SampleArea, SampleGrid, Survey,
)


@pytest.fixture
def two_grids(db, regions, eclasses):
    """Two grids, each containing one sample area in the same parcel."""
    parcel = Parcel.objects.create(
        name='trigger-test', region=regions[0], eclass=eclasses[0],
        area_ha=Decimal('1.00'),
    )
    g1 = SampleGrid.objects.create(name='Grid A')
    g2 = SampleGrid.objects.create(name='Grid B')
    sa1 = SampleArea.objects.create(
        sample_grid=g1, parcel=parcel, number='1',
        lat=0.0, lon=0.0, r_m=12,
    )
    sa2 = SampleArea.objects.create(
        sample_grid=g2, parcel=parcel, number='1',
        lat=0.0, lon=0.0, r_m=12,
    )
    survey_a = Survey.objects.create(name='SurveyA', sample_grid=g1)
    survey_b = Survey.objects.create(name='SurveyB', sample_grid=g2)
    survey_unstructured = Survey.objects.create(name='SurveyUnstructured')
    return {
        'g1': g1, 'g2': g2,
        'sa1': sa1, 'sa2': sa2,
        'survey_a': survey_a, 'survey_b': survey_b,
        'survey_unstructured': survey_unstructured,
    }


def test_matching_grid_inserts_ok(two_grids):
    """Sample with matching sample_area.grid == survey.grid succeeds."""
    f = two_grids
    Sample.objects.create(
        sample_area=f['sa1'], survey=f['survey_a'], date=date.today(),
    )
    assert Sample.objects.filter(survey=f['survey_a']).count() == 1


def test_mismatched_grid_insert_blocks(two_grids):
    """Sample with mismatched grids is rejected by the trigger."""
    f = two_grids
    # sa1 is in grid A, but survey_b is on grid B → mismatch.
    with pytest.raises((IntegrityError, DatabaseError)) as exc:
        Sample.objects.create(
            sample_area=f['sa1'], survey=f['survey_b'], date=date.today(),
        )
    assert 'sample_grid' in str(exc.value).lower()


def test_mismatched_grid_update_blocks(two_grids):
    """Updating a Sample to point at a mismatched area also blocks."""
    f = two_grids
    s = Sample.objects.create(
        sample_area=f['sa1'], survey=f['survey_a'], date=date.today(),
    )
    s.sample_area = f['sa2']  # sa2 is in grid B, survey_a is on grid A
    with pytest.raises((IntegrityError, DatabaseError)):
        s.save()


def test_unstructured_sample_without_area_inserts_ok(two_grids):
    """Unstructured surveys store samples without sample areas."""
    f = two_grids
    Sample.objects.create(
        sample_area=None, survey=f['survey_unstructured'], date=date.today(),
    )
    assert Sample.objects.filter(survey=f['survey_unstructured']).count() == 1


def test_structured_sample_without_area_blocks(two_grids):
    """Structured surveys must keep pointing at a sample area."""
    f = two_grids
    with pytest.raises((IntegrityError, DatabaseError)) as exc:
        Sample.objects.create(
            sample_area=None, survey=f['survey_a'], date=date.today(),
        )
    assert 'sample_grid' in str(exc.value).lower()


def test_unstructured_sample_with_area_blocks(two_grids):
    """Unstructured surveys must not point at a grid sample area."""
    f = two_grids
    with pytest.raises((IntegrityError, DatabaseError)) as exc:
        Sample.objects.create(
            sample_area=f['sa1'], survey=f['survey_unstructured'],
            date=date.today(),
        )
    assert 'sample_grid' in str(exc.value).lower()


def test_unstructured_survey_cannot_be_active(db):
    """Only structured surveys can feed dendrometric settings."""
    with pytest.raises((IntegrityError, DatabaseError)) as exc:
        Survey.objects.create(name='ActiveUnstructured', active=True)
    assert 'survey_active_requires_sample_grid' in str(exc.value).lower()


def test_structured_survey_can_be_active(two_grids):
    f = two_grids
    survey = Survey.objects.create(
        name='ActiveStructured', sample_grid=f['g1'], active=True,
    )
    assert survey.active is True
