"""The reserved survey is inactive by schema, not by a silent repair write."""
import pytest
from django.db import IntegrityError, transaction

from apps.base.models import SampleGrid, Survey
from apps.base.preserved_trees import PRESERVED_SURVEY_NAME, get_preserved_tree_survey

pytestmark = pytest.mark.django_db


def test_existing_reserved_survey_is_a_complete_noop():
    survey = get_preserved_tree_survey()
    before = (survey.version, survey.created_at, survey.modified_at, survey.history.count())
    result = get_preserved_tree_survey()
    assert result.pk == survey.pk and not result.active
    assert (result.version, result.created_at, result.modified_at, result.history.count()) == before


def test_active_unstructured_reserved_survey_is_impossible():
    with pytest.raises(IntegrityError), transaction.atomic():
        Survey.objects.create(name=PRESERVED_SURVEY_NAME, active=True)


def test_structured_name_collision_is_rejected_without_repairing_user_data():
    grid = SampleGrid.objects.create(name='Structured survey grid')
    survey = Survey.objects.create(name=PRESERVED_SURVEY_NAME, active=True, sample_grid=grid)
    before = (survey.version, survey.modified_at, survey.history.count())
    with pytest.raises(RuntimeError, match='structured'):
        get_preserved_tree_survey()
    survey.refresh_from_db()
    assert survey.active and survey.sample_grid_id == grid.pk
    assert (survey.version, survey.modified_at, survey.history.count()) == before
