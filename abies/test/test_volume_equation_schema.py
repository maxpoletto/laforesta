"""Frozen upgrade defaults must preserve species IDs and existing estimates."""
import importlib
from decimal import Decimal
from types import SimpleNamespace

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

from apps.base.models import Species

pytestmark = pytest.mark.django_db


def test_upgrade_recognizes_a_renamed_reference_and_is_idempotent():
    fir = Species.objects.create(common_name='Abete bianco', latin_name='Abies alba')
    bucket = Species.objects.create(common_name='Altra conifera')
    # Simulate the pre-upgrade schema's data: model creation defaults were not
    # available when these rows were originally inserted.
    from apps.base.models import VolumeEquation
    VolumeEquation.objects.all().delete()
    before = (fir.pk, fir.version, fir.modified_at)
    migration = importlib.import_module('apps.base.migrations.0032_species_equations')
    state = MigrationExecutor(connection).loader.project_state([
        ('base', '0032_species_equations'),
    ]).apps
    migration.seed_species_equations(state, SimpleNamespace(connection=connection))
    equation = VolumeEquation.objects.get(species_id=fir.pk)
    assert equation.b0 == Decimal('-1.8381')
    assert VolumeEquation.objects.get(species_id=bucket.pk).proxy_species_id == fir.pk
    history_count = VolumeEquation.history.count()
    migration.seed_species_equations(state, SimpleNamespace(connection=connection))
    assert VolumeEquation.objects.count() == 2
    assert VolumeEquation.history.count() == history_count
    fir.refresh_from_db()
    assert (fir.pk, fir.version, fir.modified_at) == before


def test_upgrade_does_not_assign_an_unavailable_spruce_equation():
    spruce = Species.objects.create(common_name='Abete rosso', latin_name='Picea abies')
    assert spruce.volume_coefficients is None


def test_default_initialization_does_not_reset_an_edited_equation():
    fir = Species.objects.create(common_name='Abete', latin_name='Abies alba')
    equation = fir.volume_equation
    equation.b0 = Decimal('123')
    equation.save()
    fir.density = 10
    fir.save()
    equation.refresh_from_db()
    assert equation.b0 == 123
