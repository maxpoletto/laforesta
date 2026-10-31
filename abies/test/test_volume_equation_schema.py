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


@pytest.mark.django_db(transaction=True)
def test_real_upgrade_preserves_captured_zero_estimates_and_height_links():
    """Exercise the schema operations against actual pre-0032 rows."""
    executor = MigrationExecutor(connection)
    latest = executor.loader.graph.leaf_nodes()
    previous = [('base', '0031_user_version')]
    executor.migrate(previous)
    try:
        old = executor.loader.project_state(previous).apps
        region = old.get_model('base', 'Region').objects.create(name='Upgrade region')
        eclass = old.get_model('base', 'Eclass').objects.create(name='A')
        parcel = old.get_model('base', 'Parcel').objects.create(name='1', region=region,
                                                               eclass=eclass, area_ha=1)
        sp = old.get_model('base', 'Species').objects.create(
            common_name='Abete bianco', latin_name='Abies alba', density='9')
        tree = old.get_model('base', 'Tree').objects.create(species=sp)
        survey = old.get_model('base', 'Survey').objects.create(name='Upgrade survey')
        sample = old.get_model('base', 'Sample').objects.create(survey=survey, date='2024-01-01')
        observation = old.get_model('base', 'TreeSample').objects.create(
            sample=sample, tree=tree, parcel=parcel, number=1,
            d_cm=30, h_m=20, volume_m3=0, mass_q=0)
        parameter_set = old.get_model('base', 'HypsoParamSet').objects.create(source='computed', min_n=5)
        parameter = old.get_model('base', 'HypsoParam').objects.create(
            param_set=parameter_set, region=region, species=sp, a=7, b=-4, n=20, r2='.6')
        before = (observation.pk, observation.version, observation.modified_at)
        executor = MigrationExecutor(connection)
        executor.migrate(latest)
        from apps.base.models import HypsoParam, TreeSample, VolumeEquation
        current = TreeSample.objects.get(pk=observation.pk)
        assert (current.pk, current.version, current.modified_at) == before
        assert current.volume_m3 == current.mass_q == 0
        assert current.tree.species_id == sp.pk
        equation = VolumeEquation.objects.get(species_id=sp.pk)
        assert equation.b0 == Decimal('-1.8381')
        current_parameter = HypsoParam.objects.get(pk=parameter.pk)
        assert current_parameter.species_id == sp.pk
        assert current_parameter.origin_set_id == parameter_set.pk
        assert current_parameter.source == 'computed'
    finally:
        MigrationExecutor(connection).migrate(latest)
