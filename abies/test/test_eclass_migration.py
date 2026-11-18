import pytest
from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor

from apps.base.models import Eclass, Parcel


@pytest.mark.django_db(transaction=True)
def test_upgrade_preserves_existing_catalog_and_parcel_constraints():
    executor = MigrationExecutor(connection)
    latest = executor.loader.graph.leaf_nodes()
    previous = [('base', '0035_map_features')]
    executor.migrate(previous)
    try:
        old = executor.loader.project_state(previous).apps
        classes = old.get_model('base', 'Eclass').objects
        classes.all().delete()
        fir = classes.create(name='A', min_harvest_volume=85)
        # This site uses F for high forest and a custom code for coppice.
        classes.create(name='F', coppice=False)
        classes.create(name='Z', coppice=True)
        region = old.get_model('base', 'Region').objects.create(name='Upgrade')
        parcel = old.get_model('base', 'Parcel').objects.create(
            region=region, name='1', eclass=fir, area_ha=1)

        MigrationExecutor(connection).migrate(latest)
        assert list(Eclass.objects.values_list('name', 'coppice', 'description')) == [
            ('A', False, 'Fustaia di abete'), ('F', False, ''), ('Z', True, '')]
        linked = Parcel.objects.get(pk=parcel.pk).eclass
        assert linked.pk == fir.pk and linked.min_harvest_volume == 85
        assert linked.version == 1 and linked.history.get().description == linked.description
        with pytest.raises(IntegrityError), transaction.atomic():
            Eclass.objects.filter(pk=fir.pk).update(coppice=True)
        with pytest.raises(IntegrityError), transaction.atomic():
            Parcel.objects.filter(pk=parcel.pk).update(intervention_interval=20)
    finally:
        MigrationExecutor(connection).migrate(latest)
