"""Preserve harvest species before cascading deletion removes the breakdown."""

from decimal import Decimal

from django.db.models.signals import pre_delete
from django.dispatch import receiver
from simple_history.signals import pre_create_historical_record

from apps.prelievi.models import Harvest, HarvestSpecies


@receiver(pre_delete, sender=Harvest)
def capture_deleted_species_masses(sender, instance, using, **kwargs):
    # simple-history records deletion in post_delete, after child rows are gone.
    # Freeze names as well as IDs: later renames must not rewrite this record.
    breakdown = HarvestSpecies.objects.using(using).filter(
        harvest_id=instance.pk,
    ).select_related('species').order_by(
        'species__sort_order', 'species__common_name', 'species_id',
    )
    snapshot = []
    for part in breakdown:
        mass = Decimal(instance.mass_q) * part.percent / Decimal(100)
        if mass != 0:
            snapshot.append({
                'species_id': part.species_id,
                'species_name': part.species.common_name,
                'mass_q': str(mass),
            })
    instance._audit_species_masses = snapshot


@receiver(pre_create_historical_record, sender=Harvest.history.model)
def attach_deleted_species_masses(sender, instance, history_instance, **kwargs):
    if history_instance.history_type == '-':
        history_instance.species_masses = instance._audit_species_masses
