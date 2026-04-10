"""Prelievi domain models — harvest operations."""

from django.db import models

from apps.base.models import Crew, Note, Optype, Parcel, Species, TimestampedModel, Tractor
from config import strings as S


class HarvestOp(TimestampedModel):
    """A single harvest operation by one crew on a given day."""
    date = models.DateField()
    optype = models.ForeignKey(Optype, on_delete=models.PROTECT)
    parcel = models.ForeignKey(Parcel, on_delete=models.PROTECT)
    crew = models.ForeignKey(Crew, on_delete=models.PROTECT)
    record1 = models.IntegerField(null=True, blank=True)
    record2 = models.IntegerField(null=True, blank=True)
    quintals = models.DecimalField(max_digits=10, decimal_places=2)
    note = models.ForeignKey(Note, on_delete=models.SET_NULL, null=True, blank=True)
    extra_note = models.TextField(blank=True)

    class Meta:
        verbose_name = S.HARVEST_OP
        verbose_name_plural = S.HARVEST_OPS

    def __str__(self):
        return f'{self.date} {self.parcel} {self.crew}'


class HarvestSpecies(models.Model):
    """Species breakdown of a harvest operation (percentages must sum to 100)."""
    harvest_op = models.ForeignKey(HarvestOp, on_delete=models.CASCADE)
    species = models.ForeignKey(Species, on_delete=models.PROTECT)
    percent = models.IntegerField()

    class Meta:
        verbose_name = S.HARVEST_SPECIES
        verbose_name_plural = S.HARVEST_SPECIES_PLURAL
        unique_together = [('harvest_op', 'species')]


class HarvestTractor(models.Model):
    """Tractor breakdown of a harvest operation (percentages must sum to 100)."""
    harvest_op = models.ForeignKey(HarvestOp, on_delete=models.CASCADE)
    tractor = models.ForeignKey(Tractor, on_delete=models.PROTECT)
    percent = models.IntegerField()

    class Meta:
        verbose_name = S.HARVEST_TRACTOR
        verbose_name_plural = S.HARVEST_TRACTORS
        unique_together = [('harvest_op', 'tractor')]
