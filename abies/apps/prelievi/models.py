"""Prelievi domain models — harvest operations."""

from collections.abc import Iterable
from decimal import Decimal

from django.db import models
from simple_history.models import HistoricalRecords

from apps.base.models import (
    Crew, HarvestPlanItem, Parcel, Product, Region, Species, TimestampedModel, Tractor,
)
from config import strings as S

# Harvest volume is materialised from mass and per-species percentages, then
# stored quantized to 0.001 m³ (one litre).
VOLUME_QUANTUM_M3 = Decimal('0.001')


def harvest_volume_m3(mass_q: Decimal,
                      density_pcts: Iterable[tuple[Decimal | None, int]]) -> Decimal:
    """Materialised harvest volume: ``SUM(mass_q × pct/100 / density)`` over
    (density, pct) pairs, quantized to 0.001 m³.  A pair whose density is
    missing or non-positive contributes nothing (no finite volume without a
    positive density)."""
    total = Decimal('0')
    hundred = Decimal(100)
    for density, pct in density_pcts:
        if density and density > 0:
            total += mass_q * Decimal(pct) / hundred / density
    return total.quantize(VOLUME_QUANTUM_M3)


class Harvest(TimestampedModel):
    """A single harvest operation by one crew on a given day."""
    date = models.DateField()
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    # Region XOR parcel — exactly one must be set (enforced by SQLite trigger
    # in migration 0003; this mirrors HarvestPlanItem's region/parcel XOR).
    region = models.ForeignKey(
        Region, on_delete=models.PROTECT, null=True, blank=True,
        related_name='harvests',
    )
    parcel = models.ForeignKey(
        Parcel, on_delete=models.PROTECT, null=True, blank=True,
    )
    crew = models.ForeignKey(Crew, on_delete=models.PROTECT)
    # New harvests must link to a HarvestPlanItem in state {open,
    # harvesting} — enforced at the view layer. Historical CSV-imported
    # rows leave this NULL.  Deletion of a linked plan item is blocked
    # at the DB level (ON DELETE PROTECT) so the link cannot orphan;
    # see `docs/database.md`.
    harvest_plan_item = models.ForeignKey(
        HarvestPlanItem, on_delete=models.PROTECT,
        null=True, blank=True, related_name='harvests',
    )
    record1 = models.IntegerField(null=True, blank=True)
    record2 = models.IntegerField(null=True, blank=True)
    mass_q = models.DecimalField(max_digits=10, decimal_places=2)
    volume_m3 = models.DecimalField(
        max_digits=10, decimal_places=3, default=0,
        help_text='Computed at write time: SUM(mass_q × pct/100 / species.density).',
    )
    # Boolean flags rendered as a comma-joined string in the Note column.
    # When `harvest_plan_item` is set, the three flags must equal those
    # on the linked item (enforced by SQLite trigger).
    damaged = models.BooleanField(default=False)
    unhealthy = models.BooleanField(default=False)
    psr = models.BooleanField(default=False)
    note = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        verbose_name = S.HARVEST
        verbose_name_plural = S.HARVESTS

    def clean(self):
        from django.core.exceptions import ValidationError
        if (self.region_id is None) == (self.parcel_id is None):
            raise ValidationError(S.ERR_HARVEST_REGION_XOR_PARCEL)

    def __str__(self):
        loc = self.parcel or self.region
        return f'{self.date} {loc} {self.crew}'


class HarvestSpecies(models.Model):
    """Species breakdown of a harvest (percentages must sum to 100)."""
    harvest = models.ForeignKey(Harvest, on_delete=models.CASCADE)
    species = models.ForeignKey(Species, on_delete=models.PROTECT)
    percent = models.IntegerField()

    class Meta:
        verbose_name = S.HARVEST_SPECIES
        verbose_name_plural = S.HARVEST_SPECIES_PLURAL
        unique_together = [('harvest', 'species')]


class HarvestTractor(models.Model):
    """Tractor breakdown of a harvest (percentages must sum to 100)."""
    harvest = models.ForeignKey(Harvest, on_delete=models.CASCADE)
    tractor = models.ForeignKey(Tractor, on_delete=models.PROTECT)
    percent = models.IntegerField()

    class Meta:
        verbose_name = S.HARVEST_TRACTOR
        verbose_name_plural = S.HARVEST_TRACTORS
        unique_together = [('harvest', 'tractor')]
