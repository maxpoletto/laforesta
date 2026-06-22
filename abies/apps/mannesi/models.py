"""Squadre domain models: team work hours and production credits."""

from django.db import models
from simple_history.models import HistoricalRecords

from apps.base.models import Crew, TimestampedModel
from config import strings as S


class WorkHour(TimestampedModel):
    """Hours worked by a team on a date."""
    date = models.DateField()
    crew = models.ForeignKey(Crew, on_delete=models.PROTECT)
    hours = models.DecimalField(max_digits=8, decimal_places=2)
    note = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        db_table = 'mannesi_hours'
        verbose_name = S.TABLE_MANNESI_HOURS
        verbose_name_plural = S.TABLE_MANNESI_HOURS
        ordering = ['-date', 'id']

    def __str__(self):
        return f'{self.date} {self.crew} {self.hours}'


class ProductionCredit(TimestampedModel):
    """Production credit/acconto for a team on a date."""
    date = models.DateField()
    crew = models.ForeignKey(Crew, on_delete=models.PROTECT)
    mass_q = models.DecimalField(max_digits=10, decimal_places=2)
    note = models.TextField(blank=True)
    history = HistoricalRecords()

    class Meta:
        db_table = 'mannesi_credits'
        verbose_name = S.TABLE_MANNESI_CREDIT
        verbose_name_plural = S.TABLE_MANNESI_CREDITS
        ordering = ['-date', 'id']

    def __str__(self):
        return f'{self.date} {self.crew} {self.mass_q}'
