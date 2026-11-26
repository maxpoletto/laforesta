"""Shared validation for harvest rows.

Manual Prelievi entry and CSV import both write ``Harvest`` plus percentage
breakdowns.  Keep the common scalar rules here so the paths do not drift.
"""

from __future__ import annotations

from decimal import Decimal

from config import strings as S
from config.constants import FIELD_DAMAGED, FIELD_PSR, FIELD_UNHEALTHY


HARVEST_FLAG_LABELS = {
    FIELD_DAMAGED: S.FLAG_DAMAGED,
    FIELD_UNHEALTHY: S.FLAG_UNHEALTHY,
    FIELD_PSR: S.FLAG_PSR,
}


def checked_harvest_flags(source):
    return [field for field in HARVEST_FLAG_LABELS if getattr(source, field, False)]


def missing_worksite_flags(worksite, values):
    return [field for field in checked_harvest_flags(worksite) if not values.get(field, True)]


def valid_harvest_mass(mass_q: Decimal | None) -> bool:
    return mass_q is not None and mass_q > 0


def valid_percentage(pct: int | None) -> bool:
    return pct is not None and 0 <= pct <= 100


def percentages_sum_to_100(values) -> bool:
    return sum(values) == 100


def percentages_sum_to_0_or_100(values) -> bool:
    return sum(values) in (0, 100)
