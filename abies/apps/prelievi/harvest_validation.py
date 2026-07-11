"""Shared validation for harvest rows.

Manual Prelievi entry and CSV import both write ``Harvest`` plus percentage
breakdowns.  Keep the common scalar rules here so the paths do not drift.
"""

from __future__ import annotations

from decimal import Decimal


def valid_harvest_mass(mass_q: Decimal | None) -> bool:
    return mass_q is not None and mass_q > 0


def valid_percentage(pct: int | None) -> bool:
    return pct is not None and 0 <= pct <= 100


def percentages_sum_to_100(values) -> bool:
    return sum(values) == 100


def percentages_sum_to_0_or_100(values) -> bool:
    return sum(values) in (0, 100)
