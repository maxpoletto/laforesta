"""Shared validation for sampled-tree measurements.

Manual entry, CSV import, and Ipso import all eventually create ``TreeSample``
rows.  Keep the row-level numeric rules here so the three paths reject the same
invalid values before they reach the model/database layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from config.constants import PRESSLER_DEFAULT, TREE_H_QUANTUM


@dataclass(frozen=True)
class SampleShootValues:
    shoot: int
    d_cm: int
    h_m: Decimal
    l10_mm: int


@dataclass(frozen=True)
class SampleTreeValues:
    number: int
    shoot: int
    d_cm: int
    h_m: Decimal
    l10_mm: int
    pressler_coeff: Decimal


def normalize_sample_shoot_values(
        *, shoot: int | None, d_cm: int | None, h_m: Decimal | None,
        l10_mm: int | None,
) -> SampleShootValues | None:
    """Return normalized coppice-shoot values, or ``None`` when invalid."""
    if shoot is None or d_cm is None or h_m is None:
        return None
    if l10_mm is None:
        l10_mm = 0
    if shoot < 0 or d_cm <= 0 or h_m <= 0 or l10_mm < 0:
        return None
    return SampleShootValues(
        shoot=shoot,
        d_cm=d_cm,
        h_m=h_m.quantize(TREE_H_QUANTUM, rounding=ROUND_HALF_UP),
        l10_mm=l10_mm,
    )


def normalize_sample_tree_values(
        *, number: int | None, d_cm: int | None, h_m: Decimal | None,
        shoot: int | None = 0, l10_mm: int | None = 0,
        pressler_coeff: Decimal | None = None,
) -> SampleTreeValues | None:
    """Return normalized sampled-tree values, or ``None`` when invalid."""
    if number is None or number <= 0:
        return None
    if pressler_coeff is None:
        pressler_coeff = PRESSLER_DEFAULT
    if pressler_coeff <= 0:
        return None
    shoot_values = normalize_sample_shoot_values(
        shoot=shoot, d_cm=d_cm, h_m=h_m, l10_mm=l10_mm,
    )
    if shoot_values is None:
        return None
    return SampleTreeValues(
        number=number,
        shoot=shoot_values.shoot,
        d_cm=shoot_values.d_cm,
        h_m=shoot_values.h_m,
        l10_mm=shoot_values.l10_mm,
        pressler_coeff=pressler_coeff.quantize(
            TREE_H_QUANTUM, rounding=ROUND_HALF_UP,
        ),
    )
