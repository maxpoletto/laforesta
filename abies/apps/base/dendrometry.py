"""Shared diameter-class and basal-area calculations.

Both digest generation and CSV exports use this module so a site-wide class
mode cannot drift between browser charts and downloaded matrices.
"""

import math

from config.constants import (
    DIAMETER_CLASS_CENTERED, DIAMETER_CLASS_MODES,
    DIAMETER_CLASS_SHIFTED_DOWN, DIAMETER_CLASS_SHIFTED_UP,
)


_DIAMETER_CLASS_OFFSETS = {
    DIAMETER_CLASS_CENTERED: 2,
    DIAMETER_CLASS_SHIFTED_UP: 0,
    DIAMETER_CLASS_SHIFTED_DOWN: 4,
}


def diameter_class_cm(d_cm: int, mode: str = DIAMETER_CLASS_CENTERED) -> int:
    """Return the five-centimetre class for an integer diameter.

    The mode controls where the named class lies within its five integer
    diameters: centered maps 18..22 to 20, shifted up maps 20..24 to 20,
    and shifted down maps 16..20 to 20.
    """
    if mode not in DIAMETER_CLASS_MODES:
        raise ValueError(f'unknown diameter-class mode: {mode!r}')
    return int((int(d_cm) + _DIAMETER_CLASS_OFFSETS[mode]) // 5 * 5)


def current_diameter_class_mode() -> str:
    """Return the persisted site-wide diameter-class mode."""
    from apps.base.models import SiteSettings
    return SiteSettings.load().diameter_class_mode


def basal_area_m2(d_cm: int) -> float:
    """Basal area in square metres for a diameter in centimetres."""
    radius_m = float(d_cm) / 200.0
    return math.pi * radius_m * radius_m
