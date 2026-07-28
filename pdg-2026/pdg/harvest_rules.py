"""
Harvest rules for the Bosco forest.

Encapsulates the forest-specific harvest policy that was previously
spread across three CSV files (comparti.csv, provv_eta.csv, provv_vol.csv).
"""

import math
from functools import partial
from typing import Callable

# Map economic class -> minimum standing value (m3/ha).
# Negative = coppice / ceduo (no volume-based harvest).
_MIN_VOLUME = {
    'A': 350, 'B': 350, 'C': 250, 'D': 250, 'E': 350,
    'F': -1,  # Ceduo
}

# (threshold as % of min volume, max harvest %)
_VOLUME_RULES = [
    (180, 25), (160, 20), (140, 15), (120, 10), (0, 0),
]

# (average age, max area % harvest) (None -> use volume table)
_AGE_RULES = [
    (60, None),  # age >= 60: use volume rules
    (30, 20),    # age >= 30: 20% of basal area
    (0,  15),    # age >= 0:  15% of basal area
]

HarvestRulesFunc = Callable[[str, int, float, float], tuple[float, float]]

def max_harvest(comparto: str, eta_media: int,
                volume_per_ha: float, area_basimetrica_per_ha: float,
                giovani_solo_area: bool = False) -> tuple[float, float]:
    """Maximum harvest limits for a parcel.

    Args:
        comparto: Compartment code (e.g. 'A')
        eta_media: Mean stand age (years)
        volume_per_ha: Volume of mature trees (D > 20cm), m3/ha
        area_basimetrica_per_ha: Basal area of mature trees, m2/ha
        giovani_solo_area: govern stands below the oldest age band by their
            basal-area limit alone, dropping the volume limit.  The volume
            table sets a floor (no harvest below 120% of the provvigione
            minima) that a young stand cannot meet, so with the default False
            the floor blocks it however much basal area it carries.  Stands in
            the oldest band are unaffected either way.

    Returns:
        (volume_limit_m3_ha, area_limit_m2_ha).
        math.inf = no limit on that dimension. (0, 0) = no harvest.
    """
    provv_min = _MIN_VOLUME.get(comparto)
    if provv_min is None:
        raise ValueError(f"Comparto sconosciuto: {comparto}")
    if provv_min < 0:
        return 0.0, 0.0  # Ceduo

    pp_max = _volume_pp_max(volume_per_ha, provv_min)
    vol_max = volume_per_ha * pp_max / 100
    for min_age, pp_max_basal in _AGE_RULES:
        if eta_media >= min_age:
            if pp_max_basal is None:  # oldest band: the volume rules govern
                return vol_max, math.inf
            area_max = area_basimetrica_per_ha * pp_max_basal / 100
            return (math.inf if giovani_solo_area else vol_max), area_max

    return 0.0, 0.0


# Harvest rule sets selectable from the command line.  'entrambi' applies both
# the volume and basal-area limits to young stands; 'area' applies only the
# basal-area limit.  Module-level singletons, so pdg.core.plan_events can key
# its cache on the rules object.
LIMITI_GIOVANI_ENTRAMBI = 'entrambi'
LIMITI_GIOVANI_AREA = 'area'

RULE_SETS: dict[str, HarvestRulesFunc] = {
    LIMITI_GIOVANI_ENTRAMBI: max_harvest,
    LIMITI_GIOVANI_AREA: partial(max_harvest, giovani_solo_area=True),
}


def _volume_pp_max(volume_per_ha, provv_minima):
    """Look up harvest % from volume rules.

    Compares actual volume to thresholds expressed as percentages
    of provvigione minima.
    """
    for threshold_ppm, pp_max in _VOLUME_RULES:
        if volume_per_ha > threshold_ppm * provv_minima / 100:
            return pp_max
    return 0.0
