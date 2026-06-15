"""Tabacchi volume equations for Italian forestry species.

Vendored from `pdg-2026/pdg/computation.py:TABACCHI` (which holds the
canonical parameter table along with covariance matrices and residual
variances used for confidence-interval computation in pdg-2026).
Abies only needs the volume coefficients `b`, so we keep just those.

Volume equation per species:
    if len(b) == 2:  V_dm3 = b[0] + b[1] * D²h
    if len(b) == 3:  V_dm3 = b[0] + b[1] * D²h + b[2] * D
    V_m3 = V_dm3 / 1000

D in cm, h in m.

Used by the campionamenti sampled-trees import core
(`apps/campionamenti/csv_trees.py`).  Manual
form entry uses the parallel JS implementation in
`apps/base/static/base/js/volume.js`.  A parity test in
`test/test_tabacchi.py` keeps the two in sync.
"""

from decimal import Decimal
from typing import Sequence


# Species name constants — must match Species.common_name in the Abies
# DB.  Same set as pdg-2026.
SP_ABETE = 'Abete'
SP_ACERO = 'Acero'
SP_CASTAGNO = 'Castagno'
SP_CERRO = 'Cerro'
SP_CILIEGIO = 'Ciliegio'
SP_DOUGLAS = 'Douglas'
SP_FAGGIO = 'Faggio'
SP_LECCIO = 'Leccio'
SP_ONTANO = 'Ontano'
SP_PINO_LARICIO = 'Pino Laricio'
SP_PINO_MARITTIMO = 'Pino Marittimo'
SP_PINO_NERO = 'Pino Nero'
SP_SORBO = 'Sorbo'


# Per-species `b` coefficient vector.  Length 2 or 3.
TABACCHI_B: dict[str, tuple[float, ...]] = {
    SP_ABETE:          (-1.8381,    3.7836e-2, 3.9934e-1),
    SP_ACERO:          ( 1.6905,    3.7082e-2),
    SP_CASTAGNO:       (-2.0,       3.6524e-2, 7.4466e-1),
    SP_CERRO:          (-4.3221e-2, 3.8079e-2),
    SP_CILIEGIO:       ( 2.3118,    3.1278e-2, 3.7159e-1),
    SP_DOUGLAS:        (-7.9946,    3.3343e-2, 1.2186),
    SP_FAGGIO:         ( 8.1151e-1, 3.8965e-2),
    SP_LECCIO:         (-2.2219,    3.9685e-2, 6.2762e-1),
    SP_ONTANO:         (-2.2932e1,  3.2641e-2, 2.991),
    SP_PINO_LARICIO:   ( 6.4383,    3.8594e-2),
    SP_PINO_MARITTIMO: ( 2.9963,    3.8302e-2),
    SP_PINO_NERO:      (-2.1480e1,  3.3448e-2, 2.9088),
    SP_SORBO:          ( 2.3118,    3.1278e-2, 3.7159e-1),
}


def tabacchi_volume_m3(d_cm: Decimal | float | int,
                       h_m: Decimal | float | int,
                       species_common_name: str) -> Decimal:
    """Compute tree volume in m³ via the species-specific Tabacchi equation.

    Raises ValueError if the species isn't in the Tabacchi table.

    >>> tabacchi_volume_m3(30, 20, 'Faggio')
    Decimal('0.7022')
    """
    b = TABACCHI_B.get(species_common_name)
    if b is None:
        raise ValueError(
            f"species '{species_common_name}' not in Tabacchi table"
        )
    return _volume_from_b(float(d_cm), float(h_m), b)


def _volume_from_b(d: float, h: float, b: Sequence[float]) -> Decimal:
    d2h = d * d * h
    if len(b) == 2:
        v_dm3 = b[0] + b[1] * d2h
    else:
        v_dm3 = b[0] + b[1] * d2h + b[2] * d
    return (Decimal(v_dm3) / Decimal(1000)).quantize(Decimal('0.0001'))


def has_species(species_common_name: str) -> bool:
    return species_common_name in TABACCHI_B
