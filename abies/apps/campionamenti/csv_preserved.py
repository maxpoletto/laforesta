"""Preserved-trees CSV import core: the three-phase contract
(``db_indexes`` → ``validate_rows`` → ``apply``) for the PAI/preserved-trees
bootstrap file.

Simpler than ``csv_trees`` (no Sample/TreeSample, no survey): each row maps
directly to a ``Tree(preserved=True, coppice=False)``.  Species resolved by
``common_name.lower()`` (strict canonical — no GENERE_MAP aliasing).  Lon/Lat
converted via ``reader.decimal`` → ``coord_float``.
"""

from dataclasses import dataclass

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import Parcel, Species, Tree
from apps.base.numparse import coord_float
from config import strings as S
from config.constants import BOSCO_TREE_DIGESTS, FIELD_LAT, FIELD_LON, FIELD_PARCEL, FIELD_SPECIES

# Required columns for the canonical preserved-trees CSV.
PRESERVED_CSV_REQUIRED = [
    S.CSV_COL_REGION,
    S.CSV_COL_PARCEL,
    S.CSV_COL_SPECIES,
    S.CSV_COL_LON,
    S.CSV_COL_LAT,
]


@dataclass
class PreservedIndexes:
    """Lookups ``validate_rows`` needs, injected so validation is pure."""
    parcels: dict     # (region_name.lower(), parcel_name) -> Parcel
    species: dict     # common_name.lower() -> Species


def db_indexes() -> PreservedIndexes:
    """Build ``PreservedIndexes`` from the live database."""
    parcels = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }
    species = {s.common_name.lower(): s for s in Species.objects.all()}
    return PreservedIndexes(parcels, species)


def validate_rows(reader, idx: PreservedIndexes):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed, errors)``.  Each parsed row is a dict ready for
    ``apply``.  Unknown species or parcel is an error; unparseable Lon/Lat
    is an error.
    """
    errors = []
    parsed = []
    for i, row in enumerate(reader, 2):
        compresa = (row.get(S.CSV_COL_REGION) or '').strip()
        particella = (row.get(S.CSV_COL_PARCEL) or '').strip()
        parcel = idx.parcels.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(i, compresa, particella))
            continue

        genere = (row.get(S.CSV_COL_SPECIES) or '').strip()
        sp = idx.species.get(genere.lower())
        if sp is None:
            errors.append(S.ERR_CSV_ROW_SPECIES.format(i, genere))
            continue

        lat = coord_float(reader.decimal(row.get(S.CSV_COL_LAT)))
        lon = coord_float(reader.decimal(row.get(S.CSV_COL_LON)))
        if lat is None or lon is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_LAT}/{S.CSV_COL_LON}'))
            continue

        parsed.append({
            FIELD_PARCEL: parcel,
            FIELD_SPECIES: sp,
            FIELD_LAT: lat,
            FIELD_LON: lon,
        })
    return parsed, errors


def apply(parsed) -> int:
    """Persist validated rows as preserved Trees.  Returns count of created trees."""
    with transaction.atomic():
        for r in parsed:
            Tree.objects.create(
                species=r[FIELD_SPECIES],
                parcel=r[FIELD_PARCEL],
                lat=r[FIELD_LAT],
                lon=r[FIELD_LON],
                preserved=True,
                coppice=False,
            )
        mark_stale(*BOSCO_TREE_DIGESTS, 'audit')
    return len(parsed)
