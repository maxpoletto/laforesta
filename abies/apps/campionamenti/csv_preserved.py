"""Preserved-trees CSV import core: the three-phase contract
(``db_indexes`` → ``validate_rows`` → ``apply``) for the PAI/preserved-trees
bootstrap file.

Each valid row creates a physical ``Tree(preserved=True, coppice=False)`` and
one preserved ``TreeSample`` observation row.  Species are resolved by
``common_name.lower()`` (strict canonical — no GENERE_MAP aliasing).  Lon/Lat
are converted via ``reader.decimal`` → ``coord_float``.
"""

from dataclasses import dataclass
from datetime import date as date_type
from decimal import ROUND_HALF_UP

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import Parcel, Sample, Species, Survey, Tree, TreeSample
from apps.base.numparse import coord_float
from apps.base.preserved_trees import (
    PRESERVED_IMPORT_SURVEY_NAME, current_preserved_number_keys,
)
from config import strings as S
from config.constants import (
    BOSCO_TREE_DIGESTS, FIELD_ACC_M, FIELD_DATE, FIELD_D_CM,
    FIELD_ESTIMATED_BIRTH_YEAR, FIELD_H_M, FIELD_H_MEASURED, FIELD_LAT,
    FIELD_LON, FIELD_NOTE, FIELD_NUMBER, FIELD_OPERATOR, FIELD_PARCEL,
    FIELD_SPECIES, PRESSLER_DEFAULT, TREE_H_QUANTUM,
)

# Required columns for the canonical preserved-trees CSV.
PRESERVED_CSV_REQUIRED = [
    S.CSV_COL_REGION,
    S.CSV_COL_PARCEL,
    S.CSV_COL_SPECIES,
    S.CSV_COL_NUMBER,
    S.CSV_COL_LON,
    S.CSV_COL_LAT,
    S.CSV_COL_DATA,
    S.CSV_COL_D_CM,
    S.CSV_COL_H_M,
]


@dataclass
class PreservedIndexes:
    """Lookups ``validate_rows`` needs, injected so validation is pure."""
    parcels: dict             # (region_name.lower(), parcel_name) -> Parcel
    species: dict             # common_name.lower() -> Species
    preserved_numbers: set    # (parcel_id, number)


def db_indexes() -> PreservedIndexes:
    """Build ``PreservedIndexes`` from the live database."""
    parcels = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }
    species = {s.common_name.lower(): s for s in Species.objects.all()}
    preserved_numbers = current_preserved_number_keys()
    return PreservedIndexes(parcels, species, preserved_numbers)


def validate_rows(reader, idx: PreservedIndexes):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed, errors)``.  Each parsed row is a dict ready for
    ``apply``.  Unknown species or parcel is an error; unparseable required
    values are errors.
    """
    errors = []
    parsed = []
    seen_numbers = set(idx.preserved_numbers)
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

        number = reader.integer(row.get(S.CSV_COL_NUMBER))
        lat = coord_float(reader.decimal(row.get(S.CSV_COL_LAT)))
        lon = coord_float(reader.decimal(row.get(S.CSV_COL_LON)))
        if number is None or number <= 0 or lat is None or lon is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_NUMBER}/{S.CSV_COL_LAT}/{S.CSV_COL_LON}',
            ))
            continue
        number_key = (parcel.id, number)
        if number_key in seen_numbers:
            errors.append(S.ERR_CSV_DUPLICATE_KEY.format(
                i, f'{S.CSV_COL_PARCEL}/{S.CSV_COL_NUMBER}',
                f'{compresa} {particella}/{number}',
            ))
            continue
        seen_numbers.add(number_key)

        date, date_ok = _required_date(row.get(S.CSV_COL_DATA))
        birth_year, birth_ok = reader.opt_int(row.get(S.CSV_COL_ESTIMATED_BIRTH_YEAR, ''))
        d_cm, d_ok = reader.opt_int(row.get(S.CSV_COL_D_CM, ''))
        h_m, h_ok = reader.opt_decimal(row.get(S.CSV_COL_H_M, ''))
        h_measured, h_measured_ok = reader.opt_bool(row.get(S.CSV_COL_H_MEASURED, ''))
        acc_m, acc_ok = reader.opt_int(row.get(S.CSV_COL_ACC_M, ''))
        if not all((date_ok, birth_ok, d_ok, h_ok, h_measured_ok, acc_ok)):
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_FILE_PRESERVED_TREES))
            continue
        if d_cm is None or d_cm <= 0:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_COL_D_CM))
            continue
        if h_m is None or h_m <= 0:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_COL_H_M))
            continue
        h_m = h_m.quantize(TREE_H_QUANTUM, rounding=ROUND_HALF_UP)
        if h_measured is None:
            h_measured = True

        parsed.append({
            FIELD_PARCEL: parcel,
            FIELD_SPECIES: sp,
            FIELD_NUMBER: number,
            FIELD_DATE: date,
            FIELD_ESTIMATED_BIRTH_YEAR: birth_year,
            FIELD_D_CM: d_cm,
            FIELD_H_M: h_m,
            FIELD_H_MEASURED: h_measured,
            FIELD_LAT: lat,
            FIELD_LON: lon,
            FIELD_ACC_M: acc_m,
            FIELD_OPERATOR: (row.get(S.CSV_COL_OPERATOR) or '').strip(),
            FIELD_NOTE: (row.get(S.CSV_COL_NOTE) or '').strip(),
        })
    return parsed, errors


def _required_date(value):
    raw = (value or '').strip()
    if not raw:
        return None, False
    try:
        return date_type.fromisoformat(raw), True
    except ValueError:
        return None, False


def apply(parsed) -> int:
    """Persist validated rows as PAI observations.  Returns created count."""
    with transaction.atomic():
        survey, _ = Survey.objects.get_or_create(
            name=PRESERVED_IMPORT_SURVEY_NAME,
            defaults={
                'sample_grid': None,
                'description': 'Rilevamento libero per alberi PAI.',
                'active': False,
            },
        )
        if survey.sample_grid_id is not None:
            raise RuntimeError(
                f'Existing survey {PRESERVED_IMPORT_SURVEY_NAME!r} is structured.'
            )
        if survey.active:
            survey.active = False
            survey.save(update_fields=['active'])
        sample_by_key = {}
        for r in parsed:
            key = (r[FIELD_DATE], r[FIELD_PARCEL].id)
            sample = sample_by_key.get(key)
            if sample is None:
                sample = Sample.objects.create(
                    sample_area=None, survey=survey, date=r[FIELD_DATE],
                )
                sample_by_key[key] = sample
            tree = Tree.objects.create(
                species=r[FIELD_SPECIES],
                parcel=r[FIELD_PARCEL],
                estimated_birth_year=r[FIELD_ESTIMATED_BIRTH_YEAR],
                lat=r[FIELD_LAT],
                lon=r[FIELD_LON],
                acc_m=r[FIELD_ACC_M],
                preserved=True,
                coppice=False,
            )
            TreeSample.objects.create(
                sample=sample,
                tree=tree,
                parcel=r[FIELD_PARCEL],
                number=r[FIELD_NUMBER],
                preserved_number=r[FIELD_NUMBER],
                shoot=0,
                standard=False,
                d_cm=r[FIELD_D_CM],
                h_m=r[FIELD_H_M],
                h_measured=r[FIELD_H_MEASURED],
                l10_mm=0,
                pressler_coeff=PRESSLER_DEFAULT,
                volume_m3=None,
                mass_q=None,
                lat=r[FIELD_LAT],
                lon=r[FIELD_LON],
                acc_m=r[FIELD_ACC_M],
                operator=r[FIELD_OPERATOR],
                note=r[FIELD_NOTE],
            )
        mark_stale(*BOSCO_TREE_DIGESTS, 'audit')
    return len(parsed)
