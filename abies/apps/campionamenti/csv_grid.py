"""Sample-grid CSV import core: the three-phase contract
(``resolve_columns`` → ``validate_rows`` → ``apply``) shared by the in-app
upload view and (later) bootstrap.

``validate_rows`` is pure — it performs no DB writes and resolves foreign keys
against an injected ``GridIndexes`` — so the same code can back a true
``--check`` dry-run against a staged index.
"""

from dataclasses import dataclass

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import Parcel, SampleArea
from apps.base.numparse import coord_float
from config import strings as S
from config.constants import (
    DEFAULT_RADIUS_M, FIELD_ALTITUDE, FIELD_LAT, FIELD_LON, FIELD_NOTE,
    FIELD_NUMBER, FIELD_PARCEL, FIELD_R_M,
)

# Column resolution (aliasing retained for now — see plan scope note).
GRID_CSV_REQUIRED = [S.CSV_COL_REGION, S.CSV_COL_PARCEL,
                     S.CSV_COL_SAMPLE_AREA, S.CSV_COL_LON, S.CSV_COL_LAT,
                     S.CSV_COL_ALT]
GRID_CSV_OPTIONAL = [S.CSV_COL_RADIUS]
GRID_CSV_ALIASES = {
    # CSV exports use Quota/Raggio.  Also accept the unit-bearing display labels
    # operators may get by exporting visible table data.
    S.CSV_COL_ALT: [S.CSV_COL_ALT, S.COL_ALT],
    S.CSV_COL_RADIUS: [S.CSV_COL_RADIUS, S.COL_RADIUS],
}
GRID_CSV_MISSING_LABELS = {
    S.CSV_COL_ALT: f'{S.COL_ALT} / {S.CSV_COL_ALT}',
    S.CSV_COL_RADIUS: f'{S.COL_RADIUS} / {S.CSV_COL_RADIUS}',
}


def resolve_columns(fieldnames):
    """Map each canonical column to the header present in ``fieldnames`` (via
    aliases).  Returns ``(found, missing)`` where ``found`` maps canonical →
    actual header and ``missing`` lists human labels for absent *required*
    columns.
    """
    found = {}
    missing = []
    available = set(fieldnames)
    for canonical in GRID_CSV_REQUIRED:
        candidates = list(dict.fromkeys(GRID_CSV_ALIASES.get(canonical, [canonical])))
        match = next((name for name in candidates if name in available), None)
        if match is None:
            missing.append(GRID_CSV_MISSING_LABELS.get(canonical, canonical))
        else:
            found[canonical] = match
    for canonical in GRID_CSV_OPTIONAL:
        candidates = list(dict.fromkeys(GRID_CSV_ALIASES.get(canonical, [canonical])))
        match = next((name for name in candidates if name in available), None)
        if match is not None:
            found[canonical] = match
    return found, missing


@dataclass
class GridIndexes:
    """The lookups ``validate_rows`` needs, injected so the same validation runs
    against the live DB (view) or a staged index (bootstrap)."""
    parcels: dict          # (region_name.lower(), parcel_name) -> Parcel
    existing_keys: set     # (region_id, number) already in the target grid


def db_indexes(grid) -> GridIndexes:
    """Build ``GridIndexes`` from the live database for the given grid."""
    parcels = {
        (p.region.name.lower(), p.name): p
        for p in Parcel.objects.select_related('region')
    }
    existing_keys = set(
        SampleArea.objects.filter(sample_grid=grid)
                          .values_list('parcel__region_id', FIELD_NUMBER)
    )
    return GridIndexes(parcels, existing_keys)


def validate_rows(reader, cols, idx: GridIndexes):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed_rows, errors)``.  Each parsed row is a dict ready for
    ``apply``.  ``errors`` are user-facing ``S.ERR_CSV_*`` strings keyed to the
    1-based data row number (header is row 1, first data row is row 2).
    """
    errors = []
    parsed = []
    seen_in_csv = set()
    for i, row in enumerate(reader, 2):
        compresa = row[cols[S.CSV_COL_REGION]].strip()
        particella = row[cols[S.CSV_COL_PARCEL]].strip()
        parcel = idx.parcels.get((compresa.lower(), particella))
        if parcel is None:
            errors.append(
                S.ERR_CSV_PARCEL_NOT_FOUND.format(i, compresa, particella))
            continue
        number = row[cols[S.CSV_COL_SAMPLE_AREA]].strip()
        key = (parcel.region_id, number)
        if key in idx.existing_keys or key in seen_in_csv:
            errors.append(S.ERR_CSV_ROW_AREA_DUPLICATE.format(
                i, compresa, particella, number))
            continue
        seen_in_csv.add(key)
        lat = reader.decimal(row.get(cols[S.CSV_COL_LAT]))
        lon = reader.decimal(row.get(cols[S.CSV_COL_LON]))
        if lat is None or lon is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_LAT}/{S.CSV_COL_LON}'))
            continue
        radius_col = cols.get(S.CSV_COL_RADIUS)
        altitude_col = cols[S.CSV_COL_ALT]
        raggio = (row.get(radius_col) or '').strip() if radius_col else ''
        r_m = reader.integer(raggio) if raggio else DEFAULT_RADIUS_M
        if r_m is None:  # present but unparseable → flag, don't silently default
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, radius_col))
            continue
        parsed.append({
            FIELD_PARCEL: parcel,
            FIELD_NUMBER: number,
            FIELD_LAT: coord_float(lat),
            FIELD_LON: coord_float(lon),
            FIELD_ALTITUDE: reader.integer(row.get(altitude_col)),
            FIELD_R_M: r_m,
            FIELD_NOTE: '',
        })
    return parsed, errors


def apply(grid, parsed) -> None:
    """Persist validated rows as ``SampleArea``s and mark digests stale.

    Wrapped in a transaction so a mid-batch failure leaves no partial rows.
    """
    with transaction.atomic():
        SampleArea.objects.bulk_create([
            SampleArea(
                sample_grid=grid,
                parcel=r[FIELD_PARCEL], number=r[FIELD_NUMBER],
                lat=r[FIELD_LAT], lon=r[FIELD_LON],
                altitude_m=r[FIELD_ALTITUDE], r_m=r[FIELD_R_M], note=r[FIELD_NOTE],
            )
            for r in parsed
        ])
        # Surveys digest carries N. aree totali per grid → stale when we add
        # areas to a grid that has surveys.
        mark_stale('grids', 'sample_areas', 'surveys', 'audit')
