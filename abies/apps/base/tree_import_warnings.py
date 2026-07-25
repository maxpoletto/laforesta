"""Non-blocking warnings shared by tree CSV and Ipso imports."""

from __future__ import annotations

from apps.base.parcel_geometry import ParcelGeometryIndex
from config import strings as S
from config.constants import FIELD_H_MEASURED, FIELD_LAT, FIELD_LON, FIELD_PARCEL


def tree_import_warnings(
        rows: list[dict], *, first_row_number: int,
        geometry: ParcelGeometryIndex | None = None,
) -> list[str]:
    """Return warning strings for validated tree-import rows.

    Rows are already validated and normalized.  Warnings never mutate the row:
    operator-provided parcel data remains authoritative when GPS disagrees.
    """
    geometry = geometry or ParcelGeometryIndex()
    warnings = []
    for row_number, row in enumerate(rows, start=first_row_number):
        if not bool(row.get(FIELD_H_MEASURED)):
            warnings.append(S.WARN_IMPORT_H_NOT_MEASURED.format(row_number))

        provided = row.get(FIELD_PARCEL)
        if provided is None or row.get(FIELD_LAT) is None or row.get(FIELD_LON) is None:
            continue
        gps_parcel = geometry.parcel_at(row.get(FIELD_LAT), row.get(FIELD_LON))
        if gps_parcel is not None and gps_parcel.id != provided.id:
            warnings.append(S.WARN_IMPORT_PARCEL_MISMATCH.format(
                row_number, _parcel_label(gps_parcel), _parcel_label(provided),
            ))
    return warnings


def _parcel_label(parcel) -> str:
    return f'{parcel.region.name} / {parcel.name}'
