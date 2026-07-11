"""Shared validation for harvest-plan item writes."""

from config import strings as S


_MISSING = object()
_CONSTRAINED_BY_HARVEST_FIELDS = (
    'region_id',
    'parcel_id',
    'damaged',
    'unhealthy',
    'psr',
)
_RELATED_ID_ALIASES = {
    'region_id': 'region',
    'parcel_id': 'parcel',
}


def _submitted_value(values, field):
    if field in values:
        return values[field]
    related_field = _RELATED_ID_ALIASES.get(field)
    if related_field in values:
        related = values[related_field]
        return related.id if related is not None else None
    return _MISSING


def plan_item_harvest_invariant_errors(item, values):
    """Reject edits that would desynchronise linked harvest rows.

    Harvest rows copy a plan item's target and three boolean flags, and DB
    triggers enforce equality only when harvest rows themselves are written.
    Therefore plan-item writes must reject changes to those fields once
    harvests point at the item.
    """
    changed = False
    for field in _CONSTRAINED_BY_HARVEST_FIELDS:
        value = _submitted_value(values, field)
        if value is not _MISSING and getattr(item, field) != value:
            changed = True
            break
    if changed and item.harvests.exists():
        return [S.ERR_PLAN_ITEM_LINKED_HARVESTS_INVARIANT]
    return []
