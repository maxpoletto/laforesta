"""Shared validation for harvest-plan item writes."""

from apps.base.models import HarvestPlanItem, Parcel
from apps.piano_di_taglio.mark_import import mark_parcel_matches_item
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


def plan_item_invariant_errors(item, values):
    """Keep linked harvests consistent and every existing mark in scope.

    Harvest rows copy a plan item's target and three boolean flags, and DB
    triggers enforce equality only when harvest rows themselves are written.
    Therefore plan-item writes must reject changes to those fields once
    harvests point at the item. Marked parcels may span a region, but must
    remain inside the proposed target; safe widening/narrowing is allowed.
    """
    proposed = {}
    for field in _CONSTRAINED_BY_HARVEST_FIELDS:
        value = _submitted_value(values, field)
        proposed[field] = getattr(item, field) if value is _MISSING else value
    changed = any(getattr(item, field) != value for field, value in proposed.items())
    if changed and item.harvests.exists():
        return [S.ERR_PLAN_ITEM_LINKED_HARVESTS_INVARIANT]

    if (proposed['region_id'], proposed['parcel_id']) != (item.region_id, item.parcel_id):
        target = HarvestPlanItem(
            region_id=proposed['region_id'], parcel_id=proposed['parcel_id'],
        )
        # Check each parcel once, not once per marked tree, using the same
        # scope predicate as manual, CSV and Ipso mark entry.
        marked_parcels = Parcel.objects.filter(
            pk__in=item.tree_marks.values('parcel_id'),
        ).only('id', 'region_id')
        if any(not mark_parcel_matches_item(target, parcel) for parcel in marked_parcels):
            return [S.ERR_PLAN_ITEM_MARKS_OUTSIDE_TARGET]
    return []
