"""Serialized intervention totals; derived writes do not invalidate edit forms."""
from django.db import connection, transaction
from django.db.models import F, Sum

from apps.base.models import HarvestPlanItem, HarvestPlanItemState, TreeMark


def lock_plan_items(item_ids):
    """Lock parents in ID order before mutating children or reading aggregates."""
    if not connection.in_atomic_block:
        raise RuntimeError('Plan-item writes require transaction.atomic()')
    ids = sorted({pk for pk in item_ids if pk is not None})
    queryset = HarvestPlanItem.objects.filter(pk__in=ids).order_by('pk')
    if not connection.features.has_select_for_update:
        # Claim SQLite's write boundary without changing clocks/history/tokens.
        queryset.update(version=F('version'))
    return {item.pk: item for item in queryset.select_for_update(
        of=('self',), no_key=connection.features.has_select_for_no_key_update,
    )}


@transaction.atomic
def refresh_marked_total(item_id, *, mark_date=None):
    item = lock_plan_items([item_id]).get(item_id)
    if item is None:
        return None
    total = TreeMark.objects.filter(harvest_plan_item_id=item_id).aggregate(total=Sum('volume_m3'))['total']
    fields = []
    if item.volume_marked_m3 != total:
        item.volume_marked_m3 = total
        fields.append('volume_marked_m3')
    if mark_date is not None and item.state == HarvestPlanItemState.PLANNED:
        item.state = HarvestPlanItemState.MARKED
        item.date_actual = mark_date
        item.version += 1
        fields.extend(['state', 'date_actual', 'version'])
    if fields:
        item.save(update_fields=fields)
    return item
