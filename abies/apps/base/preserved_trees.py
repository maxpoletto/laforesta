"""Helpers for preserved/PAI trees backed by TreeSample rows."""

from __future__ import annotations

from django.db.models import Max, OuterRef, Subquery

from apps.base.models import TreeSample

PRESERVED_HISTORY_SURVEY_NAME = 'Alberi da preservare - storico'
PRESERVED_IMPORT_SURVEY_NAME = 'Alberi da preservare'


def latest_preserved_tree_samples(*, for_update: bool = False):
    """Return the current PAI row for each parcel-scoped preserved number.

    Current state is the latest TreeSample by sample date, with row id as a
    deterministic tie-breaker. Callers can add select_related/order_by filters.
    """
    base = TreeSample.objects.filter(preserved_number__isnull=False)
    latest_id = (
        base
        .filter(
            parcel_id=OuterRef('parcel_id'),
            preserved_number=OuterRef('preserved_number'),
        )
        .order_by('-sample__date', '-id')
        .values('id')[:1]
    )
    qs = base.filter(id=Subquery(latest_id))
    return qs.select_for_update() if for_update else qs


def latest_preserved_tree_sample(
        parcel_id: int, preserved_number: int, *, for_update: bool = False,
):
    qs = (
        TreeSample.objects
        .filter(parcel_id=parcel_id, preserved_number=preserved_number)
        .order_by('-sample__date', '-id')
    )
    if for_update:
        qs = qs.select_for_update()
    return qs.first()


def current_preserved_number_keys(parcel_ids: set[int] | None = None) -> set[tuple[int, int]]:
    qs = latest_preserved_tree_samples()
    if parcel_ids is not None:
        qs = qs.filter(parcel_id__in=parcel_ids)
    return set(qs.values_list('parcel_id', 'preserved_number'))


def preserved_number_exists(
        *, parcel_id: int, preserved_number: int, exclude_id: int | None = None,
) -> bool:
    qs = latest_preserved_tree_samples().filter(
        parcel_id=parcel_id, preserved_number=preserved_number,
    )
    if exclude_id is not None:
        qs = qs.exclude(id=exclude_id)
    return qs.exists()


def next_preserved_number(parcel_id: int, exclude_id: int | None = None) -> int:
    qs = latest_preserved_tree_samples().filter(parcel_id=parcel_id)
    if exclude_id is not None:
        qs = qs.exclude(id=exclude_id)
    return (qs.aggregate(max_number=Max('preserved_number'))['max_number'] or 0) + 1
