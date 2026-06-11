"""Shared query selectors for runtime defaults."""

from django.utils import timezone

from apps.base.models import HarvestPlan, Survey


def active_or_default_harvest_plan():
    """Harvest plan used by Bosco future production settings and digests.

    An explicitly active plan wins.  Otherwise use the plan whose range
    includes the current year and has the greatest end year.
    """
    active = HarvestPlan.objects.filter(active=True).order_by('-year_end', 'id').first()
    if active is not None:
        return active
    year = timezone.localdate().year
    return (HarvestPlan.objects
            .filter(year_start__lte=year, year_end__gte=year)
            .order_by('-year_end', 'id')
            .first())


def active_or_default_survey_ids() -> list[int]:
    """Survey ids used by Bosco dendrometry settings and digests."""
    ids = list(Survey.objects.filter(active=True).order_by('name')
               .values_list('id', flat=True))
    if ids:
        return ids
    first = Survey.objects.order_by('name').values_list('id', flat=True).first()
    return [first] if first else []
