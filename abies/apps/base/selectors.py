"""Shared query selectors for runtime defaults."""

from django.utils import timezone

from apps.base.models import HarvestPlan, HypsoParamSet, HypsoParamSource, Survey


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
    ids = list(
        Survey.objects
        .filter(active=True, sample_grid__isnull=False)
        .order_by('name')
        .values_list('id', flat=True)
    )
    if ids:
        return ids
    first = (
        Survey.objects
        .filter(sample_grid__isnull=False)
        .order_by('name')
        .values_list('id', flat=True)
        .first()
    )
    return [first] if first else []


def height_plot_survey_ids() -> list[int]:
    """Survey ids used by Bosco's height scatter plot.

    A computed hypsometric parameter set can opt the height plot into the
    surveys that produced those parameters.  Otherwise the plot follows the
    normal dendrometry survey setting.
    """
    active = (HypsoParamSet.objects.active()
              .filter(source=HypsoParamSource.COMPUTED,
                      use_for_height_plots=True)
              .first())
    if active is not None:
        ids = list(active.surveys.order_by('name').values_list('id', flat=True))
        if ids:
            return ids
    return active_or_default_survey_ids()
