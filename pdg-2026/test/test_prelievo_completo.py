"""Tests for prelievo_completo: full-coverage harvest scheduling.

See docs/prelievo-completo.md for the design.
"""

from pdg.computation import COL_COMPRESA, COL_PARTICELLA
from pdg.simulation import COL_HARVEST, COL_YEAR, schedule_harvests

# Starves parcels A and D: five eligible parcels, three years, a target so
# small that one parcel fills it, and a rest period longer than the plan.
STARVED = dict(past_harvests=None, year_range=(2026, 2028), min_gap=10,
               target_volume=1.0)


def cut_parcels(events):
    return {(e[COL_COMPRESA], e[COL_PARTICELLA]) for e in events}


class TestEligibility:
    """schedule_harvests reports what each parcel could have yielded."""

    def test_records_parcels_the_loop_never_reached(self, data_all, harvest_rules):
        eligibility = {}
        events = schedule_harvests(data_all, rules=harvest_rules,
                                   eligibility=eligibility, **STARVED)
        committed = cut_parcels(events)
        assert committed == {('Test', 'B'), ('Test', 'C'), ('Test', 'E')}
        assert set(eligibility) - committed == {('Test', 'A'), ('Test', 'D')}

    def test_records_positive_harvests_only(self, data_all, harvest_rules):
        eligibility = {}
        schedule_harvests(data_all, rules=harvest_rules,
                          eligibility=eligibility, **STARVED)
        for by_year in eligibility.values():
            assert by_year
            assert all(v > 0 for v in by_year.values())

    def test_does_not_change_the_schedule(self, data_all, harvest_rules):
        """Recording eligibility must not alter which parcels are cut."""
        base = schedule_harvests(data_all, rules=harvest_rules, **STARVED)
        recorded = schedule_harvests(data_all, rules=harvest_rules,
                                     eligibility={}, **STARVED)
        assert base == recorded

    def test_resting_parcels_are_not_eligible(self, data_all, harvest_rules):
        """A parcel inside its rest period is skipped, not recorded."""
        eligibility = {}
        schedule_harvests(data_all, rules=harvest_rules, eligibility=eligibility,
                          past_harvests=None, year_range=(2026, 2027),
                          min_gap=10, target_volume=1e9)
        # Everything is cut in 2026, so nothing may be recorded for 2027.
        assert all(2027 not in by_year for by_year in eligibility.values())
