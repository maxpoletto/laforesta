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


class TestForcedHarvests:
    """Recovery harvests are applied on top of the ordinary schedule."""

    def test_forced_parcel_is_cut(self, data_all, harvest_rules):
        forced = {2027: [('Test', 'A')]}
        events = schedule_harvests(data_all, rules=harvest_rules,
                                   forced=forced, **STARVED)
        assert ('Test', 'A') in cut_parcels(events)

    def test_forced_harvest_does_not_displace_the_schedule(self, data_all,
                                                           harvest_rules):
        """A forced cut is added; it must not change what the loop chose."""
        base = schedule_harvests(data_all, rules=harvest_rules, **STARVED)
        forced = {2027: [('Test', 'A')]}
        full = schedule_harvests(data_all, rules=harvest_rules,
                                 forced=forced, **STARVED)
        assert all(e in full for e in base)
        assert len(full) == len(base) + 1

    def test_forced_harvest_respects_the_rest_period(self, data_all,
                                                     harvest_rules):
        """Forcing a parcel the loop already cut this year is a no-op."""
        forced = {2026: [('Test', 'C')]}   # C is cut normally in 2026
        base = schedule_harvests(data_all, rules=harvest_rules, **STARVED)
        full = schedule_harvests(data_all, rules=harvest_rules,
                                 forced=forced, **STARVED)
        assert full == base

    def test_forced_harvest_skipped_when_still_resting(self, data_all,
                                                       harvest_rules):
        """A parcel cut in 2026 cannot be forced again in 2027."""
        forced = {2027: [('Test', 'C')]}
        base = schedule_harvests(data_all, rules=harvest_rules, **STARVED)
        full = schedule_harvests(data_all, rules=harvest_rules,
                                 forced=forced, **STARVED)
        assert full == base
