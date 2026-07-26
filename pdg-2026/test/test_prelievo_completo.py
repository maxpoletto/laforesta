"""Tests for prelievo_completo: full-coverage harvest scheduling.

See docs/prelievo-completo.md for the design.
"""

from pdg.computation import COL_COMPRESA, COL_PARTICELLA
from pdg.core import OPT_PRELIEVO_COMPLETO, plan_events
from pdg.simulation import (
    COL_HARVEST, COL_YEAR, place_debts, schedule_harvests,
    schedule_harvests_complete,
)

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


class TestPlaceDebts:
    """Debts are assigned to years deterministically, levelling the load."""

    def test_picks_the_lowest_total_eligible_year(self):
        eligibility = {('R', 'p'): {2030: 100.0, 2031: 100.0}}
        assert place_debts(eligibility, set(), {2030: 500.0, 2031: 200.0}) == \
            {('R', 'p'): 2031}

    def test_ties_go_to_the_earliest_year(self):
        eligibility = {('R', 'p'): {2030: 10.0, 2031: 10.0}}
        assert place_debts(eligibility, set(), {2030: 100.0, 2031: 100.0}) == \
            {('R', 'p'): 2030}

    def test_committed_parcels_are_not_debts(self):
        eligibility = {('R', 'p'): {2030: 10.0}}
        assert place_debts(eligibility, {('R', 'p')}, {2030: 0.0}) == {}

    def test_largest_debt_of_a_year_is_placed_first(self):
        """Both come due in 2030; the big one picks the emptier year."""
        eligibility = {
            ('R', 'big'): {2030: 100.0, 2031: 100.0},
            ('R', 'small'): {2030: 10.0, 2031: 10.0},
        }
        out = place_debts(eligibility, set(), {2030: 50.0, 2031: 0.0})
        assert out == {('R', 'big'): 2031, ('R', 'small'): 2030}

    def test_earlier_debt_year_is_settled_first(self):
        eligibility = {
            ('R', 'late'): {2031: 100.0},
            ('R', 'early'): {2030: 10.0, 2031: 10.0},
        }
        out = place_debts(eligibility, set(), {2030: 20.0, 2031: 0.0})
        assert out == {('R', 'early'): 2031, ('R', 'late'): 2031}

    def test_result_is_independent_of_input_order(self):
        a = {('R', 'x'): {2030: 5.0, 2031: 5.0},
             ('R', 'y'): {2030: 5.0, 2031: 5.0}}
        b = dict(reversed(list(a.items())))
        totals = {2030: 0.0, 2031: 0.0}
        assert place_debts(a, set(), totals) == place_debts(b, set(), totals)

    def test_parcel_names_sort_naturally(self):
        """Tie-break uses natural order: p2 before p10."""
        eligibility = {('R', 'p10'): {2030: 5.0, 2031: 5.0},
                       ('R', 'p2'): {2030: 5.0, 2031: 5.0}}
        out = place_debts(eligibility, set(), {2030: 0.0, 2031: 0.0})
        assert out == {('R', 'p2'): 2030, ('R', 'p10'): 2031}


class TestScheduleHarvestsComplete:
    """No rules-eligible parcel is left uncut."""

    def test_covers_every_eligible_parcel(self, data_all, harvest_rules):
        eligibility = {}
        base = schedule_harvests(data_all, rules=harvest_rules,
                                 eligibility=eligibility, **STARVED)
        ever_eligible = set(eligibility)
        assert ever_eligible - cut_parcels(base), "fixture no longer starves"

        full = schedule_harvests_complete(data_all, rules=harvest_rules,
                                          **STARVED)
        assert ever_eligible <= cut_parcels(full)

    def test_is_additive(self, data_all, harvest_rules):
        """The base schedule survives; recovery only appends."""
        base = schedule_harvests(data_all, rules=harvest_rules, **STARVED)
        full = schedule_harvests_complete(data_all, rules=harvest_rules,
                                          **STARVED)
        assert all(e in full for e in base)
        assert len(full) > len(base)

    def test_respects_the_rest_period(self, data_all, harvest_rules):
        full = schedule_harvests_complete(data_all, rules=harvest_rules,
                                          **STARVED)
        last = {}
        for e in sorted(full, key=lambda e: e[COL_YEAR]):
            key = (e[COL_COMPRESA], e[COL_PARTICELLA])
            if key in last:
                assert e[COL_YEAR] - last[key] >= 10
            last[key] = e[COL_YEAR]

    def test_no_change_when_nothing_is_starved(self, data_all, harvest_rules):
        """A target large enough to cut everything leaves the plan alone."""
        kw = dict(past_harvests=None, year_range=(2026, 2027), min_gap=10,
                  target_volume=1e9, rules=harvest_rules)
        assert schedule_harvests_complete(data_all, **kw) == \
            schedule_harvests(data_all, **kw)

    def test_is_deterministic(self, data_all, harvest_rules):
        a = schedule_harvests_complete(data_all, rules=harvest_rules, **STARVED)
        b = schedule_harvests_complete(data_all, rules=harvest_rules, **STARVED)
        assert a == b


class TestDirectiveOption:
    """prelievo_completo reaches the scheduler and keys the plan cache."""

    def test_option_name(self):
        assert OPT_PRELIEVO_COMPLETO == 'prelievo_completo'

    def test_default_matches_plain_scheduling(self, data_all, harvest_rules):
        events = plan_events(data_all, rules=harvest_rules, **STARVED)
        assert cut_parcels(events) == {('Test', 'B'), ('Test', 'C'),
                                       ('Test', 'E')}

    def test_complete_covers_every_eligible_parcel(self, data_all,
                                                   harvest_rules):
        events = plan_events(data_all, rules=harvest_rules,
                             complete=True, **STARVED)
        assert {('Test', 'A'), ('Test', 'D')} <= cut_parcels(events)

    def test_cache_distinguishes_complete(self, data_all, harvest_rules):
        plain = plan_events(data_all, rules=harvest_rules, **STARVED)
        full = plan_events(data_all, rules=harvest_rules,
                           complete=True, **STARVED)
        assert plain is not full
        assert plan_events(data_all, rules=harvest_rules,
                           complete=True, **STARVED) is full
