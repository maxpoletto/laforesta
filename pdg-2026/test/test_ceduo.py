"""Tests for coppice scheduling algorithm."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdg.ceduo import (
    CoppiceParcel, CoppiceEvent, schedule_coppice,
    load_coppice_parcels, load_adjacencies, last_harvests_from_calendario,
    coppice_gantt_bars,
)

TEST_DATA_DIR = Path(__file__).parent / 'data'


class TestScheduleCoppice:
    """Test cases from ceduo.md examples + edge cases."""

    def test_standalone_small_parcel(self):
        """Example 1: 8 ha, intervallo=12, last harvest 2015 -> 2027, 2039."""
        parcels = [CoppiceParcel('X', 'A', 8.0, 12)]
        events = schedule_coppice(parcels, set(), {('X', 'A'): 2015}, (2027, 2045))
        years = [(e.year, e.area_ha) for e in events]
        assert years == [(2027, 8.0), (2039, 8.0)]
        assert all(e.cycle_start == e.year for e in events)

    def test_standalone_large_parcel(self):
        """Example 2: 25 ha, intervallo=12, last harvest 2015 -> 3 equal sub-harvests."""
        parcels = [CoppiceParcel('X', 'A', 25.0, 12)]
        events = schedule_coppice(parcels, set(), {('X', 'A'): 2015}, (2027, 2045))
        year_area = [(e.year, e.area_ha) for e in events]
        chunk = 25.0 / 3
        assert year_area == [
            (2027, chunk), (2029, chunk), (2031, chunk),
            (2039, chunk), (2041, chunk), (2043, chunk),
        ]
        # First sub-harvest of each cycle: cycle_start == year
        assert events[0].cycle_start == 2027
        assert events[3].cycle_start == 2039
        # Continuations reference the cycle start
        assert events[1].cycle_start == 2027
        assert events[2].cycle_start == 2027
        assert events[4].cycle_start == 2039
        assert events[5].cycle_start == 2039

    def test_exactly_10ha(self):
        """10 ha: single harvest, no sub-division."""
        parcels = [CoppiceParcel('X', 'A', 10.0, 12)]
        events = schedule_coppice(parcels, set(), {('X', 'A'): 2015}, (2027, 2045))
        year_area = [(e.year, e.area_ha) for e in events]
        assert year_area == [(2027, 10.0), (2039, 10.0)]

    def test_exactly_20ha(self):
        """20 ha: two sub-harvests of 10 ha each."""
        parcels = [CoppiceParcel('X', 'A', 20.0, 12)]
        events = schedule_coppice(parcels, set(), {('X', 'A'): 2015}, (2027, 2045))
        year_area = [(e.year, e.area_ha) for e in events]
        assert year_area == [
            (2027, 10.0), (2029, 10.0),
            (2039, 10.0), (2041, 10.0),
        ]

    def test_no_harvest_history(self):
        """Parcel with no past harvest: last_harvest=0, eligible from anno_inizio."""
        parcels = [CoppiceParcel('X', 'A', 8.0, 12)]
        events = schedule_coppice(parcels, set(), {}, (2027, 2045))
        # 0 + 12 = 12 < 2027, so eligible from 2027
        years = [e.year for e in events]
        assert years == [2027, 2039]

    def test_partial_cycle_at_end(self):
        """Sub-harvests that would exceed anno_fine are truncated."""
        parcels = [CoppiceParcel('X', 'A', 25.0, 12)]
        events = schedule_coppice(parcels, set(), {('X', 'A'): 2015}, (2027, 2040))
        year_area = [(e.year, e.area_ha) for e in events]
        # First cycle fits: 2027, 2029, 2031. Second cycle: 2039 fits, 2041 > 2040.
        chunk = 25.0 / 3
        assert year_area == [
            (2027, chunk), (2029, chunk), (2031, chunk),
            (2039, chunk),
        ]

    # ----- Adjacency constraint tests -----

    def test_adjacent_small_parcels(self):
        """Example 3: Two adjacent 8 ha, intervallo=12, last 2015/2016."""
        parcels = [
            CoppiceParcel('X', 'A', 8.0, 12),
            CoppiceParcel('X', 'B', 8.0, 12),
        ]
        adj = {(('X', 'A'), ('X', 'B'))}
        last = {('X', 'A'): 2015, ('X', 'B'): 2016}
        events = schedule_coppice(parcels, adj, last, (2027, 2045))
        a_years = [(e.year, e.area_ha) for e in events if e.particella == 'A']
        b_years = [(e.year, e.area_ha) for e in events if e.particella == 'B']
        assert a_years == [(2027, 8.0), (2039, 8.0)]
        assert b_years == [(2029, 8.0), (2041, 8.0)]

    def test_adjacent_large_parcels(self):
        """Example 4: Two adjacent 12 ha, intervallo=12, last 2015/2016. Each splits into 2x6 ha."""
        parcels = [
            CoppiceParcel('X', 'A', 12.0, 12),
            CoppiceParcel('X', 'B', 12.0, 12),
        ]
        adj = {(('X', 'A'), ('X', 'B'))}
        last = {('X', 'A'): 2015, ('X', 'B'): 2016}
        events = schedule_coppice(parcels, adj, last, (2027, 2050))
        a_events = [(e.year, e.area_ha) for e in events if e.particella == 'A']
        b_events = [(e.year, e.area_ha) for e in events if e.particella == 'B']
        assert a_events == [(2027, 6.0), (2029, 6.0), (2039, 6.0), (2041, 6.0)]
        assert b_events == [(2031, 6.0), (2033, 6.0), (2043, 6.0), (2045, 6.0)]

    def test_chain_adjacency_not_transitive(self):
        """A-B and B-C adjacent: A and C can be scheduled same year."""
        parcels = [
            CoppiceParcel('X', 'A', 8.0, 12),
            CoppiceParcel('X', 'B', 8.0, 12),
            CoppiceParcel('X', 'C', 8.0, 12),
        ]
        adj = {(('X', 'A'), ('X', 'B')), (('X', 'B'), ('X', 'C'))}
        # All same last harvest -> all eligible 2027
        last = {('X', 'A'): 2015, ('X', 'B'): 2015, ('X', 'C'): 2015}
        events = schedule_coppice(parcels, adj, last, (2027, 2045))
        a_years = [e.year for e in events if e.particella == 'A']
        b_years = [e.year for e in events if e.particella == 'B']
        c_years = [e.year for e in events if e.particella == 'C']
        # A gets 2027 (first in queue). C not adjacent to A, gets 2027 too.
        # B adjacent to both, pushed to 2029.
        assert a_years[0] == 2027
        assert c_years[0] == 2027
        assert b_years[0] == 2029

    def test_cross_compresa_no_adjacency(self):
        """Parcels in different comprese are never adjacent."""
        parcels = [
            CoppiceParcel('X', 'A', 8.0, 12),
            CoppiceParcel('Y', 'A', 8.0, 12),
        ]
        last = {('X', 'A'): 2015, ('Y', 'A'): 2015}
        events = schedule_coppice(parcels, set(), last, (2027, 2045))
        x_years = [e.year for e in events if e.compresa == 'X']
        y_years = [e.year for e in events if e.compresa == 'Y']
        assert x_years[0] == 2027
        assert y_years[0] == 2027

    def test_adjacency_sub_harvest_blocks_neighbor(self):
        """All sub-harvest years of a parcel block adjacent parcels."""
        # A is 25 ha (sub-harvests 2027, 2029, 2031). B is 8 ha, adjacent to A.
        parcels = [
            CoppiceParcel('X', 'A', 25.0, 12),
            CoppiceParcel('X', 'B', 8.0, 12),
        ]
        adj = {(('X', 'A'), ('X', 'B'))}
        last = {('X', 'A'): 2015, ('X', 'B'): 2015}
        events = schedule_coppice(parcels, adj, last, (2027, 2050))
        b_events = [e for e in events if e.particella == 'B']
        # B blocked by A's 2027, 2029, 2031 -> blocked [2026..2032]
        assert b_events[0].year == 2033


class TestIO:
    """Test I/O helpers for coppice data."""

    def test_load_coppice_parcels(self):
        """Loads only Ceduo parcels with correct fields."""
        parcels = load_coppice_parcels(TEST_DATA_DIR / 'particelle-ceduo.csv')
        assert len(parcels) == 2  # C is Fustaia, excluded
        assert parcels[0].compresa == 'X'
        assert parcels[0].particella == 'A'
        assert parcels[0].area_ha == 8.0
        assert parcels[0].intervallo == 12
        assert parcels[1].particella == 'B'
        assert parcels[1].intervallo == 15

    def test_load_adjacencies(self):
        """Builds sorted-pair adjacency set."""
        adj = load_adjacencies(TEST_DATA_DIR / 'adiacenze-ceduo.csv')
        assert (('X', 'A'), ('X', 'B')) in adj
        assert len(adj) == 1

    def test_last_harvests_from_calendario(self):
        """Extracts max year per Ceduo parcel, ignores Fustaia."""
        last = last_harvests_from_calendario(TEST_DATA_DIR / 'calendario-ceduo.csv')
        assert last[('X', 'A')] == 2015
        assert last[('X', 'B')] == 2018  # max of 2016, 2018
        assert ('X', 'C') not in last    # Fustaia excluded


def _evt(year, cycle_start, area_ha=10.0, intervallo=15,
         compresa='X', particella='A', area_totale_ha=10.0) -> CoppiceEvent:
    return CoppiceEvent(year, compresa, particella, area_ha,
                        area_totale_ha, intervallo, cycle_start)


class TestCoppiceGanttBars:
    """Test the Gantt bar layout for preserved-shoot batch lifetimes."""

    def test_10ha_single_sub_harvest(self):
        """User's first example: 10 ha / interval 15, single sub-harvest per cycle.

        Three cycles produce three bars on two alternating lanes.
        """
        parcel = CoppiceParcel('X', 'A', 10.0, 15)
        events = [
            _evt(1, 1), _evt(16, 16), _evt(31, 31),
        ]
        rows = coppice_gantt_bars([parcel], events)
        assert len(rows) == 1
        row = rows[0]
        assert (row.compresa, row.particella) == ('X', 'A')
        assert row.n_lanes == 2
        tuples = [(b.start_year, b.end_year, b.lane, b.cycle_idx, b.sub_idx, b.n_sub)
                  for b in row.bars]
        assert tuples == [
            (1, 31, 0, 1, 1, 1),
            (16, 46, 1, 2, 1, 1),
            (31, 61, 0, 3, 1, 1),
        ]

    def test_16ha_two_sub_harvests(self):
        """User's second example: 16 ha / interval 15, sub-harvests in consecutive years.

        Each sub-slot alternates between two lanes across cycles; the two sub-slots
        occupy disjoint lane pairs, giving 4 lanes total.
        """
        parcel = CoppiceParcel('X', 'A', 16.0, 15)
        events = [
            _evt(1, 1, area_ha=8.0, area_totale_ha=16.0),
            _evt(2, 1, area_ha=8.0, area_totale_ha=16.0),
            _evt(16, 16, area_ha=8.0, area_totale_ha=16.0),
            _evt(17, 16, area_ha=8.0, area_totale_ha=16.0),
            _evt(31, 31, area_ha=8.0, area_totale_ha=16.0),
            _evt(32, 31, area_ha=8.0, area_totale_ha=16.0),
        ]
        rows = coppice_gantt_bars([parcel], events)
        assert len(rows) == 1
        row = rows[0]
        assert row.n_lanes == 4
        tuples = sorted(
            (b.start_year, b.end_year, b.lane, b.cycle_idx, b.sub_idx, b.n_sub)
            for b in row.bars)
        assert tuples == [
            (1, 31, 0, 1, 1, 2), (2, 32, 2, 1, 2, 2),
            (16, 46, 1, 2, 1, 2), (17, 47, 3, 2, 2, 2),
            (31, 61, 0, 3, 1, 2), (32, 62, 2, 3, 2, 2),
        ]

    def test_empty_parcel_still_has_row(self):
        """Parcels with no scheduled events still produce a row (for display)."""
        parcel = CoppiceParcel('X', 'A', 8.0, 12)
        rows = coppice_gantt_bars([parcel], [])
        assert len(rows) == 1
        assert rows[0].bars == []
        assert rows[0].n_lanes == 2  # minimum row height

    def test_three_sub_harvests(self):
        """25 ha parcel: n_sub=3 (ceil(25/10)), so 6 lanes; each sub-slot uses its own pair."""
        parcel = CoppiceParcel('X', 'A', 25.0, 12)
        events = [
            _evt(2027, 2027, area_ha=10.0, area_totale_ha=25.0, intervallo=12),
            _evt(2029, 2027, area_ha=10.0, area_totale_ha=25.0, intervallo=12),
            _evt(2031, 2027, area_ha=5.0, area_totale_ha=25.0, intervallo=12),
            _evt(2039, 2039, area_ha=10.0, area_totale_ha=25.0, intervallo=12),
            _evt(2041, 2039, area_ha=10.0, area_totale_ha=25.0, intervallo=12),
            _evt(2043, 2039, area_ha=5.0, area_totale_ha=25.0, intervallo=12),
        ]
        rows = coppice_gantt_bars([parcel], events)
        row = rows[0]
        assert row.n_lanes == 6
        # Sub-slots 0,1,2 in cycle 0 → lanes 0,2,4; in cycle 1 → lanes 1,3,5.
        by_start = {b.start_year: b for b in row.bars}
        assert by_start[2027].lane == 0 and by_start[2027].cycle_idx == 1 and by_start[2027].sub_idx == 1
        assert by_start[2029].lane == 2 and by_start[2029].cycle_idx == 1 and by_start[2029].sub_idx == 2
        assert by_start[2031].lane == 4 and by_start[2031].cycle_idx == 1 and by_start[2031].sub_idx == 3
        assert by_start[2039].lane == 1 and by_start[2039].cycle_idx == 2 and by_start[2039].sub_idx == 1
        assert by_start[2041].lane == 3 and by_start[2041].cycle_idx == 2 and by_start[2041].sub_idx == 2
        assert by_start[2043].lane == 5 and by_start[2043].cycle_idx == 2 and by_start[2043].sub_idx == 3
        assert all(b.n_sub == 3 for b in row.bars)

    def test_partial_cycle_at_end(self):
        """A cycle truncated within the planning window still lays out correctly."""
        parcel = CoppiceParcel('X', 'A', 16.0, 15)
        events = [
            _evt(1, 1, area_ha=8.0, area_totale_ha=16.0),
            _evt(2, 1, area_ha=8.0, area_totale_ha=16.0),
            _evt(16, 16, area_ha=8.0, area_totale_ha=16.0),
        ]
        rows = coppice_gantt_bars([parcel], events)
        row = rows[0]
        assert row.n_lanes == 4
        tuples = sorted(
            (b.start_year, b.end_year, b.lane, b.cycle_idx, b.sub_idx)
            for b in row.bars)
        assert tuples == [(1, 31, 0, 1, 1), (2, 32, 2, 1, 2), (16, 46, 1, 2, 1)]

    def test_rows_follow_input_parcel_order(self):
        """Row order follows the input parcel list (natsort happens upstream)."""
        parcels = [
            CoppiceParcel('X', 'A', 8.0, 12),
            CoppiceParcel('X', 'B', 8.0, 12),
            CoppiceParcel('Y', 'A', 8.0, 12),
        ]
        rows = coppice_gantt_bars(parcels, [])
        assert [(r.compresa, r.particella) for r in rows] == [
            ('X', 'A'), ('X', 'B'), ('Y', 'A'),
        ]
