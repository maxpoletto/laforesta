"""Tests for coppice scheduling algorithm."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdg.ceduo import CoppiceParcel, CoppiceEvent, schedule_coppice


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
        """Example 2: 25 ha, intervallo=12, last harvest 2015 -> sub-harvests."""
        parcels = [CoppiceParcel('X', 'A', 25.0, 12)]
        events = schedule_coppice(parcels, set(), {('X', 'A'): 2015}, (2027, 2045))
        year_area = [(e.year, e.area_ha) for e in events]
        assert year_area == [
            (2027, 10.0), (2029, 10.0), (2031, 5.0),
            (2039, 10.0), (2041, 10.0), (2043, 5.0),
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
        assert year_area == [
            (2027, 10.0), (2029, 10.0), (2031, 5.0),
            (2039, 10.0),
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
        """Example 4: Two adjacent 12 ha, intervallo=12, last 2015/2016."""
        parcels = [
            CoppiceParcel('X', 'A', 12.0, 12),
            CoppiceParcel('X', 'B', 12.0, 12),
        ]
        adj = {(('X', 'A'), ('X', 'B'))}
        last = {('X', 'A'): 2015, ('X', 'B'): 2016}
        events = schedule_coppice(parcels, adj, last, (2027, 2050))
        a_events = [(e.year, e.area_ha) for e in events if e.particella == 'A']
        b_events = [(e.year, e.area_ha) for e in events if e.particella == 'B']
        assert a_events == [(2027, 10.0), (2029, 2.0), (2039, 10.0), (2041, 2.0)]
        assert b_events == [(2031, 10.0), (2033, 2.0), (2043, 10.0), (2045, 2.0)]

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
