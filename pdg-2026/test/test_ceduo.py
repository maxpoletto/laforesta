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
