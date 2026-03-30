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
