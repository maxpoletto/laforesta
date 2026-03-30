"""Coppice (ceduo) scheduling: rotation-based harvest calendar with adjacency constraints."""

import heapq
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from natsort import natsort_keygen

from pdg.computation import COL_COMPRESA, COL_PARTICELLA, COL_GOVERNO, COL_AREA_PARCEL

# Domain constants
MAX_HARVEST_AREA_HA = 10   # Maximum area per sub-harvest
MIN_ADJACENCY_GAP = 2      # Minimum years between adjacent parcel events
SUB_HARVEST_GAP = 2         # Minimum years between sub-harvests of same parcel

# Column names specific to coppice output
COL_YEAR = 'year'
COL_AREA_HA = 'area_ha'
COL_CYCLE_START = 'cycle_start'

# Column names in input CSVs
COL_PARAMETRO = 'Parametro'
COL_ANNO = 'Anno'
GOV_CEDUO = 'Ceduo'
COL_ADJ_A = 'A'
COL_ADJ_B = 'B'


@dataclass
class CoppiceParcel:
    """A coppice parcel eligible for scheduled harvest."""
    compresa: str
    particella: str
    area_ha: float
    intervallo: int


@dataclass
class CoppiceEvent:
    """One scheduled harvest (or sub-harvest) event."""
    year: int
    compresa: str
    particella: str
    area_ha: float
    cycle_start: int  # year of first sub-harvest in this cycle (== year for first)


ParcelKey = tuple[str, str]  # (compresa, particella)
Adjacencies = set[tuple[ParcelKey, ParcelKey]]  # sorted pairs: first < second


def _has_adjacency_conflict(key: ParcelKey, year: int,
                            adjacencies: Adjacencies,
                            scheduled_years: dict[ParcelKey, list[int]]) -> bool:
    """Check if scheduling key in year conflicts with any adjacent parcel."""
    for a, b in adjacencies:
        if a == key:
            other = b
        elif b == key:
            other = a
        else:
            continue
        for adj_year in scheduled_years.get(other, []):
            if abs(year - adj_year) < MIN_ADJACENCY_GAP:
                return True
    return False


def _schedule_one_cycle(
    parcel: CoppiceParcel, key: ParcelKey,
    eligible: int, last_year: int,
    adjacencies: Adjacencies,
    scheduled_years: dict[ParcelKey, list[int]],
) -> list[CoppiceEvent]:
    """Schedule one harvest cycle (possibly multiple sub-harvests) for a parcel."""
    remaining = parcel.area_ha
    year = eligible
    cycle_events: list[CoppiceEvent] = []
    cycle_start: int | None = None

    while remaining > 0 and year <= last_year:
        if _has_adjacency_conflict(key, year, adjacencies, scheduled_years):
            year += 1
            continue

        chunk = min(MAX_HARVEST_AREA_HA, remaining)
        if cycle_start is None:
            cycle_start = year
        cycle_events.append(CoppiceEvent(
            year, parcel.compresa, parcel.particella, chunk, cycle_start))

        remaining -= chunk
        year += SUB_HARVEST_GAP

    return cycle_events


def schedule_coppice(
    parcels: list[CoppiceParcel],
    adjacencies: Adjacencies,
    last_harvests: dict[ParcelKey, int],
    year_range: tuple[int, int],
) -> list[CoppiceEvent]:
    """Schedule coppice harvests using a priority queue.

    Args:
        parcels: Ceduo parcels with area and rotation interval.
        adjacencies: Set of sorted (parcel_key_a, parcel_key_b) pairs, a < b.
        last_harvests: Most recent harvest year per parcel (0 if unknown).
        year_range: (first_year, last_year) inclusive planning window.

    Returns:
        List of CoppiceEvent sorted by (year, compresa, particella).
    """
    first_year, last_year = year_range
    natsort_key = natsort_keygen()

    # Priority queue: (eligible_year, compresa, sort_key, particella, parcel)
    heap: list[tuple[int, str, object, str, CoppiceParcel]] = []
    for p in parcels:
        last = last_harvests.get((p.compresa, p.particella), 0)
        eligible = max(first_year, last + p.intervallo)
        heapq.heappush(heap, (eligible, p.compresa, natsort_key(p.particella),
                               p.particella, p))

    scheduled_years: dict[ParcelKey, list[int]] = {}
    events: list[CoppiceEvent] = []

    while heap:
        eligible, _, _, _, parcel = heapq.heappop(heap)
        if eligible > last_year:
            continue

        key = (parcel.compresa, parcel.particella)
        cycle_events = _schedule_one_cycle(
            parcel, key, eligible, last_year, adjacencies, scheduled_years)

        if not cycle_events:
            continue

        events.extend(cycle_events)
        scheduled_years.setdefault(key, []).extend(e.year for e in cycle_events)

        # Re-insert for next cycle: intervallo years after first sub-harvest
        next_eligible = cycle_events[0].year + parcel.intervallo
        if next_eligible <= last_year:
            heapq.heappush(heap, (
                next_eligible, parcel.compresa,
                natsort_key(parcel.particella), parcel.particella, parcel))

    events.sort(key=lambda e: (e.year, e.compresa, natsort_key(e.particella)))
    return events
