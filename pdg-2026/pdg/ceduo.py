"""Coppice (ceduo) scheduling: rotation-based harvest calendar with adjacency constraints."""

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from natsort import natsort_keygen

from pdg.computation import COL_COMPRESA, COL_PARTICELLA, COL_GOVERNO, COL_AREA_PARCEL, GOV_CEDUO

# Domain constants
MAX_HARVEST_AREA_HA = 10   # Maximum area per sub-harvest
MIN_ADJACENCY_GAP = 2      # Minimum years between adjacent parcel events
SUB_HARVEST_GAP = 2         # Minimum years between sub-harvests of same parcel

# Column names specific to coppice output
COL_YEAR = 'year'
COL_AREA_HA = 'area_ha'
COL_AREA_TOTALE_HA = 'area_totale_ha'
COL_INTERVALLO = 'intervallo'
COL_CYCLE_START = 'cycle_start'

# Column names in input CSVs
COL_PARAMETRO = 'Parametro'
COL_ANNO = 'Anno'
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
    area_totale_ha: float  # total parcel area
    intervallo: int        # rotation interval (years)
    cycle_start: int       # year of first sub-harvest in this cycle (== year for first)


@dataclass
class CoppiceBar:
    """One gantt-chart bar: the lifetime of a preserved-shoot batch seeded by one sub-harvest."""
    start_year: int           # harvest year (batch seeded)
    end_year: int             # start_year + 2 * intervallo (batch cut)
    lane: int                 # 0..n_lanes-1 within the parcel row
    cycle_idx: int            # 1-based index of the harvest cycle on this parcel
    sub_idx: int              # 1-based sub-harvest index within the cycle
    n_sub: int                # sub-harvests per cycle for this parcel (1 if ≤ 10 ha)


@dataclass
class CoppiceRow:
    """One parcel's row in the gantt chart."""
    compresa: str
    particella: str
    n_lanes: int              # 2 * n_sub_harvests, ≥ 2 even for empty rows
    bars: list[CoppiceBar]


ParcelKey = tuple[str, str]  # (compresa, particella)
Adjacencies = set[tuple[ParcelKey, ParcelKey]]  # sorted pairs: first < second


# =============================================================================
# I/O
# =============================================================================

def load_coppice_parcels(particelle_path: Path) -> list[CoppiceParcel]:
    """Load coppice parcels from particelle.csv (Governo=Ceduo only)."""
    df = pd.read_csv(particelle_path, encoding='utf-8-sig', comment='#')
    df = df[df[COL_GOVERNO] == GOV_CEDUO]
    df[COL_PARTICELLA] = df[COL_PARTICELLA].astype(str)
    natsort_key = natsort_keygen()
    df = df.sort_values(
        [COL_COMPRESA, COL_PARTICELLA],
        key=lambda col: col.map(natsort_key) if col.name == COL_PARTICELLA else col)
    return [
        CoppiceParcel(row[COL_COMPRESA], row[COL_PARTICELLA],
                      row[COL_AREA_PARCEL], int(row[COL_PARAMETRO]))
        for _, row in df.iterrows()
    ]


def load_adjacencies(adiacenze_path: Path) -> Adjacencies:
    """Load adjacency pairs from cedui-adiacenti.csv as sorted-pair set."""
    df = pd.read_csv(adiacenze_path, encoding='utf-8-sig', comment='#')
    df[COL_ADJ_A] = df[COL_ADJ_A].astype(str)
    df[COL_ADJ_B] = df[COL_ADJ_B].astype(str)
    adj: Adjacencies = set()
    for _, row in df.iterrows():
        key_a = (row[COL_COMPRESA], row[COL_ADJ_A])
        key_b = (row[COL_COMPRESA], row[COL_ADJ_B])
        pair = (min(key_a, key_b), max(key_a, key_b))
        adj.add(pair)
    return adj


def last_harvests_from_calendario(calendario_path: Path) -> dict[ParcelKey, int]:
    """Extract most recent harvest year per Ceduo parcel from calendario CSV."""
    df = pd.read_csv(calendario_path, comment='#')
    df[COL_PARTICELLA] = df[COL_PARTICELLA].astype(str)
    if COL_GOVERNO in df.columns:
        df = df[df[COL_GOVERNO] == GOV_CEDUO]
    result: dict[ParcelKey, int] = {}
    for _, row in df.iterrows():
        key = (row[COL_COMPRESA], row[COL_PARTICELLA])
        result[key] = max(result.get(key, 0), int(row[COL_ANNO]))
    return result


# =============================================================================
# SCHEDULING
# =============================================================================

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
    n_harvests = math.ceil(parcel.area_ha / MAX_HARVEST_AREA_HA)
    chunk = parcel.area_ha / n_harvests
    year = eligible
    cycle_events: list[CoppiceEvent] = []
    cycle_start: int | None = None

    while len(cycle_events) < n_harvests and year <= last_year:
        if _has_adjacency_conflict(key, year, adjacencies, scheduled_years):
            year += 1
            continue

        if cycle_start is None:
            cycle_start = year
        cycle_events.append(CoppiceEvent(
            year, parcel.compresa, parcel.particella, chunk,
            parcel.area_ha, parcel.intervallo, cycle_start))

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


# =============================================================================
# GANTT LAYOUT
# =============================================================================

# Preserved shoots from one sub-harvest survive two harvest cycles before being cut:
# - harvest N seeds batch;
# - harvest N+1 keeps it alongside a new batch;
# - harvest N+2 cuts it.
# So a batch's lifetime = 2 * intervallo years.
BATCH_LIFETIME_CYCLES = 2

# Minimum lanes per parcel row — lets empty rows still occupy visible height.
MIN_LANES_PER_PARCEL = 2


def coppice_gantt_bars(
    parcels: list[CoppiceParcel],
    events: list[CoppiceEvent],
) -> list[CoppiceRow]:
    """Compute Gantt rows/bars illustrating preserved-shoot batch lifetimes.

    Each scheduled sub-harvest event seeds a bar spanning
    [event.year, event.year + 2 * intervallo]. Within a parcel, bars are laid
    out on 2 * n_sub_harvests lanes: for each sub-slot, consecutive cycles
    alternate between an odd/even lane so the two concurrently-live batches
    do not overlap, and batch N+2 reuses batch N's lane (batch N is cut just
    as batch N+2 is seeded).

    Rows follow the order of the input `parcels` list.
    """
    # Group events by parcel key.
    by_parcel: dict[ParcelKey, list[CoppiceEvent]] = {}
    for e in events:
        by_parcel.setdefault((e.compresa, e.particella), []).append(e)

    rows: list[CoppiceRow] = []
    for p in parcels:
        key = (p.compresa, p.particella)
        n_sub = math.ceil(p.area_ha / MAX_HARVEST_AREA_HA)
        n_lanes = max(BATCH_LIFETIME_CYCLES * n_sub, MIN_LANES_PER_PARCEL)

        cycles: dict[int, list[CoppiceEvent]] = {}
        for e in by_parcel.get(key, []):
            cycles.setdefault(e.cycle_start, []).append(e)

        bars: list[CoppiceBar] = []
        for cycle_idx0, cycle_start in enumerate(sorted(cycles)):
            cycle_events = sorted(cycles[cycle_start], key=lambda e: e.year)
            for sub_idx0, e in enumerate(cycle_events):
                lane = BATCH_LIFETIME_CYCLES * sub_idx0 + (cycle_idx0 % BATCH_LIFETIME_CYCLES)
                bars.append(CoppiceBar(
                    start_year=e.year,
                    end_year=e.year + BATCH_LIFETIME_CYCLES * e.intervallo,
                    lane=lane,
                    cycle_idx=cycle_idx0 + 1,
                    sub_idx=sub_idx0 + 1,
                    n_sub=n_sub,
                ))

        rows.append(CoppiceRow(
            compresa=p.compresa,
            particella=p.particella,
            n_lanes=n_lanes,
            bars=bars,
        ))

    return rows
