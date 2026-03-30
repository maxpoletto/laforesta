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
    return []
