"""Data loading: CSV/tree file I/O with caching."""

from pathlib import Path

import pandas as pd

from pdg.computation import (
    COL_COMPRESA, COL_PARTICELLA, COL_FUSTAIA,
    COL_INTERVALLO_CEDUO,
)

file_cache = {}

# Column name used for the coppice rotation before the Abies exports
LEGACY_COL_PARAMETRO = 'Parametro'


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV, detecting the dialect from the header line.

    Abies exports use ';' as separator and ',' as decimal mark; legacy files
    are plain comma-separated with decimal dots.
    """
    with open(path, encoding='utf-8-sig') as f:
        header = f.readline()
    if ';' in header:
        return pd.read_csv(path, sep=';', decimal=',', comment='#')
    return pd.read_csv(path, comment='#')


def load_csv(filenames: list[str] | str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load CSV file(s), skipping comment lines starting with #."""
    if isinstance(filenames, str):
        filenames = [filenames]
    key = (data_dir, tuple(sorted(filenames)))
    if key not in file_cache:
        files = [ data_dir / filename if data_dir else Path(filename) for filename in filenames]
        file_cache[key] = pd.concat(
            [_read_csv(file) for file in files],
            ignore_index=True)
    return file_cache[key]


def load_parcels(filenames: list[str] | str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load parcel metadata (particelle CSV), normalizing legacy files.

    In legacy files the coppice rotation column is called 'Parametro';
    Abies exports (and this function's result) call it 'Intervallo'.
    """
    df = load_csv(filenames, data_dir).copy()
    if COL_INTERVALLO_CEDUO not in df.columns and LEGACY_COL_PARAMETRO in df.columns:
        df.rename(columns={LEGACY_COL_PARAMETRO: COL_INTERVALLO_CEDUO},
                  inplace=True)
    return df

def load_trees(filenames: list[str] | str, data_dir: Path | None = None,
               ceduo: bool = False) -> pd.DataFrame:
    """Load trees from CSV file(s), keeping only fustaia or ceduo rows."""
    df = load_csv(filenames, data_dir)
    mask = ~df[COL_FUSTAIA] if ceduo else df[COL_FUSTAIA]
    result = df[mask].copy()
    result[COL_PARTICELLA] = result[COL_PARTICELLA].astype(str)
    return result

def read_past_harvests(path: Path) -> pd.DataFrame:
    """Read past harvests CSV (columns: Anno, Compresa, Particella).

    Cached: repeated calls return the same DataFrame object, so downstream
    caches (e.g. pdg.core.plan_cache) can compare inputs by identity.
    Callers must not mutate the result.
    """
    key = ('past_harvests', str(path))
    if key not in file_cache:
        df = pd.read_csv(path, comment='#')
        required = {'Anno', COL_COMPRESA, COL_PARTICELLA}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Calendario tagli {path}: colonne mancanti: {missing}")
        df[COL_PARTICELLA] = df[COL_PARTICELLA].astype(str)
        file_cache[key] = df
    return file_cache[key]
