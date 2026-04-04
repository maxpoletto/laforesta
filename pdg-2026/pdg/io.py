"""Data loading: CSV/tree file I/O with caching."""

from pathlib import Path

import pandas as pd

from pdg.computation import COL_COMPRESA, COL_PARTICELLA, COL_FUSTAIA

file_cache = {}

def load_csv(filenames: list[str] | str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load CSV file(s), skipping comment lines starting with #."""
    if isinstance(filenames, str):
        filenames = [filenames]
    key = (data_dir, tuple(sorted(filenames)))
    if key not in file_cache:
        files = [ data_dir / filename if data_dir else Path(filename) for filename in filenames]
        file_cache[key] = pd.concat(
            [pd.read_csv(file, comment='#') for file in files],
            ignore_index=True)
    return file_cache[key]

def load_trees(filenames: list[str] | str, data_dir: Path | None = None,
               ceduo: bool = False) -> pd.DataFrame:
    """Load trees from CSV file(s), keeping only fustaia or ceduo rows."""
    df = load_csv(filenames, data_dir)
    mask = ~df[COL_FUSTAIA] if ceduo else df[COL_FUSTAIA]
    result = df[mask].copy()
    result[COL_PARTICELLA] = result[COL_PARTICELLA].astype(str)
    return result

def read_past_harvests(path: Path) -> pd.DataFrame:
    """Read past harvests CSV (columns: Anno, Compresa, Particella)."""
    df = pd.read_csv(path, comment='#')
    required = {'Anno', COL_COMPRESA, COL_PARTICELLA}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Calendario tagli {path}: colonne mancanti: {missing}")
    df[COL_PARTICELLA] = df[COL_PARTICELLA].astype(str)
    return df
