"""Tests for calendario_mannesi.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from calendario_mannesi import governo_effettivo, governo_from_particelle, OTHER_SPECIES

SPECIES_COLS = ["castagno"] + OTHER_SPECIES


def make_mannesi_df(rows: list[dict]) -> pd.DataFrame:
    """Build a mannesi-like DataFrame from minimal row dicts.

    Each dict must have Anno, Compresa, Particella, castagno; other species default to 0.
    """
    for r in rows:
        for sp in SPECIES_COLS:
            r.setdefault(sp, 0.0)
    return pd.DataFrame(rows).astype({"Particella": str})


def write_particelle_csv(path: Path, rows: list[tuple[str, str, str]]):
    """Write a minimal particelle.csv with (Compresa, Particella, Governo) tuples."""
    df = pd.DataFrame(rows, columns=["Compresa", "Particella", "Governo"])
    df.to_csv(path, index=False)


class TestGovernoEffettivo:
    def test_ceduo_when_castagno_dominates(self):
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 100, "abete": 10},
        ])
        result = governo_effettivo(df)
        assert result.loc[(2020, "Serra", "1")] == "Ceduo"

    def test_fustaia_when_others_dominate(self):
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 10, "abete": 100},
        ])
        result = governo_effettivo(df)
        assert result.loc[(2020, "Serra", "1")] == "Fustaia"

    def test_fustaia_when_equal(self):
        """When castagno == others, should be Fustaia (not strictly greater)."""
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 50, "abete": 50},
        ])
        result = governo_effettivo(df)
        assert result.loc[(2020, "Serra", "1")] == "Fustaia"

    def test_aggregates_multiple_deliveries(self):
        """Multiple deliveries in one year/parcel should be summed."""
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 30, "abete": 60},
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 80, "abete": 10},
        ])
        # castagno: 110, abete: 70 → Ceduo
        result = governo_effettivo(df)
        assert result.loc[(2020, "Serra", "1")] == "Ceduo"

    def test_multiple_parcels_and_years(self):
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 100},
            {"Anno": 2020, "Compresa": "Serra", "Particella": "2a", "abete": 100},
            {"Anno": 2021, "Compresa": "Serra", "Particella": "1", "abete": 100},
        ])
        result = governo_effettivo(df)
        assert result.loc[(2020, "Serra", "1")] == "Ceduo"
        assert result.loc[(2020, "Serra", "2a")] == "Fustaia"
        assert result.loc[(2021, "Serra", "1")] == "Fustaia"

    def test_multiple_other_species_summed(self):
        """All non-castagno species contribute to 'others'."""
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1",
             "castagno": 50, "abete": 10, "pino": 10, "douglas": 10, "faggio": 10, "ontano": 10, "altro": 10},
        ])
        # others total: 60 > castagno 50 → Fustaia
        result = governo_effettivo(df)
        assert result.loc[(2020, "Serra", "1")] == "Fustaia"


class TestGovernoFromParticelle:
    def test_basic_lookup(self, tmp_path):
        particelle_csv = tmp_path / "particelle.csv"
        write_particelle_csv(particelle_csv, [
            ("Serra", "1", "Fustaia"),
            ("Serra", "2a", "Ceduo"),
        ])
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 999},
            {"Anno": 2020, "Compresa": "Serra", "Particella": "2a", "castagno": 999},
        ])
        result = governo_from_particelle(df, particelle_csv)
        assert result.loc[(2020, "Serra", "1")] == "Fustaia"
        assert result.loc[(2020, "Serra", "2a")] == "Ceduo"

    def test_ignores_species_data(self, tmp_path):
        """Governo comes from particelle.csv regardless of actual species mix."""
        particelle_csv = tmp_path / "particelle.csv"
        write_particelle_csv(particelle_csv, [("Serra", "1", "Ceduo")])
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 0, "abete": 999},
        ])
        result = governo_from_particelle(df, particelle_csv)
        assert result.loc[(2020, "Serra", "1")] == "Ceduo"

    def test_unknown_parcel_gives_nan(self, tmp_path):
        """Parcels not in particelle.csv get NaN Governo."""
        particelle_csv = tmp_path / "particelle.csv"
        write_particelle_csv(particelle_csv, [("Serra", "1", "Fustaia")])
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "X"},
        ])
        result = governo_from_particelle(df, particelle_csv)
        assert pd.isna(result.loc[(2020, "Serra", "X")])

    def test_same_parcel_multiple_years(self, tmp_path):
        """Same parcel in different years gets the same Governo."""
        particelle_csv = tmp_path / "particelle.csv"
        write_particelle_csv(particelle_csv, [("Serra", "1", "Fustaia")])
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1"},
            {"Anno": 2021, "Compresa": "Serra", "Particella": "1"},
        ])
        result = governo_from_particelle(df, particelle_csv)
        assert result.loc[(2020, "Serra", "1")] == "Fustaia"
        assert result.loc[(2021, "Serra", "1")] == "Fustaia"

    def test_deduplicates_deliveries(self, tmp_path):
        """Multiple deliveries for same year/parcel produce one output row."""
        particelle_csv = tmp_path / "particelle.csv"
        write_particelle_csv(particelle_csv, [("Serra", "1", "Fustaia")])
        df = make_mannesi_df([
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 50},
            {"Anno": 2020, "Compresa": "Serra", "Particella": "1", "castagno": 70},
        ])
        result = governo_from_particelle(df, particelle_csv)
        assert len(result) == 1
