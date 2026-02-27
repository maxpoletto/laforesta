"""Tests for production.py — preprocessing of production data."""

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

# Import from same directory
sys.path.insert(0, str(Path(__file__).parent))
from production import build_timeseries, parse_csv


@pytest.fixture
def valid_parcels():
    """Simulated GeoJSON valid parcels for two regions."""
    return {
        "Serra": {"Serra-1", "Serra-2", "Serra-3"},
        "Fabrizia": {"Fabrizia-1", "Fabrizia-2"},
    }


def make_rows(*tuples):
    """Helper: build CSV-like dicts from (anno, compresa, particella, qli) tuples."""
    return [
        {"Anno": str(a), "Compresa": c, "Particella": p, "Q.li": str(q)}
        for a, c, p, q in tuples
    ]


class TestBuildTimeseries:
    def test_basic_structure(self, valid_parcels):
        rows = make_rows(
            (2020, "Serra", "1", 100),
            (2021, "Serra", "2", 200),
        )
        result = build_timeseries(rows, valid_parcels)

        assert "Serra" in result
        assert "Fabrizia" in result
        ts = result["Serra"]
        assert ts["years"] == [2020, 2021]
        assert ts["unit"] == "quintali"
        assert set(ts["parcels"]) == {"Serra-1", "Serra-2", "Serra-3"}

    def test_contiguous_years(self, valid_parcels):
        """Years should be a contiguous range from min to max across all regions."""
        rows = make_rows(
            (2018, "Serra", "1", 10),
            (2022, "Fabrizia", "1", 20),
        )
        result = build_timeseries(rows, valid_parcels)
        expected_years = [2018, 2019, 2020, 2021, 2022]
        assert result["Serra"]["years"] == expected_years
        assert result["Fabrizia"]["years"] == expected_years

    def test_values_correct_length(self, valid_parcels):
        rows = make_rows(
            (2020, "Serra", "1", 100),
            (2022, "Serra", "1", 300),
        )
        result = build_timeseries(rows, valid_parcels)
        ts = result["Serra"]
        n_years = len(ts["years"])  # 2020, 2021, 2022 = 3
        assert n_years == 3
        for cp in ts["parcels"]:
            assert len(ts["values"][cp]) == n_years

    def test_zero_fill_for_missing_years(self, valid_parcels):
        rows = make_rows(
            (2020, "Serra", "1", 100),
            (2022, "Serra", "1", 300),
        )
        result = build_timeseries(rows, valid_parcels)
        vals = result["Serra"]["values"]["Serra-1"]
        assert vals == [100, 0, 300]

    def test_zero_for_parcels_without_production(self, valid_parcels):
        """Parcels in GeoJSON but absent from CSV should get all zeros."""
        rows = make_rows(
            (2020, "Serra", "1", 100),
        )
        result = build_timeseries(rows, valid_parcels)
        assert result["Serra"]["values"]["Serra-2"] == [0]
        assert result["Serra"]["values"]["Serra-3"] == [0]

    def test_forest_total(self, valid_parcels):
        rows = make_rows(
            (2020, "Serra", "1", 100),
            (2020, "Serra", "2", 200),
            (2021, "Serra", "1", 150),
        )
        result = build_timeseries(rows, valid_parcels)
        # 2020: 100 + 200 + 0 = 300; 2021: 150 + 0 + 0 = 150
        assert result["Serra"]["forest_total"] == [300, 150]

    def test_unknown_parcels_warned_and_skipped(self, valid_parcels, capsys):
        rows = make_rows(
            (2020, "Serra", "1", 100),
            (2020, "Serra", "X", 999),
            (2020, "Altro", "1", 50),
        )
        result = build_timeseries(rows, valid_parcels)
        stderr = capsys.readouterr().err
        assert "Serra-X" in stderr
        assert "Altro-1" in stderr
        # Unknown parcel should not appear in values
        assert "Serra-X" not in result["Serra"]["values"]

    def test_empty_csv(self, valid_parcels):
        result = build_timeseries([], valid_parcels)
        assert result == {}


class TestParseCsv:
    def test_parse(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Anno,Compresa,Particella,Q.li\n2020,Serra,1,100\n")
        rows = parse_csv(str(csv_file))
        assert len(rows) == 1
        assert rows[0]["Anno"] == "2020"
        assert rows[0]["Q.li"] == "100"
