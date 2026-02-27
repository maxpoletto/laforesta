#!/usr/bin/env python3
"""
Preprocess per-parcel annual production CSV into per-compresa JSON files.

Reads produzione-particelle-anno.csv and terreni.geojson, produces one
timeseries.json per compresa under the output directory.

Usage:
    python produzione.py \
        --csv ../data/produzione-particelle-anno.csv \
        --geojson ../data/terreni.geojson \
        --output-dir ../data/produzione
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_valid_parcels(geojson_path: str) -> dict[str, set[str]]:
    """Return {compresa: set of CP keys} from GeoJSON features."""
    with open(geojson_path) as f:
        gj = json.load(f)
    parcels: dict[str, set[str]] = {}
    for feat in gj["features"]:
        layer = feat["properties"]["layer"]
        name = feat["properties"]["name"]
        if layer not in parcels:
            parcels[layer] = set()
        parcels[layer].add(name)
    return parcels


def parse_csv(csv_path: str) -> list[dict]:
    """Read production CSV into list of dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_timeseries(
    rows: list[dict],
    valid_parcels: dict[str, set[str]],
) -> dict[str, dict]:
    """
    Build per-compresa timeseries data.

    Returns {compresa: {years, parcels, unit, values, forest_total}}.
    Warns on stderr about CP keys not in GeoJSON.
    """
    # Collect all years across ALL compresas for a contiguous range
    all_years: set[int] = set()
    # Group rows by compresa
    rows_by_compresa: dict[str, list[dict]] = {}
    warned: set[str] = set()

    for row in rows:
        compresa = row["Compresa"].strip()
        particella = row["Particella"].strip()
        anno = int(row["Anno"])
        all_years.add(anno)

        if compresa not in rows_by_compresa:
            rows_by_compresa[compresa] = []
        rows_by_compresa[compresa].append(row)

        # Check validity
        cp = f"{compresa}-{particella}"
        if cp not in warned and (
            compresa not in valid_parcels or cp not in valid_parcels[compresa]
        ):
            print(f"Warning: {cp} not in GeoJSON, skipping", file=sys.stderr)
            warned.add(cp)

    if not all_years:
        return {}

    year_min = min(all_years)
    year_max = max(all_years)
    years = list(range(year_min, year_max + 1))
    year_to_idx = {y: i for i, y in enumerate(years)}

    result: dict[str, dict] = {}

    for compresa, comp_valid in valid_parcels.items():
        parcel_list = sorted(comp_valid)
        n_years = len(years)

        # Initialize values: every valid parcel gets a zero-filled array
        values: dict[str, list[int]] = {
            cp: [0] * n_years for cp in parcel_list
        }

        # Fill in CSV data
        for row in rows_by_compresa.get(compresa, []):
            particella = row["Particella"].strip()
            cp = f"{compresa}-{particella}"
            if cp not in values:
                continue  # unknown parcel, already warned
            anno = int(row["Anno"])
            qli = int(row["Q.li"])
            values[cp][year_to_idx[anno]] = qli

        # Forest total: sum across all valid parcels per year
        forest_total = [0] * n_years
        for arr in values.values():
            for i, v in enumerate(arr):
                forest_total[i] += v

        result[compresa] = {
            "years": years,
            "parcels": parcel_list,
            "unit": "quintali",
            "values": values,
            "forest_total": forest_total,
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-compresa production timeseries JSON"
    )
    parser.add_argument("--csv", required=True, help="Path to production CSV")
    parser.add_argument("--geojson", required=True, help="Path to terreni.geojson")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    valid_parcels = load_valid_parcels(args.geojson)
    rows = parse_csv(args.csv)
    timeseries = build_timeseries(rows, valid_parcels)

    output_dir = Path(args.output_dir)
    for compresa, data in timeseries.items():
        comp_dir = output_dir / compresa
        comp_dir.mkdir(parents=True, exist_ok=True)
        out_path = comp_dir / "timeseries.json"
        with open(out_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Wrote {out_path} ({len(data['parcels'])} parcels, {len(data['years'])} years)")


if __name__ == "__main__":
    main()
