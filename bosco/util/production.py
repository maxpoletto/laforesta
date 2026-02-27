#!/usr/bin/env python3
"""
Preprocess annual production CSV into per-region JSON files with
per-parcel production time series data.

Example usage:
    python production.py \
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

REGION_COL = 'Compresa'
PARCEL_COL = 'Particella'
YEAR_COL = 'Anno'

def load_valid_parcels(geojson_path: str) -> dict[str, set[str]]:
    """Return {region: set of region-parcel keys} from GeoJSON features."""
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
    Build per-region timeseries data.

    Returns {region: {years, parcels, unit, values, forest_total}}.
    Warns on stderr about region-parcel keys not in GeoJSON.
    """
    # Collect all years across ALL regions for a contiguous range
    all_years: set[int] = set()
    # Group rows by compresa
    rows_by_region: dict[str, list[dict]] = {}
    warned: set[str] = set()

    for row in rows:
        region = row[REGION_COL].strip()
        parcel = row[PARCEL_COL].strip()
        year = int(row[YEAR_COL])
        all_years.add(year)

        if region not in rows_by_region:
            rows_by_region[region] = []
        rows_by_region[region].append(row)

        # Check validity
        rp = f"{region}-{parcel}"
        if rp not in warned and (
            region not in valid_parcels or rp not in valid_parcels[region]
        ):
            print(f"Warning: {rp} not in GeoJSON, skipping", file=sys.stderr)
            warned.add(rp)

    if not all_years:
        return {}

    year_min = min(all_years)
    year_max = max(all_years)
    years = list(range(year_min, year_max + 1))
    year_to_idx = {y: i for i, y in enumerate(years)}

    result: dict[str, dict] = {}

    for region, comp_valid in valid_parcels.items():
        parcel_list = sorted(comp_valid)
        n_years = len(years)

        # Initialize values: every valid parcel gets a zero-filled array
        values: dict[str, list[int]] = {
            cp: [0] * n_years for cp in parcel_list
        }

        # Fill in CSV data
        for row in rows_by_region.get(region, []):
            parcel = row["Particella"].strip()
            rp = f"{region}-{parcel}"
            if rp not in values:
                continue  # unknown parcel, already warned
            year = int(row["Anno"])
            qli = int(row["Q.li"])
            values[rp][year_to_idx[year]] = qli

        # Forest total: sum across all valid parcels per year
        forest_total = [0] * n_years
        for arr in values.values():
            for i, v in enumerate(arr):
                forest_total[i] += v

        result[region] = {
            "years": years,
            "parcels": parcel_list,
            "unit": "quintali",
            "values": values,
            "forest_total": forest_total,
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-region production timeseries JSON"
    )
    parser.add_argument("--csv", required=True, help="Path to production CSV")
    parser.add_argument("--geojson", required=True, help="Path to terreni.geojson")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    valid_parcels = load_valid_parcels(args.geojson)
    rows = parse_csv(args.csv)
    timeseries = build_timeseries(rows, valid_parcels)

    output_dir = Path(args.output_dir)
    for region, data in timeseries.items():
        comp_dir = output_dir / region
        comp_dir.mkdir(parents=True, exist_ok=True)
        out_path = comp_dir / "timeseries.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Wrote {out_path} ({len(data['parcels'])} parcels, {len(data['years'])} years)")


if __name__ == "__main__":
    main()
