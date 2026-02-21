#!/usr/bin/env python3
"""
Sentinel-2 satellite data downloader for forest biomass analysis.

Queries the Copernicus Data Space Ecosystem (CDSE) to find cloud-free
Sentinel-2 L2A imagery, downloads selected bands via the Sentinel Hub
Process API, computes vegetation indices (NDVI, NDMI, EVI), and saves
as single-band uint8 GeoTIFFs.

Requirements:
    pip install requests numpy rasterio

Authentication (for 'fetch' only):
    1. Register at https://dataspace.copernicus.eu
    2. Go to https://shapps.dataspace.copernicus.eu/dashboard/
    3. Under User Settings > OAuth Clients, create a new client
    4. Set environment variables:
         export CDSE_CLIENT_ID="sh-..."
         export CDSE_CLIENT_SECRET="..."
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Copernicus API endpoints and authorization
# ---------------------------------------------------------------------------

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def get_access_token() -> str:
    """OAuth2 client_credentials token from CDSE."""
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "Error: set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.\n"
            "Register at https://dataspace.copernicus.eu, then create\n"
            "OAuth credentials in the Sentinel Hub dashboard.",
            file=sys.stderr,
        )
        sys.exit(1)
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Auth failed ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BANDS = ["B02", "B04", "B08", "B11"]
INDICES = ["ndvi", "ndmi", "evi"]
EXPECTED_TIFFS = [b.lower() + ".tif" for b in BANDS] + [i + ".tif" for i in INDICES]
RESOLUTION_M = 10
MAX_CLOUD_DEFAULT = 0.1
FIRST_YEAR = 2015            # Sentinel-2A launch year
WGS84 = rasterio.CRS.from_epsg(4326)
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bbox_from_geojson(path: Path, region: str | None = None) -> list[float]:
    """Return [west, south, east, north] bounding box from a GeoJSON file.

    If region is given, only features whose properties.layer matches are included.
    """
    with open(path, encoding='utf-8') as f:
        gj = json.load(f)
    features = gj["features"]
    if region:
        features = [f for f in features if f["properties"].get("layer") == region]
        if not features:
            print(f"Error: no features for region '{region}'", file=sys.stderr)
            sys.exit(1)
    lons, lats = [], []
    for feature in features:
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            rings = geom["coordinates"]
        elif geom["type"] == "MultiPolygon":
            rings = [r for poly in geom["coordinates"] for r in poly]
        else:
            continue
        for ring in rings:
            for coord in ring:
                lons.append(coord[0])
                lats.append(coord[1])
    return [min(lons), min(lats), max(lons), max(lats)]


def pixel_dims(bbox: list[float]) -> tuple[int, int]:
    """Compute (width, height) in pixels for a WGS84 bbox at RESOLUTION_M."""
    west, south, east, north = bbox
    mid_lat_rad = math.radians((south + north) / 2)
    w = round((east - west) * 111320 * math.cos(mid_lat_rad) / RESOLUTION_M)
    h = round((north - south) * 111320 / RESOLUTION_M)
    return w, h


def reflectance_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Reflectance [0, ~1] -> uint8 [0, 255]."""
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def uint8_to_reflectance(arr: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> reflectance [0, 1]."""
    return arr.astype(np.float64) / 255.0


def index_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Index [-1, 1] -> uint8 [0, 255].  128 ~ 0."""
    return ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def uint8_to_index(arr: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> index [-1,+1]."""
    return arr.astype(np.float64) / 127.5 - 1


# ---------------------------------------------------------------------------
# Find command (catalog search)
# ---------------------------------------------------------------------------

def cmd_find(args: argparse.Namespace) -> None:
    """Find cloud-free L2A scenes, optionally filtered by month."""
    bbox = bbox_from_geojson(args.geojson, args.region)
    width, height = pixel_dims(bbox)
    print(f"Bounding box: {bbox}")
    print(f"Pixel dimensions: {width} x {height} ({width * height:,} pixels)")

    # Parse --months (e.g. "6,11") into a set of ints
    if args.months:
        target_months = set(int(m) for m in args.months.split(","))
    else:
        target_months = set(range(1, 13))
    print(f"Months:       {sorted(target_months)}\n")

    west, south, east, north = bbox
    wkt = (
        f"POLYGON(({west} {south},{east} {south},"
        f"{east} {north},{west} {north},{west} {south}))"
    )
    year_start = args.year_start or FIRST_YEAR
    year_end = args.year_end or datetime.now().year
    # Query window spans min..max of target months
    m_start = min(target_months)
    m_end = max(target_months)
    all_scenes = []

    for year in range(year_start, year_end + 1):
        date_start = f"{year}-{m_start:02d}-01T00:00:00.000Z"
        # First day of next month as exclusive upper bound (avoids invalid day-31)
        if m_end == 12:
            date_end = f"{year + 1}-01-01T00:00:00.000Z"
        else:
            date_end = f"{year}-{m_end + 1:02d}-01T00:00:00.000Z"

        filter_parts = [
            "Collection/Name eq 'SENTINEL-2'",
            "contains(Name, 'MSIL2A')",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
            f"ContentDate/Start gt {date_start}",
            f"ContentDate/Start lt {date_end}",
            (
                "Attributes/OData.CSC.DoubleAttribute/any("
                "att:att/Name eq 'cloudCover' and "
                f"att/OData.CSC.DoubleAttribute/Value lt {args.max_cloud})"
            ),
        ]

        resp = requests.get(
            CATALOG_URL,
            params={
                "$filter": " and ".join(filter_parts),
                "$top": 50,
                "$expand": "Attributes",
            },
            timeout=30,
        )
        resp.raise_for_status()

        for p in resp.json().get("value", []):
            date_str = p["ContentDate"]["Start"][:10]
            month = int(date_str[5:7])
            if month not in target_months:
                continue
            cloud = next(
                (
                    a["Value"]
                    for a in p.get("Attributes", [])
                    if a.get("Name") == "cloudCover"
                ),
                None,
            )
            all_scenes.append({
                "date": date_str,
                "cloud": cloud,
                "name": p["Name"],
                "id": p["Id"],
            })

    all_scenes.sort(key=lambda s: (s["date"], s["cloud"] or 100))

    if not all_scenes:
        print("No scenes found.")
        return

    # Find best (lowest cloud) per group
    best = {}
    for s in all_scenes:
        key = s["date"][:4] if args.overall_best else s["date"][:7]
        if key not in best or s["cloud"] < best[key]["cloud"]:
            best[key] = s

    best_scenes = set(id(s) for s in best.values())
    group_label = "year" if args.overall_best else "year-month"

    print(f"{'Date':<12} {'Cloud':>6}  Product")
    print("-" * 78)
    for s in all_scenes:
        marker = " <--" if id(s) in best_scenes else ""
        cloud_str = f"{s['cloud']:.1f}%"
        print(f"{s['date']:<12} {cloud_str:>6}  {s['name']}{marker}")

    print(f"\n{len(all_scenes)} scenes, {len(best)} {group_label} groups")
    print(f"<-- = lowest cloud cover per {group_label} (best candidate)")

    if args.pick:
        picked = sorted(s["date"] for s in best.values())
        args.pick.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pick, "w", encoding='utf-8') as f:
            for d in picked:
                f.write(d + "\n")
        print(f"\nWrote {len(picked)} dates to {args.pick}")


# ---------------------------------------------------------------------------
# Fetch command (download of band data)
# ---------------------------------------------------------------------------


def fetch_band(bbox: list[float], width: int, height: int,
               date_str: str, band_name: str, token: str) -> np.ndarray:
    """Download one band as a float32 GeoTIFF. Returns a 2D numpy array."""
    # Sentinel Hub returns reflectance [0, 1] by default (no units specified).
    evalscript = f"""//VERSION=3
function setup() {{
  return {{
    input: [{{bands: ["{band_name}"]}}],
    output: {{bands: 1, sampleType: "FLOAT32"}}
  }};
}}
function evaluatePixel(s) {{
  return [s.{band_name}];
}}"""
    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326",
                },
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_str}T00:00:00Z",
                        "to": f"{date_str}T23:59:59Z",
                    },
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"},
            }],
        },
        "evalscript": evalscript,
    }
    resp = requests.post(
        PROCESS_URL,
        json=body,
        headers={"Authorization": f"Bearer {token}", "Accept": "image/tiff"},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"  Error fetching {band_name}: {resp.status_code}", file=sys.stderr)
        print(f"  {resp.text[:500]}", file=sys.stderr)
        sys.exit(1)

    with rasterio.open(io.BytesIO(resp.content)) as src:
        return src.read(1)  # 2D float32 array


def compute_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute NDVI, NDMI, EVI from float32 reflectance arrays."""
    b02, b04, b08, b11 = bands["B02"], bands["B04"], bands["B08"], bands["B11"]
    eps = np.float32(1e-10)
    return {
        "ndvi": np.clip((b08 - b04) / (b08 + b04 + eps), -1, 1),
        "ndmi": np.clip((b08 - b11) / (b08 + b11 + eps), -1, 1),
        "evi":  np.clip(2.5 * (b08 - b04) / (b08 + 6*b04 - 7.5*b02 + 1 + eps), -1, 1),
    }


def save_geotiff(path: Path, data_uint8: np.ndarray,
                 bbox: list[float], width: int, height: int) -> None:
    """Write a single-band uint8 GeoTIFF with DEFLATE compression."""
    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        dtype="uint8",
        width=width,
        height=height,
        count=1,
        crs=WGS84,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data_uint8, 1)


def fetch_one_date(date_str: str, bbox: list[float], width: int, height: int,
                   output_dir: Path, token: str) -> bool:
    """Download bands and compute indices for a single date. Returns True on success."""
    print(f"\n--- {date_str} ---")

    # Download bands
    bands = {}
    for band in BANDS:
        print(f"  Downloading {band}...", end="", flush=True)
        arr = fetch_band(bbox, width, height, date_str, band, token)
        bands[band] = arr
        print(f" done  (min={arr.min():.4f}, max={arr.max():.4f})")

    # Check for empty data (no satellite pass on this date)
    if all(b.max() == 0 for b in bands.values()):
        print(f"  Warning: all bands are zero — skipping {date_str}", file=sys.stderr)
        return False

    # Compute indices
    indices = compute_indices(bands)
    for name, arr in indices.items():
        print(f"  {name.upper():<6} (min={arr.min():.3f}, max={arr.max():.3f})")

    # Save
    out_dir = output_dir / date_str
    for name, arr in bands.items():
        path = out_dir / f"{name.lower()}.tif"
        save_geotiff(path, reflectance_to_uint8(arr), bbox, width, height)
        print(f"  Saved {path}")

    for name, arr in indices.items():
        path = out_dir / f"{name}.tif"
        save_geotiff(path, index_to_uint8(arr), bbox, width, height)
        print(f"  Saved {path}")

    total_kb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1024
    print(f"  Total: {total_kb:.0f} KB")
    return True


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download bands and compute indices for one or more dates."""
    bbox = bbox_from_geojson(args.geojson, args.region)
    width, height = pixel_dims(bbox)
    print(f"Bounding box: {bbox}")
    print(f"Dimensions:   {width} x {height} pixels")
    print(f"Bands:        {', '.join(BANDS)}")

    # Collect dates from positional arg or --dates-file
    if args.date and args.dates_file:
        print("Error: specify either a date or --dates-file, not both.", file=sys.stderr)
        sys.exit(1)
    if args.dates_file:
        dates = []
        with open(args.dates_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    dates.append(line)
        if not dates:
            print(f"Error: no dates in {args.dates_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Dates:        {len(dates)} from {args.dates_file}")
    elif args.date:
        dates = [args.date]
        print(f"Date:         {args.date}")
    else:
        print("Error: specify a date or --dates-file.", file=sys.stderr)
        sys.exit(1)

    token = get_access_token()
    ok = 0
    for date_str in dates:
        if fetch_one_date(date_str, bbox, width, height, args.output_dir, token):
            ok += 1
    print(f"\nFetched {ok}/{len(dates)} dates.")


# ---------------------------------------------------------------------------
# Precompute command: compute time series, forest parcel mask, and manifest
# ---------------------------------------------------------------------------

def discover_dates(output_dir: Path) -> list[str]:
    """Scan for date subdirs containing all expected TIFFs."""
    dates = []
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir() or not DATE_DIR_RE.match(entry.name):
            continue
        files = {f.name for f in entry.iterdir()}
        if all(t in files for t in EXPECTED_TIFFS):
            dates.append(entry.name)
    return dates


def rasterize_parcels(geojson_path: Path, bbox: list[float],
                      w: int, h: int,
                      region: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Rasterize parcel polygons into a uint8 mask.

    Multiple GeoJSON features with the same name are treated as fragments
    of a single parcel (multi-polygon).

    If region is given, only features whose properties.layer matches are included.

    Returns (mask, parcel_names) where:
      - parcel_names is a sorted list of unique parcel names
      - mask pixel values are: 0 = outside all parcels,
        1..N = parcel index (matching parcel_names, 1-based).
        All fragments of a multi-polygon parcel share the same index.
    """
    with open(geojson_path, encoding='utf-8') as f:
        gj = json.load(f)

    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, w, h)

    # Filter by region if requested, then sort by name.
    features = gj["features"]
    if region:
        features = [f for f in features if f["properties"].get("layer") == region]
    features = sorted(features, key=lambda f: f["properties"]["name"])
    parcel_names = sorted(set(f["properties"]["name"] for f in features))
    name_to_idx = {name: idx for idx, name in enumerate(parcel_names, start=1)}

    # All fragments of the same parcel get the same raster index.
    shapes = [(feat["geometry"], name_to_idx[feat["properties"]["name"]])
              for feat in features]

    mask = rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask, parcel_names


def compute_timeseries(mask: np.ndarray, parcel_names: list[str],
                       dates: list[str], output_dir: Path) -> dict:
    """Compute per-parcel and forest-wide averages for all layers and dates.

    Returns dict ready for JSON serialization:
      { parcels, dates, layers,
        means: { forest: {layer: [val, ...]},
                 parcels: {cp: {layer: [val, ...]} }
               }
      }
    """
    bands_lower = [b.lower() for b in BANDS]
    layers = bands_lower + INDICES
    index_set = set(INDICES)

    # Precompute pixel masks
    forest_pixels = mask > 0
    forest_count = int(np.sum(forest_pixels))
    parcel_masks = {}
    for idx, name in enumerate(parcel_names, start=1):
        pmask = mask == idx
        parcel_masks[name] = (pmask, int(np.sum(pmask)))

    means_forest = {layer: [] for layer in layers}
    means_parcels = {name: {layer: [] for layer in layers} for name in parcel_names}

    for date in dates:
        for layer in layers:
            path = output_dir / date / (layer + ".tif")
            with rasterio.open(path) as src:
                data = src.read(1)

            # Convert to real values
            if layer in index_set:
                real = uint8_to_index(data)
            else:
                real = uint8_to_reflectance(data)

            # Forest-wide average (only pixels inside any parcel)
            forest_mean = float(np.mean(real[forest_pixels])) if forest_count > 0 else 0.0
            means_forest[layer].append(round(forest_mean, 4))

            # Per-parcel averages
            for name in parcel_names:
                pmask, pcount = parcel_masks[name]
                val = float(np.mean(real[pmask])) if pcount > 0 else 0.0
                means_parcels[name][layer].append(round(val, 4))

    return {
        "parcels": parcel_names,
        "dates": dates,
        "layers": layers,
        "means": {
            "forest": means_forest,
            "parcels": means_parcels,
        },
    }


def cmd_precompute(args: argparse.Namespace) -> None:
    """Discover dates, generate manifest, build parcel mask + time series."""
    output_dir = args.output_dir
    bbox = bbox_from_geojson(args.geojson, args.region)
    w, h = pixel_dims(bbox)
    print(f"Bounding box: {bbox}")
    print(f"Dimensions:   {w} x {h} pixels\n")

    # Discover date directories
    dates = discover_dates(output_dir)
    if not dates:
        print(f"No complete date directories found in {output_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(dates)} dates: {dates[0]} .. {dates[-1]}")

    # Generate manifest.json
    manifest = {
        "dates": dates,
        "bands": [b.lower() for b in BANDS],
        "indices": INDICES,
        "bbox": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        "width": w,
        "height": h,
        "resolution_m": RESOLUTION_M,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {manifest_path}")

    # Rasterize parcels
    print("\nRasterizing parcels...")
    mask, parcel_names = rasterize_parcels(args.geojson, bbox, w, h, args.region)

    for idx, name in enumerate(parcel_names, start=1):
        count = int(np.sum(mask == idx))
        print(f"  {name}: {count} pixels")
    forest_count = int(np.sum(mask > 0))
    print(f"  Total forest: {forest_count} pixels")

    # Save parcel mask
    mask_path = output_dir / "parcel-mask.tif"
    save_geotiff(mask_path, mask, bbox, w, h)
    print(f"\nSaved {mask_path}")

    # Compute and save time series
    print("\nComputing time series averages...")
    ts = compute_timeseries(mask, parcel_names, dates, output_dir)
    ts_path = output_dir / "timeseries.json"
    with open(ts_path, "w", encoding='utf-8') as f:
        json.dump(ts, f, indent=2)
    print(f"Saved {ts_path} ({ts_path.stat().st_size / 1024:.1f} KB)")

    # Quick sanity check
    ndvi_forest = ts["means"]["forest"].get("ndvi", [])
    if ndvi_forest:
        print(f"\nNDVI forest averages: {ndvi_forest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point and command line parsing."""

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        """Add --geojson, --output-dir, and --region to a subcommand parser."""
        subparser.add_argument(
            "--geojson", type=Path, required=True, help="Path to GeoJSON file (required)",
        )
        subparser.add_argument(
            "--output-dir", type=Path, required=True, help="Satellite data directory (required)",
        )
        subparser.add_argument(
            "--region", type=str, default=None,
            help="Restrict to a single region (GeoJSON layer name)",
        )

    parser = argparse.ArgumentParser(
        description="Sentinel-2 data for forest biomass analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="Find cloud-free scenes")
    add_common_args(p_find)
    p_find.add_argument(
        "--max-cloud", type=float, default=MAX_CLOUD_DEFAULT,
        help=f"Max cloud cover %% (default {MAX_CLOUD_DEFAULT})",
    )
    p_find.add_argument(
        "--year-start", type=int, default=None,
        help=f"First year to search (default {FIRST_YEAR})",
    )
    p_find.add_argument(
        "--year-end", type=int, default=None,
        help="Last year to search (default: current year)",
    )
    p_find.add_argument(
        "--months", type=str, default=None,
        help="Comma-separated months to include (e.g. 6,11). Default: all.",
    )
    p_find.add_argument(
        "--pick", type=Path, default=None,
        help="Write best dates (one per line) to this file.",
    )
    p_find.add_argument(
        "--overall-best", action="store_true",
        help="Pick one best per year (across all months) instead of per year-month.",
    )

    p_fetch = sub.add_parser("fetch", help="Download bands and indices")
    add_common_args(p_fetch)
    p_fetch.add_argument("date", nargs="?", default=None, help="Scene date (YYYY-MM-DD)")
    p_fetch.add_argument(
        "--dates-file", type=Path, default=None,
        help="File with one date per line (instead of positional date).",
    )

    p_precompute = sub.add_parser("precompute",
                                  help="Precompute time series of mean values, generate manifest")
    add_common_args(p_precompute)

    args = parser.parse_args()
    {"find": cmd_find, "fetch": cmd_fetch, "precompute": cmd_precompute}[args.command](args)


if __name__ == "__main__":
    main()
