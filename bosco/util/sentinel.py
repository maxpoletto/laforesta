#!/usr/bin/env python3
"""
Sentinel-2 satellite data downloader for forest biomass analysis.

Queries the Copernicus Data Space Ecosystem (CDSE) to find cloud-free
Sentinel-2 L2A imagery, downloads selected bands via the Sentinel Hub
Process API, computes vegetation indices (NDVI, NDMI, EVI), and saves
as single-band uint8 GeoTIFFs.

Usage:
    ./sentinel.py find [--max-cloud 5]
    ./sentinel.py fetch <YYYY-MM-DD>

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

import argparse
import io
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
GEOJSON_PATH = DATA_DIR / "serra.geojson"
OUTPUT_DIR = DATA_DIR / "satellite"

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BANDS = ["B02", "B04", "B08", "B11"]
RESOLUTION_M = 10
MAX_CLOUD_DEFAULT = 5.0
SEARCH_MONTH_RANGE = (6, 7)  # June through July
FIRST_YEAR = 2015             # Sentinel-2A launch year
WGS84 = CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_from_geojson(path):
    """Return [west, south, east, north] bounding box from a GeoJSON file."""
    with open(path) as f:
        gj = json.load(f)
    lons, lats = [], []
    for feature in gj["features"]:
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


def pixel_dims(bbox):
    """Compute (width, height) in pixels for a WGS84 bbox at RESOLUTION_M."""
    west, south, east, north = bbox
    mid_lat_rad = math.radians((south + north) / 2)
    w = round((east - west) * 111320 * math.cos(mid_lat_rad) / RESOLUTION_M)
    h = round((north - south) * 111320 / RESOLUTION_M)
    return w, h


# ---------------------------------------------------------------------------
# CDSE authentication
# ---------------------------------------------------------------------------

def get_access_token():
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
# Catalog search
# ---------------------------------------------------------------------------

def cmd_find(args):
    """Find cloud-free L2A scenes in June-July of each year."""
    bbox = bbox_from_geojson(GEOJSON_PATH)
    width, height = pixel_dims(bbox)
    print(f"Bounding box: {bbox}")
    print(f"Pixel dimensions: {width} x {height} ({width * height:,} pixels)\n")

    west, south, east, north = bbox
    wkt = (
        f"POLYGON(({west} {south},{east} {south},"
        f"{east} {north},{west} {north},{west} {south}))"
    )
    current_year = datetime.now().year
    all_scenes = []

    for year in range(FIRST_YEAR, current_year + 1):
        m_start, m_end = SEARCH_MONTH_RANGE
        date_start = f"{year}-{m_start:02d}-01T00:00:00.000Z"
        date_end = f"{year}-{m_end:02d}-31T23:59:59.999Z"

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
            cloud = next(
                (
                    a["Value"]
                    for a in p.get("Attributes", [])
                    if a.get("Name") == "cloudCover"
                ),
                None,
            )
            all_scenes.append({
                "date": p["ContentDate"]["Start"][:10],
                "cloud": cloud,
                "name": p["Name"],
                "id": p["Id"],
            })

    all_scenes.sort(key=lambda s: (s["date"], s["cloud"] or 100))

    if not all_scenes:
        print("No scenes found.")
        return

    # Find best (lowest cloud) per year
    best_per_year = {}
    for s in all_scenes:
        year = s["date"][:4]
        if year not in best_per_year or (s["cloud"] or 100) < (best_per_year[year]["cloud"] or 100):
            best_per_year[year] = s

    print(f"{'Date':<12} {'Cloud':>6}  Product")
    print("-" * 78)
    for s in all_scenes:
        marker = " <--" if s is best_per_year.get(s["date"][:4]) else ""
        cloud_str = f"{s['cloud']:.1f}%" if s["cloud"] is not None else "?"
        print(f"{s['date']:<12} {cloud_str:>6}  {s['name']}{marker}")

    print(f"\n{len(all_scenes)} scenes across {len(best_per_year)} years")
    print("<-- = lowest cloud cover per year (best candidate)")


# ---------------------------------------------------------------------------
# Band download via Sentinel Hub Process API
# ---------------------------------------------------------------------------

def fetch_band(bbox, width, height, date_str, band_name, token):
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


# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------

def compute_indices(bands):
    """Compute NDVI, NDMI, EVI from float32 reflectance arrays."""
    b02, b04, b08, b11 = bands["B02"], bands["B04"], bands["B08"], bands["B11"]
    eps = np.float32(1e-10)
    return {
        "ndvi": np.clip((b08 - b04) / (b08 + b04 + eps), -1, 1),
        "ndmi": np.clip((b08 - b11) / (b08 + b11 + eps), -1, 1),
        "evi":  np.clip(2.5 * (b08 - b04) / (b08 + 6*b04 - 7.5*b02 + 1 + eps), -1, 1),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def reflectance_to_uint8(arr):
    """Reflectance [0, ~1] -> uint8 [0, 255]."""
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def index_to_uint8(arr):
    """Index [-1, 1] -> uint8 [0, 255].  128 ~ 0."""
    return ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def save_geotiff(path, data_uint8, bbox, width, height):
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


# ---------------------------------------------------------------------------
# Fetch command
# ---------------------------------------------------------------------------

def cmd_fetch(args):
    """Download bands and compute indices for a specific date."""
    bbox = bbox_from_geojson(GEOJSON_PATH)
    width, height = pixel_dims(bbox)
    date_str = args.date
    print(f"Bounding box: {bbox}")
    print(f"Dimensions:   {width} x {height} pixels")
    print(f"Date:         {date_str}")
    print(f"Bands:        {', '.join(BANDS)}\n")

    token = get_access_token()

    # Download bands
    bands = {}
    for band in BANDS:
        print(f"  Downloading {band}...", end="", flush=True)
        arr = fetch_band(bbox, width, height, date_str, band, token)
        bands[band] = arr
        print(f" done  (min={arr.min():.4f}, max={arr.max():.4f})")

    # Check for empty data (no satellite pass on this date)
    if all(b.max() == 0 for b in bands.values()):
        print("\nWarning: all bands are zero. No data for this date?", file=sys.stderr)
        sys.exit(1)

    # Compute indices
    indices = compute_indices(bands)
    for name, arr in indices.items():
        print(f"  {name.upper():<6} min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")

    # Save
    out_dir = OUTPUT_DIR / date_str
    for name, arr in bands.items():
        path = out_dir / f"{name.lower()}.tif"
        save_geotiff(path, reflectance_to_uint8(arr), bbox, width, height)
        print(f"  Saved {path.relative_to(DATA_DIR)}")

    for name, arr in indices.items():
        path = out_dir / f"{name}.tif"
        save_geotiff(path, index_to_uint8(arr), bbox, width, height)
        print(f"  Saved {path.relative_to(DATA_DIR)}")

    # Metadata
    metadata = {
        "date": date_str,
        "bbox": bbox,
        "bbox_leaflet": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        "width": width,
        "height": height,
        "resolution_m": RESOLUTION_M,
        "bands": {
            b: "uint8, reflectance [0,1] -> [0,255]" for b in BANDS
        },
        "indices": {
            "ndvi": "uint8, [-1,1] -> [0,255], 128=zero",
            "ndmi": "uint8, [-1,1] -> [0,255], 128=zero",
            "evi":  "uint8, [-1,1] -> [0,255], 128=zero",
        },
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path.relative_to(DATA_DIR)}")

    total_kb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1024
    print(f"\nTotal: {total_kb:.0f} KB in {out_dir.relative_to(DATA_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 data for forest biomass analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="Find cloud-free scenes")
    p_find.add_argument(
        "--max-cloud", type=float, default=MAX_CLOUD_DEFAULT,
        help=f"Max cloud cover %% (default {MAX_CLOUD_DEFAULT})",
    )

    p_fetch = sub.add_parser("fetch", help="Download bands and indices")
    p_fetch.add_argument("date", help="Scene date (YYYY-MM-DD)")

    args = parser.parse_args()
    {"find": cmd_find, "fetch": cmd_fetch}[args.command](args)


if __name__ == "__main__":
    main()
