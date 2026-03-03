#!/usr/bin/env python3
"""
Extract named features from a GeoJSON file and produce a zipped shapefile bundle.

Matches features by properties.name. Optionally reprojects to UTM (auto-detected
from feature centroid).

Usage:
    geojson2shp.py parcels.geojson "Serra-17"
    geojson2shp.py -u parcels.geojson "Serra-17"    # reproject to UTM
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape


SHAPEFILE_EXTENSIONS = ('.shp', '.shx', '.dbf', '.prj', '.cpg')


def load_matching_features(geojson_path: Path, feature_name: str) -> list[dict]:
    """Load GeoJSON and return features where properties.name == feature_name."""
    with open(geojson_path, encoding='utf-8') as f:
        data = json.load(f)

    return [
        feat for feat in data.get('features', [])
        if feat.get('properties', {}).get('name') == feature_name
    ]


def utm_epsg(lon: float, lat: float) -> int:
    """Derive UTM EPSG code from a lon/lat point."""
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def features_to_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from a list of GeoJSON feature dicts."""
    geometries = [shape(f['geometry']) for f in features]
    properties = [f.get('properties', {}) for f in features]
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
    return gdf


def write_shapefile_zip(gdf: gpd.GeoDataFrame, name: str) -> Path:
    """Write GeoDataFrame as a zipped shapefile bundle named <name>.zip."""
    output_zip = Path(f'{name}.zip')
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = Path(tmpdir) / f'{name}.shp'
        gdf.to_file(shp_path, driver='ESRI Shapefile')

        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for ext in SHAPEFILE_EXTENSIONS:
                component = Path(tmpdir) / f'{name}{ext}'
                if component.exists():
                    zf.write(component, f'{name}/{component.name}')

    return output_zip


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Extract named features from GeoJSON to a zipped shapefile'
    )
    parser.add_argument('geojson_file', type=Path, help='Input GeoJSON file')
    parser.add_argument('feature_name', help='Feature name to match (properties.name)')
    parser.add_argument('-u', '--utm', action='store_true',
                        help='Reproject to UTM (auto-detect zone from centroid)')
    args = parser.parse_args()

    features = load_matching_features(args.geojson_file, args.feature_name)
    if not features:
        print(f"Error: no features matching '{args.feature_name}' "
              f"in {args.geojson_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(features)} feature(s) matching '{args.feature_name}'",
          file=sys.stderr)

    gdf = features_to_gdf(features)

    if args.utm:
        centroid = gdf.geometry.union_all().centroid
        epsg = utm_epsg(centroid.x, centroid.y)
        zone = int((centroid.x + 180) / 6) + 1
        print(f"Reprojecting to EPSG:{epsg} (UTM zone {zone})", file=sys.stderr)
        gdf = gdf.to_crs(epsg=epsg)

    output = write_shapefile_zip(gdf, args.feature_name)
    print(f"Written to {output}", file=sys.stderr)


if __name__ == '__main__':
    main()
