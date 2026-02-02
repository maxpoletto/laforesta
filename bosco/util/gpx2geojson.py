#!/usr/bin/env python3
"""
Convert GPX files to GeoJSON LineString features with layer labeling.

Extracts tracks and routes from GPX files and converts them to GeoJSON LineStrings.
All features are tagged with a region/layer label for integration with the parcel editor.

Usage:
    gpx2geojson.py -r Serra roads/*.gpx > serra-roads.geojson
    gpx2geojson.py -r "Monte Cocuzzo" trail1.gpx trail2.gpx > cocuzzo-trails.geojson
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate approximate distance in meters between two lat/lon points.
    Uses a simple Euclidean approximation, accurate enough for short distances.
    """
    lat_diff = (lat2 - lat1) * 111000  # 1 degree latitude ≈ 111km
    lon_diff = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))  # Adjust for latitude
    return math.sqrt(lat_diff**2 + lon_diff**2)


def thin_coordinates(coords: list[list[float]], min_distance: float) -> list[list[float]]:
    """
    Reduce point density by keeping only points at least min_distance meters apart.
    Always keeps the first and last points.
    """
    if min_distance <= 0 or len(coords) <= 2:
        return coords

    result = [coords[0]]  # Always keep first point
    last_kept = coords[0]

    for coord in coords[1:-1]:  # Check middle points
        lon, lat = coord[0], coord[1]
        last_lon, last_lat = last_kept[0], last_kept[1]

        if distance_meters(last_lat, last_lon, lat, lon) >= min_distance:
            result.append(coord)
            last_kept = coord

    result.append(coords[-1])  # Always keep last point
    return result


def parse_gpx_file(file_path: Path, min_distance: float = 0.0) -> list[dict[str, Any]]:
    """
    Parse a GPX file and extract all tracks and routes as GeoJSON features.

    Returns a list of feature dictionaries (not yet wrapped in FeatureCollection).
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)
        return []

    # GPX namespace (handles both with and without namespace)
    ns = {'': 'http://www.topografix.com/GPX/1/1'}

    # Try to detect namespace from root
    if root.tag.startswith('{'):
        ns_uri = root.tag[1:root.tag.index('}')]
        ns = {'gpx': ns_uri}
        ns_prefix = 'gpx:'
    else:
        ns = {}
        ns_prefix = ''

    features = []

    # Extract tracks
    for trk in root.findall(f'.//{ns_prefix}trk', ns):
        name_elem = trk.find(f'{ns_prefix}name', ns)
        track_name = name_elem.text.strip() if name_elem is not None and name_elem.text else file_path.stem

        # A track can have multiple segments; we'll create one LineString per segment
        for trkseg in trk.findall(f'{ns_prefix}trkseg', ns):
            coordinates = []
            for trkpt in trkseg.findall(f'{ns_prefix}trkpt', ns):
                lat = trkpt.get('lat')
                lon = trkpt.get('lon')
                if lat and lon:
                    ele_elem = trkpt.find(f'{ns_prefix}ele', ns)
                    ele = float(ele_elem.text) if ele_elem is not None and ele_elem.text else 0.0
                    # GeoJSON uses [lon, lat, elevation] order
                    coordinates.append([float(lon), float(lat), ele])

            if len(coordinates) >= 2:  # LineString needs at least 2 points
                coordinates = thin_coordinates(coordinates, min_distance)
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': coordinates
                    },
                    'properties': {
                        'name': track_name,
                        'type': 'viabilita',
                        'source': file_path.name
                    }
                })

    # Extract routes
    for rte in root.findall(f'.//{ns_prefix}rte', ns):
        name_elem = rte.find(f'{ns_prefix}name', ns)
        route_name = name_elem.text.strip() if name_elem is not None and name_elem.text else file_path.stem

        coordinates = []
        for rtept in rte.findall(f'{ns_prefix}rtept', ns):
            lat = rtept.get('lat')
            lon = rtept.get('lon')
            if lat and lon:
                ele_elem = rtept.find(f'{ns_prefix}ele', ns)
                ele = float(ele_elem.text) if ele_elem is not None and ele_elem.text else 0.0
                coordinates.append([float(lon), float(lat), ele])

        if len(coordinates) >= 2:
            coordinates = thin_coordinates(coordinates, min_distance)
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coordinates
                },
                'properties': {
                    'name': route_name,
                    'type': 'viabilita',
                    'source': file_path.name
                }
            })

    return features


def convert_gpx_to_geojson(gpx_files: list[Path], region_label: str, min_distance: float = 0.0) -> dict[str, Any]:
    """
    Convert multiple GPX files to a single GeoJSON FeatureCollection.

    All features are tagged with the specified region/layer label.
    Points are thinned to keep only those at least min_distance meters apart.
    """
    all_features = []
    files_processed = 0
    files_failed = 0

    for gpx_file in gpx_files:
        if not gpx_file.exists():
            print(f"Warning: File not found: {gpx_file}", file=sys.stderr)
            files_failed += 1
            continue

        features = parse_gpx_file(gpx_file, min_distance)

        # Add layer label to each feature
        for feature in features:
            feature['properties']['layer'] = region_label

        all_features.extend(features)
        files_processed += 1
        print(f"Processed {gpx_file.name}: {len(features)} feature(s)", file=sys.stderr)

    print(f"\nTotal: {files_processed} file(s) processed, {len(all_features)} feature(s) extracted",
          file=sys.stderr)
    if files_failed > 0:
        print(f"Warning: {files_failed} file(s) failed", file=sys.stderr)
    if min_distance > 0:
        print(f"Thinning: kept points at least {min_distance}m apart", file=sys.stderr)

    return {
        'type': 'FeatureCollection',
        'features': all_features
    }


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Convert GPX files to GeoJSON with layer labeling',
        epilog='Output is written to stdout. Redirect to a file: ... > output.geojson'
    )
    parser.add_argument('-r', '--region', required=True,
                       help='Region/layer label for all features (e.g., "Serra", "Monte Cocuzzo")')
    parser.add_argument('-d', '--distance', type=float, default=0.0, metavar='N',
                       help='Keep only points at least N meters apart (default: 0, no reduction)')
    parser.add_argument('-output', '-o', type=Path,
                       help='Output GeoJSON file (default: stdout)')
    parser.add_argument('gpx_files', nargs='+',
                       help='One or more GPX files to convert')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output (default: compact)')

    args = parser.parse_args()

    # Convert paths
    gpx_paths = [Path(f) for f in args.gpx_files]

    # Convert
    geojson = convert_gpx_to_geojson(gpx_paths, args.region, args.distance)

    # Output to stdout
    f = sys.stdout
    if args.output:
        f = open(args.output, 'w', encoding='utf-8')
    if args.pretty:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    else:
        json.dump(geojson, f, ensure_ascii=False)

    print(file=f)  # Final newline
    f.close() if args.output else None

if __name__ == '__main__':
    main()
