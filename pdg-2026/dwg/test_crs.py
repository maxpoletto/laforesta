#!/usr/bin/env python3
"""
Test different Italian coordinate reference systems to find the correct one.

Common Italian CRS that look similar but have different datums:
- EPSG:32633 - WGS84 / UTM zone 33N (what we assumed)
- EPSG:23033 - ED50 / UTM zone 33N (common in older Italian data)
- EPSG:3004  - Monte Mario / Italy zone 2 (Italian national grid)
- EPSG:3005  - Monte Mario / Italy zone 2 with different params

The offset you're seeing (50-100m) is classic datum shift.
"""

import argparse
import json
from pathlib import Path

from pyproj import Transformer, CRS
import matplotlib.pyplot as plt


# Common Italian CRS to test
ITALIAN_CRS = {
    'WGS84_UTM33N': 'EPSG:32633',      # What we assumed
    'ED50_UTM33N': 'EPSG:23033',        # European Datum 1950
    'MonteMario_Zone2': 'EPSG:3004',    # Italian national (Gauss-Boaga)
    'WGS84_UTM32N': 'EPSG:32632',       # Wrong zone?
    'ED50_UTM32N': 'EPSG:23032',        # ED50 wrong zone
}


def load_raw_coords_from_geojson(geojson_path):
    """Load coordinates, assuming they're already in WGS84 (incorrectly projected)."""
    with open(geojson_path) as f:
        data = json.load(f)
    
    coords = []
    for feature in data.get('features', []):
        geom = feature.get('geometry', {})
        if geom.get('type') == 'Polygon':
            for ring in geom.get('coordinates', []):
                coords.extend(ring)
        elif geom.get('type') == 'MultiPolygon':
            for polygon in geom.get('coordinates', []):
                for ring in polygon:
                    coords.extend(ring)
    
    return coords


def test_crs_options(dxf_path, layer_name, output_dir):
    """Extract raw coordinates and test different CRS assumptions."""
    import ezdxf
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract raw UTM coordinates from DXF
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    
    raw_coords = []
    for entity in msp:
        if entity.dxf.layer != layer_name:
            continue
        
        etype = entity.dxftype()
        if etype == 'LINE':
            raw_coords.append((entity.dxf.start.x, entity.dxf.start.y))
            raw_coords.append((entity.dxf.end.x, entity.dxf.end.y))
        elif etype in ('LWPOLYLINE', 'POLYLINE'):
            if hasattr(entity, 'get_points'):
                for p in entity.get_points(format='xy'):
                    raw_coords.append((p[0], p[1]))
    
    if not raw_coords:
        print(f"No coordinates found in layer {layer_name}")
        return
    
    print(f"Extracted {len(raw_coords)} coordinate points")
    
    # Get bounding box of raw coords
    xs = [c[0] for c in raw_coords]
    ys = [c[1] for c in raw_coords]
    print(f"Raw coordinate range:")
    print(f"  X: {min(xs):.1f} to {max(xs):.1f}")
    print(f"  Y: {min(ys):.1f} to {max(ys):.1f}")
    
    # Test each CRS
    results = {}
    for name, epsg in ITALIAN_CRS.items():
        try:
            transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
            
            # Transform centroid
            cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
            lon, lat = transformer.transform(cx, cy)
            
            # Transform all points
            wgs_coords = [transformer.transform(x, y) for x, y in raw_coords]
            
            results[name] = {
                'epsg': epsg,
                'centroid_wgs84': (lon, lat),
                'coords_wgs84': wgs_coords,
            }
            
            print(f"\n{name} ({epsg}):")
            print(f"  Centroid -> {lat:.6f}°N, {lon:.6f}°E")
            
        except Exception as e:
            print(f"\n{name} ({epsg}): FAILED - {e}")
    
    # Calculate offsets between different CRS results
    if 'WGS84_UTM33N' in results:
        ref = results['WGS84_UTM33N']['centroid_wgs84']
        print(f"\nOffsets from WGS84/UTM33N (at centroid):")
        for name, data in results.items():
            if name == 'WGS84_UTM33N':
                continue
            cent = data['centroid_wgs84']
            # Approximate meters (1 degree ≈ 111km lat, varies for lon)
            dlat_m = (cent[1] - ref[1]) * 111000
            dlon_m = (cent[0] - ref[0]) * 111000 * 0.77  # cos(38°) ≈ 0.77 for Calabria
            print(f"  {name}: {dlon_m:.1f}m E, {dlat_m:.1f}m N")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(results.items()):
        if i >= 6:
            break
        ax = axes[i]
        
        lons = [c[0] for c in data['coords_wgs84']]
        lats = [c[1] for c in data['coords_wgs84']]
        
        ax.scatter(lons, lats, s=1, alpha=0.5)
        ax.set_title(f"{name}\n{data['epsg']}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        
        # Add centroid marker
        cx, cy = data['centroid_wgs84']
        ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"crs_comparison_{layer_name}.png", dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir}/crs_comparison_{layer_name}.png")
    
    # Save GeoJSON for each CRS option
    for name, data in results.items():
        geojson = {
            "type": "FeatureCollection",
            "properties": {
                "source_crs": data['epsg'],
                "crs_name": name,
            },
            "features": [{
                "type": "Feature",
                "properties": {"type": "points"},
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": data['coords_wgs84']
                }
            }]
        }
        
        out_path = output_dir / f"{layer_name}_{name}.geojson"
        with open(out_path, 'w') as f:
            json.dump(geojson, f)
    
    print(f"\nSaved GeoJSON files for each CRS option in {output_dir}/")
    print("Load these in the digitizer to see which aligns with satellite imagery!")
    
    return results


def apply_manual_offset(geojson_path, offset_east_m, offset_north_m, output_path):
    """Apply a manual offset to a GeoJSON file."""
    with open(geojson_path) as f:
        data = json.load(f)
    
    # Convert meter offset to approximate degrees (at ~38°N latitude)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(lat) ≈ 87 km at 38°N
    offset_lon = offset_east_m / 87000
    offset_lat = offset_north_m / 111000
    
    def offset_coords(coords):
        if isinstance(coords[0], (int, float)):
            # Single coordinate pair
            return [coords[0] + offset_lon, coords[1] + offset_lat]
        else:
            # Nested list
            return [offset_coords(c) for c in coords]
    
    for feature in data.get('features', []):
        geom = feature.get('geometry', {})
        if 'coordinates' in geom:
            geom['coordinates'] = offset_coords(geom['coordinates'])
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Applied offset: {offset_east_m}m E, {offset_north_m}m N")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test CRS options for Italian cadastral data")
    subparsers = parser.add_subparsers(dest='command')
    
    # Test CRS command
    test_parser = subparsers.add_parser('test', help='Test different CRS options')
    test_parser.add_argument("input", help="Input DXF file")
    test_parser.add_argument("--layer", "-l", required=True, help="Layer name")
    test_parser.add_argument("--output-dir", "-o", default="crs_test", help="Output directory")
    
    # Manual offset command
    offset_parser = subparsers.add_parser('offset', help='Apply manual offset to GeoJSON')
    offset_parser.add_argument("input", help="Input GeoJSON file")
    offset_parser.add_argument("--east", "-e", type=float, required=True, help="Offset east (meters, negative for west)")
    offset_parser.add_argument("--north", "-n", type=float, required=True, help="Offset north (meters, negative for south)")
    offset_parser.add_argument("--output", "-o", required=True, help="Output GeoJSON file")
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_crs_options(args.input, args.layer, args.output_dir)
    elif args.command == 'offset':
        apply_manual_offset(args.input, args.east, args.north, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
