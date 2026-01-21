#!/usr/bin/env python3
"""
Analyze DXF structure to find parcel boundaries and extract them individually.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import ezdxf
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, Point, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid


# UTM Zone 33N -> WGS84
transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


def reproject_coords(coords):
    return [transformer.transform(x, y) for x, y in coords]


def transform_point(x, y, insert_point, scale, rotation):
    x *= scale[0]
    y *= scale[1]
    if rotation != 0:
        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r
    x += insert_point[0]
    y += insert_point[1]
    return x, y


def get_insert_transform(insert_entity):
    """Get transformation parameters from an INSERT entity."""
    insert_point = (insert_entity.dxf.insert.x, insert_entity.dxf.insert.y)
    scale = (
        getattr(insert_entity.dxf, 'xscale', 1.0),
        getattr(insert_entity.dxf, 'yscale', 1.0),
    )
    rotation = getattr(insert_entity.dxf, 'rotation', 0.0)
    return (insert_point, scale, rotation)


def extract_polyline_raw(entity, insert_transform=None):
    """Extract raw UTM coordinates from polyline, return (coords, is_closed)."""
    try:
        if hasattr(entity, 'get_points'):
            points = [(p[0], p[1]) for p in entity.get_points(format='xy')]
        else:
            points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        
        if len(points) < 2:
            return None, False
        
        if insert_transform:
            insert_pt, scale, rotation = insert_transform
            points = [transform_point(x, y, insert_pt, scale, rotation) for x, y in points]
        
        is_closed = getattr(entity.dxf, 'flags', 0) & 1
        return points, bool(is_closed)
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)
        return None, False


def analyze_dxf(dxf_path: Path):
    """Analyze DXF structure and report on potential parcel organization."""
    print(f"Analyzing: {dxf_path}\n")
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    
    # 1. Analyze layers
    print("=" * 60)
    print("LAYERS")
    print("=" * 60)
    layer_entities = defaultdict(lambda: defaultdict(int))
    for entity in msp:
        layer_entities[entity.dxf.layer][entity.dxftype()] += 1
    
    for layer, types in sorted(layer_entities.items()):
        print(f"  {layer}: {dict(types)}")
    
    # 2. Analyze blocks
    print("\n" + "=" * 60)
    print("BLOCKS (referenced in modelspace)")
    print("=" * 60)
    
    block_usage = defaultdict(int)
    for entity in msp:
        if entity.dxftype() == 'INSERT':
            block_usage[entity.dxf.name] += 1
    
    block_details = {}
    for block_name, count in sorted(block_usage.items()):
        try:
            block = doc.blocks.get(block_name)
            entity_types = defaultdict(int)
            closed_polylines = 0
            for e in block:
                etype = e.dxftype()
                entity_types[etype] += 1
                if etype in ('LWPOLYLINE', 'POLYLINE'):
                    if getattr(e.dxf, 'flags', 0) & 1:
                        closed_polylines += 1
            
            block_details[block_name] = {
                'usage_count': count,
                'entity_types': dict(entity_types),
                'closed_polylines': closed_polylines,
            }
            print(f"  {block_name} (used {count}x): {dict(entity_types)}")
            if closed_polylines:
                print(f"    -> {closed_polylines} closed polylines (potential boundaries!)")
        except KeyError:
            print(f"  {block_name}: NOT FOUND")
    
    # 3. Find all closed polylines (potential parcel boundaries)
    print("\n" + "=" * 60)
    print("CLOSED POLYLINES (potential parcel boundaries)")
    print("=" * 60)
    
    parcels = []
    
    # Direct polylines in modelspace
    for entity in msp:
        etype = entity.dxftype()
        if etype in ('LWPOLYLINE', 'POLYLINE'):
            coords, is_closed = extract_polyline_raw(entity)
            if coords and is_closed and len(coords) >= 3:
                parcels.append({
                    'source': 'modelspace',
                    'layer': entity.dxf.layer,
                    'coords_utm': coords,
                    'block': None,
                })
    
    # Polylines inside blocks (via INSERT)
    for entity in msp:
        if entity.dxftype() != 'INSERT':
            continue
        
        block_name = entity.dxf.name
        insert_transform = get_insert_transform(entity)
        insert_layer = entity.dxf.layer
        
        try:
            block = doc.blocks.get(block_name)
        except KeyError:
            continue
        
        for e in block:
            etype = e.dxftype()
            if etype in ('LWPOLYLINE', 'POLYLINE'):
                coords, is_closed = extract_polyline_raw(e, insert_transform)
                if coords and is_closed and len(coords) >= 3:
                    entity_layer = getattr(e.dxf, 'layer', '0')
                    parcels.append({
                        'source': 'block',
                        'layer': entity_layer if entity_layer != '0' else insert_layer,
                        'coords_utm': coords,
                        'block': block_name,
                    })
    
    print(f"  Found {len(parcels)} closed polylines")
    
    # Group by block name
    by_block = defaultdict(list)
    for p in parcels:
        by_block[p['block'] or 'direct'].append(p)
    
    print(f"  Grouped by source:")
    for block, items in sorted(by_block.items()):
        print(f"    {block}: {len(items)} polygons")
    
    # Group by layer
    by_layer = defaultdict(list)
    for p in parcels:
        by_layer[p['layer']].append(p)
    
    print(f"  Grouped by layer:")
    for layer, items in sorted(by_layer.items()):
        print(f"    {layer}: {len(items)} polygons")
    
    return parcels, doc


def plot_individual_parcels(parcels, output_dir: Path, max_plots=20):
    """Generate individual PNG for each parcel."""
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating individual parcel plots (max {max_plots})...")
    
    # Convert to shapely polygons with WGS84 coords
    parcel_polys = []
    for i, p in enumerate(parcels):
        coords = p['coords_utm']
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        
        try:
            wgs_coords = reproject_coords(coords)
            poly = Polygon(wgs_coords)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_valid and poly.area > 0:
                parcel_polys.append({
                    'index': i,
                    'polygon': poly,
                    'layer': p['layer'],
                    'block': p['block'],
                    'area': poly.area,
                })
        except Exception as e:
            print(f"  Skipping parcel {i}: {e}")
    
    # Sort by area (largest first) for more interesting plots
    parcel_polys.sort(key=lambda x: -x['area'])
    
    # Plot each parcel
    n_plots = min(len(parcel_polys), max_plots)
    for i, pp in enumerate(parcel_polys[:n_plots]):
        poly = pp['polygon']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot this parcel
        if poly.geom_type == 'Polygon':
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.4, color='steelblue')
            ax.plot(xs, ys, color='darkblue', linewidth=1.5)
        elif poly.geom_type == 'MultiPolygon':
            for geom in poly.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.4, color='steelblue')
                ax.plot(xs, ys, color='darkblue', linewidth=1.5)
        
        # Add context: show all other parcels faintly
        for other in parcel_polys:
            if other['index'] == pp['index']:
                continue
            opoly = other['polygon']
            if opoly.geom_type == 'Polygon':
                xs, ys = opoly.exterior.xy
                ax.plot(xs, ys, color='gray', linewidth=0.3, alpha=0.3)
            elif opoly.geom_type == 'MultiPolygon':
                for geom in opoly.geoms:
                    xs, ys = geom.exterior.xy
                    ax.plot(xs, ys, color='gray', linewidth=0.3, alpha=0.3)
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        title = f"Parcel {pp['index']}"
        if pp['layer']:
            title += f" (Layer: {pp['layer']})"
        if pp['block']:
            title += f" [Block: {pp['block']}]"
        ax.set_title(title)
        ax.set_aspect('equal')
        
        # Zoom to this parcel with some padding
        minx, miny, maxx, maxy = poly.bounds
        pad_x = (maxx - minx) * 0.2 or 0.001
        pad_y = (maxy - miny) * 0.2 or 0.001
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        
        plt.tight_layout()
        
        out_path = output_dir / f"parcel_{pp['index']:03d}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out_path}")
    
    print(f"\nGenerated {n_plots} parcel plots in {output_dir}/")
    return parcel_polys


def main():
    parser = argparse.ArgumentParser(description="Analyze DXF for parcel boundaries")
    parser.add_argument("input", help="Input DXF file")
    parser.add_argument("--output-dir", "-o", default="parcels", help="Output directory for PNGs")
    parser.add_argument("--max-plots", "-n", type=int, default=20, help="Max number of parcel plots")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't plot")
    args = parser.parse_args()
    
    dxf_path = Path(args.input)
    if not dxf_path.exists():
        print(f"Error: {dxf_path} not found", file=sys.stderr)
        sys.exit(1)
    
    parcels, doc = analyze_dxf(dxf_path)
    
    if not args.analyze_only and parcels:
        output_dir = Path(args.output_dir)
        parcel_polys = plot_individual_parcels(parcels, output_dir, args.max_plots)
        
        # Also save a summary GeoJSON
        features = []
        for pp in parcel_polys:
            features.append({
                "type": "Feature",
                "properties": {
                    "parcel_index": pp['index'],
                    "layer": pp['layer'],
                    "block": pp['block'],
                    "area_deg2": pp['area'],
                },
                "geometry": mapping(pp['polygon'])
            })
        
        geojson_path = output_dir / "all_parcels.geojson"
        with open(geojson_path, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        print(f"Saved GeoJSON: {geojson_path}")


if __name__ == "__main__":
    main()
