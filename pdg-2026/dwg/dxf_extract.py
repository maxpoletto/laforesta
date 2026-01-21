#!/usr/bin/env python3
"""
DXF Boundary Extractor Prototype

Extracts geometric entities from a DXF file, reprojects from UTM 33N to WGS84,
and provides both a quick matplotlib plot and GeoJSON output for deck.gl.

Handles both direct geometry and block references (INSERT entities).

Usage:
    python dwg_extract.py input.dxf [--output boundaries.geojson]

Requirements:
    pip install ezdxf pyproj matplotlib shapely --break-system-packages
"""

import argparse
import json
import math
import sys
from pathlib import Path

import ezdxf
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, Point, mapping
from shapely.ops import transform
from shapely import affinity


# UTM Zone 33N -> WGS84
transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


def reproject_coords(coords):
    """Transform coordinates from UTM 33N to WGS84 (lon, lat)."""
    return [transformer.transform(x, y) for x, y in coords]


def transform_point(x, y, insert_point, scale, rotation):
    """Transform a point according to INSERT parameters (before reprojection)."""
    # Scale
    x *= scale[0]
    y *= scale[1]

    # Rotate (rotation is in degrees)
    if rotation != 0:
        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r

    # Translate
    x += insert_point[0]
    y += insert_point[1]

    return x, y


def extract_polyline_coords(entity):
    """Extract raw coordinates from LWPOLYLINE or POLYLINE (no reprojection)."""
    try:
        if hasattr(entity, 'get_points'):
            points = [(p[0], p[1]) for p in entity.get_points(format='xy')]
        else:
            points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]

        if len(points) < 2:
            return None, False

        is_closed = getattr(entity.dxf, 'flags', 0) & 1
        return points, is_closed
    except Exception as e:
        print(f"  Warning: Could not extract polyline: {e}", file=sys.stderr)
        return None, False


def extract_line_coords(entity):
    """Extract raw coordinates from LINE entity (no reprojection)."""
    try:
        start = (entity.dxf.start.x, entity.dxf.start.y)
        end = (entity.dxf.end.x, entity.dxf.end.y)
        return [start, end]
    except Exception as e:
        print(f"  Warning: Could not extract line: {e}", file=sys.stderr)
        return None


def extract_circle_coords(entity):
    """Extract center from CIRCLE entity (no reprojection)."""
    try:
        return (entity.dxf.center.x, entity.dxf.center.y)
    except Exception as e:
        print(f"  Warning: Could not extract circle: {e}", file=sys.stderr)
        return None


def extract_polyline(entity, insert_transform=None):
    """Extract coordinates from LWPOLYLINE or POLYLINE."""
    points, is_closed = extract_polyline_coords(entity)
    if points is None:
        return None

    # Apply block transform if present
    if insert_transform:
        insert_pt, scale, rotation = insert_transform
        points = [transform_point(x, y, insert_pt, scale, rotation) for x, y in points]

    if is_closed and len(points) >= 3:
        if points[0] != points[-1]:
            points.append(points[0])
        return Polygon(reproject_coords(points))
    else:
        return LineString(reproject_coords(points))


def extract_line(entity, insert_transform=None):
    """Extract coordinates from LINE entity."""
    coords = extract_line_coords(entity)
    if coords is None:
        return None

    if insert_transform:
        insert_pt, scale, rotation = insert_transform
        coords = [transform_point(x, y, insert_pt, scale, rotation) for x, y in coords]

    return LineString(reproject_coords(coords))


def extract_circle(entity, insert_transform=None):
    """Extract CIRCLE as a point with radius metadata."""
    center = extract_circle_coords(entity)
    if center is None:
        return None

    if insert_transform:
        insert_pt, scale, rotation = insert_transform
        center = transform_point(center[0], center[1], insert_pt, scale, rotation)

    lon, lat = transformer.transform(center[0], center[1])
    return Point(lon, lat)


def extract_from_entity(entity, doc, insert_transform=None):
    """Extract geometry from a single entity, with optional INSERT transform."""
    etype = entity.dxftype()

    extractors = {
        'LWPOLYLINE': extract_polyline,
        'POLYLINE': extract_polyline,
        'LINE': extract_line,
        'CIRCLE': extract_circle,
    }

    if etype in extractors:
        geom = extractors[etype](entity, insert_transform)
        if geom and not geom.is_empty:
            return geom
    return None


def extract_block_geometry(doc, block_name, insert_entity):
    """Extract all geometry from a block definition, transformed by INSERT params."""
    try:
        block = doc.blocks.get(block_name)
    except KeyError:
        print(f"  Warning: Block '{block_name}' not found", file=sys.stderr)
        return []

    # Get INSERT transformation parameters
    insert_point = (insert_entity.dxf.insert.x, insert_entity.dxf.insert.y)
    scale = (
        getattr(insert_entity.dxf, 'xscale', 1.0),
        getattr(insert_entity.dxf, 'yscale', 1.0),
    )
    rotation = getattr(insert_entity.dxf, 'rotation', 0.0)
    insert_transform = (insert_point, scale, rotation)

    layer = insert_entity.dxf.layer
    geometries = []

    for entity in block:
        etype = entity.dxftype()

        # Recursive handling of nested INSERTs
        if etype == 'INSERT':
            nested = extract_block_geometry(doc, entity.dxf.name, entity)
            # Apply our transform to nested results... this gets complex
            # For now, skip nested blocks (uncommon for boundary data)
            continue

        geom = extract_from_entity(entity, doc, insert_transform)
        if geom:
            # Use entity's layer if set, otherwise INSERT's layer
            entity_layer = getattr(entity.dxf, 'layer', None)
            if entity_layer and entity_layer != '0':
                geom_layer = entity_layer
            else:
                geom_layer = layer
            geometries.append((geom, geom_layer, etype))

    return geometries


def extract_geometries(dxf_path: Path) -> list[dict]:
    """
    Extract all geometric entities from a DXF file.
    Returns list of GeoJSON features.
    """
    print(f"Reading: {dxf_path}")
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    features = []

    # Count entities by type for reporting
    entity_counts = {}
    for entity in msp:
        etype = entity.dxftype()
        entity_counts[etype] = entity_counts.get(etype, 0) + 1

    print(f"Found entity types in modelspace: {entity_counts}")

    # Also report block definitions
    block_info = {}
    for block in doc.blocks:
        if block.name.startswith('*'):  # Skip anonymous/internal blocks
            continue
        entities_in_block = {}
        for e in block:
            et = e.dxftype()
            entities_in_block[et] = entities_in_block.get(et, 0) + 1
        if entities_in_block:
            block_info[block.name] = entities_in_block

    if block_info:
        print(f"Found {len(block_info)} block definitions:")
        for bname, bentities in list(block_info.items())[:10]:  # Show first 10
            print(f"  {bname}: {bentities}")
        if len(block_info) > 10:
            print(f"  ... and {len(block_info) - 10} more blocks")

    # Process direct geometry in modelspace
    for entity in msp:
        etype = entity.dxftype()

        if etype == 'INSERT':
            # Extract geometry from block reference
            block_name = entity.dxf.name
            block_geoms = extract_block_geometry(doc, block_name, entity)
            for geom, layer, orig_type in block_geoms:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "layer": layer,
                        "entity_type": orig_type,
                        "block": block_name,
                    },
                    "geometry": mapping(geom)
                }
                features.append(feature)
        else:
            geom = extract_from_entity(entity, doc)
            if geom:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "layer": entity.dxf.layer,
                        "entity_type": etype,
                    },
                    "geometry": mapping(geom)
                }
                features.append(feature)

    print(f"Extracted {len(features)} features")
    return features


def plot_features(features: list[dict], title: str = "DXF Boundaries"):
    """Quick matplotlib plot for visual verification."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by layer
    layers = list(set(f["properties"]["layer"] for f in features))
    colors = plt.cm.tab20(range(len(layers)))
    layer_colors = dict(zip(layers, colors))

    for feature in features:
        geom_type = feature["geometry"]["type"]
        coords = feature["geometry"]["coordinates"]
        layer = feature["properties"]["layer"]
        color = layer_colors[layer]

        if geom_type == "LineString":
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color, linewidth=0.5, alpha=0.7)
        elif geom_type == "Polygon":
            # Exterior ring
            xs, ys = zip(*coords[0])
            ax.fill(xs, ys, alpha=0.3, color=color)
            ax.plot(xs, ys, color=color, linewidth=0.5)
        elif geom_type == "Point":
            ax.plot(coords[0], coords[1], 'o', color=color, markersize=2)

    # Legend (if not too many layers)
    if len(layers) <= 20:
        handles = [plt.Line2D([0], [0], color=layer_colors[l], label=l) for l in layers]
        ax.legend(handles=handles, loc='upper left', fontsize=6)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()

    return fig


def save_geojson(features: list[dict], output_path: Path):
    """Save features as GeoJSON FeatureCollection."""
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    print(f"Saved GeoJSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract boundaries from DXF")
    parser.add_argument("input", nargs='+', help="Input DXF file(s)")
    parser.add_argument("--output-dir", "-o", default="", help="Output directory")
    parser.add_argument("--no-plot", default=False, action="store_true", help="Skip matplotlib plot")
    parser.add_argument("--no-display", default=False, action="store_true", help="Skip displaying the plot")
    parser.add_argument("--layer", "-l", default=[], action="append", help="Filter to specific layer(s)")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]

    # Validate input files
    for dxf_path in input_paths:
        if not dxf_path.exists():
            print(f"Error: File not found: {dxf_path}", file=sys.stderr)
            sys.exit(1)

    # Process each input file
    for i, dxf_path in enumerate(input_paths):
        # Extract
        features = extract_geometries(dxf_path)

        # Filter by layer if requested
        if args.layer:
            features = [f for f in features if f["properties"]["layer"] in args.layer]
            print(f"\tFiltered {dxf_path.name} to {len(features)} features in layers: {args.layer}")

        if not features:
            print(f"\tWarning: No features extracted from {dxf_path.name}!", file=sys.stderr)
            continue

        for feature in features:
            feature["properties"]["source_file"] = dxf_path.name

        # Save GeoJSON
        output_path = Path(args.output_dir) / Path(dxf_path.name).with_suffix('.geojson')
        print(f"\tSaving GeoJSON to {output_path}")
        save_geojson(features, output_path)

        # Plot
        if not args.no_plot:
            fig = plot_features(features, title=dxf_path.stem)
            plot_path = Path(args.output_dir) / Path(dxf_path.name).with_suffix('.png')
            fig.savefig(plot_path, dpi=150)
            if not args.no_display:
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
