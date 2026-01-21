#!/usr/bin/env python3
"""
Analyze DXF structure and extract parcel boundaries via polygonization.

Handles the common CAD case where boundaries are open polylines/lines
that visually form closed regions but aren't flagged as closed.

Supports multiple Italian coordinate reference systems.
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
from shapely.geometry import LineString, Polygon, Point, MultiLineString, mapping
from shapely.ops import polygonize, unary_union, linemerge
from shapely.validation import make_valid


# Common Italian CRS options
CRS_OPTIONS = {
    'wgs84-33n': 'EPSG:32633',    # WGS84 / UTM zone 33N
    'ed50-33n': 'EPSG:23033',      # ED50 / UTM zone 33N (common in older Italian data)
    'monte-mario-2': 'EPSG:3004',  # Monte Mario / Italy zone 2 (Gauss-Boaga)
    'wgs84-32n': 'EPSG:32632',     # WGS84 / UTM zone 32N
    'ed50-32n': 'EPSG:23032',      # ED50 / UTM zone 32N
}

# Default - can be overridden via command line
SOURCE_CRS = 'EPSG:32633'
transformer = None


def init_transformer(source_crs):
    """Initialize the coordinate transformer."""
    global transformer, SOURCE_CRS
    SOURCE_CRS = source_crs
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)


# Initialize with default
init_transformer('EPSG:32633')


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
    insert_point = (insert_entity.dxf.insert.x, insert_entity.dxf.insert.y)
    scale = (
        getattr(insert_entity.dxf, 'xscale', 1.0),
        getattr(insert_entity.dxf, 'yscale', 1.0),
    )
    rotation = getattr(insert_entity.dxf, 'rotation', 0.0)
    return (insert_point, scale, rotation)


def extract_linestring_coords(entity, insert_transform=None):
    """Extract coordinates as a list of (x, y) tuples - works for LINE, LWPOLYLINE, POLYLINE."""
    etype = entity.dxftype()
    coords = []

    try:
        if etype == 'LINE':
            coords = [
                (entity.dxf.start.x, entity.dxf.start.y),
                (entity.dxf.end.x, entity.dxf.end.y),
            ]
        elif etype in ('LWPOLYLINE', 'POLYLINE'):
            if hasattr(entity, 'get_points'):
                coords = [(p[0], p[1]) for p in entity.get_points(format='xy')]
            else:
                coords = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]

            # Check if explicitly closed
            is_closed = getattr(entity.dxf, 'flags', 0) & 1
            if is_closed and len(coords) >= 3 and coords[0] != coords[-1]:
                coords.append(coords[0])

        if len(coords) < 2:
            return None

        # Apply INSERT transform if present
        if insert_transform:
            insert_pt, scale, rotation = insert_transform
            coords = [transform_point(x, y, insert_pt, scale, rotation) for x, y in coords]

        return coords

    except Exception as e:
        print(f"  Warning extracting {etype}: {e}", file=sys.stderr)
        return None


def collect_linework_by_layer(dxf_path: Path):
    """Collect all line/polyline geometry grouped by layer."""
    print(f"Reading: {dxf_path}")
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Report structure
    entity_counts = defaultdict(lambda: defaultdict(int))
    for entity in msp:
        entity_counts[entity.dxf.layer][entity.dxftype()] += 1

    print("\nLayers and entity counts:")
    for layer, types in sorted(entity_counts.items()):
        print(f"  {layer}: {dict(types)}")

    # Collect linework by layer
    layer_lines = defaultdict(list)  # layer -> list of LineString (in UTM)

    for entity in msp:
        etype = entity.dxftype()
        layer = entity.dxf.layer

        if etype in ('LINE', 'LWPOLYLINE', 'POLYLINE'):
            coords = extract_linestring_coords(entity)
            if coords and len(coords) >= 2:
                layer_lines[layer].append(LineString(coords))

    return layer_lines, doc


def snap_endpoints(lines, tolerance=1.0):
    """Snap nearby endpoints together using clustering."""
    if not lines or len(lines) < 2:
        return lines

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        import numpy as np
    except ImportError:
        print("  scipy not available, skipping endpoint snapping")
        return lines

    # Collect all endpoints
    all_endpoints = []
    for line in lines:
        coords = list(line.coords)
        all_endpoints.append(coords[0])
        all_endpoints.append(coords[-1])

    if len(all_endpoints) < 2:
        return lines

    points_arr = np.array(all_endpoints)

    # Handle degenerate cases
    if len(set(map(tuple, points_arr))) <= 1:
        return lines

    try:
        distances = pdist(points_arr)
        if len(distances) == 0 or np.max(distances) == 0:
            return lines
        linkage_matrix = linkage(distances, method='single')
        clusters = fcluster(linkage_matrix, t=tolerance, criterion='distance')
    except Exception as e:
        print(f"  Clustering failed: {e}")
        return lines

    # Compute cluster centroids
    cluster_centroids = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_centroids:
            cluster_centroids[cluster_id] = []
        cluster_centroids[cluster_id].append(points_arr[i])

    for cluster_id in cluster_centroids:
        pts = np.array(cluster_centroids[cluster_id])
        cluster_centroids[cluster_id] = tuple(pts.mean(axis=0))

    # Map original endpoints to centroids
    endpoint_map = {}
    for i, pt in enumerate(all_endpoints):
        cluster_id = clusters[i]
        endpoint_map[tuple(pt)] = cluster_centroids[cluster_id]

    # Rebuild lines with snapped endpoints
    snapped_lines = []
    for line in lines:
        coords = list(line.coords)
        new_coords = list(coords)

        start = tuple(coords[0])
        if start in endpoint_map:
            new_coords[0] = endpoint_map[start]

        end = tuple(coords[-1])
        if end in endpoint_map:
            new_coords[-1] = endpoint_map[end]

        if len(new_coords) >= 2:
            snapped_lines.append(LineString(new_coords))

    return snapped_lines


def close_gaps(lines, max_gap=50):
    """
    Close gaps by adding short line segments between nearby endpoints.
    Unlike snap_endpoints which just moves points, this creates new geometry.
    """
    try:
        from scipy.spatial import cKDTree
        import numpy as np
    except ImportError:
        print("  scipy not available, skipping gap closing")
        return lines

    endpoints = []
    endpoint_info = []

    for i, line in enumerate(lines):
        coords = list(line.coords)
        endpoints.append(coords[0])
        endpoint_info.append((i, 'start', coords[0]))
        endpoints.append(coords[-1])
        endpoint_info.append((i, 'end', coords[-1]))

    endpoints_arr = np.array(endpoints)
    tree = cKDTree(endpoints_arr)

    # Find pairs of endpoints within max_gap that aren't from same line
    added_segments = []
    used_endpoints = set()

    for i, (pt, (line_idx, end_type, _)) in enumerate(zip(endpoints, endpoint_info)):
        if i in used_endpoints:
            continue

        indices = tree.query_ball_point(pt, max_gap)

        for j in indices:
            if j <= i or j in used_endpoints:
                continue

            other_line_idx = endpoint_info[j][0]
            if other_line_idx == line_idx:
                continue

            other_pt = endpoints[j]
            dist = np.linalg.norm(np.array(pt) - np.array(other_pt))

            if 0.01 < dist < max_gap:
                added_segments.append(LineString([pt, other_pt]))
                used_endpoints.add(i)
                used_endpoints.add(j)
                break

    if added_segments:
        print(f"    Added {len(added_segments)} gap-closing segments (max {max_gap}m)")

    return lines + added_segments


def node_lines(lines):
    """Split lines at all intersections (noding)."""
    if not lines:
        return lines

    merged = unary_union(lines)

    if merged.geom_type == 'MultiLineString':
        return list(merged.geoms)
    elif merged.geom_type == 'LineString':
        return [merged]
    elif merged.geom_type == 'GeometryCollection':
        result = []
        for geom in merged.geoms:
            if geom.geom_type == 'LineString':
                result.append(geom)
            elif geom.geom_type == 'MultiLineString':
                result.extend(geom.geoms)
        return result
    else:
        return lines


def polygonize_layer(lines, layer_name, max_gap=0):
    """
    Convert a collection of lines into polygons.

    Uses noding (split at intersections) and endpoint snapping to handle
    common CAD issues like crossing lines and small gaps.

    max_gap: if > 0, add line segments to close gaps up to this distance (meters)
    """
    if not lines:
        return []

    print(f"\n  Polygonizing layer '{layer_name}' ({len(lines)} lines)...")

    # Optionally close gaps first
    if max_gap > 0:
        lines = close_gaps(lines, max_gap)

    # Strategy 1: Basic polygonization
    polys_basic = list(polygonize(lines))
    print(f"    Basic: {len(polys_basic)} polygons")

    # Strategy 2: Node first (split at intersections)
    noded = node_lines(lines)
    print(f"    After noding: {len(noded)} segments")
    polys_noded = list(polygonize(noded))
    print(f"    Noded polygonization: {len(polys_noded)} polygons")

    # Strategy 3: Snap endpoints then node
    best_polys = polys_noded
    best_count = len([p for p in polys_noded if p.area > 100])

    for snap_tol in [0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            snapped = snap_endpoints(lines, tolerance=snap_tol)
            noded_snapped = node_lines(snapped)
            polys_snapped = list(polygonize(noded_snapped))

            # Filter slivers (< 100 sq meters)
            valid = [p for p in polys_snapped if p.area > 100]

            if len(valid) > best_count:
                best_count = len(valid)
                best_polys = valid
                print(f"    Snap {snap_tol}m + node: {len(valid)} polygons (new best)")
        except Exception as e:
            pass  # Skip failed attempts

    # Final validation
    valid_polygons = []
    for p in best_polys:
        if not p.is_valid:
            p = make_valid(p)
        if p.is_valid and p.area > 100:
            valid_polygons.append(p)

    print(f"    Final: {len(valid_polygons)} valid polygons")
    return valid_polygons


def analyze_and_extract(dxf_path: Path, parcel_layers=None, max_gap=0):
    """
    Main extraction routine.

    parcel_layers: list of layer names to process, or None for auto-detect
    max_gap: if > 0, add line segments to close gaps up to this distance (meters)
    """
    layer_lines, doc = collect_linework_by_layer(dxf_path)

    # Auto-detect parcel layers if not specified
    if parcel_layers is None:
        # Heuristic: layers with multiple polylines, excluding labels/etichette
        parcel_layers = []
        for layer, lines in layer_lines.items():
            if len(lines) >= 5:  # At least 5 lines
                if 'ETICHETT' not in layer.upper() and 'LABEL' not in layer.upper():
                    parcel_layers.append(layer)
        print(f"\nAuto-detected parcel layers: {parcel_layers}")

    # Polygonize each layer
    all_parcels = []
    for layer in parcel_layers:
        if layer not in layer_lines:
            print(f"  Warning: layer '{layer}' not found")
            continue

        polygons = polygonize_layer(layer_lines[layer], layer, max_gap)

        for i, poly in enumerate(polygons):
            all_parcels.append({
                'layer': layer,
                'index_in_layer': i,
                'polygon_utm': poly,
            })

    print(f"\nTotal parcels extracted: {len(all_parcels)}")
    return all_parcels, layer_lines


def reproject_polygon(poly_utm):
    """Reproject a polygon from UTM to WGS84."""
    if poly_utm.geom_type == 'Polygon':
        exterior = reproject_coords(list(poly_utm.exterior.coords))
        interiors = [reproject_coords(list(ring.coords)) for ring in poly_utm.interiors]
        return Polygon(exterior, interiors)
    elif poly_utm.geom_type == 'MultiPolygon':
        return MultiPolygon([reproject_polygon(p) for p in poly_utm.geoms])
    else:
        return poly_utm


def plot_individual_parcels(parcels, output_dir: Path, max_plots=50):
    """Generate individual PNG for each parcel."""
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating parcel plots...")

    # Convert to WGS84
    parcel_data = []
    for i, p in enumerate(parcels):
        try:
            poly_wgs = reproject_polygon(p['polygon_utm'])
            if poly_wgs.is_valid and poly_wgs.area > 0:
                parcel_data.append({
                    'global_index': i,
                    'layer': p['layer'],
                    'index_in_layer': p['index_in_layer'],
                    'polygon': poly_wgs,
                    'area': poly_wgs.area,
                })
        except Exception as e:
            print(f"  Skipping parcel {i}: {e}")

    # Sort by area descending
    parcel_data.sort(key=lambda x: -x['area'])

    n_plots = min(len(parcel_data), max_plots)

    for i, pd in enumerate(parcel_data[:n_plots]):
        poly = pd['polygon']

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot this parcel highlighted
        if poly.geom_type == 'Polygon':
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.4, color='steelblue')
            ax.plot(xs, ys, color='darkblue', linewidth=1.5)

        # Show other parcels faintly for context
        for other in parcel_data:
            if other['global_index'] == pd['global_index']:
                continue
            opoly = other['polygon']
            if opoly.geom_type == 'Polygon':
                xs, ys = opoly.exterior.xy
                ax.plot(xs, ys, color='gray', linewidth=0.3, alpha=0.3)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{pd['layer']} - Parcel {pd['index_in_layer']}")
        ax.set_aspect('equal')

        # Zoom to parcel
        minx, miny, maxx, maxy = poly.bounds
        pad_x = (maxx - minx) * 0.3 or 0.001
        pad_y = (maxy - miny) * 0.3 or 0.001
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        plt.tight_layout()

        safe_layer = pd['layer'].replace('/', '_').replace(' ', '_')
        out_path = output_dir / f"{safe_layer}_parcel_{pd['index_in_layer']:03d}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    print(f"Generated {n_plots} plots in {output_dir}/")
    return parcel_data


def save_geojson(parcel_data, output_path: Path):
    """Save all parcels as GeoJSON."""
    features = []
    for pd in parcel_data:
        features.append({
            "type": "Feature",
            "properties": {
                "layer": pd['layer'],
                "parcel_index": pd['index_in_layer'],
                "area_deg2": pd['area'],
            },
            "geometry": mapping(pd['polygon'])
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    print(f"Saved: {output_path}")


def plot_polygon(ax, poly, alpha=0.4, linewidth=0.5):
    """Plot a polygon or multipolygon using matplotlib's default color cycle."""
    if poly.geom_type == 'Polygon':
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=alpha)
        ax.plot(xs, ys, 'b-', linewidth=linewidth)
    elif poly.geom_type == 'MultiPolygon':
        for p in poly.geoms:
            plot_polygon(ax, p, alpha, linewidth)


def plot_overview_utm(parcels, output_path: Path):
    """Plot all parcels in UTM coordinates (like find_gaps.py)."""
    fig, ax = plt.subplots(figsize=(14, 12))

    for p in parcels:
        plot_polygon(ax, p['polygon_utm'])

    ax.set_title(f"All Parcels - UTM ({len(parcels)} total)")
    ax.set_aspect('equal')
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved UTM overview: {output_path}")


def plot_overview(parcel_data, output_path: Path):
    """Plot all parcels (WGS84)."""
    fig, ax = plt.subplots(figsize=(14, 12))

    for pd in parcel_data:
        plot_polygon(ax, pd['polygon'])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"All Parcels ({len(parcel_data)} total)")
    # Note: no set_aspect('equal') - lat/lng degrees aren't equal distances
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved overview: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract parcels from DXF via polygonization")
    parser.add_argument("input", help="Input DXF file")
    parser.add_argument("--output-dir", "-o", default="parcels", help="Output directory")
    parser.add_argument("--max-plots", "-n", type=int, default=50, help="Max individual plots")
    parser.add_argument("--layers", "-l", nargs='+', help="Specific layers to process")
    parser.add_argument("--max-gap", "-g", type=float, default=0, help="Max gap to close by adding segments (meters)")
    parser.add_argument("--no-individual", action="store_true", help="Skip individual parcel plots")
    parser.add_argument("--crs", "-c", default="wgs84-33n",
                        choices=list(CRS_OPTIONS.keys()),
                        help="Source coordinate system (default: wgs84-33n). "
                             "Options: wgs84-33n, ed50-33n, monte-mario-2, wgs84-32n, ed50-32n")
    args = parser.parse_args()

    # Initialize transformer with specified CRS
    source_crs = CRS_OPTIONS[args.crs]
    init_transformer(source_crs)
    print(f"Using source CRS: {args.crs} ({source_crs})")

    dxf_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract parcels
    parcels, layer_lines = analyze_and_extract(
        dxf_path,
        parcel_layers=args.layers,
        max_gap=args.max_gap
    )

    if not parcels:
        print("No parcels extracted!")
        sys.exit(1)

    # Convert and plot
    parcel_data = []
    for i, p in enumerate(parcels):
        try:
            poly_wgs = reproject_polygon(p['polygon_utm'])
            if not poly_wgs.is_valid:
                print(f"  Parcel {i}: invalid after reprojection (area_utm={p['polygon_utm'].area:.0f})")
                poly_wgs = make_valid(poly_wgs)
            if poly_wgs.is_valid and poly_wgs.area > 0:
                parcel_data.append({
                    'global_index': i,
                    'layer': p['layer'],
                    'index_in_layer': p['index_in_layer'],
                    'polygon': poly_wgs,
                    'area': poly_wgs.area,
                })
            else:
                print(f"  Parcel {i}: dropped (valid={poly_wgs.is_valid}, area={poly_wgs.area})")
        except Exception as e:
            print(f"  Skipping parcel {i}: {e}")

    print(f"After reprojection: {len(parcel_data)} of {len(parcels)} parcels valid")

    # Save GeoJSON
    save_geojson(parcel_data, output_dir / "all_parcels.geojson")

    # Overview plots
    plot_overview_utm(parcels, output_dir / "overview_utm.png")
    plot_overview(parcel_data, output_dir / "overview.png")

    # Individual plots
    if not args.no_individual:
        plot_individual_parcels(parcels, output_dir, args.max_plots)


if __name__ == "__main__":
    main()
