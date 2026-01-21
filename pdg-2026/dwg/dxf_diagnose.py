#!/usr/bin/env python3
"""
Diagnose and repair linework issues for polygonization.
"""

import argparse
import math
import sys
from pathlib import Path
from collections import defaultdict

import ezdxf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, MultiLineString, box
from shapely.ops import polygonize, unary_union, linemerge, snap
from shapely.strtree import STRtree


# UTM Zone 33N -> WGS84
transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)


def extract_lines_from_dxf(dxf_path, layer_name):
    """Extract all lines from a specific layer."""
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    
    lines = []
    for entity in msp:
        if entity.dxf.layer != layer_name:
            continue
        
        etype = entity.dxftype()
        coords = []
        
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
            
            # Handle closed flag
            is_closed = getattr(entity.dxf, 'flags', 0) & 1
            if is_closed and len(coords) >= 3 and coords[0] != coords[-1]:
                coords.append(coords[0])
        
        if len(coords) >= 2:
            lines.append(LineString(coords))
    
    return lines


def find_endpoint_gaps(lines, tolerance=1.0):
    """Find gaps between line endpoints."""
    # Collect all endpoints
    endpoints = []
    for i, line in enumerate(lines):
        coords = list(line.coords)
        endpoints.append((Point(coords[0]), i, 'start'))
        endpoints.append((Point(coords[-1]), i, 'end'))
    
    # Build spatial index
    points = [ep[0] for ep in endpoints]
    tree = STRtree(points)
    
    # Find isolated endpoints (not near any other endpoint)
    gaps = []
    for i, (pt, line_idx, end_type) in enumerate(endpoints):
        nearby = tree.query(pt.buffer(tolerance))
        # Exclude self
        nearby_others = [j for j in nearby if j != i]
        if len(nearby_others) == 0:
            gaps.append({
                'point': pt,
                'line_index': line_idx,
                'end_type': end_type,
            })
    
    return gaps


def node_lines(lines):
    """
    Node the linework - split lines at all intersections.
    This is often necessary when lines cross but don't share vertices.
    """
    from shapely.ops import unary_union
    
    # Union all lines - this creates a noded geometry
    merged = unary_union(lines)
    
    # Extract individual line segments
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


def snap_endpoints(lines, tolerance=1.0):
    """Snap nearby endpoints together."""
    if not lines:
        return lines
    
    # Collect all endpoints
    all_endpoints = []
    for line in lines:
        coords = list(line.coords)
        all_endpoints.append(coords[0])
        all_endpoints.append(coords[-1])
    
    # Cluster nearby endpoints
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    
    if len(all_endpoints) < 2:
        return lines
    
    points_arr = np.array(all_endpoints)
    
    # Handle case where all points are identical
    if len(set(map(tuple, points_arr))) == 1:
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
        
        # Snap start
        start = tuple(coords[0])
        if start in endpoint_map:
            new_coords[0] = endpoint_map[start]
        
        # Snap end
        end = tuple(coords[-1])
        if end in endpoint_map:
            new_coords[-1] = endpoint_map[end]
        
        if len(new_coords) >= 2:
            snapped_lines.append(LineString(new_coords))
    
    return snapped_lines


def diagnose_layer(dxf_path, layer_name, output_dir):
    """Analyze a layer and try various repair strategies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSING LAYER: {layer_name}")
    print(f"{'='*60}")
    
    lines = extract_lines_from_dxf(dxf_path, layer_name)
    print(f"Extracted {len(lines)} lines")
    
    if not lines:
        print("No lines found!")
        return
    
    # Basic stats
    total_length = sum(l.length for l in lines)
    print(f"Total linework length: {total_length:.1f} meters")
    
    # Find gaps
    for tol in [0.1, 0.5, 1.0, 2.0, 5.0]:
        gaps = find_endpoint_gaps(lines, tolerance=tol)
        print(f"Endpoints with no neighbor within {tol}m: {len(gaps)}")
    
    # Plot raw linework
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Plot 1: Raw linework
    ax = axes[0, 0]
    for line in lines:
        xs, ys = line.xy
        ax.plot(xs, ys, 'b-', linewidth=0.5)
    ax.set_title(f"Raw linework ({len(lines)} lines)")
    ax.set_aspect('equal')
    
    # Mark endpoints
    for line in lines:
        coords = list(line.coords)
        ax.plot(coords[0][0], coords[0][1], 'go', markersize=2, alpha=0.5)
        ax.plot(coords[-1][0], coords[-1][1], 'ro', markersize=2, alpha=0.5)
    
    # Try basic polygonization
    ax = axes[0, 1]
    polys = list(polygonize(lines))
    print(f"\nBasic polygonization: {len(polys)} polygons")
    
    for line in lines:
        xs, ys = line.xy
        ax.plot(xs, ys, 'gray', linewidth=0.3, alpha=0.5)
    
    for poly in polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.4)
        ax.plot(xs, ys, 'b-', linewidth=0.5)
    ax.set_title(f"Basic polygonize: {len(polys)} polygons")
    ax.set_aspect('equal')
    
    # Try with noding
    ax = axes[1, 0]
    noded = node_lines(lines)
    print(f"After noding: {len(noded)} line segments")
    
    polys_noded = list(polygonize(noded))
    print(f"Polygonization after noding: {len(polys_noded)} polygons")
    
    for line in noded:
        xs, ys = line.xy
        ax.plot(xs, ys, 'gray', linewidth=0.3, alpha=0.5)
    
    for poly in polys_noded:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.4)
        ax.plot(xs, ys, 'b-', linewidth=0.5)
    ax.set_title(f"After noding: {len(polys_noded)} polygons")
    ax.set_aspect('equal')
    
    # Try with snapping + noding
    ax = axes[1, 1]
    best_count = 0
    best_tol = 0
    best_polys = []
    
    for snap_tol in [0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            snapped = snap_endpoints(lines, tolerance=snap_tol)
            noded_snapped = node_lines(snapped)
            polys_snapped = list(polygonize(noded_snapped))
            
            # Filter tiny slivers
            valid = [p for p in polys_snapped if p.area > 100]  # > 100 sq meters
            
            print(f"Snap tolerance {snap_tol}m: {len(valid)} polygons (filtered from {len(polys_snapped)})")
            
            if len(valid) > best_count:
                best_count = len(valid)
                best_tol = snap_tol
                best_polys = valid
        except Exception as e:
            print(f"Snap tolerance {snap_tol}m failed: {e}")
    
    for line in lines:
        xs, ys = line.xy
        ax.plot(xs, ys, 'gray', linewidth=0.3, alpha=0.5)
    
    for poly in best_polys:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.4)
        ax.plot(xs, ys, 'b-', linewidth=0.5)
    ax.set_title(f"Best result (snap={best_tol}m): {len(best_polys)} polygons")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    out_path = output_dir / f"diagnose_{layer_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved diagnostic plot: {out_path}")
    
    # Zoom into problem areas - find where gaps exist
    gaps = find_endpoint_gaps(lines, tolerance=2.0)
    if gaps:
        print(f"\nPlotting {min(len(gaps), 6)} gap locations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, gap in enumerate(gaps[:6]):
            ax = axes[i]
            pt = gap['point']
            
            # Plot area around gap
            buffer = 50  # meters
            minx, miny = pt.x - buffer, pt.y - buffer
            maxx, maxy = pt.x + buffer, pt.y + buffer
            
            for line in lines:
                if line.intersects(box(minx, miny, maxx, maxy)):
                    xs, ys = line.xy
                    ax.plot(xs, ys, 'b-', linewidth=1)
                    coords = list(line.coords)
                    ax.plot(coords[0][0], coords[0][1], 'go', markersize=6)
                    ax.plot(coords[-1][0], coords[-1][1], 'ro', markersize=6)
            
            ax.plot(pt.x, pt.y, 'kx', markersize=15, markeredgewidth=3)
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_title(f"Gap at line {gap['line_index']} ({gap['end_type']})")
            ax.set_aspect('equal')
        
        plt.tight_layout()
        out_path = output_dir / f"gaps_{layer_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved gap locations: {out_path}")
    
    return best_polys, best_tol


def main():
    parser = argparse.ArgumentParser(description="Diagnose linework issues")
    parser.add_argument("input", help="Input DXF file")
    parser.add_argument("--layer", "-l", required=True, help="Layer to diagnose")
    parser.add_argument("--output-dir", "-o", default="diagnose", help="Output directory")
    args = parser.parse_args()
    
    # Need scipy for clustering
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "--break-system-packages", "-q"])
    
    diagnose_layer(args.input, args.layer, args.output_dir)


if __name__ == "__main__":
    main()
