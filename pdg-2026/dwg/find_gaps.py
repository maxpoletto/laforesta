#!/usr/bin/env python3
"""
Find the largest gaps in linework and attempt to close them.
"""

import argparse
import sys
from pathlib import Path

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union, nearest_points
from scipy.spatial import cKDTree


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
            
            is_closed = getattr(entity.dxf, 'flags', 0) & 1
            if is_closed and len(coords) >= 3 and coords[0] != coords[-1]:
                coords.append(coords[0])
        
        if len(coords) >= 2:
            lines.append(LineString(coords))
    
    return lines


def find_gaps_with_distances(lines, max_distance=1000):
    """
    Find all endpoint gaps and their distances to nearest other endpoint.
    Returns sorted list of (distance, point, nearest_point, line_idx, end_type)
    """
    # Collect endpoints
    endpoints = []
    endpoint_info = []
    
    for i, line in enumerate(lines):
        coords = list(line.coords)
        endpoints.append(coords[0])
        endpoint_info.append((i, 'start'))
        endpoints.append(coords[-1])
        endpoint_info.append((i, 'end'))
    
    endpoints_arr = np.array(endpoints)
    tree = cKDTree(endpoints_arr)
    
    gaps = []
    for i, (pt, (line_idx, end_type)) in enumerate(zip(endpoints, endpoint_info)):
        # Find 2 nearest (first is self)
        distances, indices = tree.query(pt, k=2)
        
        if len(distances) > 1:
            nearest_dist = distances[1]
            nearest_idx = indices[1]
            nearest_pt = endpoints[nearest_idx]
            
            if nearest_dist > 0.01 and nearest_dist < max_distance:  # Not coincident, within range
                gaps.append({
                    'distance': nearest_dist,
                    'point': pt,
                    'nearest_point': nearest_pt,
                    'line_index': line_idx,
                    'end_type': end_type,
                    'nearest_line_index': endpoint_info[nearest_idx][0],
                })
    
    # Sort by distance descending (largest gaps first)
    gaps.sort(key=lambda x: -x['distance'])
    
    return gaps


def close_gaps(lines, max_gap=50):
    """
    Close gaps by adding short line segments between nearby endpoints.
    """
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
            
        # Find nearby endpoints
        indices = tree.query_ball_point(pt, max_gap)
        
        for j in indices:
            if j <= i or j in used_endpoints:
                continue
            
            other_line_idx = endpoint_info[j][0]
            if other_line_idx == line_idx:
                continue  # Same line
            
            other_pt = endpoints[j]
            dist = np.linalg.norm(np.array(pt) - np.array(other_pt))
            
            if 0.01 < dist < max_gap:
                # Add connecting segment
                added_segments.append(LineString([pt, other_pt]))
                used_endpoints.add(i)
                used_endpoints.add(j)
                break
    
    return lines + added_segments


def main():
    parser = argparse.ArgumentParser(description="Analyze and close gaps")
    parser.add_argument("input", help="Input DXF file")
    parser.add_argument("--layer", "-l", required=True, help="Layer name")
    parser.add_argument("--output-dir", "-o", default="gaps", help="Output directory")
    parser.add_argument("--max-gap", "-g", type=float, default=50, help="Max gap to close (meters)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    lines = extract_lines_from_dxf(args.input, args.layer)
    print(f"Extracted {len(lines)} lines from layer '{args.layer}'")
    
    # Find gaps
    gaps = find_gaps_with_distances(lines)
    
    print(f"\nLargest gaps (endpoint to nearest other endpoint):")
    for g in gaps[:15]:
        print(f"  {g['distance']:.1f}m: line {g['line_index']} ({g['end_type']}) -> line {g['nearest_line_index']}")
    
    # Plot the largest gaps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, gap in enumerate(gaps[:6]):
        ax = axes[i]
        pt = np.array(gap['point'])
        nearest = np.array(gap['nearest_point'])
        
        # Plot area around gap
        center = (pt + nearest) / 2
        buffer = max(gap['distance'] * 2, 100)
        
        for line in lines:
            coords = np.array(line.coords)
            # Check if any part is in view
            if (coords[:, 0].max() > center[0] - buffer and 
                coords[:, 0].min() < center[0] + buffer and
                coords[:, 1].max() > center[1] - buffer and
                coords[:, 1].min() < center[1] + buffer):
                ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1)
                ax.plot(coords[0, 0], coords[0, 1], 'go', markersize=4)
                ax.plot(coords[-1, 0], coords[-1, 1], 'ro', markersize=4)
        
        # Highlight the gap
        ax.plot([pt[0], nearest[0]], [pt[1], nearest[1]], 'r--', linewidth=2)
        ax.plot(pt[0], pt[1], 'kx', markersize=15, markeredgewidth=3)
        ax.plot(nearest[0], nearest[1], 'k+', markersize=15, markeredgewidth=3)
        
        ax.set_xlim(center[0] - buffer, center[0] + buffer)
        ax.set_ylim(center[1] - buffer, center[1] + buffer)
        ax.set_title(f"Gap {i+1}: {gap['distance']:.1f}m")
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"largest_gaps_{args.layer}.png", dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir}/largest_gaps_{args.layer}.png")
    
    # Try closing gaps and polygonizing
    print(f"\nAttempting to close gaps up to {args.max_gap}m...")
    
    extended_lines = close_gaps(lines, max_gap=args.max_gap)
    print(f"Added {len(extended_lines) - len(lines)} gap-closing segments")
    
    # Node and polygonize
    noded = list(unary_union(extended_lines).geoms) if unary_union(extended_lines).geom_type == 'MultiLineString' else [unary_union(extended_lines)]
    polys = list(polygonize(noded))
    valid = [p for p in polys if p.area > 100]
    print(f"Polygonization result: {len(valid)} polygons")
    
    # Plot result
    fig, ax = plt.subplots(figsize=(14, 12))
    
    for line in lines:
        coords = np.array(line.coords)
        ax.plot(coords[:, 0], coords[:, 1], 'gray', linewidth=0.3, alpha=0.5)
    
    for poly in valid:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.4)
        ax.plot(xs, ys, 'b-', linewidth=0.5)
    
    ax.set_title(f"After closing gaps ≤{args.max_gap}m: {len(valid)} polygons")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / f"closed_gaps_{args.layer}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/closed_gaps_{args.layer}.png")


if __name__ == "__main__":
    main()
