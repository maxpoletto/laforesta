#!/usr/bin/env python3
"""Add a hole to a polygon feature where a smaller feature is fully contained.

Usage:
    python add_hole.py --geojson ../data/terreni.geojson \
        --outer Capistrano-14b --inner Capistrano-14a

Finds the inner feature that is geometrically contained within the outer
feature, and adds its boundary as a hole in the outer polygon.  Writes
the updated GeoJSON back to the same file.
"""

import argparse
import json

from shapely.geometry import Polygon, shape


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geojson", required=True, help="Path to GeoJSON file")
    parser.add_argument("--outer", required=True, help="Name of the outer feature (gets the hole)")
    parser.add_argument("--inner", required=True, help="Name of the inner feature (becomes the hole)")
    args = parser.parse_args()

    with open(args.geojson) as f:
        data = json.load(f)

    # Collect all features matching each name
    outers = [(i, f) for i, f in enumerate(data["features"])
              if f["properties"].get("name") == args.outer]
    inners = [(i, f) for i, f in enumerate(data["features"])
              if f["properties"].get("name") == args.inner]

    if not outers:
        raise SystemExit(f"No feature named '{args.outer}'")
    if not inners:
        raise SystemExit(f"No feature named '{args.inner}'")

    # Find the (inner, outer) pair where inner is contained in outer
    found = False
    for inner_idx, inner_feat in inners:
        inner_geom = shape(inner_feat["geometry"])
        for outer_idx, outer_feat in outers:
            outer_geom = shape(outer_feat["geometry"])
            if outer_geom.contains(inner_geom):
                # Add inner boundary as hole in outer polygon
                assert outer_feat["geometry"]["type"] == "Polygon", \
                    f"Expected Polygon, got {outer_feat['geometry']['type']}"
                outer_ring = outer_feat["geometry"]["coordinates"][0]
                inner_ring = inner_feat["geometry"]["coordinates"][0]

                new_geom = Polygon(outer_ring, [inner_ring])
                assert new_geom.is_valid, f"Result is not a valid polygon"

                old_area = outer_geom.area
                new_area = new_geom.area
                hole_area = inner_geom.area

                from shapely.geometry import mapping
                data["features"][outer_idx]["geometry"] = mapping(new_geom)

                print(f"Added hole from {args.inner} (feat {inner_idx}) "
                      f"to {args.outer} (feat {outer_idx})")
                print(f"  Outer area: {old_area:.10f} -> {new_area:.10f} "
                      f"(hole: {hole_area:.10f})")
                found = True
                break
        if found:
            break

    if not found:
        raise SystemExit(f"No feature named '{args.inner}' is contained "
                         f"within any feature named '{args.outer}'")

    with open(args.geojson, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {args.geojson}")


if __name__ == "__main__":
    main()
