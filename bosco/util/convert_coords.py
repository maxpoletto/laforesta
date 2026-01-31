#!/usr/bin/env python3
"""
Convert tree coordinates from UTM 33N or Gauss-Boaga Zone 2 (Italy East) to WGS84 lat/lon.

Required CSV columns:
  UTMLon,UTMLat,GaussLon,GaussLat,Lon,Lat
"""

import argparse
import csv
from typing import cast
from pyproj import Transformer

def process_csv(input_path: str, output_path: str) -> None:
    """Process input CSV and write output with converted coordinates."""
    # From UTM 33N to WGS84
    utm_transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    # From Gauss-Boaga Zone 2 to WGS84
    gauss_transformer = Transformer.from_crs("EPSG:3004", "EPSG:4326", always_xy=True)

    with open(input_path, 'r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        fieldnames = cast(list[str], reader.fieldnames)
        required_fields = {'Lon', 'Lat', 'GaussLon', 'GaussLat', 'UTMLon', 'UTMLat'}
        if not required_fields.issubset(set(fieldnames)):
            missing = required_fields - set(fieldnames)
            raise ValueError(f"Input CSV is missing required fields: {', '.join(missing)}")

        rows, converted, skipped = [], 0, 0

        for row in reader:
            utm_lon = row.get('UTMLon', '').strip()
            utm_lat = row.get('UTMLat', '').strip()
            gauss_lon = row.get('GaussLon', '').strip()
            gauss_lat = row.get('GaussLat', '').strip()

            new_row = row.copy()
            if utm_lon and utm_lat:
                assert not gauss_lon and not gauss_lat, \
                    "Row has both UTM and Gauss coordinates; ambiguous input."
                try:
                    easting = float(utm_lon)
                    northing = float(utm_lat)
                    lon, lat = utm_transformer.transform(easting, northing)
                    new_row['Lon'] = f"{lon:.6f}"
                    new_row['Lat'] = f"{lat:.6f}"
                    converted += 1
                except (ValueError, TypeError):
                    new_row['Lon'] = ''
                    new_row['Lat'] = ''
                    skipped += 1
            elif gauss_lon and gauss_lat:
                try:
                    easting = float(gauss_lon)
                    northing = float(gauss_lat)
                    lon, lat = gauss_transformer.transform(easting, northing)
                    new_row['Lon'] = f"{lon:.6f}"
                    new_row['Lat'] = f"{lat:.6f}"
                    converted += 1
                except (ValueError, TypeError):
                    new_row['Lon'] = ''
                    new_row['Lat'] = ''
                    skipped += 1
            else:
                print(row)
                assert False, "Row has neither UTM nor Gauss coordinates; cannot convert."
            rows.append(new_row)

    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} rows: {converted} converted, {skipped} skipped")
    print(f"Output written to: {output_path}")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Convert coordinates from UTM 33N or Gauss-Boaga Zone 2 to WGS84'
    )
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output CSV file')
    args = parser.parse_args()

    process_csv(args.input, args.output)


if __name__ == '__main__':
    main()
