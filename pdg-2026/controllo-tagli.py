#!/usr/bin/env python3
"""Check that the same parcel (compresa + particella) is never harvested
less than N years apart across one or more harvest-calendar CSV files.

Each CSV must have at least the columns: Anno, Compresa, Particella.

Usage:
    python check_harvest_spacing.py [-n YEARS] file1.csv [file2.csv ...]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_MIN_YEARS = 10
REQUIRED_COLUMNS = {'Anno', 'Compresa', 'Particella'}


def read_harvests(paths: list[Path]) -> dict[tuple[str, str], list[int]]:
    """Read harvest events from CSV files, grouped by (compresa, particella).

    Returns a dict mapping (compresa, particella) -> sorted list of years.
    """
    events: dict[tuple[str, str], set[int]] = defaultdict(set)
    for path in paths:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                sys.exit(f"Errore: {path} manca le colonne {missing}")
            for row in reader:
                key = (row['Compresa'].strip(), row['Particella'].strip())
                events[key].add(int(row['Anno']))
    return {k: sorted(v) for k, v in events.items()}


def check_spacing(harvests: dict[tuple[str, str], list[int]],
                   min_years: int) -> list[str]:
    """Return violation messages for parcels harvested too frequently."""
    violations = []
    for (compresa, particella), years in sorted(harvests.items()):
        for prev, cur in zip(years, years[1:]):
            if cur - prev < min_years:
                violations.append(
                    f"La particella {compresa}/{particella} "
                    f"è stata tagliata nel {prev} e poi nel {cur} "
                    f"(meno di {min_years} anni)"
                )
    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Verifica che nessuna particella sia tagliata "
                    "meno di N anni dopo il taglio precedente.")
    parser.add_argument('files', nargs='+', type=Path,
                        help='CSV con colonne Anno, Compresa, Particella')
    parser.add_argument('-n', '--min-years', type=int, default=DEFAULT_MIN_YEARS,
                        help=f'Intervallo minimo tra tagli (default: {DEFAULT_MIN_YEARS})')
    args = parser.parse_args()

    harvests = read_harvests(args.files)
    violations = check_spacing(harvests, args.min_years)

    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        sys.exit(1)
    else:
        print(f"OK: nessuna particella tagliata meno di {args.min_years} anni dopo il taglio precedente.")


if __name__ == '__main__':
    main()
