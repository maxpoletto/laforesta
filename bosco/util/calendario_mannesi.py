#!/usr/bin/env python3
"""Convert mannesi.csv (delivery records) to calendario-tagli format (one row per parcel/year).

Governo is determined in one of two ways:
  - Default: looked up from particelle.csv by Compresa/Particella.
  - --effettivo: computed from delivery data — Ceduo if castagno quintals
    exceed all other species combined, Fustaia otherwise.
"""

import argparse
import pathlib

import pandas as pd

GROUPBY_COLS = ["Anno", "Compresa", "Particella"]
OTHER_SPECIES = ["abete", "pino", "douglas", "faggio", "ontano", "altro"]


def governo_from_particelle(df: pd.DataFrame, particelle_path: pathlib.Path) -> pd.Series:
    """Look up Governo from particelle.csv for each row group."""
    particelle = pd.read_csv(particelle_path, dtype={"Particella": str})
    merged = df[GROUPBY_COLS].drop_duplicates().merge(
        particelle[["Compresa", "Particella", "Governo"]],
        on=["Compresa", "Particella"],
        how="left",
    )
    return merged.set_index(GROUPBY_COLS)["Governo"]


def governo_effettivo(df: pd.DataFrame) -> pd.Series:
    """Ceduo if castagno quintals exceed all other species combined."""
    grouped = df.groupby(GROUPBY_COLS)
    castagno = grouped["castagno"].sum()
    altri = grouped[OTHER_SPECIES].sum().sum(axis=1)
    return (castagno > altri).map({True: "Ceduo", False: "Fustaia"}).rename("Governo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mannesi", type=pathlib.Path, help="Input mannesi.csv")
    parser.add_argument("output", type=pathlib.Path, help="Output calendario CSV")
    parser.add_argument(
        "--effettivo", action="store_true",
        help="Calcola il governo in base ai dati di prelievo invece che ai dati particellari (default: Falso)",
    )
    parser.add_argument(
        "--particelle", type=pathlib.Path, default=None,
        help="File con dati particellari (default: ../data/particelle.csv relative to mannesi)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.mannesi, dtype={"Particella": str})

    if args.effettivo:
        governo = governo_effettivo(df)
    else:
        particelle_path = args.particelle or args.mannesi.parent / "particelle.csv"
        governo = governo_from_particelle(df, particelle_path)

    cal = governo.reset_index()
    cal = cal.sort_values(GROUPBY_COLS).reset_index(drop=True)
    cal.to_csv(args.output, index=False)
    print(f"Generato {len(cal)} righe in {args.output}")


if __name__ == "__main__":
    main()
