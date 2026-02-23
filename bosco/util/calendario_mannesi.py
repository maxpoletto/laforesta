#!/usr/bin/env python3
"""Convert mannesi.csv (delivery records) to calendario-tagli format (one row per parcel/year)."""

import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
cal = (
    df.groupby(["Anno", "Compresa", "Particella"], sort=True)["Ceduo?"]
    .apply(lambda s: "Ceduo" if s.mean() > 0.5 else "Fustaia")
    .rename("Governo")
    .reset_index()
)
cal.to_csv(sys.argv[2], index=False)
print(f"Wrote {len(cal)} rows to {sys.argv[2]}")
