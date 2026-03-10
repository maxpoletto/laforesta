#!/usr/bin/env python3
"""
Confronta due snapshot CSV mannesi e riporta le differenze.

Output:
  Sezione 1 (DIFFERENZE) — righe modificate (filtrate per evidenziare solo cambiamenti importanti)
  Sezione 2 (DUPLICATI)  — righe con chiave duplicata nel file più recente
  Sezione 3 (INCOERENZE) — Governo e Ceduo? incoerenti nel file nuovo

Usage: python3 diff_mannesi.py [--anche_nuovi] VECCHIO.csv NUOVO.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEDUP_KEY = ["Data", "VDP", "Squadra", "Q.li", "Tipo"]
SORT_COLS = ["_date", "VDP", "Squadra", "Q.li", "Tipo"]
NORMALIZE_COLS = ["Ceduo?"]
NOT_AVAILABLE = "n/d"
STANDARD_NOTES = {"", "catastrofato", "fitosanitario", "PSR"}

SUMMARY_COLS = [
    "Data", "VDP", "Squadra", "Q.li", "Tipo",
    "Compresa", "Particella", "Ceduo?", "Governo",
]
NOTE_COLS = DEDUP_KEY + ["Note", "Altre note"]
CP_COLS = DEDUP_KEY + ["Compresa", "Particella", "CP", "Ceduo?", "Governo"]

# Columns explained by Note/CP changes — anything else is "other".
_EXPLAINED = {"Note", "Particella", "CP", "Compresa", "Ceduo?"}


# ── I/O helpers ────────────────────────────────────────────────────

def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["_date"] = pd.to_datetime(df["Data"])
    for col in NORMALIZE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
    return df


def load_governo_lookup(mannesi_path: str) -> dict[tuple[str, str], str]:
    """Build a (Compresa, Particella) -> Governo map from particelle.csv."""
    candidates = [
        Path(mannesi_path).resolve().parent / "particelle.csv",
        Path(__file__).resolve().parent.parent / "data" / "particelle.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig",
                             usecols=["Compresa", "Particella", "Governo"])
            return {
                (str(r["Compresa"]), str(r["Particella"])): str(r["Governo"])
                for _, r in df.iterrows() if pd.notna(r["Governo"])
            }
    return {}


def cell(value) -> str:
    return "" if pd.isna(value) else str(value)


# ── Sorting / dedup ───────────────────────────────────────────────

def _vdp_sort_val(v) -> tuple[int, str]:
    if pd.isna(v):
        return (999999, "")
    s = str(v).strip()
    try:
        return (int(s), "")
    except ValueError:
        m = re.match(r"(\d+)", s)
        return (int(m.group(1)), s) if m else (999999, s)


def _row_sort_key(row: pd.Series) -> tuple:
    return (
        row["_date"],
        _vdp_sort_val(row["VDP"]),
        str(row["Squadra"]) if pd.notna(row["Squadra"]) else "",
        float(row["Q.li"]) if pd.notna(row["Q.li"]) else float("inf"),
        str(row["Tipo"]) if pd.notna(row["Tipo"]) else "",
    )


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df.duplicated(subset=DEDUP_KEY, keep="first")
    return df[~mask].copy(), df[mask].copy()


# ── Merge walk ────────────────────────────────────────────────────

def merge_walk(old_df, new_df, common_cols):
    """Walk two sorted dataframes in parallel, classifying rows."""
    old_keys = [_row_sort_key(old_df.iloc[i]) for i in range(len(old_df))]
    new_keys = [_row_sort_key(new_df.iloc[i]) for i in range(len(new_df))]

    changed, only_old, only_new = [], [], []
    i = j = 0
    while i < len(old_keys) and j < len(new_keys):
        ok, nk = old_keys[i], new_keys[j]
        if ok == nk:
            orow, nrow = old_df.iloc[i], new_df.iloc[j]
            diffs = [c for c in common_cols if cell(orow[c]) != cell(nrow[c])]
            if diffs:
                changed.append((orow, nrow, diffs))
            i += 1
            j += 1
        elif ok < nk:
            only_old.append(old_df.iloc[i])
            i += 1
        else:
            only_new.append(new_df.iloc[j])
            j += 1
    while i < len(old_df):
        only_old.append(old_df.iloc[i])
        i += 1
    while j < len(new_df):
        only_new.append(new_df.iloc[j])
        j += 1
    return changed, only_old, only_new


# ── Classification helpers ────────────────────────────────────────

def _particella_num(val: str) -> str | None:
    m = re.match(r"(\d+)", val.strip())
    return m.group(1) if m else None


def _is_meaningful_note(orow, nrow) -> bool:
    old_note = cell(orow["Note"])
    new_note = cell(nrow["Note"])
    new_altre = cell(nrow["Altre note"]) if "Altre note" in nrow.index else ""
    # "malato" elaborated into "fitosanitario" + "... malato" is routine.
    if new_note in STANDARD_NOTES and old_note and old_note in new_altre:
        return False
    return new_altre != old_note or new_note not in STANDARD_NOTES


def _is_meaningful_cp(orow, nrow) -> bool:
    if cell(orow["Compresa"]) != cell(nrow["Compresa"]):
        return True
    return _particella_num(cell(orow["Particella"])) != _particella_num(
        cell(nrow["Particella"]))


# ── Table printing ────────────────────────────────────────────────

def _row_dict(row, cols, governo_map):
    """Build a display dict, looking up Governo when needed."""
    d = {c: cell(row[c]) if c in row.index else "" for c in cols if c != "Governo"}
    if "Governo" in cols:
        d["Governo"] = governo_map.get(
            (cell(row["Compresa"]), cell(row["Particella"])), NOT_AVAILABLE)
    return d


def print_table(rows, cols, markers=None, highlights=None):
    """Print aligned table with optional -/+ markers and red highlights."""
    marker_w = (max(len(m) for m in markers) + 1) if markers else 0
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(r.get(c, "")))

    header = " " * marker_w + "  ".join(c.rjust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows):
        pfx = (markers[i].ljust(marker_w - 1) + " ") if markers else ""
        parts = []
        for c in cols:
            val = r.get(c, "").rjust(widths[c])
            if highlights and i < len(highlights) and c in highlights[i]:
                val = RED + val + RESET
            parts.append(val)
        print(pfx + "  ".join(parts))


def _print_diff_group(items, cols, governo_map, label, skipped=0, skip_label=""):
    """Print one group of changed rows (CP, Note, or Other)."""
    if skipped:
        print(f"  ({skipped} {skip_label})")
    if not items:
        return
    # Collect any extra diff columns beyond the base set.
    all_cols = list(cols)
    for _, _, diffs in items:
        for c in diffs:
            if c not in all_cols:
                all_cols.append(c)
    print(f"{BOLD}--- {len(items)} righe: {label} ---{RESET}")
    table_rows, markers, highlights = [], [], []
    for orow, nrow, diffs in items:
        for marker, row in [("-", orow), ("+", nrow)]:
            table_rows.append(_row_dict(row, all_cols, governo_map))
            markers.append(marker)
            highlights.append(set(diffs) if marker == "+" else set())
    print_table(table_rows, all_cols, markers, highlights)
    print()


# ── Main diff section ─────────────────────────────────────────────

def print_diff_section(changed, only_old, only_new, governo_map,
                       show_new=False):
    print(f"\n{BOLD}=== DIFFERENZE ==={RESET}")
    print(f"  Totale modificate:    {len(changed)}")
    print(f"  Solo nel vecchio:     {len(only_old)}")
    print(f"  Solo nel nuovo:       {len(only_new)}\n")

    cp_rows, note_rows, other_rows = [], [], []
    cp_skip = note_skip = 0

    for orow, nrow, diffs in changed:
        has_cp = "CP" in diffs
        has_note = "Note" in diffs
        m_cp = has_cp and _is_meaningful_cp(orow, nrow)
        m_note = has_note and _is_meaningful_note(orow, nrow)

        if m_cp:
            cp_rows.append((orow, nrow, diffs))
        elif has_cp:
            cp_skip += 1

        if m_note and not m_cp:
            note_rows.append((orow, nrow, diffs))
        elif has_note and not m_cp:
            note_skip += 1

        if not m_cp and not m_note and set(diffs) - _EXPLAINED:
            other_rows.append((orow, nrow, diffs))

    _print_diff_group(cp_rows, CP_COLS, governo_map,
                      "CP modificate", cp_skip,
                      "righe CP omesse: solo cambio di suffisso")
    _print_diff_group(note_rows, NOTE_COLS, governo_map,
                      "Note modificate", note_skip,
                      "righe Note omesse: reclassificazione standard")
    if other_rows:
        _print_diff_group(other_rows, DEDUP_KEY, governo_map, "altre modifiche")

    if only_old:
        print(f"{BOLD}--- Solo nel vecchio ({len(only_old)}) ---{RESET}")
        print_table([_row_dict(r, SUMMARY_COLS, governo_map) for r in only_old],
                    SUMMARY_COLS)
        print()
    if only_new and show_new:
        print(f"{BOLD}--- Solo nel nuovo ({len(only_new)}) ---{RESET}")
        print_table([_row_dict(r, SUMMARY_COLS, governo_map) for r in only_new],
                    SUMMARY_COLS)
        print()


# ── Governo/Ceduo? mismatches ─────────────────────────────────────

def find_governo_ceduo_mismatches(df, governo_map):
    results, seen = [], set()
    for i in range(len(df)):
        row = df.iloc[i]
        cp = (cell(row["Compresa"]), cell(row["Particella"]))
        governo = governo_map.get(cp)
        if governo is None:
            continue
        ceduo = cell(row["Ceduo?"]).upper()
        if not ((governo == "Fustaia" and ceduo == "TRUE")
                or (governo == "Ceduo" and ceduo == "FALSE")):
            continue
        key = (*cp, ceduo)
        if key in seen:
            continue
        seen.add(key)
        d = {c: cell(row[c]) for c in SUMMARY_COLS if c != "Governo"}
        d["Governo"] = governo
        results.append(d)
    return results


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 2:
        print(f"Uso: {sys.argv[0]} [--anche_nuovi] VECCHIO.csv NUOVO.csv",
              file=sys.stderr)
        sys.exit(1)
    show_new = "--anche_nuovi" in flags

    df_old = read_csv(args[0])
    df_new = read_csv(args[1])
    governo_map = load_governo_lookup(args[1])
    common_cols = [c for c in df_old.columns if c in df_new.columns and c != "_date"]

    old_uniq, _ = deduplicate(df_old)
    new_uniq, new_dupes = deduplicate(df_new)

    old_sorted = old_uniq.sort_values(SORT_COLS).reset_index(drop=True)
    new_sorted = new_uniq.sort_values(SORT_COLS).reset_index(drop=True)
    changed, only_old, only_new = merge_walk(old_sorted, new_sorted, common_cols)

    print_diff_section(changed, only_old, only_new, governo_map, show_new)

    print(f"{BOLD}=== DUPLICATI (file nuovo): {len(new_dupes)} righe ==={RESET}")
    if len(new_dupes) > 0:
        print_table([_row_dict(new_dupes.iloc[i], SUMMARY_COLS, governo_map)
                     for i in range(len(new_dupes))], SUMMARY_COLS)
    print()

    mismatches = find_governo_ceduo_mismatches(new_uniq, governo_map)
    print(f"{BOLD}=== INCOERENZE GOVERNO/CEDUO: {len(mismatches)} righe ==={RESET}")
    if mismatches:
        print_table(mismatches, SUMMARY_COLS,
                    highlights=[{"Ceduo?", "Governo"}] * len(mismatches))
    print()


if __name__ == "__main__":
    main()
