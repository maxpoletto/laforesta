"""One-off ETL: convert La Foresta's legacy CSVs into the canonical abies inputs
that ``manage.py bootstrap`` consumes.

This is the "sub-project 2" converter of the import-decoupling effort.  It reads
the legacy files in ``src_dir`` (see ``abies-data/README``) and writes a directory
of canonical CSVs (plus ``terreni.geojson``) in the canonical dialect:

    read  utf-8-sig, write utf-8, comma-delimited, dot decimals.

Stdlib ``csv`` only — no project imports, no new dependencies — so it can run
standalone against a checkout of the legacy data.  The *column names* it emits
are the canonical headers documented in ``config/strings_it.py`` (the
``S.CSV_COL_*`` constants) and exercised by ``test/test_bootstrap.py``; they are
duplicated here as literals on purpose, to keep this throwaway tool decoupled
from the Django app.

See ``ingest/README.md`` for the full source→canonical mapping, the assumptions
made on ambiguous tree-survey data, and the files deferred to a later increment.

Run:

    python3 -m ingest.convert_laforesta <src_dir> <out_dir>
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# --- canonical dialect ------------------------------------------------------

READ_ENCODING = 'utf-8-sig'   # legacy files may carry a BOM
WRITE_ENCODING = 'utf-8'
DELIMITER = ','               # canonical pairing: comma field / dot decimal

# --- magic names emitted into the canonical files ---------------------------

GRID_NAME = 'Aree di saggio PDG 2026'
PLAN_NAME = 'PDG 2026'
SURVEY_CALCULATED = 'Campionamento calcolato'
SURVEY_HEIGHTS = 'Campionamento altezze'
DEFAULT_SURVEY_DATE = '2024-09-15'

# Legacy hard-coded eclass rule: comparti A–E are high forest (fustaia,
# non-coppice); comparto F is coppice (ceduo).
COPPICE_COMPARTO = 'F'

# Canonical product names (the distinct values of apps.base.refdata.PRODUCT_MAP).
# Duplicated here as literals to keep this ETL standalone; if PRODUCT_MAP changes
# this list should be revisited.
CANONICAL_PRODUCTS = [
    'Tronchi', 'Cippato', 'Ramaglia', 'Pertiche-Puntelli', 'Pertiche-Tronchi',
]

# --- legacy source filenames ------------------------------------------------

SRC_PARCELS = 'particelle.csv'
SRC_CREWS = 'mannesi.csv'
SRC_SAMPLE_AREAS = 'aree-di-saggio.csv'
SRC_TREES_CALCULATED = 'alberi-calcolati.csv'
SRC_TREES_HEIGHTS = 'alberi-altezze.csv'
SRC_HYPSO = 'equazioni_ipsometro.csv'
SRC_PLAN_FUSTAIA = 'piano_fustaia.csv'
SRC_PLAN_CEDUO = 'piano_ceduo.csv'
SRC_GEOJSON = 'terreni.geojson'
SRC_SPECIES = 'apps/base/data/species.csv'  # in-repo canonical species list

# --- canonical output filenames (must match bootstrap's expectations) -------

OUT_REGIONS = 'regions.csv'
OUT_ECLASSES = 'eclasses.csv'
OUT_CREWS = 'crews.csv'
OUT_SPECIES = 'species.csv'
OUT_PRODUCTS = 'products.csv'
OUT_PARCELS = 'particelle.csv'
OUT_SAMPLE_GRIDS = 'sample_grids.csv'
OUT_HARVEST_PLANS = 'harvest_plans.csv'
OUT_SAMPLE_AREAS = 'sample_areas.csv'
OUT_SURVEYS = 'surveys.csv'
OUT_SAMPLED_TREES = 'sampled-trees.csv'
OUT_HYPSO = 'hypso_params.csv'
OUT_GEOJSON = 'terreni.geojson'

# --- canonical column headers (the S.CSV_COL_* literals) --------------------

COL_REGION = 'Compresa'
COL_CLASS = 'Comparto'
COL_COPPICE = 'Ceduo'
COL_CREW = 'Squadra'
COL_PRODUCT = 'Tipo'
COL_PARCEL = 'Particella'
COL_AREA_HA = 'Area (ha)'
COL_AVE_AGE = 'Età media'
COL_LOCATION = 'Località'
COL_ALT_MIN = 'Altitudine min'
COL_ALT_MAX = 'Altitudine max'
COL_ASPECT = 'Esposizione'
COL_GRADE_PCT = 'Pendenza %'
COL_GEO_DESC = 'Stazione'       # geological station
COL_VEG_DESC = 'Soprassuolo'    # vegetation description
COL_GRID = 'Griglia'
COL_PLAN = 'Piano'
COL_YEAR_START = 'Anno inizio'
COL_YEAR_END = 'Anno fine'
COL_SAMPLE_AREA = 'Area saggio'
COL_LON = 'Lon'
COL_LAT = 'Lat'
COL_ALT = 'Quota'
COL_RADIUS = 'Raggio'
COL_SURVEY = 'Rilevamento'
COL_DATA = 'Data'
COL_TREE = 'Albero'
COL_SHOOT = 'Pollone'
COL_STANDARD = 'Matricina'
COL_D_CM = 'D_cm'
COL_H_M = 'H_m'
COL_L10_MM = 'L10_mm'
COL_SPECIES = 'Genere'
COL_HIGHFOREST = 'Fustaia'

# Species reference headers (canonical SPECIES RefTable column names).
COL_LATIN = 'Nome latino'
COL_DENSITY = 'Densità (q/m³)'
COL_MINOR = 'Minore'
COL_SORT_ORDER = 'Ordine'

# Legacy plan year column (shared by piano_fustaia / piano_ceduo).
LEGACY_COL_ANNO = 'Anno'

# Legacy tree column names.
LEGACY_TREE_N = 'n'
LEGACY_TREE_POLL = 'poll'
LEGACY_TREE_D = 'D(cm)'
LEGACY_TREE_H = 'h(m)'
LEGACY_TREE_L10 = 'L10(mm)'

# In alberi-calcolati the ``poll`` column is normally a shoot number, but the
# sentinel ``mat`` marks the tree as a coppice standard (matricina) rather than a
# numbered shoot.  Canonically that is the dedicated ``Matricina`` boolean, so a
# ``mat`` row maps to Matricina=True with a blank Pollone (→ 0).
LEGACY_POLL_STANDARD = 'mat'


# --- IO helpers -------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    """Read a legacy CSV (delimiter auto-detected as ';' or ',') into dicts."""
    text = path.read_text(encoding=READ_ENCODING)
    delimiter = ';' if ';' in text.split('\n', 1)[0] else ','
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def _write(path: Path, header: list[str], rows: list[list]) -> int:
    """Write rows under header in the canonical dialect.  Returns the row count."""
    with path.open('w', encoding=WRITE_ENCODING, newline='') as f:
        writer = csv.writer(f, delimiter=DELIMITER)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def _distinct(rows: list[dict], col: str) -> list[str]:
    """Sorted distinct non-blank stripped values of ``col``."""
    return sorted({(r.get(col) or '').strip() for r in rows if (r.get(col) or '').strip()})


# --- converters (one per canonical file) ------------------------------------

def _convert_regions(parcels: list[dict], out_dir: Path) -> int:
    rows = [[v] for v in _distinct(parcels, COL_REGION)]
    return _write(out_dir / OUT_REGIONS, [COL_REGION], rows)


def _convert_eclasses(parcels: list[dict], out_dir: Path) -> int:
    rows = [[c, 1 if c == COPPICE_COMPARTO else 0]
            for c in _distinct(parcels, COL_CLASS)]
    return _write(out_dir / OUT_ECLASSES, [COL_CLASS, COL_COPPICE], rows)


def _convert_crews(src_dir: Path, out_dir: Path) -> int:
    crews = _read(src_dir / SRC_CREWS)
    rows = [[v] for v in _distinct(crews, COL_CREW)]
    return _write(out_dir / OUT_CREWS, [COL_CREW], rows)


def _convert_species(repo_root: Path, out_dir: Path) -> int:
    """Reshape the in-repo canonical species list into the localized SPECIES
    RefTable headers (Genere, Nome latino, Densità (q/m³), Minore, Ordine)."""
    species = _read(repo_root / SRC_SPECIES)
    header = [COL_SPECIES, COL_LATIN, COL_DENSITY, COL_MINOR, COL_SORT_ORDER]
    rows = [
        [s['common'], s['latin'], s['density_q_m3'], s['minor'], s['sort_order']]
        for s in species
    ]
    return _write(out_dir / OUT_SPECIES, header, rows)


def _convert_products(out_dir: Path) -> int:
    rows = [[p] for p in CANONICAL_PRODUCTS]
    return _write(out_dir / OUT_PRODUCTS, [COL_PRODUCT], rows)


def _convert_parcels(parcels: list[dict], out_dir: Path) -> int:
    """Canonical subset of legacy particelle: drop CP, Governo, Piano del
    taglio, Parametro, Matricine; keep/reorder the rest (header names already
    match the legacy file)."""
    header = [
        COL_REGION, COL_CLASS, COL_PARCEL, COL_AREA_HA, COL_AVE_AGE, COL_LOCATION,
        COL_ALT_MIN, COL_ALT_MAX, COL_ASPECT, COL_GRADE_PCT, COL_GEO_DESC,
        COL_VEG_DESC,
    ]
    rows = [[(r.get(c) or '').strip() for c in header] for r in parcels]
    return _write(out_dir / OUT_PARCELS, header, rows)


def _convert_sample_grids(out_dir: Path) -> int:
    return _write(out_dir / OUT_SAMPLE_GRIDS, [COL_GRID], [[GRID_NAME]])


def _convert_harvest_plans(src_dir: Path, out_dir: Path) -> int:
    """One harvest plan ``PDG 2026`` spanning the min..max Anno across the
    legacy fustaia and ceduo plan files."""
    years = []
    for fn in (SRC_PLAN_FUSTAIA, SRC_PLAN_CEDUO):
        for r in _read(src_dir / fn):
            raw = (r.get(LEGACY_COL_ANNO) or '').strip()
            if raw:
                years.append(int(raw))
    if not years:
        raise ValueError(
            f'no {LEGACY_COL_ANNO} values in {SRC_PLAN_FUSTAIA}/{SRC_PLAN_CEDUO}')
    header = [COL_PLAN, COL_YEAR_START, COL_YEAR_END]
    rows = [[PLAN_NAME, min(years), max(years)]]
    return _write(out_dir / OUT_HARVEST_PLANS, header, rows)


def _convert_sample_areas(src_dir: Path, out_dir: Path) -> int:
    """Select Lon/Lat/Quota/Raggio from legacy aree-di-saggio; add the constant
    Griglia; drop CP / UTM* / Gauss*."""
    areas = _read(src_dir / SRC_SAMPLE_AREAS)
    header = [COL_GRID, COL_REGION, COL_PARCEL, COL_SAMPLE_AREA,
              COL_LON, COL_LAT, COL_ALT, COL_RADIUS]
    rows = [
        [GRID_NAME] + [(r.get(c) or '').strip()
                       for c in (COL_REGION, COL_PARCEL, COL_SAMPLE_AREA,
                                 COL_LON, COL_LAT, COL_ALT, COL_RADIUS)]
        for r in areas
    ]
    return _write(out_dir / OUT_SAMPLE_AREAS, header, rows)


def _convert_surveys(out_dir: Path) -> int:
    header = [COL_SURVEY, COL_GRID, COL_DATA]
    rows = [
        [SURVEY_CALCULATED, GRID_NAME, DEFAULT_SURVEY_DATE],
        [SURVEY_HEIGHTS, GRID_NAME, DEFAULT_SURVEY_DATE],
    ]
    return _write(out_dir / OUT_SURVEYS, header, rows)


def _convert_sampled_trees(src_dir: Path, out_dir: Path) -> int:
    """Union of the two tree surveys into one canonical sampled-trees file.

    - ``alberi-calcolati`` (survey ``Campionamento calcolato``) carries the
      Albero/Pollone/D/H/L10 detail directly; Matricina is left blank (the
      strict core defaults blank Matricina → False, blank Pollone/L10 → 0).
    - ``alberi-altezze`` (survey ``Campionamento altezze``) has the correct
      height data but lacks the Albero/Pollone/Matricina shape, so Albero is
      synthesized as a 1-based sequence per (Compresa, Particella, Area saggio)
      and Pollone/Matricina/L10 are left blank.

    NOTE: ``alberi-columns.csv`` has the Albero/Pollone/Matricina shape but the
    README states its *data* is wrong, so it is deliberately NOT merged.  The
    Matricina/Pollone/L10 detail it would provide for the heights survey is left
    as a reviewer decision (see ingest/README.md).
    """
    header = [
        COL_SURVEY, COL_REGION, COL_PARCEL, COL_SAMPLE_AREA, COL_TREE, COL_SHOOT,
        COL_STANDARD, COL_D_CM, COL_H_M, COL_L10_MM, COL_SPECIES, COL_HIGHFOREST,
    ]
    rows: list[list] = []
    rows += _calculated_tree_rows(_read(src_dir / SRC_TREES_CALCULATED))
    rows += _heights_tree_rows(_read(src_dir / SRC_TREES_HEIGHTS))
    return _write(out_dir / OUT_SAMPLED_TREES, header, rows)


def _calculated_tree_rows(trees: list[dict]) -> list[list]:
    out = []
    for r in trees:
        poll = (r.get(LEGACY_TREE_POLL) or '').strip()
        if poll.lower() == LEGACY_POLL_STANDARD:
            shoot, standard = '', 'True'   # coppice standard (matricina)
        else:
            shoot, standard = poll, ''     # numbered shoot (blank → 0)
        out.append([
            SURVEY_CALCULATED,
            (r.get(COL_REGION) or '').strip(),
            (r.get(COL_PARCEL) or '').strip(),
            (r.get(COL_SAMPLE_AREA) or '').strip(),
            (r.get(LEGACY_TREE_N) or '').strip(),       # Albero
            shoot,                                       # Pollone
            standard,                                    # Matricina
            (r.get(LEGACY_TREE_D) or '').strip(),       # D_cm
            (r.get(LEGACY_TREE_H) or '').strip(),       # H_m
            (r.get(LEGACY_TREE_L10) or '').strip(),     # L10_mm
            (r.get(COL_SPECIES) or '').strip(),
            (r.get(COL_HIGHFOREST) or '').strip(),
        ])
    return out


def _heights_tree_rows(trees: list[dict]) -> list[list]:
    """Synthesize Albero as a 1-based sequence per (region, parcel, area)."""
    seq: dict[tuple, int] = {}
    out = []
    for r in trees:
        region = (r.get(COL_REGION) or '').strip()
        parcel = (r.get(COL_PARCEL) or '').strip()
        area = (r.get(COL_SAMPLE_AREA) or '').strip()
        key = (region, parcel, area)
        seq[key] = seq.get(key, 0) + 1
        out.append([
            SURVEY_HEIGHTS, region, parcel, area,
            seq[key],   # synthesized Albero
            '',         # Pollone (blank → 0)
            '',         # Matricina (blank → False)
            (r.get(LEGACY_TREE_D) or '').strip(),   # D_cm
            (r.get(LEGACY_TREE_H) or '').strip(),   # H_m
            '',         # L10_mm (blank → 0)
            (r.get(COL_SPECIES) or '').strip(),
            (r.get(COL_HIGHFOREST) or '').strip(),
        ])
    return out


def _convert_hypso(src_dir: Path, out_dir: Path) -> int:
    """Copy the legacy ipsometer equations verbatim (its lowercase headers are
    accepted case-insensitively by hypsometry.parse_param_csv)."""
    src = src_dir / SRC_HYPSO
    dst = out_dir / OUT_HYPSO
    text = src.read_text(encoding=READ_ENCODING)
    dst.write_text(text, encoding=WRITE_ENCODING)
    # Report the data-row count (exclude header) for the summary.
    return max(0, len(text.strip().splitlines()) - 1)


def _copy_geojson(src_dir: Path, out_dir: Path) -> None:
    shutil.copyfile(src_dir / SRC_GEOJSON, out_dir / OUT_GEOJSON)


# --- entry point ------------------------------------------------------------

def main(src_dir: Path, out_dir: Path) -> dict[str, int]:
    """Convert all legacy files in ``src_dir`` into canonical files in
    ``out_dir``.  Returns a ``{filename: row_count}`` summary (geojson is copied
    verbatim and omitted from the counts)."""
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The in-repo species list lives under the abies project root, which is the
    # parent of this ``ingest`` package.
    repo_root = Path(__file__).resolve().parent.parent

    parcels = _read(src_dir / SRC_PARCELS)

    counts = {
        OUT_REGIONS: _convert_regions(parcels, out_dir),
        OUT_ECLASSES: _convert_eclasses(parcels, out_dir),
        OUT_CREWS: _convert_crews(src_dir, out_dir),
        OUT_SPECIES: _convert_species(repo_root, out_dir),
        OUT_PRODUCTS: _convert_products(out_dir),
        OUT_PARCELS: _convert_parcels(parcels, out_dir),
        OUT_SAMPLE_GRIDS: _convert_sample_grids(out_dir),
        OUT_HARVEST_PLANS: _convert_harvest_plans(src_dir, out_dir),
        OUT_SAMPLE_AREAS: _convert_sample_areas(src_dir, out_dir),
        OUT_SURVEYS: _convert_surveys(out_dir),
        OUT_SAMPLED_TREES: _convert_sampled_trees(src_dir, out_dir),
        OUT_HYPSO: _convert_hypso(src_dir, out_dir),
    }
    _copy_geojson(src_dir, out_dir)
    return counts


def _cli(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert La Foresta legacy CSVs into canonical abies '
                    'bootstrap inputs.')
    parser.add_argument('src_dir', type=Path, help='Legacy data directory.')
    parser.add_argument('out_dir', type=Path, help='Output (canonical) directory.')
    args = parser.parse_args(argv)
    counts = main(args.src_dir, args.out_dir)
    for name, n in counts.items():
        print(f'  {name:<22} {n} rows')
    print(f'  {OUT_GEOJSON:<22} copied')
    return 0


if __name__ == '__main__':
    sys.exit(_cli())
