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
PRESSLER_DEFAULT = '2'

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
SRC_HARVESTS = SRC_CREWS
SRC_SAMPLE_AREAS = 'aree-di-saggio.csv'
SRC_TREES_CALCULATED = 'alberi-calcolati.csv'
SRC_TREES_HEIGHTS = 'alberi-altezze.csv'
SRC_HYPSO = 'equazioni_ipsometro.csv'
SRC_PLAN_FUSTAIA = 'piano_fustaia.csv'
SRC_PLAN_CEDUO = 'piano_ceduo.csv'
SRC_PAI = 'piante-accrescimento-indefinito.csv'
SRC_GEOJSON = 'terreni.geojson'
SRC_SPECIES = 'apps/base/data/species.csv'  # in-repo canonical species list

# --- canonical output filenames (must match bootstrap's expectations) -------

OUT_REGIONS = 'regions.csv'
OUT_ECLASSES = 'eclasses.csv'
OUT_CREWS = 'crews.csv'
OUT_SPECIES = 'species.csv'
OUT_PRODUCTS = 'products.csv'
OUT_TRACTORS = 'tractors.csv'
OUT_PARCELS = 'particelle.csv'
OUT_SAMPLE_GRIDS = 'sample_grids.csv'
OUT_HARVEST_PLANS = 'harvest_plans.csv'
OUT_SAMPLE_AREAS = 'sample_areas.csv'
OUT_SURVEYS = 'surveys.csv'
OUT_SAMPLED_TREES = 'sampled-trees.csv'
OUT_HYPSO = 'hypso_params.csv'
OUT_HARVESTS = 'harvests.csv'
OUT_HARVEST_PLAN_ITEMS = 'harvest_plan_items.csv'
OUT_PRESERVED = 'preserved-trees.csv'
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
COL_PRESSLER = 'Pressler'
COL_SPECIES = 'Genere'
COL_HIGHFOREST = 'Fustaia'
COL_ACTIVE = 'Attivo'

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

# --- tractor / harvest / plan-item / preserved-tree column headers -----------

# Canonical tractor-table headers (S.CSV_COL_TRACTOR_NAME / _MANUFACTURER / _MODEL / CSV_COL_YEAR).
COL_TRACTOR_NAME = 'Trattore'
COL_MANUFACTURER = 'Produttore'
COL_MODEL        = 'Modello'
COL_YEAR         = 'Anno'

# Canonical harvest column headers (static portion).
COL_QUINTALS        = 'Q.li'
COL_VDP             = 'VDP'
COL_PROT            = 'Prot.'
COL_HARVEST_DAMAGED   = 'Danneggiato'
COL_HARVEST_UNHEALTHY = 'Fitosanitario'
COL_HARVEST_PSR       = 'PSR'
COL_EXTRA_NOTE        = 'Altre note'

# Canonical harvest-plan-item column headers.
COL_HARVEST_M3    = 'Prelievo (m³)'
COL_SURFACE_HA    = 'Superficie intervento (ha)'
COL_PERIOD_Y      = 'Turno (a)'
COL_NOTE          = 'Note'

# Dynamic-column prefix literals (must match S.CSV_COL_SPECIES_PREFIX / _TRACTOR_PREFIX).
SPECIES_PREFIX = 'Specie:'
TRACTOR_PREFIX = 'Trattore:'

# Hard-coded La Foresta tractors: [name, manufacturer, model, year(blank)].
# The ``name`` values are the canonical Tractor.name keys matched case-sensitively
# in the harvest dynamic columns.
_TRACTORS = [
    ['Equus 175N UN', 'Equus', '175N UN', ''],
    ['Fiat 110-90',   'Fiat',  '110-90',  ''],
    ['Fiat 80-66',    'Fiat',  '80-66',   ''],
    ['Landini 135',   'Landini', '135',   ''],
    ['New Holland T5050', 'New Holland', 'T5050', ''],
]

# Map legacy mannesi.csv ``Tipo`` values → canonical product names.
# The three matching values are identity; only the two long descriptive names differ.
_PRODUCT_MAP = {
    'Tronchi':                                    'Tronchi',
    'Cippato':                                    'Cippato',
    'Ramaglia':                                   'Ramaglia',
    'Pertiche-puntelli-tronchi castagno venduti': 'Pertiche-Puntelli',
    'Pertiche-tronchi castagno':                  'Pertiche-Tronchi',
}

# Map mannesi.csv species %-columns → canonical Species.common_name (exact case).
# The key is the full column header (e.g. ``'abete %'``); the value is the
# common_name as it appears in species.csv (``'Abete'``, etc.).
# ``pino`` is Pino Laricio, the dominant pine of the Calabrian forests in this dataset.
_SPECIES_COL_MAP = {
    'abete %':    'Abete',
    'pino %':     'Pino Laricio',
    'douglas %':  'Douglas',
    'faggio %':   'Faggio',
    'castagno %': 'Castagno',
    'ontano %':   'Ontano',
    'altro %':    'Altro',
}

# Map mannesi.csv tractor %-columns → canonical Tractor.name (exact case, from _TRACTORS).
_TRACTOR_COL_MAP = {
    'Equus %':              'Equus 175N UN',
    'Fiat 110-90 %':        'Fiat 110-90',
    'Fiat 80-66 %':         'Fiat 80-66',
    'Landini 135 %':        'Landini 135',
    'New Holland T5050 %':  'New Holland T5050',
}

# Map the legacy Note flag tokens (lowercase) → (damaged, unhealthy, psr) booleans.
# Multiple tokens can co-occur (comma/space separated in practice but the real
# data has exactly one token per non-blank Note cell).
_NOTE_FLAG_MAP = {
    'catastrofato':  (True,  False, False),
    'fitosanitario': (False, True,  False),
    'psr':           (False, False, True),
}

# Map piante-accrescimento-indefinito.csv ``Genere`` → canonical Species.common_name.
# Only entries that do NOT already match canonical common_name case-insensitively.
# Unmappable species (Betulla, Farnia, Noce, Pioppo) collapse to 'Altro'.
_PAI_SPECIES_MAP = {
    'Abete Bianco':   'Abete',     # Abies alba = canonical 'Abete'
    'Betulla Bianca': 'Altro',
    'Farnia':         'Altro',     # Quercus robur — not in canonical set
    'Noce':           'Altro',
    'Pioppo Tremulo': 'Altro',
}


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
    RefTable headers (Genere, Nome latino, Densità (q/m³), Pressler, Minore, Ordine)."""
    species = _read(repo_root / SRC_SPECIES)
    header = [COL_SPECIES, COL_LATIN, COL_DENSITY, COL_PRESSLER, COL_MINOR, COL_SORT_ORDER]
    rows = [
        [
            s['common'], s['latin'], s['density_q_m3'],
            s.get('pressler_default') or PRESSLER_DEFAULT, s['minor'], s['sort_order'],
        ]
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
    header = [COL_SURVEY, COL_GRID, COL_DATA, COL_ACTIVE]
    rows = [
        [SURVEY_CALCULATED, GRID_NAME, DEFAULT_SURVEY_DATE, 'True'],
        [SURVEY_HEIGHTS, GRID_NAME, DEFAULT_SURVEY_DATE, 'False'],
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
        COL_STANDARD, COL_D_CM, COL_H_M, COL_L10_MM, COL_PRESSLER,
        COL_SPECIES, COL_HIGHFOREST,
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
            PRESSLER_DEFAULT,                           # Pressler
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
            PRESSLER_DEFAULT,  # Pressler
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


def _sanitize_int(raw: str) -> str:
    """Return ``raw`` if it parses as a non-fractional integer, else blank.

    Used to sanitize optional integer fields (VDP) that may carry legacy
    non-numeric values (``'nd'``, ``'783 bis'``) or fractional data-entry
    errors (``'360.9'``) in the source data.  The strict import core rejects
    any non-integer value, so non-conforming cells are silenced to blank."""
    s = raw.strip()
    if not s:
        return ''
    try:
        f = float(s)
        if f == int(f):
            return s
        return ''
    except (ValueError, OverflowError):
        return ''


def _convert_tractors(out_dir: Path) -> int:
    """Emit the five hard-coded La Foresta tractors."""
    header = [COL_TRACTOR_NAME, COL_MANUFACTURER, COL_MODEL, COL_YEAR]
    return _write(out_dir / OUT_TRACTORS, header, list(_TRACTORS))


def _parse_note_flags(note: str) -> tuple[str, str, str]:
    """Return ('true'/'false', 'true'/'false', 'true'/'false') for
    (damaged, unhealthy, psr) from a legacy Note flag-string.

    Handles multiple comma- or space-separated tokens by OR-accumulating each
    token's flags.  The single-token case behaves exactly as before.
    All _NOTE_FLAG_MAP keys are single words, so splitting on whitespace after
    normalising commas is correct."""
    damaged = unhealthy = psr = False
    for token in note.strip().lower().replace(',', ' ').split():
        d, u, p = _NOTE_FLAG_MAP.get(token, (False, False, False))
        damaged |= d; unhealthy |= u; psr |= p
    return (
        'true' if damaged   else 'false',
        'true' if unhealthy else 'false',
        'true' if psr       else 'false',
    )


def _convert_harvests(src_dir: Path, out_dir: Path) -> int:
    """Convert mannesi.csv → harvests.csv.

    Dynamic species columns: ``Specie: <common_name>`` in _SPECIES_COL_MAP order.
    Dynamic tractor columns: ``Trattore: <name>`` in _TRACTOR_COL_MAP order.
    ``Particella == 'X'`` → blank (region-wide row).
    ``Note`` token → three explicit boolean flag columns.
    """
    rows = _read(src_dir / SRC_HARVESTS)

    # Build the canonical dynamic column headers in a fixed order so the
    # bootstrap column-resolution and species-sum check see a consistent layout.
    species_headers = [f'{SPECIES_PREFIX} {name}' for name in _SPECIES_COL_MAP.values()]
    tractor_headers = [f'{TRACTOR_PREFIX} {name}' for name in _TRACTOR_COL_MAP.values()]

    static_header = [
        COL_REGION, COL_PARCEL, COL_DATA, COL_CREW, COL_PRODUCT,
        COL_QUINTALS, COL_VDP, COL_PROT,
        COL_HARVEST_DAMAGED, COL_HARVEST_UNHEALTHY, COL_HARVEST_PSR,
        COL_EXTRA_NOTE,
    ]
    header = static_header + species_headers + tractor_headers

    legacy_species_cols = list(_SPECIES_COL_MAP.keys())
    legacy_tractor_cols = list(_TRACTOR_COL_MAP.keys())

    out = []
    vdp_blanked = 0
    for r in rows:
        parcel = (r.get(COL_PARCEL) or '').strip()
        if parcel == 'X':
            parcel = ''

        tipo_raw = (r.get(COL_PRODUCT) or '').strip()
        tipo = _PRODUCT_MAP.get(tipo_raw, tipo_raw)

        damaged, unhealthy, psr = _parse_note_flags(r.get('Note') or '')

        vdp_raw = (r.get(COL_VDP) or '').strip()
        vdp = _sanitize_int(vdp_raw)
        if vdp_raw and not vdp:
            vdp_blanked += 1

        species_pcts = [
            (r.get(col) or '').strip() or '0'
            for col in legacy_species_cols
        ]
        tractor_pcts = [
            (r.get(col) or '').strip() or '0'
            for col in legacy_tractor_cols
        ]

        out.append([
            (r.get(COL_REGION) or '').strip(),
            parcel,
            (r.get(COL_DATA) or '').strip(),
            (r.get(COL_CREW) or '').strip(),
            tipo,
            (r.get(COL_QUINTALS) or '').strip(),
            vdp,
            (r.get(COL_PROT) or '').strip(),
            damaged, unhealthy, psr,
            (r.get(COL_EXTRA_NOTE) or '').strip(),
        ] + species_pcts + tractor_pcts)

    n = _write(out_dir / OUT_HARVESTS, header, out)
    if vdp_blanked > 0:
        print(f'  [warn] {vdp_blanked} VDP values blanked (non-integer; rows kept)', file=sys.stderr)
    return n


def _convert_harvest_plan_items(src_dir: Path, out_dir: Path) -> int:
    """Unified harvest plan items from piano_fustaia.csv + piano_ceduo.csv.

    Highforest rows: ``Piano, Compresa, Particella, Anno, Prelievo (m³),
    Superficie intervento (ha), Turno (a), Danneggiato, Fitosanitario, PSR, Note``.
    Coppice rows: same header, Prelievo left blank.
    ``Particella == 'X'`` → blank (region-wide).
    Boolean flags from the legacy ``Note`` column (fustaia) or left false (ceduo).
    """
    header = [
        COL_PLAN, COL_REGION, COL_PARCEL, COL_YEAR,
        COL_HARVEST_M3, COL_SURFACE_HA, COL_PERIOD_Y,
        COL_HARVEST_DAMAGED, COL_HARVEST_UNHEALTHY, COL_HARVEST_PSR,
        COL_NOTE,
    ]

    out = []

    # Highforest rows from piano_fustaia.csv.
    for r in _read(src_dir / SRC_PLAN_FUSTAIA):
        parcel = (r.get(COL_PARCEL) or '').strip()
        if parcel == 'X':
            parcel = ''
        note_raw = (r.get(COL_NOTE) or '').strip()
        damaged, unhealthy, psr = _parse_note_flags(note_raw)
        out.append([
            PLAN_NAME,
            (r.get(COL_REGION) or '').strip(),
            parcel,
            (r.get(COL_YEAR) or '').strip(),
            (r.get(COL_HARVEST_M3) or '').strip(),  # Prelievo (m³)
            '',                                       # Superficie intervento (ha) — not in fustaia
            '',                                       # Turno (a) — not in fustaia
            damaged, unhealthy, psr,
            '',                                       # free-text Note — unused in fustaia
        ])

    # Coppice rows from piano_ceduo.csv (no damaged/psr flags; Note is free-text).
    for r in _read(src_dir / SRC_PLAN_CEDUO):
        parcel = (r.get(COL_PARCEL) or '').strip()
        if parcel == 'X':
            parcel = ''
        out.append([
            PLAN_NAME,
            (r.get(COL_REGION) or '').strip(),
            parcel,
            (r.get(COL_YEAR) or '').strip(),
            '',                                                    # Prelievo (m³) — not in ceduo
            (r.get(COL_SURFACE_HA) or '').strip(),
            (r.get(COL_PERIOD_Y) or '').strip(),
            'false', 'false', 'false',                            # no flag info in ceduo
            (r.get(COL_NOTE) or '').strip(),                      # free-text Note
        ])

    return _write(out_dir / OUT_HARVEST_PLAN_ITEMS, header, out)


def _convert_preserved(src_dir: Path, out_dir: Path) -> int:
    """Convert piante-accrescimento-indefinito.csv → preserved-trees.csv.

    Maps legacy Genere via _PAI_SPECIES_MAP to canonical common_name.  Passes
    Lon/Lat verbatim.  Rows with missing/blank Lon or Lat are silently skipped
    (a handful of PAI entries have coordinate data quality issues in the legacy
    source, flagged in their Note column).
    """
    header = [COL_REGION, COL_PARCEL, COL_SPECIES, COL_LON, COL_LAT]
    pai_rows = _read(src_dir / SRC_PAI)
    out = []
    for r in pai_rows:
        lon = (r.get(COL_LON) or '').strip()
        lat = (r.get(COL_LAT) or '').strip()
        if not lon or not lat:
            continue
        genere_raw = (r.get('Genere') or '').strip()
        # Use the explicit map first; fall back to identity (canonical names match verbatim).
        genere = _PAI_SPECIES_MAP.get(genere_raw, genere_raw)
        out.append([
            (r.get(COL_REGION) or '').strip(),
            (r.get(COL_PARCEL) or '').strip(),
            genere,
            lon,
            lat,
        ])
    skipped = len(pai_rows) - len(out)
    n = _write(out_dir / OUT_PRESERVED, header, out)
    if skipped > 0:
        print(f'  [warn] {skipped} preserved-tree rows skipped (missing Lon/Lat)', file=sys.stderr)
    return n


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
        OUT_TRACTORS: _convert_tractors(out_dir),
        OUT_PARCELS: _convert_parcels(parcels, out_dir),
        OUT_SAMPLE_GRIDS: _convert_sample_grids(out_dir),
        OUT_HARVEST_PLANS: _convert_harvest_plans(src_dir, out_dir),
        OUT_SAMPLE_AREAS: _convert_sample_areas(src_dir, out_dir),
        OUT_SURVEYS: _convert_surveys(out_dir),
        OUT_SAMPLED_TREES: _convert_sampled_trees(src_dir, out_dir),
        OUT_HYPSO: _convert_hypso(src_dir, out_dir),
        OUT_HARVESTS: _convert_harvests(src_dir, out_dir),
        OUT_HARVEST_PLAN_ITEMS: _convert_harvest_plan_items(src_dir, out_dir),
        OUT_PRESERVED: _convert_preserved(src_dir, out_dir),
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
