# ingest — legacy → canonical ETL

One-off ETL that converts La Foresta's **legacy** survey/plan CSVs into the
**canonical abies CSV inputs** that `manage.py bootstrap` consumes.  This is
"sub-project 2" of the import-decoupling effort: it decouples the messy source
data from the strict canonical contract enforced by `bootstrap`.

## Usage

```sh
python3 -m ingest.convert_laforesta <src_dir> <out_dir>
```

- `src_dir` — the legacy data dir (e.g. `../abies-data`); see its `README`.
- `out_dir` — written with the canonical files below, then loadable with
  `manage.py bootstrap <out_dir> [--check]`.

Dialect: reads `utf-8-sig` (BOM-tolerant), writes `utf-8`, comma-delimited, dot
decimals (the canonical pairing).  Stdlib `csv` only — no project imports, no new
dependencies — so the tool runs standalone against a legacy-data checkout.

## Output files (source → canonical mapping)

| Output | Source | Mapping |
|---|---|---|
| `regions.csv` (`Compresa`) | `particelle.csv` | distinct `Compresa` |
| `eclasses.csv` (`Comparto,Ceduo`) | `particelle.csv` | distinct `Comparto`; `Ceduo=1` iff `Comparto=='F'`, else `0` |
| `crews.csv` (`Squadra`) | `mannesi.csv` | distinct non-blank `Squadra` |
| `species.csv` (`Genere,Nome latino,Densità (q/m³),Pressler,Minore,Ordine`) | in-repo `apps/base/data/species.csv` | reshape `common/latin/density_q_m3/pressler_default/minor/sort_order` to the canonical localized headers (full species set so tree `Genere` resolves) |
| `products.csv` (`Tipo`) | constant | the distinct canonical products of `apps.base.refdata.PRODUCT_MAP.values()` |
| `particelle.csv` | `particelle.csv` | select/reorder `Compresa,Comparto,Particella,Area (ha),Età media,Località,Altitudine min,Altitudine max,Esposizione,Pendenza %,Stazione,Soprassuolo`; drop CP, Governo, Piano del taglio, Parametro, Matricine |
| `sample_grids.csv` (`Griglia`) | constant | one row: `Aree di saggio PDG 2026` |
| `harvest_plans.csv` (`Piano,Anno inizio,Anno fine`) | `piano_fustaia.csv` + `piano_ceduo.csv` | one row `PDG 2026`, `Anno inizio`/`Anno fine` = min/max `Anno` across both files |
| `sample_areas.csv` (`Griglia,Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio`) | `aree-di-saggio.csv` | add the constant `Griglia`; select Lon/Lat/Quota/Raggio; drop CP/UTM\*/Gauss\* |
| `surveys.csv` (`Rilevamento,Griglia,Data`) | constant | two rows, both `Griglia=Aree di saggio PDG 2026`, `Data=2024-09-15`: `Campionamento calcolato`, `Campionamento altezze` |
| `sampled-trees.csv` | `alberi-calcolati.csv` + `alberi-altezze.csv` | union of the two surveys (see below) |
| `hypso_params.csv` | `equazioni_ipsometro.csv` | copied verbatim (lowercase headers accepted case-insensitively by `hypsometry.parse_param_csv`) |
| `tractors.csv` (`Trattore,Produttore,Modello,Anno`) | constant | the five hard-coded La Foresta tractors |
| `harvests.csv` | `mannesi.csv` | `Tipo` via `_PRODUCT_MAP`; `Particella='X'` → blank; species `abete %` etc. → `Specie: <canonical>` dynamic cols; `Equus %` etc. → `Trattore: <name>` dynamic cols; `Note` token → `Danneggiato/Fitosanitario/PSR` booleans; invalid VDP values (``nd``, ``bis`` variants, fractional) → blank |
| `harvest_plan_items.csv` | `piano_fustaia.csv` + `piano_ceduo.csv` | unified rows with `Piano=PDG 2026`; `Particella='X'` → blank; highforest flags from `Note` token; coppice flags left `false` (no flag column in legacy) |
| `preserved-trees.csv` (`Compresa,Particella,Genere,Lon,Lat`) | `piante-accrescimento-indefinito.csv` | `Genere` via `_PAI_SPECIES_MAP` to canonical common name; rows missing `Lon`/`Lat` silently skipped |
| `terreni.geojson` | `terreni.geojson` | copied verbatim (bootstrap does not load it, but the canonical dir carries the source geometry) |

## Tree-survey assumptions (the interesting part)

`sampled-trees.csv` is the **union** of two surveys, mapped to the canonical
`Rilevamento,Compresa,Particella,Area saggio,Albero,Pollone,Matricina,D_cm,H_m,L10_mm,Pressler,Genere,Fustaia`:

1. **`Campionamento calcolato`** (from `alberi-calcolati.csv`): `n`→`Albero`,
   `poll`→`Pollone`, `D(cm)`→`D_cm`, `h(m)`→`H_m`, `L10(mm)`→`L10_mm`,
   `Pressler` is set to `2`, plus `Genere`, `Fustaia`.
   - **`poll == 'mat'` sentinel:** in the legacy file the `poll` column is
     normally a shoot number, but the literal `mat` marks the tree as a coppice
     **standard** (matricina), not a numbered shoot.  Canonically that is the
     dedicated `Matricina` boolean, so a `mat` row maps to `Matricina=True` with
     a blank `Pollone` (the strict core defaults blank Pollone → 0).  All 87
     `mat` rows are coppice (`Fustaia=FALSE`), consistent with this reading.
     Non-`mat` rows leave `Matricina` blank (→ False).

2. **`Campionamento altezze`** (from `alberi-altezze.csv`): `D(cm)`→`D_cm`,
   `h(m)`→`H_m`, `Genere`, `Fustaia`.  This file has the **correct** height
   data but lacks the Albero/Pollone/Matricina shape, so:
   - `Albero` is **synthesized** as a 1-based sequence per
     `(Compresa, Particella, Area saggio)`.
   - `Pollone`, `Matricina`, `L10_mm` are left **blank** (strict core defaults
     blank Pollone/L10 → 0, blank Matricina → False), and `Pressler` is set to
     `2`.

3. **`alberi-columns.csv` tension (left for review):** `alberi-columns.csv` has
   the useful `Albero/Pollone/Matricina` shape, but the legacy `README` states
   its **data is wrong**.  We deliberately do **not** merge it and do **not**
   attempt a fuzzy join.  The Matricina/Pollone/L10 detail it could provide for
   the *heights* survey is therefore **not** carried — it is left as a reviewer
   decision.  `alberi-columns.csv` is not imported.

`Genere` values are passed through **verbatim**; `csv_trees` applies its own
`GENERE_MAP` synonym mapping (e.g. `Pino`→`Pino Laricio`, lowercase `leccio`
resolves case-insensitively) when loading.

## Eclass coppice rule

The legacy eclass set is hard-coded A–E = high forest (non-coppice), F = coppice.
The converter reproduces this purely from the data: `Ceduo=1` iff `Comparto=='F'`.

## Legacy data-quality sanitization

The converter silently sanitizes a handful of non-conforming legacy values to
allow a clean bootstrap load:

- **VDP column** (`mannesi.csv`): 57 rows carry non-integer values (`nd`,
  `783 bis`, `360.9`, etc.).  These are sanitized to blank (the bootstrap
  import core treats blank as NULL, which is valid).  The full original value
  is preserved in the legacy source file.
- **PAI coordinates**: a handful of preserved-tree rows in
  `piante-accrescimento-indefinito.csv` have empty `Lon` or `Lat` (noted in
  their `Note` column).  These rows are skipped silently; they cannot be
  loaded without coordinates.

## Validation

`test/test_ingest_roundtrip.py` runs the converter on the real legacy data and
asserts `bootstrap --check` validates it clean against an empty DB (plus sanity
counts).  Run from the abies root:

```sh
env DJANGO_DEBUG=1 DJANGO_SECRET_KEY=django-insecure-local-dev \
    python3 -m pytest test/test_ingest_roundtrip.py -v
```
