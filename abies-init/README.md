# abies-init — legacy → canonical initialization

One-off ETL that converts La Foresta's **legacy** survey/plan CSVs into the
**canonical abies CSV inputs** that `manage.py bootstrap` consumes.  This is
"sub-project 2" of the import-decoupling effort: it decouples the messy source
data from the strict canonical contract enforced by `bootstrap`.

## Usage

```sh
python3 convert_laforesta.py <src_dir> <out_dir> [branding_out_dir]
```

- `src_dir` — the legacy data dir (e.g. `../abies-data`); see its `README`.
- `out_dir` — written with the canonical files below, then loadable with
  `manage.py bootstrap <out_dir> [--check]`.
- `branding_out_dir` — optional; when set, receives `logo.png`,
  `favicon.gif`, and `ipso-logo.gif` from this repository's `branding/`
  directory.  The Abies
  checkout expects these under `data/branding` before build/deploy.

Dialect: reads `utf-8-sig` (BOM-tolerant), writes `utf-8`, comma-delimited, dot
decimals (the canonical pairing).  Stdlib `csv` only — no project imports, no new
dependencies — so the tool runs standalone against a legacy-data checkout.

## Makefile

The default target writes both Abies bootstrap data and branding inputs:

```sh
make
```

By default this writes canonical CSVs to `../abies/data/canonical` and branding
assets to `../abies/data/branding`. Override `DST_DIR` or `BRANDING_DST_DIR` if
needed.

## Output files (source → canonical mapping)

| Output | Source | Mapping |
|---|---|---|
| `regions.csv` (`Compresa`) | `particelle.csv` | distinct `Compresa` |
| `eclasses.csv` (`Comparto,Ceduo`) | `particelle.csv` | distinct `Comparto`; `Ceduo=1` iff `Comparto=='F'`, else `0` |
| `crews.csv` (`Squadra,Attivo`) | `mannesi.csv` plus constant | distinct non-blank `Squadra`; `Attivo=true` only for crews with 2026 rows, plus `Zaffino-Santaguida`; all other legacy crews are inactive |
| `species.csv` (`Genere,Nome latino,Densità (q/m³),Pressler,Minore,Ordine`) | in-repo `apps/base/data/species.csv` | reshape `common/latin/density_q_m3/pressler_default/minor/sort_order` to the canonical localized headers; omit `Pino Laricio` because legacy generic pine aliases are flattened to `Pino Nero` |
| `products.csv` (`Tipo`) | constant | the distinct canonical products of `apps.base.refdata.PRODUCT_MAP.values()` |
| `particelle.csv` | `particelle.csv` | select/reorder `Compresa,Comparto,Particella,Area (ha),Età media,Località,Altitudine min,Altitudine max,Esposizione,Pendenza %,Stazione,Soprassuolo`; drop CP, Governo, Piano del taglio, Parametro, Matricine |
| `sample_grids.csv` (`Griglia`) | constant | one row: `Aree di saggio PDG 2026` |
| `harvest_plans.csv` (`Piano,Anno inizio,Anno fine`) | `piano_fustaia.csv` + `piano_ceduo.csv` | one row `PDG 2026`, `Anno inizio`/`Anno fine` = min/max `Anno` across both files |
| `sample_areas.csv` (`Griglia,Compresa,Particella,Area saggio,Lon,Lat,Quota,Raggio`) | `aree-di-saggio.csv` | add the constant `Griglia`; select Lon/Lat/Quota/Raggio; drop CP/UTM\*/Gauss\* |
| `surveys.csv` (`Rilevamento,Griglia,Data`) | constant | two rows with `Griglia=Aree di saggio PDG 2026`: `Campionamento PDG Sabatino` dated `2022-08-01`, and `Campionamento altezze Luca` dated `2025-12-01` |
| `sampled-trees.csv` | `alberi-calcolati.csv` + `alberi-altezze.csv` | union of the two surveys (see below) |
| `hypso_params.csv` | `equazioni_ipsometro.csv` | copied verbatim (lowercase headers accepted case-insensitively by `hypsometry.parse_param_csv`) |
| `tractors.csv` (`Trattore,Produttore,Modello,Anno`) | constant | the six hard-coded La Foresta tractors, including `Scania P380` |
| `harvests.csv` | `mannesi.csv` | `Tipo` via `_PRODUCT_MAP`; `Particella='X'` → blank; species `abete %` etc. → `Specie: <canonical>` dynamic cols, with `pino %` mapped to `Pino Nero`; `Equus %` etc. → `Trattore: <name>` dynamic cols; `Note` token → `Danneggiato/Fitosanitario/PSR` booleans; invalid VDP values (``nd``, ``bis`` variants, fractional) → blank |
| `harvest_plan_items.csv` | `piano_fustaia.csv` + `piano_ceduo.csv` | unified rows with `Piano=PDG 2026`; `Particella='X'` → blank; highforest flags from `Note` token; coppice flags left `false` (no flag column in legacy) |
| `preserved-trees.csv` (`Compresa,Particella,Numero,Genere,Data,Anno di nascita stimato,D_cm,H_m,H_measured,Lon,Lat,Acc_m,Operatore,Note`) | `piante-accrescimento-indefinito.csv` | `Genere` preserved when it already names a canonical species; legacy `Abete Bianco` aliases to `Abete`; `Pino`, `Pino Laricio`, and `Pino Nero` flatten to `Pino Nero`; `Pino Marittimo` and `Pino Strobo` stay distinct; `Numero` regenerated from 1 within each parcel because the source column is unreliable; date/birth year/accuracy/operator left blank; rows missing `Lon`/`Lat` silently skipped |
| `marks/*.csv` (`Data,Compresa,Particella,Catastrofata,Numero,Genere,D_cm,H_m,H_measured,Lat,Lon,Acc_m,Operatore`) | `martellate/*.csv` | upload-ready mark files for Piano di taglio; legacy `Specie` header becomes `Genere`, `Lng` becomes `Lon`, decimal commas become dot decimals, and `Pino` aliases to `Pino Nero` |
| `terreni.geojson` | `terreni.geojson` | copied verbatim (bootstrap does not load it, but the canonical dir carries the source geometry) |

## Tree-survey assumptions (the interesting part)

`sampled-trees.csv` is the **union** of two surveys, mapped to the canonical
`Rilevamento,Compresa,Particella,Area saggio,Albero,Pollone,Matricina,D_cm,H_m,L10_mm,Pressler,Genere,Fustaia`:

1. **`Campionamento PDG Sabatino`** (from `alberi-calcolati.csv`): `n`→`Albero`,
   `poll`→`Pollone`, `D(cm)`→`D_cm`, `h(m)`→`H_m`, `L10(mm)`→`L10_mm`,
   `Pressler` is set to `2`, plus `Genere`, `Fustaia`.
   - **`poll == 'mat'` sentinel:** in the legacy file the `poll` column is
     normally a shoot number, but the literal `mat` marks the tree as a coppice
     **standard** (matricina), not a numbered shoot.  Canonically that is the
     dedicated `Matricina` boolean, so a `mat` row maps to `Matricina=True` with
     a blank `Pollone` (the strict core defaults blank Pollone → 0).  All 87
     `mat` rows are coppice (`Fustaia=FALSE`), consistent with this reading.
     Non-`mat` rows leave `Matricina` blank (→ False).

2. **`Campionamento altezze Luca`** (from `alberi-altezze.csv`): `D(cm)`→`D_cm`,
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

`Genere` values are normalized before writing canonical CSVs: `Pino`,
`Pino Laricio`, and `Pino Nero` all become `Pino Nero`; `Pino Marittimo` and
`Pino Strobo` remain distinct.  Other names are passed through as-is.

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

`test/test_roundtrip.py` runs the converter on the real legacy data and
asserts `bootstrap --check` validates it clean against an empty DB (plus sanity
counts).  Run from the abies-init root:

```sh
python3 -m pytest -v
```
