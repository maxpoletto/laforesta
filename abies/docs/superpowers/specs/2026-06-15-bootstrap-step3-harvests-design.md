# Bootstrap step 3 — harvests, plan items, preserved trees (design)

> **Status (2026-06-15):** approved; ready for writing-plans. Branch `dev/import`.
> Extends the master spec
> [`2026-06-14-bootstrap-import-contract-design.md`](2026-06-14-bootstrap-import-contract-design.md)
> — §5 (file contract), §6 (new harvest format + region-wide rows), §13.3
> (sequencing). This increment implements §13.3 **plus** the two adjacent optional
> cores (plan items, preserved trees) so that `make dev` populates Prelievi, Piano
> di taglio, and the Bosco PAI overlay for a near-term demo.

## 1. Goal & scope

Cut the last legacy data over to the canonical path: load **harvests**, **harvest
plan items**, **preserved trees**, and **tractors** through `bootstrap`, and emit
them from the La Foresta converter — so a single `make dev` yields a fully
populated instance to demo end-to-end.

**In scope**

- `Harvest` and `Tractor` model changes (+ migrations, + XOR trigger).
- Import cores: `harvests.csv` (new), `tractors.csv` (reference), and the
  `harvest_plan_items.csv` and `preserved-trees.csv` loaders.
- `bootstrap` loads all four (removed from the deferred list).
- Converter (`ingest/convert_laforesta.py`) emits all four canonical files.
- Tests (core unit tests, bootstrap, extended converter round-trip) and a
  `make dev` demo smoke.

**Deferred (NOT in this increment)**

- `export <dir>` + the export→bootstrap→export round-trip (master spec step 4).
- `--force` / safe restorable wipe (`make dev` already does `reset-db`).
- The in-app harvest **upload view** (master spec marks it a later feature; only
  the shared import core is built here).
- Removing aliasing / `GENERE_MAP` / `*_LEGACY` / synthetic-`X` (step 5).

## 2. Model changes

**`Harvest`** (`apps/prelievi/models.py`), mirroring `HarvestPlanItem`:

- `parcel` → `null=True, blank=True` (keep `on_delete=PROTECT`).
- add `region = ForeignKey(Region, on_delete=PROTECT, null=True, blank=True,
  related_name='harvests')`.
- enforce **region XOR parcel** two ways: a `clean()` mirror (early form
  feedback) and a new SQLite trigger, added the same way as `HarvestPlanItem`'s
  in `apps/base/migrations/0002_triggers.py` (a new triggers migration).
- `__str__` handles the null-parcel (region-wide) case.

**Downstream effects of a nullable `Harvest.parcel`** — these must ship with the
model change (chunk 1), or region-wide harvests break:

- **Prelievi digest** (`apps/base/digests.py`, harvest row ≈ line 292)
  dereferences `op.parcel.region_id` / `op.parcel.region.name` / `op.parcel.name`
  and crashes when `parcel` is NULL. Make it null-safe: a region-wide row takes
  `region_id`/region name from `op.region`, `parcel_id` is NULL, and `Particella`
  renders as the display mark `X` (`PARCEL_WHOLE_REGION_MARK`, the master spec §6
  convention). Add a digest test for a region-wide harvest row.
- **Existing harvest consistency triggers** (`0002_triggers.py`,
  `harvest_parcel_consistency_*` ≈ line 91) join `base_parcel ON p.id =
  NEW.parcel_id`, assuming a non-null parcel when a harvest links to a plan item.
  Update them to also accept a region-wide harvest (`NEW.parcel_id IS NULL`)
  whose `region_id` matches a region-wide linked item. Bootstrap-loaded harvests
  set `harvest_plan_item = NULL` (so the trigger does not fire for them), but the
  fix removes a latent bug for the future upload view. Add a trigger test.

**`Tractor`** (`apps/base/models.py`):

- add `name` — the unique `Trattore` display key used verbatim in the
  `Trattore: <name>` harvest headers (manufacturer+model is not unique once year
  matters). Declared `unique=True, null=True` so the migration is safe on an
  already-populated table (SQLite permits multiple NULLs); the `tractors.csv`
  core always populates it, and both `make dev` (reset) and a prod re-bootstrap
  start from an empty table, so no data backfill is required.
- **Expose `name` in settings + audit** (decision: editable in the UI, not
  bootstrap-only immutable): add it to the Impostazioni → Tractors form
  (`apps/impostazioni/templates/impostazioni/_tractor_form_it.html` + the
  create/update view in `apps/impostazioni/views.py`, with uniqueness validation)
  and to the tractor audit-digest table (`apps/base/digests.py` ≈ line 447).
  UI-created tractors set `name` directly; bootstrap sets it from `tractors.csv`.

## 3. Import cores

3.1 **`tractors.csv`** → a new `TRACTORS` `RefTable` in `apps/base/csv_reference.py`
(key column `Trattore`→`name`; plus `Produttore`/`Modello`/`Anno`). Reuses the
existing reference engine (idempotent upsert, version/`modified_at` bump). Added
to `ALL_TABLES`.

3.2 **`harvests.csv`** → new core `apps/prelievi/csv_harvests.py`, mirroring the
three-phase shape of `apps/campionamenti/csv_trees.py`:

- `resolve_columns` — static headers + the dynamic `Specie:` / `Trattore:`
  prefixed columns (master spec §6).
- `validate_rows` (pure; injected indexes for parcels, regions, crews, products,
  species-by-name, tractors-by-name):
  - static cols: `Compresa`, `Particella` (blank ⇒ region-wide ⇒ `parcel=None`,
    `region=<Compresa>`), `Data`, `Squadra`, `Tipo`, `Q.li` (mass, primitive),
    `VDP`, `Prot.`, explicit `Danneggiato`/`Fitosanitario`/`PSR` booleans,
    `Altre note`.
  - dynamic cols: `Specie: <Genere>` and `Trattore: <name>` percentages —
    blank ⇒ 0; species %-sum-to-100 when `Q.li > 0`; tractor %-sum-to-100 when
    any tractor is present, else all blank/0. **Header-matching rule:** a column
    matches the `Specie:` / `Trattore:` prefix, then the remaining key is trimmed
    of surrounding whitespace and matched **exactly, case-sensitively** against
    `Species.common_name` / `Tractor.name`. Two headers whose trimmed keys are
    equal are a **duplicate** error; an unmatched key is an **unknown-key** error.
    (This is the master spec §6 "normalized" = trim-only, no case-folding.)
  - `volume_m3` computed from `Q.li` + species densities via the existing
    `apps.prelievi.models.harvest_volume_m3` (no authoritative volume column).
- `apply` — bulk-create `Harvest` + `HarvestSpecies` + `HarvestTractor`.

3.3 **`harvest_plan_items.csv`** (unified fustaia+ceduo, explicit boolean flag
columns) → a thin bootstrap-side loader that **reuses `apps/piano_di_taglio/
csv_plan.py::apply`**. The canonical file carries a `Piano` column (master spec
§5), and `apply` operates on **one** target plan at a time, so the loader first
**groups rows by `Piano`**, resolves each existing `HarvestPlan` (an unknown
`Piano` is an error), and calls `apply(target_plan=<plan>, …)` once per group —
mirroring how `bootstrap` groups `sample_areas` by `Griglia` and `sampled-trees`
by `Rilevamento`. `apply` already consumes parsed dicts and owns the proven
HarvestDetail / ParcelPlanDetail / dedup / region-wide write logic, so within each
group the loader builds the fustaia-/ceduo-shaped parsed dicts directly: it splits
each row by `parcel.eclass.coppice` (highforest → fustaia dict with `Prelievo
(m³)`; coppice → ceduo dict with `Superficie intervento (ha)` + `Turno (a)`),
routes blank-`Particella` region-wide rows through the fustaia/region branch, and
reads the explicit `Danneggiato`/`Fitosanitario`/`PSR` boolean columns (not the
legacy `Note`-string). The in-app view's existing `parse_fustaia_rows`/
`parse_ceduo_rows` (legacy `Note`-string + `X` marker) are left untouched.
*(Rejected alternative: a standalone items core — it would duplicate `apply`.)*

3.4 **`preserved-trees.csv`** → new small core `apps/campionamenti/csv_preserved.py`
(grouped with `csv_trees`, the other `Tree`-creating core): `Tree(preserved=True,
coppice=False)` with strict canonical `Genere`→`Species` lookup and `Lon`/`Lat`.
No `Sample`/`TreeSample`, so much simpler than `csv_trees`.

## 4. `bootstrap` wiring (`apps/base/management/commands/bootstrap.py`)

- Add `TRACTORS` to the reference loop (between crews and the bulk loaders).
- Load, in dependency order after parcels/regions/crews/products/species/
  tractors/plans: `harvest_plan_items` (needs plans + parcels), `preserved-trees`
  (needs parcels + species), `harvests` (needs parcels/regions/crews/products/
  species/tractors).
- Remove `CSV_FILE_PRESERVED_TREES`, `CSV_FILE_HARVEST_PLAN_ITEMS`,
  `CSV_FILE_HARVESTS` from `DEFERRED_FILES`. No `GUARD_MODELS` change needed: the
  relevant models (`Tractor`, `Tree`, `HarvestPlanItem`, `Harvest`) are already
  guarded (their children `HarvestSpecies`/`HarvestTractor` are covered by
  guarding `Harvest`).

## 5. Converter (`ingest/convert_laforesta.py`)

Emit four more canonical files (stdlib only, standalone):

- `tractors.csv` — the five hard-coded La Foresta tractors (`Trattore`,
  `Produttore`, `Modello`, `Anno`).
- `harvests.csv` — from `mannesi.csv`: `Tipo` via the canonical product map
  (`refdata.PRODUCT_MAP` values), `Particella='X'` → blank (region-wide),
  `abete %…` → `Specie: <canonical Genere>`, `Equus %…` → `Trattore: <name>`,
  the `Note` flag-string → explicit `Danneggiato`/`Fitosanitario`/`PSR` booleans,
  `Q.li`/`VDP`/`Prot.`/`Data`/`Squadra`/`Altre note` carried through.
- `harvest_plan_items.csv` — unified from `piano_fustaia.csv` + `piano_ceduo.csv`
  (explicit boolean flags; blank `Particella` for `X` region-wide rows).
- `preserved-trees.csv` — from `piante-accrescimento-indefinito.csv` (`Genere`
  mapped to canonical, `Lon`/`Lat`).

Legacy→canonical column/value maps are duplicated as literals in the converter
(it stays dependency-free), consistent with the existing `CANONICAL_PRODUCTS`
pattern.

## 6. Constants (`config/strings_it.py`)

Add: `CSV_FILE_TRACTORS`; dedicated explicit-boolean column headers
`CSV_COL_DAMAGED = 'Danneggiato'`, `CSV_COL_UNHEALTHY = 'Fitosanitario'`,
`CSV_COL_PSR = 'PSR'` (the master spec §6 header names), shared by `harvests.csv`
and `harvest_plan_items.csv`. These are **not** the existing `FLAG_DAMAGED/
UNHEALTHY/PSR` — those are the `Note`-string rendering labels
(`FLAG_DAMAGED = 'Catastrofato'`), a different concept; do not conflate them.
Also add the `Specie:` / `Trattore:` localized prefixes and the `tractors.csv`
column headers (`Trattore`, `Produttore`, `Modello`, `Anno`). Reuse existing
`CSV_COL_REGION/PARCEL/DATA/CREW/PRODUCT/NOTE/QUINTALS/VDP/PROT/EXTRA_NOTE`,
`CSV_FILE_HARVESTS/HARVEST_PLAN_ITEMS/PRESERVED_TREES`, and
`PARCEL_WHOLE_REGION_MARK`.

## 7. Testing

- Core unit tests: `csv_harvests` (static + dynamic columns, the header-matching
  rule incl. unknown-key and duplicate-key errors, region-wide rows,
  species/tractor sum-to-100, computed volume); `tractors` RefTable;
  `preserved-trees`; the unified `harvest_plan_items` loader (grouping by `Piano`
  incl. unknown-`Piano` error, highforest/coppice split, region-wide, explicit
  booleans).
- Digest test: a region-wide harvest yields a valid prelievi digest row (region
  name + `Particella = X`, no crash).
- Trigger/model tests: Harvest region-XOR-parcel (new trigger + `clean()`); the
  updated `harvest_parcel_consistency` trigger accepts a region-wide harvest
  linked to a region-wide plan item.
- Settings test: `Tractor.name` is editable in the Impostazioni form and its
  uniqueness is enforced.
- `bootstrap` tests: loads all four; region-wide harvest; refusal of a harvest
  row with both/neither parcel and region.
- Extend `test/test_ingest_roundtrip.py`: `bootstrap --check` clean on the
  now-complete canonical set, with sanity counts (≈11941 harvests incl. 1468
  region-wide; plan items; preserved trees; 5 tractors).

## 8. Demo verification

`make dev` → Prelievi (~11941 harvests), Piano di taglio (plan + items), Bosco
PAI overlay (preserved trees), all populated; manual smoke of those pages.

## 9. Implementation chunks (for writing-plans)

Each shippable with `make test` green:

1. **`Harvest` model + downstream** — nullable parcel + region FK + `clean()` +
   region-XOR-parcel trigger; update the `harvest_parcel_consistency` triggers for
   a NULL parcel; null-safe prelievi digest. Tests: XOR, region-wide digest row,
   trigger consistency.
2. **`Tractor.name`** — model field + migration; expose in the Impostazioni
   tractor form/view (uniqueness) and the tractor audit digest. Tests: editable +
   unique.
3. **`harvests.csv` core** (`csv_harvests`) + unit tests.
4. **`tractors` RefTable + `preserved-trees` core + `harvest_plan_items` loader**
   (group by `Piano`, reuse `csv_plan.apply`) + unit tests.
5. **`bootstrap` wiring** — load the four in dependency order; drop from
   `DEFERRED_FILES`; bootstrap tests.
6. **Converter outputs** — `tractors`/`harvests`/`harvest_plan_items`/
   `preserved-trees`; extend the round-trip test.
7. **Demo smoke** — `make dev`, verify Prelievi / Piano di taglio / Bosco PAI.
