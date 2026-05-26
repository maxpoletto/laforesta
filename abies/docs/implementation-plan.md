# Implementation plan

State for the build sequence that takes Abies from Stage 1 (Prelievi MVP,
already shipped) through to a full version-1 of the app.  Page-level UX,
schema, URL parameters, and digest formats are settled in the per-page docs
in `docs/page-*.md` and in `docs/database.md`; this file is just the
*ordering* of the work.

## Principles

- **Demo soon.**  Every milestone after M0 must land a user-visible
  feature.  No long stretches of pure plumbing.
- **Cluster schema work at phase boundaries.**  Migrations are the heaviest
  cost; no mid-phase migrations.
- **Don't break Prelievi.**  Stage 1 is in users' hands; every milestone
  ships green.
- **Reimport over migrate** while we still can.  No harvest data has been
  UI-entered, so M0's rename is truncate-and-reimport, not an in-place
  data migration.

## Milestone order

```
M0  Foundation               (no new tab visible)
M3  Campionamenti            ← first new-feature demo
M4  Bosco (six sub-phases)   ← progressive, biggest UX impact
M1  Piano di taglio: Calendario
M2  Piano di taglio: Marks   ← lights up the Operazioni-recenti gap on Bosco
```

The numbering keeps the original "topological" labels for cross-reference;
the *order* of execution is what matters.  M0 → M3 → M4 → M1 → M2.

Rationale for putting Piano di taglio last despite the original M1/M2
naming: there is no schema-level dependency from Bosco or Campionamenti
on Piano di taglio (`survey.harvest_plan_id` is nullable; `tree` is added
in M3 and referenced by M2's `tree_mark`).  Bosco's per-parcel "Operazioni
recenti → ultime martellate" sub-list is the only thing that has to wait
— it ships dark and lights up automatically once M2 lands.  In exchange,
Bosco becomes demo-able far earlier than under the M1-first order.

## M0 — Foundation

No new tab; pure plumbing so the new pages have ground to stand on.

**Schema**
- Rename `HarvestOp` → `Harvest`, `Optype` → `Product` (drop tables and
  recreate; truncate harvest data; reimport from `mannesi.csv`).
- Add `species.density:real` (q/m³).
- Add `harvest.volume_m3:real`, computed at write time as
  `SUM_over_species(quintals × percent/100 / species.density)`.
- Drop `parcel.harvest_plan_id` (no longer per-parcel; plan items handle
  the relationship).

**Digests**
- Update `prelievi.json` — column rename (Tipo: optype→product), add
  `Volume (m³)` materialized companion column.
- Update `parcel_year_production.json` — both q.li and m³ totals.
- Update `parcels.json`, `crews.json`, `audit.json` for the rename.
- New: `species.json` (id, common_name, latin_name, density, sort_order,
  active).

**Frontend**
- Tab order in shell template: Bosco · Piano di taglio · Campionamenti ·
  Prelievi · Controllo · Impostazioni.  Empty placeholder tabs OK for the
  ones not yet built.
- Prelievi UI: rename Tipo label, add small `Volume (m³)` companion
  column next to `Q.li`.

**Exit criteria**
- All existing Prelievi tests pass.
- `make digest && make import` reproduces a clean dataset from CSVs.
- Audit page renders correctly with the renamed model history.

## M3 — Campionamenti

First user-visible new tab.  Brings the heaviest schema additions; later
milestones reuse them.

**Schema**
- Add `sample_grid` (id, name, description), `survey` (id, name,
  harvest_plan_id nullable, sample_grid_id, description), `sample` (id,
  sample_area_id, survey_id, date), `tree` (id, species_id, year, lat
  nullable, lng nullable, parcel_id, preserved, coppice), `tree_sample`
  (id synthetic + UNIQUE on (sample_id, tree_id, shoot); fields per
  `database.md`).
- Reshape `sample_area`: drop `plan_year`; add `sample_grid_id` FK,
  `r_m`, `note`.
- Drop `preserved_tree`; ETL existing PAI CSV → `tree` rows with
  `preserved=true`.
- Trigger: `sample.sample_area.sample_grid_id ==
  sample.survey.sample_grid_id`.

**Tabacchi**
- New JS module `apps/base/static/base/js/volume.js` — Tabacchi
  parameters + `volume(d, h, species)` + `mass(volume, density)`.
- Python equivalent on the server side (used only by the CSV import
  path).  Both sides reference `pdg-2026/pdg/computation.py` as
  canonical source.

**Digests**
- New: `grids.json`, `surveys.json`, `sample_areas.json`, `samples.json`.
- New: `sampled_trees_<survey_id>.json` — lazy per survey.

**Frontend**
- Three sections of the Campionamenti page (Griglie map-centric,
  Rilevamenti map-centric, Alberi campionati table).
- Grid CSV import + Tree-and-sample CSV import (Impostazioni-adjacent
  flows, but invoked from each section's "Nuovo" modal).
- Manual tree+sample entry form with V/m live preview.
- ETL: load existing `aree-di-saggio.csv` and `alberi-calcolati.csv`
  for demo data on day one.

**Exit criteria**
- All three sections functional with imported demo data.
- Manual entry round-trips through the optimistic-locking and digest
  staleness flow.
- JS Tabacchi vs Python Tabacchi parity confirmed against a fixture of
  ~20 mixed-species trees.

## M4 — Bosco

Six independently shippable sub-phases.  The first sub-phase
(`M4a`) makes Bosco the landing page (replacing Prelievi as default).

- **M4a — Map + Caratteristiche mode**.  Port from `bosco/b`.  No
  overlay yet.  `parcels.json` already exists; add satellite digests
  (port from Boscoscopio).
- **M4b — Per-parcel/per-region overlay: Metadati + Operazioni
  recenti**.  Cheapest two sections.  Reads existing digests; cross-page
  links light up.  "Ultime martellate" sub-list ships *dark* (waiting
  on M2).
- **M4c — Dendrometria**.  New digest: `parcel_dendrometry.json`.
  Survey selector wired to actual data (active-survey rule per
  `bosco.md`).
- **M4d — Produzione storica**.  Reuses `prelievi.json`; three
  breakdowns (specie / prodotto / squadra) derived client-side.
- **M4e — Evoluzione mode**.  Port from Boscoscopio.
- **M4f — PAI mode**.  New digest: `preserved_trees.json`.  Add-PAI
  form + edit/delete affordances.

**Exit criteria** (per sub-phase): each is independently shippable; no
sub-phase blocks others.

## M1 — Piano di taglio: Calendario

Smallest possible Piano-di-taglio surface.  No marks yet; chip moves
between *pianificato* / *raccolto (parz.)* / *raccolto* directly.

**Schema**
- `HarvestPlanItem.quintals:int` → `volume_m3:real nullable`; add
  `intervention_area_ha:real nullable`, `note:string nullable`.

**Plan import** (admin-only, Impostazioni)
- Parses `piano.csv` (fustaia) + `ceduo.csv` (coppice) +
  `equazioni_ipsometro.csv` (stored but unused this milestone).
- Transactional with overwrite confirmation.  Decimal-comma handling
  at parse time.
- Column-name constants in Python module so future drift is a one-line
  edit.

**Digests**
- New: `harvest_plans.json`, `harvest_plan_items.json`.

**Frontend**
- Piano di taglio tab gets Section 1 (Calendario) only.  Sections 2/3
  not yet in DOM.
- Inline expansion shows "no mark yet" placeholder for fustaia items.
- Cross-page link from chip → Prelievi (`?c=N&pa=N&y1=…&y2=…`).
- Particella cells link to Bosco per-parcel overlay (already exists
  by this point).

**Exit criteria**
- Plan import round-trips with `pdg-2026`'s actual output.
- Status chip computes correctly for fustaia and coppice without marks.

## M2 — Piano di taglio: Marks

The richer half.  Wires up everything for the full plan→mark→harvest
trace.

**Schema**
- Add `mark` (id, parcel_id, date, harvest_plan_item_id, note),
  `tree_mark` (id synthetic + UNIQUE on (mark_id, tree_id);
  d_cm, h_m, h_measured, volume_m3, mass_q),
  `tree_height_regression` (composite PK).
- `mark.harvest_plan_item_id` is `ON DELETE PROTECT`.
- Trigger: `mark.parcel = harvest.parcel` when `harvest.mark_id` is
  set.

**Plan import** activates the regression CSV path.

**Digests**
- New: `marks.json`, `mark_trees.json` (all-marks-in-one for now;
  per-mark split is future work), `tree_height_regressions.json`.

**Frontend**
- JS regression-lookup module: `h_for(d, plan_id, region_id,
  species_id)`.
- Sections 2 & 3 of Piano di taglio.
- New-mark form, new-marked-tree form (regression auto-fill + Tabacchi
  V/m preview, reusing `volume.js` from M3).
- Status chip gains *martellato* state + 0.9× ⚠ discrepancy flag.
  Calendar inline expansion shows mark detail.
- Cascade-delete UI (warning + forced CSV export) for mark deletion.
- Bosco's "ultime martellate" sub-list of Operazioni recenti lights up.

**Exit criteria**
- Full plan→mark→harvest loop demonstrable end-to-end.
- Plan re-import correctly blocks on existing marks (PROTECT).

## Cross-cutting (deferrable, do once)

- Mobile portrait QA pass per page (one shared pass near the end is
  fine).
- `mark_trees.json` per-mark split if size warrants (annotated in
  `piano-di-taglio.md`).
- `harvest_detail` UI in Impostazioni (the import doesn't populate
  Istruzioni; manual editing is a Settings sub-section eventually).
- Audit page coverage for new tables (django-simple-history is mostly
  automatic; just register the new models).

## Status

Tracked via TaskCreate / TaskUpdate; initial tasks 1–5 correspond to
M0, M3, M4, M1, M2 with `blockedBy` set to enforce the order above.
