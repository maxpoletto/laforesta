# Piano di taglio — fine-grained implementation plan

This file is the authoritative implementation order for the Piano di taglio
domain. It supersedes the M1 ("Calendario") and M2 ("Marks") subsections in
[`implementation-plan.md`](implementation-plan.md); the rest of that
document (M0, M3, M4) is unchanged and remains the source of truth for
those milestones.

The split exists because the redesign that landed in 2026-05 fused the
schema and Prelievi-coupling work that used to be split across M1 and M2,
and replaced the three-section page UI with a two-calendar + bookmarkable
modal layout. The PT-NN increments below are sized to be ~100–250 lines
of diff each, thematically self-contained, and independently reviewable.

> **Context for downstream work.** No data is in production yet during
> this phase. Dev DB can be blown away freely; schema changes do not
> require data migrations. The user will explicitly flag when this
> changes.

## Conventions

- Each item is intended to be a single commit. Tests ride with the change
  that introduces the behaviour they cover.
- Order is the suggested execution order. Many items can run in parallel
  after the schema stabilises (end of Phase 1).
- Italian terms are used in URLs and user-visible labels; English in
  Python identifiers and JS variable names (see `abies/CLAUDE.md`
  "Internationalization").
- Reference docs: [`database.md`](database.md),
  [`pages/piano-di-taglio.md`](pages/piano-di-taglio.md),
  [`pages/prelievi.md`](pages/prelievi.md), [`pages/campionamenti.md`](pages/campionamenti.md).

## Phase 0 — Spec hygiene

- **PT-00** Edit `database.md`, `pages/piano-di-taglio.md`,
  `pages/prelievi.md`, and `abies/CLAUDE.md` to reflect the agreed
  A1–A10 / B1–B8 decisions from the redesign review. No code, no
  migrations.
- **PT-01** Create this file (`docs/piano-di-taglio-plan.md`) and
  reference it from `abies/CLAUDE.md`. No code.

## Phase 1 — Schema

All schema changes are migration-only — no data preservation needed
(see context note above). Each item is one migration plus the matching
model edits. SQLite-level triggers land in the last item of the phase.

- **PT-10** `apps/base/models.py`: `HarvestPlan` add
  `name:CharField(unique=True)`. (Existing `description` is preserved
  for the longer text.) Migration.
- **PT-11** `apps/base/models.py`: rewrite `HarvestPlanItem` to the new
  shape — `region` and `parcel` both nullable (XOR), `state:int` with
  `STATE_*` constants, `year_planned`, `date_actual` nullable,
  materialized `volume_planned_m3 / volume_marked_m3 / volume_actual_m3`,
  `intervention_area_ha`, `damaged / unhealthy / psr` booleans, `note`.
  Drops the old `quintals` column. Includes a Python-level monotonic-
  state validator on save. Migration.
- **PT-12** `apps/base/models.py`: new `HarvestTransition` model with
  `harvest_plan_item:FK PROTECT`, `open:bool`, `date`, `note`. Migration.
- **PT-13** `apps/base/models.py`: new `TreeHeightRegression` model
  (composite unique constraint on `(harvest_plan, region, species)`).
  Migration.
- **PT-14** `apps/base/models.py`: new `TreeMark` model —
  `harvest_plan_item:FK PROTECT`, `tree:FK`, `date`, `d_cm`, `h_m`,
  `h_measured`, `volume_m3`, `mass_q`, `lat`, `lon`, `acc_m` nullable,
  `operator`, `import_fingerprint` (the row-content hash used for A7
  idempotency). `UNIQUE(harvest_plan_item, tree)`. Migration.
- **PT-15** `apps/base/models.py`: `Tree` — rename `lng → lon`, add
  `acc_m:int nullable`. `SampleArea` — rename `lng → lon`. Migration
  + grep-fix any code referencing `.lng`.
- **PT-16** `apps/prelievi/models.py`: `Harvest` — add
  `harvest_plan_item:FK PROTECT nullable` (nullable for historical
  rows; new harvests carry it via Django-level validation, not a
  schema constraint), add `damaged / unhealthy / psr:bool`, rename
  `quintals → mass_q`, rename `extra_note → note`, drop `note:FK`.
  Migration.
- **PT-17** Drop the `Note` model and its migrations once nothing
  references it. Migration.
- **PT-18** SQLite triggers in a dedicated migration: (a) enforce
  `harvest_plan_item.region_id XOR parcel_id`; (b) when
  `harvest.harvest_plan_item_id` is non-null, the linked item's parcel
  must match `harvest.parcel_id` (or its region must match
  `harvest.parcel.region_id` for region-wide items); (c) when
  `harvest.harvest_plan_item_id` is non-null,
  `harvest.{damaged,unhealthy,psr} == harvest_plan_item.{...}`. Land
  the triggers with Python-level checks that mirror them for
  app-side error messages. Tests for both happy and trigger-rejection
  paths.

## Phase 2 — Constants & strings

- **PT-20** `config/strings_it.py` and
  `apps/base/static/base/js/strings_it.js`: add `STATE_*` Italian
  labels, boolean-flag labels (`CATASTROFATO`, `FITOSANITARIO`, `PSR`),
  the rendered comma-joined-flags string format helper, and the CSV
  column-name constants (`CSV_COL_COMPRESA`, `CSV_COL_PARTICELLA`,
  `CSV_COL_ANNO`, `CSV_COL_PRELIEVO_M3`, `CSV_COL_SUP_INTERVENTO_HA`,
  `CSV_COL_TURNO_A`, `CSV_COL_NOTE`, plus the lowercase regression
  CSV columns) used by the import paths.

## Phase 3 — Digest generators

No UI consumers yet, but tests in place.

- **PT-30** `apps/base/digests.py`: `generate_harvest_plans` +
  `build_harvest_plan_record`. Stale on `HarvestPlan` write.
- **PT-31** `apps/base/digests.py`: `generate_harvest_plan_items` +
  `build_harvest_plan_item_record`. Stale on `HarvestPlanItem`,
  `TreeMark`, `Harvest`, `HarvestSpecies`, and `HarvestTransition`
  writes. Includes the computed flag-string per A5 and `Tipo`
  derived from `parcel.eclass.coppice` (region-wide items derive
  `Tipo` from their region's predominant eclass family, falling back
  to `"misto"` if mixed; row lands in the fustaia section in that
  case).
- **PT-32** `apps/base/digests.py`: `generate_tree_height_regressions`.
  Stale on `TreeHeightRegression` writes (essentially: plan import).
- **PT-33** `apps/base/digests.py`: per-item lazy
  `generate_mark_trees_<id>` (pattern from
  `sampled_trees_<survey>`). Stale on `TreeMark` writes that touch
  that item.
- **PT-34** `apps/base/digests.py`: update `generate_prelievi` and
  `generate_parcel_year_production` for the new harvest schema —
  Volume column carries `harvest.volume_m3`; Cantiere column carries
  `harvest.harvest_plan_item_id`; Note column carries the
  flag-string from the harvest's own booleans (not the plan item's,
  for historical rows whose link is null).
- **PT-35** Lock the invalidation chain with `TestDigestInvalidation`
  cases for every (write, digest) pair in the materialization table
  documented in `database.md` "Harvest plan item materialization".

## Phase 4 — Backend endpoints

- **PT-40** New `apps/piano_di_taglio/` skeleton (`apps.py`,
  `urls.py`, empty `views.py`, `templates/piano_di_taglio/`,
  `static/piano_di_taglio/`). Register in `INSTALLED_APPS` and
  `config/urls.py`. The path `/piano-di-taglio` resolves to the SPA
  shell with the new domain registered. No behaviour, just plumbing.
- **PT-41** Plan CRUD: `plan_data_view`, `plan_form_view`,
  `plan_save_view`, `plan_delete_view`. The delete view validates
  the plan-level dangerous-delete gate (every item in `state =
  planned`).
- **PT-42** Plan CSV import: `plan_csv_import_view` (one endpoint;
  dispatches on which CSV file is uploaded — fustaia / ceduo /
  regression). Resolves region+parcel and region+species lookups,
  creates rows transactionally. Reuses the CSV column constants
  from PT-20.
- **PT-43** Plan-level Esporta CSV: `plan_export_view` returning a
  zip of `piano.csv + ceduo.csv + equazioni_ipsometro.csv` (B4
  round-trip format).
- **PT-44** Item CRUD: `item_data_view` (modal metadata),
  `item_form_view`, `item_save_view`, `item_delete_view` (gated
  `state = planned` at app level; DB-level PROTECT is the backstop).
- **PT-45** Per-item Esporta CSV: `item_export_view` returning a zip
  with `martellate_<id>.csv` and `prelievi_<id>.csv`. Reused as the
  forced-download step before per-item deletion.
- **PT-46** Transition save: `transition_save_view` — Apri/Chiudi
  cantiere. Validates the monotonic state machine server-side
  (`planned/marked → open` or `open/harvesting → closed`), creates a
  `HarvestTransition` row, advances state.
- **PT-47** `apps/prelievi/views.py`: rewrite form view + save view —
  Cantiere pulldown query (items in state `open` or `harvesting`),
  auto-populate `damaged / unhealthy / psr` from the chosen item per
  B8, mandatory `harvest_plan_item_id` on new harvest (Django-level
  validation), state auto-advance to `harvesting` on first linked
  harvest insert.

## Phase 5 — Frontend: plan + calendars (no item modal yet)

- **PT-50** `static/piano_di_taglio/js/piano-di-taglio.js` skeleton —
  `PianoDiTaglioPage` class with `mount/unmount/onQueryChange`,
  registered in `app.js` route table. Renders the plan selector
  header (plan pulldown + pencil + trash + Esporta CSV + `+ Nuovo
  piano`) and the plan description below.
- **PT-51** Nuovo piano modal — "Crea vuoto" tab. Posts to PT-41's
  endpoints.
- **PT-52** Nuovo piano modal — "Importa calendario da CSV" tab
  with the checkbox-driven fustaia / ceduo choice. Posts to PT-42.
- **PT-53** Nuovo piano modal — "Importa equazioni da CSV" tab.
  Posts to PT-42.
- **PT-54** Edit-plan modal (pencil) — name + description fields.
  Posts to PT-41.
- **PT-55** Plan dangerous-delete flow — shared confirmation-modal
  template parameterised by object type (will be reused by per-item
  delete in PT-59 and by future grid/sample deletes if not already).
  Uses PT-43 as the forced-download step.
- **PT-56** Calendario fustaia section: sortable-table render,
  per-section search box, per-table CSV export (sortable-table-level,
  separate from the plan-level export). No action affordances yet.
- **PT-57** Add-harvest-plan-item modal (Nuovo intervento) for
  fustaia. Posts to PT-44.
- **PT-58** Calendario ceduo section: same shape as fustaia (A3 —
  has `+ Aggiungi`, looking-glass, trash icons; same modals as
  fustaia, dispatched on `Tipo`). Includes the coppice-specific
  columns (`Superficie intervento (ha)`, `Superficie totale (ha)`,
  `Turno (a)`).
- **PT-59** Per-item trash icon → dangerous-delete flow (PT-55
  template). Trash is disabled with tooltip unless `state =
  planned`.

## Phase 5R — UX retrofit (post-Phase-5 review)

After Phase 5 landed, a review surfaced four issues to fix before Phase 6
opens up the per-item modal:

1. **+Nuovo piano confuses identity and content.** Picking a name while
   simultaneously choosing an import flavor (fustaia / ceduo) triggered
   spurious "duplicate name" errors when the operator flipped between
   tabs to fix a wrong checkbox. Fix: identity (name + description) is
   set at creation; content (CSV imports) lands later via the pencil
   modal or the empty-state CTA. The same paradigm will propagate to
   griglie + rilevamenti in a later phase (out of scope here).
2. **Add-harvest-plan-item validator rejects parcel-scoped items.** The
   form sends both `region_id` and `parcel_id` (the cascading pickers),
   and the old XOR validator treated that as "both set → error". The
   storage invariant is XOR but the UI contract is "Compresa required,
   Particella optional"; the server normalises the submission rather
   than rejecting it.
3. **Aesthetic polish.** Title should be plural; the trash icon in the
   pdt-header wraps to a new line on narrow layouts; two-word column
   headers (`Anno effettivo`, `Volume previsto`, etc.) hog horizontal
   space because they are forced single-line.
4. **Empty-state discoverability.** With imports no longer in the create
   modal, a freshly created empty plan needs an obvious in-page
   affordance pointing at the import flows; otherwise the operator has
   to know to open the pencil first.

- **PT-5R-0** Aesthetic fixes (CSS-only). Plural title; widen
  `.pdt-header-left` so the trash icon doesn't wrap; allow two-word
  column headers to wrap at the space and centre them; trim
  `min-width`s where they force the wider single-line layout.
- **PT-5R-1** Fix `_parse_item_body` in `apps/piano_di_taglio/views.py`:
  if `parcel_id` is provided, ignore any submitted `region_id` and
  store the row as parcel-scoped; if `parcel_id` is blank, require
  `region_id` + `damaged OR unhealthy`; if both are blank, reject
  with a "Compresa obbligatoria" message (new
  `ERR_PLAN_ITEM_COMPRESA_REQUIRED`). Drop the old XOR error string
  from server use; tests adjust accordingly. Update
  `docs/page-piano-di-taglio.md` validator wording.
- **PT-5R-2** Strip `+Nuovo piano` modal to a single panel (no tabs):
  name + description. Year range is no longer asked at create time —
  the empty plan defaults to `year_start = year_end = current civil
  year`, and it expands implicitly on CSV import (handled in PT-5R-3).
- **PT-5R-3** Pencil modal grows three tabs: `Dettagli` (existing
  name/description/year-range edit), `Importa calendario da CSV`,
  `Importa equazioni da CSV`. The import tabs reuse `submitCsvImport`
  but post to the existing `plan_csv_import_view` with an additional
  `plan_id` field. Server side: when `plan_id` is set, upsert items
  into the existing plan rather than creating it — `(parcel,
  year_planned)` is the dedup key for both fustaia and ceduo rows
  (an existing match is overwritten, a new row is appended). Widen
  `year_start` / `year_end` to cover any new years. Upsert regressions
  on `(region, species)`. Tests cover re-import idempotency and the
  overwrite-with-revised-values path.
- **PT-5R-4** Empty-state CTA in fustaia + ceduo calendar sections:
  when the active plan has zero items of that family, replace the
  table with an inline panel: `[Importa calendario CSV] [Importa
  equazioni CSV (fustaia only)] [+ Aggiungi manualmente]`. The two
  import buttons open the Modifica piano modal pre-focused on the
  matching tab; `+ Aggiungi manualmente` opens the existing
  add-item modal.

Verification rides the existing Phase-5 manual smoke checklist plus
two new cases: (a) re-create a plan with the same name after deleting
the old one (used to fail when import-flavor was conflated with
create); (b) `+ Aggiungi` an intervento with Compresa + Particella +
Anno only (no flag, no volume) — used to fail on the XOR validator.

## Phase 6 — Frontend: view/edit-item modal

- **PT-60** Modal shell + URL routing for `i=N` parameter, with
  bookmarkability and back-button behaviour. Metadata pane
  (region, parcel, planned year, actual year, state, planned
  amount, flag-string, free-text note) + pencil for inline edit.
- **PT-61** Per-item Esporta CSV button in the modal header
  (uses PT-45).
- **PT-62** Apri cantiere flow: small modal + state advance via
  PT-46.
- **PT-63** Chiudi cantiere flow: small modal + state advance via
  PT-46.
- **PT-64** Prelievi section in the modal (read-only listing) —
  reuses `prelievi.json` filtered client-side on
  `harvest_plan_item_id`. Visible only when `state >= open`.

## Phase 7 — Prelievi page changes

- **PT-70** Prelievi form — Cantiere pulldown replacing the
  Compresa / Particella pulldowns (sources from
  `harvest_plan_items.json` filtered client-side on `Stato ∈ {open,
  harvesting}`). Selection mandatory for new harvests.
- **PT-71** Prelievi form — read-only flag-string summary line below
  the form, auto-populated from the selected Cantiere per B8.
- **PT-72** End-to-end smoke: import real `pdg-2026/csv/*.csv`,
  open a cantiere (skipping marking — manual `planned → open`),
  enter a harvest, verify `volume_actual_m3` updates and the
  calendar `Stato` advances to `harvesting`.

## Phase 8 — Marks (the M2 surface)

Gated by Phase 7 being functional end-to-end.

- **PT-80** Martellata section in the view/edit-item modal:
  read-only table rendering `tree_mark` rows via PT-33. Total
  `volume_marked_m3` shown at the top.
- **PT-81** Nuovo albero martellato modal — D / h (with regression
  auto-fill on Specie + D from `tree_height_regressions.json`,
  operator override sets `h_measured = true`) / lat-lon (shared
  component) / live V/m preview via `volume.js` (the existing
  client-side Tabacchi reuse from campionamenti). Server stores
  client-provided V and m as-is; no server-side recompute on the
  manual path.
- **PT-82** `[+ Nuova martellata]` ipso CSV import — uses
  `apps/base/tabacchi.py` server-side to fill `volume_m3` and
  `mass_q` for each row, applies the A7 row-level fingerprint
  dedup, auto-advances state from `planned` to `marked` on the
  first successful insert under this item.
- **PT-83** Closed-cantiere disable — `[+ Nuova martellata]` and
  `+ Aggiungi` are greyed with tooltip; an inline banner above the
  table reads "Il cantiere è chiuso, non si possono aggiungere
  martellate."
- **PT-84** Edit + delete affordances on individual `tree_mark`
  rows (pencil / trash). State stays monotonic on delete (B3): the
  count can return to zero, but state does not revert to
  `planned`.
- **PT-85** End-to-end smoke: import a real ipso CSV, verify
  tree_marks land and state advances to `marked`, re-import the
  same CSV → no duplicates, ipsometric auto-fill works when a
  regression exists.

## Phase 9 — Cleanup

- **PT-90** Refresh TOC in `abies/CLAUDE.md` (line numbers shift as
  the docs evolve).
- **PT-91** Audit `pages/controllo.md` and confirm the new tables
  (`HarvestPlanItem`, `TreeMark`, `HarvestTransition`,
  `TreeHeightRegression`) are registered with
  `django-simple-history` and surface in the audit log.

## Verification — overall end-to-end

```
cd abies && make dev && make test
# In a browser, after `make dev`:
# 1. /piano-di-taglio → "+ Nuovo piano" → import the three
#    pdg-2026/csv/* files into a fresh plan.
# 2. Calendar shows fustaia + ceduo rows with correct counts and
#    volumes (zeroes for marked/actual).
# 3. Open a fustaia item: state=planned. Click "Apri cantiere"
#    (skip-marking path). State becomes "open".
# 4. Switch to /prelievi → select that cantiere from the Cantiere
#    pulldown → enter a harvest. Save.
# 5. Back to /piano-di-taglio: the item's state is now
#    "harvesting" and volume_actual_m3 reflects the harvest.
# 6. (Phase 8) From the View/Edit modal, click "+ Nuova
#    martellata" → upload a real ipso CSV. tree_marks land;
#    state advances to "marked" (well, it was already past
#    "marked" — state monotonicity).
# 7. Re-upload the same ipso CSV → no duplicate tree_marks.
# 8. Click "Chiudi cantiere". State becomes "closed". Re-open
#    /prelievi; the cantiere is no longer in the pulldown.
# 9. Per-item "Esporta CSV" in the modal produces a zip with
#    martellate_<id>.csv and prelievi_<id>.csv.
# 10. Plan-level "Esporta CSV" produces the three round-trip
#     CSVs in a single zip.
```
