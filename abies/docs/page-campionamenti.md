# Campionamenti page

## Overview

Recording and exploration of forest sampling operations. Trees are
measured in circular sample areas (~12 m radius) for diameter, height,
and optionally L10 (outer-10-ring width). Coppice trees are measured
per-shoot. Per-hectare biomass is extrapolated to parcel and region.

Three schema entities organize the data:

- **Grid** (`sample_grid`): a set of physical sample areas. Multiple
  surveys may reuse the same grid over time.
- **Survey** (`survey`): visits the areas of exactly one grid, typically
  over a season. "Complete" when every area has been visited.
- **Sample** (`sample`): one visit to one area within a survey; holds
  the date. Each `tree_sample` row hangs off a sample.

## Visual layout

No page-level filter bar. Three collapsible sections:

1. **Griglie di campionamento** (map, collapsed by default).
2. **Rilevamenti** (map, open by default — main entry point).
3. **Alberi campionati** (sortable-table, collapsed, renders lazily).

Sections are independent except:
- selecting a survey in section 2 drives section 3 (that survey's trees are
  shown);
- selecting a sample area in section further restricts section 3 (only that
  sample area's trees are shown).

### Section 1 — Griglie (Grids)

Section is built around a map.

**Top row** — a "Griglia" pulldown that selects the active grid by name (e.g.,
`Aree di saggio PDG 2026`). Immediately to the right of the pulldown writers see
pencil and garbage icons for editing (see "Modifica griglia modal") and deleting
the active grid (if any surveys reference it, clicking "garbage" generates an
error modal telling the user to first delete them).

On the far right of the top row, the CSV export and "Nuova Griglia" buttons (see
UI Design Patterns > Buttons).

Below the top row, a short summary of the active grid: n. aree, comprese
coperte, n. rilevamenti che la usano, data ultimo aggiornamento, descrizione.

**Map** — Leaflet map of the active grid's sample areas, drawn at their `r_m`
radius. This map is purely a grid-management surface; it does not show
survey/visited information (that's section 2's job). A short note below the map
clarifies the distinction.

Map interactions:

- **Hover** on a sample area: a small tooltip shows just the region, parcel,
  and `numero` of the area di campionamento (e.g., `Serra 2a / adc 17`). Kept
  terse to avoid noise during navigation.
- **Click** on an existing sample area: opens a popover with the full per-area
  fields (parcel, numero, lat, lng, quota, raggio, note). Writers also see
  "Modifica" and "Elimina" buttons (Elimina is greyed out if sample area has
  samples).
- **Click** on empty map space (writers only): prompts "Inserire una
  nuova area qui?". On confirm, opens the new-area form (below) with
  lat/lng pre-filled to the clicked coordinates. When the click falls inside a
  parcel polygon, the form also pre-selects that Compresa and Particella
  (resolved from the map geometry); a click outside every parcel leaves them
  for the writer to pick.

Below the map sit (writers only):

- A short hint: "Clicca sulla mappa per aggiungere un'area, o su un'area per
  modificarla." — addresses the discoverability of the click-to-create gesture.
- "+ Aggiungi area" button — opens the new-area form with empty lat/lng (for
  manual entry from a GPS device or paper notes).

The new-area form:

- Compresa (pulldown).
- Particella (pulldown, scoped to Compresa).
- Numero (auto-suggested as the next free integer for the compresa — area
  numbers are unique per compresa, not per particella). The suggestion tracks
  the selected Compresa and may be overridden.
- Lat/lng (shared lat-lng component — see below). Pre-filled from the click
  location when entered via map-click.
- Alt. (m).
- Raggio (m) (default 10 m).
- Note (optional).

The "Esporta CSV" button at the top right exports the active grid's sample areas
in the same column shape as the import flow (see "Grid CSV import" below) —
useful for programming GPS devices for the field crew.

The "Nuova griglia" button opens an overlay modal with two creation
paths:

- *Crea vuota* — creates an empty grid with a name and optional
  description, ready for manual area additions via the map or CSV import
  (via the pencil modal).
- *Genera automaticamente* — runs the grid generator (similar to `bosco/pac`)
  across user-selected regions, writes `sample_grid` plus `sample_area` rows in
  one transaction. Generated areas are numbered per compresa, restarting at 1
  in each (matching manually-added areas — see §"The new-area form").

This deliberately does *not* offer CSV import at creation time: identity
(the grid exists with a name) is decoupled from content (sample areas).
Imports happen later via the pencil modal.

Note that a grid can be edited (a sample area added, or an unused sample area
deleted) after a survey has started. A sample area cannot be deleted once it is
used in any sample.

#### Modifica griglia modal (pencil)

A two-tab modal acting on the currently-selected grid:

- **Dettagli** — name and description, both editable.
- **Importa aree da CSV** — upload a CSV of sample areas. Rows are
  added to the active grid; duplicates (same compresa + area number)
  are rejected. See "Grid CSV import" below.

#### Empty state

If no grids exist yet, the pulldown is disabled and the section shows a centred
prompt: "Nessuna griglia. Premi 'Nuova griglia' per crearne una." with the same
Nuova-griglia button.

### Section 2 — Rilevamenti (Surveys)

Same top-row layout as Griglie (pulldown + pencil/trash + Esporta CSV + Nuovo
rilevamento). Deleting an empty survey shows a simple warning. Deleting a
survey with trees prompts with a deletion-after-export modal (see UI Design
Patterns > Deletion-after-export).

Summary line: descrizione, griglia, n. aree visitate / totali, data primo/ultimo
campione.

Selecting a survey drives Section 3 (only that survey's trees). The
section-level "Esporta CSV" exports the full survey; Section 3 has its
own export for the filtered subset.

**Nuovo rilevamento** modal: Nome (required), Griglia (required pulldown),
Descrizione (optional). Creates an empty survey and binds the survey to an
existing grid. CSV import happens later via the pencil modal, or trees are added
individually.

#### Modifica rilevamento modal (pencil)

Two-tab modal (same pattern as Modifica griglia):

- **Dettagli** — name and description.
- **Importa alberi da CSV** — upload + optional "Data predefinita"
  fallback date. See "Tree-and-sample CSV import" below.

**Map** — sample areas drawn at `r_m` radius. Visited areas in dark
green, unvisited in light green. Click an area to filter Section 3;
click empty space to show all trees. Hover tooltip: region, parcel,
area number, n. trees sampled.

### Section 3 — Alberi campionati (Sampled trees)

Sortable-table of `tree_sample` rows joined with `tree`, `sample`,
`sample_area`, `parcel`, `species`, with its own search box and CSV export.

Without an active survey the section shows an empty state ("Seleziona un
rilevamento per visualizzare gli alberi campionati.") rather than every sampled
tree in the database.

With an active survey, shows all trees from that survey's samples.

With an active survey and sample area, narrows to that area's trees within that
survey.

Columns: compresa, particella, area di campionamento, n. albero
(`tree_sample.number`), specie, tipo (fustaia / ceduo), pollone, matricina, D
(cm), h (m), L10 (mm), V (m³), m (q), PAI, Lat/Lon. Pressler coefficient is not
displayed.

Pollone value is blank (not 0) for high forest (fustaia) rows.

V (m³) and m (q) are blank for coppice (ceduo) rows (no per-shoot volume
estimates — see `database.md`).

L10 column is blank if value is 0.

Writers see "+ Aggiungi", pencil, garbage. The "+" pops up an error unless both
a survey and a sample area are selected in section 2. It opens the manual
tree-entry flow (below).

## Data entry flows

### Grid CSV import

Required columns: `Compresa`, `Particella`, `Area saggio` (→
`sample_area.number`), `Lon`, `Lat`, `Quota` (or `Alt. (m)`), and
`Raggio` (or `Raggio (m)`).

Flow:
1. Writer opens the pencil modal on the target grid → "Importa aree da
   CSV" tab, uploads file.
2. Importer resolves `Compresa` → region, `(region, Particella)` →
   parcel. Failure on either lookup aborts the entire import (no
   partial state).
3. For each row, creates a `sample_area` row with
   `sample_grid_id` pointing at the target grid. Duplicates (same
   compresa + area number) are rejected.
4. Transactional. Reports counts and a per-row error list at the
   end.

### Tree-and-sample CSV import

Required columns: `Compresa`, `Particella`, `Area saggio`, `Albero` (→
`tree_sample.number`), `Pollone` (→ `tree_sample.shoot`), `Matricina` (bool, →
`tree_sample.standard`), `D_cm`, `H_m`, `L10_mm`, `Genere` (→ species),
`Fustaia` (bool; `Fustaia=false` → `tree.coppice=true`), Pressler (→
`tree_sample.pressler_coeff`). Importer aborts with a helpful message if any of
these fields is missing. However, on a per-row level, it is ok for `L10_mm` to
be 0 (not all trees are cored).

Optional columns: `Data` (→ `sample.date`), `PAI` (bool, →
`tree.preserved`).

`V_m3` is computed at import time from `D_cm`, `H_m`, and the species-specific
Tabacchi parameters in `apps/base/tabacchi.py` (vendored from `pdg-2026`).

`mass_q` is then `tree_mass_q(V_m3, species.density)` (= `V_m3 × density`, from
the stored 4-dp `V_m3`) for fustaia rows; left NULL for coppice rows and for
species outside the Tabacchi table. The interactive form mirrors this in JS
(`volume.js`) — see the V/m preview note below for the ≤1-ULP caveat.

Flow:
1. Writer opens the pencil modal on the target survey → "Importa alberi
   da CSV" tab.
2. Uploads CSV. If the file lacks a `Data` column, the form asks for a default
   sample date applied to all rows.
3. Importer maps each `(Compresa, Particella, Area saggio)` to the one
   `sample` row for that area in the target survey. All rows for the same area
   must carry the same sample date; if a sample already exists for that area in
   the survey, the CSV date must match it. Conflicting dates abort the import.
4. For each row, resolves `tree_id` via the cross-sample identity convention
   (see below): same `(sample_area, Albero)` → reuse existing `tree.id`;
   otherwise create a new `tree` row.
5. Writes `tree_sample`. `PAI=true` sets `tree.preserved=true`. `Fustaia=false`
   sets `tree.coppice=true`.
6. Transactional. Reports success counts and a per-row error list.

The Compresa+Particella+Area saggio referenced by each row must already exist in
the survey's grid; the schema-level trigger (see `database.md`) prevents writing
samples to areas outside the grid.

### Manual tree + sample entry

"+ Aggiungi" opens an overlay modal. Read-only header: compresa,
particella, area di saggio.

Fields: numero albero (pulldown — see "Cross-sample tree identity" below),
specie, ceduo checkbox (toggles coppice shoot block), D (cm), h (m), L10 (mm,
optional), Pressler coeff (defaults to per-species default), lat/lon (shared
component), PAI checkbox.

For fustaia trees whose species has Tabacchi hypsometric params, a live V/m
preview (`V = X.XX m³ · m = X.X q`) recomputes as D/h/specie change. V via
the Tabacchi formula in JS (`volume.js`); m = V × density from `species.json`,
derived from the **stored** 4-dp V so it equals `round(V × density)`, matching
the import path. The server stores the client-provided V/m without recomputing,
so the same tree entered here vs imported from CSV (Python `tabacchi.py`) may
differ by ≤1 ULP (≤0.0001 m³ / ≤0.001 q): JS rounds in float, Python in Decimal.
The two are held in parity by `test/test_tabacchi.py`.

The **numero albero pulldown** lists "nuovo albero" (auto-numbered) then
all previously recorded trees in this area with species and last
measurements. Selecting an existing tree locks specie, ceduo, and
lat/lng.

**Coppice block** (ceduo checked): D/h/L10 are per-shoot. Each row:
`[N. pollone] [matricina] [D] [h] [L10]`. Starting shoot number
continues from prior samples. "+ Aggiungi pollone" adds rows.

Submit: "Salva" / "Salva e continua" (batch entry).

### Cross-sample tree identity

Within a single sample area, a physical tree carries the same
`tree_sample.number` (`Albero`) across all samples in which it appears. This
convention is enforced by the app — not by a schema constraint — and rests on
physical tagging or careful field notes to preserve numbering between visits.

When operator picks an existing tree from  previous-samples list during manual
entry, app reuses the corresponding `tree_id` and propagates its `Albero` number
into the new `tree_sample` row. When they declare a new tree, the app assigns
`Albero = max(existing in this area) + 1`.

If field-numbering integrity is lost (e.g., a tag falls off, or the operator
can't distinguish two trees), the cross-sample link is lost for that pair: a new
`tree` row is created, and the historical row remains in place but no longer
linkable. This is a recoverable workflow problem, not data corruption.

### Editing / deletion

Pencil opens the row in an edit form; garbage prompts for confirmation. For
surveys and samples, the database-level cascade rules (see `database.md`) mean
deletion can destroy a lot of work: the UI raises a strong, distinct-styled
warning ("Questa operazione cancellerà N campioni e M misure di alberi che non
possono essere recuperati") and forces an "Esporta CSV" of the affected rows
before the delete button is enabled.

Deleting a single `tree_sample` row leaves both the `sample` and the underlying
`tree` row intact.

## Lat/lon entry component (shared)

Used here and on Piano di taglio (mark-tree add). Two coupled inputs
(latitudine, longitudine) plus a "Usa posizione attuale" button. The button is
enabled only when the browser's `navigator.geolocation` reports availability and
the user has granted permission; otherwise it is hidden. On click it populates
the inputs with the device's current coordinates.

## Cross-page links

- Any particella reference on this page (the area popover in section
  1, the new-area form, table cells in section 3) links to the Bosco
  per-parcel page for that parcel.
- The per-parcel page on Bosco shows recent surveys touching that
  parcel and links back here pre-filtered to the relevant survey.
- The Bosco per-parcel page's *Dendrometria* section reads from the
  global survey selector in the Bosco sidebar (a page-level filter,
  not a per-parcel control — see "Knock-on changes" below).

## URL parameters

- Path: `/campionamenti`
- Query parameters:
  - `o=...`: which sections are expanded — single-char tokens, order
    irrelevant.  Tokens: `g` = Griglie, `r` = Rilevamenti, `t` =
    Alberi campionati.  Absent → default (`r`); explicit empty
    (`?o=`) → all closed.
  - `g=N`: id of the active grid in section 1.  Absent or pointing
    at a deleted grid → most recently updated grid (or empty state
    if none exist).
  - `s=N`: id of the active survey in section 2.  Drives section 3.
    Absent or pointing at a deleted survey → most recently active
    survey (or empty state if none exist).
  - `a=N`: id of the active sample area within the active survey
    (set by clicking an area on section 2's map).  Narrows section
    3 to that area's trees and surfaces the sample date inline.
    Silently cleared if the URL's `a=N` does not belong to the
    active survey's grid (handles stale shares).
  - Section 3 sortable-table state:
    - `tf=...`: URL-encoded search-box filter.
    - `tsc=N`: sort column index.
    - `tso=0|1`: sort order (0 = ascending, 1 = descending).
    Default sort: by Compresa, Particella, n. area, n. albero,
    pollone (natural reading order).
  - `mt=o|t|s`: shared map type for both maps (Section 1 and
    Section 2).  Defaults to `s` (Satellite).

Map center and zoom are *not* encoded — both maps auto-fit to the
active grid (Section 1) / active survey's grid (Section 2) on every
render, and pan/zoom is treated as transient view state.  This keeps
URLs short and avoids the "shared link reproduces a stale viewport"
trap.

The active grid (`g=`) and active survey (`s=`) are independent
controls: they can point at different grids without contradiction.
Section 3 only depends on `s=` and `a=` — `g=` does not narrow it.

## Data tables

Six digests serve this page.  Four are eagerly loaded on first
navigation (small, drive the always-visible Section 2 map and the
pulldowns); one is lazily loaded per active survey when Section 3
expands; one is shared with Piano di taglio.

### Cache invalidation

| Write | Digests marked stale | Optimistic client patch |
|---|---|---|
| Grid save (create) | `grids`, `audit` | `grids` via `patches` |
| Grid edit (rename) | `grids`, `audit` | `grids` via `patches`; pulldown option text updated |
| Grid delete | `grids`, `audit` | `grids` (row removed) |
| Grid CSV import (areas) | `grids`, `sample_areas`, `surveys`, `audit` | All three via `patches` |
| Survey save (create) | `surveys`, `grids`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | `surveys` + `grids` via `applySideEffects` |
| Survey edit (rename) | `surveys`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | `surveys` via `patches`; pulldown option text updated |
| Survey delete | `sampled_trees_<id>`, `samples`, `surveys`, `grids`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | Force-refresh via `cache.load` (cascade) |
| Survey CSV import (trees) | `sampled_trees_<id>`, `samples`, `surveys`, `parcel_dendrometry`, `parcel_dendrometry_points`, `preserved_trees`, `audit` | Force-refresh all three via `cache.load` (no records returned); the `sampled_trees_<id>` reload fires the Section 3 table's `onUpdate`; Section 2 summary + map re-rendered; survey pulldown rebuilt |
| Area save (create/update) | `sample_areas`, `grids`, `surveys`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | All three via `applySideEffects`; survey pulldown rebuilt; both maps re-rendered if affected |
| Area delete | `sample_areas`, `grids`, `surveys`, `audit` | Same as area save |
| Tree save (create/update) | `sampled_trees_<id>`, `samples`, `surveys`, `parcel_dendrometry`, `parcel_dendrometry_points`, `preserved_trees`, `audit` | All three via `applySideEffects`; Section 2 map re-rendered |
| Tree delete | `sampled_trees_<id>`, `samples`, `surveys`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | Same as tree save |

### `grids.json`

Drives the Section 1 pulldown and active-grid summary line.
Eager-loaded only when Section 1 opens (Section 1 is closed by
default).  Invalidated on `sample_grid`, `sample_area`, and `survey`
writes (the last because `N. rilevamenti` counts surveys per grid).

Columns: `row_id`, `version`, `Nome`, `Descrizione`, `N. aree`,
`Comprese`, `N. rilevamenti`, `Ultimo aggiornamento`.  Sorted by
`Ultimo aggiornamento` descending.

`Comprese` is a comma-separated string of distinct region names
covered by the grid's areas.  `N. rilevamenti` is `COUNT(survey
WHERE sample_grid_id = this.id)`.  `Ultimo aggiornamento` is
`max(sample_grid.modified_at, max sample_area.modified_at within
the grid)`.

### `surveys.json`

Drives the Section 2 pulldown and active-survey summary line.
Eager-loaded.  Invalidated on `survey`, `sample`, and `sample_area`
writes.

Columns: `row_id`, `version`, `Nome`, `Descrizione`, `Griglia`
(grid id), `N. aree visitate`, `N. aree totali`, `Data primo`,
`Data ultimo`.  Sorted by `Data ultimo` descending (most recently
active first), falling back to `created_at` for surveys without
samples yet.

`N. aree visitate` is `COUNT(DISTINCT sample.sample_area_id WHERE
survey_id = this.id)`.  `N. aree totali` is the size of the survey's
grid (`COUNT(sample_area WHERE sample_grid_id = survey.sample_grid_id)`).
Grid name is looked up client-side via `grids.json`.

### `sample_areas.json`

All sample-area rows across all grids.  Eager-loaded; the client
filters by active grid (Section 1) or by the active survey's grid
(Section 2).  Invalidated on `sample_area` writes.

Columns: `row_id`, `version`, `Griglia` (sample_grid_id), `Compresa`,
`Particella`, `Numero`, `Lat`, `Lon`, `Alt. (m)`, `Raggio (m)`, `Note`.

### `samples.json`

All `sample` rows.  Drives Section 2's visited/unvisited map coloring,
the per-area "n. alberi" hover tooltip, and the inline sample-date
display in Section 3.  Eager-loaded.  Invalidated on `sample` writes
and on `tree_sample` writes (since `N. alberi` is materialized).

Columns: `row_id`, `version`, `Survey`, `Sample area`, `Data`,
`N. alberi`.

`N. alberi` is `COUNT(tree_sample WHERE sample_id = this.id)`,
materialized so the Section 2 hover tooltip can render without
loading the heavy per-survey tree digest.

### `sampled_trees_<survey_id>.json` (lazy, per survey)

One digest per survey, denormalized for Section 3.  Lazily fetched
the first time a given `s=<id>` enters scope (either via URL on page
entry or via Section 2 pulldown change), and cached client-side per
survey for the rest of the session.  Invalidated on `tree_sample`
writes whose sample's survey matches.

Columns: `row_id`, `version`, `Sample area`, `Data campione`,
`Compresa`, `Particella`, `N. area`, `N. albero`, `Specie`, `Tipo`,
`Coppice`, `Pollone`, `Matricina`, `D (cm)`, `h (m)`, `L10 (mm)`, `V (m³)`,
`m (q)`, `PAI`, `Lat`, `Lon`.  Sort: by `Compresa`, `Particella`,
`N. area`, `N. albero`, `Pollone`.

`row_id` = `tree_sample.id` (the synthetic id, see `database.md`).
`Coppice` is the stable boolean copied from `tree.coppice`; `Tipo` is the
localized display label derived from the same value.
`V (m³)` and `m (q)` are NULL for ceduo rows (per `database.md`
invariant).  `Lat`, `Lon` come from `tree.lat/lon` if set, else fall
back to the sample-area center.  `Data campione` is `sample.date`
(useful for cross-tab tracking even when Section 3 isn't narrowed
to a single area).

This digest is shared between the page's Section 3 sortable-table
*and* its CSV export from Section 3 / Section 2 (the latter via the
"export the active survey's full set of trees" button — same data,
the section-2 export bypasses the section-3 search filter).

Bosco's per-parcel Dendrometria charts read a different digest
(`parcel_dendrometry.json` — pre-aggregated, not per-tree); they do
not consume `sampled_trees_<survey_id>.json`.

### `species.json`

Shared with Piano di taglio (mark form).  Small lookup digest
(`row_id`, `version`, `Nome`, `Nome latino`, `Densità (q/m³)`,
`Sort order`, `Attiva`) used by:
- the live V/m preview in this page's manual entry form;
- the live V/m preview in Piano di taglio's mark form;
- digest-generation joins for `marks.json` and any other materialized
  mass total derived from V × density.
Invalidated on `species` writes.

## Knock-on changes

- **Bosco sidebar**: a page-level *Survey* pulldown sits in the top controls
  (alongside the region selector). It drives every Bosco view that reads
  sample-derived data — the per-parcel page's *Dendrometria* section, the
  visited-vs-unvisited coloring in *Aree di saggio* mode, and dendrometric
  entries in the *Caratteristiche* metric pulldown. The selector is global, not
  per-parcel: this keeps behavior predictable as the user clicks between
  parcels. Defaults to the most recent survey overall.
- **Multi-region surveys**: a single survey points at one grid. For surveys that
  cover multiple regions, prefer making the grid generator produce a single
  forest-wide grid in one operation, rather than making the schema's grid-survey
  relationship many-to-many.
