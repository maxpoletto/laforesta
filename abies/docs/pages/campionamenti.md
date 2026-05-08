# Campionamenti page

## Overview

Recording and exploration of forest sampling operations. A *sampling* takes
place in a circular *sample area* (typically ~12 m radius) where every tree
inside the radius is measured for diameter at breast height and total height.
For a subset of trees we also take a core sample to measure the "L10 distance",
the radial width of the 10 outer rings. For coppices, we measure the number of
shoots on a stump and whether each shoot is a *standard* (a shoot allowed to
grow — see https://en.wikipedia.org/wiki/Coppicing). Per-hectare biomass and
stand-structure statistics are extrapolated from these measurements to the
parcel and region.

Three concepts organize the data, all first-class schema entities:

- **Grid** (`sample_grid`): a layout of physical sample areas across one or
  more regions, where each `sample_area` row carries a `sample_grid_id` FK
  pointing at its grid (a sample area belongs to exactly one grid).
  Typically generated from a uniform spatial grid by the sample-area
  generation tool (similar to `bosco/pac`, but supporting multiple regions
  at the same time). Multiple surveys may use the same grid over time.
- **Survey** (`survey`): a high-level operation that visits the sample areas of
  exactly one grid, typically over a season or year. May be associated with a
  harvest plan or stand on its own. A survey is "complete" when every area in
  its grid has been visited.
- **Sample** (`sample`): a single visit to one sample area within a survey.
  Holds the date of the visit. Each `tree_sample` row hangs off a sample.

This page is the canonical surface for managing all three.

## Visual layout

The page has no page-level filter bar or other control.

The page is a vertical sequence of three collapsible sections separated by
dark-green 4 px horizontal rules:

1. Sampling grids ("Griglie di campionamento") (map-centered, collapsed by
   default).
2. Surveys ("Rilevamenti") (map-centered, open by default — the main entry
   point).
3. Sampled trees ("Alberi campionati") (table-based, collapsed by default;
   renders lazily).

The sampled trees table follows the standard sortable-table idiom from
`CLAUDE.md` ("Tabular data"): an inline search box on the upper left and
"Esporta CSV" on the upper right.

### Display and selection model

Opening a collapsed section to browse it has no effect on the state of other
sections.

Likewise, selecting a sampling grid ("griglia di campionamento") in section 1
has no effect on what is visible in sections 2 and 3.

However, if a survey ("rilevamento") is selected in section 2, then only its
trees (and no others) are displayed in section 3.

### Section 1 — Griglie (Grids)

Grids typically hold 100–500 sample areas, so a flat sortable-table is
not a useful primary view. Instead the section is map-centered.

**Top row** — a "Griglia" pulldown that selects the active grid by name (e.g.,
`Aree di saggio PDG 2026`). To the right of the pulldown sit "Nuova griglia" and
"Esporta CSV" buttons. Writers also see pencil and garbage icons next to the
pulldown for editing the active grid's `name` / `description` and for deletion
(the garbage icon is disabled when any survey references the grid).

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
  pencil and garbage icons in the popover.
- **Click** on empty map space (writers only): prompts "Inserire una
  nuova area qui?". On confirm, opens the new-area form (below) with
  lat/lng pre-filled to the clicked coordinates.

Below the map sit (writers only):

- A short hint: "Clicca sulla mappa per aggiungere un'area, oppure usa il
  pulsante." — addresses the discoverability of the click-to-create gesture.
- "+ Aggiungi area" button — opens the new-area form with empty lat/lng (for
  manual entry from a GPS device or paper notes).

The new-area form:

- Compresa (pulldown).
- Particella (pulldown, scoped to Compresa).
- Numero (auto-suggested as the next free integer for the parcel).
- Lat/lng (shared lat-lng component — see below). Pre-filled from the click
  location when entered via map-click.
- Quota (m).
- Raggio (default 10 m).
- Note (optional).

The "Esporta CSV" button at the top right exports the active grid's sample areas
in the same column shape as the import flow (see "Grid CSV import" below) —
useful for programming GPS devices for the field crew.

The "Nuova griglia" button opens a full-page modal with three
creation paths:

- *Genera automaticamente* — runs the grid generator (similar to `bosco/pac`)
  across user-selected regions, writes `sample_grid` plus `sample_area` rows in
  one transaction.
- *Importa da CSV* — see "Grid CSV import" below.
- *Crea vuota* — creates an empty grid, ready for manual area
  additions via the map.

Note that a grid can be edited (a sample area added, or an unused sample area
deleted) after a survey has started. A sample area cannot be deleted once it is
used in any sample.

#### Empty state

If no grids exist yet, the pulldown is disabled and the section shows a centred
prompt: "Nessuna griglia. Premi 'Nuova griglia' per crearne una." with the same
Nuova-griglia button.

### Section 2 — Rilevamenti (Surveys)

Layout is identical to the Griglie section above.

**Top row** — a "Rilevamenti" pulldown that selects the active survey by name
(e.g., `Bosco completo 2026`). To the right of the pulldown sit "Nuovo
rilevamento" and "Esporta CSV" buttons. Writers also see pencil and garbage
icons next to the pulldown for editing the active survey's `name` /
`description` and for deletion (the garbage icon is disabled when any samples
reference the survey).

Below the top row, a short summary of the active survey: Descrizione, griglia
(name), piano di taglio (nullable), n. aree visitate / n. aree totali, data
primo campione, data ultimo campione.

Selecting a survey displays all its corresponding trees in Section 3 (see
below).

The "Esporta CSV" button at the top right exports the active survey's full set
of trees in the same column shape as the import flow (see "Tree-and-sample CSV
import" below) — useful for round-tripping the whole survey. Section 3 has its
own separate "Esporta CSV" that exports only the currently displayed subset
(after the section's search filter and any area-click narrowing).

The "Nuovo rilevamento" button opens a full-page modal with two creation paths:

- *Importa da CSV* — see "Tree-and-sample CSV import" below.
- *Crea vuoto* — prompts user to choose a grid (via a pulldown of all available
  grids), then creates an empty survey based on that grid ready for manual data
  addition (in section 3).

**Map** — Leaflet map of the active survey's sample areas, drawn at their `r_m`
radius. Visited sample areas are in one color (abies palette dark green),
unvisited ones are in another (abies palette light green). After creating an
empty survey via "Crea vuoto" above, all grid dots on the map are in the
unvisited color, of course.

When no survey is selected (e.g., immediately after page load before a default
is picked, or after the user explicitly clears the pulldown), the map is empty
and Section 3 shows its empty state.

Map interactions:

- **Hover** on a sample area: a small tooltip shows just the region, parcel,
  sample area `numero`, and number of trees sampled (e.g., `Serra 2a / adc 17 /
  43 alberi`). Kept terse to avoid noise during navigation.
- **Click** on a sample area: filters Section 3 to just the trees in this sample
  area.
- **Click** on empty map space: sets Section 3 to display all trees in the
  current survey.

### Section 3 — Alberi campionati (Sampled trees)

Sortable-table of `tree_sample` rows joined with `tree`, `sample`,
`sample_area`, `parcel`, `species`, with its own search box and CSV export.

Without an active survey the section shows an empty state ("Seleziona un
rilevamento nella sezione Rilevamenti per visualizzare gli alberi") rather than
every sampled tree in the database.

With an active survey, shows all trees from that survey's samples.

With an active survey and sample area, narrows to that area's trees within that
survey. The header above the table also shows the date of the *sample* for that
area in that survey (editable inline; defaults to today if the sample row had no
date set). Date edits are simple metadata changes — they go through the
standard form-intercept path and don't trigger the warning + forced-CSV-export
flow used for destructive cascade operations.

Columns: comprea, particella, area di campionamento, n. albero
(`tree_sample.number`), specie, tipo (fustaia / ceduo), pollone, matricina, D
(cm), h (m), L10 (mm), PAI.

Writers see "+ Aggiungi", pencil, garbage. The "+" is enabled when both a survey
and sample area are selected in section 2. It opens the manual tree-entry flow
(below).

## Data entry flows

### Grid CSV import

Required columns: `Compresa`, `Particella`, `Area saggio` (→
`sample_area.number`), `Lon`, `Lat`, `Quota`, `Raggio`. Reference file:
`bosco/data/aree-di-saggio.csv` — currently lacks `Raggio`. Importer aborts with
a helpful message if any of these fields are missing.

Flow:
1. Writer picks "Nuova griglia" → "Importa da CSV", uploads file,
   provides a name and description for the new grid.
2. Importer resolves `Compresa` → region, `(region, Particella)` →
   parcel. Failure on either lookup aborts the entire import (no
   partial state).
3. Creates one `sample_grid` row.
4. For each row, creates a `sample_area` row with
   `sample_grid_id` pointing at the new grid. Each grid owns its
   own areas — there is no sharing of `sample_area` rows across
   grids.
5. Transactional. Reports counts and a per-row error list at the
   end.

### Tree-and-sample CSV import

Required columns: `Compresa`, `Particella`, `Area saggio`, `Albero` (→
`tree_sample.number`), `Pollone` (→ `tree_sample.shoot`), `Matricina` (bool, →
`tree_sample.standard`), `D_cm`, `H_m`, `L10_mm`, `Genere` (→ species),
`Fustaia` (bool; `Fustaia=false` → `tree.coppice=true`). Importer aborts with a
helpful message if any of these fields is missing. However, on a per-row level,
it is ok for L10_mm to be 0 (not all trees are cored).

Optional columns: `Data` (→ `sample.date`), `PAI` (bool, →
`tree.preserved`).

Flow:
1. Writer selects the target survey.
2. Uploads CSV. If the file lacks a `Data` column, the form asks for  a default
   sample date applied to all rows.
3. Importer groups rows by (Compresa, Particella, Area saggio, Data). Each group
   becomes one `sample` row in the target survey (skipped if a sample already
   exists for that area+date in the survey, with a conflict prompt).
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

Clicking on "+ Aggiungi" at the bottom of the sampled trees table  triggers a
full-page modal with the tree input form.

Top of form shows the following data (displayed, not editable):
data, compresa, particella, area di saggio

Editable fields:
- numero albero (`tree_sample.number`) (pulldown—see below)
- specie (pulldown)
- fustaia (checkbox). Defaults to fustaia, except in parcels whose
  `eclass.coppice = true`, where it defaults to ceduo.
- D (cm)
- h (m)
- L10 (mm) (can be left blank, will default to 0)
- Lat/lng of the individual tree (optional — shared component;
  defaults to the sample-area center).
- Pianta ad accrescimento indefinito (checkbox)

The "numero albero" pulldown has the following entries:
- The top entry is "nuovo albero", in which case the tree is assigned a new
  number never used before in this sample area (across any sample).
- The next entries are, in numerical order, all previously recorded tree numbers
  in this sample area, with in parentheses the species and last measured
  diameter and height, e.g., "n.1 (abete, d=40cm, h=20m)"

If the user selects a previous tree, "specie", "fustaia", and lat/lng (optional)
are locked.

If fustaia is not checked, the D, h, and L10 fields are indented and refer to
individual shoots. The indented block looks like:
    [numero pollone] [matricina (checkbox, default off)] [D] [h] [L10]
    [aggiungi pollone button]
The numero pollone is non-editable. Its starting value is 1 for a brand-new
tree, or `max(existing tree_sample.shoot for this tree across all samples) + 1`
when the operator picked an existing tree from the pulldown. `Aggiungi pollone`
adds a further row with the next sequential number.

At the bottom of the form are "Salva" / "Salva e aggiungi" submit buttons, the
latter for batch entry of consecutive trees in the same sample, styled
identically to the harvest input form.

### Cross-sample tree identity

Within a single sample area, a physical tree carries the same
`tree_sample.number` (`Albero`) across all samples in which it appears. This
convention is enforced by the app — not by a schema constraint — and rests on
physical tagging or careful field notes to preserve numbering between visits.

When the operator picks an existing tree from the previous-samples list during
manual entry, the app reuses the corresponding `tree_id` and propagates its
`Albero` number into the new `tree_sample` row. When they declare a new tree,
the app assigns `Albero = max(existing in this area) + 1`.

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

## Lat/lng entry component (shared)

Used here, on Bosco (PAI add), and on Piano di taglio (mark-tree add). Two
coupled inputs (latitudine, longitudine) plus a "Usa posizione attuale" button.
The button is enabled only when the browser's `navigator.geolocation` reports
availability and the user has granted permission; otherwise it is hidden. On
click it populates the inputs with the device's current coordinates.

Manual entry remains the primary path (office staff entering data from paper
field notes).

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

TBD — defer until the rest of the UX is settled.

## Data tables

TBD — defer. Likely:

- `grids.json`: list of grids with sample-area counts, regions covered.
- `surveys.json`: list of surveys with completeness counts.
- `sample_areas.json`: as on Bosco.
- `sampled_trees.json`: denormalized tree+tree_sample+sample for
  table rendering.

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
