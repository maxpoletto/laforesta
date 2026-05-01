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

The page is a vertical sequence of four collapsible sections separated by
dark-green 4 px horizontal rules:

1. Sampling grids ("Griglie di campionamento") (collapsed by default).
2. Surveys ("Rilevamenti") (open by default — the main entry point).
3. Sampled trees ("Alberi campionati") (collapsed by default; renders lazily).

Each table follows the standard sortable-table idiom from `CLAUDE.md` ("Tabular
data"): an inline search box on the upper left and "Esporta CSV" on the upper
right, scoped to that section's data only.

### Display and selection model

Opening a collapsed section to browse it has no effect on the state of other
sections.

Likewise, selecting a sampling grid ("griglia di campionamento") in section 1
has no effect on what is visible in sections 2 and 3.

However, if a survey ("rilevamento") is selected in section 2, then only its
trees (and no others) are displayed in section 3.

### Section 1 — Griglie

Grids typically hold 100–500 sample areas, so a flat sortable-table is
not a useful primary view.  Instead the section is map-centered.

**Top row** — a "Griglia" pulldown that selects the active grid by name;
pulldown options also show a short metadata summary (e.g., `Bosco completo 2027
(218 aree, 3 rilevamenti)`).  To the right of the pulldown sit "Nuova griglia"
and "Esporta CSV" buttons.  Writers also see pencil and garbage icons next to
the pulldown for editing the active grid's `name` / `description` and for
deletion (the garbage icon is disabled when any survey references the grid).

Below the top row, a short summary of the active grid: n. aree, comprese
coperte, n. rilevamenti che la usano, data ultimo aggiornamento, descrizione.

**Map** — Leaflet map of the active grid's sample areas, drawn at
their `r_m` radius.  This map is purely a grid-management surface; it
does not show survey/visited information (that's section 2's job).
A short note below the map clarifies the distinction.

Map interactions:

- **Hover** on a sample area: a small tooltip shows just the region and parcel
  and `numero` (e.g., `Serra 2a / 17`).  Kept terse to avoid noise during
  navigation.
- **Click** on an existing sample area: opens a popover with the full
  per-area fields (parcel, numero, lat, lng, quota, raggio, note).
  Writers also see pencil and garbage icons in the popover.
- **Click** on empty map space (writers only): prompts "Inserire una
  nuova area qui?".  On confirm, opens the new-area form (below) with
  lat/lng pre-filled to the clicked coordinates.

Below the map sit (writers only):

- A short hint: "Clicca sulla mappa per aggiungere un'area, oppure
  usa il pulsante." — addresses the discoverability of the
  click-to-create gesture.
- "+ Aggiungi area" button — opens the new-area form with empty
  lat/lng (for manual entry from a GPS device or paper notes).

The new-area form:

- Compresa (pulldown).
- Particella (pulldown, scoped to Compresa).
- Numero (auto-suggested as the next free integer for the parcel).
- Lat/lng (shared lat-lng component — see below).  Pre-filled from
  the click location when entered via map-click.
- Quota (m).
- Raggio (default 10 m).
- Note (optional).

The "Esporta CSV" button at the top right exports the active grid's
sample areas in the same column shape as the import flow (see "Grid
CSV import" below) — useful for programming GPS devices for the
field crew.

The "Nuova griglia" button opens a full-page modal with three
creation paths:

- *Genera automaticamente* — runs the grid generator (similar to
  `bosco/pac`) across user-selected regions, writes `sample_grid`
  plus `sample_area` rows in one transaction.
- *Importa da CSV* — see "Grid CSV import" below.
- *Crea vuota* — creates an empty grid, ready for manual area
  additions via the map.

#### Empty state

If no grids exist yet, the pulldown is disabled and the section shows
a centred prompt: "Nessuna griglia.  Premi 'Nuova griglia' per
crearne una." with the same Nuova-griglia button.

### Section 2 — Rilevamenti

Sortable-table of surveys, with its own search box, "filtra per
griglia" pulldown alongside the search, and CSV export.

Columns: descrizione, griglia (link to section 1), piano di taglio
(nullable), n. aree visitate / n. aree totali, data primo sample,
data ultimo sample, stato (*in corso* / *completo*, computed from
completeness).

Selecting a row marks it as the active survey for sections 3 and 4
(see "Selection model" above).

Writers see "+", pencil, and garbage. The garbage icon triggers the
strong-warning + forced-export flow documented in `database.md`
(equivalent to "Esporta CSV" of the affected rows before deletion is
permitted).

The "+" opens the new-survey form:

- Descrizione.
- Griglia (pulldown of grids). No auto-fill from section 1, since
  selecting a grid there does not cascade.
- Piano di taglio (pulldown, optional).

### Section 3 — Mappa

Standard Leaflet map. The region pulldown lives in the map nav bar
(see "Maps" in `CLAUDE.md`) and is the only place region selection
exists on this page. Shows parcel borders and sample-area circles
drawn at their actual `r_m` radius.

Coloring depends on the active survey selected in section 2:

- No active survey: all areas in the region tinted neutrally.
- Active survey: areas split into *visitate* (one color) vs *non
  visitate* (another), based on whether a `sample` row exists for
  the area in this survey. Areas outside the survey's grid appear
  as faint outlines for context.

Click a sample-area circle to mark it as the active area, which
further narrows section 4. Click again to deselect.

### Section 4 — Alberi campionati

Sortable-table of `tree_sample` rows joined with `tree`, `sample`,
`sample_area`, `parcel`, `species`, with its own search box and CSV
export.

Without an active survey the section shows an empty state
("Seleziona un rilevamento nella sezione Rilevamenti per
visualizzare gli alberi") rather than every sampled tree in the
database. With an active survey, shows all trees from that
survey's samples. With an active area (via section 3), narrows to
that area's trees within that survey.

Columns: data, regione, particella, area, n. albero
(`tree_sample.number`), specie, tipo (fustaia / ceduo), pollone,
matricina, D (cm), h (m), L10 (mm), PAI.

Writers see "+", pencil, garbage. The "+" is enabled when a sample
area is selected in section 3 *and* a survey in section 2; it opens the
manual tree-entry flow (below).

## Data entry flows

### Grid CSV import

Required columns: `Compresa`, `Particella`, `Area saggio` (→
`sample_area.number`), `Lon`, `Lat`, `Quota`, `Raggio`. Reference file:
`bosco/data/aree-di-saggio.csv` — currently lacks `Raggio`; the
importer defaults it to 20 m when missing.

Flow:
1. Writer picks "Nuova griglia" → "Importa da CSV", uploads file,
   provides a name and description for the new grid.
2. Importer resolves `Compresa` → region, `(region, Particella)` →
   parcel.  Failure on either lookup aborts the entire import (no
   partial state).
3. Creates one `sample_grid` row.
4. For each row, creates a `sample_area` row with
   `sample_grid_id` pointing at the new grid.  Each grid owns its
   own areas — there is no sharing of `sample_area` rows across
   grids.
5. Transactional.  Reports counts and a per-row error list at the
   end.

### Batch tree-and-sample CSV import

Required columns: `Compresa`, `Particella`, `Area saggio`, `Albero`
(→ `tree_sample.number`), `Pollone` (→ `tree_sample.shoot`),
`Matricina` (bool, → `tree_sample.standard`), `D_cm`, `H_m`,
`L10_mm`, `Genere` (→ species), `Fustaia` (bool;
`Fustaia=false` → `tree.coppice=true`).

Optional columns: `Data` (→ `sample.date`), `PAI` (bool, →
`tree.preserved`).

Flow:
1. Writer selects the target survey.
2. Uploads CSV. If the file lacks a `Data` column, the form asks for
. a default sample date applied to all rows.
3. Importer groups rows by (Compresa, Particella, Area saggio, Data).
. Each group becomes one `sample` row in the target survey
. (skipped if a sample already exists for that area+date in the
. survey, with a conflict prompt).
4. For each row, resolves `tree_id` via the cross-sample identity
. convention (see below): same `(sample_area, Albero)` → reuse
. existing `tree.id`; otherwise create a new `tree` row.
5. Writes `tree_sample`. `PAI=true` sets `tree.preserved=true`.
. `Fustaia=false` sets `tree.coppice=true`.
6. Transactional. Reports success counts and a per-row error list.

The Compresa+Particella+Area saggio referenced by each row must
already exist in the survey's grid; the schema-level trigger (see
`database.md`) prevents writing samples to areas outside the grid.

### Manual tree + sample entry

Two-stage flow, triggered by "+" in section 4 with a survey + sample
area selected.

Stage 1 — sample setup (skipped if a sample already exists in the
selected survey for the selected area on the chosen date):

- Data (defaults to today).
- "Crea sample" button creates the `sample` row.

Stage 2 — tree entry, with the sample now active:

- The form lists all trees previously sampled in this `sample_area`
  across any survey, by `Albero` number, species, and last-known
  measurements. The operator either:
  - Picks an existing tree from the list → the form pre-fills species
. and inherits `Albero` (= `tree_sample.number`). The operator
. enters the new measurements.
  - Declares a new tree → the form opens with `Albero` =
. max(existing in this area) + 1.

- Specie (pulldown — locked when reusing).
- N. albero (locked when reusing; suggested when new).
- D (cm), h (m), L10 (mm).
- Pollone (default 0; auto-increments per shoot of the same coppice
  stump within the entry session).
- Matricina (default off).
- Fustaia / ceduo radio (defaults to fustaia, except in parcels whose
  `eclass.coppice = true`, where it defaults to ceduo).
- PAI checkbox (default off — sets `tree.preserved=true` on save).
- Lat/lng of the individual tree (optional — shared component;
  defaults to the sample-area center).

"Salva" / "Salva e aggiungi" submit buttons, the latter for batch
entry of consecutive trees in the same sample.

### Cross-sample tree identity

Within a single sample area, a physical tree carries the same
`tree_sample.number` (`Albero`) across all samples in which it
appears. This convention is enforced by the app — not by a schema
constraint — and rests on physical tagging or careful field notes to
preserve numbering between visits.

When the operator picks an existing tree from the previous-samples
list during manual entry, the app reuses the corresponding `tree_id`
and propagates its `Albero` number into the new `tree_sample` row.
When they declare a new tree, the app assigns
`Albero = max(existing in this area) + 1`.

If field-numbering integrity is lost (e.g., a tag falls off, or the
operator can't distinguish two trees), the cross-sample link is lost
for that pair: a new `tree` row is created, and the historical row
remains in place but no longer linkable. This is a recoverable
workflow problem, not data corruption.

### Editing / deletion

Pencil opens the row in an edit form; garbage prompts for confirmation.
For surveys and samples, the database-level cascade rules (see
`database.md`) mean deletion can destroy a lot of work: the UI raises a
strong, distinct-styled warning ("Questa operazione cancellerà N
campioni e M misure di alberi che non possono essere recuperati") and
forces an "Esporta CSV" of the affected rows before the delete button
is enabled.

Deleting a single `tree_sample` row leaves both the `sample` and
the underlying `tree` row intact.

## Lat/lng entry component (shared)

Used here, on Bosco (PAI add), and on Piano di taglio (mark-tree add).
Two coupled inputs (latitudine, longitudine) plus a "Usa posizione
attuale" button. The button is enabled only when the browser's
`navigator.geolocation` reports availability and the user has granted
permission; otherwise it is hidden. On click it populates the inputs
with the device's current coordinates.

Manual entry remains the primary path (office staff entering data from
paper field notes).

## Cross-page links

- Any particella reference on this page (the area popover in section
  1, the new-area form, table cells in section 4) links to the Bosco
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

- **Bosco sidebar**: a page-level *Survey* pulldown sits in the top
  controls (alongside the region selector). It drives every Bosco view
  that reads sample-derived data — the per-parcel page's *Dendrometria*
  section, the visited-vs-unvisited coloring in *Aree di saggio* mode,
  and dendrometric entries in the *Caratteristiche* metric pulldown.
  The selector is global, not per-parcel: this keeps behavior
  predictable as the user clicks between parcels. Defaults to the most
  recent survey overall.
- **Multi-region surveys**: a single survey points at one grid. For
  surveys that cover multiple regions, prefer making the grid generator
  produce a single forest-wide grid in one operation, rather than
  making the schema's grid-survey relationship many-to-many. See
  "Open questions".

## Open questions

1. **Cross-sample tree numbering**: should the convention "same
. `tree_sample.number` for the same physical tree within a sample
. area" be enforced by a SQLite trigger, or remain an app-level
. convention. Trigger pros: catches data-quality bugs early. Cons:
. complicates legitimate corrections (e.g., renumbering after a tag
. replacement). Lean: app-level for now, revisit if drift is
. observed.

3. **Grid editing scope**: are grids editable after a survey starts
. referencing them. Adding sample areas to a grid mid-survey would
. change "complete" semantics retroactively. Likely safer:
. grid is frozen once any survey references it; further additions
. require cloning the grid. TBD.

4. **Where session-level metadata lived in the previous draft**: now
. moot — `survey.description` carries it, and the proposed
. `sampling_session` table is not needed.
