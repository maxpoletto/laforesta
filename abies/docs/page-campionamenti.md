# Campionamenti page

## Overview

Recording and exploration of forest sampling operations. Trees are
measured in circular sample areas (~12 m radius) for diameter, height,
and optionally L10 (outer-10-ring width). Coppice trees are measured
per-shoot. Per-hectare biomass is extrapolated to parcel and region.

Three schema entities organize the data:

- **Grid** (`sample_grid`): a set of physical sample areas. Multiple
  surveys may reuse the same grid over time.
- **Survey** (`survey`): a tree survey. Structured surveys visit the areas of
  exactly one grid, typically over a season; unstructured surveys have no grid
  and are not statistically expandable by default.
- **Sample** (`sample`): a dated group of tree measurements within a survey.
  Structured samples point at one sample area; unstructured samples do not.
  Each `tree_sample` row hangs off a sample.

## Visual layout

No page-level filter bar. Two collapsible workspaces:

1. **Griglie di campionamento** (grid management map, collapsed by default).
2. **Rilevamenti** (survey overview, titled map, sampled-tree table and
   dendrometric summary; open by default). Structured surveys map sample areas;
   free surveys map individual trees. Both maps are 600px tall.

The trees belong directly to the survey workspace, without a separate Alberi
campionati foldable section. Selecting a survey changes its tree table;
selecting an area on a structured survey's map narrows that table. Grid and
survey selections remain independent: managing another grid does not change
the selected survey or its trees.

### Section 1 — Griglie (Grids)

Section is built around a map.

**Top row** — a "Griglia" pulldown that selects the active grid by name (e.g.,
`Aree di saggio PDG 2026`). Immediately to the right of the pulldown writers see
pencil and garbage icons for editing (see "Modifica griglia modal") and deleting
the active grid (if any surveys reference it, clicking "garbage" generates an
error modal telling the user to first delete them).

On the far right of the top row, writers see **+ Nuova griglia**.

The next row shows the active grid's summary (number of areas, regions, surveys
using it, last update and description), with `[Esporta] [Importa CSV]
[+ Nuova area]` aligned to the right. Readers see only Esporta.
The row wraps on narrow screens. Import and manual area creation target the
selected grid, including an empty grid.

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
  "Modifica" and "Elimina" buttons (Elimina is grayed out if sample area has
  samples).
- **Click** on empty map space (writers only): prompts "Inserire una
  nuova area qui?". On confirm, opens the new-area form (below) with
  lat/lng pre-filled to the clicked coordinates. When the click falls inside a
  parcel polygon, the form also pre-selects that Compresa and Particella
  (resolved from the map geometry); a click outside every parcel leaves them
  for the writer to pick.

Below the map, writers see the hint "Clicca sulla mappa per aggiungere
un'area, o su un'area per modificarla." The **+ Nuova area**
button in the summary row opens the same form with empty coordinates, for
manual entry from a GPS device or paper notes.

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

The "Esporta" button in the summary row exports the active grid's sample areas
in the same column shape as the import flow (see "Grid CSV import" below),
including optional notes — useful for programming GPS devices for the field
crew and as a lossless backup.

The "Nuova griglia" button opens an overlay modal with two creation
paths:

- *Crea vuota* — creates an empty grid with a name and optional
  description, ready for manual area additions via the map or CSV import
  (via the workspace Importa CSV button).
- *Genera automaticamente* — runs the grid generator
  across user-selected regions, writes `sample_grid` plus `sample_area` rows in
  one transaction. Generated areas are numbered per compresa, restarting at 1
  in each (matching manually-added areas — see §"The new-area form").
  Generated locations form a cell-centered lattice in a local WGS84 meter plane,
  aligned north/east at the selected parcels' center and clipped to their
  geometry. Spacing and area estimates use the shared ellipsoidal calculations
  described in [map architecture](ui-maps.md#local-metric-planes-and-sampling-grids).

This deliberately does *not* offer CSV import at creation time: identity
(the grid exists with a name) is decoupled from content (sample areas).
Imports happen later via the workspace Importa CSV button.

Note that a grid can be edited (a sample area added, or an unused sample area
deleted) after a survey has started. A sample area cannot be deleted once it is
used in any sample.

#### Modifica griglia modal (pencil)

A single form edits the selected grid's name and description. It has no tabs;
**Importa CSV** in the grid summary row opens the separate file-import dialog.
Rows are added to the selected grid and duplicate area identities are rejected
(see "Grid CSV import" below).

#### Empty state

With no grids, the selector is hidden and the section asks the user to create
a grid. **+ Nuova griglia** stays in its usual top-row position; the summary
row and map remain hidden until a grid exists.

### Section 2 — Rilevamenti (Surveys)

Same top-row layout as Griglie (pulldown + pencil/trash, with
**+ Nuovo rilevamento** on the far right). Deleting an empty survey shows a
simple warning. Deleting a survey with trees prompts with a deletion-after-export
modal (see UI Design Patterns > Deletion-after-export).

The pulldown separates surveys into **Rilevamenti liberi** and
**Rilevamenti predefiniti** optgroups, in that order, and sorts each group
alphabetically. Free surveys are labelled `Nome (N alberi)`, where the count
is the sum of their samples' materialized `N. alberi` values. Structured
surveys retain the `Nome (visitate/totali aree)` label. The groups and labels
are rebuilt from the survey and sample caches so tree edits and imports update
the free-survey counts without a page reload.

Every survey shows a descriptive overview such as
`Dal 2026-10-01 al 2026-10-16. 43 alberi campionati.` The count covers all
measurements in the selected survey, independently of the table or sample-area
filter. If there are no samples, the date range is replaced by `Nessun
campione.` The user-provided description follows on its own line, preserving
line breaks.

**Nuovo rilevamento** modal: Nome (required), Griglia (required pulldown),
Descrizione (optional). Creates an empty survey and binds the survey to an
existing grid. CSV import happens later via the tree-table Importa CSV button,
or trees are added individually.

#### Modifica rilevamento modal (pencil)

A single metadata form edits name and description. It has no tabs. The tree
table's **Importa CSV** button opens a separate file-import dialog naming the
target survey, with an optional default date for files lacking `Data`.

**Map** — appears above the tree table in both survey types, at 600px tall.
Its heading explains what the markers represent:

- **Mappa delle aree di saggio** for a structured survey. Markers represent
  the grid's areas, including areas that have not been visited. Visited areas
  use the dark semantic marker color and unvisited areas the pale color (green
  on OSM/topographic maps, yellow on satellite). Hover shows region, parcel,
  area number and the survey's sampled-tree count. The gray italic note reads
  "Clicca su un'area di saggio per filtrare gli alberi."
- **Mappa degli alberi** for a free survey. Points represent sampled trees with
  valid coordinates. Trees without coordinates remain in the table and
  summary. A free survey with no measurements hides the map; an empty text
  filter result keeps the map visible with no tree points.

Clicking a sample area **replaces** the table's text filter with its complete
region/parcel/area identity, for example
`compresa:Serra particella:9 area:72`. Names with spaces are quoted.
Qualified region, parcel and area terms match whole identifiers, so area `7`
does not select `72` or an equally numbered area in another region. There is
no additional hidden area filter. The input, table, summary and data/report
exports all describe the same selection.

The selected map marker follows these full identity terms. Keeping the terms
and adding a species filter retains the selected area as manual-entry context.
Clearing or replacing them clears the marker selection; clicking empty map
space while an area is selected clears the filter. Unvisited areas can be
selected too: the table/summary are empty, and **+ Nuovo albero** can add the
first measurement to that area. The separate selected-area description is
unnecessary because the scope is visible in the filter input.

For a free survey, editing the table filter updates the map's tree points.
For a structured survey, filtering leaves all area markers and their visit
progress visible; the markers are a survey-navigation surface, not a map of
individual filtered trees.

Changing survey **from the dropdown** clears the text/area filter, retaining
the sort order. Opening a shared URL honors its supplied filter instead. The
old table is removed while another survey loads; a map selection made during
that load is retained when its rows arrive. A late response from a previous
survey cannot replace the current view.

#### Sampled trees within the survey workspace

Sortable-table of `tree_sample` rows joined with `tree`, `sample`,
`sample_area`, `parcel`, `species`. Its search/pagination row carries
`[Esporta] [Importa CSV] [+ Nuovo albero]` at the right. Readers see only
Esporta. The same controls remain available when an active survey is empty.
The table displays 20 rows per page. Pagination limits visible rows only;
filter results, exports, maps and summary include every matching measurement.

Without an active survey the section shows an empty state ("Seleziona un
rilevamento per visualizzare gli alberi campionati.") rather than every sampled
tree in the database.

With an active survey, shows all trees from that survey's samples. The
**Filtra** input is the single filter for the table and dendrometric summary.
It also filters the free survey's tree map.

Qualified parcel filters such as `part:1` match the full parcel identifier,
excluding `1b`, `10` and `10b`; unqualified text search remains substring-based.
Parcel-column sorting is natural (`1`, `1b`, `2`, `2a`, …, `10`, `11`) in either
direction. The filtered tree map, summary and export use the same exact parcel
selection as the table.

Both survey types show **Riassunto dendrometrico** below the table, using the
shared Bosco/Martellate three-chart layout, species legend, and totals for
sampled measurements, volume and basal area. These are totals of the filtered
sampled rows, not estimates per hectare. Bosco owns the per-hectare estimates.

The shared `TreeDetail` component renders this summary for both types. For free
surveys its tree map is mounted in the separate map host above the table; for
structured surveys its tree map is disabled, leaving the area-map controller
responsible for the markers and visit progress. Marking-page consumers retain
the component's default map-then-summary layout.

The summary consumes every row retained by the table's filter, so charts,
legend and totals remain synchronized with the selection. The free tree map
uses those same rows. The summary's gray **Esporta** button
downloads `riassunto_dendrometrico_rilevamento_<survey-id>.zip` containing
`numero_alberi.csv`, `volume_m3.csv`, and `area_basimetrica_m2.csv`. Each CSV
has species on rows and five-centimeter diameter classes on columns. Chart and
ZIP boundaries follow the site-wide **Classi diametriche** setting; the export
endpoint reads it directly. Species whose entire row is zero for that metric
are omitted. With an active text filter (including one written by an area
click), the client submits all matching `tree_sample` IDs, across pagination
pages, so the ZIP matches the visible summary. With no filter the server
exports the whole survey.

Display columns, in order: `Data`, `Compresa`, `Particella`, `Area`, `Albero`,
`Specie`, `Tipo`, `Poll`, `Mat`, `Diam (cm)`, `Alt (m)`, `L10 (mm)`, `Vol (m3)`,
`Massa (q)`, `PAI`, `Lat`, `Lon`, `Acc (m)`. Abbreviations have no trailing dots.
Measurement headers can wrap at the units to retain compact column widths.
Pressler is not displayed. Missing legacy accuracy is shown as `-`.

The first word of each label is an unambiguous selector, so users can type
`area:72`, `albero:17`, `diam:>30`, `alt:>15`, `vol:>1` or `massa:>10` without
quoting column names. Values containing spaces still need quotes, for example
`compresa:"Serra San Bruno"`. The selectors follow the localized display labels.
The table, map selection and manual-entry form use the same label selectors.
An exact visible label takes precedence over an unambiguous prefix. For an
older filter such as `"n. area":72`, the current selector is `area:72`.

Pollone value is blank (not 0) for high forest (fustaia) rows.

V (m³) and m (q) are blank for coppice (ceduo) rows (no per-shoot volume
estimates — see `database.md`).

L10 column is blank if value is 0.

Writers see "+ Nuovo albero", pencil and trash icons. Creating a tree requires
a survey. A predefined survey also requires a selected sample area, while a free survey
does not. It opens the manual tree-entry flow (below).

## Data entry flows

CSV validation names the incorrect field and explains its required value.
Messages count the header as row 1, combine scalar problems per row and abort
an invalid file without partial writes. See [CSV imports](csv-imports.md) for
shared diagnostics and the full list of import paths.

### Grid CSV import

Required columns: `Compresa`, `Particella`, `Area saggio` (→
`sample_area.number`), `Lon`, `Lat`, `Quota` (or `Alt. (m)`), and
`Raggio` (or `Raggio (m)`).

Flow:
1. Writer selects the target grid, then clicks **Importa CSV** in its summary
   row and uploads a file. The dialog names the grid.
2. Importer resolves `Compresa` → region, `(region, Particella)` →
   parcel. Failure on either lookup aborts the entire import (no
   partial state).
3. For each row, creates a `sample_area` row with
   `sample_grid_id` pointing at the target grid. Duplicates (same
   compresa + area number) are rejected.
4. Transactional. Reports counts and a per-row error list at the
   end.

### Tree-and-sample CSV import

The tree CSV import targets an existing survey. Structured surveys and free
surveys share the same measurement validation (`Albero`, `Pollone`, `D_cm`,
`H_m`, `L10_mm`, `Pressler`, `Matricina`, `Fustaia`, `H_measured`) but differ
in how the sample scope is resolved.

Required columns for structured surveys: `Compresa`, `Particella`,
`Area saggio`, `Albero` (→ `tree_sample.number`), `Pollone` (→
`tree_sample.shoot`), `Matricina` (bool → `tree_sample.standard`), `D_cm`,
`H_m`, `L10_mm`, `Pressler` (→ `tree_sample.pressler_coeff`), `Genere`
(→ species), and `Fustaia` (bool; `Fustaia=false` → `tree.coppice=true`).
`Area saggio` must already exist in the survey grid.

Required columns for free surveys: the same fields except `Area saggio`. Free
rows resolve the parcel directly from `Compresa` + `Particella`. The optional
`Campione` column groups rows into separate dated samples within a file. Its
values are file-local labels; exports number the groups from 1. Each group
must have one date, supplied by `Data` or the modal's default date. Tree/shoot
numbers are unique within a group but may restart in another, including two
samples on the same date. A file without `Campione` creates one sample, as do
rows with blank group values.

An ordinary free-survey row may leave `Albero` blank. Within each sample group,
the importer reserves explicit numbers, then numbers blanks above their
maximum (or from 1 when all are blank). A PAI row still requires an explicit
positive `Albero`. This also lets recovery CSVs from Ipso preserve genuinely
unnumbered source rows without inventing source values.

Optional columns: `Data` (→ `sample.date`), `PAI` (bool), `Numero PAI`,
`Campione`, `H_measured` (bool), `Lat`, `Lon`, `Acc_m`, `Operatore`, and `Note`.
Blank `H_measured` is false. If either `Lat` or `Lon` is present, both must be
present. After validation, rows with `H_measured=false` generate a non-blocking
warning. Rows with both coordinates also generate a warning if the point falls
in a different parcel from the submitted `Compresa`/`Particella`; after
confirmation, the submitted parcel remains authoritative.

`PAI=true` stores a parcel-scoped `tree_sample.preserved_number`. It uses
`Numero PAI` when supplied and otherwise falls back to `Albero`. The two
numbers may differ: `Albero` is local to the sample, while `Numero PAI`
identifies the preserved tree across visits. A nonblank `Numero PAI` must be a
positive integer and requires `PAI=true`; duplicates within a sample and parcel
are rejected. If `(parcel_id, preserved_number)` already exists, the import
reuses that `tree_id`; otherwise it creates a new preserved `tree`. Exports
include both optional identity columns, so restoring a free survey preserves
its sample groups and restoring either survey type preserves both numbers.

`V_m3` is computed at import time from `D_cm`, `H_m`, and the selected species'
configured volume equation, resolved by species ID through `apps/base/equations.py`.
The numeric Tabacchi core remains in `apps/base/tabacchi.py`. Coefficients and
explicit proxy references are editable in Settings; renaming a species leaves
the equation association intact. Initially, `Altra conifera` references Abete
bianco (`Abete`) and `Altra latifoglia` references Faggio. These references apply
to manual entry, CSV and Ipso imports without changing the observation's species
or substituting height-regression parameters.

`mass_q` is then `tree_mass_q(V_m3, species.density)` (= `V_m3 × density`, from
the stored 4-dp `V_m3`) for fustaia rows; left NULL for coppice rows and for
species without a configured volume equation. The interactive form mirrors this in JS
(`volume.js`) — see the V/m preview note below for the ≤1-ULP caveat.
Mass always uses the selected species' current density, not the proxy's.

Flow:
1. Writer selects the target survey, then clicks **Importa CSV** in its tree
   table controls. The dialog names the survey.
2. Uploads CSV. If the file lacks a `Data` column, the form asks for a default
   sample date applied to all rows.
3. Structured import maps each `(Compresa, Particella, Area saggio)` to the one
   `sample` row for that area in the target survey. All rows for the same area
   must carry the same sample date; if a sample already exists for that area in
   the survey, the CSV date must match it. Conflicting dates abort the import.
4. Free import creates one null-area sample per `Campione` group (one sample
   for files without groups). Ordinary trees are identified by `Albero` within
   that group, so coppice shoots share their physical tree. Preserved rows
   reuse the parcel-scoped PAI identity across groups and visits.
5. If warnings are present, the modal shows them with proceed/abort controls.
   Proceeding resubmits with confirmation; validation and warnings are
   recomputed before the write.
6. Writes `tree_sample`. `PAI=true` stores a non-null
   `tree_sample.preserved_number`. `Fustaia=false` sets `tree.coppice=true`.
7. Transactional. Reports success counts and a per-row error list.

The Compresa+Particella+Area saggio referenced by each structured row must
already exist in the survey's grid; the schema-level trigger (see
`database.md`) prevents writing samples to areas outside the grid.

### Manual tree + sample entry

"+ Nuovo albero" opens an overlay modal. Read-only header: compresa,
particella, area di saggio.

Fields: numero albero (pulldown — see "Cross-sample tree identity" below),
specie, ceduo checkbox (toggles coppice shoot block), D (cm), h (m), L10 (mm,
optional), Pressler coeff (defaults to per-species default), lat/lon (shared
component), PAI checkbox.

New trees always default to fustaia, including in coppice-managed parcels.
Species choices include `Altra conifera` and `Altra latifoglia`, but exclude
historical `Altro`. Existing Altro trees remain editable and can be selected
from the prior-tree picker for a new measurement of the same physical tree;
their locked species is not a new-tree choice. CSV imports and old Ipso uploads
retain their historical species IDs/names rather than relabeling them.

Parcel economic management and individual-tree morphology are independent;
the operator must select **Ceduo** explicitly when recording coppice shoots.

For fustaia trees whose species has a configured volume equation (including
explicit proxies), a live V/m
preview (`V = X.XX m³ · m = X.X q`) recomputes as D/h/specie change. V via
the configured coefficients passed with species options in JS (`volume.js`);
m = V × density from `species.json`,
derived from the **stored** 4-dp V so it equals `round(V × density)`, matching
the import path. The server recomputes V/m with the same Python helper used
by CSV imports. The JS preview can differ from the persisted Python estimate
by ≤1 ULP (≤0.0001 m³ / ≤0.001 q): JS rounds in float, Python in Decimal.
The two are held in parity by `test/test_tabacchi.py`.
Missing equations show an explicit unavailable-volume message, not a zero
estimate. Equation edits never recalculate existing samples.

The **numero albero pulldown** lists "nuovo albero" (auto-numbered) then
all previously recorded trees in this area with species and last
measurements. Selecting an existing tree locks specie, ceduo, and
lat/lng.

**Coppice block** (ceduo checked): D/h/L10 are per-shoot. Each row:
`[N. pollone] [matricina] [D] [h] [L10]`. Starting shoot number
continues from prior samples. "+ Aggiungi pollone" adds rows.

Submit: "Salva" / "Salva e continua" (batch entry).

For a free survey, an edit whose date is unchanged stays in its original
sample. A new manual row, or an edit moved to another date, uses the oldest
existing sample for that survey and date, creating one only if needed. There
may legitimately be several same-date samples because each CSV import and each
Bosco/PAI entry creates its own sample.

Resolving the target sample during validation is read-only: a missing sample
is only an unsaved candidate. Species, prior-tree and PAI identity checks, tree
number/shoot conflicts, and (for edits) the row's current version must all pass
before it is persisted. The parent sample, physical tree and all measurements
are then saved in one transaction. A rejected create/edit or failed shoot write
cannot leave an empty sample or partial measurements behind, with or without a
request nonce. Existing samples are not deleted when a measurement moves to
another date.

New rows in the canonical free survey `Alberi da preservare` default the PAI
checkbox on, so they immediately participate in the parcel-scoped preserved
tree identity and appear in Bosco/PAI. The checkbox remains editable. Other
free surveys continue to default it off.

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
possono essere recuperati") and forces an "Esporta" of the affected rows
before the delete button is enabled.

Deleting `tree_sample` rows leaves both their `sample` and underlying `tree`
rows intact. The sampled-tree table's header trash can enters multi-row
deletion mode. Select individual measurements across pages, review the listed
records in one confirmation, and delete them together. Selecting replaces the
pencil/trash icons with checkboxes and suspends row editing. Search and area
filter changes clear selection; sorting, pagination and unchanged background
refreshes preserve it.

The server confines a batch to its survey and checks every selected version
before deleting anything. A rejected selection leaves every measurement in
place. Accepted deletions update sample/survey counts, preserved-tree summaries, the
dendrometric summary, and the appropriate survey map. Survey deletion retains its
separate export safeguard. See
[Multi-row deletion](ui-design-patterns.md#multi-row-deletion).

Tree/area deletions and all their digest dirty marks share one transaction,
with or without a request nonce. Tree, area, grid and survey deletes also
recheck the submitted version at the write boundary, so a change after the
preflight read cannot cause deletion of a newer version.

## Lat/lon entry component (shared)

Used here and on Piano di taglio (mark-tree add). Two coupled inputs
(latitudine, longitudine) plus a "Usa posizione attuale" button. The button is
enabled only when the browser's `navigator.geolocation` reports availability and
the user has granted permission; otherwise it is hidden. On click it populates
the inputs with the device's current coordinates.

## Cross-page links

- Any particella reference on this page (the area popover in section
  1, the new-area form, sampled-tree table cells) links to the Bosco
  per-parcel page for that parcel.
- The per-parcel page on Bosco shows recent surveys touching that
  parcel and links back here pre-filtered to the relevant survey.
- The Bosco per-parcel page's *Dendrometria* section reads from the
  global survey selector in the Bosco sidebar (a page-level filter,
  not a per-parcel control — see "Knock-on changes" below).

## URL parameters

- Path: `/rilevamenti`
- Query parameters:
  - `o=...`: which sections are expanded — single-char tokens, order
    irrelevant.  Tokens: `g` = Griglie and `r` = Rilevamenti. A legacy `t`
    token opens Rilevamenti too; generated URLs use only `g` and `r`.
    Absent → default (`r`); explicit empty (`?o=`) → all closed.
  - `g=N`: id of the active grid in section 1.  Absent or pointing
    at a deleted grid → most recently updated grid (or empty state
    if none exist).
  - `s=N`: id of the active survey in section 2, including its tree table.
    Absent or pointing at a deleted survey → most recently active
    survey (or empty state if none exist).
  - Legacy `a=N` sample-area links are read as a visible region/parcel/area
    filter. New URLs express selection in `tf` alone. An area outside the
    selected survey's grid, or an area supplied for a free survey, is ignored.
  - Survey tree-table state:
    - `tf=...`: URL-encoded search-box filter.
    - `tsc=N`: sort column index.
    - `tso=0|1`: sort order (0 = ascending, 1 = descending).
    Default sort: by Compresa, Particella, n. area, n. albero,
    pollone (natural reading order).
  - `mt=o|t|s`: shared map type for every active map, including a free
    survey's tree map. Defaults to `s` (Satellite).

Map center and zoom are *not* encoded. Grid/sample-area maps fit their active
grid, while a free survey's tree map fits its parcels. Pan/zoom is treated as
transient view state. This keeps URLs short and avoids the "shared link
reproduces a stale viewport" trap.

The active grid (`g=`) and active survey (`s=`) are independent
controls: they can point at different grids without contradiction.
The tree table depends on `s=` and its visible `tf` filter; `g=` does not
narrow it. A structured area selection is represented by normal `tf` terms.

## Data tables

Six digests serve this page. Five are eagerly loaded on first navigation
(small, drive the always-visible controls and maps); one is lazily loaded per
active survey when Rilevamenti opens.

### Cache invalidation

| Write | Digests marked stale | Optimistic client patch |
|---|---|---|
| Grid save (create) | `grids`, `audit` | `grids` via `patches` |
| Grid edit (rename) | `grids`, `audit` | `grids` via `patches`; pulldown option text updated |
| Grid delete | `grids`, `sample_areas`, `audit` | `grids` (row removed) |
| Grid CSV import (areas) | `grids`, `sample_areas`, `surveys`, `audit` | All three via `patches` |
| Survey save (create) | `surveys`, `grids`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | `surveys` + `grids` patched; full aggregate caches evicted |
| Survey edit (rename) | `surveys`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | `surveys` patched; full aggregate caches evicted |
| Survey delete | `sampled_trees_<id>`, `samples`, `surveys`, `grids`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | Survey row deleted; full aggregate caches evicted and cascade data refreshed |
| Survey CSV import (trees) | `sampled_trees_<id>`, `samples`, `surveys`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `preserved_trees`, `audit` | Bulk tree/sample/survey caches force-refreshed; `parcels` evicted |
| Area save (create/update) | `sample_areas`, `grids`, `surveys`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `audit` | Area/grid/survey rows patched; `parcels` evicted; maps re-rendered if affected |
| Area delete | `sample_areas`, `grids`, `surveys`, `audit` | Area row deleted; grid/survey summary rows patched |
| Tree save (create/update) | every `sampled_trees_<id>` containing the shared physical tree, `samples`, `surveys`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `preserved_trees`, `audit` | Current survey tree/sibling rows, sample, survey, and current PAI rows patched; other affected survey caches and `parcels` evicted |
| Tree delete | `sampled_trees_<id>`, `samples`, `surveys`, `parcels`, `parcel_dendrometry`, `parcel_dendrometry_points`, `preserved_trees`, `audit` | Same direct row/summary patches as tree save; removed PAI row deleted or replaced; `parcels` evicted |
| Species save (Impostazioni) | all existing `sampled_trees_*` and `mark_trees_*`, shared species/Bosco/Prelievi digests, `audit` | Both dynamic client-cache families evicted |
| Diameter-class setting save (Impostazioni) | all existing `sampled_trees_*` and `mark_trees_*`, `parcel_dendrometry`, `audit` | All affected chart-cache families evicted; lazy regeneration on next open |

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

For structured surveys, `N. aree visitate` is
`COUNT(DISTINCT sample.sample_area_id WHERE survey_id = this.id)` and
`N. aree totali` is the size of the survey's grid
(`COUNT(sample_area WHERE sample_grid_id = survey.sample_grid_id)`). For
unstructured surveys, `N. aree visitate` is the number of dated samples and
`N. aree totali` is `0`. Grid name is looked up client-side via `grids.json`;
`Griglia` is null for unstructured surveys.

### `sample_areas.json`

All sample-area rows across all grids.  Eager-loaded; the client
filters by active grid (Section 1) or by the active survey's grid
(Section 2).  Invalidated on `sample_area` writes.

Columns: `row_id`, `version`, `Griglia` (sample_grid_id), `Compresa`,
`Particella`, `Numero`, `Lat`, `Lon`, `Alt. (m)`, `Raggio (m)`, `Note`.

### `samples.json`

All `sample` rows. Drives Section 2's visited/unvisited map coloring and the
per-area "n. alberi" hover tooltip. Eager-loaded. Invalidated on `sample` writes
and on `tree_sample` writes (since `N. alberi` is materialized).

Columns: `row_id`, `version`, `Survey`, `Sample area`, `Data`,
`N. alberi`. `Sample area` is null for unstructured samples.

`N. alberi` is `COUNT(tree_sample WHERE sample_id = this.id)`,
materialized so the Section 2 hover tooltip can render without
loading the heavy per-survey tree digest.

### `sampled_trees_<survey_id>.json` (lazy, per survey)

One digest per survey, denormalized for its tree table.  Lazily fetched
the first time a given `s=<id>` enters scope (either via URL on page
entry or via Section 2 pulldown change), and cached client-side per
survey for the rest of the session. Invalidated on relevant `tree_sample`
writes, on species-name changes, and on diameter-class setting changes. A
shared physical-tree edit invalidates every survey that contains that tree;
the current survey receives sibling-row patches while other affected client
caches are evicted.

Columns: `row_id`, `version`, `Sample area`, `Data campione`,
`Compresa`, `Particella`, `N. area`, `N. albero`, `Specie`, `Tipo`,
`Coppice`, `Pollone`, `Matricina`, `D (cm)`, `h (m)`, `L10 (mm)`, `V (m³)`,
`m (q)`, `PAI`, `Lat`, `Lon`, `Acc. (m)`. Structured rows derive parcel/area
display from the sample area, with row coordinates falling back to area
coordinates.
Unstructured rows derive parcel display and coordinates from the `TreeSample`
row, with `Sample area` null and `N. area` blank. Sort: by `Compresa`,
`Particella`, `N. area`, `N. albero`, `Pollone`.

Top-level `diameter_class_mode` metadata tells the shared
`TreeDetail` chart controller which convention to apply. After a setting save,
the next open of an evicted survey digest regenerates it on demand.

`row_id` = `tree_sample.id` (the synthetic id, see `database.md`).
`Coppice` is the stable boolean copied from `tree.coppice`; `Tipo` is the
localized display label derived from the same value.
`V (m³)` and `m (q)` are NULL for ceduo rows (per `database.md`
invariant).  `Lat`, `Lon` come from row-level `TreeSample` coordinates
if set, else fall back to the sample-area center. `Acc. (m)` is the nullable
row-level GPS accuracy and has no sample-area fallback; the UI displays null
as `-`. `Data campione` is `sample.date` (useful for cross-tab tracking even
when the table isn't narrowed to a single area).

The page uses this digest for the survey tree table and client-side
dendrometric summary in both survey types, plus the tree map for free surveys.
The table's Esporta submits the row IDs retained by its filters. The server renders normalized database
rows, preserving hidden fields, free-sample groups and PAI numbers. With no
area or text filter this exports the entire survey. The separate survey-delete
backup always exports all measurements, regardless of the active filters.
The dendrometry ZIP endpoint
(`/api/campionamenti/survey/dendrometry/export/<survey-id>/`) uses the same
shared server-side matrix builder as Martellate and accepts the visible row IDs
when the text filter is active.

Bosco's per-parcel Dendrometria charts read a different digest
(`parcel_dendrometry.json` — pre-aggregated, not per-tree); they do
not consume `sampled_trees_<survey_id>.json`.

### `species.json`

Shared with Piano di taglio (mark form).  Small lookup digest
(`row_id`, `version`, `Nome`, `Nome latino`, `Densità (q/m³)`,
`Pressler`, `Sort order`, `Attivo`, `Minore`, `Tipo`) used by:
- the live V/m preview in this page's manual entry form;
- the live V/m preview in Piano di taglio's mark form;
- digest-generation joins for `marks.json` and any other materialized
  mass total derived from V × density.
Invalidated on `species` writes.
Settings uses the same column/record builder for its table and write responses,
so optimistic patches cannot corrupt the shared species lookup cache. Settings
hides version and sort order and renders Tipo (a localized botanical-type
label, blank when unclassified) as a dropdown, or read-only text for catch-alls.

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
