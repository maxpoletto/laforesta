# Bosco page

## Overview

The "bosco" page is the eventual landing page of Abies. It is derived from the
Boscoscopio app, but with substantial additional data.

At a high level, the page provides insight into the state of the forest, in the
past, present, and future. It provides a map view of the forest. In some cases,
this view serves to display data directly (e.g., a heatmap of productivity per
region). In others cases, it serves as a graphical mechanism for the user to
select a region or parcel of the forest about which they want to learn more
information (e.g., select a parcel to view dendrometric information about it).

Bosco displays several kinds of information:

1. Parcel metadata

    - Location, altitude, exposure, surface area (cadastral and computed), etc.
    - Info about geology
    - Info about vegetation

2. Dendrometric information by diameter class and species.

    - Tree count: stacked bar chart
    - Tree volume: stacked bar chart
    - Tree basimetric area: stacked bar chart
    - Average height: scatter plot
    - % increment: line graph

    For stacked bar charts, different species are represented by different
    stacked bar areas. For scatter plots and line graphs, they are
    different-colored dots or lines.

3. Historical production

    - By parcel
    - By region

4. Preserved trees (PAI)

5. Marks ("martellate")

    - Per-mark-operation metadata
    - Individual trees

6. Satellite data

Sample areas and individual sampled trees are not displayed here — that view
lives on the Campionamenti page.

## URL structure

- Path: `/bosco`
- Query parameters are grouped below by purpose.  See "Query parameter
  details" further down for per-mode specifics.

### Map and page-level controls

- `mt=o|t|s` — map type (OSM, Topo, Satellite).  Default: `s`.
- `mc=NN.NNNNNN,NN.NNNNNN` — map center (lat,lng).  Absent → fits the
  active region's bounding box.
- `mz=NN` — map zoom.  Absent → fits the active region.
- `c=N` — id of the active region.  Effectively always present
  (every page state is scoped to one region; see also CLAUDE.md
  "Maps").  Stale `c=N` (deleted region) falls back to the first
  region by name.
- `s=N` — active survey id (page-level filter for sample-derived
  views — see the *Survey* pulldown above).  Default per the rule
  documented in "Visual appearance": most recent survey whose grid
  touches the active region, falling through to most-recent-overall.
  Stale `s=N` (deleted survey) re-applies the default.
- `m=1|2|3` — mode: 1 = Caratteristiche (default), 2 = Evoluzione,
  3 = Piante ad accrescimento indefinito.

### Per-parcel / per-region overlay

- `v=1|2` — overlay open: `1` = per-parcel, `2` = per-region.
  Absent → no overlay (the default map view).
- `pa=N` — id of the parcel for the per-parcel overlay; required if
  `v=1`.  Same convention as Prelievi (see `prelievi.md`).
- `vo=<tokens>` — open sections within the overlay.  Single-char
  tokens, order irrelevant.  Tokens: `m` = Metadati, `d` =
  Dendrometria, `p` = Produzione storica, `o` = Operazioni recenti.
  Absent → default (`m`).  Explicit empty (`?vo=`) → all closed.
- `dm=N` — Dendrometria metric: 1 = numero alberi, 2 = volume,
  3 = area basimetrica, 4 = altezza media, 5 = incremento.
  Default: 1.
- `ds=<id list>` — Dendrometria visible-species filter:
  comma-separated `species.id` values (e.g., `3,5,11`).  Absent →
  all species visible.

When the overlay is closed, all `v=`/`pa=`/`vo=`/`dm=`/`ds=`
params are stripped from the URL — closing returns the user to a
clean map state.  Reopening uses the documented defaults.

### Mode-specific parameters

See "Query parameter details" below for the full list per mode.
Cross-mode summary:
- Caratteristiche (`m=1`): `q=` metric id, `fc=` cadastral flag.
- Evoluzione (`m=2`): `q=`, `d1=`/`d2=`, `fa=`, `fc=`.
- PAI (`m=3`): `pp=` parcels list, `ps=` species list.

### Cross-page links into Bosco

- From Campionamenti: `/bosco?c=<region_id>&v=1&pa=<parcel_id>` to
  land on a per-parcel overlay; `s=` carries through if specified.
- From Piano di taglio: same pattern when a particella cell is
  clicked.
- From Prelievi: same pattern.

### Cross-page links out of Bosco (per-parcel overlay)

These are the links from the per-parcel/per-region page sections:

- *Produzione storica* row → `/prelievi?c=<region_id>&pa=<parcel_id>`
  pre-filtered to the harvest operation.
- *Operazioni recenti* → respective page (`/campionamenti?s=…`,
  `/piano-di-taglio?p=…&mk=…`, `/prelievi?c=…&pa=…`) pre-filtered
  to the relevant entity.

## Visual appearance

The page layout is as described under "Maps" in `CLAUDE.md`.  The map
fills the viewport; the navbar on the right hosts (top to bottom):

1. The standard map nav header (status panel, OSM/Topo/Satellite
   selector, region pulldown — see "Maps" in `CLAUDE.md`).
2. **Survey** ("Rilevamenti") pulldown — selects the active `survey` (see
   `campionamenti.md`).  This is a page-level filter that drives every view
   reading sample-derived data: the per-parcel page's *Dendrometria* section and
   the dendrometric entries in the *Caratteristiche* metric pulldown.  Default
   selection is the most recent survey *whose grid touches the currently
   selected region*; if the user later switches region, the survey selection is
   re-evaluated against the new region the same way (and updated only if the
   prior choice doesn't touch the new region — preserving an explicit override
   when it still applies).  If no survey's grid touches the region, falls
   through to the most recent survey overall, and dendrometric views show an
   empty state ("Nessun dato per questa selezione").  Modes that do not read
   sample data (*Evoluzione*, *PAI*) ignore the selector but it remains visible
   for stability.
3. **Mode** radio group (this page's primary control):
   - Caratteristiche (default)
   - Evoluzione
   - Piante ad accrescimento indefinito
4. Per-mode controls (see below).

Sample-area visualization (list of parcels with sample-area counts,
visited-vs-unvisited coloring, click-to-drill-in) lives on the Campionamenti
page, not here.

### Hover and click on the map

- **Hover** on a parcel: a small unobtrusive label shows the parcel
  name (e.g., `11`, `2a`).  No additional stats are shown; the goal is
  fast orientation while sweeping.
- **Click** on a parcel: opens the **per-parcel page** as a
  full-screen overlay above the map (see "Per-parcel page" below).
  Closing the overlay restores the map exactly as it was.
- **Click** on empty map space (no parcel underneath): opens the
  **per-region page** for the currently selected region (the same
  layout as the per-parcel page, scoped one level up — see below).
  This is the only entry point to the per-region view, so the gesture
  is documented unobtrusively in the sidebar (small-print hint under
  the region pulldown: "Clicca su una particella per i dettagli, o
  fuori dalle particelle per il riepilogo della regione").

### Mode panels

- **Caratteristiche**

  Pull-down with the list of features from the "Visualizza
  caratteristiche" part of Boscoscopio (parcel-level metadata and
  satellite-derived metrics).  Below it, a checkbox for "Aree
  catastali".

  Map paints a heatmap per pixel or per parcel depending on the
  feature and the "media per particella" toggle (when applicable),
  identically to Boscoscopio.  Color range is yellow-green (low) to
  dark-green (high), except for raw normalized satellite data
  (0 = black → 1 = white).

- **Evoluzione**

  Pull-down for the metric and two date pickers (years or
  year-months) to compare.  Behavior identical to Boscoscopio's
  "Visualizza differenze" with "limita al bosco" always true.

  Below the pickers, checkboxes for "media per particella" and "aree
  catastali".

  Map shows red-to-green diff heatmap (new − old): high → dark
  green, low → dark red, white at zero.

- **Piante ad accrescimento indefinito**

  Two scrollable lists (as in `bosco/pai`).  Top: parcels of the
  current region (with PAI count in parens).  Bottom: species
  (color-coded dot, name, count).  "mostra tutte" / "nascondi tutte"
  buttons under each list.

  Map shows parcel borders and per-tree dots colored by species.

  Writers see a "+ Aggiungi PAI" button below the species list.
  Clicking it opens a form: specie, lat/lng (shared lat-lng
  component — manual entry plus an optional "Usa posizione attuale"
  button when device geolocation is available), anno (defaults to
  current year).  The parcel is auto-derived from the geometry on
  save; if the lat/lng falls outside any parcel, the form prompts
  the user to pick one explicitly.

  Click on an existing PAI dot opens a popover with species, year,
  parcel, lat/lng; writers see a pencil and garbage icon.

## Per-parcel and per-region page

Reached by clicking a parcel on the map (per-parcel) or by clicking
empty map space outside any parcel (per-region; uses the currently
selected region from the pulldown).  Rendered as a full-screen
overlay above the map; dismissable with Escape, the back button, or
a close button at the top right.  The page is bookmarkable (the URL
encodes the scope) so it can be shared directly.

Both scopes use the same layout — the per-region view is useful
because many dendrometric stats (volume/ha, basimetric area/ha,
increment) are more meaningful at region level than at parcel level.
A small breadcrumb at the top of the page indicates scope (regione |
regione + particella); from a per-parcel view the breadcrumb's
"regione" segment is clickable and opens the per-region overlay for
the parcel's region.

The page is a single scrollable column of collapsible sections,
following the standard Abies idiom:

1. **Metadati** (open by default)
   - Location, altitude min/max, esposizione, pendenza.
   - Surface area: cadastral and computed.
   - Età media, classe economica, tipo (alto fusto / ceduo).
   - Descrizione vegetazione (free text, multi-paragraph).
   - Descrizione geologia (free text, multi-paragraph).
   
   Writers see a pencil icon next to each editable field that flips
   it into an inline editor.  Save commits via the standard
   form-intercept path; cancel reverts.

2. **Dendrometria** (closed by default)
   - Metric pulldown: numero alberi / volume / area basimetrica /
     altezza media / incremento.
   - Species filter: checkbox list of species observed in this
     scope, with "mostra tutte" / "nascondi tutte".
   - Survey scope: the active survey from the global sidebar
     selector.  This section does not have its own survey control —
     changing surveys is a page-level action, not a per-parcel one,
     to keep behavior predictable as the user navigates between
     parcels.  At region scope the chart aggregates across all
     parcels of the region using the active survey's samples.  When
     the active survey's grid does not touch this scope, the section
     shows an empty state and points the user to the sidebar
     selector.
   - Chart: one of
     - Stacked bar (count / volume / basimetric area): x = classe
       diametrica, y = metric, stacks = species.
     - Scatter (altezza media): x = classe diametrica, y = metric,
       points colored by species.
     - Line (incremento %): x = classe diametrica, y = metric, lines
       colored by species.
   - All values per ettaro, extrapolated from sample areas.
   - Diameter-class binning: TBD (likely 5 cm bins from 5 cm).

3. **Produzione storica** (closed by default)
   - Stacked bar chart of yearly q.li harvested in this scope, with
     a per-chart pulldown to break down by specie / prodotto /
     squadra.
   - Below the chart, a small sortable-table of individual harvest
     operations in this scope, with a link per row to Prelievi
     pre-filtered to that operation.

4. **Operazioni recenti** (closed by default)
   - Three sub-lists: ultimi rilevamenti (surveys touching this
     scope), ultime martellate (marks), ultimi prelievi (harvests).
     Each entry is a one-line summary with a link to the relevant
     page (Campionamenti, Piano di taglio, Prelievi) pre-filtered.
   - This is the connective tissue between Bosco and the
     operations pages.

Sections render lazily on first expand.

## Query parameter details

### Caratteristiche (`m=1`)

- `q=1`–`14` — metric id, matching the entries in the
  "caratteristica" pulldown (parcel-level metadata + satellite-derived
  metrics; same set as Boscoscopio).
- `fc=1` — "aree catastali" checked.

### Evoluzione (`m=2`)

- `q=1`–`4` — metric id, matching the entries in the
  "metrica" pulldown.
- `d1=YYYYMMDD`, `d2=YYYYMMDD` — start and end dates of the
  comparison.  Year granularity uses `YYYY0101`; month granularity
  uses `YYYYMM01`.
- `fa=1` — "media per particella" checked.
- `fc=1` — "aree catastali" checked.

### Piante ad accrescimento indefinito (`m=3`)

- `pp=<id list>` — comma-separated `parcel.id` values within the
  active region (e.g., `12,13,17,42`).  Absent → all parcels in the
  region visible.
- `ps=<id list>` — comma-separated `species.id` values (e.g.,
  `3,5,11`).  Absent → all species visible.

### Renamed parameters

`pp=` and `ps=` replace the earlier `p=` and `s=` from the
Boscoscopio-era spec.  `s=` is now the page-level survey selector
(matching Campionamenti's convention), and `p=` is reserved (it has
plan-id semantics on Piano di taglio; here it's free but kept
unused to avoid cross-page confusion).

All entity references in URLs use integer ids (`c=`, `pa=`, `pp=`,
`ps=`, `ds=`), not names.  Names are unstable under rename; ids
keep shared/bookmarked URLs working.  The lookups are cheap because
the relevant digests (`parcels.json`, `species.json`, etc.) are
already loaded by the time the URL is parsed.

## Data tables

Bosco mixes Bosco-specific digests with several it shares with other
pages.  Loading is staged: a small set is eager on page entry to
render the map immediately; the rest is lazy on overlay-open or
mode-switch.

### Eager-loaded (page entry)

- **`parcels.json`** — denormalized parcel table.  Drives the
  Caratteristiche heatmap (parcel-level metrics) and the per-parcel
  page's *Metadati* section.  Invalidated on `parcel` writes.

  Columns: `row_id`, `version`, `Compresa`, `Particella`, `Classe`,
  `Area (ha)`, `Area cat. (ha)`, `Età media`, `Località`, `Alt. min`,
  `Alt. max`, `Esposizione`, `Pendenza %`, `Tipo` (alto fusto / ceduo),
  `Desc. veg.`, `Desc. geo`.  `row_id` = `parcel.id`; `Tipo` derived
  from `parcel.eclass.coppice`.

- **`surveys.json`** — shared with Campionamenti.  Needed eagerly to
  populate the *Survey* pulldown and to compute the default-survey
  selection rule (most recent survey whose grid touches the active
  region).

- **`sample_areas.json`** — shared with Campionamenti.  Needed
  eagerly to evaluate "does this survey's grid touch this region?"
  for the default-survey rule.  Not used to draw anything on the
  Bosco map.

- **`species.json`** — shared with Campionamenti and Piano di taglio.
  Used here for color-coded species swatches in the PAI mode lists
  and per-parcel charts.

### Lazy on PAI mode (m=3)

- **`preserved_trees.json`** — `tree` rows with `preserved=true`,
  denormalized.  Invalidated on `tree` writes (specifically those
  toggling `preserved` or moving location).

  Columns: `row_id`, `version`, `Compresa`, `Particella`, `Specie`,
  `Anno`, `Lat`, `Lng`, `Note`.  `row_id` = `tree.id`.  Sorted by
  `(Compresa, Particella, Specie, Anno)`.

  Drives both the species/parcel scrollable lists and the per-tree
  dot map.  Filtered client-side by `pp=` / `ps=`.

### Lazy on per-parcel/per-region overlay open

- **`parcel_dendrometry.json`** — per-(parcel, survey, species,
  classe diametrica) aggregated stats.  Invalidated on `tree_sample`
  writes within any of the involved surveys.

  Columns: `row_id`, `version`, `Parcel`, `Survey`, `Specie`,
  `Classe (cm)`, `N. alberi/ha`, `Volume (m³/ha)`, `Area bas. (m²/ha)`,
  `Altezza media (m)`, `Incremento %`.  Sorted by `(Parcel, Survey,
  Specie, Classe)`.

  Per-hectare values are computed at digest-generation time by
  scaling sample-area observations by the surveyed area's coverage
  of the parcel.  Region-scope aggregation is done client-side by
  summing across the region's parcels (with appropriate per-ha
  arithmetic).  Diameter-class binning: 5 cm bins starting at 5 cm
  (TBD if pdg-2026 settles on different bins).

  `row_id` is a sequential synthetic index; this digest is read-only
  and does not participate in the standard edit/cache-update flow.
  Same convention applies to the other aggregation digests below.

- **`prelievi.json`** — shared with Prelievi (already documented).
  Drives both the per-parcel page's *Produzione storica* stacked
  bar chart *and* the small "individual harvest operations in this
  scope" sortable-table beneath it.  One row per harvest operation,
  with per-species quintal columns and `Tipo` / `Squadra` group-by
  columns already present, so the three breakdown options
  (specie / prodotto / squadra) all derive client-side via
  filter-by-parcel + group-by-year + sum.  No additional digest
  needed for the chart.

- **`parcel_year_production.json`** — shared with Prelievi and
  Piano di taglio (already documented under `prelievi.md`).  Not
  read by the per-parcel chart (covered by `prelievi.json` above);
  consumed elsewhere on Bosco where a single Q.li / m³ rollup
  per (region, parcel, year) is enough.

- **`marks.json`** — shared with Piano di taglio.  Drives the
  *Operazioni recenti* "ultime martellate" sub-list, filtered to
  marks touching this scope.

### Map data

- **`particelle.geojson`** — parcel polygons, as in Boscoscopio.
  Eager-loaded.  Updated only on geometry-data refresh (rare).
- **Satellite data** — preprocessed GeoTIFFs and per-pixel digests as
  in Boscoscopio.  Lazy per (metric, date) tuple.

### Notes on aggregation digests

Digests like `parcel_dendrometry.json`,
`parcel_year_production.json`, and
`parcel_year_production_breakdown.json` are read-only aggregations:
no per-row writes, no edit forms.  They follow the standard
`{columns, rows}` shape with a synthetic `row_id` (sequential 1, 2,
3, …) so the cache layer treats them uniformly, but row_id has no
semantic meaning and is not used for cache updates.

A schema change that affects an aggregation's inputs marks the
aggregation digest stale via the same staleness-flag pattern used
for tabular digests (see CLAUDE.md "JSON digest regeneration").
