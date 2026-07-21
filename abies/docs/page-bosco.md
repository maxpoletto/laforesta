# Bosco page

## Overview

The "bosco" page provides insight into forest health and productivity.

The main view is map-based. Some data (e.g., harvest amount by region) is
displayed on the map directly. For other data, the map serves as a parcel- or
region-selection surface and the app then displays a full-page modal with
information about that area.

Information provided:

1. Parcel metadata
2. Dendrometric information by diameter class and species.
3. Historical production by parcel and region
4. Preserved trees (PAI)
5. Satellite data

## Visual appearance

The page layout is a map with sidebar, as described under `Maps` in
`ui-design-patterns.md`. The map fills the viewport; the navbar on the right
hosts (top to bottom):

1. Standard map nav header (status panel, region pulldown).
2. **Mode** radio group (the page's primary control):
   - Caratteristiche (default)
   - Evoluzione
   - Piante ad accrescimento indefinito
3. Per-mode controls (see below).

### Hover and click on the map

- **Hover** on a parcel: a small unobtrusive label shows the region and parcel
  name (e.g., `Capistrano 11`, `Serra 2a`) and below it, the parcel type
  (Fustaia or Ceduo) and area (e.g., "Fustaia 24,3 ha", "Ceduo 7,9 ha").
  Uses same mechanism as Campionamenti maps. No additional details are shown;
  the goal is fast orientation while sweeping.
- In Caratteristiche and Evoluzione, **click** on a parcel opens the
  **per-parcel page** as a full-screen overlay above the map (see "Per-parcel
  page" below). Closing the overlay restores the map exactly as it was.
- In Caratteristiche and Evoluzione, **click** on empty map space (no parcel
  underneath) opens the **per-region page** for the currently selected region
  (the same layout as the per-parcel page, scoped one level up — see below).
  This is the only entry point to the per-region view, so the gesture is
  documented unobtrusively in the sidebar (small-print hint under the region
  pulldown: "Clicca su una particella per vederne i dettagli, o fuori dalle
  particelle per il riepilogo della compresa").
- In PAI mode, **click** on the map prompts writers to insert a new preserved
  tree at that coordinate; confirming opens the PAI add form with lat/lon
  prefilled, and with the parcel preselected when the click was inside one.

### Mode panels

- **Caratteristiche**

  Pull-down with a subset of the  features from the "Visualizza caratteristiche"
  part of Boscoscopio:
  - Caratteristiche
    - Età media (a)
    - Governo
    - Alt. media (m)
  - Produzione
    - Prelievo storico
    - Prelievo previsto
  - Dati satellitari
    - NDVI
    - NDMI
    - EVI

  Below the pullodown, a checkbox for "Aree catastali" (default unchecked, uses
  computed surface area from parcel geometries).

  If harvest volume ("prelievo") is shown, then we also display another checkbox,
  "Valori per ettaro" (default unchecked) to display normalized per-hectare
  productivity.

  The choice of harvest plan used in the "prelievo previsto" is configured on
  the settings page, see page-impostazioni.md.

- **Evoluzione**

  Pull-down for the metric and two date pickers. Satellite metrics use
  year-month choices and follow Boscoscopio's "Visualizza differenze" behavior
  with "limita al bosco" always true. Prelievo uses years for which historical
  production data exists and maps `(to-year q.li) - (from-year q.li)` per
  parcel.

  Below the pickers, checkboxes for "media per particella" and "aree catastali";
  Prelievo also shows "valori per ettaro". By default satellite maps render
  the detailed pixel heatmap; "media per particella" switches to the
  precomputed per-parcel means.

  Map shows red-to-green diff heatmap (new − old).

- **Piante ad accrescimento indefinito**

  Two scrollable lists (like in `bosco/pai`):
  - Top: parcels of the current region (with PAI count in parens).
  - Bottom: species (color-coded dot, name, count).

  Under each list are `Tutte` / `Nessuna` buttons.

  Map shows parcel borders and per-tree dots colored by species.

  Writers see a "+ Aggiungi" button below the species list. Clicking it opens a
  modal form: species, tree number, survey date, optional estimated birth year,
  parcel, diameter/height, lat/lon (shared lat-lon component — manual entry plus
  "Usa GPS" button when device geolocation is available), and free-text note.
  Species, number, date, diameter, height, and lat/lon cannot be empty:
  showFormError on submit if they are. Parcel is auto-derived (via parcel
  geometries) from lat/lon when user sets lat/lon, but user can edit it
  explicitly also.

  Click on an existing tree dot opens a popover with species, tree number,
  parcel, survey date, estimated birth year, diameter/height, coordinates, and
  note; writers see a pencil and garbage icon.

### Map conventions

When mapping parcel types, coppices are yellow, high forest green.

When mapping values in a positive range ([0, 1], > 0, etc.), color range is
yellow-green (low) to dark-green (high).

When mapping values in a range around 0 (e.g., differences in NDVI values, [-1,
1]), range is red (min) to white (0) to green (max). Ranges are normalized to be
symmetric around 0 ([a,b] --> [-max(abs(a),abs(b)), max(abs(a),abs(b))]).

Heatmaps are rendered per pixel, unless "media per particella" is selected, in
which case the average pixel value per parcel is computed, and that value is
applied to every pixel in the parcel (behavior identical to "Boscoscopio").

## Per-parcel and per-region page

Reached by clicking a parcel on the map (per-parcel) or by clicking
empty map space outside any parcel (per-region; uses the currently
selected region from the pulldown).  Rendered as a full-screen
overlay above the map; dismissable with Escape, the back button, or
a close button (x) at the top right.  The page is bookmarkable (the URL
encodes the scope) so it can be shared directly.

Both scopes use the same layout — the per-region view is useful
because many dendrometric stats (volume/ha, basimetric area/ha,
increment) are more meaningful at region level than at parcel level.
A small header at the top of the page indicates scope (regione |
regione + particella).

The page is a single scrollable column of collapsible sections,
following the standard Abies idiom:

1. **Caratteristiche** (open by default)
   - Location, altitude min/max, esposizione, pendenza.
   - Surface area: cadastral and computed.
   - Classe economica, tipo (alto fusto / ceduo).
   - For parcels only: età media (a).
   - Descrizione vegetazione ("soprassuolo") (free text, multi-paragraph).
   - Descrizione geologia ("stazione") (free text, multi-paragraph).
   
   Writers see a pencil icon next to each editable field that flips it into an
   inline editor with the standard annulla/salva buttons (see UI Design
   Patterns). Hitting Escape is equivalent to annulla.

2. **Dendrometria** (closed by default)
   - Species filter: a not very tall checkbox list of species to be displayed in
     the following graphs, with `Tutte` / `Nessuna` buttons.

   - Units checkbox for the  ("Per ettaro"): if checked, data in the quantity
     charts is per hectare, otherwise for the whole parcel/region. (Default checked.)

   - Quantity charts: Three stacked-bar charts (stacked by species), for three
     parameters, all with x-axis = diameter class (5 cm buckets centered on
     multiples of 5 (E.g., diamater class 20 means 17.5 < d <= 22.5)).
     - Tree count ("Numero alberi"), y-axis = number of trees
     - Tree total volume ("Volume totale"), y-axis = total volume
     - Tree basimetric area ("Area basimetrica"), y-axis = total basimetric area

   - Average height: scatter plot of individual trees x-axis = diameter, y-axis
     = height. Fit curves (with regression coefficients) plotted for species
     with more than N data points.

   - Percentage growth: line graph, one line per species, x-axis = diameter class, y-axis =
     Pressler percentage volume growth for diameter class.

   The choice of surveys used for this data is made in the settings page, see
   page-impostazioni.md. If no survey exists, displays a message to that effect.

   For every graph, displays the number of trees used to compute the data.

3. **Produzione storica** (closed by default)
   - Units checkbox as above: if checked, production is per hectare, otherwise
     per parcel/region.
   - Stacked bar chart of monthly/yearly q.li harvested in this scope (parcel or
     region), otherwise identical to the chart in Prelievi (see page-prelievi.md
     > Produzione).

Sections render lazily on first expand.

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
- `m=1|2|3` — mode: 1 = Caratteristiche (default), 2 = Evoluzione,
  3 = Piante ad accrescimento indefinito.

### Per-parcel / per-region overlay

- `v=1|2` — overlay open: `1` = per-parcel, `2` = per-region.
  Absent → no overlay (the default map view).
- `pa=N` — id of the parcel for the per-parcel overlay; required if
  `v=1`.  Same convention as Prelievi (see `prelievi.md`).
- `vo=<tokens>` — open sections within the overlay.  Single-char
  tokens, order irrelevant.  Tokens: `m` = Metadati (Caratteristiche),
  `d` = Dendrometria, `p` = Produzione storica,
  Absent → default (`m`).  Explicit empty (`?vo=`) → all closed.
- `ds=<id list>` — Dendrometria visible-species filter:
  comma-separated `species.id` values (e.g., `3,5,11`).  Absent →
  all species visible.

When the overlay is closed, all `v=`/`pa=`/`vo=`/`ds=`
params are stripped from the URL — closing returns the user to a
clean map state.  Reopening uses the documented defaults.

### Mode-specific parameters

See "Query parameter details" below for the full list per mode.
Cross-mode summary:
- Caratteristiche (`m=1`): `q=` metric id, `fa=` parcel-average satellite flag, `fc=` cadastral flag, `fh=` per-hectare harvest flag.
- Evoluzione (`m=2`): `q=`, `d1=`/`d2=`, `fa=`, `fc=`, `fh=`.
- PAI (`m=3`): `pp=` parcels list, `ps=` species list.

### Cross-page links into Bosco

- From Campionamenti: `/bosco?c=<region_id>&v=1&pa=<parcel_id>` to
  land on a per-parcel overlay; `s=` carries through if specified.
- From Piano di taglio: same pattern when a particella cell is
  clicked.
- From Prelievi: same pattern.

### Cross-page links out of Bosco (per-parcel overlay)

- *Produzione storica* row → `/prelievi?c=<region_id>&pa=<parcel_id>`
  pre-filtered to the harvest operation.

### Query parameter details

#### Caratteristiche (`m=1`)

- `q=1`–`8` — metric id, matching the entries in the
  "caratteristica" pulldown (parcel-level metadata + satellite-derived
  metrics; same set as Boscoscopio).
- `fc=1` — "aree catastali" checked.
- `fh=1` — "valori per ettaro" checked for harvest metrics (Caratteristiche) or `q=4` Prelievo (Evoluzione).
- `fa=1` — "media per particella" checked for satellite metrics. Unchecked satellite metrics render raster pixels; checked metrics render parcel averages from `timeseries.json`.

#### Evoluzione (`m=2`)

- `q=1`–`4` — metric id, matching the entries in the
  "parametro" pulldown.
- `d1=YYYYMMDD`, `d2=YYYYMMDD` for satellite metrics — start and end
  dates of the comparison. Year granularity uses `YYYY0101`; month granularity
  uses `YYYYMM01`.
- `d1=YYYY`, `d2=YYYY` for `q=4` Prelievo — historical production years.
  The rendered value is `(d2 q.li) - (d1 q.li)` for each parcel.
- `fa=1` — "media per particella" checked for satellite metrics. Unchecked satellite metrics render raster pixels; checked metrics render parcel averages from `timeseries.json`.
- `fc=1` — "aree catastali" checked.
- `fh=1` — "valori per ettaro" checked when `q=4` Prelievo.

#### Piante ad accrescimento indefinito (`m=3`)

- `pp=<id list>` — comma-separated `parcel.id` values within the
  active region (e.g., `12,13,17,42`).  Absent → all parcels in the
  region visible; explicit empty `pp=` → none visible.
- `ps=<id list>` — comma-separated `species.id` values (e.g.,
  `3,5,11`).  Absent → all species visible; explicit empty `ps=` → none visible.

#### Entity references

All entity references in URLs use integer ids (`c=`, `pa=`, `pp=`,
`ps=`, `ds=`), not names.  Names are unstable under rename; ids
keep shared/bookmarked URLs working.  The lookups are cheap because
the relevant digests (`parcels.json`, `species.json`, etc.) are
already loaded by the time the URL is parsed.

## Data tables

Bosco mixes page-specific digests with several it shares with other
pages. Loading is staged: a small set is eager on page entry to
render the map immediately; the rest is lazy on overlay-open or
mode-switch.

### Eager-loaded (page entry)

- **`parcels.json`** — denormalized parcel table.  Drives the
  Caratteristiche heatmap (parcel-level metrics) and the per-parcel
  page's *Metadati* section.  Invalidated on `parcel` writes.

  Columns: `row_id`, `version`, `Region id`, `Compresa`, `Particella`, `Classe`,
  `Coppice`, `Area (ha)`, `Area cat. (ha)`, `Età media (a)`, `Località`,
  `Alt. min. (m)`, `Alt. max. (m)`, `Esposizione`, `Pendenza (%)`, `Tipo`
  (alto fusto / ceduo), `Soprassuolo`, `Stazione`, `Piano dei tagli`,
  `Intervallo interventi`, `Matricine / ha`. `row_id` = `parcel.id`;
  `Coppice` is the stable boolean from `parcel.eclass.coppice`; `Tipo` is
  its localized display label.  `version` participates in optimistic locking
  for parcel metadata edits.

- **`species.json`** — shared with Campionamenti and Piano di taglio.
  Used here for color-coded species swatches in the PAI mode lists
  and per-parcel charts.

### Lazy on PAI mode (m=3)

- **`preserved_trees.json`** — current preserved/PAI tree rows, derived
  from the latest `tree_sample` row with a non-null `preserved_number` for
  each `(parcel_id, preserved_number)` identity. Invalidated on preserved-tree
  sample writes.

  Columns: `row_id`, `version`, `Tree id`, `Parcel id`, `Species id`,
  `Compresa`, `Particella`, `Specie`, `Numero`, `Data`,
  `Anno di nascita stimato`, `D (cm)`, `h (m)`, `h misurata`, `Lat`, `Lon`,
  `Note`. `h (m)` is blank for historical PAI rows whose height is unknown.
  `row_id` = `tree_sample.id`; `Numero` is the parcel-scoped
  `preserved_number`. Sorted by `(Compresa, Particella, Numero)`.

  Drives both the species/parcel scrollable lists and the per-tree
  dot map.  Filtered client-side by `pp=` / `ps=`.

### Lazy on per-parcel/per-region overlay open

- **`parcel_dendrometry.json`** — per-(parcel, survey, species, classe
  diametrica) aggregated stats.  Invalidated on `tree_sample` writes within any
  of the involved surveys, and on change of dendtrometry surveys in the settings
  page.

  Columns: `row_id`, `Parcel id`, `Survey id`, `Species id`,
  `Compresa`, `Particella`, `Rilevamento`, `Area saggi (ha)`, `Specie`,
  `Classe diam. (cm)`, `N. alberi`, `Volume (m³)`, `Area bas. (m²)`, `Altezza media (m)`,
  `Incremento %`.  Sorted by `(Compresa, Particella, Rilevamento, Specie,
  Classe diam. (cm))`.

  Tree count, volume, and basal area are expanded client-side by sampled-area
  coverage: per-hectare values divide observations by `Area saggi (ha)`, and
  whole-scope values multiply by selected parcel/region area over
  `Area saggi (ha)`. Region-scope aggregation is also done client-side by
  summing across the region's parcels.

  `row_id` is a sequential synthetic index; this digest is read-only
  and does not participate in the standard edit/cache-update flow.

- **`parcel_dendrometry_points.json`** — per-tree points for the Dendrometria
  section's diameter/height scatter plot. If the active computed hypsometric
  parameter set has "Usa questi rilevamenti per i grafici altezza/diametro"
  enabled, it uses that set's source surveys, otherwise it uses the same
  active-survey setting as `parcel_dendrometry.json`. It is invalidated when
  dendrometry settings or active hypsometric parameters change.

  Columns: `row_id`, `Parcel id`, `Survey id`, `Tree id`, `Species id`,
  `Compresa`, `Particella`, `Rilevamento`, `Specie`, `D (cm)`, `h (m)`.
  `row_id` = `tree_sample.id`.  Read-only.

- **`future_production.json`** — per-year production for high forest (fustaia)
  parcels based on the currently selected harvest plan. Invalidated on harvest
  plan edits (in Piani di Taglio) or when the selected harvest plan is changed
  in Impostazioni.

- **`prelievi.json`** — shared with Prelievi (already documented). Drives the
  per-parcel page's *Produzione storica* stacked bar chart, as in
  page-prelievi.md.

### Map data

- **`terreni.geojson`** — parcel polygons, as in Boscoscopio, enriched
  during `build_geo` with static parcel metadata such as
  `properties.coppice`.  Eager-loaded.  Updated only on geometry-data refresh
  (rare).
- **Satellite data** — per-compresa `manifest.json`, `timeseries.json`,
  raw index rasters, and `parcel-mask.tif`, served from `SATELLITE_DIR` via
  `/api/bosco/satellite/<region_id>/...`.  Built from `terreni.geojson` with
  `manage.py build_satellite`; satellite metrics in both Caratteristiche and
  Evoluzione render raster pixels by default and use the precomputed per-parcel
  means in `timeseries.json` when `fa=1`. Raster-mode overlays and
  tooltips are rendered in the browser from the same prefetched raw raster
  values.
