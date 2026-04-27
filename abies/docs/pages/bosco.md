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

4. Sampled trees

    - Sample areas
    - Trees within a sample area
    - Preserved trees

5. Marks ("martellate")

    - Per-mark-operation metadata
    - Individual trees

6. Satellite data

## URL structure

TODO: Needs to be updated.

- Path: /abies/bosco
- Query parameters:
  - Map type (mt={o,t,s} (OSM, Topo, Sat))
  - Map center (Lat/lng) (mc=NN.NNNNNN,NN.NNNNNN)
  - Map zoom level (mz=NN)
  - Region (c={[region name]})
  - Mode (m=N): denotes display mode (see below)
  - Parameter (q=N): denotes parameter to display (see below)
  - Parcels (p=...): denotes parcels to display (see below)
  - Species (s=...): denotes species to display (see below)
  - Dates (d1=YYYYMMDD, d2=YYYYMMDD): used for historical comparisons (see below)
  - Boolean flags (false if not present):
    - Cadastral area (fc): whether to display cadastral or computed area ("Aree
      catastali" in Boscoscopio)
    - Parcel averages (fa): whether satellite values should be rendered
      per-satellite-pixel or average per parcel ("medie per particella" in
      Boscoscopio)

## Visual appearance

The page layout is as described under "Maps" in `CLAUDE.md`.  The map
fills the viewport; the navbar on the right hosts (top to bottom):

1. The standard map nav header (status panel, OSM/Topo/Satellite
   selector, region pulldown — see "Maps" in `CLAUDE.md`).
2. **Mode** radio group (this page's primary control):
   - Caratteristiche (default)
   - Evoluzione
   - Aree di saggio
   - Piante ad accrescimento indefinito
3. Per-mode controls (see below).

### Hover and click on the map

- **Hover** on a parcel: a small unobtrusive label shows the parcel
  name (e.g., `11`, `2a`).  No additional stats are shown; the goal is
  fast orientation while sweeping.
- **Click** on a parcel: opens the **per-parcel page** as a
  full-screen overlay above the map (see "Per-parcel page" below).
  Closing the overlay restores the map exactly as it was.
- **Click** on empty map space: dismisses any selection.

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

- **Aree di saggio**

  Scrollable list of parcels in the current region, with a checkbox
  per parcel and the number of contained sample areas in
  parentheses.  Below the list, "mostra tutte" / "nascondi tutte"
  buttons.

  Map displays parcel borders and yellow circles drawn at each
  sample area's actual `r_m` radius (centered at its lat/lng).
  Per-tree dots are *not* drawn — that view lives on the
  Campionamenti page.

  Click on a sample-area circle opens a small popover with the
  area's number, parcel, group label, and a "vai a Campionamenti"
  link that pre-filters Campionamenti to that area.

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

## Per-parcel page

Reached by clicking a parcel on the map.  Rendered as a full-screen
overlay above the map; dismissable with Escape, the back button, or a
close button at the top right.  The page is bookmarkable (the URL
encodes the parcel id) so it can be shared directly.

The same layout serves a per-region view, reached by clicking the
"info regione" button next to the region pulldown — useful because
many dendrometric stats (volume/ha, basimetric area/ha, increment)
are more meaningful at region level than at parcel level.  A small
breadcrumb at the top of the page indicates scope (regione | regione
+ particella) and lets the user toggle up to the region-aggregate
view from a parcel page.

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
   - Sample/year selector: which sampling session to display
     (defaults to the most recent).  At region scope, aggregates
     across all parcels of the region for the chosen session.
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
   - Three sub-lists: ultimi rilievi (samples), ultime martellate
     (marks), ultimi prelievi (harvests).  Each entry is a one-line
     summary with a link to the relevant page (Campionamenti,
     Piano di taglio, Prelievi) pre-filtered.
   - This is the connective tissue between Bosco and the
     operations pages.

Sections render lazily on first expand.

## Query parameter details

- Caratteristiche
  - m=1
  - q=1-14 (corresponding to the entries in the pulldown menu)
  - fc=1 if "aree catastali" is checked

- Evoluzione
  - m=2
  - q=1-4 (corresponding to the entries in the pulldown menu)
  - d1=YYYYMMDD,d2=YYYYMMDD (start and end dates of comparison).

    If the granularity of data is year, then dates are of the form YYYY0101.
    If the granularity is month, then the dates are of the form YYYYMM01.

  - fa=1 if "media per particella" is checked.
  - fc=1 if "aree catastali" is checked.

- Aree di saggio
  - m=3
  - p=[comma-separated list of parcels (e.g., "1,2,4a,4b,14,15c")]

- Piante ad accrescimento indefinito
  - m=4
  - p=[comma-separated list of parcels (e.g., "1,2,4a,4b,14,15c")]
  - s=[comma-separated list of down-cased species names, with spaces replaced by
    underscores, e.g., "abete_rosso,castagno,betulla_bianca"]

## Data tables

TBD — defer detailed digest design until UX is settled.  The current
sketch is:

Statistical data:
- `parcels.json`: JSON version of the parcel table (columns TBD).
- `sample_areas.json`: JSON version of the sample_area table.
- `preserved_trees.json`: digest of `tree` rows with `preserved=true`.
- `parcel_year_production.json`: per-parcel-per-year q.li totals,
  conceptually `SELECT region, parcel, year, SUM(quintals) FROM
  harvest GROUP BY region, parcel, year`, organized like the
  timeseries files in Boscoscopio.
- `parcel_dendrometry.json`: per-(parcel, session, species, classe
  diametrica) aggregated stats (count, volume, area basimetrica,
  altezza media, incremento) for the per-parcel page's dendrometry
  charts.  Region-aggregate values are computed client-side by
  summing.

Map data:
- `particelle.geojson` as in Boscoscopio.
- Satellite data as in Boscoscopio.
