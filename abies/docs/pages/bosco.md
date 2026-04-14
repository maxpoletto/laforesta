# Bosco page

The "bosco" page is the eventual landing page of Abies. It is close
to a clone of the Boscoscopio app, with the addition of data from other Bosco
apps ("aree di saggio", "piante ad accrescimento indefinito") and bookmarkable
URLs.

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

The page layout is as described under "Maps" in CLAUDE.md. The lower part of
the control panel has radio buttons titled:

- Caratteristiche (default)
- Evoluzione
- Aree di saggio
- Piante ad accrescimento indefinito

When each button is selected, the rest of the control panel looks as follows:

- Caratteristiche

  Panel contains pull-down with the same list of features as those in the
  "Visualizza caratteristiche" part of Boscoscopio. The behavior is identical.
  Below the pulldown is a checkbox for "Aree catastali".

  Map shows heatmap per pixel or per parcel, depending on type of data and
  whether "media per particella" is checked, identically to Boscoscopio.
  Color range is yellow-green (low values) to dark-green (high values), except for raw normalized satellite data (0 = black -> 1 = white).

- Evoluzione

  Panel contains pull-down with the same list of features as in the "Visualizza differenze" part of Boscoscopio, plus pull-downs for two dates (years, or year-months) to compare. The behavior is identical to Boscoscopio with "limita al bosco" always set to true.

  Below the pull-downs are checkboxes for "media per particella" and "aree catastali".

  Map shows red-to-green heatmap showing (new - old) values per pixel, or
  average diffs per parcel if "media per particella" is selected. High values
  map to dark green, low values map to dark red, white in the middle.

- Aree di saggio

  Panel contains scrollable list of  parcels for the current region, identical
  to bosco/ads (but for only the current region, not all regions). There is a
  checkbox to the left of each parcel name, and the number of contained sample
  areas in parentheses to the right. Below the panel of parcels, there are
  "mostra tutte" and "nascondi tutte" buttons.

  Map displays parcel borders and yellow dots corresponding to the sample areas.

- Piante ad accrescimento indefinito

  Shows two scrollable lists, like bosco/pai. Top list is parcels in the current
  region, identical to "aree di saggio" above. Lower list is species to display.
  Each species has a checkbox (to display or not, a color-coded dot, a name, and
  a count in parentheses, identically to bosco/pai.) Both lists have "mostra
  tutte" and "nascondi tutte" buttons with the obvious semantics.

  Map displays parcel borders and colored dots corresponding to the trees.

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

Statistical data:
- parcels.json: JSON version of the parcel table (columns TBD)
- sample_areas.json: JSON version of the sample_area table
- preserved_trees.json: JSON version of the preserved_tree table
- parcel_year_production.json: a digest that conceptually is a "SELECT region,
  parcel, year, SUM(quintals) FROM harvest_op GROUP BY region, parcel, year", organized like the timeseries.json files in Boscoscopio.

Map data:
- particelle.geojson as in Boscoscopio
- satellite data as in Boscoscopio
