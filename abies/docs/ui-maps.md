# Map architecture

All abies maps are built from three layers, Leaflet at the bottom. Visual
rules (chrome layout, basemap appearance, parcel style) live in
[`ui-design-patterns.md`](ui-design-patterns.md) under *Maps*; this doc covers
the **code** structure — including the parcel data source, the hover-tooltip and
view-persistence mechanisms, and how the layers fit together.

## Layers

1. **`base/js/map-common.js` — the shim.**  A thin wrapper over Leaflet:
   `MapCommon.create(el, { basemap }) → wrapper`. It wires only the standard
   chrome every map shares — the basemap layer + switcher (top-right), the
   localized zoom control, and an optional coordinate readout — and exposes
   `getLeafletMap() / getBasemap() / setBasemap() / syncBasemap()`, firing
   `basemapchange` for user-initiated cross-map sync and
   `basemapstylechange` for every actual layer change; pages round-trip the
   active basemap through the `mt=` URL param (`o`/`t`/`s`). It also holds
   `BASEMAPS`. Nothing else: no measure, location, markers, or geometry.

2. **`base/js/map-tools.js` — opt-in tools.**  `attachMeasure(leaflet)`,
   `attachLocation(leaflet)`, and `attachSidebarToggle(leaflet, { sidebarId,
   mapId })`, each adding its control + handlers and returning a `{ destroy() }`
   teardown handle. A tool that captures map clicks also exposes
   `capturesClicks()` (the measure tool subscribes to the map's `click` to drop
   points); a host that *also* listens for clicks (ParcelMap) suppresses its own
   click handling while any tool reports `capturesClicks()`. Plus the pure,
   locale-aware `formatDistance(m)`. The tools know nothing about ParcelMap and
   take a raw Leaflet map.

3. **`base/js/parcel-map.js` — `ParcelMap`, the general map abstraction.**  Built
   on the shim. Renders parcel polygons (shared `PARCEL_STYLE` +
   `parcelTooltipContent` hover tooltips), frames the map (fit-to-parcels or a restored `initialView`)
   and reports view changes, exposes the unified click callback
   `onMapClick(latlng, feature|null)`, and opts into tools via `tools: { measure,
   location, sidebar }` (all off by default). It also provides a sample-area
   marker layer with active-highlight for the maps that need it. It also
   updates registered semantic marker layers when the basemap changes. Owns
   `PARCEL_STYLE` and `FIT_PADDING`.

```
map-common.js (shim) ─┐
map-tools.js  (tools) ─┴─► parcel-map.js (ParcelMap) ─► GriglieMap   (subclass)
geo.js  (geometry) ───────►                          ─► RilevamentiMap (subclass)
                                                     ─► GridPlanner  (composition)
```

## Using ParcelMap

Two patterns, picked by whether the thing *is* a map or *has* one:

- **Subclass** when your map is a parcel map with sample-area markers (Griglie,
  Rilevamenti). `class FooMap extends ParcelMap`; in the constructor call
  `super({ container, className, geojson, basemap, tools, initialView,
  onViewChange, onMapClick })`, adapting `onMapClick` to your page callbacks;
  override `setAreas(rows…)` to build markers via `_addAreaMarker(area, {
  fillColor, fillTone, fillOpacity, tooltip, onClick })`. Use `fillTone:
  'dark'|'light'` for green status markers that need the satellite contrast
  palette. The base owns marker
  bookkeeping, `setActiveAreaId` / active-highlight, `fitParcels`,
  `invalidateSize`, view persistence, the `.leaflet` / `.wrapper` handles used
  for basemap sync, and `destroy`.

- **Compose** when your component has a map but isn't one (GridPlanner). Create
  `new ParcelMap({ container, geojson, tools: {} })`, draw your own layers on
  `parcelMap.leaflet`, and call `parcelMap.destroy()` on teardown. You get the
  parcel surface + framing for free; the marker layer stays empty.

### Active-area selection (sample-area maps)

There are two write paths to `activeAreaId`, on purpose: a **marker click** sets
it directly and always re-fires the select callback (so re-clicking the active
marker still notifies the page), while the public **`setActiveAreaId(id)`** is
the page-driven path and is idempotent. Don't route the marker click through
`setActiveAreaId` — the early-out would swallow a repeat selection.

### Basemap-aware point colors

`map-palette.js` owns the two semantic point tones. Each marker keeps its
original green on OSM/topo and uses dark yellow or pale straw on satellite.
Build standalone vector options with
`semanticMarkerStyle(basemap, tone, style)`. In a `ParcelMap`, prefer
`parcelMap.semanticMarkerStyle(tone, style)` and register app-owned layer groups
with `registerSemanticMarkerLayer(layer)`; the built-in marker layer is already
registered. HTML `DivIcon` markers can use the `mc-basemap-satellite` class
which `MapCommon` maintains on the map container.

The palette update runs only on `basemapstylechange`. With the default Canvas
renderer, Leaflet coalesces the markers' `setStyle` calls into one redraw, so a
basemap switch costs one linear pass over semantic point layers; pan, zoom, and
filter updates incur no palette work. Do not use it for categorical colors
(species, forest type, or data scales), which must remain stable.

## Adding a new map

1. Decide subclass vs. compose (above).
2. Pass a parcel `geojson` sorted with `sortFeaturesByArea` (see *Parcel
   polygons*, below).
3. Enable only the tools you need. `tools` is opt-in: measure + location suit
   full field maps (Griglie, Rilevamenti, Bosco); a modal map like the grid
   planner passes `tools: {}`.
4. Persist/restore the view with `initialView` + `onViewChange` (see *View
   persistence across re-renders*, below).

## Parcel polygons

Source: `data/geo/terreni.geojson`. Each polygon feature carries
`properties.layer = "<Compresa>"` (e.g. `"Capistrano"`) and
`properties.name = "<Compresa>-<particella>"` (e.g. `"Capistrano-10a"`).
`build_geo` enriches those features after parcels are loaded, with static
DB-backed metadata such as `properties.coppice`. After fetching, sort with
`sortFeaturesByArea(geojson)` (`geo.js`) so smaller polygons render — and
tooltip — on top of the larger ones that contain them.

`ParcelMap` styles every polygon with the shared `PARCEL_STYLE` and binds the
hover tooltip automatically. `parcelLabel(feature)` (`geo.js`) stays pure and
returns `{title, type}` label data; `parcelTooltipContent(feature)`
(`parcel-map.js`) turns that data into the shared tooltip DOM. Raw features
render as `"<Compresa> <particella>"`; enriched features add a second line with
`Fustaia`/`Ceduo`. A standalone renderer that doesn't use `ParcelMap` can mount
the same tooltip with
`bindTooltip(parcelTooltipContent(f), { sticky: true, direction: 'top' })`.

## View persistence across re-renders

`ParcelMap` supports this via `initialView` + `onViewChange`: the page stashes
`savedView: { id, center, zoom }` on every `onViewChange` and passes it back as
`initialView` only when the id matches the current selection — so a data write
keeps the view, but switching grid/survey resets to `fitBounds`. See
`campionamenti.js` for the pattern.

## Geometry and i18n

Pure geometry/label helpers live in `base/js/geo.js` with **no** Leaflet
dependency: `sortFeaturesByArea`, `geodesicArea`, `featureArea`,
`geoJSONFeatureArea`, `pointInPolygon`, `findContainingParcel`,
`planGridForTarget`, `parcelLabel`, …

All map control labels are `strings.js` constants (`MAP_ZOOM_IN`,
`MAP_MEASURE_TITLE`, `MAP_LOCATION_*`, …); every displayed number goes through
`format.js` (e.g. `formatDistance` → `fmtDecimal*`), never `.toFixed()`, so
distances localize with the active locale's decimal separator.

## Distance metric

`geo.distanceMeters()` and Ipso upload inference delegate to the shared
`base/js/distance.js` numerical core: a sphere with WGS84 mean radius
6371009 m. The Ipso geometry bridge and offline precache include that core.
Inside/on/outside sample radii and nearest-area selection have shared metric
coverage; stored area IDs are preserved. Polygon geodesic area retains its
existing 6378137 m Leaflet-compatible convention.

Geometry labels accept region properties in precedence order `layer`,
`compresa`, `Compresa`, and parcel properties `name`, `particella`, `Particella`.
The first hyphen separates the existing full-name prefix from the parcel label.
`parcelIdentity()` exposes internal `{regionName, parcelName}` strings; the
legacy `parcelNames()` result remains compatible. Leading zeros and suffixes
are preserved. Python `parcel_key()` follows the same alias policy.

## Printable parcel maps

`bosco/js/characteristic-print.js` creates a standalone A4 vector layout from
parcel geometries and text labels. It uses a north-up Mercator projection,
fits the entire supplied area, and chooses portrait or landscape for the
largest legible map. It does not use Leaflet, tiles, contours, or raster images.
The layout is shared by the SVG preview and `base/js/pdf.js` PDF renderer;
both use the PDF font metrics. The PDF retains vector paths and selectable
text, including when included in LaTeX with `graphicx`.

Labels use 9 pt parcel names and 8 pt values. Interior placement tests the
whole label rectangle against polygon boundaries and holes, and avoids other
labels. Narrow parcels use ordered margin callouts with leaders and marked
interior anchors. Separate components retain all their outlines but receive
one label per parcel. The renderer reports an error when its placement strategy
cannot produce a complete, readable A4 layout; it never silently drops labels
or shrinks the font.

The layout uses a custom deterministic greedy heuristic, combining standard
geometry and label-placement techniques:

1. **Find promising interior positions.** Sample a 16×16 grid within each
   polygon component's bounding box, plus points nudged inward from boundary
   edge midpoints to cover narrow polygons. Keep interior candidates and rank
   them by their distance from the nearest boundary, including holes.
2. **Fit the actual label rectangle.** Try up to 80 candidates per parcel.
   Require the whole rectangle to stay inside the parcel, avoid holes and
   boundary crossings, and leave a gap from already placed labels.
3. **Give constrained parcels first choice.** Process parcels with less
   maximum sampled interior clearance first, preserving their limited options.
4. **Place remaining labels in the margins.** Prefer the side nearest the
   interior anchor, switching sides if needed for capacity. Sort each column
   by anchor height, then pack labels with forward and backward passes to
   maintain separation and keep them within the page. Connect them to their
   interior anchors with leaders.
5. **Compare page orientations.** Attempt both portrait and landscape, keeping
   the successful layout with the largest map scale.

The closest reference for the interior-point step is
[Mapbox's polylabel](https://github.com/mapbox/polylabel), which finds a polygon's
*pole of inaccessibility*: the interior point furthest from its boundary.
Polylabel uses adaptive subdivision and guarantees an optimum within a supplied
precision. Abies uses fixed sampling and retains multiple candidates for
rectangle fitting and collision avoidance; it does not depend on polylabel.
The complete layout has no global optimality guarantee and may fail to find a
feasible arrangement even when one exists.

`bosco-calendar-print.js` adapts the calendar snapshot to this generic layout.
It joins parcel names to the database identities used by the snapshot, merges
split GeoJSON features, and rejects unmatched or missing parcel geometries.
GeoJSON feature IDs are not database parcel IDs. Other metrics can supply
different text labels without changing the geometry or PDF renderer.
