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

## Coordinates, rendering and measurements

Source GeoJSON coordinates are WGS84 longitude/latitude, in that order.
Leaflet APIs use latitude/longitude. `build_geo` preserves the supplied geometry;
it enriches properties without transforming the source datum. A source from a
different CRS must be correctly transformed to WGS84 before import.

Bosco and the other interactive maps use Leaflet's EPSG:3857 Web/Pseudo-Mercator
projection, matching their basemaps. Its spherical projection radius is the
WGS84 semi-major axis, 6378137 m, as prescribed by EPSG:3857. This makes the
east/west projection scale exact at the WGS84 equator. A mean Earth radius
would define different projected coordinates, rather than improve EPSG:3857.
Printable maps use the same formulas, with Y reversed for page coordinates;
the radius cancels during page fitting. This
display projection does not change the geographic coordinates of boundaries,
trees, observations or GPS fixes. Absolute alignment depends on the source
datum, boundary survey, imagery and GPS accuracy. The approximately 28% scale
enlargement around Calabria affects projected lengths, not coordinate placement.
See [PROJ's Web Mercator definition](https://proj.org/en/stable/operations/projections/webmerc.html).

All application point-to-point distances, the map ruler, and parcel polygon
areas use the **WGS84 ellipsoid**, through GeographicLib's Karney algorithms:
semi-major axis `a = 6378137 m`, flattening `f = 1 / 298.257223563`.
`base/js/geodesy.js` is the shared core for Abies ES modules and Ipso classic
scripts; it obtains the constants from GeographicLib's WGS84 instance. The
locked GeographicLib browser asset and its license are vendored locally and
the numerical asset/core are included in Ipso's offline precache.
There is no application Haversine fallback or separately tuned mobile radius.
The Web Mercator rendering radius is part of the display CRS definition.

`geo.distanceMeters()` returns the shortest ellipsoidal surface distance.
Polygon areas follow geodesic edges, ignore ring winding, subtract holes,
and sum MultiPolygon components. They are horizontal reference-ellipsoid
measurements, not terrain slope lengths/areas. Cadastral areas and stored
forestry sample areas (`pi * radius²`) retain their independent meanings.
`sortFeaturesByArea()` calculates each feature's `_areaM2` from its geometry
when loading parcel data, replacing any supplied cached area value. Bosco
sums component areas per parcel for geometric-area displays and per-hectare
values. The database's separately supplied parcel area is the cadastral-area
source, rather than a cache of this geometric calculation.
Sample-radius inference and nearest-area selection use the same metric on
desktop and mobile. An explicit sample-area ID takes precedence over coordinate
inference.
See [GeographicLib's algorithms and API](https://geographiclib.sourceforge.io/JavaScript/doc/).

Coordinate entry/display retain five decimal places. Around Calabria, the
rounding steps are about 1.11 m north/south and 0.87 m east/west; rounding
error is at most half a step on each axis. Stored/imported coordinates and
parcel vertices retain their supplied precision. Measurement calculations use
those numbers directly, without first rounding them for display.

Regression fixtures in `test/fixtures/wgs84-geodesy.json` come from the
independent GeographicLib Python implementation. Tests cover equatorial and
Calabrian distances, coincidence, antipodes, polar and date-line cases, ring
winding, holes, MultiPolygons, classic-script/module parity, the map ruler,
and sample-radius inclusion/exclusion. Vendor checks ensure the served library
matches the lock; Ipso precache checks cover offline availability.
The fixture generator `test/fixtures/generate_wgs84.py` records direct/inverse
and area references and independently minimizes geodesic segment distances;
it can be rerun with the locked Python dependencies.

### Local metric planes and sampling grids

`localProjection(lat, lng)` builds an ellipsoidal azimuthal-equidistant plane
around a chosen origin. A GeographicLib inverse calculation supplies each
point's distance and azimuth from the origin; these become east/north meter
coordinates. The inverse mapping uses GeographicLib's direct calculation.
Both directions use the same WGS84 ellipsoid as point distances and areas.
This is a computation plane; the map display uses Web Mercator.

`distanceToBoundaryMeters()` centers this plane at the query location and
finds the closest point on projected polygon segments, including hole and
MultiPolygon boundaries. Approximating a curved geodesic segment by a straight
segment in this local plane is tested within 1 cm of independently minimized
ellipsoidal segment distances over 5 km forestry extents. It is not a global
nearest-geodesic-segment solver.

Automatic sampling-grid construction places a cell-centered square lattice in
a plane centered on the selected parcels, then converts its points back to
WGS84 and clips their centers to the parcel geometry. Axes follow true
north/east at the origin. Grid spacing is expressed in meters and requires no
UTM-zone choice. The target-count search reuses one projection
and projected extent for all spacing trials. Tests verify 500 m neighbor
spacing within 1 mm across a 5 km square, point containment, parcel labels,
projection round trips, and the target-count contract.

### Satellite raster sizing

Server calculations use GeographicLib Python's `Geodesic.WGS84` through
`apps.base.geodesy`, with the same ellipsoid as the browser. Satellite
grids derive their nominal 10 m dimensions from ellipsoidal distances along
the bounding-box midlines. The rasters remain in WGS84 geographic coordinates.
Every date and band must use this canonical pixel grid. Downloads validate
retained dates against it, and precomputation verifies dimensions, CRS and
transform before publishing a mask or manifest. Rebuild into a fresh output
directory when changing source geometry or raster-grid calculations; see the
[satellite build commands](../README.md#local-development).

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
both use the PDF font metrics. An optional 8 pt legend is placed after the
subtitle and participates in header wrapping and map-space calculations.
The PDF retains vector paths and selectable
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

`bosco-metric-print.js` prepares the content and availability of printable
metrics. It supports average age, forest type, average altitude and the cutting
calendar. Altitude is the mean of both altitude bounds, in whole meters; a
missing bound produces a missing value.
Values use the same calculation and localized formatting as map tooltips, via
`bosco-characteristics.js`; missing values become `-` and zero remains a value.
Age, forest type and altitude maps need no selected harvest plan. Forest type labels
use the parcel digest's localized display value. `bosco-print-ui.js` freezes the
content in one layout for preview and download, using a shared modal template.

`bosco-parcel-print.js` joins parcel names to database identities, merges split
GeoJSON features, and rejects unmatched or missing parcel geometries.
Labels append the localized forest-type marker from the parcel's `coppice`
flag. `bosco-calendar-print.js` supplies calendar labels and heading ranges.
Heading ranges group calendar years by that flag across all parcels in
the selected plan, independently of the printed compresa. Parcel ordering uses
the unadorned parcel name. The legend explains forest types and unscheduled
parcels below the heading; other maps explain missing data instead. The footer
contains the export date. GeoJSON feature IDs are not database parcel IDs.
Metric adapters supply text without changing the geometry or PDF renderer.
