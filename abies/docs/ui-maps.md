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
   `basemapchange` for cross-map sync; pages round-trip the active basemap
   through the `mt=` URL param (`o`/`t`/`s`). It also holds `BASEMAPS`. Nothing else:
   no measure, location, markers, or geometry.

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
   marker layer with active-highlight for the maps that need it. Owns
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
  fillColor, fillOpacity, tooltip, onClick })`. The base owns marker
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
`build_geo` enriches those features after `import_parcels` with static DB-backed
metadata such as `properties.coppice`. After fetching, sort with
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
