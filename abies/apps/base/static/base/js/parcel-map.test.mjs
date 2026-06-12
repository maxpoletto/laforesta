/**
 * Unit tests for ParcelMap's duplication-prone core: tool opt-in, view
 * framing, the unified click dispatch, sample-area marker bookkeeping,
 * active-highlight (incl. the idempotent public setter vs. the direct
 * marker-click path), and teardown.
 *
 * Run with: node apps/base/static/base/js/parcel-map.test.mjs (part of `make
 * test-js`).  Leaflet is a runtime global, so rather than fake its whole
 * control chrome we stub `MapCommon.create` (the one thing ParcelMap calls into)
 * and install a small `global.L` for ParcelMap's own layer/marker calls.
 */

import MapCommon from './map-common.js';

let pass = 0;
const failures = [];
const check = (ok, msg) => { if (ok) pass++; else failures.push(msg); };

// --- Stub MapCommon.create: capture opts, return a fake wrapper/leaflet -------
let lastCreateOpts = null;
let lastMap = null;

function makeFakeMap() {
  const handlers = {};
  return {
    handlers,
    controlsAdded: 0,
    controlsRemoved: 0,
    on(ev, cb) { (handlers[ev] ||= []).push(cb); return this; },
    off(ev, cb) { handlers[ev] = (handlers[ev] || []).filter(h => h !== cb); return this; },
    fire(ev, payload) { (handlers[ev] || []).forEach(cb => cb(payload)); },
    setView(center, zoom) { this.view = { center, zoom }; return this; },
    fitBounds(b, o) { this.fitted = { b, o }; return this; },
    getCenter() { return { lat: 1, lng: 2 }; },
    getZoom() { return 12; },
    getContainer() { return { style: {} }; },
    invalidateSize() { this.invalidated = (this.invalidated || 0) + 1; },
    addControl(c) { this.controlsAdded++; c.onAdd?.(this); return this; },
    removeControl() { this.controlsRemoved++; return this; },
    locate() { this.locating = true; return this; },
    stopLocate() { this.locating = false; return this; },
    distance() { return 1; },
    remove() { this.removed = true; },
  };
}

MapCommon.create = (el, opts) => {
  lastCreateOpts = opts;
  lastMap = makeFakeMap();
  return {
    leaflet: lastMap,
    getLeafletMap: () => lastMap,
    getBasemap: () => opts.basemap,
    setBasemap() {},
    syncBasemap() {},
  };
};

// --- Minimal global.L for ParcelMap's own calls ------------------------------
function fakeLayerGroup() {
  return {
    layers: [],
    addTo() { return this; },
    clearLayers() { this.layers.length = 0; },
    addLayer(l) { this.layers.push(l); },
  };
}
let lastParcelLayer = null;
function fakeMarker(latlng, style) {
  return {
    latlng, style: { ...style }, handlers: {},
    bindTooltip(t) { this.tooltip = t; return this; },
    on(ev, cb) { this.handlers[ev] = cb; return this; },
    addTo(layer) { layer.addLayer(this); return this; },
    setStyle(s) { Object.assign(this.style, s); },
    bringToFront() { this.front = true; },
    click() { this.handlers.click?.({}); },
  };
}
function fakeClassList() {
  const s = new Set();
  return {
    add: c => s.add(c), remove: c => s.delete(c),
    toggle: (c, on) => { const v = on === undefined ? !s.has(c) : on; v ? s.add(c) : s.delete(c); return v; },
    contains: c => s.has(c),
  };
}
global.L = {
  layerGroup() { return fakeLayerGroup(); },
  geoJSON(data, opts) {
    const layer = {
      featureClicks: [],
      tooltips: [],
      addTo() { return this; },
      getBounds() { return { isValid: () => true }; },
    };
    (data.features || []).forEach(f => {
      const lyr = {
        bindTooltip(t, options) { layer.tooltips.push({ f, tooltip: t, options }); return this; },
        on(ev, cb) { if (ev === 'click') layer.featureClicks.push({ f, cb }); },
      };
      opts.onEachFeature?.(f, lyr);
    });
    lastParcelLayer = layer;
    return layer;
  },
  circleMarker(latlng, style) { return fakeMarker(latlng, style); },
  polyline() { return { addTo() { return this; } }; },
  marker() { return { addTo() { return this; } }; },
  circle() { return { addTo() { return this; } }; },
  divIcon() { return {}; },
  // Enough of the control/DOM surface for map-tools' attach* to run.
  Control: { extend(proto) { return function Ctl() { Object.assign(this, proto); }; } },
  DomUtil: { create(tag, cls) { return { className: cls || '', href: '', title: '', textContent: '', classList: fakeClassList() }; } },
  DomEvent: { on() {}, off() {}, stop() {}, stopPropagation() {}, preventDefault() {}, disableClickPropagation() {} },
};

// --- Minimal DOM ------------------------------------------------------------
global.document = {
  createElement(tagName) {
    return {
      tagName, className: '', textContent: '', children: [],
      appendChild(child) { this.children.push(child); return child; },
      remove() {},
    };
  },
};
const makeContainer = () => ({ appendChild() {} });

function parcel(layer, name, props = {}) {
  return {
    type: 'Feature',
    properties: { layer, name, ...props },
    geometry: { type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] },
  };
}

const { ParcelMap } = await import('./parcel-map.js');

function newMap(opts = {}) {
  return new ParcelMap({
    container: makeContainer(),
    className: 'test-map',
    geojson: { type: 'FeatureCollection', features: [parcel('A', 'A-1'), parcel('B', 'B-2')] },
    ...opts,
  });
}

// --- Tool opt-in (off by default; attached only when requested) -------------
const m0 = newMap({ tools: {}, basemap: 'topo' });
check(lastCreateOpts.basemap === 'topo', 'basemap passed through to MapCommon.create');
check(m0._toolHandles.length === 0, 'tools:{} → no tools attached');
const mt = newMap({ tools: { measure: true, location: true } });
check(mt._toolHandles.length === 2 && lastMap.controlsAdded === 2,
      'tools:{measure,location} → both attached (2 controls)');
check(typeof mt._toolHandles[0].capturesClicks === 'function',
      'measure handle exposes capturesClicks() for click suppression');
check(!m0._toolHandles.some(h => h.capturesClicks?.()),
      'tools:{} → nothing captures clicks');

// --- View framing -----------------------------------------------------------
newMap();
check(!!lastMap.fitted && !lastMap.view, 'no initialView → fitBounds, not setView');
newMap({ initialView: { center: [45, 9], zoom: 14 } });
check(!!lastMap.view && !lastMap.fitted, 'initialView → setView, not fitBounds');

// --- Parcel tooltip DOM ----------------------------------------------------
newMap({
  geojson: {
    type: 'FeatureCollection',
    features: [parcel('Capistrano', 'Capistrano-3a', { coppice: false })],
  },
});
const parcelTooltip = lastParcelLayer.tooltips[0].tooltip;
check(lastParcelLayer.tooltips[0].options.sticky === true,
      'parcel tooltip bound with sticky option');
check(parcelTooltip.className === 'parcel-tooltip'
      && parcelTooltip.children[0].className === 'parcel-tooltip-title'
      && parcelTooltip.children[0].textContent === 'Capistrano 3a'
      && parcelTooltip.children[1].textContent === 'Fustaia',
      'parcel tooltip DOM includes title + forest type');

// --- View-change reporting --------------------------------------------------
let reported = null;
newMap({ onViewChange: (c, z) => { reported = { c, z }; } });
lastMap.fire('moveend');
check(reported && reported.c[0] === 1 && reported.c[1] === 2 && reported.z === 12,
      'moveend → onViewChange([lat,lng], zoom)');

// --- Unified click dispatch -------------------------------------------------
let clicks = [];
const m = newMap({ onMapClick: (latlng, feature) => clicks.push({ latlng, feature }) });
lastMap.fire('click', { latlng: { lat: 7, lng: 8 } });
check(clicks.length === 1 && clicks[0].feature === null,
      'map empty click → onMapClick(latlng, null)');
lastParcelLayer.featureClicks[0].cb({ latlng: { lat: 3, lng: 4 } });
check(clicks.length === 2 && clicks[1].feature.properties.layer === 'A',
      'parcel click → onMapClick(latlng, feature)');

// --- Click suppression while a tool captures clicks -------------------------
// While a tool (measure) is capturing clicks, map + parcel clicks build the
// measurement instead of reaching the page (which would otherwise open the
// new-area prompt).
let mc = [];
const pmMeasure = newMap({
  tools: { measure: true },
  onMapClick: (latlng, feature) => mc.push({ latlng, feature }),
});
const measureHandle = pmMeasure._toolHandles[0];
measureHandle.capturesClicks = () => true;
lastMap.fire('click', { latlng: { lat: 1, lng: 1 } });
lastParcelLayer.featureClicks[0].cb({ latlng: { lat: 2, lng: 2 } });
check(mc.length === 0,
      'tool capturing clicks: empty-space + parcel clicks suppressed (no onMapClick)');
measureHandle.capturesClicks = () => false;
lastMap.fire('click', { latlng: { lat: 3, lng: 3 } });
lastParcelLayer.featureClicks[0].cb({ latlng: { lat: 4, lng: 4 } });
check(mc.length === 2 && mc[0].feature === null && mc[1].feature.properties.layer === 'A',
      'not capturing: clicks dispatch onMapClick again');

// --- Marker bookkeeping + active-highlight ----------------------------------
let picked = [];
const a1 = { id: 11, lat: 1, lon: 2 };
const a2 = { id: 22, lat: 3, lon: 4 };
m._addAreaMarker(a1, { fillColor: '#0a0', fillOpacity: 0.8, tooltip: 't1',
                       onClick: (a) => picked.push(a.id) });
m._addAreaMarker(a2, { fillColor: '#0a0', fillOpacity: 0.8, tooltip: 't2',
                       onClick: (a) => picked.push(a.id) });
check(m.markers.size === 2 && m.markerLayer.layers.length === 2,
      '_addAreaMarker: tracked in markers map + marker layer');

m.markers.get(a1.id).click();
check(m.activeAreaId === 11 && picked.length === 1 && picked[0] === 11,
      'marker click → sets active + fires onClick');
check(m.markers.get(a1.id).style.color === '#ffcc00'
      && m.markers.get(a2.id).style.color === '#000',
      'active marker highlighted, others not');

// Re-clicking the already-active marker re-fires onClick (NOT idempotent).
m.markers.get(a1.id).click();
check(picked.length === 2 && picked[1] === 11,
      're-click active marker re-fires onClick');

// Public setActiveAreaId IS idempotent — no refresh when unchanged.
m.setActiveAreaId(22);
check(m.activeAreaId === 22 && m.markers.get(a2.id).front === true,
      'setActiveAreaId(other) → switches active + bringToFront');
m.markers.get(a2.id).front = false;
m.setActiveAreaId(22);
check(m.markers.get(a2.id).front === false,
      'setActiveAreaId(same) → early-out, no re-highlight');

m._clearMarkers();
check(m.markers.size === 0 && m.markerLayer.layers.length === 0,
      '_clearMarkers: empties markers map + layer');

// --- Teardown (tools torn down before the map is removed) -------------------
const d = newMap({ tools: { measure: true, location: true } });
const itsMap = lastMap;
d.destroy();
check(itsMap.removed === true && itsMap.controlsRemoved === 2
      && d.leaflet === null && d.markers.size === 0 && d._toolHandles.length === 0,
      'destroy: tools removed, leaflet removed, refs nulled');

// --- Report -----------------------------------------------------------------
console.log('parcel-map.js');
if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  console.log(`\n${pass} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`\n${pass} passed, 0 failed`);
