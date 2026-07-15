/**
 * Unit tests for TreePointsMap's digest mapping and geolocated tree overlay.
 * Run with: node apps/base/static/base/js/tree-points-map.test.mjs (part of
 * `make test-js`).
 */

import MapCommon from './map-common.js';
import { ROW_ID } from './constants.js';
import * as S from './strings.js';

let pass = 0;
const failures = [];
const check = (ok, msg) => { if (ok) pass++; else failures.push(msg); };

class MockNode {
  constructor(tagName = null, text = '') {
    this.tagName = tagName;
    this._text = text;
    this.children = [];
    this.className = '';
  }
  get textContent() {
    return this.children.length
      ? this.children.map(child => child.textContent).join('')
      : this._text;
  }
  set textContent(value) {
    this.children = [];
    this._text = String(value ?? '');
  }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { for (const child of children) this.appendChild(child); }
  remove() {}
}

global.document = {
  documentElement: { lang: 'it' },
  createElement: tag => new MockNode(tag),
  createTextNode: text => new MockNode('#text', String(text)),
};

let lastMap = null;
function makeFakeMap() {
  return {
    on() { return this; },
    fitBounds(bounds, opts) { this.fitted = { bounds, opts }; return this; },
    remove() { this.removed = true; },
  };
}
MapCommon.create = () => {
  lastMap = makeFakeMap();
  return {
    getLeafletMap: () => lastMap,
    getBasemap: () => 'satellite',
    setBasemap() {},
    syncBasemap() {},
  };
};

function fakeLayerGroup() {
  return {
    layers: [],
    addTo() { return this; },
    addLayer(layer) { this.layers.push(layer); },
    clearLayers() { this.layers.length = 0; },
  };
}
function fakeBounds(valid) {
  return { isValid: () => valid };
}
function fakeMarker(latlng, style) {
  return {
    latlng,
    style,
    handlers: {},
    bindTooltip(content, options) { this.tooltip = { content, options }; return this; },
    on(event, handler) { this.handlers[event] = handler; return this; },
    addTo(layer) { layer.addLayer(this); return this; },
    click() { this.handlers.click?.({}); },
  };
}

global.L = {
  layerGroup: () => fakeLayerGroup(),
  geoJSON(data, opts) {
    const layer = {
      addTo() { return this; },
      getBounds() { return fakeBounds((data.features || []).length > 0); },
    };
    for (const feature of data.features || []) {
      opts.onEachFeature?.(feature, { bindTooltip() {}, on() {} });
    }
    return layer;
  },
  circleMarker: (latlng, style) => fakeMarker(latlng, style),
  DomEvent: { stopPropagation() {} },
};

const { TreePointsMap, treePointsFromDigest } = await import('./tree-points-map.js');

const columns = [
  ROW_ID, S.COL_NUMBER, S.COL_SPECIES, S.COL_D_CM, S.COL_H_M, S.COL_LAT, S.COL_LON,
];
const rows = [
  [101, 12, 'Abete bianco', 31, 18.25, 38.1, 16.2],
  [102, 13, 'Faggio', 20, 12, null, 16.3],
  [103, 14, 'Cerro', 22, 13, '38.4', '16.5'],
];
const points = treePointsFromDigest(rows, columns);
check(points.length === 3, 'treePointsFromDigest keeps every source row');
check(points[0].id === 101 && points[0].row === rows[0]
      && points[0].number === 12 && points[0].species === 'Abete bianco'
      && points[0].diameter === 31 && points[0].height === 18.25
      && points[0].lat === 38.1 && points[0].lon === 16.2,
      'treePointsFromDigest maps id, source row, and standard tree fields');

const host = new MockNode('div');
let clicked = null;
const map = new TreePointsMap({
  container: host,
  className: 'test-tree-map',
  geojson: { type: 'FeatureCollection', features: [] },
  tools: {},
  onTreeClick: tree => { clicked = tree; },
});
map.setTrees(points);
check(map.markerLayer.layers.length === 2,
      'setTrees renders only rows with finite Lat/Lon');
const first = map.markerLayer.layers[0];
check(first.latlng[0] === 38.1 && first.latlng[1] === 16.2,
      'setTrees uses [lat, lon] marker coordinates');
check(first.style.fillColor === '#2d5d2c' && first.style.fillOpacity === 0.85,
      'tree marker uses the shared dark-green style');
check(first.tooltip.options.sticky === true,
      'tree marker tooltip is sticky like parcel hover labels');
first.click();
check(clicked?.id === 101 && clicked?.row === rows[0],
      'tree marker click reports the mapped tree row');
check(first.tooltip.content.textContent.includes('Numero: 12')
      && first.tooltip.content.textContent.includes('Specie: Abete bianco')
      && first.tooltip.content.textContent.includes('D (cm): 31')
      && first.tooltip.content.textContent.includes('h (m): 18,25'),
      'tree marker tooltip lists number, species, diameter, and height');

map.destroy();
check(lastMap.removed === true, 'destroy delegates to ParcelMap teardown');

console.log('tree-points-map.js');
if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  console.log(`\n${pass} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`\n${pass} passed, 0 failed`);
