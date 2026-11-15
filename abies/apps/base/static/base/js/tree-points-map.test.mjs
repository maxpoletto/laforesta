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
  const handlers = {};
  return {
    on(name, callback) { (handlers[name] ||= []).push(callback); return this; },
    off(name, callback) {
      handlers[name] = (handlers[name] || []).filter(item => item !== callback);
      return this;
    },
    fire(name, detail) { for (const callback of handlers[name] || []) callback(detail); },
    size: {x:0,y:0}, invalidations: 0, fits: 0,
    getSize() { return {...this.size}; },
    invalidateSize() { this.invalidations++; this.size={x:this.container.clientWidth,y:this.container.clientHeight}; },
    setView(center,zoom) { this.view={center,zoom}; },
    fitBounds(bounds, opts) { this.fits++; this.fitted = { bounds, opts }; return this; },
    remove() { this.removed = true; },
  };
}
MapCommon.create = container => {
  lastMap = makeFakeMap();
  lastMap.container=container;
  let basemap = 'satellite';
  return {
    getLeafletMap: () => lastMap,
    getBasemap: () => basemap,
    setBasemap(name) {
      if (name === basemap) return;
      basemap = name;
      lastMap.fire('basemapstylechange', { name });
    },
    syncBasemap(name) { this.setBasemap(name); },
  };
};

function fakeLayerGroup() {
  return {
    layers: [],
    addTo() { return this; },
    addLayer(layer) { this.layers.push(layer); },
    clearLayers() { this.layers.length = 0; },
    eachLayer(callback) { this.layers.forEach(callback); },
  };
}
function fakeBounds(valid) {
  return { isValid: () => valid };
}
function fakeMarker(latlng, style) {
  return {
    latlng,
    style: { ...style },
    options: { ...style },
    handlers: {},
    bindTooltip(content, options) { this.tooltip = { content, options }; return this; },
    on(event, handler) { this.handlers[event] = handler; return this; },
    addTo(layer) { layer.addLayer(this); return this; },
    setStyle(update) { Object.assign(this.style, update); Object.assign(this.options, update); },
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

const sampleColumns = columns.map(column => (
  column === S.COL_NUMBER ? S.COL_TREE_NUM : column
));
const samplePoints = treePointsFromDigest(rows, sampleColumns, {
  number: S.COL_TREE_NUM,
});
check(samplePoints[0].number === 12 && samplePoints[0].species === 'Abete bianco'
      && samplePoints[0].lat === 38.1,
      'a partial column override preserves the shared tree field defaults');

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
check(first.style.fillColor === '#d6a800' && first.style.fillOpacity === 0.85,
      'tree marker uses the shared dark-yellow satellite style');
map.wrapper.syncBasemap('topo');
check(first.style.fillColor === '#2d5d2c',
      'tree marker restores the shared dark-green style on topo');
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

const observers=[];
globalThis.ResizeObserver=class {
  constructor(callback) { this.callback=callback; observers.push(this); }
  observe(target) { this.target=target; }
  disconnect() { this.disconnected=true; }
};
const {ParcelMap}=await import('./parcel-map.js');
const geographicContext={type:'FeatureCollection',features:[{properties:{layer:'Serra',name:'Serra-9'}}]};
for(const initialView of [undefined,{center:[38.1,16.2],zoom:14}]) {
  const sized=new ParcelMap({container:new MockNode('div'),geojson:geographicContext,initialView});
  const leaflet=sized.leaflet;
  const observer=observers.at(-1);
  check(Boolean(observer),'ParcelMap observes its actual container dimensions');
  if(observer) {
    const fits=leaflet.fits;
    sized.mapEl.clientWidth=800;sized.mapEl.clientHeight=600;
    observer.callback();
    check(leaflet.size.x===800&&leaflet.size.y===600,'a first visible box replaces the cached zero-size viewport');
    check(leaflet.fits===fits+(initialView?0:1),'first visibility fits parcels without replacing a saved view');
    const initialFits=leaflet.fits;
    sized.mapEl.clientWidth=1000;
    observer.callback();
    check(leaflet.size.x===1000&&leaflet.fits===initialFits,'later container resizing retains the current map view');
    const invalidations=leaflet.invalidations;
    observer.callback();
    sized.mapEl.clientHeight=0;observer.callback();
    check(leaflet.invalidations===invalidations,'unchanged and hidden boxes do not invalidate the map');
    sized.destroy();observer.callback();
    check(observer.disconnected&&leaflet.invalidations===invalidations,'teardown disconnects sizing observation and ignores queued callbacks');
  } else sized.destroy();
}
delete globalThis.ResizeObserver;

console.log('tree-points-map.js');
if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  console.log(`\n${pass} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`\n${pass} passed, 0 failed`);
