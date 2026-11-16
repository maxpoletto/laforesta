import { stagePrintModules } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const tmp = stagePrintModules('abies-print-context-');
const C = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-print-context.js')));
const M = await import(pathToFileURL(path.join(tmp, 'bosco/js/characteristic-print.js')));
const polygon = { type: 'Polygon', coordinates: [[[16, 38], [16.01, 38],
  [16.01, 38.01], [16, 38.01], [16, 38]]] };
const line = (layer, properties = {}) => ({ type: 'Feature', properties: { layer, ...properties },
  geometry: { type: 'LineString', coordinates: [[15.99, 38.004], [16.003, 38.004], [16.02, 38.004]] } });
const water = { type: 'Feature', properties: { layer: 'water', kind: 'water' },
  geometry: { type: 'Polygon', coordinates: [polygon.coordinates[0],
    [[16.003, 38.003], [16.004, 38.003], [16.004, 38.004], [16.003, 38.004], [16.003, 38.003]]] } };
const features = [line('roads', { kind: 'track' }), line('roads', { kind: 'primary' }), water,
  line('water', { kind: 'stream' }), line('contours', { elevation: 100 }), line('contours', { elevation: 120 })];
const input = { items: [{ name: '1 (F)', geometry: polygon, lines: ['37 a'] }], footer: 'Export' };
const plain = M.buildPrintMapLayout(input);
const credits = C.contextCredits({ build: { dem_source: 'copernicus-glo-30' } }, C.CONTEXT_LAYERS);
const layout = M.buildPrintMapLayout({ ...input, contextFeatures: features, credits });
assert.equal(layout.scale, plain.scale);
assert.equal(layout.mapWidthCm, plain.mapWidthCm);
assert.equal(layout.mapHeightCm, plain.mapHeightCm);
assert.equal(layout.context.shapes.length, features.length);
assert.ok(layout.footerLines.some(line => line.text.includes('OpenStreetMap')));
assert.ok(layout.footerLines.some(line => line.text.includes('Copernicus')));
assert.ok(layout.footerLines.every(line => line.y >= layout.mapBox.y + layout.mapBox.height));
const pdf = M.printMapPDF(layout);
assert.ok(pdf.includes(' re W n'), 'clip backgrounds to the map area, outside headings and callouts');
assert.ok(pdf.includes('[3 2] 0 d'), 'tracks use dashed vector lines');
assert.ok(pdf.includes(' B* Q'), 'water polygons use even/odd fills so islands stay empty');
assert.ok(!pdf.includes('/Subtype /Image'));
const shapes = pdf.indexOf(' re W n');
const label = pdf.indexOf('(1 \\(F\\))');
assert.ok(label > shapes, 'parcel labels are drawn above context');
assert.notEqual(C.contextStyle({ layer: 'contours', elevation: 100 }).width,
  C.contextStyle({ layer: 'contours', elevation: 120 }).width, 'index contours are emphasized');
assert.equal(C.contextCredits({ build: {} }, []).length, 0);
assert.ok(C.contextCredits({ build: {} }, ['water']).every(text => !text.includes('Copernicus')));
assert.ok(C.contextCredits({ build: { dem_source: 'local' } }, ['contours']).every(text => !text.includes('Copernicus')));
globalThis.document = { createElementNS(ns, tag) {
  return { tag, attrs: {}, children: [], setAttribute(key, value) { this.attrs[key] = value; },
    appendChild(child) { this.children.push(child); } };
} };
const svg = M.buildPrintMapPreview(layout);
const group = svg.children.find(child => child.tag === 'g');
assert.equal(group.attrs['clip-path'], 'url(#bosco-print-context-clip)');
assert.ok(group.children.some(child => child.attrs['fill-rule'] === 'evenodd'));
assert.ok(group.children.some(child => child.attrs['stroke-dasharray'] === '3 2'));

// Input GeoJSON may retain multipart geometries instead of the builder's
// individual components. Each component must reach both vector renderers.
const multipart = [
  { ...water, geometry: { type: 'MultiPolygon', coordinates: [water.geometry.coordinates, water.geometry.coordinates] } },
  { ...line('water'), geometry: { type: 'MultiLineString', coordinates: [
    [[0, 10], [100, 10]], [[0, 20], [100, 20]],
  ] } },
  { properties: { layer: 'water' }, geometry: { type: 'Point', coordinates: [5, 5] } },
];
const multipartLayout = C.layoutPrintContext(multipart, point => point,
  { x: 0, y: 0, width: 100, height: 100 }, []);
assert.equal(multipartLayout.shapes.length, 4);
assert.equal(multipartLayout.shapes.filter(shape => shape.close).length, 2);
const indexLine = (elevation, points) => ({ properties: { layer: 'contours', elevation },
  geometry: { type: 'LineString', coordinates: points } });
const index = C.layoutPrintContext([
  indexLine(100, [[0, 10], [10, 10]]),
  indexLine(200, [[-200, 50], [0, 50], [50, 50], [100, 50], [300, 50]]),
  indexLine(300, [[-100, 80], [30, 80], [100, 80], [150, 80], [200, 80]]),
  indexLine(400, [[-100, 80], [30, 80], [100, 80], [150, 80], [200, 80]]),
], point => point, { x: 0, y: 0, width: 100, height: 100 }, [
  { x: 35, y: 40, width: 30, height: 20 },
]);
assert.deepEqual(index.labels.map(label => label.text), ['300 m'],
  'short, clipped and colliding index labels are omitted; a free quarter-point is used');
const vectorLayout = { ...layout, context: index };
const labelledSVG = M.buildPrintMapPreview(vectorLayout);
assert.ok(labelledSVG.children.find(child => child.tag === 'g').children.some(child => child.textContent === '300 m'));
assert.ok(M.printMapPDF(vectorLayout).includes('(300 m)'));

let calls = [];
const data = { available: true, schema_version: 1, features };
globalThis.fetch = async (url, options) => {
  calls.push({ url, options });
  return { ok: true, status: calls.length === 1 ? 200 : 304,
    headers: { get: key => key === 'ETag' ? '"context-v1"' : null }, json: async () => data };
};
assert.equal((await C.loadPrintContext(1)).features, features);
assert.equal((await C.loadPrintContext(1)).features, features, 'a 304 retains the cached vectors');
assert.equal(calls[1].options.headers['If-None-Match'], '"context-v1"');
assert.equal(calls[0].url, '/api/bosco/map-context/1/');
assert.deepEqual(await C.loadPrintContext(null), { available: false });
assert.equal(calls.length, 2);
console.log('Print context: vector clipping, holes, fixed scale, sources, styling and conditional cache passed.');
