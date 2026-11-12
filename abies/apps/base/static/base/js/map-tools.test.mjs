/**
 * Unit tests for map-tools.js's pure `formatDistance` helper.
 *
 * Run with: node apps/base/static/base/js/map-tools.test.mjs (part of `make
 * test-js`).  format.js has no `document` in node, so it defaults to the
 * Italian locale — these assertions therefore also prove the distance is
 * formatted locale-aware (comma decimal), NOT via `.toFixed()` (dot).
 */
import { attachMeasure, formatDistance } from './map-tools.js';

let pass = 0;
const failures = [];
const eq = (actual, expected, msg) => {
  if (actual === expected) pass++;
  else failures.push(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
};

console.log('map-tools.js');

eq(formatDistance(0), '0,0 m', 'zero metres → "0,0 m"');
eq(formatDistance(12.3), '12,3 m', 'metres: 1 decimal, comma separator');
eq(formatDistance(999), '999,0 m', 'just below 1 km stays in metres');
eq(formatDistance(1000), '1,00 km', '1000 m → "1,00 km" (km threshold, 2 decimals)');
eq(formatDistance(1234.5), '1,23 km', 'kilometres: 2 decimals, comma separator');

let controlClick, mapClick;
const measuredLabels = [];
const node = { classList: { toggle() {} } };
const layer = { addTo() { return this; } };
globalThis.L = {
  Control: { extend: definition => class { onAdd() { return definition.onAdd(); } } },
  DomUtil: { create: () => node },
  DomEvent: { disableClickPropagation() {}, stopPropagation() {}, preventDefault() {},
    on(_node, _event, handler) { controlClick = handler; } },
  layerGroup: () => layer, circleMarker: () => layer, polyline: () => layer,
  divIcon: options => options,
  marker(_point, options) { measuredLabels.push(options.icon.html); return layer; },
};
const map = {
  addControl(control) { control.onAdd(); },
  on(_event, handler) { mapClick = handler; },
  getContainer: () => ({ style: {} }),
  distance() { throw new Error('Ruler must use the shared WGS84 core'); },
};
attachMeasure(map);
controlClick({});
mapClick({ latlng: { lat: 0, lng: 0 } });
mapClick({ latlng: { lat: 0, lng: 1 } });
eq(measuredLabels[0].includes('111,32 km'), true, 'ruler measures the WGS84 equatorial degree');

if (failures.length) {
  for (const f of failures) console.error(`FAIL ${f}`);
  console.log(`\n${pass} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`\n${pass} passed, 0 failed`);
