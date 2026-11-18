import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import { installDOM } from '../../../../../test/js-dom.mjs';
// Regression tests for the automatic grid planner.
// Run with: node apps/campionamenti/static/campionamenti/js/grid-planner.test.mjs

import {
  fileURLToPath, pathToFileURL,
} from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-grid-planner-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'campionamenti'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'campionamenti', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
copyGeodesicVendor(staticRoot);
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const {
  FIELD_DESCRIPTION, FIELD_NAME, FIELD_NONCE, FIELD_POINTS, FIELD_R_M,
} = await import(staticModule('base/js/constants.js'));
const { GridPlanner } = await import(staticModule('campionamenti/js/grid-planner.js'));

let passed = 0;
let failed = 0;

function check(ok, msg) {
  if (ok) passed++;
  else { failed++; console.error(`FAIL ${msg}`); }
}
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
}

globalThis.document = { body: { dataset: { csrf: 'csrf-token' } } };
Object.defineProperty(globalThis, 'crypto', {
  value: { randomUUID: () => 'grid-planner-nonce' }, configurable: true,
});

let posted = null;
globalThis.fetch = async (url, opts) => {
  posted = { url, opts, body: JSON.parse(opts.body) };
  return {
    status: 200,
    ok: true,
    json: async () => ({ row_id: 42 }),
  };
};

const fields = new Map([
  ['#grid-auto-name', { value: 'Auto grid' }],
  ['#grid-auto-description', { value: 'Descrizione' }],
  ['#grid-auto-radius', { value: '12' }],
]);
const host = { querySelector: sel => fields.get(sel) };
let created = null;
const planner = new GridPlanner({ host, onCreated: (rowId, data) => { created = { rowId, data }; } });
planner.points = [{ compresa: 'Capistrano', particella: '1', lat: 38.5, lon: 16.1 }];
planner.submitBtn = { disabled: false };
planner.statusEl = {
  textContent: '',
  classList: { toggle() {} },
};

await planner._save();

eq(posted.url, '/api/campionamenti/grid/save-auto/', 'planner posts to the auto-grid endpoint');
eq(posted.body, {
  [FIELD_NAME]: 'Auto grid',
  [FIELD_DESCRIPTION]: 'Descrizione',
  [FIELD_R_M]: 12,
  [FIELD_POINTS]: planner.points,
  [FIELD_NONCE]: 'grid-planner-nonce',
}, 'planner save body uses shared wire constants and includes a nonce');
eq(created, { rowId: 42, data: { row_id: 42 } }, 'planner calls onCreated after save');

// Newly imported maps can use standard GeoJSON without exporter-specific properties.
const mapHost = installDOM();
const { default: mapCommon } = await import(staticModule('base/js/map-common.js'));
mapCommon.create = () => ({ getLeafletMap: () => ({ on() {} }) });
const layer = { addTo() { return this; }, getBounds: () => ({ isValid: () => false }) };
globalThis.L = { geoJSON: () => layer, layerGroup: () => layer };
const ring = [[16, 38], [16.1, 38], [16, 38.1], [16, 38]];
const single = { type: 'Feature', properties: { layer: 'Costa-Nord', name: 'Costa-Nord-01' },
  geometry: { type: 'Polygon', coordinates: [ring] } };
const multi = { type: 'Feature', properties: { Compresa: 'Serra', Particella: '1' },
  geometry: { type: 'MultiPolygon', coordinates: [[ring]] } };
const polygonPlanner = new GridPlanner({ host: mapHost,
  geojson: { type: 'FeatureCollection', features: [single, multi] } });
await polygonPlanner.init();
eq(polygonPlanner.regionListEl.querySelectorAll('input').map(input => input.value),
   ['Costa-Nord', 'Serra'], 'the planner offers regions from standard Polygon and MultiPolygon features');
eq(polygonPlanner.parcelMap.geojson.features, [single, multi], 'both outlines reach the map');

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
