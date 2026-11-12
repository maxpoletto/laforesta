import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-metric-print-'));
for (const app of ['base', 'bosco']) {
  fs.cpSync(path.resolve(here, `../../../../${app}/static/${app}/js`),
    path.join(tmp, app, 'js'), { recursive: true });
}
copyGeodesicVendor(tmp);
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const P = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-metric-print.js')));
const polygon = x => ({ type: 'Polygon', coordinates: [
  [[x, 38], [x + .01, 38], [x + .01, 38.01], [x, 38.01], [x, 38]],
] });
const entries = [
  { id: 1, region: 'Forest', parcel: '1', coppice: false, type: 'Fustaia', aveAge: 70.6 },
  { id: 2, region: 'Forest', parcel: '2b', coppice: true, type: 'Ceduo', aveAge: null },
  { id: 3, region: 'Forest', parcel: '10', coppice: false, type: 'Fustaia', aveAge: 0 },
];
const geojson = { features: entries.map((entry, i) => ({
  geometry: polygon(16 + i * .02), properties: { layer: entry.region, name: `${entry.region}-${entry.parcel}` },
})) };
const options = { geojson, region: { id: 1, name: 'Forest' }, entries };
const age = P.buildMetricPrint({ ...options, metricId: '1' });
assert.equal(age.title, 'Età media - Forest');
assert.deepEqual(age.items.map(item => item.name), ['1 (F)', '2b (C)', '10 (F)']);
assert.deepEqual(age.items.map(item => item.lines), [['71 a'], ['-'], ['0 a']]);
assert.equal(age.filename, 'eta_media_Forest.pdf');
assert.ok(age.legend.includes('Dato non disponibile'));
assert.equal(P.printAvailability('1', { hasRegion: true }).disabled, false,
  'age printing does not require a harvest plan');
assert.equal(P.printAvailability('1', { hasRegion: false }).disabled, true);
assert.equal(P.printAvailability('10', { hasRegion: true }).disabled, true,
  'calendar still requires its plan snapshot');
assert.equal(P.printAvailability('6', { hasRegion: true }).visible, false,
  'satellite fields are a separate group');
assert.throws(() => P.buildMetricPrint({ ...options, metricId: '6' }));
const forestType = P.buildMetricPrint({ ...options, metricId: '2' });
assert.equal(forestType.title, 'Governo - Forest');
assert.equal(forestType.filename, 'governo_Forest.pdf');
assert.deepEqual(forestType.items.map(item => item.lines), [['Fustaia'], ['Ceduo'], ['Fustaia']]);
assert.deepEqual(forestType.items.map(item => item.name), ['1 (F)', '2b (C)', '10 (F)']);
const unknownType = P.buildMetricPrint({ ...options, metricId: '2',
  entries: entries.map(entry => ({ ...entry, type: null })) });
assert.ok(unknownType.items.every(item => item.lines[0] === '-'));
assert.equal(P.printAvailability('2', { hasRegion: true }).disabled, false,
  'forest type printing does not require a harvest plan');
const altitude = P.buildMetricPrint({ ...options, metricId: '3',
  entries: entries.map((entry, i) => ({ ...entry, altitudeMean: [971.5, null, 0][i] })) });
assert.equal(altitude.title, 'Altitudine media - Forest');
assert.equal(altitude.filename, 'altitudine_media_Forest.pdf');
assert.deepEqual(altitude.items.map(item => item.lines), [['972 m'], ['-'], ['0 m']]);
assert.equal(P.printAvailability('3', { hasRegion: true }).disabled, false);
console.log('Metric printing: age, forest type, altitude, missing data and availability passed.');
