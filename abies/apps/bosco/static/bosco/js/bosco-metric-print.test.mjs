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
const slope = P.buildMetricPrint({ ...options, metricId: '9',
  entries: entries.map((entry, i) => ({ ...entry, gradePct: [34.6, null, 0][i] })) });
assert.equal(slope.title, 'Pendenza media - Forest');
assert.equal(slope.filename, 'pendenza_media_Forest.pdf');
assert.deepEqual(slope.items.map(item => item.lines), [['35 %'], ['-'], ['0 %']]);
assert.equal(P.printAvailability('9', { hasRegion: true }).disabled, false);
const harvestEntries = entries.map((entry, i) => ({ ...entry, key: `Forest-${entry.parcel}`,
  displayAreaHa: [2, null, 1][i], areaHa: [5, null, 1][i], cadastralAreaHa: [4, null, 1][i] }));
const historical = new Map([['Forest-1', 25.6], ['Forest-2b', 10]]);
const historicalOptions = { ...options, entries: harvestEntries, metricId: '4', context: { historical } };
const history = P.buildMetricPrint(historicalOptions);
assert.equal(history.title, 'Prelievo storico - Forest');
assert.equal(history.filename, 'prelievo_storico_Forest.pdf');
assert.deepEqual(history.items.map(item => item.lines), [['26 q'], ['10 q'], ['0 q']]);
assert.ok(!history.legend.includes('per ettaro'));
const historicalPerHa = { ...historicalOptions, context: { historical, perHa: true } };
const geometricHistory = P.buildMetricPrint(historicalPerHa);
assert.deepEqual(geometricHistory.items.map(item => item.lines), [['12,80 q/ha'], ['-'], ['0,00 q/ha']]);
assert.equal(geometricHistory.filename, 'prelievo_storico_Forest_per_ettaro.pdf');
assert.ok(geometricHistory.legend.includes('per ettaro') && geometricHistory.legend.includes('aree geometriche'));
const cadastralHistory = P.buildMetricPrint({ ...historicalPerHa, useCadastralArea: true,
  entries: harvestEntries.map(entry => ({ ...entry, displayAreaHa: entry.cadastralAreaHa })) });
assert.equal(cadastralHistory.items[0].lines[0], '6,40 q/ha');
assert.ok(cadastralHistory.legend.includes('aree catastali'));
assert.equal(historical.get('Forest-1'), 25.6, 'normalization does not mutate totals');
assert.equal(P.printAvailability('4', { hasRegion: true }).disabled, true,
  'do not offer zero-valued printouts while historical data is still loading');
assert.equal(P.printAvailability('4', { hasRegion: true, harvestsLoaded: true }).disabled, false);
assert.throws(() => P.buildMetricPrint({ ...historicalOptions, context: {} }), /Caricamento prelievi/);
const planData = { selected_plan: { id: 7, name: 'Piano di taglio 2027', year_start: 2027, year_end: 2050 },
  calendar: [{ parcel_id: 1, years: [2027, 2040] }, { parcel_id: 2, years: [2027, 2045] },
    { parcel_id: 4, years: [2050] }] };
const future = new Map([[1, 25.6], [2, 10]]);
const futureOptions = { ...options, entries: harvestEntries, metricId: '5', data: planData,
  planEntries: [...harvestEntries, { id: 4, region: 'Elsewhere', parcel: '1', coppice: true }],
  context: { future } };
const forecast = P.buildMetricPrint(futureOptions);
assert.equal(forecast.title, 'Prelievo previsto - Forest');
assert.equal(forecast.subtitle, 'Piano di taglio 2027 (fustaia 2027-2040, ceduo 2027-2050)',
  'plan ranges include interventions in other comprese');
assert.equal(forecast.filename, 'prelievo_previsto_Forest_2027-2050.pdf');
assert.deepEqual(forecast.items.map(item => item.lines), [['25,60 m³'], ['10,00 m³'], ['0,00 m³']]);
const perHaForecastOptions = { ...futureOptions, context: { future, perHa: true } };
const geometricForecast = P.buildMetricPrint(perHaForecastOptions);
assert.deepEqual(geometricForecast.items.map(item => item.lines), [['12,80 m³/ha'], ['-'], ['0,00 m³/ha']]);
assert.equal(geometricForecast.filename, 'prelievo_previsto_Forest_2027-2050_per_ettaro.pdf');
assert.ok(geometricForecast.legend.includes('aree geometriche'));
const cadastralForecast = P.buildMetricPrint({ ...perHaForecastOptions, useCadastralArea: true,
  entries: harvestEntries.map(entry => ({ ...entry, displayAreaHa: entry.cadastralAreaHa })) });
assert.equal(cadastralForecast.items[0].lines[0], '6,40 m³/ha');
assert.ok(cadastralForecast.legend.includes('aree catastali'));
const emptyForecast = P.buildMetricPrint({ ...futureOptions,
  data: { selected_plan: planData.selected_plan, calendar: [] }, context: { future: new Map() } });
assert.ok(emptyForecast.items.every(item => item.lines[0] === '0,00 m³'));
assert.equal(emptyForecast.subtitle, 'Piano di taglio 2027');
assert.equal(P.printAvailability('5', { hasRegion: true }).disabled, true);
assert.equal(P.printAvailability('5', { hasRegion: true, data: planData }).disabled, false);
assert.throws(() => P.buildMetricPrint({ ...futureOptions, data: null }), /Seleziona un piano/);
assert.throws(() => P.buildMetricPrint({ ...futureOptions, context: {} }), /Seleziona un piano/);
assert.equal(future.get(1), 25.6);
console.log('Metric printing: all characteristics and production fields, units and area choices passed.');
