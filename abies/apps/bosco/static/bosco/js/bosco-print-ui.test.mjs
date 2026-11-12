import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-print-ui-'));
for (const app of ['base', 'bosco']) {
  fs.cpSync(path.resolve(here, `../../../../${app}/static/${app}/js`),
    path.join(tmp, app, 'js'), { recursive: true });
}
copyGeodesicVendor(tmp);
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const { showMetricPrint } = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-print-ui.js')));
class Element {
  constructor() { this.children = []; this.listeners = {}; this.attrs = {}; this.textContent = ''; }
  setAttribute(key, value) { this.attrs[key] = value; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  remove() {}
  classList = { add() {}, remove() {} };
}
const container = new Element(), preview = new Element(), download = new Element();
const cancel = new Element(), error = new Element(), title = new Element();
const fragment = new Element();
fragment.querySelector = selector => ({
  '[data-target="print-preview"]': preview,
  '[data-action="download-map"]': download,
  '[data-target="print-error"]': error,
  '[data-field="print-title"]': title,
})[selector];
fragment.querySelectorAll = () => [cancel];
let downloaded, blob;
globalThis.document = {
  documentElement: { lang: 'it' }, body: new Element(),
  getElementById: id => id === 'modal-container' ? container
    : { content: { cloneNode: () => fragment } },
  createElementNS: () => new Element(),
  createElement(tag) {
    const el = new Element();
    if (tag === 'a') el.click = () => { downloaded = el.download; };
    return el;
  },
  addEventListener() {}, removeEventListener() {},
};
URL.createObjectURL = value => { blob = value; return 'blob:test'; };
URL.revokeObjectURL = () => {};
const geometry = { type: 'Polygon', coordinates: [[[16, 38], [16.01, 38],
  [16.01, 38.01], [16, 38.01], [16, 38]]] };
const data = { selected_plan: { id: 1, name: 'Piano di taglio Plan A', year_start: 2027, year_end: 2050 },
  calendar: [{ parcel_id: 1, years: [2027, 2040] }, { parcel_id: 2, years: [2027, 2050] }] };
showMetricPrint({ metricId: '10', geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [
  { id: 1, region: 'Forest', parcel: '1', coppice: false },
  { id: 2, region: 'Elsewhere', parcel: '2a', coppice: true },
],
data, date: new Date(2026, 10, 11) });
assert.equal(preview.children.length, 1, 'open a preview before downloading');
assert.equal(blob, undefined, 'opening preview does not download automatically');
const heading = 'Piano di taglio Plan A (fustaia 2027-2040, ceduo 2027-2050)';
const legend = '(C) = Ceduo, (F) = Fustaia, (-) = Nessun taglio previsto.';
const previewText = preview.children[0].children.filter(element => element.textContent);
assert.ok(previewText.some(element => element.textContent === heading));
const legendElement = previewText.find(element => element.textContent === legend);
assert.equal(legendElement?.attrs['font-size'], 8, 'legend is small text under the heading');
assert.ok(legendElement.attrs.y > previewText.find(element => element.textContent === heading).attrs.y);
assert.ok(previewText.some(element => element.textContent === '1 (F)'));
assert.ok(previewText.some(element => element.textContent === 'Esportato il 2026-11-11'));
data.selected_plan.name = 'Plan B';
data.calendar[0].years = [2040];
download.listeners.click();
assert.match(downloaded, /Forest.*2027.*2050.*\.pdf$/);
const pdf = await blob.text();
assert.ok(pdf.includes('Plan A') && pdf.includes('2027, 2040'), 'download uses the preview snapshot');
assert.ok(pdf.includes('fustaia 2027-2040, ceduo 2027-2050'));
assert.ok(pdf.includes('1 \\(F\\)'));
assert.ok(pdf.includes('\\(C\\) = Ceduo, \\(F\\) = Fustaia'));
assert.ok(pdf.includes('(Esportato il 2026-11-11)'));
assert.ok(!pdf.includes('Piano di taglio:') && !pdf.includes(' | '));
assert.ok(!pdf.includes('Plan B'));
assert.ok(!pdf.includes('/Subtype /Image'));
assert.equal(error.textContent, '');
const createURL = URL.createObjectURL;
URL.createObjectURL = () => { throw new Error('Download unavailable'); };
download.listeners.click();
assert.ok(error.textContent, 'download failures stay inside the preview');
URL.createObjectURL = createURL;
download.listeners.click();
assert.equal(error.textContent, '', 'a download can be retried without reopening the preview');
cancel.listeners.click();
assert.equal(container.children.length, 0);
preview.replaceChildren();
blob = undefined;
showMetricPrint({ metricId: '1', geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [
  { id: 1, region: 'Forest', parcel: '1', coppice: false, aveAge: 37 },
], date: new Date(2026, 10, 11) });
assert.equal(title.textContent, 'Mappa stampabile - Età media');
assert.ok(preview.children[0].children.some(element => element.textContent === '37 a'));
download.listeners.click();
assert.equal(downloaded, 'eta_media_Forest.pdf');
assert.ok((await blob.text()).includes('(37 a)'));
cancel.listeners.click();
preview.replaceChildren();
const historical = new Map([['Forest-1', 25.6]]);
const harvestEntry = { id: 1, region: 'Forest', parcel: '1', key: 'Forest-1',
  coppice: false, displayAreaHa: 2 };
showMetricPrint({ metricId: '4', geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [harvestEntry],
context: { historical, perHa: true }, useCadastralArea: true });
assert.ok(preview.children[0].children.some(element => element.textContent === '12,80 q/ha'));
historical.set('Forest-1', 100);
harvestEntry.displayAreaHa = 10;
download.listeners.click();
assert.equal(downloaded, 'prelievo_storico_Forest_per_ettaro.pdf');
const historicalPDF = await blob.text();
assert.ok(historicalPDF.includes('(12,80 q/ha)'), 'download freezes the totals and area used by the preview');
assert.ok(historicalPDF.includes('aree catastali'));
cancel.listeners.click();
preview.replaceChildren();
const future = new Map([[1, 25.6]]);
const forecastData = { selected_plan: { id: 8, name: 'Selected plan', year_start: 2027, year_end: 2050 },
  calendar: [{ parcel_id: 1, years: [2027, 2040] }] };
showMetricPrint({ metricId: '5', geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [{ ...harvestEntry, displayAreaHa: 2 }],
context: { future, perHa: true }, data: forecastData });
assert.ok(preview.children[0].children.some(element => element.textContent === '12,80 m³/ha'));
future.set(1, 0);
forecastData.selected_plan.name = 'Changed plan';
download.listeners.click();
assert.equal(downloaded, 'prelievo_previsto_Forest_2027-2050_per_ettaro.pdf');
const forecastPDF = await blob.text();
assert.ok(forecastPDF.includes('Selected plan') && !forecastPDF.includes('Changed plan'));
assert.ok(forecastPDF.includes('aree geometriche'));
cancel.listeners.click();
console.log('Print UI: metric previews, snapshots, reader access, download retry and close passed.');
