import { stagePrintModules } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const tmp = stagePrintModules('abies-print-ui-');
const { showMetricPrint: openMetricPrint } = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-print-ui.js')));
const flush = () => new Promise(resolve => setTimeout(resolve, 0));
const pdfHasText = (pdf, text) => pdf.includes(text) || pdf.toLowerCase().includes(Buffer.from(text, 'latin1').toString('hex'));
const showMetricPrint = async options => {
  Object.values(layers).forEach(control => { control.checked = false; control.disabled = true; });
  openMetricPrint({ loadContext: async () => ({ available: false }), ...options });
  await flush();
};
class Element {
  constructor() { this.children = []; this.listeners = {}; this.attrs = {}; this.textContent = ''; }
  setAttribute(key, value) { this.attrs[key] = value; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  remove() {}
  focus() { this.focused = true; }
  classList = { add() {}, remove() {} };
}
const container = new Element(), preview = new Element(), download = new Element();
const cancel = new Element(), error = new Element(), title = new Element();
const scale = new Element(), dimensions = new Element(), paper = new Element();
const contextStatus = new Element(), contourNote = new Element();
const layers = Object.fromEntries(['roads', 'water', 'contours'].map(layer => {
  const control = new Element();
  control.dataset = { layer };
  control.disabled = true;
  control.checked = false;
  return [layer, control];
}));
scale.value = '10000';
const fragment = new Element();
fragment.querySelector = selector => ({
  '[data-target="print-preview"]': preview,
  '[data-action="download-map"]': download,
  '[data-target="print-error"]': error,
  '[data-field="print-title"]': title,
  '[data-role="print-scale"]': scale,
  '[data-field="print-dimensions"]': dimensions,
  '[data-field="print-paper"]': paper,
  '[data-field="print-context-status"]': contextStatus,
  '[data-field="print-contour-note"]': contourNote,
  ...Object.fromEntries(Object.entries(layers).map(([layer, control]) => [`[data-layer="${layer}"]`, control])),
})[selector];
fragment.querySelectorAll = () => [cancel];
let downloaded, blob;
const documentListeners = {};
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
  addEventListener(event, handler) { documentListeners[event] = handler; },
  removeEventListener(event) { delete documentListeners[event]; },
};
URL.createObjectURL = value => { blob = value; return 'blob:test'; };
URL.revokeObjectURL = () => {};
const geometry = { type: 'Polygon', coordinates: [[[16, 38], [16.01, 38],
  [16.01, 38.01], [16, 38.01], [16, 38]]] };
const data = { selected_plan: { id: 1, name: 'Piano di taglio Plan A', year_start: 2027, year_end: 2050 },
  calendar: [{ parcel_id: 1, years: [2027, 2040] }, { parcel_id: 2, years: [2027, 2050] }] };
await showMetricPrint({ metricId: '10', geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [
  { id: 1, region: 'Forest', parcel: '1', coppice: false },
  { id: 2, region: 'Elsewhere', parcel: '2a', coppice: true },
],
data, date: new Date(2026, 10, 11) });
assert.equal(preview.children.length, 1, 'open a preview before downloading');
assert.equal(blob, undefined, 'opening preview does not download automatically');
assert.equal(download.disabled, false);
assert.ok(scale.focused, 'focus the scale input when opening the modal');
assert.match(dimensions.textContent, /8,8 × 11,1 cm/);
assert.match(paper.textContent, /A4/);
const heading = 'Piano di taglio Plan A (fustaia 2027-2040, ceduo 2027-2050)';
const legend = '(C) = Ceduo, (F) = Fustaia, (-) = Nessun taglio previsto.';
const previewText = preview.children[0].children.filter(element => element.textContent);
assert.ok(previewText.some(element => element.textContent === heading));
const legendElement = previewText.find(element => element.textContent === legend);
assert.equal(legendElement?.attrs['font-size'], 8, 'legend is small text under the heading');
assert.ok(legendElement.attrs.y > previewText.find(element => element.textContent === heading).attrs.y);
assert.ok(previewText.some(element => element.textContent === '1 (F)'));
assert.ok(previewText.some(element => element.textContent === 'Esportato il 2026-11-11'));
const createSVG = document.createElementNS;
document.createElementNS = () => { throw new Error(); };
scale.listeners.input();
assert.equal(download.disabled, true, 'preview rendering failures cannot leave a downloadable layout');
assert.equal(preview.children.length, 0);
assert.ok(error.textContent, 'a rendering failure without a message receives the localized fallback');
document.createElementNS = createSVG;
scale.listeners.input();
assert.equal(download.disabled, false, 'preview rendering can be retried in the same modal');
assert.equal(error.textContent, '');
data.selected_plan.name = 'Plan B';
data.calendar[0].years = [2040];
geometry.coordinates[0][1][0] = 17;
scale.value = '5000';
scale.listeners.input();
assert.match(dimensions.textContent, /17,6 × 22,2 cm/, 'scale edits retain the original geometry');
assert.ok(preview.children[0].children.some(element => element.textContent === 'Scala 1 : 5000'));
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
assert.ok(pdf.includes('(Scala 1 : 5000)'));
assert.equal(error.textContent, '');
assert.equal(container.children.length, 1, 'a successful download keeps the scale modal open');
for (const invalid of ['', '0', '-1', '5000.5', 'Infinity', '9007199254740992']) {
  scale.value = invalid;
  scale.listeners.input();
  assert.equal(download.disabled, true, `reject invalid scale ${invalid}`);
  assert.equal(preview.children.length, 0, 'remove stale previews on invalid input');
  assert.equal(paper.textContent, '');
  assert.equal(dimensions.textContent, '');
  assert.equal(scale.attrs['aria-invalid'], 'true');
  assert.equal(download.title, error.textContent, 'disabled download explains the scale error');
  assert.ok(error.textContent);
  const previous = blob;
  download.listeners.click();
  assert.equal(blob, previous, 'never download a stale layout after an invalid edit');
}
scale.value = '100';
scale.listeners.input();
assert.equal(download.disabled, true);
assert.match(error.textContent, /A0/);
scale.value = '1250';
scale.listeners.input();
assert.equal(download.disabled, false, 'recover after an oversized scale');
assert.equal(scale.attrs['aria-invalid'], 'false');
assert.equal(download.title, '');
assert.equal(error.textContent, '');
assert.match(paper.textContent, /A0/);
download.listeners.click();
assert.match(await blob.text(), /\/MediaBox \[0 0 2383\.94 3370\.39\]/);
scale.value = '5000';
scale.listeners.input();
const createURL = URL.createObjectURL;
URL.createObjectURL = () => { throw new Error('Download unavailable'); };
download.listeners.click();
assert.ok(error.textContent, 'download failures stay inside the preview');
URL.createObjectURL = createURL;
download.listeners.click();
assert.equal(error.textContent, '', 'a download can be retried without reopening the preview');
assert.equal(container.children.length, 1, 'retry does not close the modal');
cancel.listeners.click();
assert.equal(container.children.length, 0);
assert.equal(documentListeners.keydown, undefined, 'close removes the Escape handler');
geometry.coordinates[0][1][0] = 16.01;
scale.value = '10000';
preview.replaceChildren();
blob = undefined;
await showMetricPrint({ metricId: '1', geojson: { features: [{ geometry,
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
await showMetricPrint({ metricId: '4', geojson: { features: [{ geometry,
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
await showMetricPrint({ metricId: '5', geojson: { features: [{ geometry,
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
// A default scale that cannot fit must still open a usable modal, so the user
// can correct it without leaving the dialog or refreshing the page.
preview.replaceChildren();
scale.value = '10000';
await showMetricPrint({ metricId: '1', geojson: { features: [{
  geometry: { type: 'Polygon', coordinates: [[[16, 38], [16.1, 38],
    [16.1, 38.1], [16, 38.1], [16, 38]]] },
  properties: { layer: 'Forest', name: 'Forest-1' },
}] }, region: { name: 'Forest' }, entries: [
  { id: 1, region: 'Forest', parcel: '1', coppice: false, aveAge: 37 },
] });
assert.equal(container.children.length, 1);
assert.equal(download.disabled, true);
assert.match(error.textContent, /A0/);
scale.value = '25000';
scale.listeners.input();
assert.equal(download.disabled, false);
assert.equal(preview.children.length, 1);
download.listeners.click();
assert.ok((await blob.text()).includes('(Scala 1 : 25000)'));
assert.equal(container.children.length, 1);
documentListeners.keydown({ key: 'Enter' });
assert.equal(container.children.length, 1, 'ordinary keys do not dismiss the dialog');
documentListeners.keydown({ key: 'Escape' });
assert.equal(container.children.length, 0);

const sourceContext = { available: true, schema_version: 1,
  build: { dem_source: 'copernicus-glo-30', contour_interval_m: 20 }, features: [
    { properties: { layer: 'roads', kind: 'track' }, geometry: { type: 'LineString',
      coordinates: [[16, 38.005], [16.01, 38.005]] } },
    { properties: { layer: 'contours', elevation: 100 }, geometry: { type: 'LineString',
      coordinates: [[16, 38.008], [16.01, 38.008]] } },
  ] };
const metricOptions = { metricId: '1', geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [
  { id: 1, region: 'Forest', parcel: '1', coppice: false, aveAge: 37 },
] };
scale.value = '10000';
await showMetricPrint({ ...metricOptions, loadContext: async () => sourceContext });
assert.equal(layers.roads.disabled, false);
assert.equal(layers.roads.checked, true);
assert.equal(layers.water.disabled, true, 'unavailable layer controls stay disabled');
assert.equal(layers.contours.checked, true);
assert.match(contourNote.textContent, /DSM/);
assert.equal(contextStatus.textContent, '');
sourceContext.features[0].geometry.coordinates[0][0] = 17;
download.listeners.click();
assert.ok(pdfHasText(await blob.text(), 'OpenStreetMap'));
layers.contours.checked = false;
layers.contours.listeners.change();
assert.equal(contourNote.textContent, '');
download.listeners.click();
assert.ok(!pdfHasText(await blob.text(), 'Copernicus'), 'source credits follow selected layers');
layers.roads.checked = false;
layers.roads.listeners.change();
download.listeners.click();
assert.ok(!pdfHasText(await blob.text(), 'OpenStreetMap'));
cancel.listeners.click();

await showMetricPrint({ ...metricOptions, loadContext: async () => { throw new Error('Offline'); } });
assert.equal(download.disabled, false, 'missing context never prevents a plain parcel PDF');
assert.match(contextStatus.textContent, /Impossibile/);
cancel.listeners.click();
let finishLoading;
Object.values(layers).forEach(control => { control.checked = false; control.disabled = true; });
openMetricPrint({ ...metricOptions, loadContext: () => new Promise(resolve => { finishLoading = resolve; }) });
await flush();
assert.equal(download.disabled, true, 'wait for the optional layer snapshot before enabling download');
cancel.listeners.click();
finishLoading(sourceContext);
await flush();
assert.equal(container.children.length, 0, 'late context results never reopen a closed modal');

await showMetricPrint({ ...metricOptions, loadContext: async () => ({ available: false, stale: true }) });
assert.equal(download.disabled, false);
assert.match(contextStatus.textContent, /aggiornati/);
cancel.listeners.click();
await showMetricPrint({ ...metricOptions, loadContext: async () => ({ ...sourceContext,
  build: { dem_source: 'local.tif', contour_interval_m: 25 } }) });
assert.match(contourNote.textContent, /25 m/);
assert.ok(!contourNote.textContent.includes('DSM'), 'a local terrain model is not described as Copernicus DSM');
cancel.listeners.click();
let failLoading;
openMetricPrint({ ...metricOptions, loadContext: () => new Promise((resolve, reject) => { failLoading = reject; }) });
await flush();
cancel.listeners.click();
failLoading(new Error('Late network failure'));
await flush();
assert.equal(container.children.length, 0, 'late failures do not reopen or replace the current modal');
console.log('Print UI: fixed-scale snapshots, validation recovery, repeated downloads and close passed.');
