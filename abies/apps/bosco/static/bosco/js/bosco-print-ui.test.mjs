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
const { showCalendarPrint } = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-print-ui.js')));
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
const cancel = new Element(), error = new Element();
const fragment = new Element();
fragment.querySelector = selector => ({
  '[data-target="print-preview"]': preview,
  '[data-action="download-map"]': download,
  '[data-target="print-error"]': error,
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
const data = { selected_plan: { id: 1, name: 'Plan A', year_start: 2026, year_end: 2040 },
  calendar: [{ parcel_id: 1, years: [2028, 2030] }] };
showCalendarPrint({ geojson: { features: [{ geometry,
  properties: { layer: 'Forest', name: 'Forest-1' } }] },
region: { id: 1, name: 'Forest' }, entries: [{ id: 1, region: 'Forest', parcel: '1' }],
data, date: new Date(2026, 10, 11) });
assert.equal(preview.children.length, 1, 'open a preview before downloading');
assert.equal(blob, undefined, 'opening preview does not download automatically');
data.selected_plan.name = 'Plan B';
data.calendar[0].years = [2040];
download.listeners.click();
assert.match(downloaded, /Forest.*2026.*2040.*\.pdf$/);
const pdf = await blob.text();
assert.ok(pdf.includes('Plan A') && pdf.includes('2028, 2030'), 'download uses the preview snapshot');
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
console.log('Calendar print UI: preview, reader access, snapshot download and close passed.');
