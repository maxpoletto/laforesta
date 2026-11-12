import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
// Controller integration: HTTP is real postJSON; table/modal primitives are
// substitutes, whose rendering/submission contracts have their own tests.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-equation-import-ui-'));
copyGeodesicVendor(root);
process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }));
fs.cpSync(path.join(here, '../../../../base/static/base/js'), path.join(root, 'base/js'), { recursive: true });
fs.mkdirSync(path.join(root, 'impostazioni/js'), { recursive: true });
fs.copyFileSync(path.join(here, 'equation-imports.js'), path.join(root, 'impostazioni/js/equation-imports.js'));
const fixture = globalThis.fixture = { posts: [], responses: [], dismissed: [], errors: [], tables: [] };
globalThis.document = { documentElement: { lang: 'it' }, body: { dataset: { csrf: 'test' } } };
fs.writeFileSync(path.join(root, 'base/js/forms.js'), `
  export function renderModalForm(html) {
    const form = { html, querySelector() { return null; }, querySelectorAll() { return []; } };
    fixture.form = form; return form;
  }
  export function interceptSubmit(form, url, callbacks) { form.url = url; form.callbacks = callbacks; }
  export function showFormError(form, message) { form.error = message; }
`);
fs.writeFileSync(path.join(root, 'base/js/modals.js'), `
  export function dismiss() { for (const cb of fixture.dismissed.splice(0)) cb(); fixture.form = null; }
  export function onDismiss(cb) { fixture.dismissed.push(cb); }
  export function showError(message) { fixture.errors.push(message); }
`);
fs.writeFileSync(path.join(root, 'base/js/ui-widgets.js'), `
  import { dismiss } from './modals.js';
  export function wireActions(root, actions) { root.actions = actions; }
  export function wireCancelButtons(form, cb) { form.cancel = cb; }
`);
fs.writeFileSync(path.join(root, 'base/js/table.js'), `
  export class TableWrapper {
    constructor(options) { this.options = options; fixture.tables.push(this); }
    destroy() { this.destroyed = true; }
  }
`);
globalThis.fetch = async (url, options) => {
  fixture.posts.push({ url, body: JSON.parse(options.body) });
  const response = fixture.responses.shift();
  assert.ok(response, 'Every network call must be expected');
  if (response.wait) await response.wait;
  return { status: response.status, json: async () => response.data };
};
const { wireEquationImports, importMappings } = await import(pathToFileURL(path.join(root, 'impostazioni/js/equation-imports.js')));
let release;
const bytes = new Promise(resolve => { release = resolve; });
const fileInput = { files: [{ arrayBuffer: () => bytes }], value: '', click() {},
  addEventListener(kind, callback) { this.callback = callback; } };
const toolbar = {};
const section = { querySelector(selector) { return selector.includes('import-file') ? fileInput : toolbar; } };
let saved = 0;
const dispose = wireEquationImports(section, async () => { saved++; });

// A second toolbar click during file reading must not change the first file's kind.
toolbar.actions['import-height']();
fixture.responses.push({ status: 200, data: { html: 'preview-one', columns: [], rows: [] } });
const first = fileInput.callback();
toolbar.actions['import-volume']();
release(new TextEncoder().encode('CSV').buffer);
await first;
assert.equal(fixture.posts[0].body.kind, 'height');
assert.ok(fixture.form.url.endsWith('/confirm/'));
fixture.form.cancel();
assert.equal(saved, 0);
assert.equal(fixture.posts.length, 1); // Cancel has no write request.
assert.equal(fixture.tables[0].destroyed, true);

// Existing explicit mappings survive when resolved rows no longer show selectors.
assert.deepEqual(importMappings({ querySelectorAll: () => [] }, { species_map: { 2: 17 } }),
  { species_map: { 2: 17 } });

fixture.responses.push({ status: 200, data: { html: 'preview-two', columns: [], rows: [] } });
await fileInput.callback();
const prior = fixture.form;
fixture.responses.push({ status: 200, data: { html: 'fresh-preview', columns: [], rows: [] } });
await prior.callbacks.onConflict({ message: 'Review changed equations' });
assert.notEqual(fixture.form, prior);
assert.equal(fixture.form.html, 'fresh-preview');
assert.equal(fixture.form.error, 'Review changed equations');
await fixture.form.callbacks.onSuccess({});
assert.equal(saved, 1);
assert.equal(fixture.form, null);

fixture.responses.push({ status: 400, data: { message: 'Invalid CSV' } });
await fileInput.callback();
assert.equal(fixture.errors.at(-1), 'Invalid CSV');
assert.equal(saved, 1);
let releasePreview;
fixture.responses.push({ status: 200, data: { html: 'obsolete', columns: [], rows: [] },
  wait: new Promise(resolve => { releasePreview = resolve; }) });
const obsolete = fileInput.callback();
await new Promise(resolve => setImmediate(resolve));
dispose();
releasePreview();
await obsolete;
assert.equal(fixture.form, null, 'A preview cannot open after leaving Settings');
await prior.callbacks.onSuccess({});
assert.equal(saved, 1, 'An old write response cannot rebuild a page that was left');
console.log('Equation import UI: read race, cancel, mappings, conflict refresh and validation passed');
