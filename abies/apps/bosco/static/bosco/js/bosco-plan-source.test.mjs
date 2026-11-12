import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-plan-source-'));
for (const app of ['base', 'bosco']) {
  fs.cpSync(path.resolve(here, `../../../../${app}/static/${app}/js`),
    path.join(tmp, app, 'js'), { recursive: true });
}
copyGeodesicVendor(tmp);
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const { wirePlanSource } = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-plan-source.js')));
const select = {
  value: '', disabled: false, options: [], listeners: new Map(),
  replaceChildren() { this.options = []; },
  appendChild(option) { this.options.push(option); },
  addEventListener(event, fn) { this.listeners.set(event, fn); },
  removeEventListener(event) { this.listeners.delete(event); },
};
const message = { textContent: '' };
const host = { querySelector: selector => selector === 'select' ? select : message };
globalThis.document = {
  body: { dataset: { csrf: 'test-token' } },
  createElement() { return {}; },
};
let selected = 1;
let failure = false;
let refreshes = 0;
globalThis.fetch = async (url, options = {}) => {
  if (!options.method) {
    assert.equal(url, '/api/bosco/plan-source/data/');
    return { ok: true, status: 200, headers: { get: () => null }, json: async () => ({
      active_id: selected,
      plans: [{ id: 1, name: 'First', year_start: 2026, year_end: 2040 },
        { id: 2, name: 'Second', year_start: 2041, year_end: 2055 }],
    }) };
  }
  assert.equal(url, '/api/bosco/plan-source/save/');
  const body = JSON.parse(options.body);
  assert.equal(typeof body.nonce, 'string');
  assert.equal(options.headers['X-CSRFToken'], 'test-token');
  if (!failure) selected = body.harvest_plan_id;
  return { status: failure ? 400 : 200, json: async () => ({
    status: failure ? 'validation_error' : 'ok', errors: ['Invalid plan'],
  }) };
};

const controller = wirePlanSource(host, {
  canEdit: true, onRefresh: async () => { refreshes++; },
});
await controller.load();
assert.equal(select.value, '1');
assert.equal(select.disabled, false);
assert.match(select.options[0].textContent, /First.*2026.*2040/);
select.value = '2';
await select.listeners.get('change')();
assert.equal(selected, 2);
assert.equal(refreshes, 1);
assert.equal(select.disabled, false);
failure = true;
select.value = '1';
await select.listeners.get('change')();
assert.equal(select.value, '2', 'failed saves restore the saved selection');
assert.equal(select.disabled, false, 'validation errors do not strand the control');
assert.ok(message.textContent);
assert.equal(refreshes, 1);
controller.destroy();
assert.equal(select.listeners.has('change'), false);

const reader = wirePlanSource(host, { canEdit: false, onRefresh() {} });
await reader.load();
assert.equal(select.disabled, true, 'readers see the saved plan without editing it');
reader.destroy();
console.log('Bosco plan selector: persistence, refresh, errors and permissions passed.');
