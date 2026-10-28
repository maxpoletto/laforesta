import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Use the real form submission, modal rendering and HTTP helpers in a served
// static layout. Only DOM and the network are simulated.
class Element {
  constructor(tag) {
    Object.assign(this, { tag, children: [], listeners: {}, style: {},
      dataset: {}, className: '', name: '', value: '', disabled: false });
    this.classList = { add() {}, remove() {} };
  }
  appendChild(child) { child.parent = this; this.children.push(child); return child; }
  replaceChildren(...children) { this.children = []; children.forEach(c => this.appendChild(c)); }
  before(child) { this.parent.children.splice(this.parent.children.indexOf(this), 0, child); }
  addEventListener(event, fn) { (this.listeners[event] ??= []).push(fn); }
  async fire(event) {
    for (const fn of this.listeners[event] ?? []) await fn({ preventDefault() {} });
  }
  matches(selector) {
    if (selector.startsWith('.')) return this.className === selector.slice(1);
    if (selector === '[data-action="cancel"]') return this.dataset.action === 'cancel';
    const match = selector.match(/^(input|button)\[(name|type)="([^"]+)"\](:checked)?$/);
    if (match) return this.tag === match[1] && this[match[2]] === match[3]
      && (!match[4] || this.checked);
    return this.tag === selector;
  }
  querySelectorAll(selector) {
    return this.children.flatMap(c => [
      ...(c.matches(selector) ? [c] : []), ...c.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
}

function makeForm(version, method) {
  const form = new Element('form');
  for (const [name, value] of [['row_id', '7'], ['version', version],
    ['email', version === '1' ? 'old@example.com' : 'fresh@example.com']]) {
    const input = form.appendChild(new Element('input'));
    Object.assign(input, { name, value });
  }
  for (const value of ['password', 'oauth']) {
    Object.assign(form.appendChild(new Element('input')), {
      name: 'login_method', value, checked: method === value,
    });
  }
  form.appendChild(new Element('div')).className = 'password-login-only';
  const actions = form.appendChild(new Element('div'));
  actions.className = 'form-actions';
  actions.appendChild(new Element('button')).dataset.action = 'cancel';
  actions.appendChild(new Element('button')).type = 'submit';
  return form;
}

const modal = new Element('div');
globalThis.document = {
  documentElement: { lang: 'it' }, body: { dataset: { csrf: 'token' } },
  getElementById: () => modal,
  querySelector: () => modal.querySelector('form'),
  createElement: tag => new Element(tag),
  createDocumentFragment: () => new Element('fragment'),
  addEventListener() {}, removeEventListener() {},
};
globalThis.DOMParser = class {
  parseFromString(html) {
    const { version, method } = JSON.parse(html);
    return { body: { childNodes: [makeForm(version, method)] } };
  }
};
globalThis.FormData = class {
  constructor(form) { this.form = form; }
  *[Symbol.iterator]() {
    for (const input of this.form.querySelectorAll('input')) {
      if (input.name !== 'login_method' || input.checked) yield [input.name, input.value];
    }
  }
};

const here = path.dirname(fileURLToPath(import.meta.url));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-settings-form-'));
process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }));
fs.mkdirSync(path.join(root, 'impostazioni/js'), { recursive: true });
fs.cpSync(path.join(here, '../../../../base/static/base/js'), path.join(root, 'base/js'), { recursive: true });
fs.copyFileSync(path.join(here, 'entity-form.js'), path.join(root, 'impostazioni/js/entity-form.js'));
const { wireSettingsForm } = await import(pathToFileURL(path.join(root, 'impostazioni/js/entity-form.js')));
const { renderModalForm } = await import(pathToFileURL(path.join(root, 'base/js/forms.js')));

const cfg = { key: 'users', saveUrl: '/users/save/' };
const state = { digest: { columns: ['row_id', 'version'], rows: [[7, 1]] },
  table: { setData(data) { this.data = structuredClone(data); } } };
let form = renderModalForm(JSON.stringify({ version: '1', method: 'password' }));
wireSettingsForm(form, cfg, state);
const originalNonce = form.querySelector('input[name="nonce"]').value;
let response = { status: 400, data: { status: 'validation_error', message: 'Invalid email' } };
const submissions = [];
globalThis.fetch = async (url, options) => {
  submissions.push(JSON.parse(options.body));
  return { status: response.status, json: async () => response.data };
};

// Ordinary 400s leave entered values and nonce in place, ready for correction.
await form.fire('submit');
assert.equal(modal.querySelector('form'), form);
assert.equal(form.querySelector('.form-error').textContent, 'Invalid email');
assert.equal(form.querySelector('input[name="nonce"]').value, originalNonce);
assert.equal(form.querySelector('button[type="submit"]').disabled, false);

// A conflict loads current values/version, refreshes the local table and wires
// the replacement form. Never just stamp a new version onto stale input values.
response.data = { status: 'conflict', message: 'Review the current account',
  row_id: 7, record: [7, 2], patches: [{ data_id: 'users', record: [7, 2] }],
  html: JSON.stringify({ version: '2', method: 'oauth' }) };
await form.fire('submit');
const fresh = modal.querySelector('form');
assert.notEqual(fresh, form);
assert.equal(fresh.querySelector('input[name="version"]').value, '2');
assert.equal(fresh.querySelector('input[name="email"]').value, 'fresh@example.com');
assert.notEqual(fresh.querySelector('input[name="nonce"]').value, originalNonce);
assert.equal(fresh.querySelector('.form-error').textContent, response.data.message);
assert.deepEqual(state.table.data.rows, [[7, 2]]);
assert.equal(fresh.querySelector('.password-login-only').style.display, 'none');
const radios = fresh.querySelectorAll('input[name="login_method"]');
radios[0].checked = true; radios[1].checked = false;
await radios[0].fire('change');
assert.equal(fresh.querySelector('.password-login-only').style.display, '');
response = { status: 200, data: { patches: [{ data_id: 'users', record: [7, 3] }] } };
await fresh.fire('submit');
assert.equal(submissions.at(-1).version, '2');
assert.deepEqual(state.table.data.rows, [[7, 3]]);
assert.equal(modal.querySelector('form'), null);
console.log('Settings forms: validation, conflict refresh, controls and resubmission passed');
