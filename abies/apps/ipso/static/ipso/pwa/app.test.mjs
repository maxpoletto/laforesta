// Regression tests for Ipso app-shell upload safety ordering.
// Run with: node apps/ipso/static/ipso/pwa/app.test.mjs

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const appSource = fs.readFileSync(path.join(here, 'app.js'), 'utf8') + `\n` +
  `globalThis.__ipsoAppTest = { State, onEnd, showResumeModal, prefillNumber };\n`;

let pass = 0;
const failures = [];
function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
}

class MockElement {
  constructor(tag, id = '') {
    this.tagName = tag.toLowerCase();
    this.id = id;
    this.children = [];
    this.parentNode = null;
    this.className = '';
    this.textContent = '';
    this.type = '';
    this._listeners = {};
    this.classList = {
      add: (...names) => this._setClasses(new Set([...this._classes(), ...names])),
      remove: (...names) => {
        const next = this._classes();
        for (const name of names) next.delete(name);
        this._setClasses(next);
      },
      contains: name => this._classes().has(name),
    };
  }
  _classes() { return new Set(this.className.split(/\s+/).filter(Boolean)); }
  _setClasses(classes) { this.className = [...classes].join(' '); }
  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  remove() {
    if (this.parentNode) {
      this.parentNode.children = this.parentNode.children.filter(c => c !== this);
      this.parentNode = null;
    }
  }
  replaceChildren(...children) {
    this.children = [];
    for (const child of children) this.appendChild(child);
  }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  async click() {
    for (const fn of this._listeners.click || []) {
      await fn({ target: this, preventDefault() {} });
    }
  }
}

function makeHarness() {
  const events = [];
  const elements = new Map();
  const buttons = [];
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new MockElement('div', id));
    return elements.get(id);
  };
  element('modal-confirm-end');
  element('modal-resume');
  element('resume-title');
  element('resume-body');
  element('resume-list');
  element('toast');

  const context = {
    console,
    setTimeout: () => 0,
    clearTimeout: () => {},
    window: { addEventListener() {} },
    document: {
      createElement(tag) {
        const el = new MockElement(tag);
        if (tag === 'button') buttons.push(el);
        return el;
      },
      getElementById: element,
      querySelectorAll: () => [],
    },
    S: {
      TOAST_EXPORT_ERROR: (msg) => `export error: ${msg}`,
      UPLOAD_RESUME_TITLE: 'Upload sospeso',
      RESUME_TITLE: 'Riprendi',
      RESUME_BODY: 'body',
      UPLOAD_RESUME_DO_NOW: 'Carica',
      UPLOAD_RESUME_KEEP_LOCAL: 'Mantieni locale',
      RESUME_RESUME: 'Riprendi',
      RESUME_EXPORT: 'Esporta',
      RESUME_DISCARD: 'Scarta',
      where: (sess) => sess.compresa || '',
    },
    IpsoModes: {
      SAMPLES: 'samples',
      PAI: 'pai',
    },
    session: {
      nextNumberDefault(trees) {
        const numbers = trees.map(tree => tree.numero)
          .filter(number => Number.isInteger(number));
        return numbers.length ? Math.max(...numbers) + 1 : null;
      },
    },
    Store: {
      STATUS_PENDING_UPLOAD: 'pending_upload',
      STATUS_OPEN: 'open',
      STATUS_EXPORTED: 'exported',
      STATUS_ABANDONED: 'abandoned',
      UPLOAD_STATUS_LOCAL_ONLY: 'local_only',
      async listTrees() {
        events.push('listTrees');
        return [{ id: 1, seq: 1, specie: 'Abete' }];
      },
      async setSessionStatus() { events.push('setSessionStatus'); },
      async setSessionUploadStatus() { events.push('setSessionUploadStatus'); },
    },
    csv: {
      formatFile() { events.push('csv'); return 'csv-text'; },
      filename() { return 'final.csv'; },
      formatDate: (ymd) => ymd,
    },
    upload: {
      buildUploadPayload() {
        events.push('buildPayload');
        throw new Error('validation failed');
      },
    },
    createUploadFlow() {
      return { enter() { events.push('uploadEnter'); }, wire() {} };
    },
    downloadText() { events.push('download'); },
  };
  vm.createContext(context);
  vm.runInContext(appSource, context, { filename: 'app.js' });
  return { context, events, elements, buttons };
}

const session = {
  id: 's1',
  status: 'pending_upload',
  mode: 'martellate',
  data: '2026-06-17',
  compresa: 'Serra',
  operatore: 'Mario',
  tree_count: 1,
};

// Ending a live session must download the local CSV before payload validation.
{
  const { context, events } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  app.State.session = { ...session, status: 'open' };
  await app.onEnd();
  check(events.indexOf('download') >= 0, 'onEnd downloads CSV even when payload validation throws');
  check(events.indexOf('download') < events.indexOf('buildPayload'),
        'onEnd downloads CSV before building the upload payload');
  check(!events.includes('setSessionStatus'), 'onEnd does not mark pending after payload validation fails');
}

// Retrying a pending upload from the resume modal has the same safety ordering.
{
  const { context, events, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  app.showResumeModal([{ ...session }]);
  const carica = buttons.find(b => b.textContent === 'Carica');
  check(Boolean(carica), 'resume modal renders the pending-upload Carica button');
  await carica.click();
  check(events.indexOf('download') >= 0, 'resume upload downloads CSV even when payload validation throws');
  check(events.indexOf('download') < events.indexOf('buildPayload'),
        'resume upload downloads CSV before building the upload payload');
}

// A prefill based on an older tree list must not replace a newer proposal when
// overlapping IndexedDB reads resolve out of order.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  const pending = [];
  let numberValue = '';

  app.State.db = {};
  app.State.session = { id: 's1', mode: 'martellate' };
  app.State.numpad = {
    value: () => numberValue,
    setValue: (_field, value) => { numberValue = value; },
  };
  context.Store.listTrees = () => new Promise(resolve => pending.push(resolve));

  const stalePrefill = app.prefillNumber();
  const freshPrefill = app.prefillNumber();
  pending[1]([{ numero: 1 }, { numero: 2 }]);
  await freshPrefill;
  eq(numberValue, '3', 'newer prefill proposes the post-save number');

  pending[0]([{ numero: 1 }]);
  await stalePrefill;
  eq(numberValue, '3', 'older prefill cannot replace the newer proposal');
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} app-shell tests passed`);
