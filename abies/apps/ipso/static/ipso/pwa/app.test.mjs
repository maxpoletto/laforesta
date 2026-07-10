// Regression tests for Ipso app-shell upload safety ordering.
// Run with: node apps/ipso/static/ipso/pwa/app.test.mjs

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const appSource = fs.readFileSync(path.join(here, 'app.js'), 'utf8') + `\n` +
  `globalThis.__ipsoAppTest = { State, boot, onEnd, showResumeModal, prefillNumber, ` +
  `validateReference, validateTerreniFeatures, restoreCachedBootResources, ` +
  `refreshBootResources };\n`;

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
    this.value = '';
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this._listeners = {};
    this.classList = {
      add: (...names) => this._setClasses(new Set([...this._classes(), ...names])),
      remove: (...names) => {
        const next = this._classes();
        for (const name of names) next.delete(name);
        this._setClasses(next);
      },
      contains: name => this._classes().has(name),
      toggle: (name, force) => {
        const next = this._classes();
        const add = force === undefined ? !next.has(name) : force;
        if (add) next.add(name); else next.delete(name);
        this._setClasses(next);
      },
    };
  }
  get options() { return this.children; }
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
  setAttribute(name, value) { this[name] = String(value); }
  closest() { return { hidden: false }; }
  querySelector() { return null; }
  reset() {}
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
  element('modal-resume').className = 'hidden';
  element('banner-reference').className = 'banner hidden';
  element('banner-storage').className = 'banner hidden';
  element('resume-title');
  element('resume-body');
  element('resume-list');
  element('toast');

  const localValues = new Map([['ipso.bearer_token', 'test-token']]);
  const modes = {
    martellate: { id: 'martellate', enabled: true },
    samples: { id: 'samples', enabled: true },
    pai: { id: 'pai', enabled: true },
  };
  const strings = new Proxy({
    ERROR_GEO_UNAVAILABLE: 'geo unavailable',
    ERROR_HTTP_STATUS: (status) => `HTTP ${status}`,
    ERROR_TOKEN_MISSING: 'token missing',
    ERROR_REFERENCE_INVALID: 'invalid reference',
    ERROR_GEOJSON_INVALID: 'invalid geojson',
    TOAST_REFERENCE_LOAD_ERROR: (msg) => `reference error: ${msg}`,
    TOAST_TERRENI_LOAD_ERROR: (msg) => `terreni error: ${msg}`,
    TOAST_DB_OPEN_ERROR: (msg) => `db error: ${msg}`,
    TOAST_BOOT_CACHE_ERROR: (msg) => `cache error: ${msg}`,
    TOAST_REFERENCE_REQUIRED: 'reference required',
    REFERENCE_OFFLINE_WARNING: 'offline reference',
    STORAGE_WARNING: 'storage warning',
    GPS_PERMISSION_BANNER: 'gps denied',
    PRE_PICK_COMPRESA: 'pick region',
    PRE_PICK_SURVEY: 'pick survey',
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
  }, {
    get(target, property) {
      return Object.prototype.hasOwnProperty.call(target, property)
        ? target[property]
        : String(property);
    },
  });
  const context = {
    console,
    setTimeout: () => 0,
    clearTimeout: () => {},
    URLSearchParams,
    APP_VERSION: 'test',
    IPSO_BEARER_STORAGE_KEY: 'ipso.bearer_token',
    IPSO_SECRET_HASH_PARAM: 'secret',
    IPSO_REF_SPECIES: 'species',
    IPSO_REF_PARCELS: 'parcels',
    IPSO_REF_HYPSOMETRY: 'ipsometrica',
    IPSO_REF_SAMPLING: 'sampling',
    IPSO_REF_SURVEYS: 'surveys',
    IPSO_REF_SAMPLE_AREAS: 'sample_areas',
    IPSO_REF_PAI: 'pai',
    IPSO_REF_PRESERVED_TREES: 'preserved_trees',
    FIELD_SURVEY_ID: 'survey_id',
    FIELD_SAMPLE_GRID_ID: 'sample_grid_id',
    FIELD_SAMPLE_AREA_ID: 'sample_area_id',
    FIELD_MAX_TREE_NUMBER: 'max_tree_number',
    IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX: 'sampling_survey:',
    window: {
      AbiesGeoReady: Promise.resolve(),
      location: { hash: '', pathname: '/ipso/', search: '' },
      history: { replaceState() {} },
      addEventListener() {},
    },
    navigator: {},
    localStorage: {
      getItem(key) { return localValues.get(key) || ''; },
      setItem(key, value) { localValues.set(key, value); },
    },
    document: {
      createElement(tag) {
        const el = new MockElement(tag);
        if (tag === 'button') buttons.push(el);
        return el;
      },
      getElementById: element,
      querySelectorAll: () => [],
      addEventListener() {},
      visibilityState: 'visible',
    },
    S: strings,
    IpsoModes: {
      MARTELLATE: 'martellate',
      SAMPLES: 'samples',
      PAI: 'pai',
      get(id) { return modes[id] || modes.martellate; },
      defaultMode() { return modes.martellate; },
      all() { return []; },
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
      async openDb() { events.push('openDb'); return {}; },
      async getCachedBootResources() {
        events.push('getCachedBootResources');
        return { reference: null, terreni: null };
      },
      async cacheReference(_db, value) { events.push(['cacheReference', value]); },
      async cacheTerreni(_db, value) { events.push(['cacheTerreni', value]); },
      async listResumableSessions() { events.push('listResumableSessions'); return []; },
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
    createNumpad() {
      return { mount() {}, value() { return ''; }, setValue() {} };
    },
    downloadText() { events.push('download'); },
    fetch: async () => { throw new Error('offline'); },
  };
  vm.createContext(context);
  vm.runInContext(appSource, context, { filename: 'app.js' });
  return { context, events, elements, buttons, localValues };
}

function referenceFixture(version = 'cached') {
  return {
    reference_version: version,
    species: [],
    parcels: [{ compresa: 'Serra', particella: '1' }],
    ipsometrica: {},
    sampling: { surveys: [], sample_areas: [] },
    pai: { preserved_trees: [] },
  };
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

// Cached protected resources are validated and restored before network work.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture();
  const terreni = [{ type: 'Feature', properties: { particella: '1' } }];
  app.State.db = {};
  context.Store.getCachedBootResources = async () => ({ reference, terreni });
  await app.restoreCachedBootResources();
  eq(app.State.reference, reference, 'boot restores the last-good reference snapshot');
  eq(app.State.terreni, terreni, 'boot restores the last-good parcel geometry');

  let invalidRejected = false;
  try { app.validateReference({ parcels: [] }); } catch (_) { invalidRejected = true; }
  check(invalidRejected, 'invalid reference snapshots are rejected');
}

// A successful online refresh validates, persists, and adopts both resources.
{
  const { context, events } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture('fresh');
  const terreni = [{ type: 'Feature', properties: { particella: '2' } }];
  app.State.db = {};
  app.State.bearerToken = 'test-token';
  context.fetch = async (url) => ({
    ok: true,
    json: async () => url === 'reference.json'
      ? reference
      : { type: 'FeatureCollection', features: terreni },
  });
  const result = await app.refreshBootResources();
  eq(result, { reference: true, terreni: true }, 'online refresh reports both resources fresh');
  eq(app.State.reference, reference, 'online refresh adopts the fresh reference');
  eq(app.State.terreni, terreni, 'online refresh adopts fresh parcel geometry');
  check(events.some((event) => Array.isArray(event) && event[0] === 'cacheReference'),
        'online refresh persists the validated reference');
  check(events.some((event) => Array.isArray(event) && event[0] === 'cacheTerreni'),
        'online refresh persists validated parcel geometry');
}

// Failed refresh leaves last-good data in place and makes staleness visible.
{
  const { context, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture();
  const terreni = [{ type: 'Feature', properties: { particella: '1' } }];
  app.State.db = {};
  app.State.bearerToken = 'test-token';
  app.State.reference = reference;
  app.State.terreni = terreni;
  context.fetch = async () => ({ ok: false, status: 503 });
  const result = await app.refreshBootResources();
  eq(result, { reference: false, terreni: false }, 'offline refresh reports cached resources');
  eq(app.State.reference, reference, 'offline refresh preserves the cached reference');
  eq(app.State.terreni, terreni, 'offline refresh preserves cached parcel geometry');
  check(elements.get('banner-reference').textContent === 'offline reference',
        'offline refresh shows a persistent stale-data warning');
  check(!elements.get('banner-reference').classList.contains('hidden'),
        'offline stale-data warning is visible');
}

// Cold-start with cached resources must not await the opportunistic network
// refresh; IndexedDB and resumable sessions are read before fetch begins.
{
  const { context, events, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture();
  const terreni = [{ type: 'Feature', properties: { particella: '1' } }];
  context.Store.openDb = async () => { events.push('openDb'); return {}; };
  context.Store.getCachedBootResources = async () => {
    events.push('getCachedBootResources');
    return { reference, terreni };
  };
  context.Store.listResumableSessions = async () => {
    events.push('listResumableSessions');
    return [{ ...session, status: 'open' }];
  };
  context.fetch = (url) => {
    events.push('fetch:' + url);
    return new Promise(() => {});
  };
  await app.boot();
  const firstFetch = events.findIndex((event) =>
    typeof event === 'string' && event.startsWith('fetch:')
  );
  check(events.indexOf('openDb') < firstFetch, 'cold boot opens IndexedDB before network fetch');
  check(events.indexOf('getCachedBootResources') < firstFetch,
        'cold boot restores protected resources before network fetch');
  check(events.indexOf('listResumableSessions') < firstFetch,
        'cold boot lists resumable sessions before network fetch');
  check(!elements.get('modal-resume').classList.contains('hidden'),
        'cold boot exposes locally stored resumable sessions immediately');
}

// Devices upgraded with an already-open session may not have a resource
// snapshot yet. Even then, failed network bootstrap must expose export/discard
// actions instead of returning before the resume modal is built.
{
  const { context, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  context.Store.listResumableSessions = async () => [
    { ...session, status: 'open' },
  ];
  context.fetch = async () => ({ ok: false, status: 503 });
  await app.boot();
  check(!elements.get('modal-resume').classList.contains('hidden'),
        'upgraded offline device still exposes its pre-cache session');
  const resume = buttons.find((button) => button.textContent === 'Riprendi');
  await resume.click();
  check(app.State.session === null, 'recording cannot resume without valid reference data');
  check(elements.get('toast').textContent === 'reference required',
        'missing reference explains why recording cannot resume');
}

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
