// Regression tests for the Campionamenti page async selection state.
// Run with: node apps/campionamenti/static/campionamenti/js/campionamenti.test.mjs

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-campionamenti-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'campionamenti'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'campionamenti', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
fs.writeFileSync(path.join(staticRoot, 'campionamenti', 'js', 'grid-planner.js'), `
export class GridPlanner {
  constructor(opts) {
    this.opts = opts;
    this.inited = false;
    this.destroyed = false;
    globalThis.__gridPlannerInstances.push(this);
  }
  init() { this.inited = true; }
  destroy() { this.destroyed = true; }
}
`);
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

let passed = 0;
let failed = 0;
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}: expected ${e}, got ${a}`);
  }
}

class MockElement {
  constructor(tag) {
    this.tagName = tag.toLowerCase();
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.className = '';
    this.id = '';
    this.textContent = '';
    this.value = '';
    this.name = '';
    this.type = '';
    this.hidden = false;
    this.href = '';
    this.rel = '';
    this.removed = false;
    this._listeners = {};
    this.style = { setProperty: (k, v) => { this.style[k] = v; } };
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
        const shouldAdd = force === undefined ? !next.has(name) : Boolean(force);
        if (shouldAdd) next.add(name);
        else next.delete(name);
        this._setClasses(next);
        return shouldAdd;
      },
    };
  }
  _classes() { return new Set(this.className.split(/\s+/).filter(Boolean)); }
  _setClasses(classes) { this.className = [...classes].join(' '); }
  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  append(...children) { for (const child of children) this.appendChild(child); }
  replaceChildren(...children) {
    this.children = [];
    for (const child of children) this.appendChild(child);
  }
  remove() { this.removed = true; }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  async click() {
    const event = { target: this, preventDefault() {} };
    let node = this;
    while (node) {
      for (const fn of node._listeners?.click || []) await fn(event);
      node = node.parentNode;
    }
  }
  removeEventListener(type, fn) {
    this._listeners[type] = (this._listeners[type] || []).filter(f => f !== fn);
  }
  matches(sel) {
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    if (sel.startsWith('.')) return this.classList.contains(sel.slice(1));
    const attr = sel.match(/^\[([^=\]]+)(?:="([^"]*)")?\](?:\.([A-Za-z0-9_-]+))?$/);
    if (attr) {
      const [, rawName, expected, cls] = attr;
      const actual = rawName.startsWith('data-') ? this.dataset[rawName.slice(5)] : this[rawName];
      if (expected !== undefined && actual !== expected) return false;
      if (expected === undefined && actual === undefined) return false;
      return !cls || this.classList.contains(cls);
    }
    return this.tagName === sel.toLowerCase();
  }
  closest(sel) {
    let node = this;
    while (node) {
      if (node.matches(sel)) return node;
      node = node.parentNode;
    }
    return null;
  }
  querySelector(sel) { return this._find(sel); }
  querySelectorAll(sel) { return this._findAll(sel); }
  _find(sel) {
    if (sel.includes(' ')) {
      const [head, ...tail] = sel.split(/\s+/);
      for (const el of this._findAll(head)) {
        const found = el._find(tail.join(' '));
        if (found) return found;
      }
      return null;
    }
    if (this.matches(sel)) return this;
    for (const child of this.children) {
      const found = child._find?.(sel);
      if (found) return found;
    }
    return null;
  }
  _findAll(sel) {
    const out = [];
    if (sel.includes(' ')) {
      const [head, ...tail] = sel.split(/\s+/);
      for (const el of this._findAll(head)) out.push(...el._findAll(tail.join(' ')));
      return out;
    }
    if (this.matches(sel)) out.push(this);
    for (const child of this.children) out.push(...(child._findAll?.(sel) || []));
    return out;
  }
  cloneNode(deep) {
    const clone = new MockElement(this.tagName);
    clone.dataset = { ...this.dataset };
    clone.className = this.className;
    clone.id = this.id;
    clone.textContent = this.textContent;
    clone.value = this.value;
    clone.name = this.name;
    clone.type = this.type;
    clone.hidden = this.hidden;
    if (deep) for (const child of this.children) clone.appendChild(child.cloneNode(true));
    return clone;
  }
}

function el(tag, { id = '', className = '', dataset = {} } = {}, children = []) {
  const node = new MockElement(tag);
  node.id = id;
  node.className = className;
  node.dataset = { ...dataset };
  for (const child of children) node.appendChild(child);
  return node;
}

function section(key) {
  const header = el('div', { className: 'collapsible-header', dataset: { section: key } });
  const body = el('div', { className: 'collapsible-body', dataset: { section: key } });
  return [header, body];
}

function buildCampionamentiTemplate() {
  const frag = el('fragment');
  frag.appendChild(el('button', { dataset: { action: 'new-grid' } }));
  for (const key of ['g', 'r', 't']) frag.append(...section(key));
  return frag;
}

function buildGridModal() {
  const root = el('div', { id: 'campionamenti-grid-modal' });
  root.appendChild(el('button', { className: 'modal-tab', dataset: { path: 'auto' } }));
  root.appendChild(el('div', { className: 'modal-tab-body grid-path-auto' }));
  root.appendChild(el('div', { id: 'campionamenti-grid-planner-host' }));
  root.appendChild(el('form', { id: 'campionamenti-grid-form-empty' }));
  root.appendChild(el('button', { dataset: { action: 'cancel' } }));
  return root;
}

const contentEl = el('main');
const modalEl = el('div', { id: 'modal-container' });
const links = [];
const templates = {
  'tmpl-campionamenti-page': { content: buildCampionamentiTemplate() },
};

globalThis.document = {
  documentElement: { lang: 'it' },
  body: { dataset: { csrf: 'csrf-token', role: 'reader' } },
  head: { appendChild: link => links.push(link) },
  createElement: tag => el(tag),
  createDocumentFragment: () => el('fragment'),
  addEventListener() {},
  removeEventListener() {},
  getElementById(id) {
    if (id === 'content') return contentEl;
    if (id === 'modal-container') return modalEl;
    return templates[id] || null;
  },
  querySelector(sel) {
    if (sel.startsWith('#modal-container ')) {
      return modalEl.querySelector(sel.slice('#modal-container '.length));
    }
    const href = sel.match(/^link\[href="([^"]+)"\]$/)?.[1];
    return href ? links.find(link => link.href === href && !link.removed) || null : null;
  },
  querySelectorAll() { return []; },
};

globalThis.location = { pathname: '/campionamenti', search: '' };
globalThis.history = {
  replaceState(_state, _title, url) {
    const u = new URL(url, 'https://example.test');
    globalThis.location = { pathname: u.pathname, search: u.search };
  },
  pushState(_state, _title, url) {
    const u = new URL(url, 'https://example.test');
    globalThis.location = { pathname: u.pathname, search: u.search };
  },
};
globalThis.window = { addEventListener() {} };
Object.defineProperty(globalThis, 'crypto', {
  configurable: true,
  value: { randomUUID: () => 'nonce-1' },
});
globalThis.__gridPlannerInstances = [];
globalThis.DOMParser = class {
  parseFromString() {
    return { body: { childNodes: [buildGridModal()] } };
  }
};

function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}

const flushAsyncWork = () => new Promise(resolve => setTimeout(resolve, 0));

const S = await import(staticModule('base/js/strings.js'));
const { ROW_ID } = await import(staticModule('base/js/constants.js'));
const cache = await import(staticModule('base/js/cache.js'));

const treeLoads = new Map([
  ['/api/campionamenti/trees/1/', deferred()],
  ['/api/campionamenti/trees/2/', deferred()],
]);
const fetches = [];

function digest(columns, rows) {
  return { columns, rows };
}

const payloads = new Map([
  ['/api/campionamenti/surveys/data/', digest(
    [ROW_ID, S.COL_NAME, S.COL_GRID, S.COL_N_AREAS_VISITED, S.COL_N_AREAS_TOTAL, S.COL_DATE_FIRST, S.COL_DATE_LAST],
    [[1, 'Rilevamento A', 10, 0, 0, '', ''], [2, 'Rilevamento B', 10, 0, 0, '', '']],
  )],
  ['/api/campionamenti/grids/data/', digest(
    [ROW_ID, S.COL_NAME, S.COL_N_AREAS, S.COL_REGIONS, S.COL_N_SURVEYS, S.COL_LAST_UPDATE],
    [[10, 'Griglia', 0, '', 2, '2026-01-01']],
  )],
  ['/api/campionamenti/sample-areas/data/', digest([ROW_ID, S.COL_GRID], [])],
  ['/api/campionamenti/samples/data/', digest([ROW_ID, S.COL_SURVEY, S.COL_SAMPLE_AREA, S.COL_N_TREES], [])],
  ['/api/geo/terreni.geojson', { type: 'FeatureCollection', features: [] }],
  ['/api/campionamenti/grid/form/', { html: '<div id=\"campionamenti-grid-modal\"></div>' }],
]);

function response(data, lastModified = 'v1') {
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? lastModified : null },
    json: async () => data,
  };
}

globalThis.fetch = async (url) => {
  fetches.push(url);
  if (treeLoads.has(url)) {
    return response(await treeLoads.get(url).promise);
  }
  if (!payloads.has(url)) throw new Error(`unexpected fetch ${url}`);
  return response(payloads.get(url));
};

const campionamenti = await import(staticModule('campionamenti/js/campionamenti.js'));

await campionamenti.mount({});
campionamenti.onQueryChange({ s: '2' });

treeLoads.get('/api/campionamenti/trees/2/').resolve(digest([ROW_ID], [[2]]));
await flushAsyncWork();
treeLoads.get('/api/campionamenti/trees/1/').resolve(digest([ROW_ID], [[1]]));
await flushAsyncWork();

fetches.length = 0;
await cache.refreshVisible();
const visibleTreeFetches = fetches.filter(url => url.includes('/trees/'));
eq(visibleTreeFetches, ['/api/campionamenti/trees/2/'],
   'stale survey selection does not replace the visible sampled-trees digest');

await contentEl.querySelector('[data-action="new-grid"]').click();
await flushAsyncWork();
await modalEl.querySelector('[data-path="auto"]').click();
eq(globalThis.__gridPlannerInstances.length, 1, 'auto grid tab lazily creates one planner');
const planner = globalThis.__gridPlannerInstances[0];
eq(planner.inited, true, 'auto grid tab initializes the planner');
await modalEl.querySelector('[data-action="cancel"]').click();
eq(planner.destroyed, true, 'grid planner is destroyed when the modal is dismissed');

campionamenti.unmount();

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
