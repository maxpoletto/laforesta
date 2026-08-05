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
  init() {
    this.inited = true;
    const cancel = document.createElement('button');
    cancel.dataset.action = 'cancel';
    this.opts.host.appendChild(cancel);
  }
  destroy() { this.destroyed = true; }
}
`);
fs.writeFileSync(path.join(staticRoot, 'campionamenti', 'js', 'rilevamenti-map.js'), `
export class RilevamentiMap {
  constructor(opts) {
    this.opts = opts;
    this.destroyed = false;
    this.wrapper = { syncBasemap: name => { this.basemap = name; } };
    this.leaflet = { on() {} };
    globalThis.__rilevamentiMapInstances.push(this);
  }
  setAreas(areas, visited) { this.areas = areas; this.visited = visited; }
  setActiveAreaId(id) { this.activeAreaId = id; }
  invalidateSize() { this.invalidated = true; }
  fitParcels() { this.fitted = true; }
  destroy() { this.destroyed = true; }
}
`);
fs.writeFileSync(path.join(staticRoot, 'base', 'js', 'tree-detail.js'), `
export class TreeDetail {
  constructor(opts) {
    this.opts = opts;
    this.rows = [...opts.digest.rows];
    this.setRowsCalls = [this.rows];
    this.destroyed = false;
    this.map = { leaflet: { on() {} } };
    this.node = document.createElement('div');
    this.node.className = 'tree-detail-stub';
    opts.container.appendChild(this.node);
    globalThis.__treeDetailInstances.push(this);
  }
  setRows(rows) {
    this.rows = [...rows];
    this.setRowsCalls.push(this.rows);
  }
  showMap() { this.showMapCalls = (this.showMapCalls || 0) + 1; }
  syncBasemap(name) { this.basemap = name; }
  destroy() {
    this.destroyed = true;
    this.node.remove();
  }
}
`);
fs.writeFileSync(path.join(staticRoot, 'base', 'js', 'router.js'), `
export function navigate(url, replace = false) {
  const target = new URL(url, 'https://example.test');
  const path = target.pathname + target.search;
  if (replace) history.replaceState(null, '', path);
  else history.pushState(null, '', path);
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
    this.offsetHeight = 0;
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
  remove() {
    this.removed = true;
    if (this.parentNode) {
      this.parentNode.children = this.parentNode.children.filter(c => c !== this);
      this.parentNode = null;
    }
  }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  dispatchEvent(event) {
    event.target ||= this;
    for (const fn of this._listeners[event.type] || []) fn(event);
    return true;
  }
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
  header.appendChild(el('span', { dataset: { field: 'title' } }));
  if (key === 'g') body.appendChild(el('select', { id: 'campionamenti-grid-select' }));
  if (key === 'r') {
    body.appendChild(el('select', { id: 'campionamenti-survey-select' }));
    body.appendChild(el('div', { dataset: { target: 'survey-summary' } }));
    body.appendChild(el('div', { dataset: { target: 'survey-map' } }));
  }
  if (key === 't') {
    body.appendChild(el('div', { dataset: { target: 'trees-empty' } }));
    body.appendChild(el('div', { dataset: { target: 'trees-table-host' } }));
    body.appendChild(el('div', { dataset: { target: 'trees-detail-host' } }));
  }
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
  root.appendChild(el('div', { className: 'modal-tabs' }, [
    el('button', { className: 'modal-tab', dataset: { path: 'empty' } }),
    el('button', { className: 'modal-tab', dataset: { path: 'auto' } }),
  ]));
  root.appendChild(el('div', { className: 'modal-tab-bodies' }, [
    el('form', { id: 'campionamenti-grid-form-empty', className: 'modal-tab-body', dataset: { path: 'empty' } }),
    el('div', { className: 'modal-tab-body', dataset: { path: 'auto' } }, [
      el('div', { id: 'campionamenti-grid-planner-host' }),
    ]),
  ]));
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
class MockSortableTable {
  constructor(opts) {
    this._allData = opts.data;
    this.data = opts.data;
    this.currentSort = opts.sort || null;
    this.currentPage = 1;
    this.onSort = opts.onSort;
    if (opts.controlsStart) opts.container.appendChild(opts.controlsStart);
    if (opts.controlsEnd) opts.container.appendChild(opts.controlsEnd);
  }
  setData(rows) { this._allData = rows; this.data = rows; }
  filter(fn) { this.data = this._allData.filter(fn); }
  clearFilter() { this.data = this._allData; }
  sort(column, _type, ascending) {
    this.currentSort = { column, ascending };
    this.onSort?.(column, ascending);
  }
  goToPage(page) { this.currentPage = page; }
  destroy() { this.destroyed = true; }
}

globalThis.window = { SortableTable: MockSortableTable, addEventListener() {} };
Object.defineProperty(globalThis, 'crypto', {
  configurable: true,
  value: { randomUUID: () => 'nonce-1' },
});
globalThis.__gridPlannerInstances = [];
globalThis.__rilevamentiMapInstances = [];
globalThis.__treeDetailInstances = [];
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
    [[1, 'Rilevamento Z', 10, 2, 4, '2025-01-01', '2025-02-01'],
     [2, 'Alberi da preservare', null, 30, 0, '1970-01-01', '1970-01-01'],
     [3, 'Abbattimenti urgenti', null, 1, 0, '2026-01-01', '2026-01-02'],
     [4, 'Rilevamento A', 10, 3, 9, '2025-03-01', '2025-03-02']],
  )],
  ['/api/campionamenti/grids/data/', digest(
    [ROW_ID, S.COL_NAME, S.COL_N_AREAS, S.COL_REGIONS, S.COL_N_SURVEYS, S.COL_LAST_UPDATE],
    [[10, 'Griglia', 0, '', 2, '2026-01-01']],
  )],
  ['/api/campionamenti/sample-areas/data/', digest([ROW_ID, S.COL_GRID], [])],
  ['/api/campionamenti/samples/data/', digest(
    [ROW_ID, S.COL_SURVEY, S.COL_SAMPLE_AREA, S.COL_N_TREES],
    [[100, 2, null, 1490], [101, 2, null, 5], [102, 3, null, 1]],
  )],
  ['/api/species/data/', digest(
    [ROW_ID, S.COL_NAME],
    [[1, 'Abete bianco'], [2, 'Faggio']],
  )],
  ['/api/geo/terreni.geojson', { type: 'FeatureCollection', features: [] }],
  ['/api/campionamenti/grid/form/', { html: '<div id=\"campionamenti-grid-modal\"></div>' }],
]);

const treeColumns = [
  ROW_ID, S.COL_SAMPLE_AREA, S.COL_REGION, S.COL_TREE_NUM, S.COL_SPECIES,
  S.COL_D_CM, S.COL_H_M, S.COL_V_M3, S.COL_LAT, S.COL_LON,
];
const freeTreeDigest = digest(treeColumns, [
  [201, null, 'A', 1, 'Abete bianco', 30, 20, 1.2, 38.1, 16.2],
  [202, null, 'A', 2, 'Faggio', 40, 22, 2.4, 38.2, 16.3],
]);

function response(data, lastModified = 'v1') {
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? lastModified : null },
    json: async () => data,
  };
}

let exportedDendrometryRowIds = null;
globalThis.fetch = async (url, options = {}) => {
  url = String(url);
  fetches.push(url);
  if (url.startsWith('/api/campionamenti/survey/dendrometry/export/') &&
      options.method === 'POST') {
    exportedDendrometryRowIds = JSON.parse(options.body || '{}').row_ids ?? null;
    return {
      status: 200,
      ok: true,
      headers: { get: h => h === 'Content-Disposition'
        ? 'attachment; filename="riassunto.zip"' : null },
      blob: async () => new Blob(['zip']),
    };
  }
  if (treeLoads.has(url)) {
    return response(await treeLoads.get(url).promise);
  }
  if (!payloads.has(url)) throw new Error(`unexpected fetch ${url}`);
  return response(payloads.get(url));
};

const campionamenti = await import(staticModule('campionamenti/js/campionamenti.js'));

await campionamenti.mount({});

const surveySummary = contentEl.querySelector('[data-target="survey-summary"]');
eq(
  surveySummary.children[0].textContent,
  'Griglia: Griglia · 2/4 aree visitate · dal 2025-01-01 al 2025-02-01',
  'structured survey summary retains grid and area progress',
);
const surveySelect = contentEl.querySelector('#campionamenti-survey-select');
const [freeSurveys, structuredSurveys] = surveySelect.children;
eq(
  surveySelect.children.map(group => group.label),
  [S.SAMPLES_FREE_SURVEYS_GROUP, S.SAMPLES_STRUCTURED_SURVEYS_GROUP],
  'survey picker groups free surveys before structured surveys',
);
eq(
  freeSurveys.children.map(option => [option.value, option.textContent]),
  [['3', 'Abbattimenti urgenti (1 albero)'],
   ['2', 'Alberi da preservare (1495 alberi)']],
  'free surveys are alphabetical and display summed tree counts',
);
eq(
  structuredSurveys.children.map(option => [option.value, option.textContent]),
  [['4', 'Rilevamento A (3/9 aree)'], ['1', 'Rilevamento Z (2/4 aree)']],
  'structured surveys are alphabetical and retain area-progress labels',
);

const surveyMap = contentEl.querySelector('[data-target="survey-map"]');
eq(surveyMap.hidden, false, 'structured survey keeps the sample-area map visible');

campionamenti.onQueryChange({ s: '2' });
eq(
  surveySummary.children[0].textContent,
  'Dal 1970-01-01 al 1970-01-01',
  'free survey summary displays only its capitalized date range',
);
eq(surveyMap.hidden, true, 'free survey hides the sample-area map');

treeLoads.get('/api/campionamenti/trees/2/').resolve(freeTreeDigest);
await flushAsyncWork();
treeLoads.get('/api/campionamenti/trees/1/').resolve(digest(
  treeColumns, [freeTreeDigest.rows[0]],
));
await flushAsyncWork();

fetches.length = 0;
await cache.refreshVisible();
const visibleTreeFetches = fetches.filter(url => url.includes('/trees/'));
eq(visibleTreeFetches, ['/api/campionamenti/trees/2/'],
   'stale survey selection does not replace the visible sampled-trees digest');
const treesHeaderSummary = contentEl.querySelector(
  '[data-target="trees-header-summary"]',
);
eq(treesHeaderSummary.textContent, '(2 alberi)',
   'free survey tree header omits the sample-area wording');

eq(globalThis.__treeDetailInstances.length, 1,
   'free survey mounts one shared tree-detail component below the table');
const detail = globalThis.__treeDetailInstances[0];
eq(detail.rows.map(row => row[0]), [201, 202],
   'tree detail starts with the sampled-tree table rows');
eq(detail.opts.pointColumnNames, { number: S.COL_TREE_NUM },
   'tree detail maps the sampled-tree number column');
eq(detail.opts.speciesNames, ['Abete bianco', 'Faggio'],
   'tree detail receives the global species palette');

const search = contentEl.querySelector('.table-search');
search.value = 'Faggio';
search.dispatchEvent({ type: 'input' });
await new Promise(resolve => setTimeout(resolve, 550));
await flushAsyncWork();
eq(detail.rows.map(row => row[0]), [202],
   'Filtra drives the free-survey map and dendrometry rows');

await detail.opts.onExport(detail.rows);
eq(exportedDendrometryRowIds, [202],
   'filtered dendrometry export submits only visible tree-sample rows');

campionamenti.onQueryChange({ s: '1' });
await flushAsyncWork();
eq(surveyMap.hidden, false, 'switching back restores the structured survey map');
eq(detail.destroyed, true,
   'structured surveys tear down the free-survey tree detail');

await contentEl.querySelector('[data-action="new-grid"]').click();
await flushAsyncWork();
await modalEl.querySelector('[data-path="auto"]').click();
eq(globalThis.__gridPlannerInstances.length, 1, 'auto grid tab lazily creates one planner');
const planner = globalThis.__gridPlannerInstances[0];
eq(planner.inited, true, 'auto grid tab initializes the planner');
eq(planner.opts.onCancel, undefined,
   'auto grid planner does not receive a private cancel callback');
await modalEl.querySelector(
  '#campionamenti-grid-planner-host [data-action="cancel"]',
).click();
eq(planner.destroyed, true,
   'standard cancel wiring dismisses and destroys the grid planner');

campionamenti.unmount();

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
