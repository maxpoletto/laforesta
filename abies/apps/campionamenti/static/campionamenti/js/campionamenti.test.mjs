import { csvImportTemplate } from '../../../../../test/js-csv-modal.mjs';
import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
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
    this.visibilityAtCreation = this.isMapVisible();
    this.setRowsCalls = [this.rows];
    this.destroyed = false;
    this.map = opts.mapEnabled === false ? null : { leaflet: { on() {} } };
    this.node = document.createElement('div');
    this.node.className = 'tree-detail-stub';
    opts.container.appendChild(this.node);
    globalThis.__treeDetailInstances.push(this);
  }
  isMapVisible() {
    for (let node=this.opts.mapContainer;node;node=node.parentNode) {
      if (node.hidden) return false;
    }
    return true;
  }
  setData(digest, {rows=digest.rows}={}) {
    this.visibilityOnUpdate=this.isMapVisible();
    this.opts.digest=digest; this.setRows(rows);
  }
  setRows(rows) {
    this.rows = [...rows];
    this.setRowsCalls.push(this.rows);
  }
  showMap() { this.visibilityOnShow=this.isMapVisible(); this.showMapCalls = (this.showMapCalls || 0) + 1; }
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
copyGeodesicVendor(staticRoot);
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
  focus() { this.focused = true; }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  setAttribute(name, value) { this[name] = String(value); }
  removeAttribute(name) { delete this[name]; }
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
  if (key === 'g') {
    body.appendChild(el('select', {id:'campionamenti-grid-select'}));
    const controls = el('div', {dataset:{target:'grid-summary-row'}});
    controls.appendChild(el('div', {dataset:{target:'grid-summary'}}));
    controls.appendChild(el('button', {dataset:{action:'import-grid-csv'}}));
    body.appendChild(controls);
  }
  if (key === 'r') {
    body.appendChild(el('select', { id: 'campionamenti-survey-select' }));
    body.appendChild(el('div', { dataset: { target: 'survey-summary' } }));
    const map = el('section', {dataset:{target:'survey-map-section'}});
    map.appendChild(el('h3',{dataset:{field:'survey-map-title'}}));
    map.appendChild(el('p',{className:'subsection-note',dataset:{target:'survey-map-help'}}));
    map.appendChild(el('div', {dataset:{target:'survey-map'}}));
    body.appendChild(map);
    body.appendChild(el('div', { dataset: { target: 'trees-empty' } }));
    body.appendChild(el('div', { dataset: { target: 'trees-table-host' } }));
    body.appendChild(el('div', { dataset: { target: 'trees-detail-host' } }));
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
  for (const key of ['g', 'r']) frag.append(...section(key));
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

function buildTreeForm(rowId) {
  const form = el('form', { id: 'campionamenti-tree-form' });
  const input = el('input');
  input.name = ROW_ID;
  input.value = rowId;
  form.appendChild(input);
  form.appendChild(el('button', { dataset: { action: 'cancel' } }));
  return form;
}

const contentEl = el('main');
const modalEl = el('div', { id: 'modal-container' });
const links = [];
const templates = {
  'tmpl-csv-import-modal': {content: csvImportTemplate(tag => el(tag))},
  'tmpl-campionamenti-page': { content: buildCampionamentiTemplate() },
};

globalThis.document = {
  documentElement: { lang: 'it' },
  body: { dataset: { csrf: 'csrf-token', role: 'writer' } },
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
const tableInstances = [];
class MockSortableTable {
  constructor(opts) {
    this._allData = opts.data;
    this.data = opts.data;
    this.columns = opts.columns;
    this.currentSort = opts.sort || null;
    this.currentPage = 1;
    this.onSort = opts.onSort;
    this.rowsPerPage = opts.rowsPerPage;
    if (opts.controlsStart) opts.container.appendChild(opts.controlsStart);
    if (opts.controlsEnd) opts.container.appendChild(opts.controlsEnd);
    const actions = opts.columns.find(column => column.key === '_actions');
    if (actions && opts.data.length) {
      const row = el('div', {
        className: 'sortable-table-row', dataset: { index: '0' },
      });
      const cell = el('div');
      actions.renderCell(cell, null, opts.data[0]);
      row.appendChild(cell);
      opts.container.appendChild(row);
    }
    tableInstances.push(this);
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
  parseFromString(source) {
    const treeRowId = source.match(/data-test-tree-row-id="([^"]*)"/)?.[1];
    if (treeRowId !== undefined) {
      return { body: { childNodes: [buildTreeForm(treeRowId)] } };
    }
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
const {matchesSearch,searchTerms}=await import(staticModule('base/js/table.js'));
const cache = await import(staticModule('base/js/cache.js'));

const treeLoads = new Map([
  ['/api/campionamenti/trees/1/', deferred()],
  ['/api/campionamenti/trees/2/', deferred()],
  ['/api/campionamenti/trees/4/', deferred()],
]);
const fetches = [];
const csvImports = [];

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
  ['/api/campionamenti/sample-areas/data/', digest(
    [ROW_ID, S.COL_GRID, S.COL_REGION, S.COL_PARCEL, S.COL_NUMBER, S.COL_LAT, S.COL_LON],
    [[50, 10, 'Serra', '01b', '7', 38.1, 16.2],
     [51, 10, 'Serra', '01b', '72', 38.2, 16.3],
     [52, 10, 'Serra San Bruno', '9', '7', 38.3, 16.4],
     [53, 10, 'Serra', '01b', '99', 38.4, 16.5]],
  )],
  ['/api/campionamenti/samples/data/', digest(
    [ROW_ID, S.COL_SURVEY, S.COL_SAMPLE_AREA, S.COL_N_TREES],
    [[99, 1, 50, 12], [100, 2, null, 1490], [101, 2, null, 5], [102, 3, null, 1]],
  )],
  ['/api/species/data/', digest(
    [ROW_ID, S.COL_NAME],
    [[1, 'Abete bianco'], [2, 'Faggio']],
  )],
  ['/api/geo/map.geojson', { type: 'FeatureCollection', features: [] }],
  ['/api/campionamenti/grid/form/', { html: '<div id=\"campionamenti-grid-modal\"></div>' }],
  ['/api/campionamenti/tree/form/201/', {
    html: '<form data-test-tree-row-id=\"\"></form>',
  }],
]);

const treeColumns = [
  ROW_ID, S.COL_SAMPLE_AREA, S.COL_REGION, S.COL_PARCEL, S.COL_TREE_NUM, S.COL_SPECIES,
  S.COL_D_CM, S.COL_H_M, S.COL_V_M3, S.COL_LAT, S.COL_LON, S.COL_ACC_M, S.COL_AREA_NUM,
  S.COL_SAMPLE_DATE,S.COL_TYPE,S.COL_COPPICE_SHOOT,S.COL_COPPICE_STD,S.COL_L10_MM,S.COL_MASS_Q,S.COL_PRESERVED,
];
const freeTreeDigest = digest(treeColumns, [
  [201, null, 'A', '1', 1, 'Abete bianco', 30, 20, 1.2, 38.1, 16.2, 4, ''],
  [202, null, 'A', '10b', 2, 'Faggio', 40, 22, 2.4, 38.2, 16.3, null, ''],
]);
freeTreeDigest.diameter_class_mode = 'shifted_up';

function response(data, lastModified = 'v1') {
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? lastModified : null },
    json: async () => data,
  };
}

let exportedDendrometryRowIds = null;
let exportedTreeRowIds = null;
globalThis.fetch = async (url, options = {}) => {
  url = String(url);
  fetches.push(url);
  if (url.endsWith('/import-csv/') && options.method === 'POST') {
    csvImports.push([url, JSON.parse(options.body)]);
    return {status:400,ok:false,json:async()=>({status:'validation_error', errors:['Riga 2: Altezza mancante.']})};
  }
  if (url.startsWith('/api/campionamenti/survey/export-csv/') && options.method === 'POST') {
    exportedTreeRowIds = JSON.parse(options.body).row_ids;
    return {status:200,ok:true,headers:{get:()=>null},blob:async()=>new Blob(['csv'])};
  }
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
  'Dal 2025-01-01 al 2025-02-01. 12 alberi campionati.',
  'survey overview shows dates and the complete sampled-tree count',
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
const mappedSurvey = globalThis.__rilevamentiMapInstances.at(-1);
const mappedArea = mappedSurvey.areas[0];
eq(S.TOOLTIP_SAMPLE_AREA(mappedArea), 'Serra 01b / adc 7',
   'map tooltip retains string parcel labels and the sample-area number');
eq(S.TOOLTIP_SAMPLE_AREA_VISITED(
  mappedArea, S.SAMPLES_TREE_COUNT(mappedSurvey.visited.get(50).treeCount)),
  'Serra 01b / adc 7 / 12 alberi',
  'visited-area tooltip retains the count from the selected survey');
eq(contentEl.querySelector('[data-field="survey-map-title"]').textContent,
   'Mappa delle aree di saggio', 'structured survey map has an explicit title');
eq(contentEl.querySelector('[data-target="survey-map-help"]').hidden, false,
   'sample-area map explains its filter interaction');

campionamenti.onQueryChange({ s: '2' });
eq(
  surveySummary.children[0].textContent,
  'Dal 1970-01-01 al 1970-01-01. 1495 alberi campionati.',
  'free survey overview shows dates and count before tree rows finish loading',
);
eq(globalThis.__rilevamentiMapInstances.at(-1).destroyed, true, 'free survey tears down the sample-area map');

treeLoads.get('/api/campionamenti/trees/2/').resolve(freeTreeDigest);
await flushAsyncWork();
const structuredTreeDigest = digest(treeColumns, [
  [301,50,'Serra','01b',1,'Abete bianco',30,20,1.2,38.1,16.2,4,'7'],
  [302,51,'Serra','01b',1,'Faggio',40,22,2.4,38.2,16.3,3,'72'],
  [303,52,'Serra San Bruno','9',1,'Faggio',40,22,2.4,38.3,16.4,3,'7'],
]);
treeLoads.get('/api/campionamenti/trees/1/').resolve(structuredTreeDigest);
await flushAsyncWork();
eq(surveySummary.children[0].textContent, 'Dal 1970-01-01 al 1970-01-01. 2 alberi campionati.',
   'the survey overview uses newly loaded rows without waiting for background refresh');

fetches.length = 0;
await cache.refreshVisible();
const visibleTreeFetches = fetches.filter(url => url.includes('/trees/'));
eq(visibleTreeFetches, ['/api/campionamenti/trees/2/'],
   'stale survey selection does not replace the visible sampled-trees digest');
const treesHeaderSummary = contentEl.querySelector(
  '[data-target="trees-scope-summary"]',
);
eq(treesHeaderSummary, null, 'area selection is represented only in the visible filter');
eq(contentEl.querySelector('[data-field="survey-map-title"]').textContent,
   'Mappa degli alberi', 'free survey map has an explicit title');
eq(contentEl.querySelector('[data-target="survey-map-help"]').hidden, true,
   'tree maps have no sample-area instruction');
eq(surveySummary.children[0].textContent, 'Dal 1970-01-01 al 1970-01-01. 2 alberi campionati.',
   'loaded tree rows update the survey overview count');

const treeTable = tableInstances.find(table =>
  table.columns.some(column => column.key === S.COL_ACC_M));
const treeParcelColumn = treeTable.columns.find(column => column.key === S.COL_PARCEL);
eq(treeParcelColumn.type, 'natural', 'sample parcel column uses natural sorting');
eq(treeParcelColumn.searchMatch, 'exact', 'sample parcel filter uses whole-value matching');
const treeAccColumn = treeTable.columns.find(
  column => column.key === S.COL_ACC_M,
);
const treeLonIndex = treeTable.columns.findIndex(
  column => column.key === S.COL_LON,
);
eq(treeTable.columns.indexOf(treeAccColumn), treeLonIndex + 1,
   'sampled-tree accuracy is immediately after longitude');
eq(treeAccColumn.formatter(null), '-',
   'sampled-tree null accuracy is displayed as a dash');
eq(treeAccColumn.formatter(4), '4',
   'sampled-tree accuracy is displayed as integer meters');
const labelledFields=[
 [S.COL_SAMPLE_DATE,'Data','data','2026-11-15','2026-11-15'],
 [S.COL_REGION,'Compresa','compresa','Serra','Serra'],
 [S.COL_PARCEL,'Particella','particella','1','1'],
 [S.COL_AREA_NUM,'Area','area','72','72'],
 [S.COL_TREE_NUM,'Albero','albero',17,'17'],
 [S.COL_SPECIES,'Specie','specie','Abete bianco','Abete'],
 [S.COL_TYPE,'Tipo','tipo','Fustaia','Fustaia'],
 [S.COL_COPPICE_SHOOT,'Poll','poll',2,'2'],
 [S.COL_COPPICE_STD,'Mat','mat',true,'✓'],
 [S.COL_D_CM,'Diam (cm)','diam',30,'>29'],
 [S.COL_H_M,'Alt (m)','alt',15.5,'>15'],
 [S.COL_L10_MM,'L10 (mm)','l10',2,'>1'],
 [S.COL_V_M3,'Vol (m3)','vol',1.25,'>1'],
 [S.COL_MASS_Q,'Massa (q)','massa',10.5,'>10'],
 [S.COL_PRESERVED,'PAI','pai',true,'✓'],
 [S.COL_LAT,'Lat','lat',38.5,'>38'],
 [S.COL_LON,'Lon','lon',16.2,'>16'],
 [S.COL_ACC_M,'Acc (m)','acc',4,'>3'],
];
const values=new Map(labelledFields.map(([key,_label,_prefix,value])=>[key,value]));
const example=treeTable.columns.map(column=>values.get(column.key)??null);
for(const [key,label,prefix,_value,criterion] of labelledFields){
 eq(treeTable.columns.find(column=>column.key===key)?.label,label,`survey display label: ${label}`);
 eq(matchesSearch(example,searchTerms(`${prefix}:${criterion}`),treeTable.columns),true,
    `unquoted ${prefix} selector resolves uniquely to its column`);
}


eq(globalThis.__treeDetailInstances.length, 1,
   'free survey mounts one shared summary component');
const detail = globalThis.__treeDetailInstances[0];
eq(treeTable.rowsPerPage,20,'survey tables request twenty rows per page');
eq(detail.opts.mapContainer===surveyMap,true,'free-survey map is mounted above the table');
eq(detail.visibilityAtCreation,true,'a first-load tree map is constructed in a visible panel');
eq(detail.visibilityOnShow,true,'Leaflet is resized only after its panel is visible');
eq(detail.visibilityOnUpdate,true,'source updates reveal a map panel before updating the component');
const savedFreeRows=freeTreeDigest.rows.map(row=>[...row]);
cache.applyResponseChanges({deletes:savedFreeRows.map(row=>({data_id:'sampled_trees_2',row_id:row[0]}))});
eq(contentEl.querySelector('[data-target="survey-map-section"]').hidden,true,
   'deleting the final free-survey tree hides its map panel');
cache.applyResponseChanges({patches:savedFreeRows.map(row=>({data_id:'sampled_trees_2',row_id:row[0],record:row}))});
eq(detail.visibilityOnUpdate,true,'a first tree after an empty source reveals its panel before map creation');

eq(detail.rows.map(row => row[0]), [201, 202],
   'tree detail starts with the sampled-tree table rows');
eq(detail.opts.digest.diameter_class_mode, 'shifted_up',
   'tree detail receives the server diameter-class mode metadata');
eq(detail.opts.pointColumnNames, { number: S.COL_TREE_NUM },
   'tree detail maps the sampled-tree number column');
eq(detail.opts.speciesNames, ['Abete bianco', 'Faggio'],
   'tree detail receives the global species palette');

const editIcon = contentEl.querySelector(
  '[data-target="trees-table-host"] .action-edit',
);
fetches.length = 0;
await editIcon.click();
await flushAsyncWork();
eq(fetches, ['/api/campionamenti/tree/form/201/'],
   'tree pencil fetches the edit form for its tree-sample row_id');
eq(modalEl.querySelector('form'), null,
   'tree pencil refuses an add-shaped form with an empty row_id');

payloads.set('/api/campionamenti/tree/form/201/', {
  html: '<form data-test-tree-row-id="201"></form>',
});
await editIcon.click();
await flushAsyncWork();
eq(modalEl.querySelector('[name="row_id"]').value, '201',
   'tree pencil accepts a form carrying the requested row_id');
await modalEl.querySelector('[data-action="cancel"]').click();

const search = contentEl.querySelector('.table-search');
search.value = 'part:1';
search.dispatchEvent({ type: 'input' });
await new Promise(resolve => setTimeout(resolve, 550));
await flushAsyncWork();
eq(detail.rows.map(row => row[0]), [201],
   'sample parcel filter excludes 10b from parcel 1 in table/map/summary');
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
eq(detail.destroyed, true, 'switching survey disposes the previous tree detail');
const structuredDetail = globalThis.__treeDetailInstances.at(-1);
eq(structuredDetail.opts.mapEnabled,false,'structured summary does not create a tree map');
eq(structuredDetail.rows.map(row=>row[0]),[301,302,303],
   'structured surveys have a summary of all trees after switching');
const structuredSearch=contentEl.querySelector('.table-search');
eq(structuredSearch.value,'','switching survey clears the previous text filter');
structuredSearch.value='Faggio';
structuredSearch.dispatchEvent({type:'input'});
await new Promise(resolve=>setTimeout(resolve,550));
const areaMap=globalThis.__rilevamentiMapInstances.at(-1);
areaMap.opts.onAreaSelect(50);
eq(structuredSearch.value,'compresa:Serra particella:01b area:7',
   'map selection overwrites the filter with its full area identity');
eq(structuredDetail.rows.map(row=>row[0]),[301],
   'area filter is exact and replaces a previous species restriction');
await structuredDetail.opts.onExport(structuredDetail.rows);
eq(exportedDendrometryRowIds,[301],'structured dendrometry export uses the selected area');
await contentEl.querySelector('[data-target="trees-table-host"] .btn-export').click();
await flushAsyncWork();
eq(exportedTreeRowIds,[301],'tree-data CSV export uses the same selected area');
areaMap.opts.onAreaSelect(51);
eq(structuredDetail.rows.map(row=>row[0]),[302],'selecting area 72 replaces area 7');
structuredSearch.value='';
structuredSearch.dispatchEvent({type:'input'});
await new Promise(resolve=>setTimeout(resolve,550));
eq(structuredDetail.rows.map(row=>row[0]),[301,302,303],
   'clearing the visible filter removes every area restriction');
eq(areaMap.activeAreaId,null,'clearing the filter clears the active map marker');
areaMap.opts.onAreaSelect(52);
eq(structuredSearch.value,'compresa:"Serra San Bruno" particella:9 area:7',
   'map filters quote region names containing spaces');
eq(structuredDetail.rows.map(row=>row[0]),[303],'area filtering separates equally numbered areas in other regions');
areaMap.opts.onAreaSelect(53);
eq(structuredDetail.rows.length,0,'unvisited areas produce an empty table and summary');
eq(areaMap.activeAreaId,53,'unvisited areas remain selected for adding their first tree');
areaMap.opts.onAreaSelect(null);
eq(structuredSearch.value,'','clicking empty map space clears the area filter');

// URL filters apply after asynchronous selection, while dropdown selection resets them.
campionamenti.onQueryChange({s:'2',tf:'Faggio'});
await flushAsyncWork();
eq(contentEl.querySelector('.table-search').value,'Faggio','deep-linked filter survives asynchronous survey activation');
surveySelect.value='1';
surveySelect.dispatchEvent({type:'change'});
await flushAsyncWork();
eq(contentEl.querySelector('.table-search').value,'','dropdown survey change resets the filter');
eq(new URLSearchParams(location.search).has('tf'),false,'dropdown selection clears the filter from the URL');
campionamenti.onQueryChange({s:'1',a:'51'});
await flushAsyncWork();
eq(contentEl.querySelector('.table-search').value,'compresa:Serra particella:01b area:72',
   'legacy area links become explicit visible filters');
campionamenti.onQueryChange({s:'1',tf:'compresa:Serra particella:01b "n. area":7'});
await flushAsyncWork();
eq(globalThis.__treeDetailInstances.at(-1).rows.map(row=>row[0]),[],
   'a digest key cannot filter a renamed column');
eq(globalThis.__rilevamentiMapInstances.at(-1).activeAreaId,null,
   'a digest key does not select a sample area for manual entry');

campionamenti.onQueryChange({s:'1',tf:'COMP:Serra PART:01b AR:7'});
await flushAsyncWork();
eq(globalThis.__treeDetailInstances.at(-1).rows.map(row=>row[0]),[301],
   'visible label prefixes filter the tree rows');
eq(globalThis.__rilevamentiMapInstances.at(-1).activeAreaId,50,
   'the same prefixes select a map area for manual entry');

campionamenti.onQueryChange({s:'1',tf:'comp:Serra part:01b a:7'});
await flushAsyncWork();
eq(globalThis.__treeDetailInstances.at(-1).rows.map(row=>row[0]),[],
   'an ambiguous visible prefix does not select any trees');
eq(globalThis.__rilevamentiMapInstances.at(-1).activeAreaId,null,
   'an ambiguous prefix clears the area context');



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

// Selection while a slow survey digest is loading must survive its arrival.
campionamenti.onQueryChange({s:'4',tf:'Faggio'});
const loadingMap=globalThis.__rilevamentiMapInstances.at(-1);
loadingMap.opts.onAreaSelect(50);
treeLoads.get('/api/campionamenti/trees/4/').resolve(structuredTreeDigest);
await flushAsyncWork();
eq(contentEl.querySelector('.table-search').value,'compresa:Serra particella:01b area:7',
   'area click while loading overrides the pending table filter');
eq(globalThis.__treeDetailInstances.at(-1).rows.map(row=>row[0]),[301],
   'slow-load selection constrains the summary after rows arrive');
campionamenti.onQueryChange({s:'1'});
await flushAsyncWork();

// The grid and tree imports are discoverable in their workspaces and retain retries.
await contentEl.querySelector('[data-action="import-grid-csv"]').click();
const gridFile = modalEl.querySelector('[name="file"]');
gridFile.files = [new Blob(['grid csv'])];
modalEl.querySelector('form').dispatchEvent({type:'submit',preventDefault(){}});
await flushAsyncWork(); await flushAsyncWork();
eq(csvImports.at(-1)[1].sample_grid_id, '10', 'grid import submits the selected grid');
eq(modalEl.querySelector('[data-field="date-row"]').hidden, true, 'grid import has no default date');
await modalEl.querySelector('[data-action="cancel"]').click();
const treeControls = contentEl.querySelector('[data-target="trees-table-host"] .table-toolbar-actions');
eq(treeControls.children.filter(node=>node.tagName==='button').map(node=>node.textContent),
  ['Esporta','Importa CSV','+ Nuovo albero'], 'sample table controls follow the common action order');
await treeControls.children.find(button=>button.textContent===S.IMPORT_CSV).click();
const treeFile = modalEl.querySelector('[name="file"]');
treeFile.files = [new Blob(['tree csv'])];
modalEl.querySelector('[name="default_date"]').value = '2026-11-14';
modalEl.querySelector('form').dispatchEvent({type:'submit',preventDefault(){}});
await flushAsyncWork(); await flushAsyncWork();
eq(csvImports.at(-1)[1].survey_id, '1', 'tree import submits the selected survey');
eq(csvImports.at(-1)[1].default_date, '2026-11-14', 'tree import preserves the default-date affordance');
eq(modalEl.querySelector('[name="file"]') === treeFile, true, 'tree CSV validation retains the file input');
await modalEl.querySelector('[data-action="cancel"]').click();
campionamenti.onQueryChange({s:'1', o:'t'});
eq(contentEl.querySelector('[data-section="r"].collapsible-body').classList.contains('open'), true,
  'older tree-section links open the combined survey workspace');

campionamenti.unmount();

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
