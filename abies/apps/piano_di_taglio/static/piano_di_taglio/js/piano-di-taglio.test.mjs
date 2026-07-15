// Regression tests for Piano di taglio async item-view state.
// Run with: node apps/piano_di_taglio/static/piano_di_taglio/js/piano-di-taglio.test.mjs

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-pdt-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'piano_di_taglio'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'piano_di_taglio', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

let passed = 0;
let failed = 0;
function check(ok, msg) {
  if (ok) passed++;
  else { failed++; console.error(`FAIL ${msg}`); }
}
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
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
    this.hidden = false;
    this.href = '';
    this.rel = '';
    this.type = '';
    this.download = '';
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
  setAttribute(name, value) {
    if (name.startsWith('data-')) this.dataset[name.slice(5)] = String(value);
    else this[name] = String(value);
  }
  getAttribute(name) {
    return name.startsWith('data-') ? this.dataset[name.slice(5)] : this[name];
  }
  removeAttribute(name) {
    if (name.startsWith('data-')) delete this.dataset[name.slice(5)];
    else delete this[name];
  }
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
  removeEventListener(type, fn) {
    this._listeners[type] = (this._listeners[type] || []).filter(f => f !== fn);
  }
  click() {
    const event = { target: this, preventDefault() {} };
    let node = this;
    while (node) {
      for (const fn of node._listeners?.click || []) fn(event);
      node = node.parentNode;
    }
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
  contains(target) {
    if (target === this) return true;
    return this.children.some(child => child.contains?.(target));
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
    clone.hidden = this.hidden;
    clone.href = this.href;
    clone.rel = this.rel;
    clone.type = this.type;
    if (deep) for (const child of this.children) clone.appendChild(child.cloneNode(true));
    return clone;
  }
}

function el(tag, { id = '', className = '', dataset = {}, type = '' } = {}, children = []) {
  const node = new MockElement(tag);
  node.id = id;
  node.className = className;
  node.dataset = { ...dataset };
  node.type = type;
  for (const child of children) node.appendChild(child);
  return node;
}

function section(key) {
  const header = el('div', { className: 'collapsible-header', dataset: { section: key } });
  const body = el('div', { className: 'collapsible-body', dataset: { section: key } });
  return [header, body];
}

function buildPageTemplate() {
  const frag = el('fragment');
  const headerLeft = el('div', { className: 'pdt-header-left' }, [
    el('label', { className: 'pdt-pulldown-label' }),
    el('select', { id: 'pdt-plan-select' }),
  ]);
  frag.appendChild(el('div', { className: 'pdt-header' }, [headerLeft]));
  frag.appendChild(el('div', { dataset: { target: 'description' } }));
  frag.appendChild(el('div', { dataset: { target: 'no-plans' } }));
  for (const key of ['f', 'c']) {
    const [header, body] = section(key);
    body.appendChild(el('div', { dataset: { target: `table-${key}` } }));
    body.appendChild(el('div', { dataset: { target: `empty-${key}` } }));
    frag.append(header, body);
  }
  return frag;
}

function buildItemViewTemplate() {
  return el('fragment', {}, [
    el('div', { className: 'pdt-item-card' }, [
      el('div', { className: 'pdt-item-header' }, [
        el('h2', { className: 'pdt-item-title', dataset: { field: 'title' } }),
        el('button', { dataset: { action: 'export-item' } }),
        el('button', { dataset: { action: 'close-item' } }),
      ]),
      el('dl', { dataset: { target: 'metadata' } }),
      el('div', { dataset: { target: 'transitions' } }),
      el('div', { dataset: { target: 'subsections' } }),
    ]),
  ]);
}

function buildSubsectionTemplate() {
  return el('fragment', {}, [
    el('div', { className: 'collapsible-header' }, [
      el('span', { dataset: { field: 'title' } }),
    ]),
    el('div', { className: 'collapsible-body' }),
  ]);
}

function buildConfirmTemplate() {
  const frag = el('fragment');
  frag.appendChild(el('p', { dataset: { field: 'message' } }));
  const actions = el('div', { className: 'form-actions' });
  actions.appendChild(el('button', { dataset: { action: 'cancel' } }));
  actions.appendChild(el('button', { dataset: { action: 'confirm' } }));
  frag.appendChild(actions);
  return frag;
}

function buildMarkPopoverTemplate() {
  const frag = el('fragment');
  frag.appendChild(el('div', { className: 'pdt-mark-popover-fields', dataset: { target: 'fields' } }));
  const actions = el('div', { className: 'form-actions' });
  actions.appendChild(el('button', { dataset: { action: 'cancel' } }));
  actions.appendChild(el('button', { className: 'btn btn-save', dataset: { action: 'edit-mark' } }));
  actions.appendChild(el('button', { className: 'btn btn-delete', dataset: { action: 'delete-mark' } }));
  frag.appendChild(actions);
  return frag;
}

const contentEl = el('main');
const modalEl = el('div', { id: 'modal-container' });
const links = [];
const templates = {
  'tmpl-pdt-page': { content: buildPageTemplate() },
  'tmpl-pdt-item-view': { content: buildItemViewTemplate() },
  'tmpl-pdt-item-subsection': { content: buildSubsectionTemplate() },
  'tmpl-pdt-mark-popover': { content: buildMarkPopoverTemplate() },
  'tmpl-confirm-modal': { content: buildConfirmTemplate() },
};

globalThis.document = {
  documentElement: { lang: 'it' },
  body: el('body', { dataset: { csrf: 'csrf-token', role: 'reader' } }),
  head: { appendChild: link => links.push(link) },
  createElement: tag => el(tag),
  createTextNode: text => { const node = el('#text'); node.textContent = text; return node; },
  createDocumentFragment: () => el('fragment'),
  addEventListener() {},
  removeEventListener() {},
  getElementById(id) {
    if (id === 'content') return contentEl;
    if (id === 'modal-container') return modalEl;
    return templates[id] || null;
  },
  querySelector(sel) {
    const href = sel.match(/^link\[href="([^"]+)"\]$/)?.[1];
    return href ? links.find(link => link.href === href && !link.removed) || null : null;
  },
  querySelectorAll() { return []; },
};

globalThis.location = { pathname: '/piano-di-taglio', search: '' };
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
function appendMockControls(opts) {
  if (!opts.controlsStart && !opts.controlsEnd) return;
  const controls = el('div', { className: 'sortable-table-controls sortable-table-controls-combined' });
  if (opts.controlsStart) {
    opts.controlsStart.classList.add('sortable-table-controls-start');
    controls.appendChild(opts.controlsStart);
  }
  controls.appendChild(el('div', { className: 'sortable-table-pagination' }));
  if (opts.controlsEnd) {
    opts.controlsEnd.classList.add('sortable-table-controls-end');
    controls.appendChild(opts.controlsEnd);
  }
  opts.container.appendChild(controls);
}
class MockSortableTable {
  constructor(opts) {
    this.container = opts.container;
    this.data = opts.data;
    this.columns = opts.columns;
    this.currentSort = opts.sort || { column: opts.columns.find(c => !c.hidden)?.key, ascending: true };
    this.onSort = opts.onSort;
    this.destroyed = false;
    appendMockControls(opts);
    tableInstances.push(this);
  }
  setData(rows) { this.data = rows; }
  filter(fn) { this.data = this.data.filter(fn); }
  clearFilter() {}
  sort(column, _type, ascending) {
    this.currentSort = { column, ascending };
    this.onSort?.(column, ascending);
  }
  destroy() { this.destroyed = true; }
}

let browserConfirmCalls = 0;
globalThis.confirm = () => { browserConfirmCalls += 1; return false; };
globalThis.window = { SortableTable: MockSortableTable, addEventListener() {} };

function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}
const flushAsyncWork = () => new Promise(resolve => setTimeout(resolve, 0));
async function flushSeveral(times = 4) {
  for (let i = 0; i < times; i++) await flushAsyncWork();
}

const S = await import(staticModule('base/js/strings.js'));
const { COL_COPPICE, ROW_ID, VERSION } = await import(staticModule('base/js/constants.js'));

const planColumns = [ROW_ID, S.COL_NAME, S.COL_DESCRIPTION, S.COL_YEAR_START, S.COL_YEAR_END];
const planDigest = { columns: planColumns, rows: [[10, 'Piano', '', 2026, 2029]] };
const itemColumns = [
  ROW_ID, VERSION, S.COL_HARVEST_PLAN,
  S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL, S.COL_REGION, S.COL_PARCEL,
  S.COL_PARCEL_AREA_HA, S.COL_TYPE, COL_COPPICE, S.COL_STATE, S.COL_NOTE,
  S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED, S.COL_VOLUME_ACTUAL,
  S.COL_INTERVENTION_AREA_HA, S.COL_PERIOD_Y,
];

function itemRow(id, { region, parcel, year, state = S.STATE_MARKED, coppice = false }) {
  return [
    id, 1, 10,
    year, null, region, parcel, coppice ? 10 : 12.5,
    coppice ? S.TYPE_COPPICE : '', coppice, state, '',
    coppice ? null : 100, coppice ? null : 20, 0,
    coppice ? 3 : null, coppice ? 12 : null,
  ];
}

const itemRows = [
  itemRow(1, { region: 'A', parcel: '1', year: 2026 }),
  itemRow(2, { region: 'B', parcel: '2', year: 2027 }),
  itemRow(11, { region: 'C', parcel: '11', year: 2026 }),
  itemRow(12, { region: 'D', parcel: '12', year: 2027 }),
  itemRow(21, { region: 'E', parcel: '21', year: 2026, state: S.STATE_OPEN, coppice: true }),
  itemRow(22, { region: 'F', parcel: '22', year: 2027, state: S.STATE_OPEN, coppice: true }),
];
const itemsDigest = { columns: itemColumns, rows: itemRows };

const itemLoads = new Map();
const markLoads = new Map();
const markDataOverrides = new Map();
const prelieviLoads = [];
const fetchUrls = [];
function deferItem(id) { const d = deferred(); itemLoads.set(`/api/piano-di-taglio/item/data/${id}/`, d); return d; }
function deferMarks(id) { const d = deferred(); markLoads.set(`/api/piano-di-taglio/mark-trees/${id}/`, d); return d; }
function deferPrelievi() { const d = deferred(); prelieviLoads.push(d); return d; }
function itemPayload(id) {
  return { record: itemRows.find(row => row[0] === id), transition_records: [] };
}
function markDigest(id, overrides = {}) {
  const row = [
    id * 100 + 1, 1, '2026-01-01', 1, 'Abete bianco', 30, 20,
    true, 1.2, 8.5, 38.1, 16.2, 'Operatore',
  ];
  const columns = [
    ROW_ID, VERSION, S.COL_DATE, S.COL_NUMBER, S.COL_SPECIES, S.COL_D_CM,
    S.COL_H_M, S.COL_H_MEASURED, S.COL_V_M3, S.COL_MASS_Q, S.COL_LAT,
    S.COL_LON, S.COL_OPERATOR,
  ];
  for (const [name, value] of Object.entries(overrides)) {
    row[columns.indexOf(name)] = value;
  }
  return { columns, rows: [row] };
}
function prelieviDigest() {
  return {
    columns: [ROW_ID, VERSION, S.COL_WORKSITE, S.COL_DATE, S.COL_VOLUME_M3],
    rows: [[2101, 1, 21, '2026-01-01', 1.1], [2201, 1, 22, '2026-01-02', 2.2]],
  };
}
function response(data, lastModified = 'v1') {
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? lastModified : null },
    json: async () => data,
  };
}

function terreniGeojson() {
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: { layer: 'A', name: 'A-1', coppice: false },
      geometry: {
        type: 'Polygon',
        coordinates: [[[16, 38], [17, 38], [17, 39], [16, 39], [16, 38]]],
      },
    }],
  };
}

let leafletMaps = [];
let leafletMarkerGroups = [];
function installLeafletMock() {
  leafletMaps = [];
  leafletMarkerGroups = [];

  function fakeClassList(node) {
    return {
      add: (...names) => node._setClasses(new Set([...node._classes(), ...names])),
      remove: (...names) => {
        const next = node._classes();
        for (const name of names) next.delete(name);
        node._setClasses(next);
      },
      toggle: (name, force) => {
        const next = node._classes();
        const shouldAdd = force === undefined ? !next.has(name) : Boolean(force);
        if (shouldAdd) next.add(name);
        else next.delete(name);
        node._setClasses(next);
        return shouldAdd;
      },
      contains: name => node._classes().has(name),
    };
  }

  function fakeMap() {
    const handlers = {};
    return {
      handlers,
      controls: [],
      on(ev, cb) { (handlers[ev] ||= []).push(cb); return this; },
      off(ev, cb) { handlers[ev] = (handlers[ev] || []).filter(h => h !== cb); return this; },
      fire(ev, payload) { (handlers[ev] || []).forEach(cb => cb(payload)); },
      addControl(control) { this.controls.push(control); control.onAdd?.(this); return this; },
      removeControl(control) { this.controls = this.controls.filter(c => c !== control); return this; },
      removeLayer(layer) { layer.removed = true; return this; },
      fitBounds(bounds, opts) { this.fitted = { bounds, opts }; return this; },
      setView(center, zoom) { this.view = { center, zoom }; return this; },
      getCenter() { return { lat: 38.5, lng: 16.5 }; },
      getZoom() { return 12; },
      getContainer() { return { style: {} }; },
      invalidateSize() {},
      locate() { return this; },
      stopLocate() { return this; },
      distance() { return 1; },
      remove() { this.removed = true; },
    };
  }

  function layerGroup() {
    const group = {
      layers: [],
      addTo() { return this; },
      addLayer(layer) { this.layers.push(layer); layer.group = this; },
      clearLayers() { this.layers = []; },
    };
    leafletMarkerGroups.push(group);
    return group;
  }

  function marker(latlng, style) {
    return {
      latlng,
      style,
      handlers: {},
      bindTooltip(content, options) { this.tooltip = { content, options }; return this; },
      on(ev, cb) { this.handlers[ev] = cb; return this; },
      addTo(layer) { layer.addLayer(this); return this; },
      setStyle(next) { this.style = { ...this.style, ...next }; },
      bringToFront() { this.front = true; },
      click() { this.handlers.click?.({}); },
    };
  }

  globalThis.L = {
    map() { const map = fakeMap(); leafletMaps.push(map); return map; },
    tileLayer() { return { addTo(map) { this.map = map; return this; } }; },
    control: { zoom: () => ({ addTo(map) { map.addControl(this); return this; } }) },
    Control: {
      extend(proto) {
        return function Control(...args) {
          Object.assign(this, proto);
          this.options = { ...(proto.options || {}) };
          this.initialize?.(...args);
        };
      },
    },
    DomUtil: {
      create(tag, cls, parent) {
        const node = el(tag, { className: cls || '' });
        node.style = {};
        node.classList = fakeClassList(node);
        parent?.appendChild?.(node);
        return node;
      },
    },
    DomEvent: {
      on(node, ev, cb) { node.addEventListener?.(ev, cb); },
      off() {},
      stop() {},
      stopPropagation() {},
      preventDefault() {},
      disableClickPropagation() {},
    },
    geoJSON(data, opts) {
      const layer = {
        addTo() { return this; },
        getBounds() { return { isValid: () => true }; },
        tooltips: [],
      };
      for (const feature of data.features || []) {
        const lyr = {
          bindTooltip(content, options) { layer.tooltips.push({ content, options }); return this; },
          on() {},
        };
        opts.onEachFeature?.(feature, lyr);
      }
      return layer;
    },
    layerGroup,
    circleMarker: marker,
    polyline: () => ({ addTo() { return this; } }),
    marker: () => ({ addTo() { return this; } }),
    circle: () => ({ addTo() { return this; } }),
    divIcon: () => ({}),
  };
}

function latestTreeMarkers() {
  return leafletMarkerGroups.at(-1)?.layers || [];
}

globalThis.fetch = async (url, options = {}) => {
  fetchUrls.push(String(url));
  if (url === '/api/piano-di-taglio/plans/data/') return response(planDigest);
  if (url === '/api/piano-di-taglio/items/data/') return response(itemsDigest);
  if (url === '/api/impostazioni/hypso-params/data/') return response({ columns: [], rows: [] });
  if (url === '/api/geo/terreni.geojson') return response(terreniGeojson());
  if (url === '/api/piano-di-taglio/mark/delete/' && options.method === 'POST') {
    const body = JSON.parse(options.body || '{}');
    const rowId = Number(body[ROW_ID]);
    const itemId = Math.floor((rowId - 1) / 100);
    markDataOverrides.set(
      `/api/piano-di-taglio/mark-trees/${itemId}/`,
      { ...markDigest(itemId), rows: [] },
    );
    return response({
      deletes: [{ data_id: `mark_trees_${itemId}`, row_id: rowId }],
    });
  }
  if (itemLoads.has(url)) return response(await itemLoads.get(url).promise);
  if (markDataOverrides.has(url)) return response(markDataOverrides.get(url), 'v2');
  if (markLoads.has(url)) return response(await markLoads.get(url).promise);
  if (url === '/api/prelievi/data/') return response(await prelieviLoads.shift().promise);
  if (String(url).startsWith('/api/piano-di-taglio/mark/form/')) {
    return { status: 404, ok: false, headers: { get: () => null }, json: async () => ({}) };
  }
  throw new Error(`unexpected fetch ${url}`);
};

const pdt = await import(staticModule('piano_di_taglio/js/piano-di-taglio.js'));
const cache = await import(staticModule('base/js/cache.js'));

function titleText() {
  return contentEl.querySelector('[data-field="title"]')?.textContent || '';
}
function expectedTitle(id) {
  const row = itemRows.find(r => r[0] === id);
  const c = itemColumns;
  return `${S.VIEW_ITEM_TITLE} del Piano, anno ${row[c.indexOf(S.COL_YEAR_PLANNED)]}, ${row[c.indexOf(S.COL_REGION)]}/${row[c.indexOf(S.COL_PARCEL)]}`;
}

async function mountItem(id) {
  await pdt.mount({ i: String(id) });
}
async function switchItem(id) {
  pdt.onQueryChange({ i: String(id) });
}
async function finish() {
  pdt.unmount();
  contentEl.replaceChildren();
  modalEl.replaceChildren();
  tableInstances.length = 0;
  markDataOverrides.clear();
  delete globalThis.L;
  leafletMaps = [];
  leafletMarkerGroups = [];
  await flushAsyncWork();
}

// Calendar tables show cadastral parcel/region area as the fifth visible column.
{
  const previousRole = document.body.dataset.role;
  document.body.dataset.role = 'writer';
  await pdt.mount({});
  const expectedLeading = [
    S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL, S.COL_REGION,
    S.COL_PARCEL, S.COL_PARCEL_AREA_HA, S.COL_STATE,
  ];
  const fustaiaVisible = tableInstances[0].columns
    .filter(col => !col.hidden)
    .map(col => col.key);
  const ceduoVisible = tableInstances[1].columns
    .filter(col => !col.hidden)
    .map(col => col.key);
  eq(
    fustaiaVisible.slice(0, expectedLeading.length),
    expectedLeading,
    'fustaia table shows area as fifth visible column',
  );
  eq(
    ceduoVisible.slice(0, expectedLeading.length),
    expectedLeading,
    'ceduo table shows area as fifth visible column',
  );
  const controls = contentEl.querySelector('[data-target="table-f"] .sortable-table-controls');
  const actionGroup = controls.querySelector('.table-toolbar-actions');
  check(Boolean(controls.querySelector('.table-search')), 'calendar search is rendered in TableWrapper controls');
  eq(
    actionGroup.children.map(child => child.textContent),
    ['Esporta', '+ Aggiungi'],
    'calendar add button is in TableWrapper toolbar immediately after export',
  );
  await finish();
  document.body.dataset.role = previousRole;
}

// A stale item/data response must not replace the newer item view.
{
  const item1 = deferItem(1);
  const item2 = deferItem(2);
  await mountItem(1);
  await switchItem(2);
  item2.resolve(itemPayload(2));
  await flushAsyncWork();
  eq(titleText(), expectedTitle(2), 'newer item renders first');
  item1.resolve(itemPayload(1));
  await flushAsyncWork();
  eq(titleText(), expectedTitle(2), 'stale item response does not replace the active view');
  await finish();
}

// A plans refresh while an item is open must not rebuild back to the calendar.
{
  const item = deferItem(1);
  await mountItem(1);
  item.resolve(itemPayload(1));
  await flushAsyncWork();
  const title = titleText();

  cache.applyResponseChanges({
    patches: [{
      data_id: 'harvest_plans',
      row_id: 10,
      record: [10, 'Piano rinominato', 'Aggiornato', 2026, 2029],
    }],
  });
  await flushAsyncWork();

  eq(titleText(), title, 'plans refresh preserves the active item view');
  check(Boolean(contentEl.querySelector('.pdt-item-card')),
        'plans refresh leaves the item card mounted');
  await finish();
}

// A stale Martellate subsection load must not destroy the active item's table.
{
  const itemA = deferItem(11);
  const itemB = deferItem(12);
  const marksA = deferMarks(11);
  const marksB = deferMarks(12);
  await mountItem(11);
  itemA.resolve(itemPayload(11));
  await flushAsyncWork();
  await switchItem(12);
  itemB.resolve(itemPayload(12));
  await flushAsyncWork();
  marksB.resolve(markDigest(12));
  await flushAsyncWork();
  const activeTable = tableInstances.at(-1);
  eq(activeTable.data.map(row => row[0]), [1201], 'active mark table belongs to item B');
  marksA.resolve(markDigest(11));
  await flushAsyncWork();
  check(!activeTable.destroyed, 'stale mark load does not destroy active mark table');
  eq(tableInstances.at(-1).data.map(row => row[0]), [1201], 'stale mark load does not replace active mark table');
  await finish();
}

// Mark edit in the item detail view must request the form for the digest ROW_ID.
{
  const previousRole = document.body.dataset.role;
  document.body.dataset.role = 'writer';
  const item = deferItem(11);
  const marks = deferMarks(11);
  fetchUrls.length = 0;
  await mountItem(11);
  item.resolve(itemPayload(11));
  await flushAsyncWork();
  marks.resolve(markDigest(11));
  await flushAsyncWork();

  const tableEl = contentEl.querySelector('.table-scroll');
  const row = el('tr', { className: 'sortable-table-row' });
  row.dataset.index = '0';
  const edit = el('span', { className: 'action-icon action-edit' });
  row.appendChild(edit);
  tableEl.appendChild(row);
  edit.click();
  await flushAsyncWork();

  const markFormUrls = fetchUrls
    .filter(u => u.startsWith('/api/piano-di-taglio/mark/form/'));
  eq(markFormUrls.at(-1), '/api/piano-di-taglio/mark/form/1101/',
     'mark edit opens the form for the digest row_id');
  await finish();
  document.body.dataset.role = previousRole;
}

// Mark deletion in the item detail view must use the shared modal, not window.confirm().
{
  const previousRole = document.body.dataset.role;
  document.body.dataset.role = 'writer';
  const item = deferItem(11);
  const marks = deferMarks(11);
  browserConfirmCalls = 0;
  await mountItem(11);
  item.resolve(itemPayload(11));
  await flushAsyncWork();
  marks.resolve(markDigest(11));
  await flushAsyncWork();

  const tableEl = contentEl.querySelector('.table-scroll');
  const row = el('tr', { className: 'sortable-table-row' });
  row.dataset.index = '0';
  const del = el('span', { className: 'action-icon action-delete' });
  row.appendChild(del);
  tableEl.appendChild(row);
  del.click();

  eq(browserConfirmCalls, 0, 'mark delete does not call the browser confirm');
  check(Boolean(modalEl.querySelector('[data-action="confirm"]')),
        'mark delete opens the shared confirm modal');
  await finish();
  document.body.dataset.role = previousRole;
}

// Mark edit from the map popover must request the same digest ROW_ID as the table action.
{
  const previousRole = document.body.dataset.role;
  document.body.dataset.role = 'writer';
  installLeafletMock();
  const item = deferItem(11);
  const marks = deferMarks(11);
  fetchUrls.length = 0;
  await mountItem(11);
  item.resolve(itemPayload(11));
  await flushAsyncWork();
  marks.resolve(markDigest(11));
  await flushSeveral();

  eq(latestTreeMarkers().length, 1, 'mark map renders a geolocated digest row');
  latestTreeMarkers()[0].click();
  modalEl.querySelector('[data-action="edit-mark"]').click();
  await flushAsyncWork();

  const markFormUrls = fetchUrls
    .filter(u => u.startsWith('/api/piano-di-taglio/mark/form/'));
  eq(markFormUrls.at(-1), '/api/piano-di-taglio/mark/form/1101/',
     'mark map edit opens the form for the digest row_id');
  await finish();
  document.body.dataset.role = previousRole;
}

// A mark edit/save response must refresh both the detail table and map marker.
{
  installLeafletMock();
  const item = deferItem(11);
  const marks = deferMarks(11);
  await mountItem(11);
  item.resolve(itemPayload(11));
  await flushAsyncWork();
  marks.resolve(markDigest(11));
  await flushSeveral();

  const updated = markDigest(11, {
    [S.COL_D_CM]: 45,
    [S.COL_H_M]: 22.5,
    [S.COL_LAT]: 38.25,
    [S.COL_LON]: 16.35,
  });
  markDataOverrides.set('/api/piano-di-taglio/mark-trees/11/', updated);
  cache.applyResponseChanges({
    patches: [{ data_id: 'mark_trees_11', row_id: 1101, record: updated.rows[0] }],
  });
  pdt.onQueryChange({});
  await flushSeveral();
  pdt.onQueryChange({ i: '11' });
  await flushSeveral();

  const latestTable = tableInstances.at(-1);
  const dCol = updated.columns.indexOf(S.COL_D_CM);
  eq(latestTable.data.map(row => row[dCol]), [45],
     'mark table reflects the edited digest row after re-render');
  eq(latestTreeMarkers().map(marker => marker.latlng), [[38.25, 16.35]],
     'mark map reflects the edited coordinates after re-render');
  await finish();
}

// Deleting a mark from the table must remove the corresponding map.
{
  const previousRole = document.body.dataset.role;
  document.body.dataset.role = 'writer';
  installLeafletMock();
  const item = deferItem(11);
  const marks = deferMarks(11);
  await mountItem(11);
  item.resolve(itemPayload(11));
  await flushAsyncWork();
  marks.resolve(markDigest(11));
  await flushSeveral();

  const tableEl = contentEl.querySelector('.table-scroll');
  const row = el('tr', { className: 'sortable-table-row' });
  row.dataset.index = '0';
  const del = el('span', { className: 'action-icon action-delete' });
  row.appendChild(del);
  tableEl.appendChild(row);
  del.click();
  modalEl.querySelector('[data-action="confirm"]').click();
  await flushSeveral(6);

  eq(tableInstances.at(-1).data, [], 'table delete refreshes the mark table to empty');
  check(!contentEl.querySelector('.pdt-mark-map-host'),
        'table delete removes the mark map when no geolocated rows remain');
  await finish();
  document.body.dataset.role = previousRole;
}

// Deleting a mark from the map popover must remove it from the table.
{
  const previousRole = document.body.dataset.role;
  document.body.dataset.role = 'writer';
  installLeafletMock();
  const item = deferItem(11);
  const marks = deferMarks(11);
  await mountItem(11);
  item.resolve(itemPayload(11));
  await flushAsyncWork();
  marks.resolve(markDigest(11));
  await flushSeveral();

  latestTreeMarkers()[0].click();
  modalEl.querySelector('[data-action="delete-mark"]').click();
  modalEl.querySelector('[data-action="confirm"]').click();
  await flushSeveral(6);

  eq(tableInstances.at(-1).data, [], 'map delete refreshes the mark table to empty');
  check(!contentEl.querySelector('.pdt-mark-map-host'),
        'map delete removes the mark map when no geolocated rows remain');
  await finish();
  document.body.dataset.role = previousRole;
}

// A stale Prelievi subsection load must not destroy the active item's table.
{
  const itemA = deferItem(21);
  const itemB = deferItem(22);
  const prelieviA = deferPrelievi();
  const prelieviB = deferPrelievi();
  await mountItem(21);
  itemA.resolve(itemPayload(21));
  await flushAsyncWork();
  await switchItem(22);
  itemB.resolve(itemPayload(22));
  await flushAsyncWork();
  prelieviB.resolve(prelieviDigest());
  await flushAsyncWork();
  const activeTable = tableInstances.at(-1);
  eq(activeTable.data.map(row => row[0]), [2201], 'active prelievi table belongs to item B');
  prelieviA.resolve(prelieviDigest());
  await flushAsyncWork();
  check(!activeTable.destroyed, 'stale prelievi load does not destroy active prelievi table');
  eq(tableInstances.at(-1).data.map(row => row[0]), [2201], 'stale prelievi load does not replace active prelievi table');
  await finish();
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
