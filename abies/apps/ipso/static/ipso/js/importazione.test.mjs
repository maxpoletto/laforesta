// Regression tests for Ipso import preview maps.
// Run with: node apps/ipso/static/ipso/js/importazione.test.mjs

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-ipso-import-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'ipso'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'ipso', 'js'), { recursive: true });
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
    this.checked = false;
    this.disabled = false;
    this.placeholder = '';
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
  get childElementCount() { return this.children.filter(c => c instanceof MockElement).length; }
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
  after(...nodes) {
    if (!this.parentNode) return;
    const siblings = this.parentNode.children;
    const idx = siblings.indexOf(this);
    let offset = 1;
    for (const node of nodes) {
      node.parentNode = this.parentNode;
      siblings.splice(idx + offset, 0, node);
      offset++;
    }
  }
  replaceChildren(...children) {
    for (const child of this.children) child.parentNode = null;
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
    const event = { target: this, preventDefault() {}, stopPropagation() {} };
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
      for (const node of this._findAll(head)) {
        const found = node._find(tail.join(' '));
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
      for (const node of this._findAll(head)) out.push(...node._findAll(tail.join(' ')));
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
    clone.checked = this.checked;
    clone.disabled = this.disabled;
    clone.placeholder = this.placeholder;
    if (deep) for (const child of this.children) clone.appendChild(child.cloneNode(true));
    return clone;
  }
}

function el(tag, { id = '', className = '', dataset = {}, type = '', hidden = false } = {}, children = []) {
  const node = new MockElement(tag);
  node.id = id;
  node.className = className;
  node.dataset = { ...dataset };
  node.type = type;
  node.hidden = hidden;
  for (const child of children) node.appendChild(child);
  return node;
}

function buildInboxTemplate() {
  const frag = el('fragment');
  frag.appendChild(el('div', { className: 'ipso-inbox-page' }, [
    el('div', { dataset: { role: 'summary' } }),
    el('div', { dataset: { role: 'table' } }),
    el('div', { dataset: { role: 'detail' }, hidden: true }),
  ]));
  return frag;
}

const contentEl = el('main');
const modalEl = el('div', { id: 'modal-container' });
const links = [];
const templates = {
  'tmpl-ipso-inbox-page': { content: buildInboxTemplate() },
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
  querySelectorAll(sel) {
    if (sel === '[data-ipso-pending-dot]') return [];
    return contentEl.querySelectorAll(sel);
  },
};

globalThis.location = { pathname: '/importazione', search: '' };
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
globalThis.alert = () => {};

const tableInstances = [];
function appendMockControls(opts) {
  if (!opts.controlsStart && !opts.controlsEnd) return;
  const controls = el('div', { className: 'sortable-table-controls sortable-table-controls-combined' });
  if (opts.controlsStart) controls.appendChild(opts.controlsStart);
  controls.appendChild(el('div', { className: 'sortable-table-pagination' }));
  if (opts.controlsEnd) controls.appendChild(opts.controlsEnd);
  opts.container.appendChild(controls);
}
class MockSortableTable {
  constructor(opts) {
    this.container = opts.container;
    this.data = opts.data;
    this.columns = opts.columns;
    this.currentSort = opts.sort || { column: opts.columns.find(c => !c.hidden)?.key, ascending: true };
    this.onSort = opts.onSort;
    this.currentPage = 1;
    this.totalPages = 1;
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
globalThis.window = { SortableTable: MockSortableTable, addEventListener() {} };

let leafletMaps = [];
let leafletMarkerGroups = [];
function installLeafletMock() {
  leafletMaps = [];
  leafletMarkerGroups = [];

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
function textOf(node) {
  if (!node) return '';
  return `${node.textContent || ''}${(node.children || []).map(textOf).join('')}`;
}

const flushAsyncWork = () => new Promise(resolve => setTimeout(resolve, 0));
async function flushSeveral(times = 4) {
  for (let i = 0; i < times; i++) await flushAsyncWork();
}

const S = await import(staticModule('base/js/strings.js'));
const {
  FIELD_ACC_M, FIELD_DATE, FIELD_D_CM, FIELD_ERROR_SUMMARY, FIELD_H_M,
  FIELD_ID, FIELD_LAT, FIELD_LON, FIELD_MODE, FIELD_MODE_LABEL,
  FIELD_NUMBER, FIELD_OPERATOR, FIELD_PARCEL, FIELD_RECEIVED_AT,
  FIELD_RECORD_DATE, FIELD_REFERENCE_VERSION, FIELD_REFERENCE_VERSION_LABEL,
  FIELD_SAMPLE_AREA_ID, FIELD_SEQ, FIELD_SESSION_ID, FIELD_SPECIES,
  FIELD_STATE, FIELD_STATE_LABEL, FIELD_TARGET_LABEL, FIELD_WORK_PACKAGE_LABEL,
  IPSO_MODE_MARTELLATE, IPSO_UPLOAD_STATE_RECEIVED, PENDING_COUNT,
  RECORD_COUNT, RECORDS, ROW_ID, TARGETS, UPLOAD,
} = await import(staticModule('base/js/constants.js'));

function response(data, lastModified = 'v1') {
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? lastModified : null },
    json: async () => data,
  };
}

function inboxDigest() {
  return {
    columns: [
      ROW_ID, '_ipso_state', S.IPSO_COL_RECEIVED, S.COL_DATE, S.IPSO_COL_MODE,
      S.IPSO_COL_OPERATOR, S.IPSO_COL_RECORDS, S.IPSO_COL_STATE,
      S.IPSO_COL_WORK_PACKAGE, S.IPSO_COL_TARGET, S.IPSO_COL_ERROR,
    ],
    rows: [
      [501, IPSO_UPLOAD_STATE_RECEIVED, '2026-07-01 10:00', '2026-07-01',
       S.IPSO_MODE_MARTELLATE_LABEL, 'Operatore A', 3, 'Ricevuto', 'WP 1', 'Cantiere 1', ''],
      [502, IPSO_UPLOAD_STATE_RECEIVED, '2026-07-02 10:00', '2026-07-02',
       S.IPSO_MODE_MARTELLATE_LABEL, 'Operatore B', 1, 'Ricevuto', 'WP 2', 'Cantiere 2', ''],
    ],
    [PENDING_COUNT]: 2,
  };
}

function record(seq, lat, lon, species = 'Abete bianco') {
  return {
    [FIELD_SEQ]: seq,
    [FIELD_DATE]: '2026-07-01',
    [FIELD_PARCEL]: 'Serra-1',
    [FIELD_SAMPLE_AREA_ID]: '',
    [FIELD_SPECIES]: species,
    [FIELD_NUMBER]: seq,
    [FIELD_D_CM]: 30 + seq,
    [FIELD_H_M]: 20 + seq / 10,
    [FIELD_LAT]: lat,
    [FIELD_LON]: lon,
    [FIELD_ACC_M]: 4,
  };
}

function uploadDetail(id, records) {
  return {
    [UPLOAD]: {
      [FIELD_ID]: id,
      [FIELD_SESSION_ID]: `session-${id}`,
      [FIELD_STATE]: IPSO_UPLOAD_STATE_RECEIVED,
      [FIELD_STATE_LABEL]: 'Ricevuto',
      [FIELD_MODE]: IPSO_MODE_MARTELLATE,
      [FIELD_MODE_LABEL]: S.IPSO_MODE_MARTELLATE_LABEL,
      [FIELD_OPERATOR]: 'Operatore',
      [FIELD_RECEIVED_AT]: '2026-07-01 10:00',
      [FIELD_RECORD_DATE]: '2026-07-01',
      [RECORD_COUNT]: records.length,
      [FIELD_REFERENCE_VERSION]: 'v1',
      [FIELD_REFERENCE_VERSION_LABEL]: 'v1',
      [FIELD_WORK_PACKAGE_LABEL]: 'WP',
      [FIELD_TARGET_LABEL]: 'Cantiere',
      [FIELD_ERROR_SUMMARY]: '',
    },
    [RECORD_COUNT]: records.length,
    [RECORDS]: records,
    [TARGETS]: [],
  };
}

function terreniGeojson() {
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: { layer: 'Serra', name: 'Serra-1' },
      geometry: {
        type: 'Polygon',
        coordinates: [[[16, 38], [17, 38], [17, 39], [16, 39], [16, 38]]],
      },
    }],
  };
}

const details = new Map([
  ['/api/ipso/uploads/501/', uploadDetail(501, [
    record(1, 38.1, 16.1),
    record(2, null, null, 'Faggio'),
    record(3, 38.3, 16.3, 'Pino'),
  ])],
  ['/api/ipso/uploads/502/', uploadDetail(502, [record(4, 38.4, 16.4, 'Leccio')])],
]);

globalThis.fetch = async (url) => {
  if (url === '/api/ipso/inbox/') return response(inboxDigest());
  if (url === '/api/geo/terreni.geojson') return response(terreniGeojson());
  if (details.has(url)) return response(details.get(url));
  throw new Error(`unexpected fetch ${url}`);
};

const importazione = await import(staticModule('ipso/js/importazione.js'));

function clickInboxOpen(index) {
  const tableEl = contentEl.querySelector('.table-scroll');
  const row = el('tr', { className: 'sortable-table-row' });
  row.dataset.index = String(index);
  const open = el('span', { className: 'action-icon action-edit' });
  row.appendChild(open);
  tableEl.appendChild(row);
  open.click();
}

installLeafletMock();
await importazione.mount({});
clickInboxOpen(0);
await flushSeveral();

const detailEl = contentEl.querySelector('[data-role="detail"]');
eq(tableInstances.at(-1).data.length, 3, 'opening an upload renders the preview table rows');
check(Boolean(detailEl.querySelector('.ipso-record-map-host')), 'opening an upload renders a preview map host');
eq(latestTreeMarkers().map(marker => marker.latlng), [[38.1, 16.1], [38.3, 16.3]],
   'preview map renders only records with finite coordinates');
check(textOf(latestTreeMarkers()[0].tooltip.content).includes('Abete bianco'),
      'preview marker tooltip includes tree species');
const firstMap = leafletMaps.at(-1);

clickInboxOpen(1);
await flushSeveral();

eq(tableInstances.at(-1).data.length, 1, 'opening a second upload replaces the preview table rows');
check(firstMap.removed, 'opening a second upload destroys the previous preview map');
eq(detailEl.querySelectorAll('.ipso-record-map-host').length, 1,
   'opening a second upload leaves exactly one preview map host');
eq(latestTreeMarkers().map(marker => marker.latlng), [[38.4, 16.4]],
   'opening a second upload replaces the preview map markers');

importazione.unmount();

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
