// Regression tests for Prelievi URL-state restoration.
// Run with: node apps/prelievi/static/prelievi/js/prelievi.test.mjs

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-prelievi-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'prelievi'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'prelievi', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;
const constants = await import(staticModule('base/js/constants.js'));
const { COL_PARCEL_ID, COL_REGION_ID, ROW_ID, VERSION } = constants;

let passed = 0;
let failed = 0;
function check(ok, msg) {
  if (ok) passed++;
  else { failed++; console.error(`FAIL ${msg}`); }
}
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
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
    this.min = '';
    this.max = '';
    this.type = '';
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
  setAttribute(name, value) {
    if (name.startsWith('data-')) this.dataset[name.slice(5)] = String(value);
    else this[name] = String(value);
  }
  getAttribute(name) {
    return name.startsWith('data-') ? this.dataset[name.slice(5)] : this[name];
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
  remove() { this.removed = true; }
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
  closest(sel) {
    let node = this;
    while (node) {
      if (node.matches(sel)) return node;
      node = node.parentNode;
    }
    return null;
  }
  matches(sel) {
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    if (sel.startsWith('.')) {
      return sel.slice(1).split('.').every(cls => this.classList.contains(cls));
    }
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
    clone.min = this.min;
    clone.max = this.max;
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

function buildConfirmTemplate() {
  const frag = el('fragment');
  frag.appendChild(el('p', { dataset: { field: 'message' } }));
  const actions = el('div', { className: 'form-actions' });
  actions.appendChild(el('button', { dataset: { action: 'cancel' } }));
  actions.appendChild(el('button', { dataset: { action: 'confirm' } }));
  frag.appendChild(actions);
  return frag;
}

function buildPrelieviTemplate() {
  const frag = el('fragment');
  const minInput = el('input', { dataset: { role: 'slider-min' }, type: 'range' });
  const maxInput = el('input', { dataset: { role: 'slider-max' }, type: 'range' });
  const search = el('input', { id: 'prelievi-search', className: 'table-search', type: 'text' });
  frag.appendChild(el('div', { className: 'prelievi-filter-bar' }, [
    el('span', { className: 'prelievi-slider-label' }),
    minInput,
    maxInput,
    search,
    el('button', { dataset: { action: 'reset-filters' } }),
    el('button', { dataset: { action: 'export-csv' } }),
    el('button', { dataset: { action: 'add' } }),
  ]));

  const [ha, ba] = section('a');
  ba.appendChild(el('select', { dataset: { role: 'year-breakdown-select' } }));
  ba.appendChild(el('label', { className: 'chart-month-toggle' }, [
    el('input', { dataset: { role: 'month-toggle' }, type: 'checkbox' }),
  ]));
  ba.appendChild(el('canvas', { dataset: { target: 'chart-a' } }));
  ba.appendChild(el('select', { dataset: { role: 'parcel-breakdown-select' } }));
  ba.appendChild(el('canvas', { dataset: { target: 'chart-b' } }));
  frag.append(ha, ba);

  const [hb, bb] = section('b');
  bb.appendChild(el('div', { className: 'prelievi-calendar-wrap', dataset: { target: 'calendar-host' } }));
  frag.append(hb, bb);

  const [hi, bi] = section('i');
  bi.dataset.target = 'table-host';
  frag.append(hi, bi);
  return frag;
}

const contentEl = el('main');
const modalEl = el('div');
const links = [];
const templates = {
  'tmpl-prelievi-page': { content: buildPrelieviTemplate() },
  'tmpl-confirm-modal': { content: buildConfirmTemplate() },
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
    const href = sel.match(/^link\[href="([^"]+)"\]$/)?.[1];
    return href ? links.find(link => link.href === href && !link.removed) || null : null;
  },
  querySelectorAll() { return []; },
};

const tableInstances = [];
class MockSortableTable {
  constructor(opts) {
    this.container = opts.container;
    this.data = opts.data;
    this.columns = opts.columns;
    this.onSort = opts.onSort;
    this.currentSort = opts.sort || { column: opts.columns.find(c => !c.hidden)?.key, ascending: true };
    this.filteredRows = opts.data;
    tableInstances.push(this);
  }
  filter(fn) { this.lastFilter = fn; this.filteredRows = this.data.filter(fn); }
  clearFilter() { this.lastFilter = null; this.filteredRows = this.data; }
  setData(rows) { this.data = rows; this.filteredRows = rows; }
  sort(column, _type, ascending) {
    this.currentSort = { column, ascending };
    this.onSort?.(column, ascending);
  }
  destroy() { this.destroyed = true; }
}

const chartInstances = [];
let browserConfirmCalls = 0;
globalThis.confirm = () => { browserConfirmCalls += 1; return false; };
const browserLocation = { origin: 'http://localhost', pathname: '/prelievi', search: '' };
globalThis.location = browserLocation;
globalThis.history = {
  replaceState(_state, _title, url) { updateLocation(url); },
  pushState(_state, _title, url) { updateLocation(url); },
};
function updateLocation(url) {
  const parsed = new URL(url, browserLocation.origin);
  browserLocation.pathname = parsed.pathname;
  browserLocation.search = parsed.search;
}
globalThis.window = {
  location: browserLocation,
  history: globalThis.history,
  SortableTable: MockSortableTable,
  Chart: class {
    constructor(_canvas, opts) {
      this.type = opts.type;
      this.data = opts.data;
      this.options = opts.options;
      chartInstances.push(this);
    }
    destroy() { this.destroyed = true; }
    update() {}
  },
};

const S = await import(staticModule('base/js/strings.js'));
const PrelieviCharts = await import(staticModule('prelievi/js/charts.js'));
const PrelieviCalendar = await import(staticModule('prelievi/js/calendar.js'));
const speciesDigest = {
  columns: [ROW_ID, S.COL_NAME],
  rows: [[1, 'Abete'], [2, 'Abete Rosso'], [3, 'Castagno']],
};

const chartColMap = {
  [S.COL_DATE]: 0, [S.COL_REGION]: 1, [S.COL_PARCEL]: 2,
  Abete: 3, Faggio: 4, [S.COL_QUINTALS]: 5, [S.COL_CREW]: 6,
  [S.COL_TYPE]: 7, 'Tractor One': 8,
};
const chartRows = [
  ['2020-01-01', 'A', '1', 10, 0, 10, 'Crew A', 'Taglio', 10],
  ['2020-01-02', 'A', '2', 0, 5, 5, 'Crew B', 'Taglio', 0],
];
const allChartSpecies = ['Abete', 'Castagno', 'Faggio'];
let speciesChart = PrelieviCharts.aggregateTimeSeries(
  chartRows, chartColMap, 'specie', false, ['Abete', 'Faggio'], [], allChartSpecies,
);
eq(speciesChart.datasets.map(d => [d.label, d.backgroundColor]),
   [['Abete', '#2e7d32'], ['Faggio', '#144b99']],
   'aggregateTimeSeries keeps species colors stable across omitted species');
const gapRows = [
  ['2020-01-01', 'A', '1', 10, 0, 10],
  ['2022-01-01', 'A', '1', 0, 5, 5],
];
let gapChart = PrelieviCharts.aggregateTimeSeries(
  gapRows, chartColMap, 'total', false, ['Abete', 'Faggio'], [], allChartSpecies,
);
eq(gapChart.labels, ['2020', '2021', '2022'],
   'aggregateTimeSeries fills missing yearly buckets');
eq(gapChart.datasets[0].data, [10, 0, 5],
   'aggregateTimeSeries zero-fills missing yearly buckets');
gapChart = PrelieviCharts.aggregateTimeSeries(
  [
    ['2020-01-01', 'A', '1', 10, 0, 10],
    ['2020-03-01', 'A', '1', 0, 5, 5],
  ],
  chartColMap, 'specie', true, ['Abete', 'Faggio'], [], allChartSpecies,
);
eq(gapChart.labels, ['2020-01', '2020-02', '2020-03'],
   'aggregateTimeSeries fills missing monthly buckets');
eq(gapChart.datasets.find(d => d.label === 'Faggio').data, [0, 0, 5],
   'aggregateTimeSeries zero-fills missing monthly species buckets');
speciesChart = PrelieviCharts.aggregateSpeciesByParcel(
  chartRows, chartColMap, ['Abete', 'Faggio'], allChartSpecies,
);
eq(speciesChart.datasets.map(d => [d.label, d.backgroundColor]),
   [['Abete', '#2e7d32'], ['Faggio', '#144b99']],
   'aggregateSpeciesByParcel keeps species colors stable across omitted species');
let parcelChart = PrelieviCharts.aggregateParcelSeries(
  chartRows, chartColMap, 'total', ['Abete', 'Faggio'], ['Tractor One'], allChartSpecies,
);
eq(parcelChart.datasets.map(d => [d.label, d.data]),
   [['Totale', [10, 5]]],
   'aggregateParcelSeries total groups quintals by parcel');
parcelChart = PrelieviCharts.aggregateParcelSeries(
  chartRows, chartColMap, 'squadra', ['Abete', 'Faggio'], ['Tractor One'], allChartSpecies,
);
eq(parcelChart.datasets.map(d => [d.label, d.data]),
   [['Crew A', [10, 0]], ['Crew B', [0, 5]]],
   'aggregateParcelSeries team breakdown groups row totals by parcel');
parcelChart = PrelieviCharts.aggregateParcelSeries(
  chartRows, chartColMap, 'trattore', ['Abete', 'Faggio'], ['Tractor One'], allChartSpecies,
);
eq(parcelChart.datasets.map(d => [d.label, d.data]),
   [['Tractor One', [10]]],
   'aggregateParcelSeries tractor breakdown pivots tractor columns by parcel');

const digest = {
  columns: [
    ROW_ID, VERSION, COL_REGION_ID, COL_PARCEL_ID,
    S.COL_DATE, S.COL_REGION, S.COL_PARCEL, S.COL_CREW,
    S.COL_TYPE, S.COL_QUINTALS, S.COL_VOLUME_M3, S.COL_NOTE,
    'Abete', 'Abete %', 'Abete Rosso', 'Abete Rosso %', 'Tractor One',
  ],
  rows: [
    [1, 1, 10, 101, '2020-03-15', 'A', '1', 'Crew', 'Taglio', 10, 1, '', 10, 100, 0, 0, 0],
    [2, 1, 10, 102, '2021-04-20', 'A', '2', 'Crew', 'Taglio', 20, 2, '', 120, 100, 0, 0, 0],
    [3, 1, 20, 203, '2022-05-10', 'B', '3', 'Crew', 'Taglio', 30, 3, '', 30, 50, 150, 50, 0],
  ],
};
const calendarData = PrelieviCalendar.buildHarvestCalendar(digest.rows, {
  [S.COL_DATE]: digest.columns.indexOf(S.COL_DATE),
  [S.COL_REGION]: digest.columns.indexOf(S.COL_REGION),
  [S.COL_PARCEL]: digest.columns.indexOf(S.COL_PARCEL),
});
eq(calendarData.periods, ['2020-03', '2021-04', '2022-05'],
   'buildHarvestCalendar uses monthly buckets from filtered harvest rows');
eq(calendarData.regions.map(region => [
  region.name, region.parcels.map(parcel => parcel.parcel),
]), [['A', ['1', '2']], ['B', ['3']]],
   'buildHarvestCalendar groups parcels by region');
eq(PrelieviCalendar.calendarSearchText(
  'squadra:zaffino 2018-01 Compresa:Serra Particella:1',
  { period: '2019-06', region: 'Fabrizia', parcel: '14b' },
), 'squadra:zaffino 2019-06 Compresa:Fabrizia Particella:14b',
   'calendarSearchText replaces prior calendar filters and preserves other terms');

let prelieviDigest = digest;
let prelieviLastModified = 'v1';

globalThis.fetch = async (url) => {
  const payloads = {
    '/api/prelievi/data/': prelieviDigest,
    '/api/species/data/': speciesDigest,
  };
  if (!payloads[url]) throw new Error(`unexpected fetch ${url}`);
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? prelieviLastModified : null },
    json: async () => payloads[url],
  };
};

const cache = await import(staticModule('base/js/cache.js'));
const prelievi = await import(staticModule('prelievi/js/prelievi.js'));

function parcelOption(value, region, name = value) {
  return {
    value: String(value),
    dataset: { region: String(region), name: String(name) },
    parentNode: null,
    remove() {
      if (!this.parentNode) return;
      this.parentNode.options = this.parentNode.options.filter(o => o !== this);
      this.parentNode = null;
    },
  };
}

function parcelSelect(value, options) {
  const select = {
    value: String(value),
    options: [...options],
    appendChild(option) {
      option.parentNode = this;
      this.options.push(option);
      // Model the browser hazard this regression pins: rebuilding a select can
      // leave it on a default option unless code explicitly restores value.
      this.value = option.value;
      return option;
    },
  };
  for (const option of select.options) option.parentNode = select;
  return select;
}

{
  const p1 = parcelOption(101, 10, '1');
  const saved = parcelOption(102, 10, '2');
  const later = parcelOption(109, 10, '9');
  const otherRegion = parcelOption(203, 20, '3');
  const allOptions = [p1, saved, otherRegion, later];
  const select = parcelSelect(saved.value, allOptions);

  prelievi.filterParcelSelectForRegion(select, allOptions, '10');

  eq(select.options.map(o => o.value), ['101', '102', '109'],
     'region-wide parcel selector filters to the selected cantiere region');
  eq(select.value, '102',
     'region-wide edit preserves the saved parcel after rebuilding options');
}

{
  const x = parcelOption(100, 10, 'X');
  const p1 = parcelOption(101, 10, '1');
  const otherRegion = parcelOption(203, 20, '3');
  const allOptions = [otherRegion, x, p1];
  const select = parcelSelect(otherRegion.value, allOptions);

  prelievi.filterParcelSelectForRegion(select, allOptions, '10');

  eq(select.value, '100',
     'region-wide parcel selector falls back to the whole-region sentinel');
}

function sliderValues() {
  const min = contentEl.querySelector('[data-role="slider-min"]');
  const max = contentEl.querySelector('[data-role="slider-max"]');
  return [Number(min.value), Number(max.value)];
}
function filteredIds() {
  return tableInstances.at(-1).filteredRows.map(row => row[0]);
}

eq(prelievi.boscoUrlForHarvestRow(digest.rows[1], digest.columns), '/bosco?c=10&v=1&pa=102',
   'boscoUrlForHarvestRow builds a parcel overlay URL from stable ids');
eq(prelievi.boscoUrlForHarvestRow(digest.rows[1], [ROW_ID, S.COL_DATE]), null,
   'boscoUrlForHarvestRow returns null without stable ids');

await prelievi.mount({ y1: '2021', y2: '2021' });
eq(sliderValues(), [2021, 2021], 'mount applies explicit year range from URL');
eq(filteredIds(), [2], 'mount filters table to explicit year range');

prelievi.onQueryChange({});
eq(sliderValues(), [2020, 2022], 'bare URL resets year slider to available endpoints');
eq(filteredIds(), [1, 2, 3], 'bare URL restores all years in the table filter');

prelievi.onQueryChange({ y1: '2021' });
eq(sliderValues(), [2021, 2022], 'partial y1 URL uses requested lower bound and default upper bound');
eq(filteredIds(), [2, 3], 'partial y1 URL filters through the default upper bound');

prelievi.onQueryChange({ y2: '2021' });
eq(sliderValues(), [2020, 2021], 'partial y2 URL uses default lower bound and requested upper bound');
eq(filteredIds(), [1, 2], 'partial y2 URL filters from the default lower bound');

prelievi.onQueryChange({});
eq(sliderValues(), [2020, 2022], 'second bare URL reset is not ignored after partial states');
eq(filteredIds(), [1, 2, 3], 'second bare URL reset restores all years again');

prelievi.onQueryChange({ c: '10' });
eq(filteredIds(), [1, 2], 'region URL filter keeps matching harvests');

prelievi.onQueryChange({ c: '10', pa: '102' });
eq(filteredIds(), [2], 'parcel URL filter narrows matching harvests by stable id');

prelievi.onQueryChange({ c: '20', pa: '102' });
eq(filteredIds(), [], 'region and parcel URL filters are both enforced');

prelievi.onQueryChange({ c: 'bad', pa: '-1' });
eq(filteredIds(), [1, 2, 3], 'invalid URL filter ids are ignored');

prelievi.onQueryChange({ o: 'b' });
let activeCalendarCells = contentEl.querySelectorAll('.prelievi-calendar-cell.active');
eq(activeCalendarCells.map(cell => cell.dataset.period), ['2020-03', '2021-04', '2022-05'],
   'calendar section renders one active month cell per filtered harvest');
activeCalendarCells.find(cell => cell.dataset.period === '2021-04').click();
eq(filteredIds(), [2],
   'clicking a calendar cell filters the table by month, region, and parcel');

prelievi.onQueryChange({ o: 'a', pb: 'specie' });
eq(chartInstances.at(-1).data.datasets.map(d => d.label), ['Abete', 'Abete Rosso'],
   'per-parcel species chart excludes tractor columns');

const castagnoDigest = {
  columns: [...digest.columns.slice(0, -1), 'Castagno', 'Castagno %', digest.columns.at(-1)],
  rows: digest.rows.map((row, index) => [
    ...row.slice(0, -1), index === 2 ? 75 : 0, index === 2 ? 100 : 0, row.at(-1),
  ]),
};
prelieviDigest = castagnoDigest;
prelieviLastModified = 'v2';
await cache.load('prelievi');
check(tableInstances.at(-1).columns.some(col => col.key === 'Castagno'),
      'background refresh rebuilds the table when digest columns change');
check(chartInstances.at(-1).data.datasets.some(d => d.label === 'Castagno'),
      'background refresh reclassifies new species columns for open charts');

prelievi.onQueryChange({ f: 'abete:>100' });
eq(filteredIds(), [2], 'exact species column search beats longer species substring matches');

prelievi.onQueryChange({ f: 'rosso' });
eq(filteredIds(), [3], 'plain search matches non-zero multi-word species columns');

prelievi.unmount();
check(tableInstances.at(-1).destroyed, 'unmount destroys the table');

// Row deletion must use the shared modal, not window.confirm().
document.body.dataset.role = 'writer';
browserConfirmCalls = 0;
await prelievi.mount({});
const tableEl = contentEl.querySelector('.table-scroll');
const row = el('tr', { className: 'sortable-table-row' });
row.dataset.index = '0';
const del = el('span', { className: 'action-icon action-delete' });
row.appendChild(del);
tableEl.appendChild(row);
del.click();
eq(browserConfirmCalls, 0, 'delete action does not call the browser confirm');
check(Boolean(modalEl.querySelector('[data-action="confirm"]')),
      'delete action opens the shared confirm modal');
prelievi.unmount();
document.body.dataset.role = 'reader';

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
