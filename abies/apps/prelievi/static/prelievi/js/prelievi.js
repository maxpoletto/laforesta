/**
 * Prelievi page: harvest operations table with year slider and CRUD forms.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import { fetchForm, renderFormHTML, interceptSubmit } from '../../base/js/forms.js';
import { postJSON } from '../../base/js/api.js';
import { showError } from '../../base/js/modals.js';
import { createRangeSlider } from '../../base/js/range-slider.js';
import * as S from '../../base/js/strings.js';
import { matchesSearch } from '../../base/js/table.js';
import {
  aggregateTimeSeries, aggregateSpeciesByParcel, renderStackedBar,
} from './charts.js';

const CSS_URL = '/static/prelievi/css/prelievi.css';
const DATA_ID = 'prelievi';
const DATA_URL = '/abies/api/prelievi/data/';
const FORM_URL = '/abies/api/prelievi/form/';
const SAVE_URL = '/abies/api/prelievi/save/';
const DELETE_URL = '/abies/api/prelievi/delete/';
const PAGE_PATH = '/abies/prelievi';

// Collapsible sections, keyed by the single-char token used in the URL `o`
// parameter ('a' = Produzione chart, 'b' = Specie-per-particella chart,
// 'i' = Interventi table).
const SECTION_KEYS = ['a', 'b', 'i'];
const DEFAULT_OPEN = 'i';                 // default when `o` param is absent

// ---------------------------------------------------------------------------
// Number formatters
// ---------------------------------------------------------------------------

/** Format quintals: one decimal, comma separator. */
function formatQuintals(value) {
  if (value == null || value === '') return '';
  return typeof value === 'number' ? value.toFixed(1).replace('.', ',') : value;
}

/** Format quintals: one decimal, comma separator. Blank for zero. */
function formatQuintalsBlankZero(value) {
  if (!value) return '';
  return typeof value === 'number' ? value.toFixed(1).replace('.', ',') : value;
}

/** Format plain integer — no thousands separator. Blank for null. */
function formatInteger(value) {
  if (value == null || value === '') return '';
  return String(value);
}

/** Column definitions for the fixed digest columns. */
const STATIC_COLS = {
  'Data': { label: S.COL_DATE, type: 'date', width: '90px' },
  'Compresa': { label: S.COL_REGION, width: '80px' },
  'Particella': { label: S.COL_PARCEL, width: '70px' },
  'Squadra': { label: S.COL_CREW, width: '108px' },
  'Tipo': { label: S.COL_OPTYPE, width: '120px' },
  'VDP': { label: S.COL_VDP, type: 'number', width: '55px', formatter: formatInteger },
  'Q.li': { label: S.COL_QUINTALS, type: 'number', width: '55px', formatter: formatQuintals },
  'Note': { label: S.COL_NOTE, width: '110px' },
  'Altre note': { label: S.COL_EXTRA_NOTE, width: '90px' },
  'version': { label: 'version', hidden: true },
};

// Column indices — resolved on first data load.
let colDate = -1;
let colVersion = -1;

// Page state.
let table = null;
let slider = null;
let unsubCache = null;
let inForm = false;
let escapeHandler = null;

// Column classification and index map — resolved on first data load.
let speciesCols = [];
let tractorCols = [];
let colMap = {};

// Section state.  Chart sections carry their own open state, canvas,
// Chart.js instance, dirty flag, and render function.  The 'i' section
// just hosts the TableWrapper's container.
const sections = {
  a: {
    title: S.CHART_PRODUCTION,
    open: false, dirty: true,
    canvas: null, instance: null, header: null, body: null,
    breakdown: 'total', byMonth: false,
    render: () => _renderChart(sections.a),
    aggregate: () => aggregateTimeSeries(
      _getFilteredRows(), colMap,
      sections.a.breakdown, sections.a.byMonth,
      speciesCols, tractorCols,
    ),
  },
  b: {
    title: S.CHART_SPECIES_BY_PARCEL,
    open: false, dirty: true,
    canvas: null, instance: null, header: null, body: null,
    render: () => _renderChart(sections.b),
    aggregate: () => aggregateSpeciesByParcel(
      _getFilteredRows(), colMap, speciesCols,
    ),
  },
  i: {
    title: S.SECTION_INTERVENTI,
    open: true,
    header: null, body: null,
  },
};

cache.register(DATA_ID, DATA_URL);

function canModify() {
  return ['admin', 'writer'].includes(document.body.dataset.role);
}

// ---------------------------------------------------------------------------
// Page lifecycle (exported for router)
// ---------------------------------------------------------------------------

export async function mount(params) {
  inForm = false;
  loadCSS(CSS_URL);
  const el = document.getElementById('content');
  el.replaceChildren();

  const loading = document.createElement('div');
  loading.className = 'loading-overlay';
  loading.textContent = S.LOADING;
  el.appendChild(loading);

  let data;
  try {
    data = await cache.load(DATA_ID);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  colDate = data.columns.indexOf('Data');
  colVersion = data.columns.indexOf('version');
  _buildColMap(data.columns);
  _classifyColumns(data.columns);

  showTableView(data, params);

  cache.setVisible([DATA_ID]);
  unsubCache = cache.onUpdate(DATA_ID, onCacheUpdate);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (unsubCache) { unsubCache(); unsubCache = null; }
  removeEscapeHandler();
  _destroyCharts();
  destroyTable();
  cache.setVisible([]);
}

export function onQueryChange(params) {
  if (inForm) return;
  applyParams(params);
}

// ---------------------------------------------------------------------------
// Table view
// ---------------------------------------------------------------------------

function showTableView(data, params) {
  inForm = false;
  removeEscapeHandler();
  _destroyCharts();                       // clean up any previous chart instances
  const el = document.getElementById('content');
  el.replaceChildren();

  const p = readParams(params);

  // Top filter bar: year slider + search + reset + CSV export.
  const fb = _buildFilterBar(el, data, p);

  // Three collapsible sections (chart A, chart B, interventi/table).
  // Section 'i'.body becomes the TableWrapper's container.
  _buildSections(el, p);

  // Table itself.
  const sort = p.sc
    ? { column: p.sc, ascending: p.so }
    : { column: 'Data', ascending: false };

  const modify = canModify();
  table = new TableWrapper({
    container: sections.i.body,
    digest: data,
    columnDefs: buildColumnDefs(data.columns),
    inlineToolbar: false,                 // we provide search + CSV in the filter bar
    canModify: modify,
    actions: modify ? {
      onAdd: () => showAddForm(),
      onEdit: (rowId) => showEditForm(rowId),
      onDelete: (rowId) => confirmDelete(rowId),
    } : {},
    sort,
    searchText: p.f,
    csvFilename: S.CSV_PRELIEVI,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => { syncURL(); _updateCharts(); },
  });

  // Wire the external search input and CSV button to the table.
  table.wireSearchInput(fb.searchInput);
  fb.csvBtn.addEventListener('click', () => table.exportCSV());

  table.setExternalFilter(yearFilter());
  _updateCharts();
}

function destroyTable() {
  if (table) { table.destroy(); table = null; }
  slider = null;
}

function onCacheUpdate() {
  if (inForm || !table) return;
  table.setData(cache.get(DATA_ID));
}

// ---------------------------------------------------------------------------
// Year slider
// ---------------------------------------------------------------------------

function buildSlider(container, years, initY1, initY2) {
  if (years.length < 2) return null;

  const title = document.createElement('span');
  title.className = 'prelievi-filter-label';
  title.textContent = S.LABEL_YEARS;

  const label = document.createElement('span');
  label.className = 'prelievi-slider-label';

  // Group label + slider in a box the same width as the search input below,
  // so their right edges align.
  const controls = document.createElement('div');
  controls.className = 'prelievi-slider-controls';

  const wrapper = document.createElement('div');
  wrapper.className = 'range-slider';
  const minInput = document.createElement('input');
  minInput.type = 'range';
  const maxInput = document.createElement('input');
  maxInput.type = 'range';
  wrapper.append(minInput, maxInput);

  controls.append(label, wrapper);
  container.append(title, controls);

  const rs = createRangeSlider(minInput, maxInput, label, () => {
    if (table) table.setExternalFilter(yearFilter());
    syncURL();
    _updateCharts();
  });

  rs.setRange(years);
  if (initY1 != null || initY2 != null) {
    rs.setValues(initY1 ?? years[0], initY2 ?? years[years.length - 1]);
  }
  return rs;
}

function extractYears(rows) {
  const s = new Set();
  for (const row of rows) {
    const d = row[colDate];
    if (d) s.add(parseInt(String(d).substring(0, 4), 10));
  }
  const arr = [...s].sort((a, b) => a - b);
  return arr.length ? arr : [new Date().getFullYear()];
}

function yearFilter() {
  if (!slider) return null;
  const [y1, y2] = slider.getRange();
  return (row) => {
    const d = row[colDate];
    if (!d) return false;
    const y = parseInt(String(d).substring(0, 4), 10);
    return y >= y1 && y <= y2;
  };
}

// ---------------------------------------------------------------------------
// URL parameter sync
// ---------------------------------------------------------------------------

function readParams(params) {
  return {
    y1: params.y1 ? parseInt(params.y1, 10) : null,
    y2: params.y2 ? parseInt(params.y2, 10) : null,
    sc: params.sc || null,
    so: params.so !== undefined ? params.so === '0' : true,
    f: params.f || '',
    // Open sections: explicit string of single-char tokens when present,
    // falling back to DEFAULT_OPEN when absent.  `?o=` (empty) is valid
    // and means "all sections closed".
    o: params.o !== undefined ? params.o : DEFAULT_OPEN,
    b: params.b || 'total',
    m: params.m === '1',
  };
}

function applyParams(params) {
  const p = readParams(params);

  // Year slider.
  if (slider && (p.y1 != null || p.y2 != null)) {
    const data = cache.get(DATA_ID);
    if (data) {
      const years = extractYears(data.rows);
      slider.setValues(p.y1 ?? years[0], p.y2 ?? years[years.length - 1]);
      if (table) table.setExternalFilter(yearFilter());
    }
  }

  // Chart A configuration.
  const a = sections.a;
  if (a.body) {
    if (a.breakdown !== p.b) {
      a.breakdown = p.b;
      const sel = a.body.querySelector('select');
      if (sel) sel.value = p.b;
      a.dirty = true;
    }
    if (a.byMonth !== p.m) {
      a.byMonth = p.m;
      const cb = a.body.querySelector('.chart-month-toggle input');
      if (cb) cb.checked = p.m;
      a.dirty = true;
    }
  }

  // Open sections.
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    const shouldBeOpen = p.o.includes(k);
    if (s.body && s.open !== shouldBeOpen) {
      s.open = shouldBeOpen;
      s.header.classList.toggle('open', shouldBeOpen);
      s.body.classList.toggle('open', shouldBeOpen);
    }
    if (s.open && s.render && s.dirty) s.render();
  }
}

function syncURL() {
  const params = new URLSearchParams();

  if (slider) {
    const data = cache.get(DATA_ID);
    if (data) {
      const years = extractYears(data.rows);
      const [y1, y2] = slider.getRange();
      if (y1 !== years[0]) params.set('y1', y1);
      if (y2 !== years[years.length - 1]) params.set('y2', y2);
    }
  }

  if (table) {
    const sort = table.getSort();
    if (sort) {
      params.set('sc', sort.column);
      params.set('so', sort.ascending ? '0' : '1');
    }
    const f = table.getSearchText();
    if (f) params.set('f', f);
  }

  // Open sections: only serialize if different from the default ('i').
  const openKeys = SECTION_KEYS.filter(k => sections[k].open).join('');
  if (openKeys !== DEFAULT_OPEN) params.set('o', openKeys);

  // Chart A config: only serialize non-default values.
  if (sections.a.breakdown !== 'total') params.set('b', sections.a.breakdown);
  if (sections.a.byMonth) params.set('m', '1');

  const qs = params.toString();
  router.navigate(PAGE_PATH + (qs ? '?' + qs : ''), true);
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

function _buildColMap(columns) {
  colMap = {};
  for (let i = 0; i < columns.length; i++) colMap[columns[i]] = i;
}

function _classifyColumns(columns) {
  speciesCols = [];
  tractorCols = [];
  for (const name of columns) {
    if (name === 'row_id' || STATIC_COLS[name] || name.endsWith(' %')) continue;
    // Species are single words; tractor labels contain a space.
    if (name.includes(' ')) tractorCols.push(name);
    else speciesCols.push(name);
  }
}

function _getFilteredRows() {
  const data = cache.get(DATA_ID);
  if (!data) return [];
  const yf = yearFilter();
  const text = table ? table.getSearchText() : '';
  const terms = text.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return data.rows.filter(row => {
    if (yf && !yf(row)) return false;
    if (terms.length && !matchesSearch(row, terms)) return false;
    return true;
  });
}

function _updateCharts() {
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    if (!s.render) continue;
    s.dirty = true;
    if (s.open) s.render();
  }
}

/** Render a chart section from its (filtered) aggregation output. */
function _renderChart(s) {
  if (!s.canvas) return;
  s.instance = renderStackedBar(s.canvas, s.aggregate(), s.instance);
  s.dirty = false;
}

function _destroyCharts() {
  for (const s of Object.values(sections)) {
    if (s.instance) { s.instance.destroy(); s.instance = null; }
    s.canvas = null;
    s.header = null;
    s.body = null;
    s.dirty = true;
  }
}

function _buildFilterBar(el, data, p) {
  const bar = document.createElement('div');
  bar.className = 'prelievi-filter-bar';

  // Year slider (appends "Anni" label + controls to bar, returns slider).
  const years = extractYears(data.rows);
  slider = buildSlider(bar, years, p.y1, p.y2);

  // "Filtra" label
  const filterLabel = document.createElement('label');
  filterLabel.className = 'prelievi-filter-label';
  filterLabel.textContent = S.FILTER_LABEL;

  // Search input
  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.className = 'prelievi-search';
  searchInput.placeholder = S.SEARCH_PLACEHOLDER;
  filterLabel.htmlFor = searchInput.id = 'prelievi-search';

  // Reset button: clears both year range and search filter.
  const resetBtn = document.createElement('button');
  resetBtn.className = 'btn btn-primary';
  resetBtn.textContent = S.RESET_FILTERS;
  resetBtn.addEventListener('click', () => {
    if (slider) {
      slider.setValues(years[0], years[years.length - 1]);
      if (table) table.setExternalFilter(yearFilter());
    }
    if (table) table.setSearchText('');
    syncURL();
    _updateCharts();
  });

  // CSV export button — wired to table.exportCSV() by caller.
  const csvBtn = document.createElement('button');
  csvBtn.className = 'btn btn-primary prelievi-csv-btn';
  csvBtn.textContent = S.EXPORT_CSV;

  bar.append(filterLabel, searchInput, resetBtn, csvBtn);
  el.appendChild(bar);

  return { searchInput, csvBtn };
}

/** Build all three collapsible sections (a, b, i) and wire toggle + URL sync. */
function _buildSections(el, p) {
  // Initialize open state + chart A config from URL params before building.
  sections.a.breakdown = p.b;
  sections.a.byMonth = p.m;
  for (const k of SECTION_KEYS) {
    sections[k].open = p.o.includes(k);
  }

  _buildSection(el, sections.a, _buildChartABody);
  _buildSection(el, sections.b, _buildChartBBody);
  _buildSection(el, sections.i, null);     // table; body is populated by caller
}

function _buildSection(el, s, buildBody) {
  const [header, body] = _collapsible(s.title, s.open);
  s.header = header;
  s.body = body;
  if (buildBody) buildBody(body, s);

  header.addEventListener('click', () => {
    s.open = !s.open;
    header.classList.toggle('open', s.open);
    body.classList.toggle('open', s.open);
    if (s.open && s.render && s.dirty) s.render();
    syncURL();
  });

  el.append(header, body);
}

function _buildChartABody(body, s) {
  const controls = document.createElement('div');
  controls.className = 'chart-controls';

  const sel = document.createElement('select');
  for (const [value, label] of [
    ['total', S.CHART_TOTAL], ['compresa', S.COL_REGION],
    ['particella', S.COL_PARCEL], ['squadra', S.COL_CREW],
    ['specie', S.LABEL_SPECIES], ['trattore', S.COL_TRACTOR],
    ['tipo', S.COL_OPTYPE],
  ]) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    if (value === s.breakdown) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => {
    s.breakdown = sel.value;
    s.render();
    syncURL();
  });
  controls.appendChild(sel);

  const monthLabel = document.createElement('label');
  monthLabel.className = 'chart-month-toggle';
  const monthCb = document.createElement('input');
  monthCb.type = 'checkbox';
  monthCb.checked = s.byMonth;
  monthCb.addEventListener('change', () => {
    s.byMonth = monthCb.checked;
    s.render();
    syncURL();
  });
  monthLabel.append(monthCb, ' ' + S.CHART_BY_MONTHS);
  controls.appendChild(monthLabel);

  body.appendChild(controls);

  s.canvas = document.createElement('canvas');
  const box = document.createElement('div');
  box.className = 'chart-container';
  box.appendChild(s.canvas);
  body.appendChild(box);
}

function _buildChartBBody(body, s) {
  s.canvas = document.createElement('canvas');
  const box = document.createElement('div');
  box.className = 'chart-container';
  box.appendChild(s.canvas);
  body.appendChild(box);
}

function _collapsible(title, open = false) {
  const header = document.createElement('div');
  header.className = 'collapsible-header' + (open ? ' open' : '');
  const span = document.createElement('span');
  span.textContent = title;
  const arrow = document.createElement('span');
  arrow.className = 'arrow';
  header.append(span, arrow);

  const body = document.createElement('div');
  body.className = 'collapsible-body' + (open ? ' open' : '');
  return [header, body];
}

// ---------------------------------------------------------------------------
// Add / Edit forms
// ---------------------------------------------------------------------------

async function showAddForm() {
  inForm = true;
  destroyTable();
  const form = await fetchForm(FORM_URL);
  if (!form) { returnToTable(); return; }
  wireForm(form);
}

async function showEditForm(rowId) {
  inForm = true;
  destroyTable();
  const form = await fetchForm(`${FORM_URL}${rowId}/`);
  if (!form) { returnToTable(); return; }
  wireForm(form);
}

/** Client-side validation before POST. Returns error message or null. */
function validateForm(body) {
  // Future date check.
  if (body.date && body.date > new Date().toISOString().slice(0, 10)) {
    return S.ERR_DATE_FUTURE;
  }
  // Species percentages must sum to 100.
  let spSum = 0;
  let trSum = 0;
  for (const [key, val] of Object.entries(body)) {
    const n = parseInt(val, 10) || 0;
    if (key.startsWith('sp_')) spSum += n;
    else if (key.startsWith('tr_')) trSum += n;
  }
  if (spSum !== 100) return S.ERR_SPECIES_PCT_SUM;
  if (trSum !== 100) return S.ERR_TRACTOR_PCT_SUM;
  return null;
}

function wireForm(form) {
  wireRegionCascade(form);
  wire100Buttons(form);

  interceptSubmit(form, SAVE_URL, {
    validate: validateForm,
    onSuccess(data, isSaveAndAdd) {
      cache.updateRow(DATA_ID, data.row_id, data.record);
      if (isSaveAndAdd) {
        showAddForm();
      } else {
        returnToTable();
      }
    },
    onConflict(data) {
      if (data.record) cache.updateRow(DATA_ID, data.row_id, data.record);
      if (data.html) {
        const newForm = renderFormHTML(data.html);
        if (newForm) wireForm(newForm);
      }
    },
    onValidationError(data) {
      if (data.html) {
        const newForm = renderFormHTML(data.html);
        if (newForm) wireForm(newForm);
      }
    },
  });

  addEscapeHandler();
}

/** Filter parcel options when region changes. */
function wireRegionCascade(form) {
  const regionSel = form.querySelector('#id_region');
  const parcelSel = form.querySelector('#id_parcel');
  if (!regionSel || !parcelSel) return;

  const allOptions = [...parcelSel.querySelectorAll('option')];

  function filterParcels() {
    const rid = regionSel.value;
    const current = parcelSel.value;
    for (const opt of allOptions) opt.remove();
    for (const opt of allOptions) {
      if (opt.dataset.region === rid) parcelSel.appendChild(opt);
    }
    if ([...parcelSel.options].some(o => o.value === current)) {
      parcelSel.value = current;
    }
  }

  regionSel.addEventListener('change', filterParcels);
  filterParcels();
}

/** Wire the "100%" quick-set buttons for species/tractor percentages. */
function wire100Buttons(form) {
  form.addEventListener('click', (e) => {
    const btn = e.target.closest('.btn-100');
    if (!btn) return;
    e.preventDefault();
    const group = btn.dataset.group;
    const target = btn.dataset.target;
    for (const input of form.querySelectorAll(`input[name^="${group}_"]`)) {
      input.value = input.name === target ? '100' : '0';
    }
  });
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

async function confirmDelete(rowId) {
  if (!confirm(S.DELETE_CONFIRM)) return;

  const data = cache.get(DATA_ID);
  if (!data) return;

  const row = data.rows.find(r => r[0] === rowId);
  if (!row) return;

  const version = colVersion >= 0 ? row[colVersion] : 0;

  let resp;
  try {
    resp = await postJSON(DELETE_URL, {
      row_id: String(rowId),
      version: String(version),
      nonce: crypto.randomUUID(),
    });
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  if (resp.status === 200) {
    cache.removeRow(DATA_ID, rowId);
    if (table) table.setData(cache.get(DATA_ID));
    return;
  }

  if (resp.data.status === 'conflict' && resp.data.record) {
    cache.updateRow(DATA_ID, resp.data.row_id, resp.data.record);
    if (table) table.setData(cache.get(DATA_ID));
  }
  showError(resp.data.message || S.ERROR_GENERIC);
}

// ---------------------------------------------------------------------------
// Escape key — cancel form, return to table
// ---------------------------------------------------------------------------

function addEscapeHandler() {
  removeEscapeHandler();
  escapeHandler = (e) => {
    if (e.key !== 'Escape') return;
    // Let modal handle its own Escape dismissal.
    if (document.getElementById('modal-container').classList.contains('open')) return;
    returnToTable();
  };
  document.addEventListener('keydown', escapeHandler);
}

function removeEscapeHandler() {
  if (escapeHandler) {
    document.removeEventListener('keydown', escapeHandler);
    escapeHandler = null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function returnToTable() {
  const data = cache.get(DATA_ID);
  const params = Object.fromEntries(new URLSearchParams(location.search));
  if (!data) { mount(params); return; }
  showTableView(data, params);
}

/**
 * Build columnDefs from digest columns.
 * Known columns get labels from STATIC_COLS.
 * Columns ending in " %" are hidden (used for form pre-population only).
 * Dynamic species/tractor quintal columns default to type 'number'.
 */
function buildColumnDefs(columns) {
  const defs = {};
  for (const name of columns) {
    if (name === 'row_id') continue;
    if (name.endsWith(' %')) {
      defs[name] = { label: name, hidden: true };
      continue;
    }
    if (STATIC_COLS[name]) {
      defs[name] = STATIC_COLS[name];
      continue;
    }
    // Dynamic quintal column: species names are single words, tractor labels
    // contain a space (manufacturer + model).
    const isTractor = name.includes(' ');
    defs[name] = {
      label: name, type: 'number',
      width: isTractor ? '100px' : '90px',
      className: 'col-wrap-header',
      formatter: formatQuintalsBlankZero,
    };
  }
  return defs;
}

function loadCSS(url) {
  if (document.querySelector(`link[href="${url}"]`)) return;
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = url;
  document.head.appendChild(link);
}

function unloadCSS(url) {
  document.querySelector(`link[href="${url}"]`)?.remove();
}
