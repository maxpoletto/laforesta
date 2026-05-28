/**
 * Prelievi page: harvest operations table with year slider and CRUD forms.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  fetchModalForm, renderModalForm, showFormError, wireCancelButtons,
} from '../../base/js/forms.js';
import { wireActions, showLoadingIn } from '../../base/js/form-widgets.js';
import { canModify } from '../../base/js/roles.js';
import { loadCSS, unloadCSS } from '../../base/js/page-css.js';
import { postJSON } from '../../base/js/api.js';
import { showError, dismiss as dismissModal, onDismiss } from '../../base/js/modals.js';
import { createRangeSlider } from '../../base/js/range-slider.js';
import * as S from '../../base/js/strings.js';
import {
  FIELD_NONCE, ROW_ID, STATUS_CONFLICT, VERSION,
} from '../../base/js/constants.js';
import {
  fmtDecimal1, fmtDecimal1BlankZero, fmtDecimal2, fmtInt,
} from '../../base/js/format.js';
import { matchesSearch } from '../../base/js/table.js';
import {
  aggregateTimeSeries, aggregateSpeciesByParcel, renderStackedBar,
} from './charts.js';
import { cloneTemplate } from '../../base/js/templates.js';

const CSS_URL = '/static/prelievi/css/prelievi.css';
const DATA_ID = 'prelievi';
const DATA_URL = '/api/prelievi/data/';
const FORM_URL = '/api/prelievi/form/';
const SAVE_URL = '/api/prelievi/save/';
const DELETE_URL = '/api/prelievi/delete/';
const PAGE_PATH = '/prelievi';
const HARVEST_PLAN_ITEMS_ID = 'harvest_plan_items';

// Collapsible sections, keyed by the single-char token used in the URL `o`
// parameter ('a' = Produzione chart, 'b' = Specie-per-particella chart,
// 'i' = Interventi table).
const SECTION_KEYS = ['a', 'b', 'i'];
const DEFAULT_OPEN = 'i';                 // default when `o` param is absent

/** Column definitions for the fixed digest columns. */
const STATIC_COLS = {
  [S.COL_DATE]:        { label: S.COL_DATE, type: 'date', width: '90px' },
  [S.COL_COMPRESA]:    { label: S.COL_COMPRESA, width: '80px' },
  [S.COL_PARCEL]:      { label: S.COL_PARCEL, width: '70px' },
  [S.COL_CREW]:        { label: S.COL_CREW, width: '108px' },
  [S.COL_TYPE]:        { label: S.COL_TYPE, width: '120px' },
  [S.COL_VDP]:         { label: S.COL_VDP, type: 'number', width: '55px', formatter: fmtInt },
  [S.COL_QUINTALS]:    { label: S.COL_QUINTALS, type: 'number', width: '55px', formatter: fmtDecimal1 },
  [S.COL_VOLUME_M3]:   { label: S.COL_VOLUME_M3, type: 'number', width: '70px', formatter: fmtDecimal2 },
  [S.COL_NOTE]:        { label: S.COL_NOTE, width: '110px' },
  [S.COL_EXTRA_NOTE]:  { label: S.COL_EXTRA_NOTE, width: '90px' },
  [S.COL_CANTIERE]:    { label: S.COL_CANTIERE, hidden: true },
  [VERSION]:     { label: VERSION, hidden: true },
};

// Column indices — resolved on first data load.
let colDate = -1;
let colVersion = -1;

// Page state.
let table = null;
let slider = null;
let unsubCache = null;
let inForm = false;
let disposePageActions = null;

// Column classification and index map — resolved on first data load.
let speciesCols = [];
let tractorCols = [];
let colMap = {};

// Section state.  Chart sections carry their own open state, canvas,
// Chart.js instance, dirty flag, and render function.  The 'i' section
// just hosts the TableWrapper's container.
const sections = {
  a: {
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
    open: false, dirty: true,
    canvas: null, instance: null, header: null, body: null,
    render: () => _renderChart(sections.b),
    aggregate: () => aggregateSpeciesByParcel(
      _getFilteredRows(), colMap, speciesCols,
    ),
  },
  i: {
    open: true,
    header: null, body: null,
  },
};

cache.register(DATA_ID, DATA_URL);


// ---------------------------------------------------------------------------
// Page lifecycle (exported for router)
// ---------------------------------------------------------------------------

export async function mount(params) {
  inForm = false;
  loadCSS(CSS_URL);
  const el = document.getElementById('content');
  showLoadingIn(el);

  let data;
  try {
    data = await cache.load(DATA_ID);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  colDate = data.columns.indexOf(S.COL_DATE);
  colVersion = data.columns.indexOf(VERSION);
  _buildColMap(data.columns);
  _classifyColumns(data.columns);

  showTableView(data, params);

  cache.setVisible([DATA_ID]);
  unsubCache = cache.onUpdate(DATA_ID, onCacheUpdate);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (unsubCache) { unsubCache(); unsubCache = null; }
  if (disposePageActions) { disposePageActions(); disposePageActions = null; }
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
  _destroyCharts();
  const el = document.getElementById('content');
  el.replaceChildren();

  const p = readParams(params);
  buildPage(el, data, p);

  const sort = p.sc
    ? { column: p.sc, ascending: p.so }
    : { column: S.COL_DATE, ascending: false };

  const modify = canModify();
  table = new TableWrapper({
    container: sections.i.body,
    digest: data,
    columnDefs: buildColumnDefs(data.columns),
    inlineToolbar: false,
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

  const searchInput = el.querySelector('#prelievi-search');
  if (searchInput) table.wireSearchInput(searchInput);

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
    if (name === ROW_ID || STATIC_COLS[name] || name.endsWith(' %')) continue;
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

function buildPage(el, data, p) {
  disposePageActions?.();
  const frag = cloneTemplate('tmpl-prelievi-page');
  el.appendChild(frag);

  // Year slider — uses the template's range inputs.
  const years = extractYears(data.rows);
  const sliderLabel = el.querySelector('.prelievi-slider-label');
  const minInput = el.querySelector('[data-role="slider-min"]');
  const maxInput = el.querySelector('[data-role="slider-max"]');
  if (minInput && maxInput && years.length >= 2) {
    slider = createRangeSlider(minInput, maxInput, sliderLabel, () => {
      if (table) table.setExternalFilter(yearFilter());
      syncURL();
      _updateCharts();
    });
    slider.setRange(years);
    if (p.y1 != null || p.y2 != null) {
      slider.setValues(p.y1 ?? years[0], p.y2 ?? years[years.length - 1]);
    }
  }

  disposePageActions = wireActions(el, {
    'reset-filters': () => {
      if (slider) {
        slider.setValues(years[0], years[years.length - 1]);
        if (table) table.setExternalFilter(yearFilter());
      }
      if (table) table.setSearchText('');
      syncURL();
      _updateCharts();
    },
    'export-csv': () => table?.exportCSV(),
  });

  // Wire collapsible sections.
  sections.a.breakdown = p.b;
  sections.a.byMonth = p.m;
  for (const key of SECTION_KEYS) {
    const s = sections[key];
    s.open = p.o.includes(key);
    s.header = el.querySelector(`[data-section="${key}"].collapsible-header`);
    s.body = el.querySelector(`[data-section="${key}"].collapsible-body`);
    s.header?.classList.toggle('open', s.open);
    s.body?.classList.toggle('open', s.open);
    s.header?.addEventListener('click', () => {
      s.open = !s.open;
      s.header.classList.toggle('open', s.open);
      s.body.classList.toggle('open', s.open);
      if (s.open && s.render && s.dirty) s.render();
      syncURL();
    });
  }

  // Chart A: wire breakdown select and month toggle.
  const a = sections.a;
  a.canvas = el.querySelector('[data-target="chart-a"]');
  const breakdownSel = el.querySelector('[data-role="breakdown-select"]');
  if (breakdownSel) {
    breakdownSel.value = a.breakdown;
    breakdownSel.addEventListener('change', () => {
      a.breakdown = breakdownSel.value;
      a.render();
      syncURL();
    });
  }
  const monthCb = el.querySelector('[data-role="month-toggle"]');
  if (monthCb) {
    monthCb.checked = a.byMonth;
    monthCb.addEventListener('change', () => {
      a.byMonth = monthCb.checked;
      a.render();
      syncURL();
    });
  }

  // Chart B.
  sections.b.canvas = el.querySelector('[data-target="chart-b"]');
}

// ---------------------------------------------------------------------------
// Add / Edit forms
// ---------------------------------------------------------------------------

async function showAddForm() {
  inForm = true;
  const form = await fetchModalForm(FORM_URL);
  if (!form) { inForm = false; return; }
  onDismiss(() => { inForm = false; });
  wireForm(form);
}

async function showEditForm(rowId) {
  inForm = true;
  const form = await fetchModalForm(`${FORM_URL}${rowId}/`);
  if (!form) { inForm = false; return; }
  onDismiss(() => { inForm = false; });
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
  wireCantiereSelect(form);
  wire100Buttons(form);
  wireCancelButtons(form, dismissModal);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const isSaveAndAdd = e.submitter?.dataset.action === 'save-and-add';
    const body = Object.fromEntries(new FormData(form));

    const err = validateForm(body);
    if (err) { showFormError(form, err); return; }

    let data, status;
    try {
      ({ data, status } = await postJSON(SAVE_URL, body));
    } catch {
      showFormError(form, S.ERROR_NETWORK);
      return;
    }

    if (status === 200) {
      cache.updateRow(DATA_ID, data.row_id, data.record);
      if (data.item_record) {
        cache.updateRow(HARVEST_PLAN_ITEMS_ID, data.item_record[0], data.item_record);
      }
      dismissModal();
      if (isSaveAndAdd) showAddForm();
      else refreshTable();
      return;
    }

    if (data.status === STATUS_CONFLICT && data.record) {
      cache.updateRow(DATA_ID, data.row_id, data.record);
    }
    if (data.html) {
      const newForm = renderModalForm(data.html);
      if (newForm) {
        wireForm(newForm);
        showFormError(newForm, data.message || S.ERROR_GENERIC);
      }
    } else {
      showFormError(form, data.message || S.ERROR_GENERIC);
    }
  });
}

function renderFlagNote(opt) {
  const parts = [];
  if (opt.dataset.damaged === '1') parts.push(S.FLAG_DAMAGED);
  if (opt.dataset.unhealthy === '1') parts.push(S.FLAG_UNHEALTHY);
  if (opt.dataset.psr === '1') parts.push(S.FLAG_PSR);
  return parts.join(', ');
}

/** Wire Cantiere pulldown: toggle parcel group, filter parcels, show flags. */
function wireCantiereSelect(form) {
  const cantiereSel = form.querySelector('#id_cantiere');
  const parcelGroup = form.querySelector('#parcel-group');
  const parcelSel = form.querySelector('#id_parcel');
  const flagsDisplay = form.querySelector('#cantiere-flags-display');
  const flagsSpan = form.querySelector('#cantiere-flags');
  if (!cantiereSel) return;

  const allParcelOpts = parcelSel
    ? [...parcelSel.querySelectorAll('option')]
    : [];

  function update() {
    const opt = cantiereSel.selectedOptions[0];
    const hasValue = opt && opt.value;

    if (flagsDisplay && flagsSpan) {
      const note = hasValue ? renderFlagNote(opt) : '';
      flagsSpan.textContent = note || '—';
      flagsDisplay.hidden = !note;
    }

    if (parcelGroup && parcelSel) {
      if (!hasValue || opt.dataset.parcelId) {
        parcelGroup.hidden = true;
      } else {
        parcelGroup.hidden = false;
        const regionId = opt.dataset.regionId;
        const current = parcelSel.value;
        for (const o of allParcelOpts) o.remove();
        for (const o of allParcelOpts) {
          if (o.dataset.region === regionId) parcelSel.appendChild(o);
        }
        if (![...parcelSel.options].some(o => o.value === current)) {
          const xOpt = [...parcelSel.options].find(o => o.dataset.name === 'X');
          parcelSel.value = xOpt ? xOpt.value : '';
        }
      }
    }
  }

  cantiereSel.addEventListener('change', update);
  update();
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
      [ROW_ID]: String(rowId),
      [VERSION]: String(version),
      [FIELD_NONCE]: crypto.randomUUID(),
    });
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  if (resp.status === 200) {
    cache.removeRow(DATA_ID, rowId);
    if (resp.data.item_record) {
      cache.updateRow(HARVEST_PLAN_ITEMS_ID, resp.data.item_record[0], resp.data.item_record);
    }
    if (table) table.setData(cache.get(DATA_ID));
    return;
  }

  if (resp.data.status === STATUS_CONFLICT && resp.data.record) {
    cache.updateRow(DATA_ID, resp.data.row_id, resp.data.record);
    if (table) table.setData(cache.get(DATA_ID));
  }
  showError(resp.data.message || S.ERROR_GENERIC);
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

function refreshTable() {
  if (table) table.setData(cache.get(DATA_ID));
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
    if (name === ROW_ID) continue;
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
      formatter: fmtDecimal1BlankZero,
    };
  }
  return defs;
}

