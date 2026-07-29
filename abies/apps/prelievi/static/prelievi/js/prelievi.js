/**
 * Prelievi page: harvest operations table with year slider and CRUD forms.
 */

import * as cache from '../../base/js/cache.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  deleteRowWithVersion, fetchModalForm, renderModalForm, showFormError,
  submitCsvImport,
} from '../../base/js/forms.js';
import {
  showConfirmModal, wireActions, wireCancelButtons, wireCollapsibleToggle,
} from '../../base/js/ui-widgets.js';
import { canModify } from '../../base/js/roles.js';
import { fileToBase64, postJSON } from '../../base/js/api.js';
import { renderStackedBar, speciesNamesFromDigest } from '../../base/js/charts.js';
import { dismiss as dismissModal, onDismiss, show as showModal } from '../../base/js/modals.js';
import { columnMap } from '../../base/js/digests.js';
import { createRangeSlider } from '../../base/js/range-slider.js';
import * as router from '../../base/js/router.js';
import * as S from '../../base/js/strings.js';
import {
  COL_PARCEL_ID, COL_REGION_ID, FIELD_DATE, FIELD_ERRORS, FIELD_FILE,
  FIELD_NONCE, FIELD_SPECIES, FIELD_SPECIES_PCT_PREFIX,
  FIELD_TRACTOR_PCT_PREFIX, PARCEL_WHOLE_REGION_MARK, ROW_ID, STATUS_CONFLICT,
} from '../../base/js/constants.js';
import { CLASS_BOSCO_LINK, STATIC_COLS, buildPrelieviColumnDefs }
  from '../../base/js/prelievi-columns.js';
import { matchesSearch, searchTerms } from '../../base/js/table.js';
import {
  aggregateTimeSeries, aggregateParcelSeries,
} from './charts.js';
import { buildHarvestCalendar, calendarSearchText } from './calendar.js';
import { cloneTemplate } from '../../base/js/templates.js';
import {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableSort, writeTableState,
} from '../../base/js/page-sync.js';
import { localISODate } from '../../base/js/format.js';

const CSS_URL = '/static/prelievi/css/prelievi.css';
const DATA_ID = 'prelievi';
const DATA_URL = '/api/prelievi/data/';
const SPECIES_ID = FIELD_SPECIES;
const SPECIES_URL = '/api/species/data/';
const FORM_URL = '/api/prelievi/form/';
const SAVE_URL = '/api/prelievi/save/';
const DELETE_URL = '/api/prelievi/delete/';
const CSV_IMPORT_URL = '/api/prelievi/import-csv/';
const PAGE_PATH = '/prelievi';
const BOSCO_PATH = '/bosco';
const DEFAULT_TABLE_SORT = { column: S.COL_DATE, ascending: false };

// Collapsible sections, keyed by the single-char token used in the URL `o`
// parameter ('a' = Riassunto charts, 'b' = Calendario, 'i' = Interventi
// table).
const SECTION_KEYS = ['a', 'b', 'i'];
const DEFAULT_OPEN = 'i';                 // default when `o` param is absent

// Column indices — resolved on first data load.
let colDate = -1;
let colRegionId = -1;
let colParcelId = -1;
let filterRegionId = null;
let filterParcelId = null;

// Page state.
let table = null;
let slider = null;
let inForm = false;
let pendingQueryParams = null;
let pendingCacheRefresh = false;
let disposePageActions = null;

// Column classification and index map — resolved on first data load.
let speciesCols = [];
let speciesNames = [];
let tractorCols = [];
let colMap = {};

// Section state.  Chart sections carry their own open state, canvas,
// Chart.js instance, dirty flag, and render function.  The 'i' section
// just hosts the TableWrapper's container.
const sections = {
  a: {
    open: false, dirty: true,
    header: null, body: null,
    yearCanvas: null, yearInstance: null, yearBreakdown: 'total', byMonth: false,
    parcelCanvas: null, parcelInstance: null, parcelBreakdown: 'total',
    render: () => _renderSummarySection(),
  },
  b: {
    open: false, dirty: true, byMonth: false,
    header: null, body: null, host: null,
    render: () => _renderCalendarSection(),
  },
  i: {
    open: true,
    header: null, body: null,
  },
};

cache.register(DATA_ID, DATA_URL);
cache.register(SPECIES_ID, SPECIES_URL);


// ---------------------------------------------------------------------------
// Page lifecycle (exported for router)
// ---------------------------------------------------------------------------

const page = createPage({
  cssUrl: CSS_URL,
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: handleQueryChange,
  onUpdate: [[DATA_ID, onCacheUpdate], [SPECIES_ID, onSpeciesUpdate]],
  visibleIds: [DATA_ID, SPECIES_ID],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

async function loadPageData() {
  const [data, speciesData] = await Promise.all([
    cache.load(DATA_ID),
    cache.load(SPECIES_ID),
  ]);
  speciesNames = speciesNamesFromDigest(speciesData);
  return data;
}

function mountPage(el, params, data) {
  inForm = false;
  pendingQueryParams = null;
  pendingCacheRefresh = false;
  _syncColumnMetadata(data.columns);
  showTableView(data, params);
}

function destroyPage() {
  if (disposePageActions) { disposePageActions(); disposePageActions = null; }
  _destroyCharts();
  destroyTable();
}

// ---------------------------------------------------------------------------
// Table view
// ---------------------------------------------------------------------------

function showTableView(data, params) {
  inForm = false;
  pendingQueryParams = null;
  pendingCacheRefresh = false;
  _destroyCharts();
  const el = document.getElementById('content');
  el.replaceChildren();

  const p = readParams(params);
  buildPage(el, data, p);

  const modify = canModify();
  table = new TableWrapper({
    container: sections.i.body,
    digest: data,
    columnDefs: buildPrelieviColumnDefs(data.columns, speciesNames),
    inlineToolbar: false,
    canModify: modify,
    actions: modify ? {
      onAdd: () => showAddForm(),
      onEdit: (rowId) => showEditForm(rowId),
      onDelete: (rowId) => confirmDelete(rowId),
    } : {},
    sort: tableSort(p.table, DEFAULT_TABLE_SORT),
    searchText: p.table.searchText,
    csvFilename: S.CSV_PRELIEVI,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => { syncURL(); _updateCharts(); },
  });

  const searchInput = el.querySelector('#prelievi-search');
  if (searchInput) table.wireSearchInput(searchInput);
  sections.i.body?.addEventListener('click', onTableClick);

  filterRegionId = p.regionId;
  filterParcelId = p.parcelId;
  table.setExternalFilter(pageFilter());
  _updateCharts();
}

function destroyTable() {
  if (sections.i.body) sections.i.body.removeEventListener('click', onTableClick);
  if (table) { table.destroy(); table = null; }
  slider = null;
}

function onCacheUpdate() {
  if (inForm) {
    pendingCacheRefresh = true;
    return;
  }
  if (!table) return;
  refreshTable();
  _updateCharts();
}

function onSpeciesUpdate(data) {
  speciesNames = speciesNamesFromDigest(data);
  const current = cache.get(DATA_ID);
  if (current) refreshTable(current);
  _updateCharts();
}

function onTableClick(e) {
  const cell = e.target.closest(`.sortable-table-cell.${CLASS_BOSCO_LINK}`);
  if (!cell) return;
  const tr = cell.closest('.sortable-table-row');
  if (!tr || !table) return;
  const row = table.rowForElement(tr);
  const url = boscoUrlForHarvestRow(row, cache.get(DATA_ID)?.columns || []);
  if (!url) return;
  e.preventDefault();
  router.navigate(url);
}

export function boscoUrlForHarvestRow(row, columns) {
  if (!row || !columns) return null;
  const regionIdx = columns.indexOf(COL_REGION_ID);
  const parcelIdx = columns.indexOf(COL_PARCEL_ID);
  if (regionIdx < 0 || parcelIdx < 0) return null;
  const regionId = positiveInt(row[regionIdx]);
  const parcelId = positiveInt(row[parcelIdx]);
  if (regionId == null || parcelId == null) return null;
  const params = new URLSearchParams();
  params.set('c', regionId);
  params.set('v', '1');
  params.set('pa', parcelId);
  return `${BOSCO_PATH}?${params.toString()}`;
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
    table: readTableState(params),
    regionId: positiveInt(params.c),
    parcelId: positiveInt(params.pa),
    // Open sections: explicit string of single-char tokens when present,
    // falling back to DEFAULT_OPEN when absent.  `?o=` (empty) is valid
    // and means "all sections closed".
    o: params.o !== undefined ? params.o : DEFAULT_OPEN,
    b: params.b || 'total',
    pb: params.pb || 'total',
    m: params.m === '1',
    cm: params.cm === '1',
  };
}

function handleQueryChange(params) {
  if (inForm) {
    pendingQueryParams = params;
    return;
  }
  applyParams(params);
}

function finishForm() {
  inForm = false;
  if (pendingQueryParams) {
    const params = pendingQueryParams;
    pendingQueryParams = null;
    applyParams(params);
  }
  if (pendingCacheRefresh) {
    pendingCacheRefresh = false;
    refreshTable();
    _updateCharts();
  }
}

function applyParams(params) {
  const p = readParams(params);
  filterRegionId = p.regionId;
  filterParcelId = p.parcelId;

  if (applyTableState(table, p.table, DEFAULT_TABLE_SORT)) _updateCharts();

  // Year slider: bare URL means the full available range, not "keep the
  // previous in-memory range" when navigating back/forward.
  if (slider) {
    const data = cache.get(DATA_ID);
    if (data) {
      const years = extractYears(data.rows);
      const target = [p.y1 ?? years[0], p.y2 ?? years[years.length - 1]];
      const current = slider.getRange();
      if (current[0] !== target[0] || current[1] !== target[1]) {
        slider.setValues(target[0], target[1]);
      }
    }
  }

  // Summary chart configuration.
  const a = sections.a;
  if (a.body) {
    if (a.yearBreakdown !== p.b) {
      a.yearBreakdown = p.b;
      const sel = a.body.querySelector('[data-role="year-breakdown-select"]');
      if (sel) sel.value = p.b;
      a.dirty = true;
    }
    if (a.parcelBreakdown !== p.pb) {
      a.parcelBreakdown = p.pb;
      const sel = a.body.querySelector('[data-role="parcel-breakdown-select"]');
      if (sel) sel.value = p.pb;
      a.dirty = true;
    }
    if (a.byMonth !== p.m) {
      a.byMonth = p.m;
      const cb = a.body.querySelector('.chart-month-toggle input');
      if (cb) cb.checked = p.m;
      a.dirty = true;
    }
  }

  const bSection = sections.b;
  if (bSection.body && bSection.byMonth !== p.cm) {
    bSection.byMonth = p.cm;
    const cb = bSection.body.querySelector('[data-role="calendar-month-toggle"]');
    if (cb) cb.checked = p.cm;
    bSection.dirty = true;
  }

  if (table) table.setExternalFilter(pageFilter());
  _updateCharts();

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

  if (filterRegionId != null) params.set('c', filterRegionId);
  if (filterParcelId != null) params.set('pa', filterParcelId);
  writeTableState(params, table);

  // Open sections: only serialize if different from the default ('i').
  const openKeys = SECTION_KEYS.filter(k => sections[k].open).join('');
  if (openKeys !== DEFAULT_OPEN) params.set('o', openKeys);

  // Summary chart config: only serialize non-default values.
  if (sections.a.yearBreakdown !== 'total') params.set('b', sections.a.yearBreakdown);
  if (sections.a.parcelBreakdown !== 'total') params.set('pb', sections.a.parcelBreakdown);
  if (sections.a.byMonth) params.set('m', '1');
  if (sections.b.byMonth) params.set('cm', '1');

  navigateWithParams(PAGE_PATH, params);
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

function _buildColMap(columns) {
  colMap = columnMap(columns);
}

function _syncColumnMetadata(columns) {
  colDate = columns.indexOf(S.COL_DATE);
  colRegionId = columns.indexOf(COL_REGION_ID);
  colParcelId = columns.indexOf(COL_PARCEL_ID);
  _buildColMap(columns);
  _classifyColumns(columns);
}

function _classifyColumns(columns) {
  speciesCols = [];
  tractorCols = [];
  const speciesColNames = new Set(speciesNames);
  for (const name of columns) {
    if (name === ROW_ID || STATIC_COLS[name] || name.endsWith(' %')) continue;
    if (speciesColNames.has(name)) speciesCols.push(name);
    else tractorCols.push(name);
  }
}

function _getFilteredRows() {
  const data = cache.get(DATA_ID);
  if (!data) return [];
  const pf = pageFilter();
  const text = table ? table.getSearchText() : '';
  const terms = searchTerms(text);
  return data.rows.filter(row => {
    if (pf && !pf(row)) return false;
    if (terms.length && !matchesSearch(row, terms, table?.searchColumns)) return false;
    return true;
  });
}

function positiveInt(value) {
  if (value == null || value === '') return null;
  const n = parseInt(value, 10);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function scopeFilter() {
  const hasRegion = filterRegionId != null && colRegionId >= 0;
  const hasParcel = filterParcelId != null && colParcelId >= 0;
  if (!hasRegion && !hasParcel) return null;
  return row => (!hasRegion || row[colRegionId] === filterRegionId)
    && (!hasParcel || row[colParcelId] === filterParcelId);
}

function pageFilter() {
  const yf = yearFilter();
  const sf = scopeFilter();
  if (!yf && !sf) return null;
  return row => (!yf || yf(row)) && (!sf || sf(row));
}

function _updateCharts() {
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    if (!s.render) continue;
    s.dirty = true;
    if (s.open) s.render();
  }
}

function _renderSummarySection() {
  const s = sections.a;
  const rows = _getFilteredRows();
  if (s.yearCanvas) {
    s.yearInstance = renderStackedBar(
      s.yearCanvas,
      aggregateTimeSeries(
        rows, colMap, s.yearBreakdown, s.byMonth,
        speciesCols, tractorCols, speciesNames,
      ),
      s.yearInstance,
    );
  }
  if (s.parcelCanvas) {
    s.parcelInstance = renderStackedBar(
      s.parcelCanvas,
      aggregateParcelSeries(
        rows, colMap, s.parcelBreakdown, speciesCols, tractorCols, speciesNames,
      ),
      s.parcelInstance,
    );
  }
  s.dirty = false;
}

function _renderCalendarSection() {
  const s = sections.b;
  if (!s.host) return;
  s.host.replaceChildren();
  const calendar = buildHarvestCalendar(_getFilteredRows(), colMap, s.byMonth);
  if (!calendar.periods.length || !calendar.regions.length) {
    const empty = document.createElement('p');
    empty.className = 'prelievi-calendar-empty';
    empty.textContent = S.PRELIEVI_CALENDAR_EMPTY;
    s.host.appendChild(empty);
    s.dirty = false;
    return;
  }

  const tableEl = document.createElement('table');
  tableEl.className = 'prelievi-calendar-table';
  tableEl.appendChild(calendarHeader(calendar.periods));
  tableEl.appendChild(calendarBody(calendar));
  tableEl.addEventListener('click', onCalendarClick);
  s.host.appendChild(tableEl);
  s.dirty = false;
}

function calendarHeader(periods) {
  const thead = document.createElement('thead');
  const tr = document.createElement('tr');
  const corner = document.createElement('th');
  corner.className = 'corner';
  tr.appendChild(corner);
  for (const period of periods) {
    const th = document.createElement('th');
    th.textContent = period;
    tr.appendChild(th);
  }
  thead.appendChild(tr);
  return thead;
}

function calendarBody(calendar) {
  const tbody = document.createElement('tbody');
  for (const region of calendar.regions) {
    const regionRow = document.createElement('tr');
    regionRow.className = 'prelievi-calendar-region';
    const regionCell = document.createElement('td');
    regionCell.colSpan = calendar.periods.length + 1;
    regionCell.textContent = region.name;
    regionRow.appendChild(regionCell);
    tbody.appendChild(regionRow);

    for (const parcel of region.parcels) {
      const tr = document.createElement('tr');
      const label = document.createElement('td');
      label.className = 'prelievi-calendar-parcel';
      label.textContent = parcel.parcel;
      tr.appendChild(label);

      for (const period of calendar.periods) {
        const td = document.createElement('td');
        td.className = 'prelievi-calendar-cell';
        const count = parcel.cells.get(period) || 0;
        if (count > 0) {
          td.classList.add('active');
          td.dataset.period = period;
          td.dataset.region = region.name;
          td.dataset.parcel = parcel.parcel;
          td.title = `${region.name} ${parcel.parcel} — ${period}`;
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }
  return tbody;
}

function onCalendarClick(e) {
  const cell = e.target.closest('.prelievi-calendar-cell.active');
  if (!cell || !table) return;
  table.setSearchText(calendarSearchText(table.getSearchText(), cell.dataset));
  syncURL();
  _updateCharts();
}

function _destroyCharts() {
  const summary = sections.a;
  for (const key of ['yearInstance', 'parcelInstance']) {
    if (summary[key]) { summary[key].destroy(); summary[key] = null; }
  }
  summary.yearCanvas = null;
  summary.parcelCanvas = null;
  sections.b.host = null;
  for (const s of Object.values(sections)) {
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
      if (table) table.setExternalFilter(pageFilter());
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
        if (table) table.setExternalFilter(pageFilter());
      }
      if (table) table.setSearchText('');
      syncURL();
      _updateCharts();
    },
    'export-csv': () => table?.exportCSV(),
    'import-csv': () => showCsvImportModal(),
    add: () => showAddForm(),
  });

  // Wire collapsible sections.
  sections.a.yearBreakdown = p.b;
  sections.a.parcelBreakdown = p.pb;
  sections.a.byMonth = p.m;
  sections.b.byMonth = p.cm;
  for (const key of SECTION_KEYS) {
    const s = sections[key];
    s.open = p.o.includes(key);
    s.header = el.querySelector(`[data-section="${key}"].collapsible-header`);
    s.body = el.querySelector(`[data-section="${key}"].collapsible-body`);
    s.header?.classList.toggle('open', s.open);
    s.body?.classList.toggle('open', s.open);
    if (s.header && s.body) {
      wireCollapsibleToggle(s.header, s.body, (open) => {
        s.open = open;
        if (open && s.render && s.dirty) s.render();
        syncURL();
      });
    }
  }

  // Summary charts: wire breakdown selects and month toggle.
  const a = sections.a;
  a.yearCanvas = el.querySelector('[data-target="chart-a"]');
  a.parcelCanvas = el.querySelector('[data-target="chart-b"]');
  const yearBreakdownSel = el.querySelector('[data-role="year-breakdown-select"]');
  if (yearBreakdownSel) {
    yearBreakdownSel.value = a.yearBreakdown;
    yearBreakdownSel.addEventListener('change', () => {
      a.yearBreakdown = yearBreakdownSel.value;
      a.render();
      syncURL();
    });
  }
  const parcelBreakdownSel = el.querySelector('[data-role="parcel-breakdown-select"]');
  if (parcelBreakdownSel) {
    parcelBreakdownSel.value = a.parcelBreakdown;
    parcelBreakdownSel.addEventListener('change', () => {
      a.parcelBreakdown = parcelBreakdownSel.value;
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
  const calendarMonthCb = el.querySelector('[data-role="calendar-month-toggle"]');
  if (calendarMonthCb) {
    calendarMonthCb.checked = sections.b.byMonth;
    calendarMonthCb.addEventListener('change', () => {
      sections.b.byMonth = calendarMonthCb.checked;
      sections.b.render();
      syncURL();
    });
  }
  sections.b.host = el.querySelector('[data-target="calendar-host"]');
}

// ---------------------------------------------------------------------------
// CSV import
// ---------------------------------------------------------------------------

function showCsvImportModal() {
  const frag = cloneTemplate('tmpl-prelievi-import-csv-modal');
  wireCancelButtons(frag, dismissModal);

  const form = frag.querySelector('[data-role="import-form"]');
  const statusBox = frag.querySelector('.csv-import-status');
  const errorsBox = frag.querySelector('.csv-import-errors');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const result = await submitCsvImport({
      form,
      statusBox,
      errorsBox,
      attempt: () => importCsv(form),
    });
    if (result?.ok) {
      await cache.load(DATA_ID);
      refreshTable();
      _updateCharts();
    }
  });

  showModal(frag);
  document.querySelector('#modal-container [name="file"]')?.focus();
}

async function importCsv(form) {
  const file = form.querySelector(`[name="${FIELD_FILE}"]`)?.files?.[0];
  if (!file) return { error: S.ERR_CSV_FILE_REQUIRED };
  const { data, status } = await postJSON(CSV_IMPORT_URL, {
    [FIELD_FILE]: await fileToBase64(file),
    [FIELD_NONCE]: crypto.randomUUID(),
  });
  if (status === 200) return { ok: true };
  return data?.[FIELD_ERRORS]?.length
    ? { errors: data[FIELD_ERRORS] }
    : { error: data?.message };
}

// ---------------------------------------------------------------------------
// Add / Edit forms
// ---------------------------------------------------------------------------

async function showAddForm() {
  inForm = true;
  const form = await fetchModalForm(FORM_URL);
  if (!form) { finishForm(); return; }
  onDismiss(finishForm);
  wireForm(form);
}

async function showEditForm(rowId) {
  inForm = true;
  const form = await fetchModalForm(`${FORM_URL}${rowId}/`);
  if (!form) { finishForm(); return; }
  onDismiss(finishForm);
  wireForm(form);
}

/** Client-side validation before POST. Returns error message or null. */
function validateForm(body) {
  // Future date check.
  if (body[FIELD_DATE] && body[FIELD_DATE] > localISODate()) {
    return S.ERR_DATE_FUTURE;
  }
  // Species percentages must sum to 100.
  let spSum = 0;
  let trSum = 0;
  for (const [key, val] of Object.entries(body)) {
    const n = parseInt(val, 10) || 0;
    if (key.startsWith(FIELD_SPECIES_PCT_PREFIX)) spSum += n;
    else if (key.startsWith(FIELD_TRACTOR_PCT_PREFIX)) trSum += n;
  }
  if (spSum !== 100) return S.ERR_SPECIES_PCT_SUM;
  if (trSum !== 0 && trSum !== 100) return S.ERR_TRACTOR_PCT_SUM;
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
      cache.applyResponseChanges(data);
      dismissModal();
      if (isSaveAndAdd) showAddForm();
      else refreshTable();
      return;
    }

    if (data.status === STATUS_CONFLICT) {
      cache.applyResponseChanges(data);
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

export function filterParcelSelectForRegion(parcelSel, allParcelOpts, regionId) {
  const current = parcelSel.value;
  for (const o of allParcelOpts) o.remove();
  for (const o of allParcelOpts) {
    if (o.dataset.region === regionId) parcelSel.appendChild(o);
  }
  if ([...parcelSel.options].some(o => o.value === current)) {
    parcelSel.value = current;
  } else {
    const xOpt = [...parcelSel.options].find(
      o => o.dataset.name === PARCEL_WHOLE_REGION_MARK,
    );
    parcelSel.value = xOpt ? xOpt.value : '';
  }
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
        filterParcelSelectForRegion(parcelSel, allParcelOpts, regionId);
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
    const prefix = btn.dataset.prefix;
    const target = btn.dataset.target;
    for (const input of form.querySelectorAll(`input[name^="${prefix}"]`)) {
      input.value = input.name === target ? '100' : '0';
    }
  });
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

function confirmDelete(rowId) {
  showConfirmModal(S.DELETE_CONFIRM, () => deleteRowWithVersion(DATA_ID, rowId, DELETE_URL, {
    confirmMessage: null,
    onSuccess: (data) => {
      cache.applyResponseChanges(data);
      if (table) table.setData(cache.get(DATA_ID));
    },
    onConflict: () => { if (table) table.setData(cache.get(DATA_ID)); },
  }));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function refreshTable(data = cache.get(DATA_ID)) {
  if (!data) return;
  _syncColumnMetadata(data.columns);
  if (table) table.setData(data, buildPrelieviColumnDefs(data.columns, speciesNames));
}

// Prelievi column definitions live in base/js/prelievi-columns.js, shared
// with the Piano-di-taglio item view's sub-table so both format identically.
