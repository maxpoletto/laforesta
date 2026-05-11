/**
 * Campionamenti page.
 *
 * M3b: survey pulldown + sortable table of sampled trees.  Map
 * sections (1 and 2) and manual entry forms come in M3c/M3d.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import { showError } from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';

const CSS_URL = '/static/campionamenti/css/campionamenti.css';
const SURVEYS_ID = 'campionamenti-surveys';
const SURVEYS_URL = '/api/campionamenti/surveys/data/';
const TREES_ID_PREFIX = 'campionamenti-trees-';
const TREES_URL_PREFIX = '/api/campionamenti/trees/';
const PAGE_PATH = '/campionamenti';

// Number formatters
function formatDecimal2(value) {
  if (value == null || value === '') return '';
  return typeof value === 'number' ? value.toFixed(2).replace('.', ',') : value;
}
function formatDecimal1(value) {
  if (value == null || value === '') return '';
  return typeof value === 'number' ? value.toFixed(1).replace('.', ',') : value;
}
function formatInt(value) {
  if (value == null || value === '') return '';
  return String(value);
}
function formatBool(value) {
  return value ? '✓' : '';
}
function formatLatLng(value) {
  if (value == null || value === '') return '';
  return typeof value === 'number' ? value.toFixed(5) : value;
}

const TREES_COLS = {
  'Sample area': { hidden: true },
  'Data campione': { label: S.LABEL_DATE, type: 'date', width: '90px' },
  'Compresa': { label: S.COL_REGION, width: '90px' },
  'Particella': { label: S.COL_PARCEL, width: '80px' },
  'N. area': { label: 'N. area', width: '70px' },
  'N. albero': { label: 'N. albero', type: 'number', width: '70px', formatter: formatInt },
  'Specie': { label: S.LABEL_SPECIES, width: '120px' },
  'Tipo': { label: S.COL_PRODUCT, width: '70px' },
  'Pollone': { label: 'Pollone', type: 'number', width: '60px', formatter: formatInt },
  'Matricina': { label: 'Matricina', type: 'boolean', width: '70px', formatter: formatBool },
  'D (cm)': { label: 'D (cm)', type: 'number', width: '60px', formatter: formatInt },
  'h (m)': { label: 'h (m)', type: 'number', width: '60px', formatter: formatDecimal2 },
  'L10 (mm)': { label: 'L10 (mm)', type: 'number', width: '70px', formatter: formatInt },
  'V (m³)': { label: S.COL_VOLUME_M3, type: 'number', width: '70px', formatter: formatDecimal2 },
  'm (q)': { label: 'm (q)', type: 'number', width: '60px', formatter: formatDecimal1 },
  'PAI': { label: 'PAI', type: 'boolean', width: '50px', formatter: formatBool },
  'Lat': { label: 'Lat', type: 'number', width: '85px', formatter: formatLatLng },
  'Lng': { label: 'Lng', type: 'number', width: '85px', formatter: formatLatLng },
  'version': { label: 'version', hidden: true },
};

// Page state.
let table = null;
let activeSurveyId = null;
let surveysData = null;
let pulldownEl = null;
let emptyEl = null;
let unsubCache = null;

cache.register(SURVEYS_ID, SURVEYS_URL);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

export async function mount(params) {
  loadCSS(CSS_URL);
  const el = document.getElementById('content');
  el.replaceChildren();

  const loading = document.createElement('div');
  loading.className = 'loading-overlay';
  loading.textContent = S.LOADING;
  el.appendChild(loading);

  try {
    surveysData = await cache.load(SURVEYS_ID);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  buildPageShell(el, params);
  cache.setVisible([SURVEYS_ID]);

  // Drive the rest from URL params / defaults.
  applyParams(params);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (unsubCache) { unsubCache(); unsubCache = null; }
  destroyTable();
  cache.setVisible([]);
  pulldownEl = null;
  emptyEl = null;
  activeSurveyId = null;
  surveysData = null;
}

export function onQueryChange(params) {
  applyParams(params);
}

// ---------------------------------------------------------------------------
// Page shell
// ---------------------------------------------------------------------------

function buildPageShell(el, params) {
  el.replaceChildren();

  const bar = document.createElement('div');
  bar.className = 'campionamenti-top-bar';

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.htmlFor = 'campionamenti-survey-select';
  label.textContent = S.SURVEY_LABEL;

  const sel = document.createElement('select');
  sel.id = 'campionamenti-survey-select';
  sel.className = 'campionamenti-pulldown';

  // Populate options.  Surveys digest columns:
  // row_id, version, Nome, Descrizione, Griglia, Piano di taglio,
  // N. aree visitate, N. aree totali, Data primo, Data ultimo.
  const surveyIdCol = surveysData.columns.indexOf('row_id');
  const nameCol = surveysData.columns.indexOf('Nome');
  const visitedCol = surveysData.columns.indexOf('N. aree visitate');
  const totalCol = surveysData.columns.indexOf('N. aree totali');

  for (const row of surveysData.rows) {
    const opt = document.createElement('option');
    opt.value = String(row[surveyIdCol]);
    const visited = row[visitedCol];
    const total = row[totalCol];
    opt.textContent = `${row[nameCol]} (${visited}/${total} aree)`;
    sel.appendChild(opt);
  }

  sel.addEventListener('change', () => {
    activateSurvey(parseInt(sel.value, 10));
    syncURL();
  });

  bar.append(label, sel);
  el.appendChild(bar);

  const section = document.createElement('div');
  section.className = 'campionamenti-section';
  section.id = 'campionamenti-section-trees';

  emptyEl = document.createElement('div');
  emptyEl.className = 'campionamenti-empty';
  emptyEl.textContent = S.CAMPIONAMENTI_EMPTY;
  section.appendChild(emptyEl);

  el.appendChild(section);
  pulldownEl = sel;
}

// ---------------------------------------------------------------------------
// Survey activation
// ---------------------------------------------------------------------------

async function activateSurvey(surveyId) {
  if (surveyId == null || isNaN(surveyId)) {
    showEmptyState();
    activeSurveyId = null;
    return;
  }
  if (pulldownEl && pulldownEl.value !== String(surveyId)) {
    pulldownEl.value = String(surveyId);
  }
  activeSurveyId = surveyId;

  const dataId = `${TREES_ID_PREFIX}${surveyId}`;
  cache.register(dataId, `${TREES_URL_PREFIX}${surveyId}/`);

  let data;
  try {
    data = await cache.load(dataId);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  renderTable(data);
  cache.setVisible([SURVEYS_ID, dataId]);

  if (unsubCache) unsubCache();
  unsubCache = cache.onUpdate(dataId, () => {
    if (table) table.setData(cache.get(dataId));
  });
}

function showEmptyState() {
  destroyTable();
  if (emptyEl) emptyEl.hidden = false;
}

function renderTable(data) {
  destroyTable();
  if (emptyEl) emptyEl.hidden = true;

  const section = document.getElementById('campionamenti-section-trees');
  if (!section) return;

  const tableHost = document.createElement('div');
  tableHost.className = 'campionamenti-table-host';
  section.appendChild(tableHost);

  const params = currentURLParams();
  const sort = params.tsc
    ? { column: params.tsc, ascending: params.tso }
    : { column: 'Compresa', ascending: true };

  table = new TableWrapper({
    container: tableHost,
    digest: data,
    columnDefs: buildColumnDefs(data.columns),
    inlineToolbar: true,
    canModify: false,                       // M3d brings writer affordances
    sort,
    searchText: params.tf,
    csvFilename: 'alberi-campionati.csv',
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
}

function destroyTable() {
  if (table) { table.destroy(); table = null; }
  const section = document.getElementById('campionamenti-section-trees');
  if (section) {
    section.querySelectorAll('.campionamenti-table-host').forEach(n => n.remove());
  }
}

function buildColumnDefs(columns) {
  const defs = {};
  for (const c of columns) {
    defs[c] = TREES_COLS[c] || { label: c, width: '90px' };
  }
  return defs;
}

// ---------------------------------------------------------------------------
// URL params
// ---------------------------------------------------------------------------

function currentURLParams() {
  const u = new URLSearchParams(location.search);
  return {
    s: u.get('s') ? parseInt(u.get('s'), 10) : null,
    tsc: u.get('tsc') || null,
    tso: u.get('tso') !== '1',              // default ascending
    tf: u.get('tf') || '',
  };
}

function applyParams(params) {
  const surveyId = params.s ? parseInt(params.s, 10) : null;

  // Default survey selection.  M3b heuristic: most-recent active
  // (already sorted in surveys.json), else first available.
  let target = surveyId;
  if (target == null && surveysData?.rows.length) {
    const idCol = surveysData.columns.indexOf('row_id');
    target = surveysData.rows[0][idCol];
  }

  if (target == null) {
    showEmptyState();
    return;
  }

  if (target !== activeSurveyId) {
    activateSurvey(target);
  }
}

function syncURL() {
  const u = new URLSearchParams();
  if (activeSurveyId != null) u.set('s', String(activeSurveyId));
  if (table) {
    const sort = table.getSort();
    if (sort) {
      u.set('tsc', sort.column);
      u.set('tso', sort.ascending ? '0' : '1');
    }
    const f = table.getSearchText();
    if (f) u.set('tf', f);
  }
  const qs = u.toString();
  router.navigate(PAGE_PATH + (qs ? '?' + qs : ''), true);
}

// ---------------------------------------------------------------------------
// CSS load / unload
// ---------------------------------------------------------------------------

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
