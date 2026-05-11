/**
 * Campionamenti page.
 *
 * M3b: survey pulldown + sortable table of sampled trees.
 * M3c: Section 2 (Rilevamenti) Leaflet map with visited/unvisited
 *      coloring + click-to-narrow-table interaction.
 * M3d (pending): Section 1 (Griglie) + manual entry forms + imports.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import { showError } from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';
import { fetchJSON } from '../../base/js/api.js';
import { RilevamentiMap } from './rilevamenti-map.js';

const CSS_URL = '/static/campionamenti/css/campionamenti.css';
const SURVEYS_ID = 'campionamenti-surveys';
const SAMPLE_AREAS_ID = 'campionamenti-sample-areas';
const SAMPLES_ID = 'campionamenti-samples';
const SURVEYS_URL = '/api/campionamenti/surveys/data/';
const SAMPLE_AREAS_URL = '/api/campionamenti/sample-areas/data/';
const SAMPLES_URL = '/api/campionamenti/samples/data/';
const TREES_ID_PREFIX = 'campionamenti-trees-';
const TREES_URL_PREFIX = '/api/campionamenti/trees/';
const PARCELLE_GEOJSON_URL = '/api/geo/particelle.geojson';
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
let map = null;
let activeSurveyId = null;
let activeAreaId = null;
let surveysData = null;
let sampleAreasData = null;
let samplesData = null;
let parcelleGeo = null;
let pulldownEl = null;
let emptyEl = null;
let mapSectionEl = null;
let unsubCache = null;
let currentTreesId = null;

cache.register(SURVEYS_ID, SURVEYS_URL);
cache.register(SAMPLE_AREAS_ID, SAMPLE_AREAS_URL);
cache.register(SAMPLES_ID, SAMPLES_URL);

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
    const [s, sa, sm, geo] = await Promise.all([
      cache.load(SURVEYS_ID),
      cache.load(SAMPLE_AREAS_ID),
      cache.load(SAMPLES_ID),
      fetchJSON(PARCELLE_GEOJSON_URL),
    ]);
    surveysData = s;
    sampleAreasData = sa;
    samplesData = sm;
    parcelleGeo = geo.data;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  buildPageShell(el);
  cache.setVisible([SURVEYS_ID, SAMPLE_AREAS_ID, SAMPLES_ID]);

  // Drive the rest from URL params / defaults.
  applyParams(params);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (unsubCache) { unsubCache(); unsubCache = null; }
  destroyTable();
  destroyMap();
  cache.setVisible([]);
  pulldownEl = null;
  emptyEl = null;
  mapSectionEl = null;
  activeSurveyId = null;
  activeAreaId = null;
  surveysData = null;
  sampleAreasData = null;
  samplesData = null;
  parcelleGeo = null;
  currentTreesId = null;
}

export function onQueryChange(params) {
  applyParams(params);
}

// ---------------------------------------------------------------------------
// Page shell
// ---------------------------------------------------------------------------

function buildPageShell(el) {
  el.replaceChildren();

  // Top bar with survey pulldown.
  const bar = document.createElement('div');
  bar.className = 'campionamenti-top-bar';

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.htmlFor = 'campionamenti-survey-select';
  label.textContent = S.SURVEY_LABEL;

  const sel = document.createElement('select');
  sel.id = 'campionamenti-survey-select';
  sel.className = 'campionamenti-pulldown';

  const surveyIdCol = surveysData.columns.indexOf('row_id');
  const nameCol = surveysData.columns.indexOf('Nome');
  const visitedCol = surveysData.columns.indexOf('N. aree visitate');
  const totalCol = surveysData.columns.indexOf('N. aree totali');

  for (const row of surveysData.rows) {
    const opt = document.createElement('option');
    opt.value = String(row[surveyIdCol]);
    opt.textContent =
      `${row[nameCol]} (${row[visitedCol]}/${row[totalCol]} aree)`;
    sel.appendChild(opt);
  }

  sel.addEventListener('change', () => {
    activateSurvey(parseInt(sel.value, 10));
    syncURL();
  });

  bar.append(label, sel);
  el.appendChild(bar);

  // Section 2 (Rilevamenti map).  Created here; the map instance is
  // built lazily in activateSurvey when we know which areas to draw.
  mapSectionEl = document.createElement('div');
  mapSectionEl.className = 'campionamenti-section campionamenti-map-section';
  mapSectionEl.id = 'campionamenti-section-map';
  el.appendChild(mapSectionEl);

  // Section 3 (Alberi campionati table).
  const tableSection = document.createElement('div');
  tableSection.className = 'campionamenti-section';
  tableSection.id = 'campionamenti-section-trees';

  emptyEl = document.createElement('div');
  emptyEl.className = 'campionamenti-empty';
  emptyEl.textContent = S.CAMPIONAMENTI_EMPTY;
  tableSection.appendChild(emptyEl);

  el.appendChild(tableSection);
  pulldownEl = sel;
}

// ---------------------------------------------------------------------------
// Survey activation (drives map + table)
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
  activeAreaId = null;     // changing survey clears area selection

  // Map: render this survey's grid + visited overlay.
  renderMap(surveyId);

  // Table: lazy-fetch the per-survey trees digest.
  const dataId = `${TREES_ID_PREFIX}${surveyId}`;
  cache.register(dataId, `${TREES_URL_PREFIX}${surveyId}/`);
  currentTreesId = dataId;

  let data;
  try {
    data = await cache.load(dataId);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  renderTable(data);
  cache.setVisible([SURVEYS_ID, SAMPLE_AREAS_ID, SAMPLES_ID, dataId]);

  if (unsubCache) unsubCache();
  unsubCache = cache.onUpdate(dataId, () => {
    if (table) table.setData(cache.get(dataId));
  });
}

function showEmptyState() {
  destroyTable();
  destroyMap();
  if (emptyEl) emptyEl.hidden = false;
}

// ---------------------------------------------------------------------------
// Map rendering
// ---------------------------------------------------------------------------

function renderMap(surveyId) {
  destroyMap();
  if (!mapSectionEl) return;

  // Look up survey → sample_grid id.
  const surveyRow = surveysData.rows.find(
    r => r[surveysData.columns.indexOf('row_id')] === surveyId,
  );
  if (!surveyRow) return;
  const gridId = surveyRow[surveysData.columns.indexOf('Griglia')];

  // Build the list of sample areas in this grid.
  const c = sampleAreasData.columns;
  const areas = sampleAreasData.rows
    .filter(r => r[c.indexOf('Griglia')] === gridId)
    .map(r => ({
      id: r[c.indexOf('row_id')],
      lat: r[c.indexOf('Lat')],
      lng: r[c.indexOf('Lng')],
      compresa: r[c.indexOf('Compresa')],
      particella: r[c.indexOf('Particella')],
      numero: r[c.indexOf('Numero')],
    }));

  // Build visited lookup for this survey from samples.json.
  const sc = samplesData.columns;
  const visitedById = new Map();
  for (const r of samplesData.rows) {
    if (r[sc.indexOf('Survey')] !== surveyId) continue;
    visitedById.set(r[sc.indexOf('Sample area')], {
      nAlberi: r[sc.indexOf('N. alberi')],
    });
  }

  map = new RilevamentiMap({
    container: mapSectionEl,
    geojson: parcelleGeo,
    onAreaSelect: (areaId) => {
      activeAreaId = areaId;
      applyAreaFilter();
      syncURL();
    },
  });
  map.setAreas(areas, visitedById);
}

function destroyMap() {
  if (map) { map.destroy(); map = null; }
}

// ---------------------------------------------------------------------------
// Table rendering
// ---------------------------------------------------------------------------

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
    csvFilename: S.CSV_SAMPLED_TREES,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
  applyAreaFilter();
}

function destroyTable() {
  if (table) { table.destroy(); table = null; }
  const section = document.getElementById('campionamenti-section-trees');
  if (section) {
    section.querySelectorAll('.campionamenti-table-host').forEach(n => n.remove());
  }
}

/** Narrow Section 3 to only rows in `activeAreaId`, or all rows when null. */
function applyAreaFilter() {
  if (!table) return;
  if (activeAreaId == null) {
    table.setExternalFilter(null);
    return;
  }
  table.setExternalFilter((row) => {
    // `Sample area` is the second column after row_id/version per
    // sampled_trees_<id>.json shape.  We look it up by name to stay
    // robust to column reordering.
    return row[areaCol()] === activeAreaId;
  });
}

let _areaColIdx = -1;
function areaCol() {
  if (_areaColIdx >= 0) return _areaColIdx;
  const data = currentTreesId ? cache.get(currentTreesId) : null;
  if (!data) return -1;
  _areaColIdx = data.columns.indexOf('Sample area');
  return _areaColIdx;
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
    a: u.get('a') ? parseInt(u.get('a'), 10) : null,
    tsc: u.get('tsc') || null,
    tso: u.get('tso') !== '1',
    tf: u.get('tf') || '',
  };
}

function applyParams(params) {
  const p = {
    s: params.s ? parseInt(params.s, 10) : null,
    a: params.a ? parseInt(params.a, 10) : null,
  };

  let target = p.s;
  if (target == null && surveysData?.rows.length) {
    const idCol = surveysData.columns.indexOf('row_id');
    target = surveysData.rows[0][idCol];
  }

  if (target == null) {
    showEmptyState();
    return;
  }

  if (target !== activeSurveyId) {
    activateSurvey(target).then(() => {
      if (p.a != null) {
        activeAreaId = p.a;
        if (map) map.setActiveAreaId(p.a);
        applyAreaFilter();
      }
    });
  } else if (p.a !== activeAreaId) {
    activeAreaId = p.a;
    if (map) map.setActiveAreaId(p.a);
    applyAreaFilter();
  }
}

function syncURL() {
  const u = new URLSearchParams();
  if (activeSurveyId != null) u.set('s', String(activeSurveyId));
  if (activeAreaId != null) u.set('a', String(activeAreaId));
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
