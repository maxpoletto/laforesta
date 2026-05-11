/**
 * Campionamenti page — 3 collapsible sections.
 *
 *   Section 1 (g) — Griglie: grid pulldown + map of its sample areas.
 *   Section 2 (r) — Rilevamenti: survey pulldown + map with visited
 *                    coloring + click-to-narrow-Section-3.
 *   Section 3 (t) — Alberi campionati: sortable table of sampled trees.
 *
 * URL params: see docs/pages/campionamenti.md (o, g, s, a, tf, tsc, tso).
 *
 * M3d-read: read-only view; writer affordances (pencil/garbage/+),
 * manual entry forms, and CSV imports ship in M3d-write.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import { showError } from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';
import { fetchJSON } from '../../base/js/api.js';
import { fetchForm, renderFormHTML, interceptSubmit } from '../../base/js/forms.js';
import { RilevamentiMap } from './rilevamenti-map.js';
import { GriglieMap } from './griglie-map.js';
import { tabacchiVolumeM3, massQ } from '../../base/js/volume.js';

const CSS_URL = '/static/campionamenti/css/campionamenti.css';
const SURVEYS_ID = 'campionamenti-surveys';
const GRIDS_ID = 'campionamenti-grids';
const SAMPLE_AREAS_ID = 'campionamenti-sample-areas';
const SAMPLES_ID = 'campionamenti-samples';
const TREES_ID_PREFIX = 'campionamenti-trees-';

const SURVEYS_URL = '/api/campionamenti/surveys/data/';
const GRIDS_URL = '/api/campionamenti/grids/data/';
const SAMPLE_AREAS_URL = '/api/campionamenti/sample-areas/data/';
const SAMPLES_URL = '/api/campionamenti/samples/data/';
const TREES_URL_PREFIX = '/api/campionamenti/trees/';
const TREE_FORM_URL = '/api/campionamenti/tree/form/';
const TREE_SAVE_URL = '/api/campionamenti/tree/save/';
const PARCELLE_GEOJSON_URL = '/api/geo/particelle.geojson';
const PAGE_PATH = '/campionamenti';

const SECTION_KEYS = ['g', 'r', 't'];
const DEFAULT_OPEN = 'r';

// --- Formatters -------------------------------------------------------------
function f2(v) { return typeof v === 'number' ? v.toFixed(2).replace('.', ',') : (v == null ? '' : v); }
function f1(v) { return typeof v === 'number' ? v.toFixed(1).replace('.', ',') : (v == null ? '' : v); }
function fInt(v) { return v == null || v === '' ? '' : String(v); }
function fBool(v) { return v ? '✓' : ''; }
function fLat(v) { return typeof v === 'number' ? v.toFixed(5) : (v == null ? '' : v); }

const TREES_COLS = {
  'Sample area': { hidden: true },
  'Data campione': { label: S.LABEL_DATE, type: 'date', width: '90px' },
  'Compresa': { label: S.COL_REGION, width: '90px' },
  'Particella': { label: S.COL_PARCEL, width: '80px' },
  'N. area': { label: 'N. area', width: '70px' },
  'N. albero': { label: 'N. albero', type: 'number', width: '70px', formatter: fInt },
  'Specie': { label: S.LABEL_SPECIES, width: '120px' },
  'Tipo': { label: S.COL_PRODUCT, width: '70px' },
  'Pollone': { label: 'Pollone', type: 'number', width: '60px', formatter: fInt },
  'Matricina': { label: 'Matricina', type: 'boolean', width: '70px', formatter: fBool },
  'D (cm)': { label: 'D (cm)', type: 'number', width: '60px', formatter: fInt },
  'h (m)': { label: 'h (m)', type: 'number', width: '60px', formatter: f2 },
  'L10 (mm)': { label: 'L10 (mm)', type: 'number', width: '70px', formatter: fInt },
  'V (m³)': { label: S.COL_VOLUME_M3, type: 'number', width: '70px', formatter: f2 },
  'm (q)': { label: 'm (q)', type: 'number', width: '60px', formatter: f1 },
  'PAI': { label: 'PAI', type: 'boolean', width: '50px', formatter: fBool },
  'Lat': { label: 'Lat', type: 'number', width: '85px', formatter: fLat },
  'Lng': { label: 'Lng', type: 'number', width: '85px', formatter: fLat },
  'version': { label: 'version', hidden: true },
};

// --- Page state -------------------------------------------------------------
const sections = {
  g: { title: S.SECTION_GRIGLIE,   open: false, header: null, body: null,
       pulldown: null, summary: null, mapEl: null, map: null },
  r: { title: S.SECTION_RILEVAMENTI, open: true, header: null, body: null,
       pulldown: null, summary: null, mapEl: null, map: null },
  t: { title: S.SECTION_ALBERI_CAMPIONATI, open: false, header: null, body: null,
       host: null, emptyEl: null },
};

let table = null;
let activeGridId = null;
let activeSurveyId = null;
let activeAreaId = null;
let surveysData = null;
let gridsData = null;
let sampleAreasData = null;
let samplesData = null;
let parcelleGeo = null;
let unsubCache = null;
let currentTreesId = null;
let _areaColIdx = -1;
let inForm = false;
let escapeHandler = null;

function canModify() {
  return ['admin', 'writer'].includes(document.body.dataset.role);
}

cache.register(SURVEYS_ID, SURVEYS_URL);
cache.register(GRIDS_ID, GRIDS_URL);
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
    const [s, g, sa, sm, geo] = await Promise.all([
      cache.load(SURVEYS_ID),
      cache.load(GRIDS_ID),
      cache.load(SAMPLE_AREAS_ID),
      cache.load(SAMPLES_ID),
      fetchJSON(PARCELLE_GEOJSON_URL),
    ]);
    surveysData = s;
    gridsData = g;
    sampleAreasData = sa;
    samplesData = sm;
    parcelleGeo = geo.data;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  buildPageShell(el, params);
  cache.setVisible([SURVEYS_ID, GRIDS_ID, SAMPLE_AREAS_ID, SAMPLES_ID]);

  applyParams(params);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (unsubCache) { unsubCache(); unsubCache = null; }
  destroyTable();
  destroyRilevamentiMap();
  destroyGriglieMap();
  cache.setVisible([]);
  resetSectionRefs();
  activeGridId = activeSurveyId = activeAreaId = null;
  surveysData = gridsData = sampleAreasData = samplesData = parcelleGeo = null;
  currentTreesId = null;
  _areaColIdx = -1;
}

export function onQueryChange(params) {
  if (inForm) return;     // suppress URL-driven rebuilds while editing
  applyParams(params);
}

function resetSectionRefs() {
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    s.header = s.body = null;
    s.pulldown = s.summary = s.mapEl = s.map = null;
    s.host = s.emptyEl = null;
  }
}

// ---------------------------------------------------------------------------
// Page shell — three collapsible sections
// ---------------------------------------------------------------------------

function buildPageShell(el, params) {
  el.replaceChildren();
  const p = readParams(params);

  // Section initial open state from URL `o=`.
  for (const k of SECTION_KEYS) sections[k].open = p.o.includes(k);

  // Section 1: Griglie
  buildSection(el, sections.g, body => buildGriglieBody(body));
  // Section 2: Rilevamenti
  buildSection(el, sections.r, body => buildRilevamentiBody(body));
  // Section 3: Alberi campionati
  buildSection(el, sections.t, body => buildAlberiBody(body));
}

function buildSection(el, s, populate) {
  const [header, body] = collapsible(s.title, s.open);
  s.header = header;
  s.body = body;
  populate(body);
  header.addEventListener('click', () => {
    s.open = !s.open;
    header.classList.toggle('open', s.open);
    body.classList.toggle('open', s.open);
    if (s.open) onSectionOpen(s);
    syncURL();
  });
  el.append(header, body);
}

function collapsible(title, open) {
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

function onSectionOpen(s) {
  // Leaflet needs invalidateSize() when a hidden container becomes visible.
  if (s === sections.g && s.map) s.map.invalidateSize();
  if (s === sections.r && s.map) s.map.invalidateSize();
}

// ---------------------------------------------------------------------------
// Section 1 — Griglie body
// ---------------------------------------------------------------------------

function buildGriglieBody(body) {
  const s = sections.g;

  const topRow = document.createElement('div');
  topRow.className = 'campionamenti-section-top';

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.textContent = S.GRID_LABEL;

  const sel = document.createElement('select');
  sel.className = 'campionamenti-pulldown';
  const idCol = gridsData.columns.indexOf('row_id');
  const nameCol = gridsData.columns.indexOf('Nome');
  for (const row of gridsData.rows) {
    const opt = document.createElement('option');
    opt.value = String(row[idCol]);
    opt.textContent = row[nameCol];
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => {
    activateGrid(parseInt(sel.value, 10));
    syncURL();
  });
  label.htmlFor = sel.id = 'campionamenti-grid-select';

  topRow.append(label, sel);
  body.appendChild(topRow);

  s.summary = document.createElement('div');
  s.summary.className = 'campionamenti-summary';
  body.appendChild(s.summary);

  s.mapEl = document.createElement('div');
  s.mapEl.className = 'campionamenti-map-host';
  body.appendChild(s.mapEl);

  s.pulldown = sel;
}

function activateGrid(gridId) {
  const s = sections.g;
  if (gridId == null || isNaN(gridId)) return;
  if (s.pulldown && s.pulldown.value !== String(gridId)) {
    s.pulldown.value = String(gridId);
  }
  activeGridId = gridId;
  renderGriglieSummary(gridId);
  renderGriglieMap(gridId);
}

function renderGriglieSummary(gridId) {
  const s = sections.g;
  if (!s.summary) return;
  const c = gridsData.columns;
  const row = gridsData.rows.find(r => r[c.indexOf('row_id')] === gridId);
  s.summary.replaceChildren();
  if (!row) return;
  const desc = row[c.indexOf('Descrizione')] || '';

  const stats = document.createElement('div');
  stats.textContent =
    `${row[c.indexOf('N. aree')]} aree · ` +
    `${row[c.indexOf('Comprese')]} · ` +
    `${row[c.indexOf('N. rilevamenti')]} rilevamenti · ` +
    `aggiornata ${formatTimestamp(row[c.indexOf('Ultimo aggiornamento')])}`;
  s.summary.appendChild(stats);
  if (desc) {
    const d = document.createElement('div');
    d.className = 'campionamenti-summary-desc';
    d.textContent = desc;
    s.summary.appendChild(d);
  }
}

function renderGriglieMap(gridId) {
  destroyGriglieMap();
  const s = sections.g;
  if (!s.mapEl) return;

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

  s.map = new GriglieMap({
    container: s.mapEl,
    geojson: parcelleGeo,
    onAreaClick: null,                      // popover in M3d-write
  });
  s.map.setAreas(areas);
}

function destroyGriglieMap() {
  if (sections.g.map) { sections.g.map.destroy(); sections.g.map = null; }
}

// ---------------------------------------------------------------------------
// Section 2 — Rilevamenti body
// ---------------------------------------------------------------------------

function buildRilevamentiBody(body) {
  const s = sections.r;

  const topRow = document.createElement('div');
  topRow.className = 'campionamenti-section-top';

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.textContent = S.SURVEY_LABEL;

  const sel = document.createElement('select');
  sel.className = 'campionamenti-pulldown';
  const idCol = surveysData.columns.indexOf('row_id');
  const nameCol = surveysData.columns.indexOf('Nome');
  const visCol = surveysData.columns.indexOf('N. aree visitate');
  const totCol = surveysData.columns.indexOf('N. aree totali');
  for (const row of surveysData.rows) {
    const opt = document.createElement('option');
    opt.value = String(row[idCol]);
    opt.textContent =
      `${row[nameCol]} (${row[visCol]}/${row[totCol]} aree)`;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => {
    activateSurvey(parseInt(sel.value, 10));
    syncURL();
  });
  label.htmlFor = sel.id = 'campionamenti-survey-select';

  topRow.append(label, sel);
  body.appendChild(topRow);

  s.summary = document.createElement('div');
  s.summary.className = 'campionamenti-summary';
  body.appendChild(s.summary);

  s.mapEl = document.createElement('div');
  s.mapEl.className = 'campionamenti-map-host';
  body.appendChild(s.mapEl);

  s.pulldown = sel;
}

async function activateSurvey(surveyId) {
  const s = sections.r;
  if (surveyId == null || isNaN(surveyId)) { showAlberiEmpty(); return; }
  if (s.pulldown && s.pulldown.value !== String(surveyId)) {
    s.pulldown.value = String(surveyId);
  }
  activeSurveyId = surveyId;
  activeAreaId = null;

  renderRilevamentiSummary(surveyId);
  renderRilevamentiMap(surveyId);

  // Section 3: lazy-fetch per-survey trees.
  const dataId = `${TREES_ID_PREFIX}${surveyId}`;
  cache.register(dataId, `${TREES_URL_PREFIX}${surveyId}/`);
  currentTreesId = dataId;
  _areaColIdx = -1;

  let data;
  try { data = await cache.load(dataId); }
  catch { showError(S.ERROR_NETWORK); return; }

  renderTable(data);
  cache.setVisible([SURVEYS_ID, GRIDS_ID, SAMPLE_AREAS_ID, SAMPLES_ID, dataId]);

  if (unsubCache) unsubCache();
  unsubCache = cache.onUpdate(dataId, () => {
    if (table) table.setData(cache.get(dataId));
  });
}

function renderRilevamentiSummary(surveyId) {
  const s = sections.r;
  if (!s.summary) return;
  const c = surveysData.columns;
  const row = surveysData.rows.find(r => r[c.indexOf('row_id')] === surveyId);
  s.summary.replaceChildren();
  if (!row) return;
  const desc = row[c.indexOf('Descrizione')] || '';
  const gridId = row[c.indexOf('Griglia')];
  const gridName = lookupGridName(gridId);

  const stats = document.createElement('div');
  stats.textContent =
    `Griglia: ${gridName} · ` +
    `${row[c.indexOf('N. aree visitate')]}/${row[c.indexOf('N. aree totali')]} aree visitate · ` +
    (row[c.indexOf('Data primo')]
      ? `dal ${row[c.indexOf('Data primo')]} al ${row[c.indexOf('Data ultimo')]}`
      : 'nessun campione');
  s.summary.appendChild(stats);
  if (desc) {
    const d = document.createElement('div');
    d.className = 'campionamenti-summary-desc';
    d.textContent = desc;
    s.summary.appendChild(d);
  }
}

function lookupGridName(gridId) {
  const c = gridsData.columns;
  const row = gridsData.rows.find(r => r[c.indexOf('row_id')] === gridId);
  return row ? row[c.indexOf('Nome')] : '';
}

function renderRilevamentiMap(surveyId) {
  destroyRilevamentiMap();
  const s = sections.r;
  if (!s.mapEl) return;

  const surveyRow = surveysData.rows.find(
    r => r[surveysData.columns.indexOf('row_id')] === surveyId,
  );
  if (!surveyRow) return;
  const gridId = surveyRow[surveysData.columns.indexOf('Griglia')];

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

  const sc = samplesData.columns;
  const visitedById = new Map();
  for (const r of samplesData.rows) {
    if (r[sc.indexOf('Survey')] !== surveyId) continue;
    visitedById.set(r[sc.indexOf('Sample area')], {
      nAlberi: r[sc.indexOf('N. alberi')],
    });
  }

  s.map = new RilevamentiMap({
    container: s.mapEl,
    geojson: parcelleGeo,
    onAreaSelect: (areaId) => {
      activeAreaId = areaId;
      applyAreaFilter();
      syncURL();
      // Auto-expand Section 3 when the user picks an area.
      if (areaId != null && !sections.t.open) {
        toggleSection(sections.t, true);
      }
    },
  });
  s.map.setAreas(areas, visitedById);
}

function destroyRilevamentiMap() {
  if (sections.r.map) { sections.r.map.destroy(); sections.r.map = null; }
}

// ---------------------------------------------------------------------------
// Section 3 — Alberi campionati body
// ---------------------------------------------------------------------------

function buildAlberiBody(body) {
  const s = sections.t;
  s.emptyEl = document.createElement('div');
  s.emptyEl.className = 'campionamenti-empty';
  s.emptyEl.textContent = S.CAMPIONAMENTI_EMPTY;
  body.appendChild(s.emptyEl);

  s.host = document.createElement('div');
  s.host.className = 'campionamenti-table-host';
  body.appendChild(s.host);
}

function showAlberiEmpty() {
  destroyTable();
  if (sections.t.emptyEl) sections.t.emptyEl.hidden = false;
}

function renderTable(data) {
  destroyTable();
  const s = sections.t;
  if (!s.host) return;
  if (s.emptyEl) s.emptyEl.hidden = true;

  const params = currentURLParams();
  const sort = params.tsc
    ? { column: params.tsc, ascending: params.tso }
    : { column: 'Compresa', ascending: true };

  const modify = canModify();
  table = new TableWrapper({
    container: s.host,
    digest: data,
    columnDefs: buildColumnDefs(data.columns),
    inlineToolbar: true,
    canModify: modify,
    actions: modify ? {
      onAdd: () => showAddTreeForm(),
      // onEdit / onDelete to follow in a later M3d-write iteration.
    } : {},
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
}

function applyAreaFilter() {
  if (!table) return;
  if (activeAreaId == null) {
    table.setExternalFilter(null);
    return;
  }
  table.setExternalFilter((row) => row[areaCol()] === activeAreaId);
}

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

function toggleSection(s, open) {
  if (!s.header || !s.body) return;
  s.open = open;
  s.header.classList.toggle('open', open);
  s.body.classList.toggle('open', open);
  if (open) onSectionOpen(s);
}

// ---------------------------------------------------------------------------
// URL params
// ---------------------------------------------------------------------------

function readParams(params) {
  return {
    o: params.o !== undefined ? params.o : DEFAULT_OPEN,
    g: params.g ? parseInt(params.g, 10) : null,
    s: params.s ? parseInt(params.s, 10) : null,
    a: params.a ? parseInt(params.a, 10) : null,
    tsc: params.tsc || null,
    tso: params.tso !== '1',
    tf: params.tf || '',
  };
}

function currentURLParams() {
  const u = new URLSearchParams(location.search);
  const o = {};
  for (const k of ['o', 'g', 's', 'a', 'tsc', 'tso', 'tf']) o[k] = u.get(k);
  return {
    o: o.o !== null ? o.o : DEFAULT_OPEN,
    g: o.g ? parseInt(o.g, 10) : null,
    s: o.s ? parseInt(o.s, 10) : null,
    a: o.a ? parseInt(o.a, 10) : null,
    tsc: o.tsc || null,
    tso: o.tso !== '1',
    tf: o.tf || '',
  };
}

function applyParams(params) {
  const p = readParams(params);

  // Sync section open/closed.
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    const shouldBeOpen = p.o.includes(k);
    if (s.header && s.open !== shouldBeOpen) toggleSection(s, shouldBeOpen);
  }

  // Active grid — default: first grid in pulldown order.
  let targetGrid = p.g;
  if (targetGrid == null && gridsData?.rows.length) {
    targetGrid = gridsData.rows[0][gridsData.columns.indexOf('row_id')];
  }
  if (targetGrid != null && targetGrid !== activeGridId) {
    activateGrid(targetGrid);
  }

  // Active survey — default: first row of surveys digest.
  let targetSurvey = p.s;
  if (targetSurvey == null && surveysData?.rows.length) {
    targetSurvey = surveysData.rows[0][surveysData.columns.indexOf('row_id')];
  }
  if (targetSurvey == null) { showAlberiEmpty(); return; }

  if (targetSurvey !== activeSurveyId) {
    activateSurvey(targetSurvey).then(() => {
      if (p.a != null) {
        activeAreaId = p.a;
        if (sections.r.map) sections.r.map.setActiveAreaId(p.a);
        applyAreaFilter();
      }
    });
  } else if (p.a !== activeAreaId) {
    activeAreaId = p.a;
    if (sections.r.map) sections.r.map.setActiveAreaId(p.a);
    applyAreaFilter();
  }
}

function syncURL() {
  const u = new URLSearchParams();

  const openKeys = SECTION_KEYS.filter(k => sections[k].open).join('');
  if (openKeys !== DEFAULT_OPEN) u.set('o', openKeys);

  if (activeGridId != null) {
    const defaultGrid = gridsData?.rows[0]?.[gridsData.columns.indexOf('row_id')];
    if (activeGridId !== defaultGrid) u.set('g', String(activeGridId));
  }
  if (activeSurveyId != null) {
    const defaultSurvey = surveysData?.rows[0]?.[surveysData.columns.indexOf('row_id')];
    if (activeSurveyId !== defaultSurvey) u.set('s', String(activeSurveyId));
  }
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
// Manual tree+sample entry form (M3d-write)
// ---------------------------------------------------------------------------

async function showAddTreeForm() {
  if (activeSurveyId == null) {
    showError(S.CAMPIONAMENTI_PICK_SURVEY_FIRST);
    return;
  }
  if (activeAreaId == null) {
    showError(S.CAMPIONAMENTI_PICK_AREA_FIRST);
    return;
  }
  inForm = true;
  const url = `${TREE_FORM_URL}?survey=${activeSurveyId}&area=${activeAreaId}`;
  const form = await fetchForm(url);
  if (!form) { returnToPage(); return; }
  wireTreeForm(form);
  addEscapeHandler();
}

function wireTreeForm(form) {
  wireVMPreview(form);
  form.querySelector('#tree-form-cancel')?.addEventListener('click', () => {
    returnToPage();
  });
  interceptSubmit(form, TREE_SAVE_URL, {
    onSuccess: async (data, isSaveAndAdd) => {
      // Server marked sampled_trees_<survey>.json stale; reload it.
      if (currentTreesId) {
        try { await cache.load(currentTreesId); } catch {}
      }
      if (isSaveAndAdd) {
        showAddTreeForm();
      } else {
        returnToPage();
      }
    },
    onConflict(data) {
      if (data.html) {
        const f = renderFormHTML(data.html);
        if (f) wireTreeForm(f);
      }
    },
    onValidationError(data) {
      if (data.html) {
        const f = renderFormHTML(data.html);
        if (f) wireTreeForm(f);
      }
    },
  });
}

/** Wire the live V/m preview line under the D/h/L10 fields. */
function wireVMPreview(form) {
  const d = form.querySelector('#id_d_cm');
  const h = form.querySelector('#id_h_m');
  const sp = form.querySelector('#id_species');
  const fustaiaCb = form.querySelector('input[type="checkbox"][name="fustaia"]');
  const preview = form.querySelector('#tree-form-vm-preview');
  const vHidden = form.querySelector('#tree-form-volume-m3');
  const mHidden = form.querySelector('#tree-form-mass-q');
  if (!d || !h || !sp || !preview || !vHidden || !mHidden) return;

  function update() {
    if (!fustaiaCb?.checked) {
      preview.textContent = S.CAMPIONAMENTI_NO_VM_FOR_CEDUO;
      vHidden.value = '';
      mHidden.value = '';
      return;
    }
    const dCm = parseFloat(d.value);
    const hM = parseFloat(h.value);
    const opt = sp.options[sp.selectedIndex];
    const speciesName = opt?.dataset.name;
    const density = parseFloat(opt?.dataset.density);
    const v = tabacchiVolumeM3(dCm, hM, speciesName);
    if (v == null) {
      preview.textContent = S.CAMPIONAMENTI_VM_INCOMPLETE;
      vHidden.value = '';
      mHidden.value = '';
      return;
    }
    const m = massQ(v, density);
    preview.textContent =
      `V = ${v.toFixed(3).replace('.', ',')} m³ · m = ${m.toFixed(2).replace('.', ',')} q`;
    vHidden.value = v.toFixed(4);
    mHidden.value = m.toFixed(3);
  }

  d.addEventListener('input', update);
  h.addEventListener('input', update);
  sp.addEventListener('change', update);
  fustaiaCb?.addEventListener('change', update);
  update();
}

function returnToPage() {
  inForm = false;
  removeEscapeHandler();
  // Re-render the whole page shell from scratch.  Section open state
  // and active grid/survey/area come back from URL params.
  const params = Object.fromEntries(new URLSearchParams(location.search));
  buildPageShell(document.getElementById('content'), params);
  applyParams(params);
}

function addEscapeHandler() {
  removeEscapeHandler();
  escapeHandler = (e) => {
    if (e.key === 'Escape') returnToPage();
  };
  document.addEventListener('keydown', escapeHandler);
}

function removeEscapeHandler() {
  if (escapeHandler) {
    document.removeEventListener('keydown', escapeHandler);
    escapeHandler = null;
  }
}

// --- Misc -------------------------------------------------------------------

function formatTimestamp(iso) {
  if (!iso) return '—';
  return iso.length >= 10 ? iso.slice(0, 10) : iso;
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
