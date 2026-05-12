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
import { show as showModal, showError, dismiss as dismissModal } from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';
import { fetchJSON, postJSON, postFormData } from '../../base/js/api.js';
import { fetchForm, renderFormHTML, interceptSubmit } from '../../base/js/forms.js';
import { RilevamentiMap } from './rilevamenti-map.js';
import { GriglieMap } from './griglie-map.js';
import { GridPlanner } from './grid-planner.js';
import { mountUseLocationButton } from '../../base/js/latlng-input.js';
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
const TREE_DELETE_URL_PREFIX = '/api/campionamenti/tree/delete/';
const SAMPLE_DATE_SAVE_URL = '/api/campionamenti/sample/date/';
const AREA_FORM_URL = '/api/campionamenti/area/form/';
const AREA_SAVE_URL = '/api/campionamenti/area/save/';
const AREA_DELETE_URL_PREFIX = '/api/campionamenti/area/delete/';
const GRID_EDIT_URL_PREFIX = '/api/campionamenti/grid/edit/';
const GRID_DELETE_URL_PREFIX = '/api/campionamenti/grid/delete/';
const SURVEY_EDIT_URL_PREFIX = '/api/campionamenti/survey/edit/';
const SURVEY_DELETE_URL_PREFIX = '/api/campionamenti/survey/delete/';
const GRID_CSV_IMPORT_URL = '/api/campionamenti/grid/import-csv/';
const TREE_CSV_IMPORT_URL = '/api/campionamenti/survey/import-csv/';
const GRID_FORM_URL = '/api/campionamenti/grid/form/';
const GRID_SAVE_URL = '/api/campionamenti/grid/save/';
const SURVEY_FORM_URL = '/api/campionamenti/survey/form/';
const SURVEY_SAVE_URL = '/api/campionamenti/survey/save/';
const PARCELLE_GEOJSON_URL = '/api/geo/particelle.geojson';
const PAGE_PATH = '/campionamenti';

const SECTION_KEYS = ['g', 'r', 't'];
const DEFAULT_OPEN = 'r';

// mt= URL param → MapCommon basemap key.  Defaults to 's' (Satellite)
// per campionamenti.md §"URL parameters".
const MAP_TYPE_TOKENS = { o: 'osm', t: 'topo', s: 'satellite' };
const DEFAULT_MAP_TYPE = 's';

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
       host: null, emptyEl: null, headerEl: null },
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
    s.host = s.emptyEl = s.headerEl = null;
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

  // Empty state — no grids exist yet.  Show a centered prompt + the
  // "Nuova griglia" button (writers only).
  if (!gridsData?.rows.length) {
    const empty = document.createElement('div');
    empty.className = 'campionamenti-empty';
    empty.textContent = S.CAMPIONAMENTI_NO_GRIDS;
    body.appendChild(empty);
    if (canModify()) {
      const addBtn = document.createElement('button');
      addBtn.type = 'button';
      addBtn.className = 'btn btn-primary';
      addBtn.textContent = S.NEW_GRID_LABEL;
      addBtn.addEventListener('click', () => showNewGridForm());
      const wrap = document.createElement('div');
      wrap.style.textAlign = 'center';
      wrap.appendChild(addBtn);
      body.appendChild(wrap);
    }
    return;
  }

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

  if (canModify()) {
    appendEditDeleteIcons(topRow, {
      onEdit: () => showRenameGridForm(),
      onDelete: () => confirmDeleteGrid(),
    });
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-primary campionamenti-add-btn';
    addBtn.textContent = S.NEW_GRID_LABEL;
    addBtn.addEventListener('click', () => showNewGridForm());
    topRow.appendChild(addBtn);
  }

  body.appendChild(topRow);

  s.summary = document.createElement('div');
  s.summary.className = 'campionamenti-summary';
  body.appendChild(s.summary);

  s.mapEl = document.createElement('div');
  s.mapEl.className = 'campionamenti-map-host';
  body.appendChild(s.mapEl);

  if (canModify()) {
    const hint = document.createElement('div');
    hint.className = 'form-help';
    hint.textContent = S.CAMPIONAMENTI_NO_AREAS_HINT;
    body.appendChild(hint);

    const addAreaBtn = document.createElement('button');
    addAreaBtn.type = 'button';
    addAreaBtn.className = 'btn btn-primary';
    addAreaBtn.textContent = S.ADD_AREA_LABEL;
    addAreaBtn.addEventListener('click', () => showAddAreaForm());
    body.appendChild(addAreaBtn);
  }

  s.pulldown = sel;
}

/**
 * Append pencil + garbage icons (writers-only affordances on the
 * Griglie / Rilevamenti pulldowns).  Same SortableTable icon glyphs.
 */
function appendEditDeleteIcons(host, { onEdit, onDelete }) {
  const edit = document.createElement('span');
  edit.className = 'action-icon action-edit campionamenti-pulldown-icon';
  edit.title = S.ACTION_EDIT;
  edit.textContent = '✎';
  edit.setAttribute('role', 'button');
  edit.addEventListener('click', onEdit);
  host.appendChild(edit);

  const del = document.createElement('span');
  del.className = 'action-icon action-delete campionamenti-pulldown-icon';
  del.title = S.ACTION_DELETE;
  del.textContent = '✕';
  del.setAttribute('role', 'button');
  del.addEventListener('click', onDelete);
  host.appendChild(del);
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
      altitude: r[c.indexOf('Quota')],
      r_m: r[c.indexOf('Raggio')],
      note: r[c.indexOf('Note')],
    }));

  const modify = canModify();
  s.map = new GriglieMap({
    container: s.mapEl,
    geojson: parcelleGeo,
    basemap: activeBasemap(),
    onAreaClick: (area) => showAreaPopover(area),
    onEmptyClick: modify
      ? (lat, lng) => promptNewAreaAt(lat, lng)
      : null,
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

  if (canModify()) {
    appendEditDeleteIcons(topRow, {
      onEdit: () => showRenameSurveyForm(),
      onDelete: () => confirmDeleteSurvey(),
    });
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-primary campionamenti-add-btn';
    addBtn.textContent = S.NEW_SURVEY_LABEL;
    addBtn.addEventListener('click', () => showNewSurveyForm());
    topRow.appendChild(addBtn);
  }

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
    basemap: activeBasemap(),
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

  // Inline header: shows the date of the active sample when an area
  // is selected (editable for writers).  Spec §Section 3.
  s.headerEl = document.createElement('div');
  s.headerEl.className = 'campionamenti-alberi-header';
  body.appendChild(s.headerEl);

  s.host = document.createElement('div');
  s.host.className = 'campionamenti-table-host';
  body.appendChild(s.host);
}

function renderAlberiHeader() {
  const s = sections.t;
  if (!s.headerEl) return;
  s.headerEl.replaceChildren();
  if (activeSurveyId == null || activeAreaId == null) return;

  // Look up the existing Sample (if any) for (survey, area) in samplesData.
  const sc = samplesData.columns;
  const row = samplesData.rows.find(
    r => r[sc.indexOf('Survey')] === activeSurveyId
      && r[sc.indexOf('Sample area')] === activeAreaId,
  );
  const currentDate = row ? row[sc.indexOf('Data')] : todayISO();

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.textContent = `${S.LABEL_DATE}:`;
  label.htmlFor = 'campionamenti-sample-date';

  if (canModify()) {
    const input = document.createElement('input');
    input.type = 'date';
    input.id = 'campionamenti-sample-date';
    input.value = currentDate || '';
    input.addEventListener('change', () => saveSampleDate(input.value));
    s.headerEl.append(label, input);
  } else {
    const span = document.createElement('span');
    span.textContent = currentDate || '—';
    s.headerEl.append(label, span);
  }
}

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

async function saveSampleDate(dateStr) {
  if (!dateStr) return;
  if (activeSurveyId == null || activeAreaId == null) return;
  try {
    const { data, status } = await postJSON(SAMPLE_DATE_SAVE_URL, {
      survey_id: activeSurveyId,
      sample_area_id: activeAreaId,
      date: dateStr,
    });
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    // Refresh samples digest + per-survey trees digest.
    await cache.load(SAMPLES_ID);
    samplesData = cache.get(SAMPLES_ID);
    if (currentTreesId) {
      try { await cache.load(currentTreesId); } catch {}
    }
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function showAlberiEmpty() {
  destroyTable();
  if (sections.t.emptyEl) sections.t.emptyEl.hidden = false;
  if (sections.t.headerEl) sections.t.headerEl.replaceChildren();
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
      onEdit: (tsId) => showEditTreeForm(tsId),
      onDelete: (tsId) => confirmDeleteTreeSample(tsId),
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
  renderAlberiHeader();
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
    mt: MAP_TYPE_TOKENS[params.mt] ? params.mt : DEFAULT_MAP_TYPE,
  };
}

function activeBasemap() {
  const u = new URLSearchParams(location.search);
  const tok = u.get('mt');
  return MAP_TYPE_TOKENS[tok] || MAP_TYPE_TOKENS[DEFAULT_MAP_TYPE];
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

  const mt = new URLSearchParams(location.search).get('mt');
  if (mt && MAP_TYPE_TOKENS[mt] && mt !== DEFAULT_MAP_TYPE) u.set('mt', mt);

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
// New grid / survey forms (M3d-write — "Crea vuota/o" path)
// ---------------------------------------------------------------------------

async function showNewGridForm() {
  inForm = true;
  const form = await fetchForm(GRID_FORM_URL);
  if (!form) { returnToPage(); return; }
  const modal = document.getElementById('campionamenti-grid-modal');
  if (!modal) { returnToPage(); return; }

  // Lazy-init the auto-generate planner when its tab is first shown.
  // Map + terreni.geojson fetch only run if the user picks that path.
  let planner = null;
  const onPathSwitch = (path) => {
    if (path === 'auto' && !planner) {
      const host = modal.querySelector('#campionamenti-grid-planner-host');
      if (host) {
        planner = new GridPlanner({
          host,
          onCreated: async (rowId) => {
            try {
              await cache.load(GRIDS_ID);
              gridsData = cache.get(GRIDS_ID);
              await cache.load(SAMPLE_AREAS_ID);
              sampleAreasData = cache.get(SAMPLE_AREAS_ID);
            } catch {}
            activeGridId = rowId;
            returnToPage();
          },
        });
        planner.init();
      }
    }
  };

  wirePathChooser(modal, onPathSwitch);
  wireGridEmptyForm(modal);
  wireGridCsvForm(modal);
  modal.querySelectorAll('#grid-form-cancel, #grid-auto-cancel, #grid-csv-cancel')
       .forEach(b => b.addEventListener('click', returnToPage));
  addEscapeHandler();
}

function wireGridEmptyForm(modal) {
  const form = modal.querySelector('#campionamenti-grid-form-empty');
  if (!form) return;
  interceptSubmit(form, GRID_SAVE_URL, {
    onSuccess: async (data) => {
      try {
        await cache.load(GRIDS_ID);
        gridsData = cache.get(GRIDS_ID);
      } catch {}
      activeGridId = data.row_id;
      returnToPage();
    },
    onValidationError(_data) {
      // Server returns empty html for simple validation errors;
      // interceptSubmit's error modal is enough.
    },
  });
}

async function showNewSurveyForm() {
  inForm = true;
  const form = await fetchForm(SURVEY_FORM_URL);
  if (!form) { returnToPage(); return; }
  const modal = document.getElementById('campionamenti-survey-modal');
  if (!modal) { returnToPage(); return; }
  wirePathChooser(modal);
  wireSurveyEmptyForm(modal);
  wireSurveyCsvForm(modal);
  modal.querySelectorAll('#survey-form-cancel, #survey-csv-cancel')
       .forEach(b => b.addEventListener('click', returnToPage));
  addEscapeHandler();
}

function wireSurveyEmptyForm(modal) {
  const form = modal.querySelector('#campionamenti-survey-form-empty');
  if (!form) return;
  interceptSubmit(form, SURVEY_SAVE_URL, {
    onSuccess: async (data) => {
      try {
        await cache.load(SURVEYS_ID);
        surveysData = cache.get(SURVEYS_ID);
      } catch {}
      activeSurveyId = data.row_id;
      returnToPage();
    },
    onValidationError(_data) {},
  });
}

/**
 * Wire the "Importa da CSV" body of a modal (grid or survey).  Submit
 * is multipart, not JSON, so we sidestep interceptSubmit.  On success
 * we update the affected caches, set the new selection, and return to
 * the main page.  On validation error we render the per-row error list
 * inside the form.
 *
 * @param {HTMLElement} modal
 * @param {{formId: string, postUrl: string,
 *          onSuccess: function(any): Promise<void>}} opts
 */
function wireCsvUploadForm(modal, opts) {
  const form = modal.querySelector(`#${opts.formId}`);
  if (!form) return;
  const errorsBox = form.querySelector('.csv-import-errors');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (errorsBox) { errorsBox.hidden = true; errorsBox.replaceChildren(); }
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) submitBtn.disabled = true;
    try {
      const fd = new FormData(form);
      const { data, status } = await postFormData(opts.postUrl, fd);
      if (status === 200) {
        await opts.onSuccess(data);
        return;
      }
      if (data?.errors && errorsBox) {
        renderCsvErrors(errorsBox, data.errors);
      } else {
        showError(data?.message || S.ERROR_GENERIC);
      }
    } catch {
      showError(S.ERROR_NETWORK);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

function renderCsvErrors(box, errors) {
  box.replaceChildren();
  const ul = document.createElement('ul');
  for (const e of errors.slice(0, 50)) {
    const li = document.createElement('li');
    li.textContent = e;
    ul.appendChild(li);
  }
  if (errors.length > 50) {
    const more = document.createElement('li');
    more.textContent = `… +${errors.length - 50} altri errori`;
    ul.appendChild(more);
  }
  box.appendChild(ul);
  box.hidden = false;
}

function wireGridCsvForm(modal) {
  wireCsvUploadForm(modal, {
    formId: 'campionamenti-grid-form-csv',
    postUrl: GRID_CSV_IMPORT_URL,
    onSuccess: async (data) => {
      await refreshGrids();
      try {
        await cache.load(SAMPLE_AREAS_ID);
        sampleAreasData = cache.get(SAMPLE_AREAS_ID);
      } catch {}
      activeGridId = data.row_id;
      returnToPage();
    },
  });
}

function wireSurveyCsvForm(modal) {
  wireCsvUploadForm(modal, {
    formId: 'campionamenti-survey-form-csv',
    postUrl: TREE_CSV_IMPORT_URL,
    onSuccess: async (data) => {
      await refreshSurveys();
      try {
        await cache.load(SAMPLES_ID);
        samplesData = cache.get(SAMPLES_ID);
      } catch {}
      activeSurveyId = data.row_id;
      returnToPage();
    },
  });
}

/**
 * Wire the [path-btn] / [modal-path-body] grouping inside a modal so
 * clicking a chooser button shows that path's body and hides the others.
 * Shared between Nuova griglia and Nuovo rilevamento modals.
 *
 * @param {HTMLElement} modal
 * @param {function(string): void} [onSwitch] — called with the new
 *   path id after each switch (used to lazy-init the auto-planner).
 */
function wirePathChooser(modal, onSwitch) {
  const buttons = modal.querySelectorAll('.path-btn');
  const bodies = modal.querySelectorAll('.modal-path-body');
  for (const btn of buttons) {
    btn.addEventListener('click', () => {
      const path = btn.dataset.path;
      for (const b of buttons) b.classList.toggle('active', b === btn);
      for (const body of bodies) {
        const isActive = body.classList.contains(`grid-path-${path}`)
                      || body.classList.contains(`survey-path-${path}`);
        body.classList.toggle('active', isActive);
      }
      onSwitch?.(path);
    });
  }
}

// ---------------------------------------------------------------------------
// Section 1 — area popover + new-area form (M3d-write)
// ---------------------------------------------------------------------------

/**
 * Open a popover on the active Griglie marker with the area's full
 * fields.  Writers get pencil/garbage icons; delete is refused when
 * the area is referenced by any Sample.
 */
function showAreaPopover(area) {
  const frag = document.createDocumentFragment();
  const list = document.createElement('div');
  list.className = 'form-readonly-block';
  for (const [label, val] of [
    ['Compresa', area.compresa],
    ['Particella', area.particella],
    ['Numero', area.numero],
    ['Lat', area.lat?.toFixed?.(5) ?? area.lat],
    ['Lng', area.lng?.toFixed?.(5) ?? area.lng],
    ['Quota', area.altitude ?? '—'],
    ['Raggio', area.r_m],
    ['Note', area.note || ''],
  ]) {
    const row = document.createElement('div');
    const b = document.createElement('strong');
    b.textContent = `${label}: `;
    row.append(b, document.createTextNode(String(val ?? '—')));
    list.appendChild(row);
  }
  frag.appendChild(list);

  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const close = document.createElement('button');
  close.className = 'btn';
  close.textContent = S.DISMISS;
  close.addEventListener('click', dismissModal);
  actions.appendChild(close);

  if (canModify()) {
    const edit = document.createElement('button');
    edit.className = 'btn btn-primary';
    edit.textContent = S.ACTION_EDIT;
    edit.addEventListener('click', () => {
      dismissModal();
      showEditAreaForm(area.id);
    });
    const del = document.createElement('button');
    del.className = 'btn btn-primary';
    del.textContent = S.ACTION_DELETE;
    del.addEventListener('click', () => {
      dismissModal();
      confirmDeleteArea(area.id);
    });
    actions.append(edit, del);
  }
  frag.appendChild(actions);
  showModal(frag);
}

function promptNewAreaAt(lat, lng) {
  if (activeGridId == null) return;
  const frag = document.createDocumentFragment();
  const p = document.createElement('p');
  p.textContent = S.CAMPIONAMENTI_INSERT_AREA_HERE;
  frag.appendChild(p);
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);
  const ok = document.createElement('button');
  ok.className = 'btn btn-primary';
  ok.textContent = S.SAVE;
  ok.addEventListener('click', () => {
    dismissModal();
    showAddAreaForm({ lat, lng });
  });
  actions.append(cancel, ok);
  frag.appendChild(actions);
  showModal(frag);
}

async function showAddAreaForm({ lat, lng } = {}) {
  if (activeGridId == null) return;
  inForm = true;
  const qs = new URLSearchParams({ grid: String(activeGridId) });
  if (lat != null) qs.set('lat', String(lat));
  if (lng != null) qs.set('lng', String(lng));
  const form = await fetchForm(`${AREA_FORM_URL}?${qs}`);
  if (!form) { returnToPage(); return; }
  wireAreaForm(form);
  addEscapeHandler();
}

async function showEditAreaForm(areaId) {
  inForm = true;
  const form = await fetchForm(`${AREA_FORM_URL}${areaId}/`);
  if (!form) { returnToPage(); return; }
  wireAreaForm(form);
  addEscapeHandler();
}

function wireAreaForm(form) {
  form.querySelector('#area-form-cancel')?.addEventListener('click', returnToPage);
  mountUseLocationButton(
    form.querySelector('#id_area_lat'),
    form.querySelector('#id_area_lng'),
  );
  // Filter Particella by Compresa.
  wireParcelByRegion(form);
  interceptSubmit(form, AREA_SAVE_URL, {
    onSuccess: async () => {
      try {
        await cache.load(SAMPLE_AREAS_ID);
        sampleAreasData = cache.get(SAMPLE_AREAS_ID);
        await cache.load(GRIDS_ID);
        gridsData = cache.get(GRIDS_ID);
      } catch {}
      returnToPage();
    },
    onValidationError(_data) {},
  });
}

function wireParcelByRegion(form) {
  const region = form.querySelector('#id_area_region');
  const parcel = form.querySelector('#id_area_parcel');
  if (!region || !parcel) return;
  const allOpts = [...parcel.options];
  function refresh() {
    const rid = region.value;
    parcel.replaceChildren();
    for (const opt of allOpts) {
      if (opt.dataset.regionId === rid) parcel.appendChild(opt.cloneNode(true));
    }
  }
  region.addEventListener('change', refresh);
  refresh();
}

function confirmDeleteArea(areaId) {
  const frag = document.createDocumentFragment();
  const p = document.createElement('p');
  p.textContent = S.DELETE_CONFIRM;
  frag.appendChild(p);
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);
  const ok = document.createElement('button');
  ok.className = 'btn btn-primary';
  ok.textContent = S.ACTION_DELETE;
  ok.addEventListener('click', async () => {
    dismissModal();
    await deleteArea(areaId);
  });
  actions.append(cancel, ok);
  frag.appendChild(actions);
  showModal(frag);
}

async function deleteArea(areaId) {
  try {
    const { data, status } = await postJSON(
      `${AREA_DELETE_URL_PREFIX}${areaId}/`, {},
    );
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    try {
      await cache.load(SAMPLE_AREAS_ID);
      sampleAreasData = cache.get(SAMPLE_AREAS_ID);
      await cache.load(GRIDS_ID);
      gridsData = cache.get(GRIDS_ID);
    } catch {}
    // Re-render the Griglie map to drop the deleted marker.
    if (activeGridId != null) renderGriglieMap(activeGridId);
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

// ---------------------------------------------------------------------------
// Rename + cascade-delete for grids and surveys (Bucket 2)
// ---------------------------------------------------------------------------

function showRenameGridForm() {
  if (activeGridId == null) return;
  const row = gridRow(activeGridId);
  if (!row) return;
  const c = gridsData.columns;
  showRenameModal({
    title: S.RENAME_TITLE_GRID,
    name: row[c.indexOf('Nome')] || '',
    description: row[c.indexOf('Descrizione')] || '',
    onSave: async ({ name, description }) => {
      try {
        const { data, status } = await postJSON(
          `${GRID_EDIT_URL_PREFIX}${activeGridId}/`, { name, description },
        );
        if (status !== 200) {
          showError(data?.message || S.ERROR_GENERIC);
          return false;
        }
        await refreshGrids();
        return true;
      } catch {
        showError(S.ERROR_NETWORK);
        return false;
      }
    },
  });
}

function showRenameSurveyForm() {
  if (activeSurveyId == null) return;
  const row = surveyRow(activeSurveyId);
  if (!row) return;
  const c = surveysData.columns;
  showRenameModal({
    title: S.RENAME_TITLE_SURVEY,
    name: row[c.indexOf('Nome')] || '',
    description: row[c.indexOf('Descrizione')] || '',
    onSave: async ({ name, description }) => {
      try {
        const { data, status } = await postJSON(
          `${SURVEY_EDIT_URL_PREFIX}${activeSurveyId}/`, { name, description },
        );
        if (status !== 200) {
          showError(data?.message || S.ERROR_GENERIC);
          return false;
        }
        await refreshSurveys();
        renderRilevamentiSummary(activeSurveyId);
        return true;
      } catch {
        showError(S.ERROR_NETWORK);
        return false;
      }
    },
  });
}

/** Shared rename modal — just Nome + Descrizione. */
function showRenameModal({ title, name, description, onSave }) {
  const frag = document.createDocumentFragment();
  const h = document.createElement('h2');
  h.textContent = title;
  frag.appendChild(h);

  const nameGroup = document.createElement('div');
  nameGroup.className = 'form-group';
  const nameLabel = document.createElement('label');
  nameLabel.textContent = S.LABEL_NAME;
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.maxLength = 100;
  nameInput.required = true;
  nameInput.value = name;
  nameGroup.append(nameLabel, nameInput);
  frag.appendChild(nameGroup);

  const descGroup = document.createElement('div');
  descGroup.className = 'form-group';
  const descLabel = document.createElement('label');
  descLabel.textContent = 'Descrizione';
  const descInput = document.createElement('textarea');
  descInput.rows = 3;
  descInput.value = description || '';
  descGroup.append(descLabel, descInput);
  frag.appendChild(descGroup);

  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);
  const ok = document.createElement('button');
  ok.className = 'btn btn-primary';
  ok.textContent = S.SAVE;
  ok.addEventListener('click', async () => {
    ok.disabled = true;
    const success = await onSave({
      name: nameInput.value.trim(),
      description: descInput.value.trim(),
    });
    if (success) dismissModal();
    else ok.disabled = false;
  });
  actions.append(cancel, ok);
  frag.appendChild(actions);
  showModal(frag);
  setTimeout(() => nameInput.focus(), 0);
}

function confirmDeleteGrid() {
  if (activeGridId == null) return;
  const row = gridRow(activeGridId);
  if (!row) return;
  const c = gridsData.columns;
  const nSurveys = row[c.indexOf('N. rilevamenti')] || 0;
  if (nSurveys > 0) {
    // Server refuses with ERR_GRID_IN_USE; surface the same message
    // without round-tripping.
    showError('La griglia è usata da uno o più rilevamenti: eliminarli prima.');
    return;
  }
  // No surveys → simple confirm.  Cascade goes to SampleAreas only.
  const nAreas = row[c.indexOf('N. aree')] || 0;
  const msg = nAreas > 0
    ? `${nAreas} aree saranno eliminate. ${S.DELETE_CONFIRM}`
    : S.DELETE_CONFIRM;
  simpleConfirmModal(msg, async () => {
    try {
      const { data, status } = await postJSON(
        `${GRID_DELETE_URL_PREFIX}${activeGridId}/`, {},
      );
      if (status !== 200) {
        showError(data?.message || S.ERROR_GENERIC);
        return;
      }
      activeGridId = null;
      await refreshGrids();
      // Refresh sample_areas too (cascaded).
      try {
        await cache.load(SAMPLE_AREAS_ID);
        sampleAreasData = cache.get(SAMPLE_AREAS_ID);
      } catch {}
      returnToPage();
    } catch {
      showError(S.ERROR_NETWORK);
    }
  });
}

function confirmDeleteSurvey() {
  if (activeSurveyId == null) return;
  const row = surveyRow(activeSurveyId);
  if (!row) return;
  const c = surveysData.columns;
  const nVisited = row[c.indexOf('N. aree visitate')] || 0;

  if (nVisited === 0) {
    simpleConfirmModal(S.DELETE_CONFIRM, () => doDeleteSurvey());
    return;
  }
  // Survey has samples → cascade flow with forced CSV export.
  const nTrees = countTreesInActiveSurvey();
  showCascadeDeleteModal({
    warning: S.CASCADE_WARN_SURVEY
      .replace('{n_samples}', nVisited)
      .replace('{n_trees}', nTrees),
    onExportCSV: () => exportSurveyCSV(activeSurveyId),
    onDelete: () => doDeleteSurvey(),
  });
}

async function doDeleteSurvey() {
  const id = activeSurveyId;
  try {
    const { data, status } = await postJSON(
      `${SURVEY_DELETE_URL_PREFIX}${id}/`, {},
    );
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    activeSurveyId = null;
    activeAreaId = null;
    // Drop the per-survey trees from the cache.
    if (currentTreesId === `${TREES_ID_PREFIX}${id}`) {
      currentTreesId = null;
    }
    await refreshSurveys();
    try {
      await cache.load(SAMPLES_ID);
      samplesData = cache.get(SAMPLES_ID);
    } catch {}
    returnToPage();
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function countTreesInActiveSurvey() {
  if (!currentTreesId) return 0;
  const d = cache.get(currentTreesId);
  return d?.rows?.length || 0;
}

/**
 * Cascade-delete confirm modal.  The "Elimina" button stays disabled
 * until the user clicks "Esporta CSV" (forces the operator to keep a
 * backup of the to-be-deleted rows).
 */
function showCascadeDeleteModal({ warning, onExportCSV, onDelete }) {
  const frag = document.createDocumentFragment();
  const h = document.createElement('h2');
  h.textContent = S.CASCADE_CONFIRM_TITLE;
  h.className = 'cascade-confirm-title';
  frag.appendChild(h);

  const warn = document.createElement('p');
  warn.className = 'cascade-confirm-warning';
  warn.textContent = warning;
  frag.appendChild(warn);

  const need = document.createElement('p');
  need.textContent = S.CASCADE_EXPORT_REQUIRED;
  frag.appendChild(need);

  const actions = document.createElement('div');
  actions.className = 'form-actions';

  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);

  const exportBtn = document.createElement('button');
  exportBtn.className = 'btn btn-primary';
  exportBtn.textContent = S.EXPORT_CSV;

  const delBtn = document.createElement('button');
  delBtn.className = 'btn btn-primary cascade-delete-btn';
  delBtn.textContent = S.ACTION_DELETE;
  delBtn.disabled = true;

  exportBtn.addEventListener('click', () => {
    onExportCSV();
    delBtn.disabled = false;
  });
  delBtn.addEventListener('click', () => {
    dismissModal();
    onDelete();
  });

  actions.append(cancel, exportBtn, delBtn);
  frag.appendChild(actions);
  showModal(frag);
}

function exportSurveyCSV(surveyId) {
  if (!currentTreesId) return;
  const d = cache.get(currentTreesId);
  if (!d) return;
  const fmt = S.TABLE_CSV_FORMAT;
  const lines = [];
  // Columns to include in the export — skip hidden synthetic ones.
  const visibleCols = d.columns.filter(
    c => c !== 'version' && c !== 'Sample area',
  );
  const idx = visibleCols.map(c => d.columns.indexOf(c));
  lines.push(visibleCols.join(fmt.separator));
  for (const row of d.rows) {
    const parts = idx.map(i => {
      const v = row[i];
      if (v == null) return '';
      if (typeof v === 'number') return String(v).replace('.', fmt.decimal);
      return String(v).replace(new RegExp(fmt.separator, 'g'), ' ');
    });
    lines.push(parts.join(fmt.separator));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = S.CSV_SAMPLED_TREES;
  a.click();
  URL.revokeObjectURL(a.href);
}

function simpleConfirmModal(message, onConfirm) {
  const frag = document.createDocumentFragment();
  const p = document.createElement('p');
  p.textContent = message;
  frag.appendChild(p);
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);
  const ok = document.createElement('button');
  ok.className = 'btn btn-primary';
  ok.textContent = S.ACTION_DELETE;
  ok.addEventListener('click', async () => {
    dismissModal();
    await onConfirm();
  });
  actions.append(cancel, ok);
  frag.appendChild(actions);
  showModal(frag);
}

async function refreshGrids() {
  try {
    await cache.load(GRIDS_ID);
    gridsData = cache.get(GRIDS_ID);
  } catch {}
}

async function refreshSurveys() {
  try {
    await cache.load(SURVEYS_ID);
    surveysData = cache.get(SURVEYS_ID);
  } catch {}
}

function gridRow(id) {
  const c = gridsData.columns;
  return gridsData.rows.find(r => r[c.indexOf('row_id')] === id);
}

function surveyRow(id) {
  const c = surveysData.columns;
  return surveysData.rows.find(r => r[c.indexOf('row_id')] === id);
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

async function showEditTreeForm(tsId) {
  inForm = true;
  const form = await fetchForm(`${TREE_FORM_URL}${tsId}/`);
  if (!form) { returnToPage(); return; }
  wireTreeForm(form);
  addEscapeHandler();
}

/**
 * Confirm + delete a single TreeSample.  Per spec §"Editing /
 * deletion", a tree_sample delete is non-cascading (the underlying
 * tree row is left intact) so no forced-CSV-export flow is needed —
 * just a standard confirm modal.
 */
function confirmDeleteTreeSample(tsId) {
  const frag = document.createDocumentFragment();
  const p = document.createElement('p');
  p.textContent = S.DELETE_CONFIRM;
  frag.appendChild(p);

  const actions = document.createElement('div');
  actions.className = 'form-actions';

  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);

  const ok = document.createElement('button');
  ok.className = 'btn btn-primary';
  ok.textContent = S.ACTION_DELETE;
  ok.addEventListener('click', async () => {
    dismissModal();
    await deleteTreeSample(tsId);
  });

  actions.append(cancel, ok);
  frag.appendChild(actions);
  showModal(frag);
}

async function deleteTreeSample(tsId) {
  try {
    const { data, status } = await postJSON(
      `${TREE_DELETE_URL_PREFIX}${tsId}/`, {},
    );
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    if (currentTreesId) {
      try { await cache.load(currentTreesId); } catch {}
    }
    // Also refresh samples digest (materialized N. alberi changed).
    try {
      await cache.load(SAMPLES_ID);
      samplesData = cache.get(SAMPLES_ID);
    } catch {}
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function wireTreeForm(form) {
  wireTreePick(form);
  wireVMPreview(form);
  mountUseLocationButton(
    form.querySelector('#id_lat'),
    form.querySelector('#id_lng'),
  );
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

/**
 * Wire the "Numero albero" pulldown.  Picking "+ nuovo albero"
 * leaves every other field freely editable; picking an existing
 * tree from the list locks specie / fustaia / lat / lng to that
 * tree's values (the server treats the existing Tree row as
 * authoritative — see views._parse_tree_body).
 *
 * The hidden #id_number field carries the actual numeric value
 * (server expects integer); options stash it in data-number /
 * data-next.
 */
function wireTreePick(form) {
  const pick = form.querySelector('#id_tree_pick');
  const num = form.querySelector('#id_number');
  if (!pick || !num) return;

  const species = form.querySelector('#id_species');
  const fustaia = form.querySelector('#id_fustaia');
  const lat = form.querySelector('#id_lat');
  const lng = form.querySelector('#id_lng');

  function setLocked(locked) {
    for (const el of [species, fustaia, lat, lng]) {
      if (el) el.disabled = locked;
    }
  }

  function apply() {
    const opt = pick.options[pick.selectedIndex];
    if (!opt) return;
    if (opt.value === 'new') {
      num.value = opt.dataset.next || '';
      setLocked(false);
      return;
    }
    // Existing tree: propagate its number, lock the rest.
    num.value = opt.dataset.number || '';
    if (species && opt.dataset.speciesId) species.value = opt.dataset.speciesId;
    if (fustaia) fustaia.checked = opt.dataset.coppice !== '1';
    if (lat && opt.dataset.lat !== undefined) lat.value = opt.dataset.lat;
    if (lng && opt.dataset.lng !== undefined) lng.value = opt.dataset.lng;
    setLocked(true);
    // Re-fire dependent listeners (V/m preview reads species/fustaia/D/h).
    species?.dispatchEvent(new Event('change'));
    fustaia?.dispatchEvent(new Event('change'));
  }

  pick.addEventListener('change', apply);
  apply();
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
  // Re-render the whole page shell from scratch.  Drop in-memory
  // selection state so applyParams triggers a full re-render of map
  // + table; otherwise the early-out in applyParams keeps the page
  // blank because the old DOM nodes were just discarded.
  destroyTable();
  destroyGriglieMap();
  destroyRilevamentiMap();
  activeGridId = activeSurveyId = activeAreaId = null;
  if (unsubCache) { unsubCache(); unsubCache = null; }
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
