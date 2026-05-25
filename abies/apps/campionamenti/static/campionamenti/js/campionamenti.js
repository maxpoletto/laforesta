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
import { show as showModal, showError, dismiss as dismissModal, onDismiss } from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';
import {
  FIELD_LAT, FIELD_LON, ROW_ID, VERSION,
} from '../../base/js/constants.js';
import { fetchJSON, postJSON, postFormData } from '../../base/js/api.js';
import {
  fetchForm, fetchModalForm, renderFormHTML, renderModalForm,
  interceptSubmit, wireCancelButtons, showFormError,
} from '../../base/js/forms.js';
import {
  mkCollapsible, mkEditDeleteIcons, renderCsvErrors,
} from '../../base/js/form-widgets.js';
import { RilevamentiMap } from './rilevamenti-map.js';
import { GriglieMap } from './griglie-map.js';
import { GridPlanner } from './grid-planner.js';
import MapCommon from '../../base/js/map-common.js';
import { mountUseLocationButton } from '../../base/js/latlng-input.js';
import { wireVMPreview as wireVMPreviewShared } from '../../base/js/tree-form.js';
import {
  fmtDecimal1, fmtDecimal2, fmtDecimal5, fmtInt, fmtBool,
} from '../../base/js/format.js';

const CSS_URL = '/static/campionamenti/css/campionamenti.css';
// Cache keys MUST match the server's `data_id` strings so that
// optimistic patches in `applySideEffects` find the right cache entry.
// See `apps/base/digests.py` for the matching generator names and
// CLAUDE.md §"Optimistic table updates".
const SURVEYS_ID = 'surveys';
const GRIDS_ID = 'grids';
const SAMPLE_AREAS_ID = 'sample_areas';
const SAMPLES_ID = 'samples';
const TREES_ID_PREFIX = 'sampled_trees_';

const SURVEYS_URL = '/api/campionamenti/surveys/data/';
const GRIDS_URL = '/api/campionamenti/grids/data/';
const SAMPLE_AREAS_URL = '/api/campionamenti/sample-areas/data/';
const SAMPLES_URL = '/api/campionamenti/samples/data/';
const TREES_URL_PREFIX = '/api/campionamenti/trees/';
const TREE_FORM_URL = '/api/campionamenti/tree/form/';
const TREE_SAVE_URL = '/api/campionamenti/tree/save/';
const TREE_DELETE_URL_PREFIX = '/api/campionamenti/tree/delete/';
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
// terreni.geojson carries the 99 per-particella polygons with
// `properties.name = "<Compresa>-<particella>"` (e.g. "Capistrano-10a")
// and `properties.layer = "<Compresa>"` — the shape `parcelLabel`
// expects.  `particelle.geojson` is the QGIS-export companion (4 outer
// boundaries + viabilità lines); it has neither per-particella
// polygons nor parsable names, so tooltips on it read like
// "03_FABRIZIA".
const TERRENI_GEOJSON_URL = '/api/geo/terreni.geojson';
const PAGE_PATH = '/campionamenti';

const SECTION_KEYS = ['g', 'r', 't'];
const DEFAULT_OPEN = 'r';

// mt= URL param → MapCommon basemap key.  Defaults to 's' (Satellite)
// per campionamenti.md §"URL parameters".
const MAP_TYPE_TOKENS = { o: 'osm', t: 'topo', s: 'satellite' };
const DEFAULT_MAP_TYPE = 's';


/** True if any cached Sample references `areaId`.  Used to disable
 *  the Section 1 "Elimina" button at popover-build time. */
function areaInUse(areaId) {
  if (!samplesData) return false;
  const c = samplesData.columns.indexOf(S.COL_SAMPLE_AREA);
  if (c < 0) return false;
  return samplesData.rows.some(r => r[c] === areaId);
}

const TREES_COLS = {
  [S.COL_SAMPLE_AREA]: { hidden: true },
  [S.COL_SAMPLE_DATE]: { label: S.LABEL_DATE, type: 'date', width: '90px' },
  [S.COL_COMPRESA]:    { label: S.COL_COMPRESA, width: '90px' },
  [S.COL_PARCEL]:      { label: S.COL_PARCEL, width: '85px' },
  [S.COL_AREA_NUM]:    { label: S.COL_AREA_NUM, width: '70px' },
  [S.COL_TREE_NUM]:    { label: S.COL_TREE_NUM_SHORT, type: 'number', width: '70px', formatter: fmtInt },
  [S.COL_SPECIES]:     { label: S.COL_SPECIES, width: '120px' },
  [S.COL_PRODUCT]:     { label: S.COL_PRODUCT, width: '70px' },
  [S.COL_POLLONE]:     { label: S.COL_POLLONE_SHORT, type: 'number', width: '55px', formatter: fmtInt },
  [S.COL_MATRICINA]:   { label: S.COL_MATRICINA_SHORT, type: 'boolean', width: '55px', formatter: fmtBool },
  [S.COL_D_CM]:        { label: S.COL_D_CM, type: 'number', width: '65px', formatter: fmtInt },
  [S.COL_H_M]:         { label: S.COL_H_M, type: 'number', width: '60px', formatter: fmtDecimal2 },
  [S.COL_L10_MM]:      { label: S.COL_L10_MM, type: 'number', width: '85px', formatter: fmtInt },
  [S.COL_V_M3]:        { label: S.COL_V_M3, type: 'number', width: '65px', formatter: fmtDecimal2 },
  [S.COL_MASS_Q]:      { label: S.COL_MASS_Q, type: 'number', width: '70px', formatter: fmtDecimal1 },
  [S.COL_PAI]:         { label: S.COL_PAI, type: 'boolean', width: '50px', formatter: fmtBool },
  [S.COL_LAT]:         { label: S.COL_LAT, type: 'number', width: '85px', formatter: fmtDecimal5 },
  [S.COL_LON]:         { label: S.COL_LON, type: 'number', width: '85px', formatter: fmtDecimal5 },
  [VERSION]: { label: VERSION, hidden: true },
};

// --- Page state -------------------------------------------------------------
const sections = {
  g: { title: S.SECTION_GRIGLIE,   open: false, header: null, body: null,
       pulldown: null, summary: null, mapEl: null, map: null,
       // {gridId, center: [lat,lng], zoom} — stashed across
       // re-renders so a grid save / area edit doesn't reset the
       // user's pan/zoom.  Keyed by gridId because `returnToPage`
       // zeroes the global `activeGridId` before re-running
       // applyParams; matching by id (not by global flag) lets the
       // re-render pick up the same view.  Discarded by the renderer
       // when the gridId changes.
       savedView: null },
  r: { title: S.SECTION_RILEVAMENTI, open: true, header: null, body: null,
       pulldown: null, summary: null, mapEl: null, map: null,
       // {surveyId, center, zoom} — same logic, keyed by surveyId.
       savedView: null },
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
// MapCommon basemap key in effect across all maps on this page.
// Initialised in mount() from the URL; updated by `basemapchange` events
// fired from any BasemapControl and by `applyParams` on back/forward.
let currentMapType = MAP_TYPE_TOKENS[DEFAULT_MAP_TYPE];

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
      fetchJSON(TERRENI_GEOJSON_URL),
    ]);
    surveysData = s;
    gridsData = g;
    sampleAreasData = sa;
    samplesData = sm;
    // Sort largest-first so small polygons render — and bind tooltip —
    // on top of their containing larger neighbours.  Mirrors
    // `bosco/b/app.js`'s use of the same helper.
    parcelleGeo = MapCommon.sortFeaturesByArea(geo.data);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  // Initialise the per-page basemap state from the URL before any map is
  // constructed.  All subsequent maps (and the BasemapControl on each)
  // start in sync with this value.
  const p0 = readParams(params);
  currentMapType = MAP_TYPE_TOKENS[p0.mt];

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
  const [header, body] = mkCollapsible(s.title, s.open);
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
  const left = document.createElement('div');
  left.className = 'campionamenti-section-left';
  const right = document.createElement('div');
  right.className = 'campionamenti-section-right';
  topRow.append(left, right);

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.textContent = S.GRID_LABEL;

  const sel = document.createElement('select');
  sel.className = 'campionamenti-pulldown page-pulldown';
  const idCol = gridsData.columns.indexOf(ROW_ID);
  const nameCol = gridsData.columns.indexOf(S.COL_NAME);
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

  left.append(label, sel);
  if (canModify()) {
    mkEditDeleteIcons(left, {
      onEdit: () => showEditGridModal(),
      onDelete: () => confirmDeleteGrid(),
      iconClass: 'campionamenti-pulldown-icon',
    });
  }

  const exportGridBtn = document.createElement('button');
  exportGridBtn.type = 'button';
  exportGridBtn.className = 'btn';
  exportGridBtn.textContent = S.EXPORT_CSV;
  exportGridBtn.addEventListener('click', () => {
    if (activeGridId != null) exportGridAreasCSV(activeGridId);
  });
  right.appendChild(exportGridBtn);

  if (canModify()) {
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-primary';
    addBtn.textContent = S.NEW_GRID_LABEL;
    addBtn.addEventListener('click', () => showNewGridForm());
    right.appendChild(addBtn);
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
  const row = gridsData.rows.find(r => r[c.indexOf(ROW_ID)] === gridId);
  s.summary.replaceChildren();
  if (!row) return;
  const desc = row[c.indexOf(S.COL_DESCRIPTION)] || '';

  const stats = document.createElement('div');
  stats.textContent =
    `${row[c.indexOf(S.COL_N_AREAS)]} aree · ` +
    `${row[c.indexOf(S.COL_REGIONS)]} · ` +
    `${row[c.indexOf(S.COL_N_SURVEYS)]} rilevamenti · ` +
    `aggiornata ${formatTimestamp(row[c.indexOf(S.COL_LAST_UPDATE)])}`;
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
    .filter(r => r[c.indexOf(S.COL_GRID)] === gridId)
    .map(r => ({
      id: r[c.indexOf(ROW_ID)],
      lat: r[c.indexOf(S.COL_LAT)],
      lon: r[c.indexOf(S.COL_LON)],
      compresa: r[c.indexOf(S.COL_COMPRESA)],
      particella: r[c.indexOf(S.COL_PARCEL)],
      numero: r[c.indexOf(S.COL_NUMBER)],
      altitude: r[c.indexOf(S.COL_QUOTA)],
      r_m: r[c.indexOf(S.COL_RAGGIO)],
      note: r[c.indexOf(S.COL_NOTE)],
    }));

  const modify = canModify();
  const sv = s.savedView;
  const initialView = (sv && sv.gridId === gridId) ? sv : null;
  s.map = new GriglieMap({
    container: s.mapEl,
    geojson: parcelleGeo,
    basemap: activeBasemap(),
    onAreaClick: (area) => showAreaPopover(area),
    onEmptyClick: modify
      ? (lat, lon) => promptNewAreaAt(lat, lon)
      : null,
    initialView,
    onViewChange: (center, zoom) => {
      s.savedView = { gridId, center, zoom };
    },
  });
  bindBasemapEvents(s.map);
  s.map.setAreas(areas);
}

function destroyGriglieMap() {
  if (sections.g.map) { sections.g.map.destroy(); sections.g.map = null; }
}

function rebuildGridSection(activeId) {
  const s = sections.g;
  destroyGriglieMap();
  s.pulldown = s.summary = s.mapEl = null;
  if (s.body) { s.body.replaceChildren(); buildGriglieBody(s.body); }
  if (activeId != null) activateGrid(activeId);
  syncURL();
}

function rebuildSurveySection(activeId) {
  const s = sections.r;
  destroyRilevamentiMap();
  s.pulldown = s.summary = s.mapEl = null;
  if (s.body) { s.body.replaceChildren(); buildRilevamentiBody(s.body); }
  if (activeId != null) activateSurvey(activeId);
  syncURL();
}

// ---------------------------------------------------------------------------
// Section 2 — Rilevamenti body
// ---------------------------------------------------------------------------

function buildRilevamentiBody(body) {
  const s = sections.r;

  const topRow = document.createElement('div');
  topRow.className = 'campionamenti-section-top';
  const left = document.createElement('div');
  left.className = 'campionamenti-section-left';
  const right = document.createElement('div');
  right.className = 'campionamenti-section-right';
  topRow.append(left, right);

  const label = document.createElement('label');
  label.className = 'campionamenti-pulldown-label';
  label.textContent = S.SURVEY_LABEL;

  const sel = document.createElement('select');
  sel.className = 'campionamenti-pulldown page-pulldown';
  const idCol = surveysData.columns.indexOf(ROW_ID);
  const nameCol = surveysData.columns.indexOf(S.COL_NAME);
  const visCol = surveysData.columns.indexOf(S.COL_N_AREAS_VISITED);
  const totCol = surveysData.columns.indexOf(S.COL_N_AREAS_TOTAL);
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

  left.append(label, sel);
  if (canModify()) {
    mkEditDeleteIcons(left, {
      onEdit: () => showEditSurveyModal(),
      onDelete: () => confirmDeleteSurvey(),
      iconClass: 'campionamenti-pulldown-icon',
    });
  }

  const exportSurveyBtn = document.createElement('button');
  exportSurveyBtn.type = 'button';
  exportSurveyBtn.className = 'btn';
  exportSurveyBtn.textContent = S.EXPORT_CSV;
  exportSurveyBtn.addEventListener('click', () => {
    if (activeSurveyId != null) exportFullSurveyCSV(activeSurveyId);
  });
  right.appendChild(exportSurveyBtn);

  if (canModify()) {
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-primary';
    addBtn.textContent = S.NEW_SURVEY_LABEL;
    addBtn.addEventListener('click', () => showNewSurveyForm());
    right.appendChild(addBtn);
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
  const row = surveysData.rows.find(r => r[c.indexOf(ROW_ID)] === surveyId);
  s.summary.replaceChildren();
  if (!row) return;
  const desc = row[c.indexOf(S.COL_DESCRIPTION)] || '';
  const gridId = row[c.indexOf(S.COL_GRID)];
  const gridName = lookupGridName(gridId);

  const stats = document.createElement('div');
  stats.textContent =
    `Griglia: ${gridName} · ` +
    `${row[c.indexOf(S.COL_N_AREAS_VISITED)]}/${row[c.indexOf(S.COL_N_AREAS_TOTAL)]} aree visitate · ` +
    (row[c.indexOf(S.COL_DATE_FIRST)]
      ? `dal ${row[c.indexOf(S.COL_DATE_FIRST)]} al ${row[c.indexOf(S.COL_DATE_LAST)]}`
      : S.STATUS_NO_SAMPLES);
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
  const row = gridsData.rows.find(r => r[c.indexOf(ROW_ID)] === gridId);
  return row ? row[c.indexOf(S.COL_NAME)] : '';
}

function renderRilevamentiMap(surveyId) {
  destroyRilevamentiMap();
  const s = sections.r;
  if (!s.mapEl) return;

  const surveyRow = surveysData.rows.find(
    r => r[surveysData.columns.indexOf(ROW_ID)] === surveyId,
  );
  if (!surveyRow) return;
  const gridId = surveyRow[surveysData.columns.indexOf(S.COL_GRID)];

  const c = sampleAreasData.columns;
  const areas = sampleAreasData.rows
    .filter(r => r[c.indexOf(S.COL_GRID)] === gridId)
    .map(r => ({
      id: r[c.indexOf(ROW_ID)],
      lat: r[c.indexOf(S.COL_LAT)],
      lon: r[c.indexOf(S.COL_LON)],
      compresa: r[c.indexOf(S.COL_COMPRESA)],
      particella: r[c.indexOf(S.COL_PARCEL)],
      numero: r[c.indexOf(S.COL_NUMBER)],
    }));

  const sc = samplesData.columns;
  const visitedById = new Map();
  for (const r of samplesData.rows) {
    if (r[sc.indexOf(S.COL_SURVEY)] !== surveyId) continue;
    visitedById.set(r[sc.indexOf(S.COL_SAMPLE_AREA)], {
      nAlberi: r[sc.indexOf(S.COL_N_TREES)],
    });
  }

  const sv = s.savedView;
  const initialView = (sv && sv.surveyId === surveyId) ? sv : null;
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
    initialView,
    onViewChange: (center, zoom) => {
      s.savedView = { surveyId, center, zoom };
    },
  });
  bindBasemapEvents(s.map);
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
    : { column: S.COL_COMPRESA, ascending: true };

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
  _areaColIdx = data.columns.indexOf(S.COL_SAMPLE_AREA);
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
  return currentMapType;
}

// Cross-map basemap sync.  Fired by either map's BasemapControl when the
// user clicks a thumbnail; mirrors the change onto the sibling map and
// writes the URL so the choice is bookmarkable.
function onBasemapChange(name) {
  if (name === currentMapType) return;
  currentMapType = name;
  if (sections.g.map?.wrapper) sections.g.map.wrapper.syncBasemap(name);
  if (sections.r.map?.wrapper) sections.r.map.wrapper.syncBasemap(name);
  syncURL();
}

// Bind once per map instance.  Safe to call after the map is constructed
// in activateGrid / activateSurvey.
function bindBasemapEvents(map) {
  if (!map?.leaflet) return;
  map.leaflet.on('basemapchange', (e) => onBasemapChange(e.name));
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

  // Back/forward: URL `mt` may differ from in-memory state.  Update
  // currentMapType and mirror onto any live maps; new maps built below
  // will pick the same value up via `activeBasemap()`.
  const desiredBasemap = MAP_TYPE_TOKENS[p.mt];
  if (desiredBasemap !== currentMapType) {
    currentMapType = desiredBasemap;
    if (sections.g.map?.wrapper) sections.g.map.wrapper.syncBasemap(desiredBasemap);
    if (sections.r.map?.wrapper) sections.r.map.wrapper.syncBasemap(desiredBasemap);
  }

  // Sync section open/closed.
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    const shouldBeOpen = p.o.includes(k);
    if (s.header && s.open !== shouldBeOpen) toggleSection(s, shouldBeOpen);
  }

  // Active grid — default to most-recently-updated when the URL is empty
  // OR points at a grid that no longer exists (handles stale URL after
  // a delete + back-button, and the post-delete returnToPage path).
  let targetGrid = p.g != null && gridRow(p.g) ? p.g : null;
  if (targetGrid == null && gridsData?.rows.length) {
    targetGrid = gridsData.rows[0][gridsData.columns.indexOf(ROW_ID)];
  }
  if (targetGrid != null && targetGrid !== activeGridId) {
    activateGrid(targetGrid);
  }

  // Active survey — same fallback as grid.
  let targetSurvey = p.s != null && surveyRow(p.s) ? p.s : null;
  if (targetSurvey == null && surveysData?.rows.length) {
    targetSurvey = surveysData.rows[0][surveysData.columns.indexOf(ROW_ID)];
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
    const defaultGrid = gridsData?.rows[0]?.[gridsData.columns.indexOf(ROW_ID)];
    if (activeGridId !== defaultGrid) u.set('g', String(activeGridId));
  }
  if (activeSurveyId != null) {
    const defaultSurvey = surveysData?.rows[0]?.[surveysData.columns.indexOf(ROW_ID)];
    if (activeSurveyId !== defaultSurvey) u.set('s', String(activeSurveyId));
  }
  if (activeAreaId != null) u.set('a', String(activeAreaId));

  // Round-trip the basemap token if it differs from the default.
  // `currentMapType` is the long-form key ('osm' / 'topo' / 'satellite');
  // invert MAP_TYPE_TOKENS to get the URL token.
  const mtToken = Object.keys(MAP_TYPE_TOKENS).find(
    k => MAP_TYPE_TOKENS[k] === currentMapType);
  if (mtToken && mtToken !== DEFAULT_MAP_TYPE) u.set('mt', mtToken);

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
  const form = await fetchModalForm(GRID_FORM_URL);
  if (!form) { inForm = false; return; }
  const modal = document.querySelector('#modal-container #campionamenti-grid-modal');
  if (!modal) { dismissModal(); inForm = false; return; }
  onDismiss(() => { inForm = false; });

  let planner = null;
  const onPathSwitch = (path) => {
    if (path === 'auto' && !planner) {
      const host = modal.querySelector('#campionamenti-grid-planner-host');
      if (host) {
        planner = new GridPlanner({
          host,
          basemap: activeBasemap(),
          onCreated: (rowId, response) => {
            if (response) applySideEffects(response);
            dismissModal();
            rebuildGridSection(rowId);
          },
        });
        planner.init();
      }
    }
  };

  wirePathChooser(modal, onPathSwitch);
  wireGridEmptyForm(modal);
  wireCancelButtons(modal, dismissModal);
}

function wireGridEmptyForm(modal) {
  const form = modal.querySelector('#campionamenti-grid-form-empty');
  if (!form) return;
  interceptSubmit(form, GRID_SAVE_URL, {
    onSuccess: (data) => {
      applySideEffects(data);
      dismissModal();
      rebuildGridSection(data.row_id);
    },
    onValidationError(_data) {},
  });
}

async function showNewSurveyForm() {
  inForm = true;
  const form = await fetchModalForm(SURVEY_FORM_URL);
  if (!form) { inForm = false; return; }
  const modal = document.querySelector('#modal-container #campionamenti-survey-modal');
  if (!modal) { dismissModal(); inForm = false; return; }
  onDismiss(() => { inForm = false; });
  wireSurveyEmptyForm(modal);
  wireCancelButtons(modal, dismissModal);
}

function wireSurveyEmptyForm(modal) {
  const form = modal.querySelector('#campionamenti-survey-form-empty');
  if (!form) return;
  interceptSubmit(form, SURVEY_SAVE_URL, {
    onSuccess: (data) => {
      applySideEffects(data);
      dismissModal();
      rebuildSurveySection(data.row_id);
    },
    onValidationError(_data) {},
  });
}

/**
 * Wire the tab bar inside a server-rendered modal so clicking a tab
 * button shows that tab's body and hides the others.
 */
function wirePathChooser(modal, onSwitch) {
  const buttons = modal.querySelectorAll('.modal-tab');
  const bodies = modal.querySelectorAll('.modal-tab-body');
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
    [S.COL_COMPRESA, area.compresa],
    [S.COL_PARCEL, area.particella],
    [S.COL_NUMBER, area.numero],
    [S.COL_LAT, area.lat?.toFixed?.(5) ?? area.lat],
    [S.COL_LON, area.lon?.toFixed?.(5) ?? area.lon],
    [S.COL_QUOTA, area.altitude ?? '—'],
    [S.COL_RAGGIO, area.r_m],
    [S.COL_NOTE, area.note || ''],
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
    // Server refuses deletion of an in-use area with ERR_AREA_IN_USE;
    // surface that at button-build time so the user doesn't go through
    // a confirm modal that's about to fail.  The server check stays
    // for the race-condition case (another writer inserts a sample
    // between this client's last digest fetch and our POST).
    if (areaInUse(area.id)) {
      del.disabled = true;
      del.title = S.AREA_IN_USE_TOOLTIP;
    } else {
      del.addEventListener('click', () => {
        dismissModal();
        confirmDeleteArea(area.id);
      });
    }
    actions.append(edit, del);
  }
  frag.appendChild(actions);
  showModal(frag);
}

function promptNewAreaAt(lat, lon) {
  if (activeGridId == null) return;
  const frag = document.createDocumentFragment();
  const p = document.createElement('p');
  p.textContent = S.CAMPIONAMENTI_INSERT_AREA_HERE;
  frag.appendChild(p);
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.dataset.action = 'cancel';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);
  const ok = document.createElement('button');
  ok.className = 'btn btn-primary';
  ok.textContent = S.CONFIRM;
  ok.addEventListener('click', () => {
    dismissModal();
    showAddAreaForm({ lat, lon });
  });
  actions.append(cancel, ok);
  frag.appendChild(actions);
  showModal(frag);
}

async function showAddAreaForm({ lat, lon } = {}) {
  if (activeGridId == null) return;
  inForm = true;
  const qs = new URLSearchParams({ grid: String(activeGridId) });
  if (lat != null) qs.set(FIELD_LAT, String(lat));
  if (lon != null) qs.set(FIELD_LON, String(lon));
  const form = await fetchModalForm(`${AREA_FORM_URL}?${qs}`);
  if (!form) { inForm = false; return; }
  onDismiss(() => { inForm = false; });
  wireAreaForm(form);
}

async function showEditAreaForm(areaId) {
  inForm = true;
  const form = await fetchModalForm(`${AREA_FORM_URL}${areaId}/`);
  if (!form) { inForm = false; return; }
  onDismiss(() => { inForm = false; });
  wireAreaForm(form);
}

function wireAreaForm(form) {
  wireCancelButtons(form, dismissModal);
  mountUseLocationButton(
    form.querySelector('#id_area_lat'),
    form.querySelector('#id_area_lon'),
    { appendTo: form.querySelector('#area-form-latlon-row') },
  );
  wireParcelByRegion(form);
  interceptSubmit(form, AREA_SAVE_URL, {
    onSuccess: (data) => {
      applySideEffects(data);
      dismissModal();
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
  cancel.dataset.action = 'cancel';
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
    cache.removeRow(SAMPLE_AREAS_ID, areaId);
    applySideEffects(data);
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

// ---------------------------------------------------------------------------
// Edit (details + CSV import) and cascade-delete for grids and surveys
// ---------------------------------------------------------------------------

function showEditGridModal() {
  if (activeGridId == null) return;
  const row = gridRow(activeGridId);
  if (!row) return;
  const c = gridsData.columns;
  showEditModal({
    title: S.RENAME_TITLE_GRID,
    name: row[c.indexOf(S.COL_NAME)] || '',
    description: row[c.indexOf(S.COL_DESCRIPTION)] || '',
    importTabLabel: S.EDIT_GRID_TAB_IMPORT,
    onDetailsSave: async ({ name, description }) => {
      const { data, status } = await postJSON(
        `${GRID_EDIT_URL_PREFIX}${activeGridId}/`, { name, description },
      );
      if (status !== 200) return data?.message || S.ERROR_GENERIC;
      applySideEffects(data);
      updatePulldownOption(sections.g, activeGridId, gridsData, 'Nome');
      return null;
    },
    buildImportForm: (form) => {
      const fileGroup = document.createElement('div');
      fileGroup.className = 'form-group';
      const fileLabel = document.createElement('label');
      fileLabel.textContent = S.LABEL_CSV_FILE;
      const fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.name = 'file';
      fileInput.accept = '.csv,text/csv';
      fileGroup.append(fileLabel, fileInput);
      form.appendChild(fileGroup);

      const help = document.createElement('p');
      help.className = 'form-help';
      help.textContent = 'Colonne necessarie: Compresa, Particella, '
        + 'Area saggio, Lon, Lat, Quota, Raggio.';
      form.appendChild(help);
    },
    onImportSubmit: async (form) => {
      const fd = new FormData(form);
      fd.append('sample_grid_id', String(activeGridId));
      fd.append('nonce', crypto.randomUUID());
      const { data, status } = await postFormData(GRID_CSV_IMPORT_URL, fd);
      if (status === 200) {
        applySideEffects(data);
        return { errors: null };
      }
      return { errors: data?.errors, message: data?.message };
    },
  });
}

function showEditSurveyModal() {
  if (activeSurveyId == null) return;
  const row = surveyRow(activeSurveyId);
  if (!row) return;
  const c = surveysData.columns;
  showEditModal({
    title: S.RENAME_TITLE_SURVEY,
    name: row[c.indexOf(S.COL_NAME)] || '',
    description: row[c.indexOf(S.COL_DESCRIPTION)] || '',
    importTabLabel: S.EDIT_SURVEY_TAB_IMPORT,
    onDetailsSave: async ({ name, description }) => {
      const { data, status } = await postJSON(
        `${SURVEY_EDIT_URL_PREFIX}${activeSurveyId}/`, { name, description },
      );
      if (status !== 200) return data?.message || S.ERROR_GENERIC;
      applySideEffects(data);
      rebuildSurveyPulldown();
      return null;
    },
    buildImportForm: (form) => {
      const fileGroup = document.createElement('div');
      fileGroup.className = 'form-group';
      const fileLabel = document.createElement('label');
      fileLabel.textContent = S.LABEL_CSV_FILE;
      const fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.name = 'file';
      fileInput.accept = '.csv,text/csv';
      fileGroup.append(fileLabel, fileInput);
      form.appendChild(fileGroup);

      const dateGroup = document.createElement('div');
      dateGroup.className = 'form-group';
      const dateLabel = document.createElement('label');
      dateLabel.textContent = 'Data predefinita (se il file non ha colonna "Data")';
      const dateInput = document.createElement('input');
      dateInput.type = 'date';
      dateInput.name = 'default_date';
      dateGroup.append(dateLabel, dateInput);
      form.appendChild(dateGroup);

      const help = document.createElement('p');
      help.className = 'form-help';
      help.textContent = 'Colonne necessarie: Compresa, Particella, '
        + 'Area saggio, Albero, Pollone, Matricina, D_cm, H_m, L10_mm, '
        + 'Genere, Fustaia. Opzionali: Data, PAI.';
      form.appendChild(help);
    },
    onImportSubmit: async (form) => {
      const fd = new FormData(form);
      fd.append('survey_id', String(activeSurveyId));
      fd.append('nonce', crypto.randomUUID());
      const { data, status } = await postFormData(TREE_CSV_IMPORT_URL, fd);
      if (status === 200) {
        await refreshSurveys();
        try {
          await cache.load(SAMPLES_ID);
          samplesData = cache.get(SAMPLES_ID);
        } catch {}
        return { errors: null };
      }
      return { errors: data?.errors, message: data?.message };
    },
  });
}

/**
 * Shared tabbed edit modal — Dettagli (name + description) and
 * Importa da CSV.  Mirrors the PDT pencil modal pattern.
 */
function showEditModal({
  title, name, description, importTabLabel,
  onDetailsSave, buildImportForm, onImportSubmit,
}) {
  const TAB_DETAILS = 'details';
  const TAB_IMPORT = 'import';

  const frag = document.createDocumentFragment();
  const card = document.createElement('div');
  card.className = 'form-card';
  frag.appendChild(card);

  const h = document.createElement('h2');
  h.textContent = title;
  card.appendChild(h);

  const tabs = document.createElement('div');
  tabs.className = 'modal-tabs';
  card.appendChild(tabs);

  const bodies = document.createElement('div');
  bodies.className = 'modal-tab-bodies';
  card.appendChild(bodies);

  const tabDefs = [
    { id: TAB_DETAILS, label: S.TAB_DETAILS },
    { id: TAB_IMPORT, label: importTabLabel },
  ];

  const bodyEls = {};
  for (const t of tabDefs) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'modal-tab';
    btn.dataset.path = t.id;
    btn.textContent = t.label;
    btn.addEventListener('click', () => switchTab(t.id));
    tabs.appendChild(btn);

    const body = document.createElement('div');
    body.className = 'modal-tab-body';
    body.dataset.path = t.id;
    bodies.appendChild(body);
    bodyEls[t.id] = body;
  }

  // --- Dettagli tab ---
  const detailsForm = document.createElement('form');
  detailsForm.className = 'campionamenti-edit-form';
  bodyEls[TAB_DETAILS].appendChild(detailsForm);

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
  detailsForm.appendChild(nameGroup);

  const descGroup = document.createElement('div');
  descGroup.className = 'form-group';
  const descLabel = document.createElement('label');
  descLabel.textContent = S.LABEL_DESCRIPTION;
  const descInput = document.createElement('textarea');
  descInput.rows = 3;
  descInput.value = description || '';
  descGroup.append(descLabel, descInput);
  detailsForm.appendChild(descGroup);

  const detailsActions = document.createElement('div');
  detailsActions.className = 'form-actions';
  const detailsCancel = document.createElement('button');
  detailsCancel.type = 'button';
  detailsCancel.className = 'btn';
  detailsCancel.textContent = S.CANCEL;
  detailsCancel.addEventListener('click', dismissModal);
  const detailsOk = document.createElement('button');
  detailsOk.type = 'submit';
  detailsOk.className = 'btn btn-primary';
  detailsOk.textContent = S.SAVE;
  detailsActions.append(detailsCancel, detailsOk);
  detailsForm.appendChild(detailsActions);

  detailsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    detailsOk.disabled = true;
    try {
      const err = await onDetailsSave({
        name: nameInput.value.trim(),
        description: descInput.value.trim(),
      });
      if (err) { showFormError(detailsForm, err); detailsOk.disabled = false; }
      else dismissModal();
    } catch {
      showFormError(detailsForm, S.ERROR_NETWORK);
      detailsOk.disabled = false;
    }
  });

  // --- Import tab ---
  const importForm = document.createElement('form');
  importForm.className = 'campionamenti-edit-form';
  bodyEls[TAB_IMPORT].appendChild(importForm);
  buildImportForm(importForm);

  const statusBox = document.createElement('div');
  statusBox.className = 'csv-import-status';
  statusBox.hidden = true;
  importForm.appendChild(statusBox);

  const errorsBox = document.createElement('div');
  errorsBox.className = 'csv-import-errors';
  errorsBox.hidden = true;
  importForm.appendChild(errorsBox);

  const importActions = document.createElement('div');
  importActions.className = 'form-actions';
  const importCancel = document.createElement('button');
  importCancel.type = 'button';
  importCancel.className = 'btn';
  importCancel.textContent = S.CANCEL;
  importCancel.addEventListener('click', dismissModal);
  const importOk = document.createElement('button');
  importOk.type = 'submit';
  importOk.className = 'btn btn-primary';
  importOk.textContent = S.IMPORT_LABEL;
  importActions.append(importCancel, importOk);
  importForm.appendChild(importActions);

  importForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    importOk.disabled = true;
    errorsBox.hidden = true;
    errorsBox.replaceChildren();
    statusBox.textContent = S.CSV_IMPORT_IN_PROGRESS;
    statusBox.hidden = false;
    try {
      const result = await onImportSubmit(importForm);
      if (!result.errors) { dismissModal(); return; }
      if (result.errors.length) renderCsvErrors(errorsBox, result.errors);
      else showFormError(importForm, result.message || S.ERROR_GENERIC);
    } catch {
      showFormError(importForm, S.ERROR_NETWORK);
    } finally {
      importOk.disabled = false;
      statusBox.hidden = true;
    }
  });

  function switchTab(id) {
    for (const btn of tabs.querySelectorAll('.modal-tab')) {
      btn.classList.toggle('active', btn.dataset.path === id);
    }
    for (const [k, b] of Object.entries(bodyEls)) {
      b.classList.toggle('active', k === id);
    }
  }

  showModal(frag);

  // Lock min-height to the tallest tab so switching doesn't reflow.
  const allBodies = Object.values(bodyEls);
  for (const b of allBodies) b.style.display = 'block';
  let maxH = 0;
  for (const b of allBodies) maxH = Math.max(maxH, b.offsetHeight);
  for (const b of allBodies) b.style.display = '';
  if (maxH > 0) bodies.style.minHeight = `${maxH}px`;

  switchTab(TAB_DETAILS);
  setTimeout(() => nameInput.focus(), 0);
}

function confirmDeleteGrid() {
  if (activeGridId == null) return;
  const row = gridRow(activeGridId);
  if (!row) return;
  const c = gridsData.columns;
  const nSurveys = row[c.indexOf(S.COL_N_SURVEYS)] || 0;
  if (nSurveys > 0) {
    // Server refuses with ERR_GRID_IN_USE; surface the same message
    // without round-tripping.
    showError(S.ERR_GRID_HAS_SURVEYS);
    return;
  }
  // No surveys → simple confirm.  Cascade goes to SampleAreas only.
  const nAreas = row[c.indexOf(S.COL_N_AREAS)] || 0;
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
  const nVisited = row[c.indexOf(S.COL_N_AREAS_VISITED)] || 0;

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
    // Survey delete cascades to samples + tree_samples and decrements
    // grids.N_rilevamenti.  Refresh all four affected caches.
    await Promise.all([refreshSurveys(), refreshGrids()]);
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
  cancel.dataset.action = 'cancel';
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
    c => c !== VERSION && c !== S.COL_SAMPLE_AREA,
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

/**
 * Section 1 "Esporta CSV": dump the active grid's sample areas in the
 * same column shape as `_grid_modal.html`'s "Importa da CSV" expects
 * (Compresa, Particella, Area saggio, Lon, Lat, Quota, Raggio).  This
 * is a round-trip with the import path — programming GPS devices is
 * the primary use case per the spec.
 */
function exportGridAreasCSV(gridId) {
  if (!sampleAreasData) return;
  const c = sampleAreasData.columns;
  const gridCol = c.indexOf(S.COL_GRID);
  // CSV output columns mirror the import format (CSV_COL_*).
  const cols = [S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA, S.CSV_COL_AREA_SAGGIO,
                S.CSV_COL_LON, S.CSV_COL_LAT,
                S.CSV_COL_QUOTA, S.CSV_COL_RAGGIO];
  // Source columns are the digest header names (COL_*).
  const srcCols = [S.COL_COMPRESA, S.COL_PARCEL, S.COL_NUMBER, S.COL_LON, S.COL_LAT,
                   S.COL_QUOTA, S.COL_RAGGIO];
  const idx = srcCols.map(s => c.indexOf(s));
  const fmt = S.TABLE_CSV_FORMAT;
  const lines = [cols.join(fmt.separator)];
  for (const row of sampleAreasData.rows) {
    if (row[gridCol] !== gridId) continue;
    const parts = idx.map(i => csvField(row[i], fmt));
    lines.push(parts.join(fmt.separator));
  }
  downloadCSV(lines, S.CSV_GRID_AREAS);
}

/**
 * Section 2 "Esporta CSV": dump every TreeSample on the active survey
 * in the same column shape as `_survey_modal.html`'s "Importa da CSV"
 * expects (Compresa, Particella, Area saggio, Albero, Pollone,
 * Matricina, D_cm, H_m, L10_mm, Genere, Fustaia, Data, PAI).  Distinct
 * from the Section 3 toolbar's CSV which respects the area / search
 * filter; this one is the whole survey.
 *
 * Lazy-fetches the per-survey digest if not already cached.
 */
async function exportFullSurveyCSV(surveyId) {
  const dataId = `${TREES_ID_PREFIX}${surveyId}`;
  cache.register(dataId, `${TREES_URL_PREFIX}${surveyId}/`);
  let d;
  try { d = await cache.load(dataId); }
  catch { showError(S.ERROR_NETWORK); return; }
  if (!d?.rows?.length) {
    showError(S.NO_RESULTS);
    return;
  }
  const c = d.columns;
  // CSV output columns mirror the import format (CSV_COL_*).
  const cols = [S.CSV_COL_COMPRESA, S.CSV_COL_PARTICELLA, S.CSV_COL_AREA_SAGGIO,
                S.CSV_COL_ALBERO, S.CSV_COL_POLLONE, S.CSV_COL_MATRICINA,
                S.CSV_COL_D_CM, S.CSV_COL_H_M, S.CSV_COL_L10_MM,
                S.CSV_COL_GENERE, S.CSV_COL_FUSTAIA, S.CSV_COL_DATA,
                S.CSV_COL_PAI];
  // Source columns are the digest header names (COL_*).
  const srcCols = [S.COL_COMPRESA, S.COL_PARCEL, S.COL_AREA_NUM, S.COL_TREE_NUM,
                   S.COL_POLLONE, S.COL_MATRICINA,
                   S.COL_D_CM, S.COL_H_M, S.COL_L10_MM,
                   S.COL_SPECIES, S.COL_PRODUCT, S.COL_SAMPLE_DATE, S.COL_PAI];
  const idx = srcCols.map(s => c.indexOf(s));
  const tipoCol = c.indexOf(S.COL_PRODUCT);
  const fmt = S.TABLE_CSV_FORMAT;
  const lines = [cols.join(fmt.separator)];
  for (const row of d.rows) {
    const parts = idx.map((i, k) => {
      if (cols[k] === S.CSV_COL_FUSTAIA) {
        // Round-trip with import: `Fustaia` = true|false, derived from
        // the digest's `Tipo` = 'fustaia' | 'ceduo'.
        return row[tipoCol] === 'fustaia' ? 'true' : 'false';
      }
      return csvField(row[i], fmt);
    });
    lines.push(parts.join(fmt.separator));
  }
  downloadCSV(lines, S.CSV_SURVEY_TREES);
}

function csvField(v, fmt) {
  if (v == null) return '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return String(v).replace('.', fmt.decimal);
  return String(v).replace(new RegExp(fmt.separator, 'g'), ' ');
}

function downloadCSV(lines, filename) {
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
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

/**
 * Update one option's text in a section's pulldown after a rename.
 * Looks up the digest row by id, reads `column` for the new label.
 */
function updatePulldownOption(section, id, digest, column) {
  if (!section?.pulldown || !digest) return;
  const c = digest.columns;
  const row = digest.rows.find(r => r[c.indexOf(ROW_ID)] === id);
  if (!row) return;
  const opt = section.pulldown.querySelector(`option[value="${id}"]`);
  if (opt) opt.textContent = row[c.indexOf(column)];
}

/** Survey pulldown label is "<name> (<n>/<m> aree)" — rebuild from cache. */
function rebuildSurveyPulldown() {
  const sel = sections.r.pulldown;
  if (!sel || !surveysData) return;
  const c = surveysData.columns;
  const idCol = c.indexOf(ROW_ID);
  const nameCol = c.indexOf(S.COL_NAME);
  const visCol = c.indexOf(S.COL_N_AREAS_VISITED);
  const totCol = c.indexOf(S.COL_N_AREAS_TOTAL);
  for (const opt of sel.options) {
    const row = surveysData.rows.find(r => r[idCol] === parseInt(opt.value, 10));
    if (row) {
      opt.textContent = `${row[nameCol]} (${row[visCol]}/${row[totCol]} aree)`;
    }
  }
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

/**
 * Apply all cache-update side effects from a write-view response and
 * re-render whichever views are affected.  This is the Prelievi
 * optimistic-update pattern adapted for Campionamenti's multi-cache
 * surface — see CLAUDE.md §"Optimistic table updates".
 *
 * Recognised payload keys (any subset, all optional):
 *   record          — single row for `data.data_id`
 *   records         — N rows for `data.data_id` (coppice multi-shoot, CSV)
 *   sample_record   — patch samplesData (one row)
 *   survey_record   — patch surveysData (one row)
 *   survey_records  — patch surveysData (N rows: area write touches every
 *                     survey on the grid)
 *   grid_record     — patch gridsData (one row)
 *   area_records    — patch sampleAreasData (N rows: bulk grid import /
 *                     auto-generate)
 *   removed_row_id  — implied: any `data.row_id` whose response carries
 *                     NO `record`/`records` is treated as a delete on
 *                     `data.data_id` (callers pass the deleted id as
 *                     `data.row_id` and the server omits the record).
 *
 * After patching the in-memory cache, mirrors the change to the
 * page-local mirrors (samplesData, surveysData, gridsData,
 * sampleAreasData) and re-renders maps / table / summary as needed.
 */
function applySideEffects(data) {
  if (!data) return;
  let touchedTreesDigest = false;
  let touchedSamples = false;
  let touchedSurveys = false;
  let touchedGrids = false;
  let touchedAreas = false;

  if (data.records && data.data_id) {
    cache.updateRows(data.data_id, data.records);
  } else if (data.record && data.data_id) {
    cache.updateRow(data.data_id, data.row_id, data.record);
  }
  // What did the primary `data_id` patch touch?  Used below to decide
  // which view to re-render.
  if (data.data_id === currentTreesId) {
    touchedTreesDigest = true;
  } else if (data.data_id === SAMPLE_AREAS_ID) {
    touchedAreas = true;
  } else if (data.data_id === SAMPLES_ID) {
    touchedSamples = true;
  } else if (data.data_id === SURVEYS_ID) {
    touchedSurveys = true;
  } else if (data.data_id === GRIDS_ID) {
    touchedGrids = true;
  }
  if (data.sample_record) {
    cache.updateRow(SAMPLES_ID, data.sample_record[0], data.sample_record);
    touchedSamples = true;
  }
  if (data.survey_record) {
    cache.updateRow(SURVEYS_ID, data.survey_record[0], data.survey_record);
    touchedSurveys = true;
  }
  if (data.survey_records?.length) {
    cache.updateRows(SURVEYS_ID, data.survey_records);
    touchedSurveys = true;
  }
  if (data.grid_record) {
    cache.updateRow(GRIDS_ID, data.grid_record[0], data.grid_record);
    touchedGrids = true;
  }
  if (data.area_records?.length) {
    cache.updateRows(SAMPLE_AREAS_ID, data.area_records);
    touchedAreas = true;
  }

  if (touchedTreesDigest && table && currentTreesId) {
    table.setData(cache.get(currentTreesId));
  }
  if (touchedSamples) {
    samplesData = cache.get(SAMPLES_ID);
    // Refresh Section 2 map markers (visited-count tooltip).
    if (activeSurveyId != null) renderRilevamentiMap(activeSurveyId);
  }
  if (touchedSurveys) {
    surveysData = cache.get(SURVEYS_ID);
    if (activeSurveyId != null) renderRilevamentiSummary(activeSurveyId);
    rebuildSurveyPulldown();
  }
  if (touchedGrids) {
    gridsData = cache.get(GRIDS_ID);
    if (activeGridId != null) renderGriglieSummary(activeGridId);
  }
  if (touchedAreas) {
    sampleAreasData = cache.get(SAMPLE_AREAS_ID);
    if (activeGridId != null) renderGriglieMap(activeGridId);
    if (activeSurveyId != null && surveyGridId(activeSurveyId) === activeGridId) {
      renderRilevamentiMap(activeSurveyId);
    }
  }
}

function gridRow(id) {
  const c = gridsData.columns;
  return gridsData.rows.find(r => r[c.indexOf(ROW_ID)] === id);
}

function surveyGridId(surveyId) {
  const row = surveyRow(surveyId);
  return row ? row[surveysData.columns.indexOf(S.COL_GRID)] : null;
}

function surveyRow(id) {
  const c = surveysData.columns;
  return surveysData.rows.find(r => r[c.indexOf(ROW_ID)] === id);
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
  const form = await fetchModalForm(url);
  if (!form) { inForm = false; return; }
  onDismiss(() => { inForm = false; });
  wireTreeForm(form);
}

async function showEditTreeForm(tsId) {
  inForm = true;
  const form = await fetchModalForm(`${TREE_FORM_URL}${tsId}/`);
  if (!form) { inForm = false; return; }
  onDismiss(() => { inForm = false; });
  wireTreeForm(form);
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
    // Optimistic remove + side-effect patches (samples, surveys).
    if (currentTreesId) {
      cache.removeRow(currentTreesId, tsId);
      if (table) table.setData(cache.get(currentTreesId));
    }
    applySideEffects(data);
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function wireTreeForm(form) {
  wireTreePick(form);
  wireCeduoToggle(form);
  wireCoppiceBlock(form);
  wireVMPreviewShared(form, { ceduoEl: form.querySelector('#tf-ceduo') });
  mountUseLocationButton(
    form.querySelector('#tf-lat'),
    form.querySelector('#tf-lon'),
    { appendTo: form.querySelector('#tf-lon')?.closest('.form-row') },
  );
  wireCancelButtons(form, dismissModal);
  interceptSubmit(form, TREE_SAVE_URL, {
    onSuccess: (data, isSaveAndAdd) => {
      applySideEffects(data);
      dismissModal();
      if (isSaveAndAdd) showAddTreeForm();
    },
    onConflict(data) {
      if (data.html) {
        const f = renderModalForm(data.html);
        if (f) wireTreeForm(f);
      }
    },
    onValidationError(data) {
      if (data.html) {
        const f = renderModalForm(data.html);
        if (f) wireTreeForm(f);
      }
    },
  });
}

/**
 * Wire the "Numero albero" pulldown.  Picking "+ nuovo albero"
 * leaves every other field freely editable; picking an existing
 * tree from the list locks specie / fustaia / lat / lon to that
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

  const species = form.querySelector('#tf-species');
  const ceduo = form.querySelector('#tf-ceduo');
  const lat = form.querySelector('#tf-lat');
  const lon = form.querySelector('#tf-lon');
  // row_id is non-empty in edit mode.  The edit path lets the user
  // adjust the underlying Tree's species / lat / lon (see
  // views._update_tree_sample), so we must NOT lock those inputs
  // when row_id is set.  But the tree number itself is fixed in edit:
  // an edit operates on one specific TreeSample, not a pivot to
  // another tree.  Lock the pulldown and hide "Salva e continua"
  // (which is a batch-entry affordance, meaningless for edits).
  const isEditMode = !!form.querySelector('input[name="row_id"]')?.value;
  if (isEditMode) {
    pick.disabled = true;
    const saveAndAdd = form.querySelector('button[data-action="save-and-add"]');
    if (saveAndAdd) saveAndAdd.style.display = 'none';
  }

  function setLocked(locked) {
    for (const el of [species, ceduo, lat, lon]) {
      if (el) el.disabled = locked;
    }
    // The "Usa posizione attuale" button is appended to the lon's
    // form-group later (mountUseLocationButton runs after this).
    // Re-query each time so we catch it once it's been mounted.
    const geoBtn = form.querySelector('.latlng-use-location');
    if (geoBtn) geoBtn.disabled = locked;
  }

  function apply() {
    const opt = pick.options[pick.selectedIndex];
    if (!opt) return;
    if (opt.value === 'new') {
      num.value = opt.dataset.next || '';
      setLocked(false);
      return;
    }
    // Existing tree: propagate its number, lock the rest UNLESS we're
    // editing (the edit path keeps Tree fields editable).  When the
    // existing tree has NULL lat/lon (option carries empty string),
    // leave the inputs alone so the template's area-centre default
    // survives.
    num.value = opt.dataset.number || '';
    if (species && opt.dataset.speciesId) species.value = opt.dataset.speciesId;
    if (ceduo) ceduo.checked = opt.dataset.coppice === '1';
    if (lat && opt.dataset.lat) lat.value = opt.dataset.lat;
    if (lon && opt.dataset.lon) lon.value = opt.dataset.lon;
    setLocked(!isEditMode);
    species?.dispatchEvent(new Event('change'));
    ceduo?.dispatchEvent(new Event('change'));
  }

  pick.addEventListener('change', apply);
  apply();
}

/**
 * Toggle visibility + submission of the fustaia / coppice subforms based
 * on the Ceduo checkbox: unchecked = simple fustaia entry (the common
 * case), checked = per-shoot coppice block.  Disabled inputs aren't
 * submitted, so we can disable instead of removing — switching modes
 * preserves any in-progress entry on the inactive side.
 */
function wireCeduoToggle(form) {
  const ceduo = form.querySelector('#tf-ceduo');
  const fustaiaBlock = form.querySelector('.tree-fustaia-fields');
  const coppiceBlock = form.querySelector('.tree-coppice-block');
  if (!ceduo || !fustaiaBlock || !coppiceBlock) return;

  function apply() {
    const isCoppice = ceduo.checked;
    fustaiaBlock.hidden = isCoppice;
    coppiceBlock.hidden = !isCoppice;
    for (const el of fustaiaBlock.querySelectorAll('input')) {
      el.disabled = isCoppice;
    }
    for (const el of coppiceBlock.querySelectorAll('input')) {
      el.disabled = !isCoppice;
    }
  }
  ceduo.addEventListener('change', apply);
  apply();
}

/**
 * Wire the coppice per-shoot block:
 *  - "+ Aggiungi pollone" clones the last row, increments its shoot
 *    number, blanks the values, attaches a "Rimuovi" button.
 *  - On form submit, serialise all coppice-shoot rows into the hidden
 *    `#id_shoots` field as JSON.  Attached before `interceptSubmit`
 *    so it runs first; that handler then sees the populated payload.
 *  - When the active tree picks an existing tree (data-next-shoot
 *    propagates), the first row's shoot number tracks that value.
 */
function wireCoppiceBlock(form) {
  const block = form.querySelector('.tree-coppice-block');
  if (!block) return;
  const shootsHost = block.querySelector('.coppice-shoots');
  const addBtn = block.querySelector('#coppice-add-btn');
  const ceduo = form.querySelector('#tf-ceduo');
  const pick = form.querySelector('#id_tree_pick');
  const shootsHidden = form.querySelector('#id_shoots');

  function rows() {
    return [...shootsHost.querySelectorAll('.coppice-shoot-row')];
  }

  function renumberLabels() {
    for (const row of rows()) {
      const lbl = row.querySelector('.coppice-shoot-label');
      if (lbl) lbl.textContent = row.dataset.shoot;
    }
  }

  function nextShootFromPick() {
    const opt = pick?.options[pick.selectedIndex];
    return parseInt(opt?.dataset.nextShoot || '1', 10);
  }

  // When operator switches the "Numero albero" pulldown to an existing
  // tree, restart the shoot numbering at that tree's next available
  // shoot.  Only on the add path (single row); edit path keeps the
  // shoot number it was rendered with.
  pick?.addEventListener('change', () => {
    if (rows().length !== 1) return;
    if (!addBtn) return;             // edit path: never renumber
    const first = rows()[0];
    first.dataset.shoot = String(nextShootFromPick());
    renumberLabels();
  });

  addBtn?.addEventListener('click', () => {
    const all = rows();
    const last = all[all.length - 1];
    const nextShoot = parseInt(last.dataset.shoot || '0', 10) + 1;
    const clone = last.cloneNode(true);
    clone.dataset.shoot = String(nextShoot);
    for (const inp of clone.querySelectorAll('input')) {
      if (inp.type === 'checkbox') inp.checked = false;
      else inp.value = '';
    }
    // cloneNode(true) copies DOM but not event listeners, so any
    // .coppice-remove-btn on the clone is dead.  Strip it and rebuild
    // — same as on the first added row (the first pre-template row
    // has no Remove button at all).
    const staleRemove = clone.querySelector('.coppice-remove-btn');
    if (staleRemove) staleRemove.remove();
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn coppice-remove-btn';
    removeBtn.textContent = S.REMOVE_POLLONE;
    removeBtn.addEventListener('click', () => {
      clone.remove();
      renumberLabels();
    });
    clone.appendChild(removeBtn);
    shootsHost.appendChild(clone);
    renumberLabels();
  });

  form.addEventListener('submit', () => {
    if (!shootsHidden) return;
    // Always serialise — the server ignores `shoots` when in fustaia
    // mode.  Disabling the coppice inputs (wireCeduoToggle) does NOT
    // remove the hidden field, so blank it explicitly here.
    if (!ceduo?.checked) {
      shootsHidden.value = '';
      return;
    }
    const list = rows().map(row => ({
      shoot: parseInt(row.dataset.shoot || '1', 10),
      standard: row.querySelector('.coppice-standard')?.checked || false,
      d_cm: row.querySelector('.coppice-d-cm')?.value || '',
      h_m: row.querySelector('.coppice-h-m')?.value || '',
      // L10 isn't meaningful per pollone; server defaults to 0.
    }));
    shootsHidden.value = JSON.stringify(list);
  });
}


/**
 * Tear down the form-modal and re-render the main page shell.
 *
 * @param {object} [overrides] — URL-param overrides merged on top of the
 *   current `location.search`.  Callers use this to pin a just-created
 *   entity as the active selection: e.g. `returnToPage({g: '7'})` opens
 *   the page with grid id 7 as the active grid.  Without overrides we
 *   honour whatever is in the URL (with the existing "fall back to first
 *   row if the id is stale" logic in applyParams).
 */
function returnToPage(overrides = {}) {
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

  // Explicit clear before rebuild.  buildPageShell would also do this,
  // but doing it here as well guarantees the form-card is GONE as soon
  // as we return — even if any of the section-build steps below throws,
  // the user is no longer staring at the now-stale upload form.  This
  // closes a class of "modal won't close after import" bugs.
  const el = document.getElementById('content');
  if (el) el.replaceChildren();

  const params = {
    ...Object.fromEntries(new URLSearchParams(location.search)),
    ...overrides,
  };
  if (el) buildPageShell(el, params);
  applyParams(params);
  // After a successful save/delete the URL may carry an id that no
  // longer exists (e.g., `s=<deleted_id>`).  applyParams falls back to
  // the first row of the digest; syncURL drops the stale param so the
  // URL stays meaningful and the next reload picks the same default.
  syncURL();
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
