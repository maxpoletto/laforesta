/**
 * Piano di taglio page.
 *
 * Hosts the plan-selector header (pulldown + edit / delete / export /
 * "+ Nuovo piano") and, below it, the fustaia and ceduo calendar
 * sections + the bookmarkable view/edit-item modal.
 *
 * URL params: see docs/pages/piano-di-taglio.md.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  show as showModal, showError, dismiss as dismissModal,
} from '../../base/js/modals.js';
import { fetchJSON, postJSON, postFormData } from '../../base/js/api.js';
import { interceptSubmit, wireCancelButtons } from '../../base/js/forms.js';
import * as S from '../../base/js/strings.js';
import { ROW_ID, VERSION } from '../../base/js/constants.js';

const CSS_URL = '/static/piano_di_taglio/css/piano-di-taglio.css';

// Cache keys MUST match server `data_id` strings (apps/base/digests.py).
const PLANS_ID = 'harvest_plans';
const ITEMS_ID = 'harvest_plan_items';
const REGRESSIONS_ID = 'tree_height_regressions';

const PLANS_URL = '/api/piano-di-taglio/plans/data/';
const ITEMS_URL = '/api/piano-di-taglio/items/data/';
const REGRESSIONS_URL = '/api/piano-di-taglio/regressions/data/';
const PLAN_SAVE_URL = '/api/piano-di-taglio/plan/save/';
const PLAN_IMPORT_CSV_URL = '/api/piano-di-taglio/plan/import-csv/';
const PLAN_DELETE_URL = '/api/piano-di-taglio/plan/delete/';
const PLAN_EXPORT_URL = '/api/piano-di-taglio/plan/export/';
const ITEM_FORM_URL = '/api/piano-di-taglio/item/form/';
const ITEM_SAVE_URL = '/api/piano-di-taglio/item/save/';
const ITEM_DELETE_URL = '/api/piano-di-taglio/item/delete/';
const ITEM_EXPORT_URL = '/api/piano-di-taglio/item/export/';

const PAGE_PATH = '/piano-di-taglio';

cache.register(PLANS_ID, PLANS_URL);
cache.register(ITEMS_ID, ITEMS_URL);
cache.register(REGRESSIONS_ID, REGRESSIONS_URL);

// --- Page state -------------------------------------------------------------

let activePlanId = null;
let plansData = null;
let itemsData = null;
let regressionsData = null;
let unsubPlans = null;
let unsubItems = null;

let descriptionEl = null;
let planSelectEl = null;

// Calendar sections — keyed by the single-char URL `o=` token.  `f`
// fills in here; `c` (Calendario ceduo) lands in a later increment.
const SECTION_KEYS = ['f', 'c'];
const DEFAULT_OPEN = 'f';

const sections = {
  f: {
    title: S.SECTION_INTERVENTI_FUSTAIA, open: true,
    kind: 'fustaia',
    header: null, body: null, host: null, table: null,
    toolbar: null, actionAdd: null, emptyState: null,
    typeMatcher: (tipo) => tipo !== S.TYPE_CEDUO,
    hiddenCols: [
      S.COL_HARVEST_PLAN, S.COL_TYPE,
      S.COL_INTERVENTION_AREA_HA, S.COL_PARCEL_AREA_HA, S.COL_TURNO_A,
      S.COL_EXTRA_NOTE,
    ],
    csvFilename: 'interventi-fustaia.csv',
  },
  c: {
    title: S.SECTION_INTERVENTI_CEDUO, open: false,
    kind: 'ceduo',
    header: null, body: null, host: null, table: null,
    toolbar: null, actionAdd: null, emptyState: null,
    typeMatcher: (tipo) => tipo === S.TYPE_CEDUO,
    hiddenCols: [
      S.COL_HARVEST_PLAN, S.COL_TYPE,
      S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED,
      // Altre note (free-text) IS shown for ceduo — pdg-2026 uses it
      // for continuation markers like "Cont. intervento 2028".
    ],
    csvFilename: 'interventi-ceduo.csv',
  },
};

function canModify() {
  return ['admin', 'writer'].includes(document.body.dataset.role);
}

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
    const [p, i, r] = await Promise.all([
      cache.load(PLANS_ID),
      cache.load(ITEMS_ID),
      cache.load(REGRESSIONS_ID),
    ]);
    plansData = p;
    itemsData = i;
    regressionsData = r;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  buildPageShell(el);
  cache.setVisible([PLANS_ID, ITEMS_ID, REGRESSIONS_ID]);
  if (unsubPlans) unsubPlans();
  if (unsubItems) unsubItems();
  unsubPlans = cache.onUpdate(PLANS_ID, onPlansUpdate);
  unsubItems = cache.onUpdate(ITEMS_ID, onItemsUpdate);

  applyParams(params);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (unsubPlans) { unsubPlans(); unsubPlans = null; }
  if (unsubItems) { unsubItems(); unsubItems = null; }
  cache.setVisible([]);
  destroyTables();
  activePlanId = null;
  plansData = itemsData = regressionsData = null;
  descriptionEl = null;
  planSelectEl = null;
}

function destroyTables() {
  for (const k of SECTION_KEYS) {
    if (sections[k].table) {
      sections[k].table.destroy();
      sections[k].table = null;
    }
  }
}

function onItemsUpdate() {
  itemsData = cache.get(ITEMS_ID);
  for (const k of SECTION_KEYS) {
    if (sections[k].table) sections[k].table.setData(itemsData);
    updateEmptyState(sections[k]);
  }
}

export function onQueryChange(params) {
  applyParams(params);
}

// ---------------------------------------------------------------------------
// URL params
// ---------------------------------------------------------------------------

function readParams(params) {
  return {
    p: params.p ? parseInt(params.p, 10) : null,
    o: params.o !== undefined ? params.o : DEFAULT_OPEN,
    ff: params.ff || '',
    fsc: params.fsc || null,
    fso: params.fso !== '1',
    cf: params.cf || '',
    csc: params.csc || null,
    cso: params.cso !== '1',
  };
}

function applyParams(params) {
  const p = readParams(params);

  // Active plan — fall back to most-recent (latest year_start) when URL is
  // empty or points to a missing id.  Digest is sorted by year_start DESC,
  // so rows[0] is the latest.
  let target = p.p != null && planRow(p.p) ? p.p : null;
  if (target == null && plansData?.rows.length) {
    target = plansData.rows[0][plansData.columns.indexOf(ROW_ID)];
  }
  if (target !== activePlanId) setActivePlan(target);

  // Section open/closed.
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    const shouldBeOpen = p.o.includes(k);
    if (s.header && s.open !== shouldBeOpen) toggleSection(s, shouldBeOpen);
  }
}

function syncURL() {
  const u = new URLSearchParams();
  if (activePlanId != null) {
    const defaultPlan = plansData?.rows[0]?.[plansData.columns.indexOf(ROW_ID)];
    if (activePlanId !== defaultPlan) u.set('p', String(activePlanId));
  }
  const openKeys = SECTION_KEYS.filter(k => sections[k].open).join('');
  if (openKeys !== DEFAULT_OPEN) u.set('o', openKeys);

  const f = sections.f;
  if (f.table) {
    const sort = f.table.getSort();
    if (sort) {
      u.set('fsc', sort.column);
      u.set('fso', sort.ascending ? '0' : '1');
    }
    const t = f.table.getSearchText();
    if (t) u.set('ff', t);
  }

  const c = sections.c;
  if (c.table) {
    const sort = c.table.getSort();
    if (sort) {
      u.set('csc', sort.column);
      u.set('cso', sort.ascending ? '0' : '1');
    }
    const t = c.table.getSearchText();
    if (t) u.set('cf', t);
  }

  const qs = u.toString();
  router.navigate(PAGE_PATH + (qs ? '?' + qs : ''), true);
}

// ---------------------------------------------------------------------------
// Page shell
// ---------------------------------------------------------------------------

function buildPageShell(el) {
  el.replaceChildren();

  const header = document.createElement('div');
  header.className = 'pdt-header';
  const left = document.createElement('div');
  left.className = 'pdt-header-left';
  const right = document.createElement('div');
  right.className = 'pdt-header-right';
  header.append(left, right);

  if (!plansData?.rows.length) {
    const empty = document.createElement('div');
    empty.className = 'pdt-empty';
    empty.textContent = S.PDT_NO_PLANS;
    left.appendChild(empty);
    if (canModify()) right.appendChild(buildNewPlanButton());
    el.appendChild(header);
    return;
  }

  const label = document.createElement('label');
  label.className = 'pdt-pulldown-label';
  label.textContent = S.LABEL_HARVEST_PLAN;

  const sel = document.createElement('select');
  sel.className = 'pdt-pulldown page-pulldown';
  const idCol = plansData.columns.indexOf(ROW_ID);
  const nameCol = plansData.columns.indexOf(S.COL_NAME);
  for (const row of plansData.rows) {
    const opt = document.createElement('option');
    opt.value = String(row[idCol]);
    opt.textContent = row[nameCol];
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => {
    setActivePlan(parseInt(sel.value, 10));
    syncURL();
  });
  label.htmlFor = sel.id = 'pdt-plan-select';
  left.append(label, sel);
  planSelectEl = sel;

  if (canModify()) {
    appendEditDeleteIcons(left, {
      onEdit: () => onEditPlan(),
      onDelete: () => onDeletePlan(),
    });
  }

  const exportBtn = document.createElement('button');
  exportBtn.type = 'button';
  exportBtn.className = 'btn';
  exportBtn.textContent = S.EXPORT_CSV;
  exportBtn.addEventListener('click', () => {
    if (activePlanId != null) downloadPlanExport(activePlanId);
  });
  right.appendChild(exportBtn);

  if (canModify()) right.appendChild(buildNewPlanButton());

  el.appendChild(header);

  descriptionEl = document.createElement('div');
  descriptionEl.className = 'pdt-description';
  el.appendChild(descriptionEl);

  buildSection(el, sections.f);
  buildSection(el, sections.c);
}

/** Build a collapsible calendar section (fustaia or ceduo). */
function buildSection(el, s) {
  const [header, body] = collapsible(s.title, s.open);
  s.header = header;
  s.body = body;

  // Per-section search box (above table; the toolbar lives inside the
  // collapsible body so it disappears when the section is collapsed).
  const toolbar = document.createElement('div');
  toolbar.className = 'pdt-section-toolbar';
  const searchLabel = document.createElement('label');
  searchLabel.className = 'pdt-search-label';
  searchLabel.textContent = S.FILTER_LABEL;
  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.className = 'table-search';
  searchInput.placeholder = S.SEARCH_PLACEHOLDER;
  searchLabel.htmlFor = searchInput.id = `pdt-search-${s === sections.f ? 'f' : 'c'}`;
  toolbar.append(searchLabel, searchInput);

  const csvBtn = document.createElement('button');
  csvBtn.type = 'button';
  csvBtn.className = 'btn btn-primary pdt-csv-btn';
  csvBtn.textContent = S.EXPORT_CSV;
  csvBtn.addEventListener('click', () => s.table?.exportCSV());
  toolbar.appendChild(csvBtn);

  body.appendChild(toolbar);
  s.toolbar = toolbar;

  s.host = document.createElement('div');
  s.host.className = 'pdt-table-host';
  body.appendChild(s.host);

  if (canModify()) {
    const addRow = document.createElement('div');
    addRow.className = 'action-add';
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn btn-primary btn-add';
    addBtn.textContent = S.ADD_ITEM_LABEL;
    addBtn.addEventListener('click', () => showAddItemModal(s));
    addRow.appendChild(addBtn);
    body.appendChild(addRow);
    s.actionAdd = addRow;

    s.emptyState = buildEmptyStateCta(s);
    s.emptyState.hidden = true;
    body.appendChild(s.emptyState);
  }

  header.addEventListener('click', () => {
    s.open = !s.open;
    header.classList.toggle('open', s.open);
    body.classList.toggle('open', s.open);
    syncURL();
  });

  el.append(header, body);

  // Build the underlying table once the section's DOM is in place.
  buildTable(s, searchInput);
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

function toggleSection(s, open) {
  if (!s.header || !s.body) return;
  s.open = open;
  s.header.classList.toggle('open', open);
  s.body.classList.toggle('open', open);
}

/**
 * Build (or rebuild) a section's TableWrapper.  Reads current URL state
 * from `location.search` so sort/search round-trip across rebuilds.
 */
function buildTable(s, searchInput) {
  if (s.table) { s.table.destroy(); s.table = null; }
  if (!s.host || !itemsData) return;

  const u = new URLSearchParams(location.search);
  const isFustaia = (s === sections.f);
  const sortColParam = isFustaia ? 'fsc' : 'csc';
  const sortOrdParam = isFustaia ? 'fso' : 'cso';
  const searchParam  = isFustaia ? 'ff'  : 'cf';

  const sortCol = u.get(sortColParam) || S.COL_YEAR_PLANNED;
  const sortAsc = u.get(sortOrdParam) !== '1';
  const searchText = u.get(searchParam) || '';

  const modify = canModify();
  s.table = new TableWrapper({
    container: s.host,
    digest: itemsData,
    columnDefs: buildItemColumnDefs(itemsData.columns, s.hiddenCols),
    inlineToolbar: false,
    canModify: modify,
    actions: modify ? {
      onEdit: (rowId) => showEditItemModal(rowId),
      onDelete: (rowId) => confirmDeleteItem(rowId),
    } : {},
    sort: { column: sortCol, ascending: sortAsc },
    searchText,
    csvFilename: s.csvFilename,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
  s.table.wireSearchInput(searchInput);
  applyPlanFilter(s);
}

function applyPlanFilter(s) {
  if (!itemsData) return;
  const planCol = itemsData.columns.indexOf(S.COL_HARVEST_PLAN);
  const typeCol = itemsData.columns.indexOf(S.COL_TYPE);
  if (s.table) {
    s.table.setExternalFilter((row) => {
      if (activePlanId != null && row[planCol] !== activePlanId) return false;
      return s.typeMatcher(row[typeCol]);
    });
  }
  updateEmptyState(s);
}

/** Toggle between the table and the empty-state CTA based on item count. */
function updateEmptyState(s) {
  if (!s.emptyState || !itemsData) return;
  const planCol = itemsData.columns.indexOf(S.COL_HARVEST_PLAN);
  const typeCol = itemsData.columns.indexOf(S.COL_TYPE);
  const hasItems = activePlanId != null && itemsData.rows.some(r =>
    r[planCol] === activePlanId && s.typeMatcher(r[typeCol]),
  );
  s.emptyState.hidden = hasItems;
  if (s.toolbar) s.toolbar.hidden = !hasItems;
  if (s.host) s.host.hidden = !hasItems;
  if (s.actionAdd) s.actionAdd.hidden = !hasItems;
}

function buildEmptyStateCta(s) {
  const wrap = document.createElement('div');
  wrap.className = 'pdt-empty-state';

  const msg = document.createElement('div');
  msg.className = 'pdt-empty-state-message';
  msg.textContent = S.PDT_SECTION_EMPTY;
  wrap.appendChild(msg);

  const actions = document.createElement('div');
  actions.className = 'pdt-empty-state-actions';

  const importCal = document.createElement('button');
  importCal.type = 'button';
  importCal.className = 'btn btn-primary';
  importCal.textContent = S.EDIT_PLAN_TAB_CALENDAR;
  importCal.addEventListener('click',
    () => openEditPlanModal(EDIT_PLAN_TAB_CALENDAR));
  actions.appendChild(importCal);

  const addManual = document.createElement('button');
  addManual.type = 'button';
  addManual.className = 'btn';
  addManual.textContent = S.PDT_EMPTY_STATE_ADD_MANUAL;
  addManual.addEventListener('click', () => showAddItemModal(s));
  actions.appendChild(addManual);

  wrap.appendChild(actions);
  return wrap;
}

function buildItemColumnDefs(columns, hiddenCols) {
  const hidden = new Set([VERSION, ...hiddenCols]);
  const defs = {};
  for (const name of columns) {
    if (name === ROW_ID) continue;
    if (hidden.has(name)) { defs[name] = { label: name, hidden: true }; continue; }
    defs[name] = ITEM_COL_DEFS[name] || { label: name };
  }
  return defs;
}

const ITEM_COL_DEFS = (() => {
  const fNum = (v) => (typeof v === 'number' ? v.toFixed(2).replace('.', ',') : (v == null || v === '' ? '' : v));
  const fArea = (v) => (typeof v === 'number' ? v.toFixed(2).replace('.', ',') : (v == null || v === '' ? '' : v));
  const fInt = (v) => (v == null || v === '' ? '' : String(v));
  return {
    [S.COL_YEAR_PLANNED]:         { label: S.COL_YEAR_PLANNED, type: 'number', width: '85px', formatter: fInt },
    [S.COL_YEAR_ACTUAL]:          { label: S.COL_YEAR_ACTUAL,  type: 'number', width: '85px', formatter: fInt },
    [S.COL_COMPRESA]:             { label: S.COL_COMPRESA,     width: '110px' },
    [S.COL_PARCEL]:               { label: S.COL_PARCEL,       width: '90px' },
    [S.COL_STATE]:                { label: S.COL_STATE,        width: '110px' },
    [S.COL_NOTE]:                 { label: S.COL_NOTE,         width: '160px' },
    [S.COL_VOLUME_PLANNED]:       { label: S.COL_VOLUME_PLANNED, type: 'number', width: '95px', formatter: fNum },
    [S.COL_VOLUME_MARKED]:        { label: S.COL_VOLUME_MARKED,  type: 'number', width: '95px', formatter: fNum },
    [S.COL_VOLUME_ACTUAL]:        { label: S.COL_VOLUME_ACTUAL,  type: 'number', width: '95px', formatter: fNum },
    [S.COL_INTERVENTION_AREA_HA]: { label: S.COL_INTERVENTION_AREA_HA, type: 'number', width: '110px', formatter: fArea },
    [S.COL_PARCEL_AREA_HA]:       { label: S.COL_PARCEL_AREA_HA,       type: 'number', width: '110px', formatter: fArea },
    [S.COL_TURNO_A]:              { label: S.COL_TURNO_A,             type: 'number', width: '75px',  formatter: fInt },
  };
})();

function buildNewPlanButton() {
  const addBtn = document.createElement('button');
  addBtn.type = 'button';
  addBtn.className = 'btn btn-primary';
  addBtn.textContent = S.NEW_PLAN_LABEL;
  addBtn.addEventListener('click', () => onNewPlan());
  return addBtn;
}

function appendEditDeleteIcons(host, { onEdit, onDelete }) {
  const edit = document.createElement('span');
  edit.className = 'action-icon action-edit pdt-pulldown-icon';
  edit.title = S.ACTION_EDIT;
  edit.textContent = '✎';
  edit.setAttribute('role', 'button');
  edit.addEventListener('click', onEdit);
  host.appendChild(edit);

  const del = document.createElement('span');
  del.className = 'action-icon action-delete pdt-pulldown-icon';
  del.title = S.ACTION_DELETE;
  del.textContent = '\u{1F5D1}\u{FE0E}';
  del.setAttribute('role', 'button');
  del.addEventListener('click', onDelete);
  host.appendChild(del);
}

// ---------------------------------------------------------------------------
// Dangerous-delete flow.
//
// `showDangerousDeleteModal` is the shared confirmation-modal template
// parameterised by object type — exported so per-item deletion can
// reuse it.  The "Elimina" button stays disabled until the user clicks
// "Esporta CSV"; that forced-download step makes sure the operator has
// a local backup before destruction.
// ---------------------------------------------------------------------------

function onDeletePlan() {
  if (activePlanId == null) return;
  const row = planRow(activePlanId);
  if (!row) return;

  // Surface the server's "must all be planned" gate up front so the
  // user doesn't have to walk through the confirm modal.
  if (planHasActiveItems(activePlanId)) {
    showError(S.ERR_PLAN_HAS_ACTIVE_ITEMS);
    return;
  }

  const planName = row[plansData.columns.indexOf(S.COL_NAME)];
  showDangerousDeleteModal({
    title: S.DELETE_PLAN_TITLE,
    warning: S.DELETE_PLAN_WARNING.replace('{name}', planName),
    onExportCSV: () => downloadPlanExport(activePlanId),
    onDelete: () => doDeletePlan(),
  });
}

/** True if any item in this plan has advanced past `planned`. */
function planHasActiveItems(planId) {
  if (!itemsData) return false;
  const c = itemsData.columns;
  const planCol = c.indexOf(S.COL_HARVEST_PLAN);
  const stateCol = c.indexOf(S.COL_STATE);
  const planned = S.STATE_PLANNED;
  return itemsData.rows.some(r =>
    r[planCol] === planId && r[stateCol] !== planned,
  );
}

async function doDeletePlan() {
  const id = activePlanId;
  try {
    const { data, status } = await postJSON(`${PLAN_DELETE_URL}${id}/`, {});
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    cache.removeRow(PLANS_ID, id);
    activePlanId = null;
    plansData = cache.get(PLANS_ID);
    // Server also marks the items + regressions digests stale because
    // the plan cascade-deletes them.  Refresh those caches so the page
    // doesn't keep rendering rows for a plan that no longer exists.
    await Promise.all([refreshItems(), refreshRegressions()]);
    onPlansUpdate();
    syncURL();
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

// ---------------------------------------------------------------------------
// Add-item modal (Nuovo intervento) — fetches the server-rendered
// fragment, injects it into a modal, and wires the region / parcel /
// coppice cascade.  Reused by both fustaia and ceduo sections; the
// section's `kind` filters the parcel pulldown to the matching family.
// ---------------------------------------------------------------------------

async function showAddItemModal(section) {
  if (activePlanId == null) return;
  let payload;
  try {
    const url = `${ITEM_FORM_URL}?plan=${activePlanId}`;
    const result = await fetchJSON(url);
    payload = result.data;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }
  if (!payload?.html) { showError(S.ERROR_GENERIC); return; }
  openItemFormModal(payload.html, section.kind);
}

/**
 * Inline-edit affordance for a calendar row (pencil icon).  Opens the
 * existing item form pre-populated with the item's fields; the form's
 * row_id hidden carries the id so the save endpoint treats this as an
 * update.  When PT-60 lands, the per-row looking-glass becomes the
 * primary affordance for full view/edit (with marks/prelievi); this
 * pencil stays as a quick-metadata-edit shortcut.
 */
async function showEditItemModal(itemId) {
  if (!itemsData) return;
  const c = itemsData.columns;
  const row = itemsData.rows.find(r => r[c.indexOf(ROW_ID)] === itemId);
  if (!row) return;
  const tipo = row[c.indexOf(S.COL_TYPE)];
  const kind = tipo === S.TYPE_CEDUO ? 'ceduo' : 'fustaia';

  let payload;
  try {
    const result = await fetchJSON(`${ITEM_FORM_URL}${itemId}/`);
    payload = result.data;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }
  if (!payload?.html) { showError(S.ERROR_GENERIC); return; }
  openItemFormModal(payload.html, kind);
}

function openItemFormModal(html, kind) {
  const frag = parseHTMLFragment(html);
  const form = frag.querySelector('form');
  if (!form) { showError(S.ERROR_GENERIC); return; }
  injectNonce(form);
  wireItemForm(form, kind);
  attachItemSubmit(form, kind);
  wireCancelButtons(form, dismissModal);

  const outer = document.createDocumentFragment();
  outer.appendChild(frag);
  showModal(outer);
}

function replaceItemFormInModal(html, kind) {
  const frag = parseHTMLFragment(html);
  const form = frag.querySelector('form');
  if (!form) return;
  injectNonce(form);
  wireItemForm(form, kind);
  attachItemSubmit(form, kind);
  wireCancelButtons(form, dismissModal);
  const modalEl = document.querySelector('#modal-container .modal');
  if (modalEl) {
    modalEl.replaceChildren();
    modalEl.appendChild(frag);
  }
}

/** DOMParser → DocumentFragment for safe HTML fragment injection. */
function parseHTMLFragment(html) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const frag = document.createDocumentFragment();
  for (const node of [...doc.body.childNodes]) frag.appendChild(node);
  return frag;
}

function attachItemSubmit(form, kind) {
  interceptSubmit(form, ITEM_SAVE_URL, {
    onSuccess: (data) => {
      if (data.record) cache.updateRow(ITEMS_ID, data.row_id, data.record);
      itemsData = cache.get(ITEMS_ID);
      for (const k of SECTION_KEYS) {
        sections[k].table?.setData(itemsData);
        updateEmptyState(sections[k]);
      }
      dismissModal();
    },
    onConflict(data) {
      if (data.html) replaceItemFormInModal(data.html, kind);
    },
    onValidationError(data) {
      if (data.html) replaceItemFormInModal(data.html, kind);
    },
  });
}

function injectNonce(form) {
  let inp = form.querySelector('input[name="nonce"]');
  if (!inp) {
    inp = document.createElement('input');
    inp.type = 'hidden';
    inp.name = 'nonce';
    form.appendChild(inp);
  }
  inp.value = crypto.randomUUID();
}

/**
 * Wire the region/parcel cascade and the fustaia/ceduo dispatch on the
 * server-rendered item form.  The form's parcel <option>s carry
 * `data-region` and `data-coppice`; we hide options that don't match
 * the selected region (cascade) or don't fit the section's family
 * (kind-filter), and toggle the volume vs intervention-area inputs to
 * match the resolved family.
 */
function wireItemForm(form, kind) {
  const regionSel = form.querySelector('#id_item_region');
  const parcelSel = form.querySelector('#id_item_parcel');
  if (!regionSel || !parcelSel) return;

  const allOpts = [...parcelSel.querySelectorAll('option[data-region]')];
  const blankOpt = parcelSel.querySelector('option:not([data-region])');

  function familyMatches(opt) {
    if (kind === 'ceduo')   return opt.dataset.coppice === '1';
    if (kind === 'fustaia') return opt.dataset.coppice === '0';
    return true;
  }

  function refresh() {
    const rid = regionSel.value;
    const current = parcelSel.value;
    parcelSel.replaceChildren();
    if (blankOpt) parcelSel.appendChild(blankOpt);
    for (const opt of allOpts) {
      if (opt.dataset.region === rid && familyMatches(opt)) {
        parcelSel.appendChild(opt);
      }
    }
    if ([...parcelSel.options].some(o => o.value === current)) {
      parcelSel.value = current;
    }
    onParcelChange();
  }

  function onParcelChange() {
    const opt = parcelSel.options[parcelSel.selectedIndex];
    const isCoppice = opt?.dataset.coppice === '1'
                   || (parcelSel.value === '' && kind === 'ceduo');
    toggleField(form, '#id_volume_planned', !isCoppice);
    toggleField(form, '#id_intervention_area', isCoppice);
  }

  regionSel.addEventListener('change', refresh);
  parcelSel.addEventListener('change', onParcelChange);
  refresh();
}

function toggleField(form, selector, visible) {
  const inp = form.querySelector(selector);
  if (!inp) return;
  const group = inp.closest('.form-group');
  if (group) group.hidden = !visible;
  inp.disabled = !visible;
}

// ---------------------------------------------------------------------------
// Per-item dangerous-delete flow.  Server's `state = planned` gate is
// the source of truth; we surface it client-side so the operator gets
// an immediate explanation instead of a generic 400.
// ---------------------------------------------------------------------------

function confirmDeleteItem(itemId) {
  if (!itemsData) return;
  const c = itemsData.columns;
  const row = itemsData.rows.find(r => r[c.indexOf(ROW_ID)] === itemId);
  if (!row) return;

  if (row[c.indexOf(S.COL_STATE)] !== S.STATE_PLANNED) {
    showError(S.ERR_PLAN_ITEM_STATE_NOT_PLANNED);
    return;
  }

  const compresa = row[c.indexOf(S.COL_COMPRESA)];
  const parcel = row[c.indexOf(S.COL_PARCEL)];
  const year = row[c.indexOf(S.COL_YEAR_PLANNED)];

  // Per-item delete is only allowed when state == planned, in which
  // case the item has no marks / harvests / transitions (DB-level
  // PROTECT blocks otherwise).  Nothing to back up → skip the
  // forced-download step.
  showDangerousDeleteModal({
    title: S.DELETE_ITEM_TITLE,
    warning: S.DELETE_ITEM_WARNING
      .replace('{year}', year)
      .replace('{region}', compresa)
      .replace('{parcel}', parcel || ''),
    onDelete: () => doDeleteItem(itemId),
  });
}

async function doDeleteItem(itemId) {
  try {
    const { data, status } = await postJSON(`${ITEM_DELETE_URL}${itemId}/`, {});
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    cache.removeRow(ITEMS_ID, itemId);
    itemsData = cache.get(ITEMS_ID);
    for (const k of SECTION_KEYS) {
      sections[k].table?.setData(itemsData);
      updateEmptyState(sections[k]);
    }
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function downloadItemExport(itemId) {
  const a = document.createElement('a');
  a.href = `${ITEM_EXPORT_URL}${itemId}/`;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/**
 * Shared dangerous-delete modal.  When `onExportCSV` is provided, the
 * modal includes a forced-download step (Esporta CSV must be clicked
 * before Elimina enables) — used for plan-level delete where the
 * planning calendar is worth backing up.  When `onExportCSV` is
 * omitted (per-item delete in PLANNED state with no deps), the
 * export step is skipped: Elimina is enabled immediately.
 */
export function showDangerousDeleteModal({ title, warning, onExportCSV, onDelete }) {
  const frag = document.createDocumentFragment();

  const h = document.createElement('h2');
  h.textContent = title;
  h.className = 'cascade-confirm-title';
  frag.appendChild(h);

  const warn = document.createElement('p');
  warn.className = 'cascade-confirm-warning';
  warn.textContent = warning;
  frag.appendChild(warn);

  if (onExportCSV) {
    const need = document.createElement('p');
    need.textContent = S.CASCADE_EXPORT_REQUIRED;
    frag.appendChild(need);
  }

  const actions = document.createElement('div');
  actions.className = 'form-actions';

  const cancel = document.createElement('button');
  cancel.className = 'btn';
  cancel.dataset.action = 'cancel';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', dismissModal);

  const delBtn = document.createElement('button');
  delBtn.className = 'btn btn-primary cascade-delete-btn';
  delBtn.textContent = S.ACTION_DELETE;
  delBtn.disabled = !!onExportCSV;

  delBtn.addEventListener('click', () => {
    dismissModal();
    onDelete();
  });

  if (onExportCSV) {
    const exportBtn = document.createElement('button');
    exportBtn.className = 'btn btn-primary';
    exportBtn.textContent = S.EXPORT_CSV;
    exportBtn.addEventListener('click', () => {
      onExportCSV();
      delBtn.disabled = false;
    });
    actions.append(cancel, exportBtn, delBtn);
  } else {
    actions.append(cancel, delBtn);
  }
  frag.appendChild(actions);
  showModal(frag);
}

// ---------------------------------------------------------------------------
// Modifica piano modal (pencil) — three tabs.  Identity (name +
// description + year range) lives under "Dettagli"; the other two
// tabs upsert CSV rows into the active plan via the plan_csv_import
// endpoint with `harvest_plan_id` set.
// ---------------------------------------------------------------------------

const EDIT_PLAN_TAB_DETAILS    = 'details';
const EDIT_PLAN_TAB_CALENDAR   = 'calendar';
const EDIT_PLAN_TAB_REGRESSION = 'regression';

function onEditPlan() {
  openEditPlanModal(EDIT_PLAN_TAB_DETAILS);
}

/**
 * Open the Modifica piano modal on a specific tab.  Exported so the
 * empty-state CTA in the calendar sections can land directly on the
 * matching import tab.
 */
export function openEditPlanModal(initialTab) {
  if (activePlanId == null) return;
  const row = planRow(activePlanId);
  if (!row) return;
  const c = plansData.columns;
  const current = {
    name: row[c.indexOf(S.COL_NAME)] || '',
    description: row[c.indexOf(S.COL_DESCRIPTION)] || '',
    year_start: row[c.indexOf(S.COL_YEAR_START)],
    year_end: row[c.indexOf(S.COL_YEAR_END)],
    version: row[c.indexOf(VERSION)] ?? 1,
  };

  const frag = document.createDocumentFragment();
  const card = document.createElement('div');
  card.className = 'form-card pdt-edit-plan-card';
  frag.appendChild(card);

  const h = document.createElement('h2');
  h.textContent = S.EDIT_PLAN_TITLE;
  card.appendChild(h);

  const tabs = document.createElement('div');
  tabs.className = 'pdt-path-tabs';
  card.appendChild(tabs);

  const bodies = document.createElement('div');
  bodies.className = 'pdt-path-bodies';
  card.appendChild(bodies);

  const tabDefs = [
    {
      id: EDIT_PLAN_TAB_DETAILS, label: S.EDIT_PLAN_TAB_DETAILS,
      build: (host) => buildEditDetailsForm(host, current),
    },
    {
      id: EDIT_PLAN_TAB_CALENDAR, label: S.EDIT_PLAN_TAB_CALENDAR,
      build: (host) => buildExistingPlanCalendarImport(host),
    },
    {
      id: EDIT_PLAN_TAB_REGRESSION, label: S.EDIT_PLAN_TAB_REGRESSION,
      build: (host) => buildExistingPlanRegressionImport(host),
    },
  ];

  const bodyEls = {};
  for (const t of tabDefs) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pdt-path-tab';
    btn.dataset.path = t.id;
    btn.textContent = t.label;
    btn.addEventListener('click', () => switchTab(t.id));
    tabs.appendChild(btn);

    const body = document.createElement('div');
    body.className = 'pdt-path-body';
    body.dataset.path = t.id;
    t.build(body);
    bodies.appendChild(body);
    bodyEls[t.id] = body;
  }

  function switchTab(id) {
    for (const t of tabs.querySelectorAll('.pdt-path-tab')) {
      t.classList.toggle('active', t.dataset.path === id);
    }
    for (const [k, b] of Object.entries(bodyEls)) {
      b.classList.toggle('active', k === id);
    }
  }

  showModal(frag);
  switchTab(initialTab || EDIT_PLAN_TAB_DETAILS);
}

/** Dettagli tab — name, year range, description.  JSON POST to /plan/save/. */
function buildEditDetailsForm(host, current) {
  const form = document.createElement('form');
  form.className = 'pdt-plan-form';
  host.appendChild(form);

  const nameRow = mkRow(form);
  mkInput(nameRow, {
    id: 'pdt-edit-name', name: 'name', label: S.LABEL_PLAN_NAME,
    type: 'text', required: true, maxLength: 100,
    value: current.name,
  });

  const yearRow = mkRow(form, 'narrow');
  mkInput(yearRow, {
    id: 'pdt-edit-y1', name: 'year_start', label: S.COL_YEAR_START,
    type: 'number', required: true, min: 1900, max: 2200,
    value: current.year_start,
  });
  mkInput(yearRow, {
    id: 'pdt-edit-y2', name: 'year_end', label: S.COL_YEAR_END,
    type: 'number', required: true, min: 1900, max: 2200,
    value: current.year_end,
  });

  const descRow = mkRow(form);
  mkTextarea(descRow, {
    id: 'pdt-edit-desc', name: 'description',
    label: S.LABEL_PLAN_DESCRIPTION, rows: 3, value: current.description,
  });

  mkFormActions(form, { onCancel: dismissModal, submitLabel: S.SAVE });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const body = {
      row_id: String(activePlanId),
      version: String(current.version),
      name: (fd.get('name') || '').toString().trim(),
      description: (fd.get('description') || '').toString().trim(),
      year_start: parseInt(fd.get('year_start'), 10),
      year_end: parseInt(fd.get('year_end'), 10),
      nonce: crypto.randomUUID(),
    };
    if (!body.name) { showError(S.ERR_PLAN_NAME_REQUIRED); return; }
    if (body.year_end < body.year_start) {
      showError(S.ERR_PLAN_YEAR_RANGE); return;
    }
    try {
      const { data, status } = await postJSON(PLAN_SAVE_URL, body);
      if (status !== 200) {
        showError(data?.message || S.ERROR_GENERIC);
        return;
      }
      if (data.record) cache.updateRow(PLANS_ID, data.row_id, data.record);
      plansData = cache.get(PLANS_ID);
      onPlansUpdate();
      dismissModal();
    } catch {
      showError(S.ERROR_NETWORK);
    }
  });
}

// ---------------------------------------------------------------------------
// Nuovo piano modal — name + description only.  An empty plan starts
// with year_start = year_end = current civil year; the range widens
// later via pencil edit or implicitly on CSV import (PT-5R-3).
// Identity (name) is deliberately decoupled from content (calendar /
// equations) — those land via the pencil modal.
// ---------------------------------------------------------------------------

function onNewPlan() {
  const frag = document.createDocumentFragment();
  const card = document.createElement('div');
  card.className = 'form-card';
  frag.appendChild(card);

  const h = document.createElement('h2');
  h.textContent = S.NEW_PLAN_TITLE;
  card.appendChild(h);

  const form = document.createElement('form');
  form.className = 'pdt-plan-form';
  card.appendChild(form);

  const nameRow = mkRow(form);
  mkInput(nameRow, {
    id: 'pdt-new-plan-name', name: 'name', label: S.LABEL_PLAN_NAME,
    type: 'text', required: true, maxLength: 100,
  });

  const descRow = mkRow(form);
  mkTextarea(descRow, {
    id: 'pdt-new-plan-desc', name: 'description',
    label: S.LABEL_PLAN_DESCRIPTION, rows: 3,
  });

  mkFormActions(form, { onCancel: dismissModal, submitLabel: S.SAVE });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const today = new Date().getFullYear();
    const body = {
      name: (fd.get('name') || '').toString().trim(),
      description: (fd.get('description') || '').toString().trim(),
      year_start: today,
      year_end: today,
      nonce: crypto.randomUUID(),
    };
    if (!body.name) { showError(S.ERR_PLAN_NAME_REQUIRED); return; }
    try {
      const { data, status } = await postJSON(PLAN_SAVE_URL, body);
      if (status !== 200) {
        showError(data?.message || S.ERROR_GENERIC);
        return;
      }
      if (data.record) cache.updateRow(PLANS_ID, data.row_id, data.record);
      plansData = cache.get(PLANS_ID);
      activePlanId = data.row_id;
      onPlansUpdate();
      syncURL();
      dismissModal();
    } catch {
      showError(S.ERROR_NETWORK);
    }
  });

  showModal(frag);
}

/**
 * Importa calendario da CSV — tab body in the pencil modal.  Single
 * file input + a "Ceduo" checkbox: unchecked sends as `fustaia_file`,
 * checked as `ceduo_file`.  POSTs to /plan/import-csv/ with
 * `harvest_plan_id` set so the server upserts rows into the active
 * plan (dedup key: parcel + year_planned).
 */
function buildExistingPlanCalendarImport(host) {
  const form = document.createElement('form');
  form.className = 'pdt-plan-form';
  host.appendChild(form);

  const kindRow = document.createElement('div');
  kindRow.className = 'form-row';
  form.appendChild(kindRow);
  const kindLabel = document.createElement('label');
  kindLabel.className = 'pdt-checkbox-label';
  const ceduoCb = document.createElement('input');
  ceduoCb.type = 'checkbox';
  ceduoCb.name = 'is_ceduo';
  kindLabel.append(ceduoCb, ' ' + S.EDIT_PLAN_CHECKBOX_CEDUO);
  kindRow.appendChild(kindLabel);

  const fileInput = mkFileInput(form, {
    id: 'pdt-edit-cal-file', label: S.LABEL_CSV_FILE,
  });

  const statusBox = mkStatusBox(form);
  const errorsBox = mkErrorsBox(form);

  mkFormActions(form, {
    onCancel: dismissModal,
    submitLabel: S.IMPORT_LABEL,
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (activePlanId == null) return;
    const file = fileInput.files[0];
    if (!file) { showError(S.ERR_CSV_FILE_REQUIRED); return; }

    const fd = new FormData();
    fd.append('harvest_plan_id', String(activePlanId));
    fd.append(ceduoCb.checked ? 'ceduo_file' : 'fustaia_file', file);
    fd.append('nonce', crypto.randomUUID());

    await submitCsvImport(form, fd, statusBox, errorsBox);
  });
}

/**
 * Importa equazioni da CSV — tab body in the pencil modal.  Same
 * /plan/import-csv/ endpoint; the regression file lands under
 * `regression_file` and rows are upserted on (region, species) under
 * the active plan.
 */
function buildExistingPlanRegressionImport(host) {
  const form = document.createElement('form');
  form.className = 'pdt-plan-form';
  host.appendChild(form);

  const fileInput = mkFileInput(form, {
    id: 'pdt-edit-reg-file', label: S.LABEL_CSV_FILE,
  });

  const statusBox = mkStatusBox(form);
  const errorsBox = mkErrorsBox(form);

  mkFormActions(form, {
    onCancel: dismissModal,
    submitLabel: S.IMPORT_LABEL,
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (activePlanId == null) return;
    const file = fileInput.files[0];
    if (!file) { showError(S.ERR_CSV_FILE_REQUIRED); return; }

    const fd = new FormData();
    fd.append('harvest_plan_id', String(activePlanId));
    fd.append('regression_file', file);
    fd.append('nonce', crypto.randomUUID());

    await submitCsvImport(form, fd, statusBox, errorsBox);
  });
}

/**
 * Submit a multipart POST to /plan/import-csv/ and, on success, refresh
 * the affected digests + land on the newly-imported plan.
 */
async function submitCsvImport(form, fd, statusBox, errorsBox) {
  const btn = form.querySelector('button[type="submit"]');
  if (btn) btn.disabled = true;
  errorsBox.hidden = true;
  errorsBox.replaceChildren();
  statusBox.textContent = S.CSV_IMPORT_IN_PROGRESS;
  statusBox.hidden = false;
  try {
    const { data, status } = await postFormData(PLAN_IMPORT_CSV_URL, fd);
    if (status === 200) {
      await Promise.all([refreshPlans(), refreshItems(), refreshRegressions()]);
      activePlanId = data.row_id;
      onPlansUpdate();
      syncURL();
      dismissModal();
      return;
    }
    if (data?.errors?.length) {
      renderCsvErrors(errorsBox, data.errors);
    } else {
      showError(data?.message || S.ERROR_GENERIC);
    }
  } catch {
    showError(S.ERROR_NETWORK);
  } finally {
    if (btn) btn.disabled = false;
    statusBox.hidden = true;
  }
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
    more.textContent = `… +${errors.length - 50}`;
    ul.appendChild(more);
  }
  box.appendChild(ul);
  box.hidden = false;
}

async function refreshPlans() {
  try { await cache.load(PLANS_ID); plansData = cache.get(PLANS_ID); } catch {}
}
async function refreshItems() {
  try { await cache.load(ITEMS_ID); itemsData = cache.get(ITEMS_ID); } catch {}
}
async function refreshRegressions() {
  try { await cache.load(REGRESSIONS_ID); regressionsData = cache.get(REGRESSIONS_ID); } catch {}
}

// ---------------------------------------------------------------------------
// Small DOM helpers (form scaffolding)
// ---------------------------------------------------------------------------

function mkRow(host, modifier) {
  const row = document.createElement('div');
  row.className = 'form-row' + (modifier ? ' ' + modifier : '');
  host.appendChild(row);
  return row;
}

function mkInput(host, opts) {
  const group = document.createElement('div');
  group.className = 'form-group';
  host.appendChild(group);
  const lbl = document.createElement('label');
  lbl.textContent = opts.label;
  lbl.htmlFor = opts.id;
  group.appendChild(lbl);
  const inp = document.createElement('input');
  inp.type = opts.type || 'text';
  inp.name = opts.name;
  inp.id = opts.id;
  if (opts.required) inp.required = true;
  if (opts.maxLength != null) inp.maxLength = opts.maxLength;
  if (opts.min != null) inp.min = opts.min;
  if (opts.max != null) inp.max = opts.max;
  if (opts.value != null) inp.value = opts.value;
  group.appendChild(inp);
  return inp;
}

function mkTextarea(host, opts) {
  const group = document.createElement('div');
  group.className = 'form-group';
  host.appendChild(group);
  const lbl = document.createElement('label');
  lbl.textContent = opts.label;
  lbl.htmlFor = opts.id;
  group.appendChild(lbl);
  const ta = document.createElement('textarea');
  ta.name = opts.name;
  ta.id = opts.id;
  ta.rows = opts.rows || 3;
  if (opts.value != null) ta.value = opts.value;
  group.appendChild(ta);
  return ta;
}

function mkFileInput(form, opts) {
  const row = mkRow(form);
  const group = document.createElement('div');
  group.className = 'form-group';
  row.appendChild(group);
  const lbl = document.createElement('label');
  lbl.textContent = opts.label;
  group.appendChild(lbl);
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = '.csv,text/csv';
  inp.required = true;
  lbl.htmlFor = inp.id = opts.id;
  group.appendChild(inp);
  return inp;
}

function mkStatusBox(form) {
  const box = document.createElement('div');
  box.className = 'csv-import-status';
  box.hidden = true;
  form.appendChild(box);
  return box;
}

function mkErrorsBox(form) {
  const box = document.createElement('div');
  box.className = 'csv-import-errors';
  box.hidden = true;
  form.appendChild(box);
  return box;
}

function mkFormActions(form, { onCancel, submitLabel }) {
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'btn';
  cancel.dataset.action = 'cancel';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', onCancel);
  const submit = document.createElement('button');
  submit.type = 'submit';
  submit.className = 'btn btn-primary';
  submit.textContent = submitLabel;
  actions.append(cancel, submit);
  form.appendChild(actions);
  return actions;
}

// ---------------------------------------------------------------------------
// Plan selection
// ---------------------------------------------------------------------------

function setActivePlan(planId) {
  if (planId == null || isNaN(planId)) {
    activePlanId = null;
    renderDescription();
    refreshSectionFilters();
    return;
  }
  activePlanId = planId;
  if (planSelectEl && planSelectEl.value !== String(planId)) {
    planSelectEl.value = String(planId);
  }
  renderDescription();
  refreshSectionFilters();
}

function refreshSectionFilters() {
  for (const k of SECTION_KEYS) applyPlanFilter(sections[k]);
}

function renderDescription() {
  if (!descriptionEl) return;
  descriptionEl.replaceChildren();
  if (activePlanId == null) return;
  const row = planRow(activePlanId);
  if (!row) return;
  const c = plansData.columns;
  const desc = row[c.indexOf(S.COL_DESCRIPTION)] || '';
  const yStart = row[c.indexOf(S.COL_YEAR_START)];
  const yEnd = row[c.indexOf(S.COL_YEAR_END)];
  const meta = document.createElement('div');
  meta.className = 'pdt-description-meta';
  meta.textContent = `${yStart}–${yEnd}`;
  descriptionEl.appendChild(meta);
  if (desc) {
    const d = document.createElement('div');
    d.className = 'pdt-description-text';
    d.textContent = desc;
    descriptionEl.appendChild(d);
  }
}

function onPlansUpdate() {
  plansData = cache.get(PLANS_ID);
  const el = document.getElementById('content');
  if (!el) return;
  buildPageShell(el);
  if (activePlanId != null && !planRow(activePlanId)) activePlanId = null;
  if (activePlanId == null && plansData?.rows.length) {
    activePlanId = plansData.rows[0][plansData.columns.indexOf(ROW_ID)];
  }
  setActivePlan(activePlanId);
}

function planRow(planId) {
  if (!plansData) return null;
  const c = plansData.columns;
  return plansData.rows.find(r => r[c.indexOf(ROW_ID)] === planId);
}

// ---------------------------------------------------------------------------
// Plan export (zip download)
// ---------------------------------------------------------------------------

function downloadPlanExport(planId) {
  // Plain anchor click: the export endpoint sets Content-Disposition
  // so the browser downloads the zip without leaving the SPA.
  const a = document.createElement('a');
  a.href = `${PLAN_EXPORT_URL}${planId}/`;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------------------------------------------------------------------------
// CSS lifecycle
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
