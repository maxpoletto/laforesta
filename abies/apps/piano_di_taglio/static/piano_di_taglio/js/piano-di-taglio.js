/**
 * Piano di taglio page.
 *
 * Hosts the plan-selector header (pulldown + edit / delete / export /
 * "+ Nuovo piano") and, below it, the fustaia and ceduo calendar
 * sections + the bookmarkable view/edit-item modal.
 *
 * URL params: see docs/page-piano-di-taglio.md.
 */

import * as cache from '../../base/js/cache.js';
import * as router from '../../base/js/router.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  show as showModal, showError, dismiss as dismissModal,
} from '../../base/js/modals.js';
import { fetchJSON, postJSON, postFormData } from '../../base/js/api.js';
import {
  fetchModalForm, interceptSubmit, wireCancelButtons, showFormError,
} from '../../base/js/forms.js';
import {
  mkRow, mkInput, mkTextarea, mkFileInput, mkFormActions,
  mkStatusBox, mkErrorsBox, renderCsvErrors,
  mkCollapsible, mkEditDeleteIcons,
} from '../../base/js/form-widgets.js';
import {
  wireVMPreview, ID_D_CM, ID_H_M, ID_SPECIES, ID_LAT, ID_LON,
} from '../../base/js/tree-form.js';
import { mountUseLocationButton } from '../../base/js/latlng-input.js';
import * as S from '../../base/js/strings.js';
import { ROW_ID, VERSION } from '../../base/js/constants.js';
import {
  fmtDecimal1, fmtDecimal2, fmtDecimal3, fmtInt, fmtCoord,
  fmtVolume, fmtArea, fmtMass,
} from '../../base/js/format.js';

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
const ITEM_DATA_URL = '/api/piano-di-taglio/item/data/';
const ITEM_FORM_URL = '/api/piano-di-taglio/item/form/';
const ITEM_SAVE_URL = '/api/piano-di-taglio/item/save/';
const ITEM_DELETE_URL = '/api/piano-di-taglio/item/delete/';
const ITEM_EXPORT_URL = '/api/piano-di-taglio/item/export/';
const TRANSITION_SAVE_URL = '/api/piano-di-taglio/transition/save/';
const MARK_TREES_URL = '/api/piano-di-taglio/mark-trees/';
const MARK_FORM_URL = '/api/piano-di-taglio/mark/form/';
const MARK_SAVE_URL = '/api/piano-di-taglio/mark/save/';
const MARK_DELETE_URL = '/api/piano-di-taglio/mark/delete/';
const MARK_CSV_IMPORT_URL = '/api/piano-di-taglio/mark/import-csv/';

const PRELIEVI_ID = 'prelievi';
const PRELIEVI_URL = '/api/prelievi/data/';

const PAGE_PATH = '/piano-di-taglio';

cache.register(PLANS_ID, PLANS_URL);
cache.register(ITEMS_ID, ITEMS_URL);
cache.register(REGRESSIONS_ID, REGRESSIONS_URL);
cache.register(PRELIEVI_ID, PRELIEVI_URL);

// --- Page state -------------------------------------------------------------

let activePlanId = null;
let plansData = null;
let itemsData = null;
let regressionsData = null;
let unsubPlans = null;
let unsubItems = null;

let descriptionEl = null;
let planSelectEl = null;
let activeItemId = null;
let prelieviData = null;
let itemPrelieviTable = null;
let itemMarkTreesTable = null;

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
  destroyItemPrelieviTable();
  activePlanId = null;
  activeItemId = null;
  plansData = itemsData = regressionsData = prelieviData = null;
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
    i: params.i ? parseInt(params.i, 10) : null,
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

  // Item view page — i=N opens the view/edit item page.
  if (p.i != null && p.i !== activeItemId) {
    openItemView(p.i);
  } else if (p.i == null && activeItemId != null) {
    closeItemView();
  }
}

function syncURL(push = false) {
  const u = new URLSearchParams();
  if (activePlanId != null) {
    const defaultPlan = plansData?.rows[0]?.[plansData.columns.indexOf(ROW_ID)];
    if (activePlanId !== defaultPlan) u.set('p', String(activePlanId));
  }
  if (activeItemId != null) u.set('i', String(activeItemId));
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
  router.navigate(PAGE_PATH + (qs ? '?' + qs : ''), !push);
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
    mkEditDeleteIcons(left, {
      onEdit: () => onEditPlan(),
      onDelete: () => onDeletePlan(),
      iconClass: 'pdt-pulldown-icon',
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
  const [header, body] = mkCollapsible(s.title, s.open);
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
      onEdit: (rowId) => navigateToItem(rowId),
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

  // Row-click → navigate to item view page (for all users, not just writers).
  s.host.addEventListener('click', (e) => {
    if (e.target.closest('.action-icon')) return;
    const tr = e.target.closest('.sortable-table-row');
    if (!tr || !s.table?._table) return;
    const rowData = s.table._table.data[parseInt(tr.dataset.index, 10)];
    if (!rowData) return;
    navigateToItem(rowData[0]);
  });
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
    () => openEditPlanModal(EDIT_PLAN_TAB_CALENDAR,
      { ceduo: s.kind === 'ceduo' }));
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
  return {
    [S.COL_YEAR_PLANNED]:         { label: S.COL_YEAR_PLANNED, type: 'number', width: '85px', formatter: fmtInt },
    [S.COL_YEAR_ACTUAL]:          { label: S.COL_YEAR_ACTUAL,  type: 'number', width: '85px', formatter: fmtInt },
    [S.COL_COMPRESA]:             { label: S.COL_COMPRESA,     width: '110px' },
    [S.COL_PARCEL]:               { label: S.COL_PARCEL,       width: '90px' },
    [S.COL_STATE]:                { label: S.COL_STATE,        width: '110px' },
    [S.COL_NOTE]:                 { label: S.COL_NOTE,         width: '160px' },
    [S.COL_VOLUME_PLANNED]:       { label: S.COL_VOLUME_PLANNED, type: 'number', width: '95px', formatter: fmtDecimal2 },
    [S.COL_VOLUME_MARKED]:        { label: S.COL_VOLUME_MARKED,  type: 'number', width: '95px', formatter: fmtDecimal2 },
    [S.COL_VOLUME_ACTUAL]:        { label: S.COL_VOLUME_ACTUAL,  type: 'number', width: '95px', formatter: fmtDecimal2 },
    [S.COL_INTERVENTION_AREA_HA]: { label: S.COL_INTERVENTION_AREA_HA, type: 'number', width: '110px', formatter: fmtDecimal2 },
    [S.COL_PARCEL_AREA_HA]:       { label: S.COL_PARCEL_AREA_HA,       type: 'number', width: '110px', formatter: fmtDecimal2 },
    [S.COL_TURNO_A]:              { label: S.COL_TURNO_A,             type: 'number', width: '75px',  formatter: fmtInt },
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
  await fetchAndOpenItemForm(`${ITEM_FORM_URL}?plan=${activePlanId}`, section.kind);
}

async function fetchAndOpenItemForm(url, kind, opts) {
  let payload;
  try {
    const { data } = await fetchJSON(url);
    payload = data;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }
  if (!payload?.html) { showError(S.ERROR_GENERIC); return; }
  openItemFormModal(payload.html, kind, opts);
}

function openItemFormModal(html, kind, { onDone } = {}) {
  const onSuccess = onDone || (() => {});
  const onCancel = dismissModal;

  function wireAndShow(frag) {
    const form = frag.querySelector('form');
    if (!form) { showError(S.ERROR_GENERIC); return; }
    injectNonce(form);
    wireItemForm(form, kind);
    attachItemSubmit(form, kind, onSuccess);
    wireCancelButtons(form, onCancel);
    return form;
  }

  const frag = parseHTMLFragment(html);
  wireAndShow(frag);
  const outer = document.createDocumentFragment();
  outer.appendChild(frag);
  showModal(outer);
}

function replaceItemFormInModal(html, kind, onSuccess) {
  const frag = parseHTMLFragment(html);
  const form = frag.querySelector('form');
  if (!form) return;
  injectNonce(form);
  wireItemForm(form, kind);
  attachItemSubmit(form, kind, onSuccess);
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

function attachItemSubmit(form, kind, onSuccess) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    let data, status;
    try {
      ({ data, status } = await postJSON(ITEM_SAVE_URL, body));
    } catch {
      showFormError(form, S.ERROR_NETWORK);
      return;
    }
    if (status === 200) {
      if (data.record) cache.updateRow(ITEMS_ID, data.row_id, data.record);
      itemsData = cache.get(ITEMS_ID);
      for (const k of SECTION_KEYS) {
        sections[k].table?.setData(itemsData);
        updateEmptyState(sections[k]);
      }
      dismissModal();
      onSuccess();
      return;
    }
    if (data.html) {
      replaceItemFormInModal(data.html, kind, onSuccess);
    } else {
      showFormError(form, data.message || S.ERROR_GENERIC);
    }
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
export function openEditPlanModal(initialTab, { ceduo = false } = {}) {
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
  card.className = 'form-card';
  frag.appendChild(card);

  const h = document.createElement('h2');
  h.textContent = S.EDIT_PLAN_TITLE;
  card.appendChild(h);

  const tabs = document.createElement('div');
  tabs.className = 'modal-tabs';
  card.appendChild(tabs);

  const bodies = document.createElement('div');
  bodies.className = 'modal-tab-bodies';
  card.appendChild(bodies);

  const tabDefs = [
    {
      id: EDIT_PLAN_TAB_DETAILS, label: S.TAB_DETAILS,
      build: (host) => buildEditDetailsForm(host, current),
    },
    {
      id: EDIT_PLAN_TAB_CALENDAR, label: S.EDIT_PLAN_TAB_CALENDAR,
      build: (host) => buildExistingPlanCalendarImport(host, { ceduo }),
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
    btn.className = 'modal-tab';
    btn.dataset.path = t.id;
    btn.textContent = t.label;
    btn.addEventListener('click', () => switchTab(t.id));
    tabs.appendChild(btn);

    const body = document.createElement('div');
    body.className = 'modal-tab-body';
    body.dataset.path = t.id;
    t.build(body);
    bodies.appendChild(body);
    bodyEls[t.id] = body;
  }

  function switchTab(id) {
    for (const t of tabs.querySelectorAll('.modal-tab')) {
      t.classList.toggle('active', t.dataset.path === id);
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
    if (!body.name) { showFormError(form, S.ERR_PLAN_NAME_REQUIRED); return; }
    if (body.year_end < body.year_start) {
      showFormError(form, S.ERR_PLAN_YEAR_RANGE); return;
    }
    try {
      const { data, status } = await postJSON(PLAN_SAVE_URL, body);
      if (status !== 200) {
        showFormError(form, data?.message || S.ERROR_GENERIC);
        return;
      }
      if (data.record) cache.updateRow(PLANS_ID, data.row_id, data.record);
      plansData = cache.get(PLANS_ID);
      onPlansUpdate();
      dismissModal();
    } catch {
      showFormError(form, S.ERROR_NETWORK);
    }
  });
}

// ---------------------------------------------------------------------------
// Nuovo piano modal — name + description only.  An empty plan starts
// with year_start = year_end = current civil year; the range widens
// later via pencil edit or implicitly on CSV import.
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
    if (!body.name) { showFormError(form, S.ERR_PLAN_NAME_REQUIRED); return; }
    try {
      const { data, status } = await postJSON(PLAN_SAVE_URL, body);
      if (status !== 200) {
        showFormError(form, data?.message || S.ERROR_GENERIC);
        return;
      }
      if (data.record) cache.updateRow(PLANS_ID, data.row_id, data.record);
      plansData = cache.get(PLANS_ID);
      activePlanId = data.row_id;
      onPlansUpdate();
      syncURL();
      dismissModal();
    } catch {
      showFormError(form, S.ERROR_NETWORK);
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
function buildExistingPlanCalendarImport(host, { ceduo = false } = {}) {
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
  ceduoCb.checked = ceduo;
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
    if (!file) { showFormError(form, S.ERR_CSV_FILE_REQUIRED); return; }

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
    if (!file) { showFormError(form, S.ERR_CSV_FILE_REQUIRED); return; }

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
      showFormError(form, data?.message || S.ERROR_GENERIC);
    }
  } catch {
    showFormError(form, S.ERROR_NETWORK);
  } finally {
    if (btn) btn.disabled = false;
    statusBox.hidden = true;
  }
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
// View/edit-item page
//
// When i=N is set in the URL, the calendar list is replaced with a
// full-width item detail view.  Back-button behaviour is preserved
// via pushState.
// ---------------------------------------------------------------------------

function navigateToItem(itemId) {
  openItemView(itemId, true);
}

function openItemView(itemId, push = false) {
  activeItemId = itemId;
  if (push) syncURL(true);
  renderItemView(itemId);
}

function closeItemView(push = false) {
  destroyItemPrelieviTable();
  activeItemId = null;
  if (push) syncURL(true);
  const el = document.getElementById('content');
  if (el) {
    buildPageShell(el);
    setActivePlan(activePlanId);
  }
}

function destroyItemPrelieviTable() {
  if (itemPrelieviTable) {
    itemPrelieviTable.destroy();
    itemPrelieviTable = null;
  }
}

function destroyItemMarkTreesTable() {
  if (itemMarkTreesTable) {
    itemMarkTreesTable.destroy();
    itemMarkTreesTable = null;
  }
}

async function renderItemView(itemId) {
  const el = document.getElementById('content');
  if (!el) return;

  el.replaceChildren();
  const loading = document.createElement('div');
  loading.className = 'loading-overlay';
  loading.textContent = S.LOADING;
  el.appendChild(loading);

  let payload;
  try {
    const { data } = await fetchJSON(`${ITEM_DATA_URL}${itemId}/`);
    payload = data;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }
  if (!payload) { showError(S.ERROR_GENERIC); return; }

  const record = payload.record;
  const transitions = payload.transition_records || [];
  if (!record) { showError(S.ERROR_GENERIC); return; }

  el.replaceChildren();
  showItemMetadata(el, itemId, record, transitions);
}

function showItemMetadata(el, itemId, record, transitions) {
  const c = itemsData?.columns;
  if (!c) return;

  const card = document.createElement('div');
  card.className = 'pdt-item-card';

  // --- Header: title + actions + close ---
  const header = document.createElement('div');
  header.className = 'pdt-item-header';

  const left = document.createElement('div');
  left.className = 'pdt-item-header-left';

  const title = document.createElement('h2');
  title.className = 'pdt-item-title';
  title.textContent = formatItemTitle(record, c);
  left.appendChild(title);

  if (canModify()) {
    const pencil = document.createElement('span');
    pencil.className = 'pdt-pulldown-icon';
    pencil.textContent = '✎';
    pencil.title = S.ACTION_EDIT;
    pencil.addEventListener('click', () => showItemEditForm(itemId));
    left.appendChild(pencil);
  }

  const right = document.createElement('div');
  right.className = 'pdt-item-header-actions';

  const exportBtn = document.createElement('button');
  exportBtn.type = 'button';
  exportBtn.className = 'btn';
  exportBtn.textContent = S.EXPORT_CSV;
  exportBtn.addEventListener('click', () => downloadItemExport(itemId));
  right.appendChild(exportBtn);

  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'pdt-item-close';
  close.textContent = '×';
  close.title = S.DISMISS;
  close.addEventListener('click', () => closeItemView(true));
  right.appendChild(close);

  header.append(left, right);
  card.appendChild(header);

  // --- Metadata pane ---
  const meta = document.createElement('dl');
  meta.className = 'pdt-item-meta';

  const planName = lookupPlanName(record[c.indexOf(S.COL_HARVEST_PLAN)]);
  const compresa = record[c.indexOf(S.COL_COMPRESA)];
  const parcel = record[c.indexOf(S.COL_PARCEL)];
  const yearPlanned = record[c.indexOf(S.COL_YEAR_PLANNED)];
  const yearActual = record[c.indexOf(S.COL_YEAR_ACTUAL)];
  const state = record[c.indexOf(S.COL_STATE)];
  const note = record[c.indexOf(S.COL_NOTE)];
  const tipo = record[c.indexOf(S.COL_TYPE)];
  const volPlanned = record[c.indexOf(S.COL_VOLUME_PLANNED)];
  const volMarked = record[c.indexOf(S.COL_VOLUME_MARKED)];
  const volActual = record[c.indexOf(S.COL_VOLUME_ACTUAL)];
  const areaIntervention = record[c.indexOf(S.COL_INTERVENTION_AREA_HA)];
  const areaParcel = record[c.indexOf(S.COL_PARCEL_AREA_HA)];
  const turno = record[c.indexOf(S.COL_TURNO_A)];
  const isCoppice = tipo === S.TYPE_CEDUO;

  addMetaRow(meta, S.LABEL_HARVEST_PLAN, planName);
  addMetaRow(meta, S.COL_COMPRESA, compresa);
  if (parcel) addMetaRow(meta, S.COL_PARCEL, parcel);
  addMetaRow(meta, S.COL_YEAR_PLANNED, yearPlanned);
  if (yearActual) addMetaRow(meta, S.COL_YEAR_ACTUAL, yearActual);
  addMetaRow(meta, S.COL_STATE, state);
  if (!isCoppice) {
    addMetaRow(meta, S.COL_VOLUME_PLANNED, fmtVolume(volPlanned));
    addMetaRow(meta, S.COL_VOLUME_MARKED, fmtVolume(volMarked));
  } else {
    addMetaRow(meta, S.COL_INTERVENTION_AREA_HA, fmtArea(areaIntervention));
    addMetaRow(meta, S.COL_PARCEL_AREA_HA, fmtArea(areaParcel));
    addMetaRow(meta, S.COL_TURNO_A, turno != null ? String(turno) : '');
  }
  addMetaRow(meta, S.COL_VOLUME_ACTUAL, fmtVolume(volActual));
  if (note) addMetaRow(meta, S.COL_NOTE, note);

  for (const t of transitions) {
    const label = t[2] ? S.LABEL_CANTIERE_OPENED : S.LABEL_CANTIERE_CLOSED;
    const value = t[4] ? `${formatDate(t[3])} — ${t[4]}` : formatDate(t[3]);
    addMetaRow(meta, label, value);
  }

  card.appendChild(meta);

  // Apri / Chiudi cantiere buttons (writers only).
  if (canModify()) {
    const btnRow = document.createElement('div');
    btnRow.className = 'form-actions';
    let hasTransition = false;

    if (state === S.STATE_PLANNED || state === S.STATE_MARKED) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-primary';
      btn.textContent = S.LABEL_OPEN_CANTIERE;
      btn.addEventListener('click', () => showTransitionForm(itemId, true));
      btnRow.appendChild(btn);
      hasTransition = true;
    }
    if (state === S.STATE_OPEN || state === S.STATE_HARVESTING) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-primary';
      btn.textContent = S.LABEL_CLOSE_CANTIERE;
      btn.addEventListener('click', () => showTransitionForm(itemId, false));
      btnRow.appendChild(btn);
      hasTransition = true;
    }

    if (hasTransition) card.appendChild(btnRow);
  }

  // Martellata sub-section (fustaia items only — coppice skips marking).
  if (!isCoppice) {
    appendItemMarkTreesSection(card, itemId, state);
  }

  // Prelievi sub-section (visible once cantiere is open or later).
  const stateIdx = [S.STATE_OPEN, S.STATE_HARVESTING, S.STATE_CLOSED];
  if (stateIdx.includes(state)) {
    appendItemPrelieviSection(card, itemId);
  }

  el.appendChild(card);
}

function addMetaRow(dl, label, value) {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value ?? '';
  dl.append(dt, dd);
}

function formatItemTitle(record, columns) {
  const compresa = record[columns.indexOf(S.COL_COMPRESA)];
  const parcel = record[columns.indexOf(S.COL_PARCEL)];
  const year = record[columns.indexOf(S.COL_YEAR_PLANNED)];
  const planName = lookupPlanName(record[columns.indexOf(S.COL_HARVEST_PLAN)]);
  const location = parcel ? `${compresa} ${parcel}` : compresa;
  return `${S.VIEW_ITEM_TITLE} — ${planName}, ${year}, ${location}`;
}

function lookupPlanName(planId) {
  const row = planRow(planId);
  return row ? row[plansData.columns.indexOf(S.COL_NAME)] : '';
}

function formatDate(iso) {
  if (!iso) return '';
  const parts = iso.split('-');
  if (parts.length === 3) return `${parts[2]}/${parts[1]}/${parts[0]}`;
  return iso;
}

function showItemEditForm(itemId) {
  const row = itemsData?.rows.find(r =>
    r[itemsData.columns.indexOf(ROW_ID)] === itemId,
  );
  const tipo = row?.[itemsData.columns.indexOf(S.COL_TYPE)];
  const kind = tipo === S.TYPE_CEDUO ? 'ceduo' : 'fustaia';
  fetchAndOpenItemForm(`${ITEM_FORM_URL}${itemId}/`, kind, {
    onDone: () => renderItemView(itemId),
  });
}

// --- Transition form (Apri / Chiudi cantiere) ---

function showTransitionForm(itemId, openFlag) {
  const frag = document.createDocumentFragment();
  const card = document.createElement('div');
  card.className = 'form-card';
  frag.appendChild(card);

  const h = document.createElement('h2');
  h.textContent = openFlag ? S.LABEL_OPEN_CANTIERE : S.LABEL_CLOSE_CANTIERE;
  card.appendChild(h);

  const form = document.createElement('form');
  card.appendChild(form);

  const dateRow = mkRow(form, 'narrow');
  mkInput(dateRow, {
    id: 'pdt-transition-date', name: 'date', label: S.LABEL_DATE,
    type: 'date', required: true,
    value: new Date().toISOString().slice(0, 10),
  });

  const noteRow = mkRow(form);
  mkInput(noteRow, {
    id: 'pdt-transition-note', name: 'note', label: S.LABEL_NOTE,
    type: 'text',
  });

  mkFormActions(form, {
    onCancel: dismissModal,
    submitLabel: S.CONFIRM,
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const dateVal = (fd.get('date') || '').toString().trim();
    if (!dateVal) { showFormError(form, S.ERR_DATE_REQUIRED); return; }

    const body = {
      harvest_plan_item_id: itemId,
      open: openFlag,
      date: dateVal,
      note: (fd.get('note') || '').toString().trim(),
      nonce: crypto.randomUUID(),
    };

    try {
      const { data, status } = await postJSON(TRANSITION_SAVE_URL, body);
      if (status !== 200) {
        showFormError(form, data?.message || S.ERROR_GENERIC);
        return;
      }
      // Patch the items cache with the updated record.
      if (data.item_record) {
        cache.updateRow(ITEMS_ID, data.row_id, data.item_record);
        itemsData = cache.get(ITEMS_ID);
      }
      dismissModal();
      renderItemView(itemId);
    } catch {
      showFormError(form, S.ERROR_NETWORK);
    }
  });

  showModal(frag);
}

// --- Martellata sub-section ---

function markTreesDataId(itemId) {
  return `mark_trees_${itemId}`;
}

async function appendItemMarkTreesSection(card, itemId, state) {
  const [header, body] = mkCollapsible(S.SECTION_MARTELLATA, true);
  card.append(header, body);

  const isClosed = state === S.STATE_CLOSED;

  // Closed-cantiere banner.
  if (isClosed) {
    const banner = document.createElement('div');
    banner.className = 'subsection-banner';
    banner.textContent = S.MARK_CLOSED_BANNER;
    body.appendChild(banner);
  }

  // Action buttons (writers only, not closed).
  if (canModify() && !isClosed) {
    const actions = document.createElement('div');
    actions.className = 'subsection-actions';

    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'btn';
    addBtn.textContent = S.NEW_MARK_LABEL;
    addBtn.addEventListener('click', () => showNewMarkForm(itemId));
    actions.appendChild(addBtn);

    const importBtn = document.createElement('button');
    importBtn.type = 'button';
    importBtn.className = 'btn';
    importBtn.textContent = S.IMPORT_MARKS_LABEL;
    importBtn.addEventListener('click', () => showImportMarksForm(itemId));
    actions.appendChild(importBtn);

    body.appendChild(actions);
  }

  // Lazy-fetch per-item mark trees.
  const dataId = markTreesDataId(itemId);
  cache.register(dataId, `${MARK_TREES_URL}${itemId}/`);

  let data;
  try { data = await cache.load(dataId); }
  catch {
    const err = document.createElement('div');
    err.textContent = S.ERROR_NETWORK;
    err.style.color = '#c00';
    body.appendChild(err);
    return;
  }
  if (!data?.rows) return;

  // Volume + mass totals.
  const c = data.columns;
  const volCol = c.indexOf(S.COL_V_M3);
  const massCol = c.indexOf(S.COL_MASS_Q);
  const volTotal = data.rows.reduce((s, r) => s + (typeof r[volCol] === 'number' ? r[volCol] : 0), 0);
  const massTotal = data.rows.reduce((s, r) => s + (typeof r[massCol] === 'number' ? r[massCol] : 0), 0);
  const hasNullVolume = data.rows.some(r => r[volCol] == null);
  const totals = document.createElement('div');
  totals.className = 'subsection-totals';
  totals.textContent =
    `${S.LABEL_VOLUME_TOTAL}: ${fmtVolume(volTotal)}  ·  ${S.LABEL_MASS_TOTAL}: ${fmtMass(massTotal)}`;
  body.appendChild(totals);
  if (hasNullVolume) {
    const note = document.createElement('div');
    note.className = 'subsection-note';
    note.textContent = S.MARK_NULL_VOLUME_NOTE;
    body.appendChild(note);
  }

  if (!data.rows.length) return;

  const host = document.createElement('div');
  host.className = 'subsection-host';
  body.appendChild(host);

  destroyItemMarkTreesTable();
  const showActions = canModify() && !isClosed;
  itemMarkTreesTable = new TableWrapper({
    container: host,
    digest: data,
    columnDefs: buildMarkTreeColumnDefs(c),
    inlineToolbar: false,
    canModify: showActions,
    actions: showActions ? {
      onEdit: (rowId) => showEditMarkForm(itemId, rowId),
      onDelete: (rowId, version) => deleteMarkRow(itemId, rowId, version),
    } : {},
    sort: { column: S.COL_DATE, ascending: false },
    csvFilename: `${S.CSV_MARTELLATE_PREFIX}${itemId}.csv`,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
  });
}

function buildMarkTreeColumnDefs(columns) {
  const hidden = new Set([VERSION]);
  const defs = {};
  for (const name of columns) {
    if (name === ROW_ID) continue;
    if (hidden.has(name)) { defs[name] = { label: name, hidden: true }; continue; }
    if (name === S.COL_DATE) { defs[name] = { label: name, type: 'date', width: '100px' }; continue; }
    if (name === S.COL_NUMERO) { defs[name] = { label: name, type: 'number', width: '60px' }; continue; }
    if (name === S.COL_D_CM) { defs[name] = { label: name, type: 'number', width: '70px' }; continue; }
    if (name === S.COL_H_M) { defs[name] = { label: name, type: 'number', width: '70px', formatter: fmtDecimal2 }; continue; }
    if (name === S.COL_H_MEASURED) { defs[name] = { label: name, type: 'boolean', width: '85px' }; continue; }
    if (name === S.COL_V_M3) { defs[name] = { label: name, type: 'number', width: '85px', formatter: fmtDecimal3 }; continue; }
    if (name === S.COL_MASS_Q) { defs[name] = { label: name, type: 'number', width: '70px', formatter: fmtDecimal2 }; continue; }
    if (name === S.COL_LAT || name === S.COL_LON) { defs[name] = { label: name, type: 'number', width: '90px', formatter: fmtCoord }; continue; }
    defs[name] = { label: name };
  }
  return defs;
}

// --- Mark form (template-based, shared wireVMPreview) ---

function findRegressions(itemId) {
  if (!regressionsData?.rows || !itemsData?.columns) return null;
  const c = itemsData.columns;
  const row = itemsData.rows.find(r => r[c.indexOf(ROW_ID)] === itemId);
  if (!row) return null;
  const planId = row[c.indexOf(S.COL_HARVEST_PLAN)];
  const compresa = row[c.indexOf(S.COL_COMPRESA)];
  const rc = regressionsData.columns;
  return regressionsData.rows.filter(
    r => r[rc.indexOf(S.COL_HARVEST_PLAN)] === planId
      && r[rc.indexOf(S.COL_COMPRESA)] === compresa,
  );
}

function wireRegressionAutoH(form, regressions) {
  if (!regressions?.length) return;
  const d = form.querySelector(`#${ID_D_CM}`);
  const h = form.querySelector(`#${ID_H_M}`);
  const sp = form.querySelector(`#${ID_SPECIES}`);
  const hMeasHidden = form.querySelector('#tf-h-measured');
  if (!d || !h || !sp) return;

  const rc = regressionsData.columns;
  let userEditedH = hMeasHidden?.value === '1';
  let programmaticH = false;

  function autoFillH() {
    if (userEditedH) return;
    const dCm = parseFloat(d.value);
    const opt = sp.options[sp.selectedIndex];
    const speciesName = opt?.dataset.name;
    if (!speciesName || !(dCm > 0)) return;
    const reg = regressions.find(
      r => r[rc.indexOf(S.COL_SPECIES)] === speciesName,
    );
    if (!reg) return;
    const fn = reg[rc.indexOf(S.COL_FUNCTION)];
    const a = reg[rc.indexOf(S.COL_A)];
    const b = reg[rc.indexOf(S.COL_B)];
    let hVal = null;
    if (fn === 'ln' && dCm > 0) hVal = a * Math.log(dCm) + b;
    if (hVal != null && hVal > 0) {
      programmaticH = true;
      h.value = hVal.toFixed(2);
      h.dispatchEvent(new Event('input'));
      programmaticH = false;
      if (hMeasHidden) hMeasHidden.value = '0';
    }
  }

  h.addEventListener('input', () => {
    if (programmaticH) return;
    userEditedH = true;
    if (hMeasHidden) hMeasHidden.value = '1';
  });
  d.addEventListener('input', autoFillH);
  sp.addEventListener('change', () => { userEditedH = false; autoFillH(); });
  autoFillH();
}

async function wireMarkForm(form, itemId) {
  if (!regressionsData) {
    try { regressionsData = await cache.load(REGRESSIONS_ID); }
    catch { /* regressions unavailable */ }
  }
  const regressions = findRegressions(itemId);
  wireVMPreview(form);
  wireRegressionAutoH(form, regressions);
  mountUseLocationButton(
    form.querySelector(`#${ID_LAT}`),
    form.querySelector(`#${ID_LON}`),
    { appendTo: form.querySelector(`#${ID_LON}`)?.closest('.form-row') },
  );
  wireCancelButtons(form, dismissModal);
  interceptSubmit(form, MARK_SAVE_URL, {
    onSuccess(data, isSaveAndAdd) {
      _applyMarkSaveResponse(data, itemId);
      dismissModal();
      if (isSaveAndAdd) {
        showNewMarkForm(itemId);
      } else {
        renderItemView(itemId);
      }
    },
    onConflict(data) {
      if (data.message) showFormError(form, data.message);
    },
    onValidationError(data) {
      if (data.message) showFormError(form, data.message);
    },
  });
}

async function showNewMarkForm(itemId) {
  const url = `${MARK_FORM_URL}?item=${itemId}`;
  const form = await fetchModalForm(url);
  if (!form) return;
  wireMarkForm(form, itemId);
}

async function showEditMarkForm(itemId, rowId) {
  const url = `${MARK_FORM_URL}${rowId}/`;
  const form = await fetchModalForm(url);
  if (!form) return;
  wireMarkForm(form, itemId);
}

async function showImportMarksForm(itemId) {
  const frag = document.createDocumentFragment();
  const card = document.createElement('div');
  card.className = 'form-card';
  frag.appendChild(card);

  const h2 = document.createElement('h2');
  h2.textContent = S.IMPORT_MARKS_TITLE;
  card.appendChild(h2);

  const form = document.createElement('form');
  card.appendChild(form);

  const row1 = mkRow(form);
  mkFileInput(row1, {
    id: 'mark-csv-file', name: 'file', label: S.LABEL_CSV_FILE,
    accept: '.csv',
  });

  const statusBox = mkStatusBox(form);
  const errorsBox = mkErrorsBox(form);

  mkFormActions(form, {
    onCancel: dismissModal,
    submitLabel: S.IMPORT_LABEL,
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = form.querySelector('#mark-csv-file');
    if (!fileInput?.files?.length) {
      showFormError(form, S.ERR_CSV_FILE_REQUIRED);
      return;
    }
    statusBox.textContent = S.CSV_IMPORT_IN_PROGRESS;
    statusBox.hidden = false;
    errorsBox.hidden = true;

    const fd = new FormData();
    fd.append('harvest_plan_item_id', String(itemId));
    fd.append('file', fileInput.files[0]);

    try {
      const { data, status } = await postFormData(MARK_CSV_IMPORT_URL, fd);
      if (status !== 200) {
        statusBox.hidden = true;
        if (data?.errors) {
          renderCsvErrors(errorsBox, data.errors);
        } else {
          showFormError(form, data?.message || S.ERROR_GENERIC);
        }
        return;
      }
      if (data.item_record) {
        cache.updateRow(ITEMS_ID, data.item_record[0], data.item_record);
        itemsData = cache.get(ITEMS_ID);
      }
      dismissModal();
      renderItemView(itemId);
    } catch {
      statusBox.hidden = true;
      showFormError(form, S.ERROR_NETWORK);
    }
  });

  showModal(frag);
}

async function deleteMarkRow(itemId, rowId, version) {
  if (!confirm(S.DELETE_CONFIRM)) return;
  try {
    const { data, status } = await postJSON(MARK_DELETE_URL, {
      row_id: rowId, version, nonce: crypto.randomUUID(),
    });
    if (status !== 200) {
      showError(data?.message || S.ERROR_GENERIC);
      return;
    }
    const dataId = markTreesDataId(itemId);
    cache.removeRow(dataId, rowId);
    if (data.item_record) {
      cache.updateRow(ITEMS_ID, data.item_record[0], data.item_record);
      itemsData = cache.get(ITEMS_ID);
    }
    renderItemView(itemId);
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function _applyMarkSaveResponse(data, itemId) {
  const dataId = markTreesDataId(itemId);
  if (data.record) {
    cache.updateRow(dataId, data.row_id, data.record);
  }
  if (data.item_record) {
    cache.updateRow(ITEMS_ID, data.item_record[0], data.item_record);
    itemsData = cache.get(ITEMS_ID);
  }
}

// --- Prelievi sub-section ---

async function appendItemPrelieviSection(card, itemId) {
  const [header, body] = mkCollapsible(S.SECTION_PRELIEVI, true);
  card.append(header, body);

  // Load prelievi data (lazy, from cache or network).
  try {
    prelieviData = await cache.load(PRELIEVI_ID);
  } catch {
    const err = document.createElement('div');
    err.textContent = S.ERROR_NETWORK;
    err.style.color = '#c00';
    body.appendChild(err);
    return;
  }

  if (!prelieviData?.rows?.length) return;

  const c = prelieviData.columns;
  const cantiereCol = c.indexOf(S.COL_CANTIERE);
  if (cantiereCol < 0) return;

  const filtered = prelieviData.rows.filter(r => r[cantiereCol] === itemId);

  // Volume total
  const volCol = c.indexOf(S.COL_VOLUME_M3);
  const total = filtered.reduce((sum, r) => sum + (typeof r[volCol] === 'number' ? r[volCol] : 0), 0);
  const totalEl = document.createElement('div');
  totalEl.className = 'subsection-totals';
  totalEl.textContent = `${S.LABEL_VOLUME_TOTAL}: ${fmtVolume(total)}`;
  body.appendChild(totalEl);

  if (!filtered.length) return;

  // Build a filtered digest for the table.
  const filteredDigest = { columns: c, rows: filtered };

  const host = document.createElement('div');
  host.className = 'subsection-host';
  body.appendChild(host);

  destroyItemPrelieviTable();
  itemPrelieviTable = new TableWrapper({
    container: host,
    digest: filteredDigest,
    columnDefs: buildPrelieviColumnDefs(c),
    inlineToolbar: false,
    canModify: false,
    actions: {},
    sort: { column: S.COL_DATE, ascending: false },
    csvFilename: `${S.CSV_PRELIEVI_PREFIX}${itemId}.csv`,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
  });
}

function buildPrelieviColumnDefs(columns) {
  const hidden = new Set([VERSION, S.COL_CANTIERE]);
  for (const name of columns) {
    if (name.endsWith(' %')) hidden.add(name);
  }
  const defs = {};
  for (const name of columns) {
    if (name === ROW_ID) continue;
    if (hidden.has(name)) { defs[name] = { label: name, hidden: true }; continue; }
    if (name === S.COL_DATE) { defs[name] = { label: name, type: 'date', width: '100px' }; continue; }
    if (name === S.COL_QUINTALS) { defs[name] = { label: name, type: 'number', width: '70px', formatter: fmtDecimal1 }; continue; }
    if (name === S.COL_VOLUME_M3) { defs[name] = { label: name, type: 'number', width: '95px', formatter: fmtDecimal2 }; continue; }
    defs[name] = { label: name };
  }
  return defs;
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
