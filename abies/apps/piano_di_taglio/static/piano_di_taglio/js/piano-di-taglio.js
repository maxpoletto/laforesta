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
  fetchModalForm, interceptSubmit, submitCsvImport, showFormError,
} from '../../base/js/forms.js';
import {
  showCascadeDeleteModal, wireActions, wireCancelButtons,
  wireCollapsibleToggle, wireTabbedModal, showLoadingIn,
} from '../../base/js/ui-widgets.js';
import { canModify } from '../../base/js/roles.js';
import { loadCSS, unloadCSS } from '../../base/js/page-css.js';
import { installEscapeHandler } from '../../base/js/escape.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { downloadFromURL } from '../../base/js/csv-export.js';
import {
  wireVMPreview, ID_D_CM, ID_H_M, ID_SPECIES, ID_LAT, ID_LON,
} from '../../base/js/tree-form.js';
import { mountUseLocationButton } from '../../base/js/latlng-input.js';
import * as S from '../../base/js/strings.js';
import {
  FIELD_DATE, FIELD_DESCRIPTION, FIELD_FILE, FIELD_HARVEST_PLAN_ID,
  FIELD_HARVEST_PLAN_ITEM_ID, FIELD_NAME, FIELD_NONCE, FIELD_NOTE,
  FIELD_OPEN, FIELD_YEAR_END, FIELD_YEAR_START,
  HYPSO_FUNC_LN, ROW_ID, VERSION,
} from '../../base/js/constants.js';
import {
  fmtDecimal1, fmtDecimal2, fmtDecimal3, fmtInt, fmtCoord,
  fmtVolume, fmtArea, fmtMass, parseDecimal,
} from '../../base/js/format.js';

const CSS_URL = '/static/piano_di_taglio/css/piano-di-taglio.css';

// Cache keys MUST match server `data_id` strings (apps/base/digests.py).
const PLANS_ID = 'harvest_plans';
const ITEMS_ID = 'harvest_plan_items';
// Hypsometric params live in a single global set, served by Impostazioni and
// consumed here only to auto-fill h in the mark form.
const HYPSO_PARAMS_ID = 'hypso_params';

const PLANS_URL = '/api/piano-di-taglio/plans/data/';
const ITEMS_URL = '/api/piano-di-taglio/items/data/';
const HYPSO_PARAMS_URL = '/api/impostazioni/hypso-params/data/';
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
cache.register(HYPSO_PARAMS_ID, HYPSO_PARAMS_URL);
cache.register(PRELIEVI_ID, PRELIEVI_URL);

// --- Page state -------------------------------------------------------------

let activePlanId = null;
let plansData = null;
let itemsData = null;
let hypsoParamsData = null;
let unsubPlans = null;
let unsubItems = null;

let descriptionEl = null;
let planSelectEl = null;
let activeItemId = null;
let prelieviData = null;
let itemPrelieviTable = null;
let itemMarkTreesTable = null;
let disposeEscape = null;
let disposePageActions = null;

// Calendar sections — keyed by the single-char URL `o=` token.  `f`
// fills in here; `c` (Calendario ceduo) lands in a later increment.
const SECTION_KEYS = ['f', 'c'];
const DEFAULT_OPEN = 'f';

const sections = {
  f: {
    open: true, kind: 'fustaia',
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
    open: false, kind: 'ceduo',
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

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

export async function mount(params) {
  loadCSS(CSS_URL);
  const el = document.getElementById('content');
  showLoadingIn(el);

  try {
    const [p, i, r] = await Promise.all([
      cache.load(PLANS_ID),
      cache.load(ITEMS_ID),
      cache.load(HYPSO_PARAMS_ID),
    ]);
    plansData = p;
    itemsData = i;
    hypsoParamsData = r;
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  buildPage(el);
  cache.setVisible([PLANS_ID, ITEMS_ID, HYPSO_PARAMS_ID]);
  if (unsubPlans) unsubPlans();
  if (unsubItems) unsubItems();
  unsubPlans = cache.onUpdate(PLANS_ID, onPlansUpdate);
  unsubItems = cache.onUpdate(ITEMS_ID, onItemsUpdate);

  applyParams(params);
}

export function unmount() {
  unloadCSS(CSS_URL);
  if (disposeEscape) { disposeEscape(); disposeEscape = null; }
  if (unsubPlans) { unsubPlans(); unsubPlans = null; }
  if (unsubItems) { unsubItems(); unsubItems = null; }
  if (disposePageActions) { disposePageActions(); disposePageActions = null; }
  cache.setVisible([]);
  destroyTables();
  destroyItemPrelieviTable();
  activePlanId = null;
  activeItemId = null;
  plansData = itemsData = hypsoParamsData = prelieviData = null;
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

function buildPage(el) {
  disposePageActions?.();
  el.replaceChildren();
  const frag = cloneTemplate('tmpl-pdt-page');
  el.appendChild(frag);

  const noPlansEl = el.querySelector('[data-target="no-plans"]');
  const hasPlans = plansData?.rows.length > 0;
  noPlansEl.hidden = hasPlans;

  // Header elements.
  const headerLeft = el.querySelector('.pdt-header-left');
  const label = headerLeft.querySelector('.pdt-pulldown-label');
  const sel = el.querySelector('#pdt-plan-select');
  if (label) label.hidden = !hasPlans;
  sel.hidden = !hasPlans;
  planSelectEl = sel;
  populatePulldown(sel, plansData);
  sel.addEventListener('change', () => {
    setActivePlan(parseInt(sel.value, 10));
    syncURL();
  });

  // Edit/delete icons (present only for writers via Django template guard).
  const editIcon = el.querySelector('[data-action="edit-plan"]');
  const deleteIcon = el.querySelector('[data-action="delete-plan"]');
  if (editIcon) editIcon.hidden = !hasPlans;
  if (deleteIcon) deleteIcon.hidden = !hasPlans;

  descriptionEl = el.querySelector('[data-target="description"]');

  // Wire sections.
  for (const k of SECTION_KEYS) {
    const s = sections[k];
    s.header = el.querySelector(`[data-section="${k}"].collapsible-header`);
    s.body = el.querySelector(`[data-section="${k}"].collapsible-body`);
    s.header?.classList.toggle('open', s.open);
    s.body?.classList.toggle('open', s.open);
    s.toolbar = el.querySelector(`[data-target="toolbar-${k}"]`);
    s.host = el.querySelector(`[data-target="table-${k}"]`);
    s.actionAdd = el.querySelector(`[data-target="add-${k}"]`) || null;
    s.emptyState = el.querySelector(`[data-target="empty-${k}"]`) || null;

    if (s.header && s.body) {
      wireCollapsibleToggle(s.header, s.body, (open) => {
        s.open = open;
        syncURL();
      });
    }

    // Hide sections when there are no plans.
    if (s.header) s.header.hidden = !hasPlans;
    if (s.body) s.body.hidden = !hasPlans;

    const searchInput = el.querySelector(`#pdt-search-${k}`);
    buildTable(s, searchInput);
  }

  disposePageActions = wireActions(el, {
    'edit-plan': () => onEditPlan(),
    'delete-plan': () => onDeletePlan(),
    'export-plan': () => { if (activePlanId != null) downloadPlanExport(activePlanId); },
    'new-plan': () => onNewPlan(),
    'export-section-csv': (btn) => {
      const sec = btn.closest('[data-section]');
      if (sec) sections[sec.dataset.section]?.table?.exportCSV();
    },
    'add-item-f': () => showAddItemModal(sections.f),
    'add-item-c': () => showAddItemModal(sections.c),
    'import-calendar-f': () => openEditPlanModal(EDIT_PLAN_TAB_CALENDAR, { ceduo: false }),
    'import-calendar-c': () => openEditPlanModal(EDIT_PLAN_TAB_CALENDAR, { ceduo: true }),
    'add-manual-f': () => showAddItemModal(sections.f),
    'add-manual-c': () => showAddItemModal(sections.c),
  });
}

function populatePulldown(sel, digest) {
  if (!sel || !digest) return;
  sel.replaceChildren();
  const idCol = digest.columns.indexOf(ROW_ID);
  const nameCol = digest.columns.indexOf(S.COL_NAME);
  for (const row of digest.rows) {
    const opt = document.createElement('option');
    opt.value = String(row[idCol]);
    opt.textContent = row[nameCol];
    sel.appendChild(opt);
  }
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

// ---------------------------------------------------------------------------
// Dangerous-delete flow.
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
  showCascadeDeleteModal({
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
    // The plan cascade-deletes its items; refresh that cache so the page
    // doesn't keep rendering rows for a plan that no longer exists.
    await refreshItems();
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
  showCascadeDeleteModal({
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
  downloadFromURL(`${ITEM_EXPORT_URL}${itemId}/`);
}

/**
 * Shared dangerous-delete modal.  When `onExportCSV` is provided, the
 * modal includes a forced-download step (Esporta CSV must be clicked
 * before Elimina enables) — used for plan-level delete where the
 * planning calendar is worth backing up.  When `onExportCSV` is
 * omitted (per-item delete in PLANNED state with no deps), the
 * export step is skipped: Elimina is enabled immediately.
 */

// ---------------------------------------------------------------------------
// Modifica piano modal (pencil) — three tabs.  Identity (name +
// description + year range) lives under "Dettagli"; the other two
// tabs upsert CSV rows into the active plan via the plan_csv_import
// endpoint with `harvest_plan_id` set.
// ---------------------------------------------------------------------------

const EDIT_PLAN_TAB_DETAILS    = 'details';
const EDIT_PLAN_TAB_CALENDAR   = 'calendar';

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

  const frag = cloneTemplate('tmpl-pdt-edit-plan');

  // Fill Dettagli form values.
  const detailsForm = frag.querySelector('[data-role="details-form"]');
  detailsForm.querySelector('#pdt-edit-name').value = current.name;
  detailsForm.querySelector('#pdt-edit-y1').value = current.year_start;
  detailsForm.querySelector('#pdt-edit-y2').value = current.year_end;
  detailsForm.querySelector('#pdt-edit-desc').value = current.description;

  // Pre-check ceduo checkbox on calendar tab.
  const calendarForm = frag.querySelector('[data-role="calendar-form"]');
  const ceduoCb = calendarForm.querySelector('input[name="is_ceduo"]');
  ceduoCb.checked = ceduo;

  // Wire cancel buttons on both forms.
  for (const f of [detailsForm, calendarForm]) {
    f.querySelector('[data-action="cancel"]')
      .addEventListener('click', dismissModal);
  }

  // Wire Dettagli submit.
  detailsForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(detailsForm);
    const body = {
      [ROW_ID]: String(activePlanId),
      [VERSION]: String(current.version),
      [FIELD_NAME]: (fd.get(FIELD_NAME) || '').toString().trim(),
      [FIELD_DESCRIPTION]: (fd.get(FIELD_DESCRIPTION) || '').toString().trim(),
      [FIELD_YEAR_START]: parseInt(fd.get(FIELD_YEAR_START), 10),
      [FIELD_YEAR_END]: parseInt(fd.get(FIELD_YEAR_END), 10),
      [FIELD_NONCE]: crypto.randomUUID(),
    };
    if (!body[FIELD_NAME]) { showFormError(detailsForm, S.ERR_PLAN_NAME_REQUIRED); return; }
    if (body[FIELD_YEAR_END] < body[FIELD_YEAR_START]) {
      showFormError(detailsForm, S.ERR_PLAN_YEAR_RANGE); return;
    }
    try {
      const { data, status } = await postJSON(PLAN_SAVE_URL, body);
      if (status !== 200) {
        showFormError(detailsForm, data?.message || S.ERROR_GENERIC);
        return;
      }
      if (data.record) cache.updateRow(PLANS_ID, data.row_id, data.record);
      plansData = cache.get(PLANS_ID);
      onPlansUpdate();
      dismissModal();
    } catch {
      showFormError(detailsForm, S.ERROR_NETWORK);
    }
  });

  // Wire Calendar submit.
  const calFileInput = calendarForm.querySelector('#pdt-edit-cal-file');
  const calStatusBox = calendarForm.querySelector('.csv-import-status');
  const calErrorsBox = calendarForm.querySelector('.csv-import-errors');

  calendarForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (activePlanId == null) return;
    const file = calFileInput.files[0];
    if (!file) { showFormError(calendarForm, S.ERR_CSV_FILE_REQUIRED); return; }

    const fd = new FormData();
    fd.append(FIELD_HARVEST_PLAN_ID, String(activePlanId));
    fd.append(ceduoCb.checked ? 'ceduo_file' : 'fustaia_file', file);
    fd.append(FIELD_NONCE, crypto.randomUUID());

    await submitPlanCsvImport(calendarForm, fd, calStatusBox, calErrorsBox);
  });

  const { lockHeight } = wireTabbedModal(frag, {
    initialTab: initialTab || EDIT_PLAN_TAB_DETAILS,
  });
  showModal(frag);
  lockHeight();
}

// ---------------------------------------------------------------------------
// Nuovo piano modal — name + description only.  An empty plan starts
// with year_start = year_end = current civil year; the range widens
// later via pencil edit or implicitly on CSV import.
// Identity (name) is deliberately decoupled from content (calendar /
// equations) — those land via the pencil modal.
// ---------------------------------------------------------------------------

function onNewPlan() {
  const frag = cloneTemplate('tmpl-pdt-new-plan');
  const form = frag.querySelector('form');

  form.querySelector('[data-action="cancel"]')
    .addEventListener('click', dismissModal);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const today = new Date().getFullYear();
    const body = {
      [FIELD_NAME]: (fd.get(FIELD_NAME) || '').toString().trim(),
      [FIELD_DESCRIPTION]: (fd.get(FIELD_DESCRIPTION) || '').toString().trim(),
      [FIELD_YEAR_START]: today,
      [FIELD_YEAR_END]: today,
      [FIELD_NONCE]: crypto.randomUUID(),
    };
    if (!body[FIELD_NAME]) { showFormError(form, S.ERR_PLAN_NAME_REQUIRED); return; }
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
 * Submit a multipart POST to /plan/import-csv/ and, on success, refresh
 * the affected digests + land on the newly-imported plan.  Thin wrapper
 * around `submitCsvImport` that knows the plan-specific URL + side
 * effects.
 */
async function submitPlanCsvImport(form, fd, statusBox, errorsBox) {
  return submitCsvImport({
    form, statusBox, errorsBox,
    attempt: async () => {
      const { data, status } = await postFormData(PLAN_IMPORT_CSV_URL, fd);
      if (status === 200) {
        await Promise.all([refreshPlans(), refreshItems()]);
        activePlanId = data.row_id;
        onPlansUpdate();
        syncURL();
        return { ok: true };
      }
      return data?.errors?.length
        ? { errors: data.errors }
        : { error: data?.message };
    },
  });
}

async function refreshPlans() {
  try { await cache.load(PLANS_ID); plansData = cache.get(PLANS_ID); } catch {}
}
async function refreshItems() {
  try { await cache.load(ITEMS_ID); itemsData = cache.get(ITEMS_ID); } catch {}
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
  buildPage(el);
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
  downloadFromURL(`${PLAN_EXPORT_URL}${planId}/`);
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
  disposeEscape?.();
  disposeEscape = installEscapeHandler(() => closeItemView(true));
  renderItemView(itemId);
}

function closeItemView(push = false) {
  if (disposeEscape) { disposeEscape(); disposeEscape = null; }
  destroyItemPrelieviTable();
  activeItemId = null;
  if (push) syncURL(true);
  const el = document.getElementById('content');
  if (el) {
    buildPage(el);
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

  showLoadingIn(el);

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

  const c = itemsData?.columns;
  if (!c) return;

  const frag = cloneTemplate('tmpl-pdt-item-view');

  // Title.
  frag.querySelector('[data-field="title"]').textContent =
    formatItemTitle(record, c);

  const card = frag.querySelector('.pdt-item-card');
  wireActions(card, {
    'edit-item': () => showItemEditForm(itemId),
    'export-item': () => downloadItemExport(itemId),
    'close-item': () => closeItemView(true),
  });

  // Metadata pane.
  const meta = frag.querySelector('[data-target="metadata"]');
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

  // Apri / Chiudi cantiere buttons (writers only — host is in template).
  const btnRow = frag.querySelector('[data-target="transitions"]');
  if (btnRow) {
    let hasTransition = false;
    if (state === S.STATE_PLANNED || state === S.STATE_MARKED) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-save';
      btn.textContent = S.LABEL_OPEN_CANTIERE;
      btn.addEventListener('click', () => showTransitionForm(itemId, true));
      btnRow.appendChild(btn);
      hasTransition = true;
    }
    if (state === S.STATE_OPEN || state === S.STATE_HARVESTING) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-save';
      btn.textContent = S.LABEL_CLOSE_CANTIERE;
      btn.addEventListener('click', () => showTransitionForm(itemId, false));
      btnRow.appendChild(btn);
      hasTransition = true;
    }
    btnRow.hidden = !hasTransition;
  }

  // Append the card to the DOM first so subsection appends land in the doc.
  el.appendChild(frag);
  const subsections = el.querySelector('[data-target="subsections"]');

  // Martellata sub-section (fustaia items only — coppice skips marking).
  if (!isCoppice) {
    appendItemMarkTreesSection(subsections, itemId, state);
  }

  // Prelievi sub-section (visible once cantiere is open or later).
  const stateIdx = [S.STATE_OPEN, S.STATE_HARVESTING, S.STATE_CLOSED];
  if (stateIdx.includes(state)) {
    appendItemPrelieviSection(subsections, itemId);
  }
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
  const location = parcel ? `${compresa}/${parcel}` : compresa;
  return `${S.VIEW_ITEM_TITLE} del ${planName}, anno ${year}, ${location}`;
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
  const frag = cloneTemplate('tmpl-pdt-transition');
  const title = frag.querySelector('[data-field="title"]');
  title.textContent = openFlag ? S.LABEL_OPEN_CANTIERE : S.LABEL_CLOSE_CANTIERE;

  const form = frag.querySelector('form');
  form.querySelector('#pdt-transition-date').value =
    new Date().toISOString().slice(0, 10);

  form.querySelector('[data-action="cancel"]')
    .addEventListener('click', dismissModal);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    const dateVal = (fd.get(FIELD_DATE) || '').toString().trim();
    if (!dateVal) { showFormError(form, S.ERR_DATE_REQUIRED); return; }

    const body = {
      [FIELD_HARVEST_PLAN_ITEM_ID]: itemId,
      [FIELD_OPEN]: openFlag,
      [FIELD_DATE]: dateVal,
      [FIELD_NOTE]: (fd.get(FIELD_NOTE) || '').toString().trim(),
      [FIELD_NONCE]: crypto.randomUUID(),
    };

    try {
      const { data, status } = await postJSON(TRANSITION_SAVE_URL, body);
      if (status !== 200) {
        showFormError(form, data?.message || S.ERROR_GENERIC);
        return;
      }
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

/**
 * Append a collapsible sub-section (header + open body) to `card` and
 * return the body element so the caller can populate it.  Starts open
 * because PDT item views always expand their sub-sections by default.
 */
function appendSubsection(card, title) {
  const frag = cloneTemplate('tmpl-pdt-item-subsection');
  frag.querySelector('[data-field="title"]').textContent = title;
  const header = frag.querySelector('.collapsible-header');
  const body = frag.querySelector('.collapsible-body');
  header.classList.add('open');
  body.classList.add('open');
  wireCollapsibleToggle(header, body);
  card.appendChild(frag);
  return body;
}

// --- Martellata sub-section ---

function markTreesDataId(itemId) {
  return `mark_trees_${itemId}`;
}

async function appendItemMarkTreesSection(card, itemId, state) {
  const body = appendSubsection(card, S.SECTION_MARTELLATA);

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
    addBtn.className = 'btn btn-create';
    addBtn.textContent = S.NEW_MARK_LABEL;
    addBtn.addEventListener('click', () => showNewMarkForm(itemId));
    actions.appendChild(addBtn);

    const importBtn = document.createElement('button');
    importBtn.type = 'button';
    importBtn.className = 'btn btn-import';
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
    if (name === S.COL_NUMERO) { defs[name] = { label: name, type: 'number', width: '60px', formatter: fmtInt }; continue; }
    if (name === S.COL_D_CM) { defs[name] = { label: name, type: 'number', width: '70px', formatter: fmtInt }; continue; }
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

function findHypsoParams(itemId) {
  if (!hypsoParamsData?.rows || !itemsData?.columns) return null;
  const c = itemsData.columns;
  const row = itemsData.rows.find(r => r[c.indexOf(ROW_ID)] === itemId);
  if (!row) return null;
  const compresa = row[c.indexOf(S.COL_COMPRESA)];
  const rc = hypsoParamsData.columns;
  return hypsoParamsData.rows.filter(
    r => r[rc.indexOf(S.COL_COMPRESA)] === compresa,
  );
}

function wireHypsoAutoH(form, hypsoParams) {
  if (!hypsoParams?.length) return;
  const d = form.querySelector(`#${ID_D_CM}`);
  const h = form.querySelector(`#${ID_H_M}`);
  const sp = form.querySelector(`#${ID_SPECIES}`);
  const hMeasHidden = form.querySelector('#tf-h-measured');
  if (!d || !h || !sp) return;

  const rc = hypsoParamsData.columns;
  let userEditedH = hMeasHidden?.value === '1';
  let programmaticH = false;

  function autoFillH() {
    if (userEditedH) return;
    const dCm = parseInt(d.value, 10);
    const opt = sp.options[sp.selectedIndex];
    const speciesName = opt?.dataset.name;
    if (!speciesName || !(dCm > 0)) return;
    const reg = hypsoParams.find(
      r => r[rc.indexOf(S.COL_SPECIES)] === speciesName,
    );
    if (!reg) return;
    const fn = reg[rc.indexOf(S.COL_FUNCTION)];
    const a = reg[rc.indexOf(S.COL_A)];
    const b = reg[rc.indexOf(S.COL_B)];
    let hVal = null;
    if (fn === HYPSO_FUNC_LN) hVal = a * Math.log(dCm) + b;
    if (hVal != null && hVal > 0) {
      programmaticH = true;
      h.value = fmtDecimal2(hVal);
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
  if (!hypsoParamsData) {
    try { hypsoParamsData = await cache.load(HYPSO_PARAMS_ID); }
    catch { /* params unavailable */ }
  }
  const hypsoParams = findHypsoParams(itemId);
  wireVMPreview(form);
  wireHypsoAutoH(form, hypsoParams);
  mountUseLocationButton(
    form.querySelector(`#${ID_LAT}`),
    form.querySelector(`#${ID_LON}`),
    { appendTo: form.querySelector(`#${ID_LON}`)?.closest('.form-row') },
  );
  wireCancelButtons(form, dismissModal);
  interceptSubmit(form, MARK_SAVE_URL, {
    validate: (body) => {
      // A mark needs D and h > 0.
      if (!(parseInt(body.d_cm, 10) > 0)) return S.ERR_D_POSITIVE;
      if (!(parseDecimal(body.h_m) > 0)) return S.ERR_H_POSITIVE;
      return null;
    },
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
  const frag = cloneTemplate('tmpl-pdt-import-marks');
  const form = frag.querySelector('form');
  const statusBox = form.querySelector('.csv-import-status');
  const errorsBox = form.querySelector('.csv-import-errors');

  form.querySelector('[data-action="cancel"]')
    .addEventListener('click', dismissModal);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = form.querySelector('#mark-csv-file');
    if (!fileInput?.files?.length) {
      showFormError(form, S.ERR_CSV_FILE_REQUIRED);
      return;
    }
    const fd = new FormData();
    fd.append(FIELD_HARVEST_PLAN_ITEM_ID, String(itemId));
    fd.append(FIELD_FILE, fileInput.files[0]);

    const result = await submitCsvImport({
      form, statusBox, errorsBox,
      attempt: async () => {
        const { data, status } = await postFormData(MARK_CSV_IMPORT_URL, fd);
        if (status === 200) {
          if (data.item_record) {
            cache.updateRow(ITEMS_ID, data.item_record[0], data.item_record);
            itemsData = cache.get(ITEMS_ID);
          }
          return { ok: true };
        }
        return data?.errors?.length
          ? { errors: data.errors }
          : { error: data?.message };
      },
    });
    if (result?.ok) renderItemView(itemId);
  });

  showModal(frag);
}

async function deleteMarkRow(itemId, rowId, version) {
  if (!confirm(S.DELETE_CONFIRM)) return;
  try {
    const { data, status } = await postJSON(MARK_DELETE_URL, {
      [ROW_ID]: rowId, [VERSION]: version, [FIELD_NONCE]: crypto.randomUUID(),
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
  const body = appendSubsection(card, S.SECTION_PRELIEVI);

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

