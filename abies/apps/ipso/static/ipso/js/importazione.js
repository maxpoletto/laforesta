/**
 * Ipso staged-upload inbox page.
 */

import * as api from '../../base/js/api.js';
import * as cache from '../../base/js/cache.js';
import { downloadFromURL } from '../../base/js/csv-export.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableSort, writeTableState,
} from '../../base/js/page-sync.js';
import { dismiss as dismissModal, show as showModal, showError } from '../../base/js/modals.js';
import {
  showCascadeDeleteModal, showConfirmModal, wireCancelButtons,
} from '../../base/js/ui-widgets.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { fmtCoord, fmtDecimal2, fmtInt } from '../../base/js/format.js';
import { installEscapeHandler } from '../../base/js/escape.js';
import * as S from '../../base/js/strings.js';
import {
  DATA_ID_IPSO_UPLOADS, FIELD_HARVEST_PLAN_ITEM_ID, FIELD_SAMPLE_AREA_ID,
  FIELD_MODE, FIELD_SURVEY_ID, FIELD_WORK_PACKAGE_LABEL, FILE_ERROR,
  IPSO_MODE_MARTELLATE, IPSO_MODE_PAI, IPSO_MODE_SAMPLES,
  IPSO_UPLOAD_STATE_IMPORTED, IPSO_UPLOAD_STATE_RECEIVED, IPSO_UPLOAD_STATE_REJECTED,
  MESSAGE, PENDING_COUNT, RECORD_COUNT, RECORDS, ROLE_ADMIN, ROLE_READER, ROWS,
  SUGGESTED_TARGET_ID, TARGETS, UPLOAD,
} from '../../base/js/constants.js';

const DATA_ID = DATA_ID_IPSO_UPLOADS;
const DATA_URL = '/api/ipso/inbox/';
const PAGE_PATH = '/importazione';
const CSS_URL = '/static/ipso/css/importazione.css';
const DEFAULT_SORT = { column: S.IPSO_COL_RECEIVED, ascending: false };
const INBOX_STATE_COL = '_ipso_state';
const DETAIL_URL = (id) => `/api/ipso/uploads/${id}/`;
const REJECT_URL = (id) => `/api/ipso/uploads/${id}/reject/`;
const DOWNLOAD_URL = (id) => `/api/ipso/uploads/${id}/download/`;
const DELETE_URL = (id) => `/api/ipso/uploads/${id}/delete/`;
const MODE_URL = (id) => `/api/ipso/uploads/${id}/mode/`;
const IMPORT_CONFIG = {
  [IPSO_MODE_MARTELLATE]: {
    url: id => `/api/ipso/uploads/${id}/import-martellate/`,
    targetField: FIELD_HARVEST_PLAN_ITEM_ID,
    targetLabel: S.IPSO_TARGET_PLAN_LABEL,
    confirm: S.IPSO_IMPORT_CONFIRM,
    requiresTarget: true,
  },
  [IPSO_MODE_SAMPLES]: {
    url: id => `/api/ipso/uploads/${id}/import-samples/`,
    targetField: FIELD_SURVEY_ID,
    targetLabel: S.IPSO_TARGET_SURVEY_LABEL,
    confirm: S.IPSO_IMPORT_SAMPLES_CONFIRM,
    requiresTarget: true,
  },
  [IPSO_MODE_PAI]: {
    url: id => `/api/ipso/uploads/${id}/import-pai/`,
    confirm: S.IPSO_IMPORT_PAI_CONFIRM,
    requiresTarget: false,
  },
};

const COLUMN_DEFS = {
  [INBOX_STATE_COL]: { hidden: true },
  [S.IPSO_COL_RECEIVED]: { label: S.IPSO_COL_RECEIVED, width: '145px' },
  [S.COL_DATE]: { label: S.COL_DATE, width: '110px' },
  [S.IPSO_COL_MODE]: { label: S.IPSO_COL_MODE, width: '120px' },
  [S.IPSO_COL_OPERATOR]: { label: S.IPSO_COL_OPERATOR, width: '150px' },
  [S.IPSO_COL_RECORDS]: { label: S.IPSO_COL_RECORDS, type: 'number', width: '80px', className: 'num' },
  [S.IPSO_COL_STATE]: { label: S.IPSO_COL_STATE, width: '120px' },
  [S.IPSO_COL_WORK_PACKAGE]: { label: S.IPSO_COL_WORK_PACKAGE, width: '190px' },
  [S.IPSO_COL_TARGET]: { label: S.IPSO_COL_TARGET, width: '150px' },
  [S.IPSO_COL_ERROR]: { label: S.IPSO_COL_ERROR, width: '260px' },
};

const PREVIEW_COLUMNS = [
  S.IPSO_COL_SEQ, S.COL_DATE, S.COL_PARCEL, FIELD_SAMPLE_AREA_ID,
  S.COL_SPECIES, S.COL_NUMBER, S.COL_D_CM, S.COL_H_M, S.COL_LAT, S.COL_LON,
  S.IPSO_COL_ACCURACY,
];
const PREVIEW_COLUMN_DEFS = {
  [S.IPSO_COL_SEQ]: { label: S.IPSO_COL_SEQ, type: 'number', width: '70px', className: 'num', formatter: intValue },
  [S.COL_DATE]: { label: S.COL_DATE, width: '110px' },
  [S.COL_PARCEL]: { label: S.COL_PARCEL, width: '150px' },
  [FIELD_SAMPLE_AREA_ID]: { label: S.COL_SAMPLE_AREA, width: '115px' },
  [S.COL_SPECIES]: { label: S.COL_SPECIES, width: '150px' },
  [S.COL_NUMBER]: { label: S.COL_NUMBER, type: 'number', width: '90px', className: 'num', formatter: intValue },
  [S.COL_D_CM]: { label: S.COL_D_CM, type: 'number', width: '90px', className: 'num', formatter: intValue },
  [S.COL_H_M]: { label: S.COL_H_M, type: 'number', width: '90px', className: 'num', formatter: decimal2Value },
  [S.COL_LAT]: { label: S.COL_LAT, type: 'number', width: '115px', formatter: coordValue },
  [S.COL_LON]: { label: S.COL_LON, type: 'number', width: '115px', formatter: coordValue },
  [S.IPSO_COL_ACCURACY]: { label: S.IPSO_COL_ACCURACY, type: 'number', width: '80px', className: 'num', formatter: intValue },
};

let table = null;
let detailTable = null;
let detailEl = null;
let summaryEl = null;
let selectedId = null;
let disposeEscape = null;
let includeImportedEl = null;
let inboxStateIndex = -1;

cache.register(DATA_ID, DATA_URL);

const page = createPage({
  cssUrl: CSS_URL,
  dataIds: [DATA_ID],
  mount: buildPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [[DATA_ID, data => refreshTable(data)]],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

function buildPage(el, params, data) {
  el.replaceChildren(cloneTemplate('tmpl-ipso-inbox-page'));
  selectedId = null;

  const root = el.querySelector('.ipso-inbox-page');
  const tableHost = root.querySelector('[data-role="table"]');
  summaryEl = root.querySelector('[data-role="summary"]');
  detailEl = root.querySelector('[data-role="detail"]');
  disposeEscape = installEscapeHandler(closeDetail);
  summaryEl.textContent = summaryText(data);

  const state = readTableState(params);
  inboxStateIndex = inboxStateColumnIndex(data);
  table = new TableWrapper({
    container: tableHost,
    digest: data,
    columnDefs: COLUMN_DEFS,
    canModify: true,
    actions: inboxActions(),
    sort: tableSort(state, DEFAULT_SORT),
    searchText: state.searchText,
    csvFilename: S.IPSO_INBOX_CSV,
    labels: {
      ...S.TABLE_LABELS,
      actionEdit: S.IPSO_ACTION_OPEN,
      actionEditIcon: S.IPSO_ACTION_OPEN_ICON,
    },
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
  installIncludeImportedToggle(tableHost);
  applyInboxFilter();
  updateNavDot(data);
}

function inboxActions() {
  const actions = {
    onEdit: id => openUpload(id),
  };
  if (isAdmin()) {
    actions.extra = [{
      key: 'mode',
      title: S.ACTION_EDIT,
      icon: '✎',
      visible: row => !isImportedInboxRow(row),
      onClick: id => showModeModal(id),
    }];
    actions.onDelete = id => confirmDeleteUpload(id);
  }
  return actions;
}

function destroyPage() {
  if (disposeEscape) { disposeEscape(); disposeEscape = null; }
  if (table) { table.destroy(); table = null; }
  destroyDetailTable();
  detailEl = null;
  summaryEl = null;
  selectedId = null;
  includeImportedEl = null;
  inboxStateIndex = -1;
}

function applyParams(params) {
  applyTableState(table, readTableState(params), DEFAULT_SORT);
}

function refreshTable(data) {
  inboxStateIndex = inboxStateColumnIndex(data);
  table?.setData(data);
  applyInboxFilter();
  updateNavDot(data);
  if (summaryEl) summaryEl.textContent = summaryText(data);
}

function syncURL() {
  const params = new URLSearchParams();
  writeTableState(params, table);
  navigateWithParams(PAGE_PATH, params);
}

function installIncludeImportedToggle(tableHost) {
  const search = tableHost.querySelector('.table-search');
  if (!search) return;

  const label = document.createElement('label');
  label.className = 'ipso-include-imported';
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = false;
  checkbox.addEventListener('change', applyInboxFilter);
  label.append(checkbox, document.createTextNode(S.IPSO_INCLUDE_IMPORTED));
  search.after(label);
  includeImportedEl = checkbox;
}

function applyInboxFilter() {
  if (!table) return;
  table.setExternalFilter(row => includeImportedEl?.checked || !isImportedInboxRow(row));
}

function isImportedInboxRow(row) {
  return inboxStateIndex >= 0 && row[inboxStateIndex] === IPSO_UPLOAD_STATE_IMPORTED;
}

function inboxStateColumnIndex(data) {
  return Array.isArray(data?.columns) ? data.columns.indexOf(INBOX_STATE_COL) : -1;
}

async function openUpload(id) {
  selectedId = id;
  table?.setSelectedRow(id);
  const needsInitialContent = detailEl.hidden || detailEl.childElementCount === 0;
  detailEl.hidden = false;
  detailEl.classList.add('is-loading');
  detailEl.setAttribute('aria-busy', 'true');
  if (needsInitialContent) {
    destroyDetailTable();
    detailEl.replaceChildren(loadingBlock(S.IPSO_LOADING_DETAIL));
  }
  try {
    const { data } = await api.fetchJSON(DETAIL_URL(id));
    if (selectedId !== id) return;
    renderDetail(data);
  } catch {
    if (selectedId === id) {
      detailEl.classList.remove('is-loading');
      detailEl.removeAttribute('aria-busy');
    }
    showError(S.ERROR_NETWORK);
  }
}

function closeDetail() {
  if (!detailEl || detailEl.hidden) return;
  detailEl.hidden = true;
  detailEl.classList.remove('is-loading');
  detailEl.removeAttribute('aria-busy');
  destroyDetailTable();
  detailEl.replaceChildren();
  selectedId = null;
  table?.setSelectedRow(null);
}

function destroyDetailTable() {
  if (!detailTable) return;
  detailTable.destroy();
  detailTable = null;
}

function confirmReject(id) {
  showConfirmModal(
    S.IPSO_REJECT_CONFIRM,
    async () => rejectUpload(id),
    { confirmLabel: S.IPSO_ACTION_REJECT },
  );
}

async function rejectUpload(id) {
  const { data, status } = await api.postJSON(REJECT_URL(id), {});
  if (status >= 400) {
    showError(data?.[MESSAGE] || S.ERROR_GENERIC);
    return;
  }
  closeDetail();
  await cache.load(DATA_ID);
}

function confirmDeleteUpload(id) {
  showCascadeDeleteModal({
    title: S.IPSO_DELETE_TITLE,
    warning: S.IPSO_DELETE_WARNING,
    exportRequired: S.IPSO_DELETE_EXPORT_REQUIRED,
    onExport: () => downloadFromURL(DOWNLOAD_URL(id)),
    onDelete: () => deleteUpload(id),
  });
}

async function deleteUpload(id) {
  const { data, status } = await api.postJSON(DELETE_URL(id), {});
  if (status >= 400) {
    showError(data?.[MESSAGE] || S.ERROR_GENERIC);
    return;
  }
  if (selectedId === id) closeDetail();
  await cache.load(DATA_ID);
}

async function showModeModal(id) {
  let detail;
  try {
    ({ data: detail } = await api.fetchJSON(DETAIL_URL(id)));
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  const upload = detail[UPLOAD] || {};
  if (upload.state === IPSO_UPLOAD_STATE_IMPORTED) {
    showError(S.IPSO_MODE_SAVE_ERROR_IMPORTED);
    return;
  }

  const frag = cloneTemplate('tmpl-ipso-upload-mode-modal');
  const form = frag.querySelector('[data-role="ipso-upload-mode-form"]');
  const select = frag.querySelector('[data-role="mode"]');
  select.value = upload.mode || IPSO_MODE_MARTELLATE;
  wireCancelButtons(form, () => dismissModal());
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await saveUploadMode(id, select.value);
  });
  showModal(frag);
}

async function saveUploadMode(id, mode) {
  const { data, status } = await api.postJSON(MODE_URL(id), { [FIELD_MODE]: mode });
  if (status >= 400) {
    showError(data?.[MESSAGE] || S.ERROR_GENERIC);
    return;
  }
  dismissModal();
  await cache.load(DATA_ID);
  if (selectedId === id) await openUpload(id);
}

function renderDetail(data) {
  detailEl.classList.remove('is-loading');
  detailEl.removeAttribute('aria-busy');
  destroyDetailTable();
  detailEl.replaceChildren();
  const upload = data[UPLOAD] || {};

  const header = document.createElement('div');
  header.className = 'ipso-detail-header';
  const title = document.createElement('h2');
  title.textContent = S.IPSO_SESSION_TITLE(upload.session_id || upload.id);
  header.appendChild(title);
  const actions = document.createElement('div');
  actions.className = 'ipso-detail-actions';
  if (canRejectUpload(upload)) {
    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'btn btn-delete';
    rejectBtn.textContent = S.IPSO_ACTION_REJECT;
    rejectBtn.addEventListener('click', () => confirmReject(upload.id));
    actions.appendChild(rejectBtn);
  }
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn ipso-detail-close';
  closeBtn.type = 'button';
  closeBtn.textContent = '\u00D7';
  closeBtn.title = S.DISMISS;
  closeBtn.setAttribute('aria-label', S.DISMISS);
  closeBtn.addEventListener('click', closeDetail);
  actions.appendChild(closeBtn);
  header.appendChild(actions);
  detailEl.appendChild(header);

  if (data[FILE_ERROR]) {
    const warning = document.createElement('p');
    warning.className = 'modal-error';
    warning.textContent = data[FILE_ERROR];
    detailEl.appendChild(warning);
  }

  detailEl.appendChild(metadataGrid([
    [S.IPSO_COL_STATE, upload.state_label],
    [S.IPSO_COL_MODE, upload.mode_label || modeLabel(upload.mode)],
    [S.IPSO_COL_OPERATOR, upload.operator],
    [S.IPSO_COL_RECEIVED, upload.received_at],
    [S.COL_DATE, upload.record_date],
    [S.IPSO_COL_RECORDS, upload.record_count],
    [S.IPSO_COL_REFERENCE, upload.reference_version_label || referenceLabel(upload.reference_version)],
    [S.IPSO_COL_WORK_PACKAGE, upload[FIELD_WORK_PACKAGE_LABEL]],
    [S.IPSO_COL_TARGET, upload.target_label],
    [S.IPSO_COL_ERROR, upload.error_summary],
  ]));

  const importEl = importTargetPanel(data);
  if (importEl) detailEl.appendChild(importEl);

  const recordsTitle = document.createElement('h3');
  recordsTitle.textContent = S.IPSO_PREVIEW_TITLE(data[RECORD_COUNT] || 0);
  detailEl.appendChild(recordsTitle);
  detailEl.appendChild(recordsTable(data[RECORDS] || []));
}

function isAdmin() {
  return document.body.dataset.role === ROLE_ADMIN;
}

function canImportUpload(upload) {
  return document.body.dataset.role !== ROLE_READER &&
    upload.state === IPSO_UPLOAD_STATE_RECEIVED && !!IMPORT_CONFIG[upload.mode];
}

function importTargetPanel(data) {
  const upload = data[UPLOAD] || {};
  const config = IMPORT_CONFIG[upload.mode];
  if (!config || !canImportUpload(upload)) return null;

  const panel = document.createElement('div');
  panel.className = 'ipso-import-target';

  let select = null;
  if (config.requiresTarget) {
    const label = document.createElement('label');
    label.textContent = config.targetLabel;
    select = document.createElement('select');
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = S.IPSO_TARGET_SELECT;
    select.appendChild(empty);
    for (const target of data[TARGETS] || []) {
      const opt = document.createElement('option');
      opt.value = String(target.id);
      opt.textContent = target.label;
      if (target.id === data[SUGGESTED_TARGET_ID]) opt.selected = true;
      select.appendChild(opt);
    }
    label.appendChild(select);
    panel.appendChild(label);
  }

  const btn = document.createElement('button');
  btn.className = 'btn btn-import';
  btn.textContent = S.IMPORT_LABEL;
  const updateEnabled = () => {
    btn.disabled = config.requiresTarget && !select.value;
  };
  if (select) select.addEventListener('change', updateEnabled);
  btn.addEventListener('click', () => {
    confirmImport(upload.id, config, select ? select.value : null);
  });
  panel.appendChild(btn);
  updateEnabled();
  return panel;
}

function confirmImport(uploadId, config, targetId) {
  showConfirmModal(
    config.confirm,
    async () => importUpload(uploadId, config, targetId),
    { confirmLabel: S.IMPORT_LABEL },
  );
}

async function importUpload(uploadId, config, targetId) {
  const body = config.requiresTarget ? {
    [config.targetField]: Number(targetId),
  } : {};
  const { data, status } = await api.postJSON(config.url(uploadId), body);
  if (status >= 400) {
    await cache.load(DATA_ID);
    await openUpload(uploadId);
    showError(data?.[MESSAGE] || S.ERROR_GENERIC);
    return;
  }
  await cache.load(DATA_ID);
  await openUpload(uploadId);
}

function canRejectUpload(upload) {
  return document.body.dataset.role !== ROLE_READER &&
    upload.state !== IPSO_UPLOAD_STATE_IMPORTED && upload.state !== IPSO_UPLOAD_STATE_REJECTED;
}

function metadataGrid(items) {
  const dl = document.createElement('dl');
  dl.className = 'ipso-meta-grid';
  for (const [label, value] of items) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value == null || value === '' ? S.IPSO_EMPTY_VALUE : String(value);
    dl.append(dt, dd);
  }
  return dl;
}

function recordsTable(records) {
  const host = document.createElement('div');
  host.className = 'ipso-record-preview';
  const rows = records.map(rec => [
    rec.seq, rec.date, rec.parcel, rec[FIELD_SAMPLE_AREA_ID], rec.species,
    rec.number, rec.d_cm, rec.h_m, rec.lat, rec.lon, rec.acc_m,
  ]);
  detailTable = new TableWrapper({
    container: host,
    digest: { columns: PREVIEW_COLUMNS, rows },
    columnDefs: PREVIEW_COLUMN_DEFS,
    canModify: false,
    inlineToolbar: false,
    sort: { column: S.IPSO_COL_SEQ, ascending: true },
    labels: { ...S.TABLE_LABELS, empty: S.IPSO_EMPTY_RECORDS },
    csvFormat: S.TABLE_CSV_FORMAT,
  });
  return host;
}

function coordValue(value) {
  return value == null || value === '' ? S.IPSO_EMPTY_VALUE : fmtCoord(value);
}

function decimal2Value(value) {
  return value == null || value === '' ? S.IPSO_EMPTY_VALUE : fmtDecimal2(value);
}

function intValue(value) {
  return value == null || value === '' ? S.IPSO_EMPTY_VALUE : fmtInt(value);
}

function modeLabel(mode) {
  if (mode === IPSO_MODE_MARTELLATE) return S.IPSO_MODE_MARTELLATE_LABEL;
  if (mode === IPSO_MODE_SAMPLES) return S.IPSO_MODE_SAMPLES_LABEL;
  if (mode === IPSO_MODE_PAI) return S.IPSO_MODE_PAI_LABEL;
  return mode || '';
}

function referenceLabel(referenceVersion) {
  return referenceVersion === 'legacy-converted'
    ? S.IPSO_REFERENCE_LEGACY_CONVERTED
    : referenceVersion || '';
}

function loadingBlock(text) {
  const p = document.createElement('p');
  p.className = 'ipso-loading';
  p.textContent = text;
  return p;
}


function summaryText(data) {
  const rows = data?.[ROWS] || [];
  const pending = data?.[PENDING_COUNT] || 0;
  if (!rows.length) return S.IPSO_SUMMARY_EMPTY;
  return S.IPSO_SUMMARY(rows.length, pending);
}

function updateNavDot(data) {
  const show = (data?.[PENDING_COUNT] || 0) > 0;
  for (const dot of document.querySelectorAll('[data-ipso-pending-dot]')) {
    dot.hidden = !show;
  }
}
