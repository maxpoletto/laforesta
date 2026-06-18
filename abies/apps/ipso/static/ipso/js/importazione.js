/**
 * Ipso staged-upload inbox page.
 */

import * as api from '../../base/js/api.js';
import * as cache from '../../base/js/cache.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableSort, writeTableState,
} from '../../base/js/page-sync.js';
import { showError } from '../../base/js/modals.js';
import { showConfirmModal } from '../../base/js/ui-widgets.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { fmtCoord } from '../../base/js/format.js';
import { installEscapeHandler } from '../../base/js/escape.js';
import * as S from '../../base/js/strings.js';
import {
  DATA_ID_IPSO_UPLOADS, FIELD_HARVEST_PLAN_ITEM_ID, FIELD_SURVEY_ID, FILE_ERROR,
  IPSO_MODE_MARTELLATE, IPSO_MODE_PAI, IPSO_MODE_SAMPLES,
  IPSO_UPLOAD_STATE_IMPORTED, IPSO_UPLOAD_STATE_RECEIVED, IPSO_UPLOAD_STATE_REJECTED,
  MESSAGE, PENDING_COUNT, RECORD_COUNT, RECORDS, ROLE_READER, ROWS,
  SUGGESTED_TARGET_ID, TARGETS, UPLOAD,
} from '../../base/js/constants.js';

const DATA_ID = DATA_ID_IPSO_UPLOADS;
const DATA_URL = '/api/ipso/inbox/';
const PAGE_PATH = '/importazione';
const CSS_URL = '/static/ipso/css/importazione.css';
const DEFAULT_SORT = { column: S.IPSO_COL_RECEIVED, ascending: false };
const DETAIL_URL = (id) => `/api/ipso/uploads/${id}/`;
const REJECT_URL = (id) => `/api/ipso/uploads/${id}/reject/`;
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
  [S.IPSO_COL_RECEIVED]: { label: S.IPSO_COL_RECEIVED, width: '145px' },
  [S.IPSO_COL_MODE]: { label: S.IPSO_COL_MODE, width: '100px' },
  [S.IPSO_COL_OPERATOR]: { label: S.IPSO_COL_OPERATOR, width: '150px' },
  [S.IPSO_COL_RECORDS]: { label: S.IPSO_COL_RECORDS, type: 'number', width: '80px', className: 'num' },
  [S.IPSO_COL_STATE]: { label: S.IPSO_COL_STATE, width: '120px' },
  [S.IPSO_COL_WORK_PACKAGE]: { label: S.IPSO_COL_WORK_PACKAGE, width: '170px' },
  [S.IPSO_COL_TARGET]: { label: S.IPSO_COL_TARGET, width: '150px' },
  [S.IPSO_COL_ERROR]: { label: S.IPSO_COL_ERROR, width: '240px' },
};

let table = null;
let detailEl = null;
let summaryEl = null;
let selectedId = null;
let disposeEscape = null;

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
  updateNavDot(data);
}

function inboxActions() {
  return {
    onEdit: id => openUpload(id),
  };
}

function destroyPage() {
  if (disposeEscape) { disposeEscape(); disposeEscape = null; }
  if (table) { table.destroy(); table = null; }
  detailEl = null;
  summaryEl = null;
  selectedId = null;
}

function applyParams(params) {
  applyTableState(table, readTableState(params), DEFAULT_SORT);
}

function refreshTable(data) {
  table?.setData(data);
  updateNavDot(data);
  if (summaryEl) summaryEl.textContent = summaryText(data);
}

function syncURL() {
  const params = new URLSearchParams();
  writeTableState(params, table);
  navigateWithParams(PAGE_PATH, params);
}

async function openUpload(id) {
  selectedId = id;
  detailEl.hidden = false;
  detailEl.replaceChildren(loadingBlock(S.IPSO_LOADING_DETAIL));
  try {
    const { data } = await api.fetchJSON(DETAIL_URL(id));
    if (selectedId !== id) return;
    renderDetail(data);
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function closeDetail() {
  if (!detailEl || detailEl.hidden) return;
  detailEl.hidden = true;
  detailEl.replaceChildren();
  selectedId = null;
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

function renderDetail(data) {
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
    [S.IPSO_COL_MODE, upload.mode],
    [S.IPSO_COL_OPERATOR, upload.operator],
    [S.IPSO_COL_RECEIVED, upload.received_at],
    [S.IPSO_COL_RECORDS, upload.record_count],
    [S.IPSO_COL_REFERENCE, upload.reference_version],
    [S.IPSO_COL_WORK_PACKAGE, upload.work_package_id],
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
  const wrap = document.createElement('div');
  wrap.className = 'ipso-record-preview table-scroll';
  const tableEl = document.createElement('table');
  tableEl.className = 'ipso-preview-table';
  const headers = [
    S.IPSO_COL_SEQ, S.COL_DATE, S.COL_PARCEL, S.COL_SPECIES, S.COL_NUMBER,
    S.COL_D_CM, S.COL_H_M, S.COL_LAT, S.COL_LON, S.IPSO_COL_ACCURACY,
  ];
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  for (const label of headers) {
    const th = document.createElement('th');
    th.textContent = label;
    trh.appendChild(th);
  }
  thead.appendChild(trh);
  tableEl.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const rec of records) {
    const tr = document.createElement('tr');
    for (const value of [
      rec.seq, rec.date, rec.parcel, rec.species, rec.number,
      rec.d_cm, rec.h_m, fmtCoord(rec.lat), fmtCoord(rec.lon), rec.acc_m,
    ]) {
      const td = document.createElement('td');
      td.textContent = value == null || value === '' ? S.IPSO_EMPTY_VALUE : String(value);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  if (!records.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = headers.length;
    td.className = 'empty';
    td.textContent = S.IPSO_EMPTY_RECORDS;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  tableEl.appendChild(tbody);
  wrap.appendChild(tableEl);
  return wrap;
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
