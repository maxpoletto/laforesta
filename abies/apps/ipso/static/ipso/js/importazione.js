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
import * as S from '../../base/js/strings.js';

const DATA_ID = 'ipso_uploads';
const DATA_URL = '/api/ipso/inbox/';
const PAGE_PATH = '/importazione';
const CSS_URL = '/static/ipso/css/importazione.css';
const DEFAULT_SORT = { column: 'Ricevuto', ascending: false };
const DETAIL_URL = (id) => `/api/ipso/uploads/${id}/`;
const REJECT_URL = (id) => `/api/ipso/uploads/${id}/reject/`;
const IMPORT_URL = (id) => `/api/ipso/uploads/${id}/import-martellate/`;

const COLUMN_DEFS = {
  Ricevuto: { label: 'Ricevuto', width: '145px' },
  Modalita: { label: 'Modalita', width: '100px' },
  Operatore: { label: 'Operatore', width: '150px' },
  Record: { label: 'Record', type: 'number', width: '80px', className: 'num' },
  Stato: { label: 'Stato', width: '120px' },
  Pacchetto: { label: 'Pacchetto', width: '170px' },
  Destinazione: { label: 'Destinazione', width: '150px' },
  Errore: { label: 'Errore', width: '240px' },
};

let table = null;
let detailEl = null;
let selectedId = null;

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
  el.replaceChildren();
  selectedId = null;

  const root = document.createElement('div');
  root.className = 'ipso-inbox-page';

  const header = document.createElement('div');
  header.className = 'ipso-inbox-header';
  const title = document.createElement('h1');
  title.textContent = 'Importazione';
  header.appendChild(title);
  const summary = document.createElement('div');
  summary.className = 'ipso-inbox-summary';
  summary.textContent = summaryText(data);
  header.appendChild(summary);
  root.appendChild(header);

  const tableHost = document.createElement('div');
  root.appendChild(tableHost);

  detailEl = document.createElement('section');
  detailEl.className = 'ipso-upload-detail';
  detailEl.hidden = true;
  root.appendChild(detailEl);

  el.appendChild(root);

  const state = readTableState(params);
  table = new TableWrapper({
    container: tableHost,
    digest: data,
    columnDefs: COLUMN_DEFS,
    canModify: true,
    actions: inboxActions(),
    sort: tableSort(state, DEFAULT_SORT),
    searchText: state.searchText,
    csvFilename: 'ipso-importazione.csv',
    labels: {
      ...S.TABLE_LABELS,
      actionEdit: 'Apri',
      actionDelete: 'Rifiuta',
    },
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
  updateNavDot(data);
}

function inboxActions() {
  const canReject = document.body.dataset.role !== 'reader';
  return {
    onEdit: id => openUpload(id),
    ...(canReject ? { onDelete: id => confirmReject(id) } : {}),
  };
}

function destroyPage() {
  if (table) { table.destroy(); table = null; }
  detailEl = null;
  selectedId = null;
}

function applyParams(params) {
  applyTableState(table, readTableState(params), DEFAULT_SORT);
}

function refreshTable(data) {
  table?.setData(data);
  updateNavDot(data);
  const summary = document.querySelector('.ipso-inbox-summary');
  if (summary) summary.textContent = summaryText(data);
}

function syncURL() {
  const params = new URLSearchParams();
  writeTableState(params, table);
  navigateWithParams(PAGE_PATH, params);
}

async function openUpload(id) {
  selectedId = id;
  detailEl.hidden = false;
  detailEl.replaceChildren(loadingBlock('Caricamento dettaglio...'));
  try {
    const { data } = await api.fetchJSON(DETAIL_URL(id));
    if (selectedId !== id) return;
    renderDetail(data);
  } catch {
    showError(S.ERROR_NETWORK);
  }
}

function confirmReject(id) {
  showConfirmModal(
    'Rifiutare questo caricamento Ipso?',
    async () => rejectUpload(id),
    { confirmLabel: 'Rifiuta' },
  );
}

async function rejectUpload(id) {
  const { data, status } = await api.postJSON(REJECT_URL(id), {});
  if (status >= 400) {
    showError(data?.message || S.ERROR_GENERIC);
    return;
  }
  detailEl.hidden = true;
  detailEl.replaceChildren();
  selectedId = null;
  await cache.load(DATA_ID);
}

function renderDetail(data) {
  detailEl.replaceChildren();
  const upload = data.upload || {};

  const header = document.createElement('div');
  header.className = 'ipso-detail-header';
  const title = document.createElement('h2');
  title.textContent = `Sessione ${upload.session_id || upload.id}`;
  header.appendChild(title);
  const actions = document.createElement('div');
  actions.className = 'ipso-detail-actions';
  if (canRejectUpload(upload)) {
    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'btn btn-delete';
    rejectBtn.textContent = 'Rifiuta';
    rejectBtn.addEventListener('click', () => confirmReject(upload.id));
    actions.appendChild(rejectBtn);
  }
  header.appendChild(actions);
  detailEl.appendChild(header);

  if (data.file_error) {
    const warning = document.createElement('p');
    warning.className = 'modal-error';
    warning.textContent = data.file_error;
    detailEl.appendChild(warning);
  }

  detailEl.appendChild(metadataGrid([
    ['Stato', upload.state_label],
    ['Modalita', upload.mode],
    ['Operatore', upload.operator],
    ['Ricevuto', upload.received_at],
    ['Record', upload.record_count],
    ['Reference', upload.reference_version],
    ['Pacchetto', upload.work_package_id],
    ['Destinazione', upload.target_label],
    ['Errore', upload.error_summary],
  ]));

  const importEl = importTargetPanel(data);
  if (importEl) detailEl.appendChild(importEl);

  const recordsTitle = document.createElement('h3');
  recordsTitle.textContent = `Anteprima record (${data.record_count || 0})`;
  detailEl.appendChild(recordsTitle);
  detailEl.appendChild(recordsTable(data.records || []));
}

function canImportUpload(upload) {
  return document.body.dataset.role !== 'reader' &&
    upload.mode === 'martellate' && upload.state === 'received';
}

function importTargetPanel(data) {
  const upload = data.upload || {};
  if (!canImportUpload(upload)) return null;

  const panel = document.createElement('div');
  panel.className = 'ipso-import-target';

  const label = document.createElement('label');
  label.textContent = 'Piano di taglio';
  const select = document.createElement('select');
  const empty = document.createElement('option');
  empty.value = '';
  empty.textContent = 'Seleziona destinazione';
  select.appendChild(empty);
  for (const target of data.targets || []) {
    const opt = document.createElement('option');
    opt.value = String(target.id);
    opt.textContent = target.label;
    if (target.id === data.suggested_target_id) opt.selected = true;
    select.appendChild(opt);
  }
  label.appendChild(select);
  panel.appendChild(label);

  const btn = document.createElement('button');
  btn.className = 'btn btn-import';
  btn.textContent = 'Importa';
  const updateEnabled = () => { btn.disabled = !select.value; };
  select.addEventListener('change', updateEnabled);
  btn.addEventListener('click', () => confirmImport(upload.id, select.value));
  panel.appendChild(btn);
  updateEnabled();
  return panel;
}

function confirmImport(uploadId, targetId) {
  showConfirmModal(
    'Importare questo caricamento nel piano selezionato?',
    async () => importUpload(uploadId, targetId),
    { confirmLabel: 'Importa' },
  );
}

async function importUpload(uploadId, targetId) {
  const { data, status } = await api.postJSON(IMPORT_URL(uploadId), {
    harvest_plan_item_id: Number(targetId),
  });
  if (status >= 400) {
    showError(data?.message || S.ERROR_GENERIC);
    return;
  }
  await cache.load(DATA_ID);
  await openUpload(uploadId);
}

function canRejectUpload(upload) {
  return document.body.dataset.role !== 'reader' &&
    upload.state !== 'imported' && upload.state !== 'rejected';
}

function metadataGrid(items) {
  const dl = document.createElement('dl');
  dl.className = 'ipso-meta-grid';
  for (const [label, value] of items) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value == null || value === '' ? '-' : String(value);
    dl.append(dt, dd);
  }
  return dl;
}

function recordsTable(records) {
  const wrap = document.createElement('div');
  wrap.className = 'ipso-record-preview table-scroll';
  const tableEl = document.createElement('table');
  tableEl.className = 'ipso-preview-table';
  const headers = ['#', 'Data', 'Particella', 'Specie', 'Numero', 'D', 'H', 'Lat', 'Lon', 'Acc.'];
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
      td.textContent = value == null || value === '' ? '-' : String(value);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  if (!records.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = headers.length;
    td.className = 'empty';
    td.textContent = 'Nessun record.';
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

function fmtCoord(value) {
  return Number.isFinite(value) ? value.toFixed(6) : value;
}

function summaryText(data) {
  const rows = data?.rows || [];
  const pending = data?.pending_count || 0;
  if (!rows.length) return 'Nessun caricamento Ipso.';
  return `${rows.length} caricamenti, ${pending} da importare.`;
}

function updateNavDot(data) {
  const show = (data?.pending_count || 0) > 0;
  for (const dot of document.querySelectorAll('[data-ipso-pending-dot]')) {
    dot.hidden = !show;
  }
}
