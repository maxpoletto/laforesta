/** Mannesi page: VDP slips, work hours, production credits, receipts. */

import * as cache from '../../base/js/cache.js';
import { fetchJSON, postJSON } from '../../base/js/api.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  deleteRowWithVersion, fetchModalForm, interceptSubmit, renderModalForm,
  showFormError,
} from '../../base/js/forms.js';
import { dismiss as dismissModal, showError } from '../../base/js/modals.js';
import { canModify } from '../../base/js/roles.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { wireCancelButtons, wireCollapsibleToggle } from '../../base/js/ui-widgets.js';
import {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableParamKeys, tableSort, writeTableState,
} from '../../base/js/page-sync.js';
import { fmtDecimal1, fmtDecimal2, parseDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';
import {
  FIELD_CREW_ID, FIELD_DATE, FIELD_HOURS, FIELD_LICENSE_PLATE, FIELD_MASS_Q,
  FIELD_MONTH, FIELD_NONCE, FIELD_SLIP_COUNT, VERSION,
} from '../../base/js/constants.js';
import { PDFDocument } from './pdf.js';

const CSS_URL = '/static/mannesi/css/mannesi.css';
const PAGE_PATH = '/mannesi';
const API = '/api/mannesi/';
const META_URL = `${API}meta/`;
const LICENSE_SAVE_URL = `${API}license-plates/save/`;
const PRELIEVI_DATA_ID = 'prelievi';
const PRELIEVI_DATA_URL = '/api/prelievi/data/';
const HOURS = {
  dataId: 'mannesi_hours',
  dataUrl: `${API}hours/data/`,
  formUrl: `${API}hours/form/`,
  saveUrl: `${API}hours/save/`,
  deleteUrl: `${API}hours/delete/`,
  csvFilename: S.CSV_MANNESI_HOURS,
  valueField: FIELD_HOURS,
  valueError: S.ERR_HOURS_POSITIVE,
};
const CREDITS = {
  dataId: 'mannesi_credits',
  dataUrl: `${API}credits/data/`,
  formUrl: `${API}credits/form/`,
  saveUrl: `${API}credits/save/`,
  deleteUrl: `${API}credits/delete/`,
  csvFilename: S.CSV_MANNESI_CREDITS,
  valueField: FIELD_MASS_Q,
  valueError: S.ERR_CREDITS_POSITIVE,
};

const SECTION_KEYS = ['v', 'h', 'a', 'r'];
const DEFAULT_OPEN = 'vhar';
const HOURS_KEYS = tableParamKeys('h');
const CREDITS_KEYS = tableParamKeys('c');
const DEFAULT_SORT = { column: S.COL_DATE, ascending: false };

let meta = null;
let hoursTable = null;
let creditsTable = null;
let sectionState = {};

cache.register(PRELIEVI_DATA_ID, PRELIEVI_DATA_URL);
cache.register(HOURS.dataId, HOURS.dataUrl);
cache.register(CREDITS.dataId, CREDITS.dataUrl);

const page = createPage({
  cssUrl: CSS_URL,
  dataIds: [HOURS.dataId, CREDITS.dataId, PRELIEVI_DATA_ID],
  visibleIds: [HOURS.dataId, CREDITS.dataId, PRELIEVI_DATA_ID],
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [
    [HOURS.dataId, data => hoursTable?.setData(data)],
    [CREDITS.dataId, data => creditsTable?.setData(data)],
  ],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

async function loadPageData() {
  const [metaResult, hours, credits, prelievi] = await Promise.all([
    fetchJSON(META_URL),
    cache.load(HOURS.dataId),
    cache.load(CREDITS.dataId),
    cache.load(PRELIEVI_DATA_ID),
  ]);
  return { meta: metaResult.data, hours, credits, prelievi };
}

function mountPage(el, params, data) {
  meta = data.meta;
  el.replaceChildren();
  el.appendChild(cloneTemplate('tmpl-mannesi-page'));

  const p = readParams(params);
  wireSections(el, p.open);
  wireVdpForm(el, data.prelievi);
  wireReceiptForm(el);
  buildEntryTable(HOURS, el.querySelector('[data-target="hours-table"]'), data.hours, p.hours);
  buildEntryTable(CREDITS, el.querySelector('[data-target="credits-table"]'), data.credits, p.credits);
}

function destroyPage() {
  hoursTable?.destroy();
  creditsTable?.destroy();
  hoursTable = null;
  creditsTable = null;
  sectionState = {};
  meta = null;
}

function readParams(params) {
  return {
    open: params.o !== undefined ? params.o : DEFAULT_OPEN,
    hours: readTableState(params, HOURS_KEYS),
    credits: readTableState(params, CREDITS_KEYS),
  };
}

function applyParams(params) {
  const p = readParams(params);
  for (const key of SECTION_KEYS) setSectionOpen(key, p.open.includes(key));
  applyTableState(hoursTable, p.hours, DEFAULT_SORT);
  applyTableState(creditsTable, p.credits, DEFAULT_SORT);
}

function syncURL() {
  const params = new URLSearchParams();
  const open = SECTION_KEYS.filter(k => sectionState[k]?.open).join('');
  if (open !== DEFAULT_OPEN) params.set('o', open);
  writeTableState(params, hoursTable, HOURS_KEYS);
  writeTableState(params, creditsTable, CREDITS_KEYS);
  navigateWithParams(PAGE_PATH, params);
}

function wireSections(el, openKeys) {
  for (const key of SECTION_KEYS) {
    const header = el.querySelector(`[data-section="${key}"].collapsible-header`);
    const body = el.querySelector(`[data-section="${key}"].collapsible-body`);
    sectionState[key] = { header, body, open: false };
    setSectionOpen(key, openKeys.includes(key));
    if (header && body) {
      wireCollapsibleToggle(header, body, (open) => {
        sectionState[key].open = open;
        syncURL();
      });
    }
  }
}

function setSectionOpen(key, open) {
  const s = sectionState[key];
  if (!s) return;
  s.open = open;
  s.header?.classList.toggle('open', open);
  s.body?.classList.toggle('open', open);
}

function buildEntryTable(cfg, container, digest, state) {
  const table = new TableWrapper({
    container,
    digest,
    columnDefs: entryColumnDefs(),
    canModify: canModify(),
    actions: canModify() ? {
      onAdd: () => showEntryForm(cfg),
      onEdit: (rowId) => showEntryForm(cfg, rowId),
      onDelete: (rowId) => confirmDelete(cfg, rowId),
    } : {},
    sort: tableSort(state, DEFAULT_SORT),
    searchText: state.searchText,
    csvFilename: cfg.csvFilename,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
  if (cfg === HOURS) hoursTable = table;
  else creditsTable = table;
}

function entryColumnDefs() {
  return {
    [VERSION]: { label: VERSION, hidden: true },
    [S.COL_DATE]: { label: S.COL_DATE, type: 'date', width: '92px' },
    [S.COL_CREW]: { label: S.COL_CREW, width: '150px' },
    [S.COL_HOURS]: { label: S.COL_HOURS, type: 'number', width: '75px', formatter: fmtDecimal2 },
    [S.COL_CREDITS_Q]: { label: S.COL_CREDITS_Q, type: 'number', width: '90px', formatter: fmtDecimal2 },
    [S.COL_NOTE]: { label: S.COL_NOTE, width: '220px' },
  };
}

async function showEntryForm(cfg, rowId = null) {
  const form = await fetchModalForm(cfg.formUrl + (rowId == null ? '' : `${rowId}/`));
  if (!form) return;
  wireEntryForm(form, cfg);
}

function wireEntryForm(form, cfg) {
  wireCancelButtons(form, dismissModal);
  interceptSubmit(form, cfg.saveUrl, {
    validate: (body) => validateEntry(body, cfg),
    onSuccess: (data) => {
      cache.applyResponseChanges(data);
      dismissModal();
      refreshEntryTable(cfg);
    },
    onConflict: (data) => {
      cache.applyResponseChanges(data);
      refreshEntryTable(cfg);
    },
    onHtml: (html, data) => {
      const newForm = renderModalForm(html);
      if (newForm) {
        wireEntryForm(newForm, cfg);
        showFormError(newForm, data.message || S.ERROR_GENERIC);
      }
    },
  });
}

function validateEntry(body, cfg) {
  if (!body[FIELD_DATE]) return S.ERR_DATE_REQUIRED;
  if (!body[FIELD_CREW_ID]) return S.ERR_CREW_REQUIRED;
  const value = parseDecimal(body[cfg.valueField]);
  if (!Number.isFinite(value) || value <= 0) return cfg.valueError;
  return null;
}

function confirmDelete(cfg, rowId) {
  return deleteRowWithVersion(cfg.dataId, rowId, cfg.deleteUrl, {
    onSuccess: (data) => {
      cache.applyResponseChanges(data);
      refreshEntryTable(cfg);
    },
    onConflict: (data) => {
      cache.applyResponseChanges(data);
      refreshEntryTable(cfg);
    },
  });
}

function refreshEntryTable(cfg) {
  const table = cfg === HOURS ? hoursTable : creditsTable;
  table?.setData(cache.get(cfg.dataId));
}

function wireVdpForm(el, prelievi) {
  const form = el.querySelector('[data-role="vdp-form"]');
  if (!form) return;
  const start = form.querySelector('[name="number"]');
  if (start) start.value = String(defaultStartNumber(prelievi));
  renderPlateDatalist(form);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    const err = validateVdp(body);
    if (err) { showFormError(form, err); return; }

    const plate = normalizePlate(body[FIELD_LICENSE_PLATE]);
    const request = {
      startNumber: parseInt(body.number, 10),
      count: parseInt(body[FIELD_SLIP_COUNT], 10),
      plate,
    };
    if (!canModify()) {
      generateVdpPDF(request);
      return;
    }

    let data, status;
    try {
      ({ data, status } = await postJSON(LICENSE_SAVE_URL, {
        [FIELD_LICENSE_PLATE]: plate,
        [FIELD_NONCE]: crypto.randomUUID(),
      }));
    } catch {
      showFormError(form, S.ERROR_NETWORK);
      return;
    }
    if (status !== 200) {
      showFormError(form, data?.message || S.ERROR_GENERIC);
      return;
    }
    meta.license_plates = data.license_plates || meta.license_plates;
    renderPlateDatalist(form);
    generateVdpPDF(request);
  });
}

function validateVdp(body) {
  const start = parseInt(body.number, 10);
  const count = parseInt(body[FIELD_SLIP_COUNT], 10);
  if (!Number.isInteger(start) || start < 1) return S.ERR_SLIP_COUNT_POSITIVE;
  if (!normalizePlate(body[FIELD_LICENSE_PLATE])) return S.ERR_LICENSE_PLATE_REQUIRED;
  if (!Number.isInteger(count) || count <= 0) return S.ERR_SLIP_COUNT_POSITIVE;
  if (count % 4 !== 0) return S.ERR_SLIP_COUNT_MULTIPLE;
  return null;
}

function renderPlateDatalist(root) {
  const list = root.querySelector('#mannesi-license-plates');
  if (!list || !meta) return;
  list.replaceChildren(...(meta.license_plates || []).map(value => {
    const opt = document.createElement('option');
    opt.value = value;
    return opt;
  }));
}

function normalizePlate(value) {
  return String(value || '').toUpperCase().replace(/\s+/g, '');
}

function defaultStartNumber(prelievi) {
  if (meta?.max_vdp) return meta.max_vdp + 1;
  const idx = prelievi?.columns?.indexOf(S.COL_VDP) ?? -1;
  if (idx < 0) return 1;
  let max = 0;
  for (const row of prelievi.rows || []) max = Math.max(max, parseInt(row[idx], 10) || 0);
  return max + 1;
}

function wireReceiptForm(el) {
  const form = el.querySelector('[data-role="receipts-form"]');
  if (!form) return;
  wireMonthPicker(form);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    const month = body[FIELD_MONTH];
    const receipts = buildReceipts(month);
    if (!receipts.length) {
      showError(S.ERR_RECEIPTS_EMPTY);
      return;
    }
    generateReceiptsPDF(month, receipts);
  });
}

function wireMonthPicker(form) {
  const picker = form.querySelector('[data-role="month-picker"]');
  if (!picker) return;
  const input = picker.querySelector(`[name="${FIELD_MONTH}"]`);
  const trigger = picker.querySelector('.mannesi-month-trigger');
  const popover = picker.querySelector('[data-role="month-popover"]');
  const yearLabel = picker.querySelector('[data-role="month-year"]');
  const grid = picker.querySelector('[data-role="month-grid"]');
  if (!input || !trigger || !popover || !yearLabel || !grid) return;

  let selected = parseMonthValue(input.value) || currentMonth();
  let visibleYear = selected.year;
  setMonthValue(input, trigger, selected);
  renderMonthGrid(grid, visibleYear, selected);
  yearLabel.textContent = String(visibleYear);

  trigger.addEventListener('click', () => {
    setMonthPopoverOpen(trigger, popover, popover.hidden);
  });
  picker.querySelector('[data-action="prev-year"]')?.addEventListener('click', () => {
    visibleYear -= 1;
    yearLabel.textContent = String(visibleYear);
    renderMonthGrid(grid, visibleYear, selected);
  });
  picker.querySelector('[data-action="next-year"]')?.addEventListener('click', () => {
    visibleYear += 1;
    yearLabel.textContent = String(visibleYear);
    renderMonthGrid(grid, visibleYear, selected);
  });
  grid.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-month]');
    if (!btn) return;
    selected = { year: visibleYear, month: parseInt(btn.dataset.month, 10) };
    setMonthValue(input, trigger, selected);
    renderMonthGrid(grid, visibleYear, selected);
    setMonthPopoverOpen(trigger, popover, false);
    trigger.focus();
  });
  picker.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setMonthPopoverOpen(trigger, popover, false);
  });
  form.addEventListener('click', (e) => {
    if (!picker.contains(e.target)) setMonthPopoverOpen(trigger, popover, false);
  });
}

function renderMonthGrid(grid, year, selected) {
  grid.replaceChildren(...Array.from({ length: 12 }, (_, i) => {
    const month = i + 1;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mannesi-month-option';
    btn.dataset.month = String(month);
    btn.textContent = monthName(month, 'short');
    btn.title = monthName(month, 'long');
    btn.classList.toggle('active', selected.year === year && selected.month === month);
    return btn;
  }));
}

function setMonthValue(input, trigger, value) {
  input.value = monthValue(value);
  trigger.textContent = monthLabel(input.value);
}

function setMonthPopoverOpen(trigger, popover, open) {
  popover.hidden = !open;
  trigger.setAttribute('aria-expanded', String(open));
}

function parseMonthValue(value) {
  const match = /^(\d{4})-(\d{2})$/.exec(value || '');
  if (!match) return null;
  const year = parseInt(match[1], 10);
  const month = parseInt(match[2], 10);
  return month >= 1 && month <= 12 ? { year, month } : null;
}

function currentMonth() {
  const now = new Date();
  return { year: now.getFullYear(), month: now.getMonth() + 1 };
}

function monthValue({ year, month }) {
  return `${year}-${String(month).padStart(2, '0')}`;
}

function monthName(month, style) {
  const d = new Date(Date.UTC(2020, month - 1, 1));
  const raw = new Intl.DateTimeFormat('it', { month: style, timeZone: 'UTC' }).format(d);
  return raw.replace('.', '');
}

// ---------------------------------------------------------------------------
// VDP PDF
// ---------------------------------------------------------------------------

function generateVdpPDF({ startNumber, count, plate }) {
  const doc = new PDFDocument();
  const w = doc.width / 2;
  const h = doc.height / 2;
  for (let i = 0; i < count; i++) {
    if (i > 0 && i % 4 === 0) doc.addPage();
    const slot = i % 4;
    drawSlip(doc, (slot % 2) * w, Math.floor(slot / 2) * h, w, h, startNumber + i, plate);
  }
  doc.save('vdp.pdf');
}

function drawSlip(doc, x, y, w, h, number, plate) {
  const left = x + 18;
  const right = x + w - 18;
  const innerWidth = right - left;
  let ruleStart = left + 55;
  let yy = y + 28;
  doc.rect(x + 8, y + 8, w - 16, h - 16);

  drawRuleField(doc, left, left + 144, yy, 'Data', { ruleStart: ruleStart, size: 10 });
  doc.textRight(right, yy, `N. ${number}`, { size: 11, bold: true });

  yy += 22;
  doc.text(left, yy, 'Targa', { size: 10, bold: true });
  doc.text(ruleStart, yy, plate, { size: 10 });

  yy += 24;
  drawRegionOptions(doc, left, right, yy, meta.regions || []);

  yy += 24;
  drawRuleField(doc, left, right, yy, 'Particella', { ruleStart: ruleStart, size: 9 });

  yy += 22;
  yy = drawProductOptions(doc, left, right, yy, meta.products || []);

  yy += 8;
  yy = drawSpeciesGrid(doc, left, yy, innerWidth, meta.species || []);

  yy = Math.max(yy + 12, y + h - 122);
  ruleStart = left + 96;
  drawRuleField(doc, left, right, yy, 'Peso lordo ql', { ruleStart, size: 10 });
  yy += 18;
  drawRuleField(doc, left, right, yy, 'Tara ql', { ruleStart, size: 10 });
  yy += 18;
  drawRuleField(doc, left, right, yy, 'Peso netto ql', { ruleStart, size: 10 });
  yy += 25;
  drawRuleField(doc, left, right, yy, 'Squadra', { ruleStart, size: 10 });
  yy += 25;
  drawRuleField(doc, left, right, yy, 'Firma', { ruleStart, size: 10 });
}

function drawRegionOptions(doc, left, right, y, regions) {
  doc.text(left, y, 'Compresa', { size: 9, bold: true });
  const startX = left + 55;
  const step = regions.length > 1 ? (right - startX - 52) / (regions.length - 1) : 0;
  regions.forEach((name, i) => {
    const x = startX + i * Math.max(60, step);
    drawCheckbox(doc, x, y - 7);
    doc.text(x + 11, y, clippedForWidth(name, 50, 9), { size: 9 });
  });
}

function drawProductOptions(doc, left, right, y, products) {
  doc.text(left, y, 'Tipo', { size: 9, bold: true });
  const colGap = 12;
  const startX = left + 55;
  const colWidth = (right - startX - colGap) / 2;
  const rowHeight = 13;
  const maxRows = Math.max(1, Math.ceil(products.length / 2));
  products.forEach((name, i) => {
    const col = Math.floor(i / maxRows);
    const row = i % maxRows;
    const x = startX + col * (colWidth + colGap);
    const yy = y + row * rowHeight;
    drawCheckbox(doc, x, yy - 7);
    doc.text(x + 11, yy, clippedForWidth(name, colWidth - 13, 8.5), { size: 8.5 });
  });
  return y + Math.max(1, Math.ceil(products.length / 2)) * rowHeight;
}

function drawSpeciesGrid(doc, x, y, width, species) {
  const rows = species.length + 1;
  const rowHeight = Math.min(12, Math.max(10, 108 / Math.max(1, rows)));
  const height = rows * rowHeight;
  const pctWidth = 100;
  const nameWidth = width - pctWidth;

  doc.rect(x, y, width, height);
  doc.line(x + nameWidth, y, x + nameWidth, y + height);
  for (let i = 1; i < rows; i++) {
    doc.line(x, y + i * rowHeight, x + width, y + i * rowHeight);
  }

  doc.text(x + 5, y + rowHeight - 4, 'Essenza', { size: 8, bold: true });
  doc.textRight(x + width - 6, y + rowHeight - 4, '%', { size: 8, bold: true });
  species.forEach((name, i) => {
    const yy = y + (i + 2) * rowHeight - 4;
    doc.text(x + 5, yy, clippedForWidth(name, nameWidth - 10, 8), { size: 8 });
  });
  return y + height;
}

function drawRuleField(doc, left, right, y, label, { ruleStart, size = 9 } = {}) {
  doc.text(left, y, label, { size, bold: true });
  doc.line(ruleStart, y + 2, right, y + 2);
}

function drawCheckbox(doc, x, y, size = 7) {
  doc.rect(x, y, size, size);
}

function clippedForWidth(value, width, size) {
  return clip(value, Math.floor(width / (size * 0.52)));
}

// ---------------------------------------------------------------------------
// Receipt PDF
// ---------------------------------------------------------------------------

function buildReceipts(month) {
  const prelievi = cache.get(PRELIEVI_DATA_ID);
  const hours = cache.get(HOURS.dataId);
  const credits = cache.get(CREDITS.dataId);
  if (!prelievi || !hours || !credits || !month) return [];

  const pc = colMap(prelievi.columns);
  const hc = colMap(hours.columns);
  const cc = colMap(credits.columns);
  const harvests = prelievi.rows.filter(row => String(row[pc[S.COL_DATE]] || '').startsWith(month));
  const teams = [...new Set(harvests.map(row => row[pc[S.COL_CREW]]).filter(Boolean))]
    .sort((a, b) => String(a).localeCompare(String(b), 'it'));

  return teams.map(crew => {
    const crewHarvests = harvests.filter(row => row[pc[S.COL_CREW]] === crew);
    const productTotals = productNames(crewHarvests, pc).map(product => ({
      product,
      mass: sum(crewHarvests.filter(row => row[pc[S.COL_TYPE]] === product), pc[S.COL_QUINTALS]),
    }));
    const totalProduction = sum(crewHarvests, pc[S.COL_QUINTALS]);
    const hoursRows = hours.rows.filter(row => row[hc[S.COL_CREW]] === crew && String(row[hc[S.COL_DATE]] || '').startsWith(month));
    const creditRows = credits.rows.filter(row => row[cc[S.COL_CREW]] === crew && String(row[cc[S.COL_DATE]] || '').startsWith(month));
    return {
      crew,
      hours: sum(hoursRows, hc[S.COL_HOURS]),
      productTotals,
      totalProduction,
      credits: sum(creditRows, cc[S.COL_CREDITS_Q]),
      harvests: crewHarvests,
      columns: pc,
    };
  });
}

function productNames(rows, pc) {
  const configured = meta.products || [];
  const seen = new Set(rows.map(row => row[pc[S.COL_TYPE]]).filter(Boolean));
  return [...configured, ...[...seen].filter(name => !configured.includes(name))];
}

function generateReceiptsPDF(month, receipts) {
  const doc = new PDFDocument({ landscape: true });
  receipts.forEach((receipt, index) => {
    if (index > 0) doc.addPage();
    drawReceipt(doc, month, receipt);
  });
  doc.save(`ricevute-mannesi-${month}.pdf`);
}

const margin = 34;
function drawReceipt(doc, month, receipt) {
  const col1 = margin, col2 = margin + 150;
  const valueComma = col2 + 44;
  let y = 32;
  doc.text(col1, y, `Squadra ${receipt.crew}`, { size: 14, bold: true });
  y += 22;
  doc.text(col1, y, monthLabel(month), { size: 11 });
  y += 34;
  doc.text(col1, y, 'Ore lavorate', { size: 10, bold: true });
  drawDecimal(doc, valueComma, y, fmtDecimal2(receipt.hours), { size: 10 });
  y += 28;
  doc.text(col1, y, 'Produzione', { size: 10, bold: true });
  doc.text(col2, y, 'Quintali', { size: 10, bold: true });
  y += 16;
  for (const item of receipt.productTotals) {
    doc.text(col1, y, item.product, { size: 10 });
    drawDecimal(doc, valueComma, y, fmtDecimal1(item.mass), { size: 10 });
    y += 14;
  }
  y += 4;
  doc.text(col1, y, 'Totale produzione', { size: 10, bold: true });
  drawDecimal(doc, valueComma, y, fmtDecimal1(receipt.totalProduction), { size: 10, bold: true });
  y += 28;
  doc.text(col1, y, 'Acconti', { size: 10 });
  drawDecimal(doc, valueComma, y, fmtDecimal1(receipt.credits), { size: 10 });
  y += 18;
  doc.text(col1, y, 'Totale', { size: 10, bold: true });
  drawDecimal(doc, valueComma, y, fmtDecimal1(receipt.totalProduction - receipt.credits), { size: 10, bold: true });
  y += 34;
  y = drawHarvestDetail(doc, receipt, margin, y, month);
}

function drawHarvestDetail(doc, receipt, x, y, month) {
  doc.text(x, y, 'Dettaglio produzione', { size: 10, bold: true });
  y += 18;
  const species = meta.species || [];
  const headers = ['Data', 'Compresa', 'Particella', 'VDP', 'Tipo', 'Q.li', 'Note', ...species.map(s => `${s} %`)];
  const widths = receiptTableWidths(doc, species.length);
  const alignments = receiptTableAlignments(species.length);
  y = drawTableRow(doc, x, y, headers, widths, true, alignments);

  for (const row of receipt.harvests) {
    if (y > doc.height - 32) {
      doc.addPage();
      y = 32;
      doc.text(x, y, `Squadra ${receipt.crew} - ${monthLabel(month)}`, { size: 10, bold: true });
      y += 18;
      y = drawTableRow(doc, x, y, headers, widths, true, alignments);
    }
    const c = receipt.columns;
    const note = [row[c[S.COL_NOTE]], row[c[S.COL_EXTRA_NOTE]]].filter(Boolean).join('; ');
    const fields = [
      row[c[S.COL_DATE]], row[c[S.COL_COMPRESA]], row[c[S.COL_PARCEL]],
      row[c[S.COL_VDP]], row[c[S.COL_TYPE]], fmtDecimal1(row[c[S.COL_QUINTALS]]),
      note,
      ...species.map(s => formatMaybe(row[c[`${s} %`]])),
    ];
    y = drawTableRow(doc, x, y, fields, widths, false, alignments);
  }
  return y;
}

function receiptTableWidths(doc, speciesCount) {
  const available = doc.width - 2 * margin;
  const base = [45, 45, 40, 30, 60, 30, 80];
  const baseTotal = base.reduce((a, b) => a + b, 0);
  const speciesWidth = speciesCount
    ? Math.max(30, Math.min(60, Math.floor((available - baseTotal) / speciesCount)))
    : 0;
  return [...base, ...Array.from({ length: speciesCount }, () => speciesWidth)];
}

function receiptTableAlignments(speciesCount) {
  return [
    'left', 'left', 'left', 'left', 'left', 'decimal', 'left',
    ...Array.from({ length: speciesCount }, () => 'decimal'),
  ];
}

function drawTableRow(doc, x, y, fields, widths, bold, alignments = []) {
  const size = bold ? 7 : 6.5;
  const rowHeight = bold ? 11 : 10;
  const rightPad = 10;
  let xx = x;
  for (let i = 0; i < fields.length; i++) {
    const text = clip(fields[i], Math.floor(widths[i] / (size * 0.52)));
    const align = alignments[i] || 'left';
    if (align === 'decimal' && !bold) {
      const commaX = xx + widths[i] - rightPad - doc.textWidth(',0', { size, bold });
      drawDecimal(doc, commaX, y, text, { size, bold });
    } else if (align === 'decimal' || align === 'right') {
      doc.textRight(xx + widths[i] - rightPad, y, text, { size, bold });
    } else {
      doc.text(xx, y, text, { size, bold });
    }
    xx += widths[i];
  }
  y += rowHeight;
  if (bold) doc.line(x, y-8, x + widths.reduce((a, b) => a + b, 0), y-8);
  return y;
}

function drawDecimal(doc, commaX, y, value, opts) {
  const text = String(value ?? '');
  const comma = text.indexOf(',');
  if (comma < 0) {
    doc.textRight(commaX, y, text, opts);
    return;
  }
  doc.textRight(commaX, y, text.slice(0, comma), opts);
  doc.text(commaX, y, text.slice(comma), opts);
}

function colMap(columns) {
  const out = {};
  columns.forEach((name, i) => { out[name] = i; });
  return out;
}

function sum(rows, idx) {
  if (idx == null || idx < 0) return 0;
  return rows.reduce((total, row) => total + (Number(row[idx]) || 0), 0);
}

function formatMaybe(value) {
  return value == null || value === '' ? '' : fmtDecimal1(value);
}

function clip(value, max) {
  const s = String(value ?? '');
  return s.length <= max ? s : `${s.slice(0, Math.max(0, max - 1))}...`;
}

function monthLabel(month) {
  const [year, monthNum] = month.split('-').map(v => parseInt(v, 10));
  const d = new Date(Date.UTC(year, monthNum - 1, 1));
  let l = new Intl.DateTimeFormat('it', { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d);
  return l.charAt(0).toUpperCase() + l.slice(1);
}
