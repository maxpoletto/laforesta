/** Squadre page: personnel, work hours, production credits, and reports. */

import * as cache from '../../base/js/cache.js';
import { fetchJSON } from '../../base/js/api.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  deleteRowWithVersion, fetchModalForm, interceptSubmit, renderModalForm,
  showFormError,
} from '../../base/js/forms.js';
import { dismiss as dismissModal, showError } from '../../base/js/modals.js';
import { canModify } from '../../base/js/roles.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { showLoadingIn, wireCancelButtons, wireCollapsibleToggle } from '../../base/js/ui-widgets.js';
import {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableParamKeys, tableSort, writeTableState,
} from '../../base/js/page-sync.js';
import { columnMap } from '../../base/js/digests.js';
import { fmtDecimal1, fmtDecimal2, parseDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';
import {
  FIELD_CREW_ID, FIELD_DATE, FIELD_HOURS, FIELD_MASS_Q, FIELD_MONTH, FIELD_NAME,
  VERSION,
} from '../../base/js/constants.js';
import { decimalRight, PDFDocument } from './pdf.js';

const CSS_URL = '/static/squadre/css/squadre.css';
const PAGE_PATH = '/squadre';
const API = '/api/squadre/';
const META_URL = `${API}meta/`;
const PRELIEVI_DATA_ID = 'prelievi';
const PRELIEVI_DATA_URL = '/api/prelievi/data/';
const PERSONNEL = {
  dataId: 'crews',
  dataUrl: `${API}crews/data/`,
  formUrl: `${API}crews/form/`,
  saveUrl: `${API}crews/save/`,
  csvFilename: S.CSV_CREWS,
};
const HOURS = {
  dataId: 'squadre_hours',
  dataUrl: `${API}hours/data/`,
  formUrl: `${API}hours/form/`,
  saveUrl: `${API}hours/save/`,
  deleteUrl: `${API}hours/delete/`,
  csvFilename: S.CSV_SQUADRE_HOURS,
  valueField: FIELD_HOURS,
  valueError: S.ERR_HOURS_POSITIVE,
};
const CREDITS = {
  dataId: 'squadre_credits',
  dataUrl: `${API}credits/data/`,
  formUrl: `${API}credits/form/`,
  saveUrl: `${API}credits/save/`,
  deleteUrl: `${API}credits/delete/`,
  csvFilename: S.CSV_SQUADRE_CREDITS,
  valueField: FIELD_MASS_Q,
  valueError: S.ERR_CREDITS_POSITIVE,
};

const SECTION_KEYS = ['p', 'h', 'a', 'r'];
const DEFAULT_OPEN = 'har';
const HOURS_KEYS = tableParamKeys('h');
const CREDITS_KEYS = tableParamKeys('c');
const DEFAULT_SORT = { column: S.COL_DATE, ascending: false };

let meta = null;
let personnelTable = null;
let personnelDigest = null;
let personnelLoaded = false;
let personnelActiveOnly = true;
let hoursTable = null;
let creditsTable = null;
let sectionState = {};

cache.register(PERSONNEL.dataId, PERSONNEL.dataUrl);
cache.register(PRELIEVI_DATA_ID, PRELIEVI_DATA_URL);
cache.register(HOURS.dataId, HOURS.dataUrl);
cache.register(CREDITS.dataId, CREDITS.dataUrl);

const page = createPage({
  cssUrl: CSS_URL,
  dataIds: [HOURS.dataId, CREDITS.dataId, PRELIEVI_DATA_ID],
  visibleIds: [PERSONNEL.dataId, HOURS.dataId, CREDITS.dataId, PRELIEVI_DATA_ID],
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [
    [PERSONNEL.dataId, updatePersonnelTable],
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
  el.appendChild(cloneTemplate('tmpl-squadre-page'));

  const p = readParams(params);
  const open = canModify() ? p.open : p.open.replace('p', '');
  if (!canModify()) removePersonnelSection(el);
  wireSections(el, open);
  if (canModify()) wirePersonnelSection(el, open.includes('p'));
  wireReportForm(el);
  buildEntryTable(HOURS, el.querySelector('[data-target="hours-table"]'), data.hours, p.hours);
  buildEntryTable(CREDITS, el.querySelector('[data-target="credits-table"]'), data.credits, p.credits);
}

function destroyPage() {
  personnelTable?.destroy();
  hoursTable?.destroy();
  creditsTable?.destroy();
  personnelTable = null;
  personnelDigest = null;
  personnelLoaded = false;
  personnelActiveOnly = true;
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
  const open = canModify() ? p.open : p.open.replace('p', '');
  for (const key of SECTION_KEYS) setSectionOpen(key, open.includes(key));
  if (open.includes('p')) {
    loadPersonnelSection(document.querySelector('[data-target="personnel-table"]'));
  }
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
        if (key === 'p' && open) {
          loadPersonnelSection(body.querySelector('[data-target="personnel-table"]'));
        }
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

function removePersonnelSection(el) {
  el.querySelector('[data-section="p"].collapsible-header')?.remove();
  el.querySelector('[data-section="p"].collapsible-body')?.remove();
}

function wirePersonnelSection(el, initiallyOpen) {
  const activeCheck = el.querySelector('[data-role="personnel-active-toggle"]');
  activeCheck?.addEventListener('change', () => {
    personnelActiveOnly = activeCheck.checked;
    applyPersonnelActiveFilter();
  });
  if (initiallyOpen) loadPersonnelSection(el.querySelector('[data-target="personnel-table"]'));
}

async function loadPersonnelSection(container) {
  if (!container || personnelLoaded) return;
  personnelLoaded = true;
  showLoadingIn(container);

  let data;
  try {
    data = await cache.load(PERSONNEL.dataId);
  } catch {
    personnelLoaded = false;
    container.replaceChildren();
    showError(S.ERROR_NETWORK);
    return;
  }

  container.replaceChildren();
  personnelDigest = data;
  personnelTable = new TableWrapper({
    container,
    digest: data,
    columnDefs: personnelColumnDefs(),
    canModify: true,
    actions: {
      onAdd: () => showPersonnelForm(),
      onEdit: (rowId) => showPersonnelForm(rowId),
    },
    csvFilename: PERSONNEL.csvFilename,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
  });
  applyPersonnelActiveFilter();
}

function personnelColumnDefs() {
  return {
    [S.LABEL_NAME]: { label: S.LABEL_NAME, width: '180px' },
    [S.LABEL_NOTES]: { label: S.LABEL_NOTES, width: '260px' },
    [S.COL_ACTIVE]: { label: S.COL_ACTIVE, type: 'boolean', width: '60px' },
  };
}

async function showPersonnelForm(rowId = null) {
  const form = await fetchModalForm(PERSONNEL.formUrl + (rowId == null ? '' : `${rowId}/`));
  if (!form) return;
  wirePersonnelForm(form);
}

function wirePersonnelForm(form) {
  wireCancelButtons(form, dismissModal);
  interceptSubmit(form, PERSONNEL.saveUrl, {
    validate: (body) => (String(body[FIELD_NAME] || '').trim() ? null : S.ERR_NAME_REQUIRED),
    onSuccess: (data) => {
      updatePersonnelFromResponse(data);
      dismissModal();
    },
    onConflict: updatePersonnelFromResponse,
  });
}

function updatePersonnelFromResponse(data) {
  cache.applyResponseChanges(data);
  updatePersonnelTable(cache.get(PERSONNEL.dataId));
}

function updatePersonnelTable(data) {
  if (!data) return;
  personnelDigest = data;
  personnelTable?.setData(data);
  applyPersonnelActiveFilter();
}

function applyPersonnelActiveFilter() {
  if (!personnelTable || !personnelDigest) return;
  const activeIdx = personnelDigest.columns.indexOf(S.COL_ACTIVE);
  if (activeIdx < 0 || !personnelActiveOnly) {
    personnelTable.setExternalFilter(null);
  } else {
    personnelTable.setExternalFilter(row => row[activeIdx] === true);
  }
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

function wireReportForm(el) {
  const form = el.querySelector('[data-role="reports-form"]');
  if (!form) return;
  wireMonthPicker(form);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    const month = body[FIELD_MONTH];
    const reports = buildReports(month);
    if (!reports.length) {
      showError(S.ERR_REPORTS_EMPTY);
      return;
    }
    generateReportsPDF(month, reports);
  });
}

function wireMonthPicker(form) {
  const picker = form.querySelector('[data-role="month-picker"]');
  if (!picker) return;
  const input = picker.querySelector(`[name="${FIELD_MONTH}"]`);
  const trigger = picker.querySelector('.squadre-month-trigger');
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
    btn.className = 'squadre-month-option';
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

function previousMonthValue(month) {
  const parsed = typeof month === 'string' ? parseMonthValue(month) : month;
  if (!parsed) return null;
  return parsed.month === 1
    ? monthValue({ year: parsed.year - 1, month: 12 })
    : monthValue({ year: parsed.year, month: parsed.month - 1 });
}

function sameMonth(value, month) {
  return Boolean(month) && String(value || '').startsWith(month);
}

function monthName(month, style) {
  const d = new Date(Date.UTC(2020, month - 1, 1));
  const raw = new Intl.DateTimeFormat(S.LOCALE, { month: style, timeZone: 'UTC' }).format(d);
  return raw.replace('.', '');
}

// ---------------------------------------------------------------------------
// Report PDF
// ---------------------------------------------------------------------------

function buildReports(month) {
  return buildReportsFromDigests(
    month, meta, cache.get(PRELIEVI_DATA_ID),
    cache.get(HOURS.dataId), cache.get(CREDITS.dataId),
  );
}

export function buildReportsFromDigests(month, metaData, prelievi, hours, credits) {
  if (!prelievi || !hours || !credits || !parseMonthValue(month)) return [];

  const pc = columnMap(prelievi.columns);
  const hc = columnMap(hours.columns);
  const cc = columnMap(credits.columns);
  const previousMonth = previousMonthValue(month);
  const harvests = prelievi.rows.filter(row => sameMonth(row[pc[S.COL_DATE]], month));
  const teams = [...new Set(harvests.map(row => row[pc[S.COL_CREW]]).filter(Boolean))]
    .sort((a, b) => String(a).localeCompare(String(b), S.LOCALE));

  return teams.map(crew => {
    const crewHarvests = harvests.filter(row => row[pc[S.COL_CREW]] === crew);
    const productTotals = productNames(crewHarvests, pc, metaData).map(product => ({
      product,
      mass: sum(crewHarvests.filter(row => row[pc[S.COL_TYPE]] === product), pc[S.COL_QUINTALS]),
    }));
    const totalProduction = sum(crewHarvests, pc[S.COL_QUINTALS]);
    const hoursRows = hours.rows.filter(row =>
      row[hc[S.COL_CREW]] === crew && sameMonth(row[hc[S.COL_DATE]], month));
    const currentCreditRows = credits.rows.filter(row =>
      row[cc[S.COL_CREW]] === crew && sameMonth(row[cc[S.COL_DATE]], month));
    const previousCreditRows = credits.rows.filter(row =>
      row[cc[S.COL_CREW]] === crew && sameMonth(row[cc[S.COL_DATE]], previousMonth));
    return {
      crew,
      hours: sum(hoursRows, hc[S.COL_HOURS]),
      productTotals,
      totalProduction,
      credits: (
        sum(currentCreditRows, cc[S.COL_CREDITS_Q]) -
        sum(previousCreditRows, cc[S.COL_CREDITS_Q])
      ),
      harvests: crewHarvests,
      columns: pc,
    };
  });
}

function productNames(rows, pc, metaData = meta) {
  const configured = metaData?.products || [];
  const seen = new Set(rows.map(row => row[pc[S.COL_TYPE]]).filter(Boolean));
  return [...configured, ...[...seen].filter(name => !configured.includes(name))];
}

function generateReportsPDF(month, reports) {
  const doc = new PDFDocument({ landscape: true });
  reports.forEach((report, index) => {
    if (index > 0) doc.addPage();
    drawReport(doc, month, report);
  });
  doc.save(S.PDF_SQUADRE_REPORTS(month));
}

const margin = 34;
export function drawReport(doc, month, report) {
  const col1 = margin, col2 = margin + 150;
  const valueComma = col2 + 44;
  let y = 32;
  doc.text(col1, y, `${S.COL_CREW} ${report.crew}`, { size: 14, bold: true });
  y += 22;
  doc.text(col1, y, monthLabel(month), { size: 11 });
  y += 34;
  doc.text(col1, y, S.SQUADRE_REPORT_HOURS, { size: 10, bold: true });
  doc.textRight(decimalRight(doc, valueComma), y, fmtDecimal2(report.hours), { size: 10 });
  y += 28;
  doc.text(col1, y, S.SQUADRE_REPORT_PRODUCTION, { size: 10, bold: true });
  doc.textRight(decimalRight(doc, valueComma), y, S.COL_CREDITS_Q, { size: 10, bold: true });
  y += 16;
  for (const item of report.productTotals) {
    doc.text(col1, y, item.product, { size: 10 });
    drawDecimal(doc, valueComma, y, fmtDecimal1(item.mass), { size: 10 });
    y += 14;
  }
  y += 4;
  doc.text(col1, y, S.SQUADRE_REPORT_TOTAL_PRODUCTION, { size: 10, bold: true });
  drawDecimal(doc, valueComma, y, fmtDecimal1(report.totalProduction), { size: 10, bold: true });
  y += 28;
  doc.text(col1, y, S.SQUADRE_REPORT_CREDITS, { size: 10 });
  drawDecimal(doc, valueComma, y, fmtDecimal1(report.credits), { size: 10 });
  y += 18;
  doc.text(col1, y, S.SQUADRE_REPORT_TOTAL, { size: 10, bold: true });
  drawDecimal(
    doc, valueComma, y, fmtDecimal1(report.totalProduction + report.credits),
    { size: 10, bold: true },
  );
  y += 34;
  y = drawHarvestDetail(doc, report, margin, y, month);
}

function drawHarvestDetail(doc, report, x, y, month) {
  doc.text(x, y, S.SQUADRE_REPORT_DETAIL, { size: 10, bold: true });
  y += 18;
  const species = meta?.species || [];
  const headers = [
    S.COL_DATE, S.COL_REGION, S.COL_PARCEL, S.COL_VDP, S.COL_TYPE, S.COL_QUINTALS, S.COL_NOTE,
    ...species.map(s => `${s} ${S.LABEL_PERCENT}`),
  ];
  const widths = reportTableWidths(doc, species.length);
  const alignments = reportTableAlignments(species.length);
  y = drawTableRow(doc, x, y, headers, widths, true, alignments);

  for (const row of report.harvests) {
    if (y > doc.height - 32) {
      doc.addPage();
      y = 32;
      doc.text(x, y, `${S.COL_CREW} ${report.crew} - ${monthLabel(month)}`, { size: 10, bold: true });
      y += 18;
      y = drawTableRow(doc, x, y, headers, widths, true, alignments);
    }
    const c = report.columns;
    const note = [row[c[S.COL_NOTE]], row[c[S.COL_EXTRA_NOTE]]].filter(Boolean).join('; ');
    const fields = [
      row[c[S.COL_DATE]], row[c[S.COL_REGION]], row[c[S.COL_PARCEL]],
      row[c[S.COL_VDP]], row[c[S.COL_TYPE]], fmtDecimal1(row[c[S.COL_QUINTALS]]),
      note,
      ...species.map(s => formatMaybe(row[c[`${s} %`]])),
    ];
    y = drawTableRow(doc, x, y, fields, widths, false, alignments);
  }
  return y;
}

function reportTableWidths(doc, speciesCount) {
  const available = doc.width - 2 * margin;
  const base = [45, 45, 40, 30, 60, 30, 80];
  const baseTotal = base.reduce((a, b) => a + b, 0);
  const speciesWidth = speciesCount
    ? Math.max(30, Math.min(60, Math.floor((available - baseTotal) / speciesCount)))
    : 0;
  return [...base, ...Array.from({ length: speciesCount }, () => speciesWidth)];
}

function reportTableAlignments(speciesCount) {
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
  let l = new Intl.DateTimeFormat(S.LOCALE, { month: 'long', year: 'numeric', timeZone: 'UTC' }).format(d);
  return l.charAt(0).toUpperCase() + l.slice(1);
}
