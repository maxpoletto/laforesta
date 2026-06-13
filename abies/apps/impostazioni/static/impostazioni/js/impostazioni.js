/**
 * Impostazioni (settings) page.
 *
 * Collapsible sections based on user role:
 *   - Password change (all password-login users)
 *   - Crews, Tractors, Species (writers and admins)
 *   - Hypsometric parameters (writers and admins)
 *   - App Users (admins only)
 *
 * Entity tables lazy-load data when the section is first opened.
 * Forms open in modals.
 */

import { fileToBase64, postJSON } from '../../base/js/api.js';
import * as cache from '../../base/js/cache.js';
import { downloadFromURL } from '../../base/js/csv-export.js';
import { TableWrapper } from '../../base/js/table.js';
import * as modals from '../../base/js/modals.js';
import { showError } from '../../base/js/modals.js';
import { fetchModalForm, interceptSubmit } from '../../base/js/forms.js';
import {
  showConfirmModal, showLoadingIn, wireActions, wireCancelButtons,
  wireCollapsibleToggle,
} from '../../base/js/ui-widgets.js';
import { loadCSS, unloadCSS } from '../../base/js/page-css.js';
import { cloneTemplate } from '../../base/js/templates.js';
import * as S from '../../base/js/strings.js';
import {
  FIELD_CREATED_AT, FIELD_FILE, FIELD_HARVEST_PLAN_ID, FIELD_MIN_N,
  FIELD_NONCE, FIELD_SOURCE, FIELD_SURVEY_IDS, FIELD_SURVEYS,
  FIELD_USE_FOR_HEIGHT_PLOTS, HYPSO_SOURCE_COMPUTED, LOGIN_METHOD_PASSWORD,
  DATA_ID, MESSAGE, PATCHES, RECORD, ROLE_ADMIN, ROLE_WRITER,
} from '../../base/js/constants.js';
import {
  fmtDecimal2, fmtDecimal4, fmtInt, parseDecimal,
} from '../../base/js/format.js';

const API = '/api/impostazioni/';
const CSS_URL = '/static/impostazioni/css/impostazioni.css';

// ---------------------------------------------------------------------------
// Section configuration
// ---------------------------------------------------------------------------

const booleanCol = (label) => ({ label, type: 'boolean', width: '50px' });
const ACTIVE_COL_DEF = booleanCol(S.COL_ACTIVE);
const MINOR_COL_DEF = booleanCol(S.COL_MINOR);

// State per entity section: { table, digest, loaded, activeOnly }.
const sections = {};

// Wrap an entity-table config as an ordered-section descriptor.  build()
// registers fresh state under cfg.key (cleared on unmount) before rendering.
function entitySection(cfg) {
  return {
    minRole: cfg.minRole,
    build: () => {
      const state = { table: null, digest: null, loaded: false, activeOnly: true };
      sections[cfg.key] = state;
      return buildEntitySection(cfg, state);
    },
  };
}

// Role-gated sections in display order.  Each: { minRole, build() → Node }.
// Hypsometric parameters sit between Specie and Utenti, so Utenti is last.
const SECTIONS = [
  entitySection({
    key: 'crews',
    minRole: ROLE_WRITER,
    title: S.SETTINGS_CREWS,
    dataUrl: `${API}crews/data/`,
    formUrl: `${API}crews/form/`,
    saveUrl: `${API}crews/save/`,
    csvFilename: S.CSV_CREWS,
    columnDefs: {
      [S.LABEL_NAME]: { label: S.LABEL_NAME, width: '180px' },
      [S.COL_ACTIVE]: ACTIVE_COL_DEF,
    },
  }),
  entitySection({
    key: 'tractors',
    minRole: ROLE_WRITER,
    title: S.SETTINGS_TRACTORS,
    dataUrl: `${API}tractors/data/`,
    formUrl: `${API}tractors/form/`,
    saveUrl: `${API}tractors/save/`,
    csvFilename: S.CSV_TRACTORS,
    columnDefs: { [S.COL_ACTIVE]: ACTIVE_COL_DEF },
  }),
  entitySection({
    key: 'species',
    minRole: ROLE_WRITER,
    title: S.SETTINGS_SPECIES,
    dataUrl: `${API}species/data/`,
    formUrl: `${API}species/form/`,
    saveUrl: `${API}species/save/`,
    csvFilename: S.CSV_SPECIES,
    columnDefs: {
      [S.COL_DENSITY]: { label: S.COL_DENSITY, type: 'number', width: '100px', formatter: fmtDecimal2 },
      [S.COL_MINOR]: MINOR_COL_DEF,
      [S.COL_ACTIVE]: ACTIVE_COL_DEF,
    },
  }),
  { minRole: ROLE_WRITER, build: buildFutureProductionSection },
  { minRole: ROLE_WRITER, build: buildDendrometrySection },
  { minRole: ROLE_WRITER, build: buildHypsoSection },
  entitySection({
    key: 'users',
    minRole: ROLE_ADMIN,
    title: S.SETTINGS_USERS,
    dataUrl: `${API}users/data/`,
    formUrl: `${API}users/form/`,
    saveUrl: `${API}users/save/`,
    csvFilename: S.CSV_USERS,
    columnDefs: {
      [S.LABEL_LAST_NAME]: { label: S.LABEL_LAST_NAME, width: '140px' },
      [S.LABEL_EMAIL]: { label: S.LABEL_EMAIL, width: '240px' },
      [S.LABEL_LOGIN_METHOD]: { label: S.LABEL_LOGIN_METHOD, width: '140px' },
      [S.COL_ACTIVE]: ACTIVE_COL_DEF,
    },
  }),
];

// ---------------------------------------------------------------------------
// Page lifecycle
// ---------------------------------------------------------------------------

export function mount() {
  loadCSS(CSS_URL);
  const el = document.getElementById('content');
  el.replaceChildren();

  const role = document.body.dataset.role;
  const loginMethod = document.body.dataset.loginMethod;

  // Password change is gated on login method, not role; it leads the page.
  if (loginMethod === LOGIN_METHOD_PASSWORD) {
    el.appendChild(buildPasswordSection());
  }

  for (const section of SECTIONS) {
    if (hasMinRole(role, section.minRole)) {
      el.appendChild(section.build());
    }
  }
}

export function unmount() {
  unloadCSS(CSS_URL);
  for (const state of Object.values(sections)) {
    if (state.table) { state.table.destroy(); state.table = null; }
  }
  // Clear state for next mount.
  for (const key of Object.keys(sections)) delete sections[key];

  // Hypso isn't an entity section: it manages a whole parameter set via a
  // module-level singleton, so it's reset directly rather than via `sections`.
  if (hypsoState.table) { hypsoState.table.destroy(); hypsoState.table = null; }
  hypsoState.digest = null;
  hypsoState.loaded = false;
}

export function onQueryChange() {}

// ---------------------------------------------------------------------------
// Password section
// ---------------------------------------------------------------------------

function buildPasswordSection() {
  const frag = cloneTemplate('tmpl-password-section');
  wireCollapsibleToggle(
    frag.querySelector('.collapsible-header'),
    frag.querySelector('.collapsible-body'),
  );

  const form = frag.querySelector('[data-role="password-form"]');
  const pw1 = form.querySelector('input[name="password1"]');
  const pw2 = form.querySelector('input[name="password2"]');
  const msg = frag.querySelector('[data-role="password-msg"]');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    msg.className = 'mt-1';

    if (pw1.value !== pw2.value) {
      msg.textContent = S.PASSWORD_MISMATCH;
      msg.className = 'mt-1 text-error';
      return;
    }

    let data, status;
    try {
      ({ data, status } = await postJSON(`${API}password/`, { password1: pw1.value, password2: pw2.value }));
    } catch {
      showError(S.ERROR_NETWORK);
      return;
    }

    msg.textContent = data.message;
    if (status === 200) {
      form.reset();
    } else {
      msg.className = 'mt-1 text-error';
    }
  });

  return frag;
}

// ---------------------------------------------------------------------------
// Entity sections (crews, tractors, species, users)
// ---------------------------------------------------------------------------

function buildEntitySection(cfg, state) {
  const frag = cloneTemplate('tmpl-entity-section');
  frag.querySelector('[data-field="title"]').textContent = cfg.title;
  const tableContainer = frag.querySelector('[data-target="table-host"]');
  const activeCheck = frag.querySelector('[data-role="active-toggle"]');

  activeCheck.addEventListener('change', () => {
    state.activeOnly = activeCheck.checked;
    applyActiveFilter(state);
  });

  wireCollapsibleToggle(
    frag.querySelector('.collapsible-header'),
    frag.querySelector('.collapsible-body'),
    () => {
      if (state.loaded) return;
      state.loaded = true;
      loadEntityData(cfg, state, tableContainer);
    },
  );

  return frag;
}

async function loadEntityData(cfg, state, container) {
  showLoadingIn(container);

  let data;
  try {
    const resp = await fetch(cfg.dataUrl);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    container.replaceChildren();
    showError(S.ERROR_NETWORK);
    return;
  }

  container.replaceChildren();
  state.digest = data;

  state.table = new TableWrapper({
    container,
    digest: data,
    columnDefs: cfg.columnDefs || {},
    canModify: true,
    actions: {
      onEdit: (rowId) => openForm(cfg, state, rowId),
      onAdd: () => openForm(cfg, state, null),
    },
    csvFilename: cfg.csvFilename,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
  });

  applyActiveFilter(state);
}

function applyActiveFilter(state) {
  if (!state.table || !state.digest) return;
  const cols = state.digest.columns;
  const activeIdx = cols.indexOf(S.COL_ACTIVE);
  if (activeIdx < 0 || !state.activeOnly) {
    state.table.setExternalFilter(null);
  } else {
    state.table.setExternalFilter(row => row[activeIdx] === true);
  }
}

// ---------------------------------------------------------------------------
// Modal forms
// ---------------------------------------------------------------------------

async function openForm(cfg, state, rowId) {
  const url = rowId ? `${cfg.formUrl}${rowId}/` : cfg.formUrl;
  const form = await fetchModalForm(url);
  if (!form) return;
  wireSettingsForm(form, cfg, state);
}

function wireSettingsForm(form, cfg, state) {
  wirePasswordToggle(form);
  wireCancelButtons(form, () => modals.dismiss());

  interceptSubmit(form, cfg.saveUrl, {
    validate(body) {
      // Species density must be > 0 (only the species form has it).
      if (body.density !== undefined && !(parseDecimal(body.density) > 0)) {
        return S.ERR_DENSITY_POSITIVE;
      }
      return null;
    },
    onSuccess(data) {
      modals.dismiss();
      // Settings tables are local state, not registered in cache.js.
      const patch = data[PATCHES]?.find(p => p[DATA_ID] === cfg.key);
      if (state.digest && patch?.[RECORD]) {
        const rows = state.digest.rows;
        const idx = rows.findIndex(r => r[0] === patch[RECORD][0]);
        if (idx >= 0) {
          rows[idx] = patch[RECORD];
        } else {
          rows.push(patch[RECORD]);
        }
        state.table?.setData(state.digest);
      }
    },
  });
}

/**
 * For the user form: toggle password-login-only fields (username,
 * password, repeat-password) based on login_method radio.  OAuth users
 * are matched by email, so we auto-use email as username server-side.
 */
function wirePasswordToggle(form) {
  const radios = form.querySelectorAll('input[name="login_method"]');
  if (!radios.length) return;

  const pwFields = form.querySelectorAll('.password-login-only');
  function toggle() {
    const method = form.querySelector('input[name="login_method"]:checked')?.value;
    const show = method === LOGIN_METHOD_PASSWORD;
    for (const el of pwFields) el.style.display = show ? '' : 'none';
  }
  for (const r of radios) r.addEventListener('change', toggle);
  toggle();
}

// ---------------------------------------------------------------------------
// Bosco source settings (writer+)
// ---------------------------------------------------------------------------

const FUTURE_PRODUCTION = {
  data: `${API}future-production/data/`,
  save: `${API}future-production/save/`,
};

const DENDROMETRY = {
  data: `${API}dendrometry/data/`,
  save: `${API}dendrometry/save/`,
};

function buildFutureProductionSection() {
  const frag = cloneTemplate('tmpl-future-production-section');
  const body = frag.querySelector('.collapsible-body');
  const form = body.querySelector('[data-role="future-production-form"]');
  const select = body.querySelector('[data-role="future-plan"]');
  const msg = body.querySelector('[data-role="future-production-msg"]');
  let loaded = false;

  wireCollapsibleToggle(
    frag.querySelector('.collapsible-header'), body,
    () => {
      if (loaded) return;
      loaded = true;
      loadFutureProduction(select, form);
    },
  );

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    const planId = parseInt(select.value, 10);
    const data = await postOrError(postJSON(FUTURE_PRODUCTION.save, {
      [FIELD_HARVEST_PLAN_ID]: planId,
      [FIELD_NONCE]: crypto.randomUUID(),
    }));
    if (data) msg.textContent = data[MESSAGE] || form.dataset.successLabel;
  });

  return frag;
}

async function loadFutureProduction(select, form) {
  setFormEnabled(form, false);
  setSelectPlaceholder(select, 'loadingLabel');

  let data;
  try {
    const resp = await fetch(FUTURE_PRODUCTION.data);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    select.replaceChildren();
    showError(S.ERROR_NETWORK);
    return;
  }

  const plans = data.plans || [];
  select.replaceChildren();
  if (!plans.length) {
    setSelectPlaceholder(select, 'emptyLabel');
    return;
  }
  for (const plan of plans) {
    const opt = selectOption(plan.id, `${plan.name} (${plan.year_start}-${plan.year_end})`);
    opt.selected = plan.id === data.active_id || plan.active === true;
    select.appendChild(opt);
  }
  setFormEnabled(form, true);
}

function buildDendrometrySection() {
  const frag = cloneTemplate('tmpl-dendrometry-section');
  const body = frag.querySelector('.collapsible-body');
  const form = body.querySelector('[data-role="dendrometry-form"]');
  const msg = body.querySelector('[data-role="dendrometry-msg"]');
  let loaded = false;

  wireCollapsibleToggle(
    frag.querySelector('.collapsible-header'), body,
    () => {
      if (loaded) return;
      loaded = true;
      loadDendrometry(body);
    },
  );

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    const data = await postOrError(postJSON(DENDROMETRY.save, {
      [FIELD_SURVEY_IDS]: selectedDendrometrySurveyIds(body),
      [FIELD_NONCE]: crypto.randomUUID(),
    }));
    if (!data) return;
    await loadDendrometry(body);
    msg.textContent = data[MESSAGE] || form.dataset.successLabel;
  });

  return frag;
}

async function loadDendrometry(body) {
  const form = body.querySelector('[data-role="dendrometry-form"]');
  const select = body.querySelector('[data-role="dendrometry-surveys"]');
  const summary = body.querySelector('[data-role="dendrometry-counts"]');

  setFormEnabled(form, false);
  renderDendrometryCounts(summary, {});
  setSelectPlaceholder(select, 'loadingLabel');

  let data;
  try {
    const resp = await fetch(DENDROMETRY.data);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    select.replaceChildren();
    showError(S.ERROR_NETWORK);
    return;
  }

  const surveys = data.surveys || [];
  const activeIds = new Set(data.active_ids || []);
  renderDendrometryCounts(summary, data.counts || {});
  select.replaceChildren();
  if (!surveys.length) {
    setSelectPlaceholder(select, 'emptyLabel');
    return;
  }
  for (const survey of surveys) {
    const opt = selectOption(survey.id, survey.name);
    opt.selected = activeIds.has(survey.id) || survey.active === true;
    select.appendChild(opt);
  }
  setFormEnabled(form, true);
}

function selectedDendrometrySurveyIds(body) {
  return [...body.querySelectorAll('[data-role="dendrometry-surveys"] option:checked')]
    .map(o => parseInt(o.value, 10));
}

function renderDendrometryCounts(root, counts) {
  for (const field of ['trees', 'regions', 'parcels']) {
    root.querySelector(`[data-field="${field}"]`).textContent = counts[field] || 0;
  }
}

function setSelectPlaceholder(select, labelName) {
  select.replaceChildren(selectOption('', select.dataset[labelName] || ''));
}

function selectOption(value, text) {
  const frag = cloneTemplate('tmpl-settings-select-option');
  const opt = frag.querySelector('option');
  opt.value = value;
  opt.textContent = text;
  return opt;
}

function setFormEnabled(form, enabled) {
  for (const el of form.querySelectorAll('select, button')) {
    el.disabled = !enabled;
  }
}

// ---------------------------------------------------------------------------
// Hypsometric parameters (writer+).  Whole-set management: a read-only table
// of the active set plus compute / import / export / clear.  See
// docs/hypsometry.md.
// ---------------------------------------------------------------------------

const HYPSO = {
  data:    `${API}hypso-params/data/`,
  active:  `${API}hypso-params/active-set/`,
  compute: `${API}hypso-params/compute/`,
  accept:  `${API}hypso-params/accept/`,
  upload:  `${API}hypso-params/import/`,
  export:  `${API}hypso-params/export/`,
  clear:   `${API}hypso-params/clear/`,
  surveys: '/api/campionamenti/surveys/data/',
};

const hypsoState = { table: null, digest: null, loaded: false };

// Hypso table numeric columns: a/b/r² are 4-dp decimals, n a count.
const HYPSO_COL_DEFS = {
  [S.COL_A]: { type: 'number', formatter: fmtDecimal4 },
  [S.COL_B]: { type: 'number', formatter: fmtDecimal4 },
  [S.COL_R2]: { type: 'number', formatter: fmtDecimal4 },
  [S.COL_N_REGRESSION]: { type: 'number', formatter: fmtInt },
};

function buildHypsoSection() {
  const frag = cloneTemplate('tmpl-hypso-section');
  const body = frag.querySelector('.collapsible-body');
  const tableHost = body.querySelector('[data-target="table-host"]');
  const descEl = body.querySelector('[data-target="description"]');
  const fileInput = body.querySelector('[data-role="import-file"]');

  wireCollapsibleToggle(
    frag.querySelector('.collapsible-header'), body,
    () => {
      if (hypsoState.loaded) return;
      hypsoState.loaded = true;
      loadHypso(tableHost, descEl);
      loadSurveys(body);
    },
  );

  wireActions(body, {
    export: () => downloadFromURL(HYPSO.export),
    import: () => fileInput.click(),
    clear: () => confirmClear(tableHost, descEl),
    compute: () => runCompute(body, tableHost, descEl),
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    fileInput.value = '';
    if (file) confirmImport(file, tableHost, descEl);
  });

  return frag;
}

async function loadHypso(tableHost, descEl) {
  showLoadingIn(tableHost);
  let digest, meta;
  try {
    const [d, m] = await Promise.all([fetch(HYPSO.data), fetch(HYPSO.active)]);
    if (!d.ok || !m.ok) throw new Error();
    digest = await d.json();
    meta = await m.json();
  } catch {
    tableHost.replaceChildren();
    showError(S.ERROR_NETWORK);
    return;
  }
  tableHost.replaceChildren();
  hypsoState.digest = digest;
  if (hypsoState.table) hypsoState.table.destroy();
  hypsoState.table = new TableWrapper({
    container: tableHost, digest, canModify: false, inlineToolbar: false,
    labels: S.TABLE_LABELS, csvFormat: S.TABLE_CSV_FORMAT,
    columnDefs: HYPSO_COL_DEFS,
  });
  renderDescription(descEl, meta);
}

function renderDescription(el, meta) {
  if (!meta || !meta[FIELD_SOURCE]) {
    el.textContent = S.HYPSO_DESC_NONE;
    return;
  }
  const label = meta[FIELD_SOURCE] === HYPSO_SOURCE_COMPUTED
    ? S.HYPSO_SOURCE_COMPUTED_LABEL : S.HYPSO_SOURCE_IMPORTED_LABEL;
  const parts = [`${label} (${meta[FIELD_CREATED_AT]})`];
  if (meta[FIELD_MIN_N] != null) {
    parts.push(`${S.HYPSO_DESC_MIN_N}: ${meta[FIELD_MIN_N]}`);
  }
  const surveys = meta[FIELD_SURVEYS] || [];
  if (surveys.length) {
    parts.push(`${S.HYPSO_DESC_SURVEYS}: ${surveys.join(', ')}`);
  }
  if (meta[FIELD_USE_FOR_HEIGHT_PLOTS]) {
    parts.push(S.HYPSO_DESC_HEIGHT_PLOTS);
  }
  el.textContent = parts.join(' · ');
}

async function loadSurveys(body) {
  const select = body.querySelector('[data-role="surveys"]');
  let digest;
  try {
    const resp = await fetch(HYPSO.surveys);
    if (!resp.ok) throw new Error();
    digest = await resp.json();
  } catch {
    return;
  }
  const nameIdx = digest.columns.indexOf(S.COL_NAME);
  select.replaceChildren();
  for (const row of digest.rows) {
    const opt = document.createElement('option');
    opt.value = row[0];
    opt.textContent = row[nameIdx];
    select.appendChild(opt);
  }
}

function selectedSurveyIds(body) {
  return [...body.querySelectorAll('[data-role="surveys"] option:checked')]
    .map(o => parseInt(o.value, 10));
}

async function runCompute(body, tableHost, descEl) {
  const minN = parseInt(body.querySelector('[data-role="min-n"]').value, 10);
  const surveyIds = selectedSurveyIds(body);
  const useForHeightPlots = body
    .querySelector('[data-role="use-for-height-plots"]')?.checked === true;
  const data = await postOrError(postJSON(HYPSO.compute,
    { [FIELD_MIN_N]: minN, [FIELD_SURVEY_IDS]: surveyIds }));
  if (data) {
    showCandidate(data, minN, surveyIds, useForHeightPlots, tableHost, descEl);
  }
}

function showCandidate(payload, minN, surveyIds, useForHeightPlots, tableHost, descEl) {
  const frag = cloneTemplate('tmpl-hypso-candidate');
  const root = frag.querySelector('.hypso-candidate');
  const host = root.querySelector('[data-target="candidate-table"]');

  wireActions(root, {
    reject: () => modals.dismiss(),
    accept: async () => {
      const data = await postOrError(postJSON(HYPSO.accept, {
        [FIELD_MIN_N]: minN,
        [FIELD_SURVEY_IDS]: surveyIds,
        [FIELD_USE_FOR_HEIGHT_PLOTS]: useForHeightPlots,
        [FIELD_NONCE]: crypto.randomUUID(),
      }));
      if (!data) return;
      modals.dismiss();
      loadHypso(tableHost, descEl);
    },
  });
  modals.show(root);

  // Same read-only digest table as the active set, in the modal.
  const table = new TableWrapper({
    container: host, digest: payload, canModify: false, inlineToolbar: false,
    labels: S.TABLE_LABELS, csvFormat: S.TABLE_CSV_FORMAT,
    columnDefs: HYPSO_COL_DEFS,
  });
  modals.onDismiss(() => table.destroy());
}

function confirmImport(file, tableHost, descEl) {
  showConfirmModal(S.HYPSO_IMPORT_CONFIRM, async () => {
    const body = {
      [FIELD_FILE]: await fileToBase64(file),
      [FIELD_NONCE]: crypto.randomUUID(),
    };
    if (await postOrError(postJSON(HYPSO.upload, body))) {
      loadHypso(tableHost, descEl);
    }
  }, { confirmLabel: S.IMPORT_LABEL });
}

function confirmClear(tableHost, descEl) {
  showConfirmModal(S.HYPSO_CLEAR_CONFIRM, async () => {
    if (await postOrError(postJSON(HYPSO.clear, {
      [FIELD_NONCE]: crypto.randomUUID(),
    }))) {
      loadHypso(tableHost, descEl);
    }
  }, { confirmLabel: S.ACTION_DELETE });
}

/**
 * Await a postJSON promise, surfacing network and non-200
 * errors in a modal.  Returns the response data, or null on error.
 */
async function postOrError(promise) {
  let result;
  try {
    result = await promise;
  } catch {
    showError(S.ERROR_NETWORK);
    return null;
  }
  if (result.status !== 200) {
    showError(result.data[MESSAGE] || S.ERROR_GENERIC);
    return null;
  }
  cache.applyResponseChanges(result.data);
  return result.data;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hasMinRole(role, minRole) {
  if (minRole === ROLE_ADMIN) return role === ROLE_ADMIN;
  if (minRole === ROLE_WRITER) return role === ROLE_ADMIN || role === ROLE_WRITER;
  return true;
}
