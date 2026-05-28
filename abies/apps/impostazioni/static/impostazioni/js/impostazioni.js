/**
 * Impostazioni (settings) page.
 *
 * Collapsible sections based on user role:
 *   - Password change (all password-login users)
 *   - Crews, Tractors, Species (writers and admins)
 *   - App Users (admins only)
 *
 * Entity tables lazy-load data when the section is first opened.
 * Forms open in modals.
 */

import { postJSON } from '../../base/js/api.js';
import { TableWrapper } from '../../base/js/table.js';
import * as modals from '../../base/js/modals.js';
import { showError } from '../../base/js/modals.js';
import { wireCancelButtons } from '../../base/js/forms.js';
import { showLoadingIn, wireCollapsibleToggle } from '../../base/js/ui-widgets.js';
import { loadCSS, unloadCSS } from '../../base/js/page-css.js';
import { cloneTemplate } from '../../base/js/templates.js';
import * as S from '../../base/js/strings.js';
import {
  LOGIN_METHOD_PASSWORD, ROLE_ADMIN, ROLE_WRITER, STATUS_CONFLICT,
} from '../../base/js/constants.js';

const API = '/api/impostazioni/';
const CSS_URL = '/static/impostazioni/css/impostazioni.css';

// ---------------------------------------------------------------------------
// Section configuration
// ---------------------------------------------------------------------------

const ACTIVE_COL_DEF = { label: S.COL_ACTIVE, type: 'boolean', width: '50px' };

const ENTITY_SECTIONS = [
  {
    key: 'crews',
    title: S.SETTINGS_CREWS,
    minRole: ROLE_WRITER,
    dataUrl: `${API}crews/data/`,
    formUrl: `${API}crews/form/`,
    saveUrl: `${API}crews/save/`,
    csvFilename: S.CSV_CREWS,
    columnDefs: {
      [S.LABEL_NAME]: { label: S.LABEL_NAME, width: '180px' },
      [S.COL_ACTIVE]: ACTIVE_COL_DEF,
    },
  },
  {
    key: 'tractors',
    title: S.SETTINGS_TRACTORS,
    minRole: ROLE_WRITER,
    dataUrl: `${API}tractors/data/`,
    formUrl: `${API}tractors/form/`,
    saveUrl: `${API}tractors/save/`,
    csvFilename: S.CSV_TRACTORS,
    columnDefs: { [S.COL_ACTIVE]: ACTIVE_COL_DEF },
  },
  {
    key: 'species',
    title: S.SETTINGS_SPECIES,
    minRole: ROLE_WRITER,
    dataUrl: `${API}species/data/`,
    formUrl: `${API}species/form/`,
    saveUrl: `${API}species/save/`,
    csvFilename: S.CSV_SPECIES,
    columnDefs: { [S.COL_ACTIVE]: ACTIVE_COL_DEF },
  },
  {
    key: 'users',
    title: S.SETTINGS_USERS,
    minRole: ROLE_ADMIN,
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
  },
];


// State per entity section: { table, digest, loaded }
const sections = {};

// ---------------------------------------------------------------------------
// Page lifecycle
// ---------------------------------------------------------------------------

export function mount() {
  loadCSS(CSS_URL);
  const el = document.getElementById('content');
  el.replaceChildren();

  const role = document.body.dataset.role;
  const loginMethod = document.body.dataset.loginMethod;

  // Password section — visible to all password-login users.
  if (loginMethod === LOGIN_METHOD_PASSWORD) {
    el.appendChild(buildPasswordSection());
  }

  // Entity sections — visible based on role.
  for (const cfg of ENTITY_SECTIONS) {
    if (!hasMinRole(role, cfg.minRole)) continue;
    const state = { table: null, digest: null, loaded: false, activeOnly: true };
    sections[cfg.key] = state;
    el.appendChild(buildEntitySection(cfg, state));
  }
}

export function unmount() {
  unloadCSS(CSS_URL);
  for (const state of Object.values(sections)) {
    if (state.table) { state.table.destroy(); state.table = null; }
  }
  // Clear state for next mount.
  for (const key of Object.keys(sections)) delete sections[key];
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

  let data;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  showFormModal(data.html, cfg, state);
}

function showFormModal(html, cfg, state) {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const wrapper = document.createElement('div');
  wrapper.append(...doc.body.childNodes);

  const form = wrapper.querySelector('form');
  if (form) {
    // Inject nonce.
    let nonce = form.querySelector('input[name="nonce"]');
    if (!nonce) {
      nonce = document.createElement('input');
      nonce.type = 'hidden';
      nonce.name = 'nonce';
      form.appendChild(nonce);
    }
    nonce.value = crypto.randomUUID();

    // Wire password field visibility for user form.
    wirePasswordToggle(form);

    wireCancelButtons(form, () => modals.dismiss());

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const body = Object.fromEntries(new FormData(form));

      // Clear any previous inline error.
      const prev = form.querySelector('.form-inline-error');
      if (prev) prev.remove();

      let data, status;
      try {
        ({ data, status } = await postJSON(cfg.saveUrl, body));
      } catch {
        showError(S.ERROR_NETWORK);
        return;
      }

      if (status === 200) {
        modals.dismiss();
        // Update table in place.
        if (state.digest && data.record) {
          const rows = state.digest.rows;
          const idx = rows.findIndex(r => r[0] === data.row_id);
          if (idx >= 0) {
            rows[idx] = data.record;
          } else {
            rows.push(data.record);
          }
          state.table?.setData(state.digest);
        }
      } else if (data.status === STATUS_CONFLICT) {
        showError(data.message || S.ERROR_CONFLICT);
      } else {
        // Inline validation error — stays inside the modal form.
        const err = document.createElement('p');
        err.className = 'form-inline-error text-error mt-1';
        err.textContent = data.message || S.ERROR_GENERIC;
        form.querySelector('.form-actions').before(err);
      }
    });
  }

  modals.show(wrapper);
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
// Helpers
// ---------------------------------------------------------------------------

function hasMinRole(role, minRole) {
  if (minRole === ROLE_ADMIN) return role === ROLE_ADMIN;
  if (minRole === ROLE_WRITER) return role === ROLE_ADMIN || role === ROLE_WRITER;
  return true;
}
