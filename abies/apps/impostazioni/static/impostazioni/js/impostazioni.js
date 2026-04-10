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
import * as S from '../../base/js/strings.js';

const API = '/abies/api/impostazioni/';

// ---------------------------------------------------------------------------
// Section configuration
// ---------------------------------------------------------------------------

const ENTITY_SECTIONS = [
  {
    key: 'crews',
    title: S.SETTINGS_CREWS,
    minRole: 'writer',
    dataUrl: `${API}crews/data/`,
    formUrl: `${API}crews/form/`,
    saveUrl: `${API}crews/save/`,
  },
  {
    key: 'tractors',
    title: S.SETTINGS_TRACTORS,
    minRole: 'writer',
    dataUrl: `${API}tractors/data/`,
    formUrl: `${API}tractors/form/`,
    saveUrl: `${API}tractors/save/`,
  },
  {
    key: 'species',
    title: S.SETTINGS_SPECIES,
    minRole: 'writer',
    dataUrl: `${API}species/data/`,
    formUrl: `${API}species/form/`,
    saveUrl: `${API}species/save/`,
  },
  {
    key: 'users',
    title: S.SETTINGS_USERS,
    minRole: 'admin',
    dataUrl: `${API}users/data/`,
    formUrl: `${API}users/form/`,
    saveUrl: `${API}users/save/`,
  },
];

const ACTIVE_COL_NAME = S.COL_ACTIVE;

// State per entity section: { table, digest, loaded }
const sections = {};

// ---------------------------------------------------------------------------
// Page lifecycle
// ---------------------------------------------------------------------------

export function mount() {
  const el = document.getElementById('content');
  el.replaceChildren();

  const role = document.body.dataset.role;
  const loginMethod = document.body.dataset.loginMethod;

  // Password section — visible to all password-login users.
  if (loginMethod === 'password') {
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
  const frag = document.createDocumentFragment();
  const { header, body } = buildCollapsible(S.SETTINGS_PASSWORD);
  frag.appendChild(header);

  const form = document.createElement('form');

  // Password1
  const g1 = formGroup(S.PASSWORD_NEW);
  const pw1 = document.createElement('input');
  pw1.type = 'password';
  pw1.id = 'id_pw1';
  pw1.name = 'password1';
  pw1.required = true;
  g1.appendChild(pw1);
  form.appendChild(g1);

  // Password2
  const g2 = formGroup(S.PASSWORD_REPEAT);
  const pw2 = document.createElement('input');
  pw2.type = 'password';
  pw2.id = 'id_pw2';
  pw2.name = 'password2';
  pw2.required = true;
  g2.appendChild(pw2);
  form.appendChild(g2);

  // Submit
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const btn = document.createElement('button');
  btn.type = 'submit';
  btn.className = 'btn btn-primary';
  btn.textContent = S.SAVE;
  actions.appendChild(btn);
  form.appendChild(actions);

  // Feedback message
  const msg = document.createElement('p');
  msg.className = 'mt-1';

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    msg.textContent = '';
    msg.className = 'mt-1';

    if (pw1.value !== pw2.value) {
      msg.textContent = S.PASSWORD_MISMATCH;
      msg.classList.add('text-error');
      return;
    }

    let data, status;
    try {
      ({ data, status } = await postJSON(`${API}password/`, { password1: pw1.value, password2: pw2.value }));
    } catch {
      showError(S.ERROR_NETWORK);
      return;
    }

    if (status === 200) {
      msg.textContent = data.message;
      form.reset();
    } else {
      msg.textContent = data.message;
      msg.classList.add('text-error');
    }
  });

  body.appendChild(form);
  body.appendChild(msg);
  frag.appendChild(body);
  return frag;
}

// ---------------------------------------------------------------------------
// Entity sections (crews, tractors, species, users)
// ---------------------------------------------------------------------------

function buildEntitySection(cfg, state) {
  const frag = document.createDocumentFragment();
  const { header, body } = buildCollapsible(cfg.title);
  frag.appendChild(header);

  // "Only active" checkbox.
  const activeRow = document.createElement('div');
  activeRow.className = 'table-toolbar';
  const activeLabel = document.createElement('label');
  activeLabel.style.cssText = 'display:flex;align-items:center;gap:4px;font-size:0.85rem;margin-left:auto';
  const activeCheck = document.createElement('input');
  activeCheck.type = 'checkbox';
  activeCheck.checked = true;
  activeCheck.addEventListener('change', () => {
    state.activeOnly = activeCheck.checked;
    applyActiveFilter(state);
  });
  activeLabel.appendChild(activeCheck);
  activeLabel.append(S.ONLY_ACTIVE);
  activeRow.appendChild(activeLabel);

  const tableContainer = document.createElement('div');

  body.appendChild(activeRow);
  body.appendChild(tableContainer);
  frag.appendChild(body);

  // Lazy load: fetch data on first open.
  header.addEventListener('click', () => {
    if (!state.loaded) {
      state.loaded = true;
      loadEntityData(cfg, state, tableContainer);
    }
  });

  return frag;
}

async function loadEntityData(cfg, state, container) {
  container.textContent = S.LOADING;

  let data;
  try {
    const resp = await fetch(cfg.dataUrl);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    container.textContent = '';
    showError(S.ERROR_NETWORK);
    return;
  }

  container.textContent = '';
  state.digest = data;

  state.table = new TableWrapper({
    container,
    digest: data,
    columnDefs: {},  // use server-provided column names as-is
    canModify: true,
    actions: {
      onEdit: (rowId) => openForm(cfg, state, rowId),
      onAdd: () => openForm(cfg, state, null),
    },
    csvFilename: `${cfg.key}.csv`,
  });

  applyActiveFilter(state);
}

function applyActiveFilter(state) {
  if (!state.table || !state.digest) return;
  const cols = state.digest.columns;
  const activeIdx = cols.indexOf(ACTIVE_COL_NAME);
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
          applyActiveFilter(state);
        }
      } else if (data.status === 'conflict') {
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
 * For the user form: toggle password fields based on login_method radio.
 */
function wirePasswordToggle(form) {
  const radios = form.querySelectorAll('input[name="login_method"]');
  if (!radios.length) return;

  const pwFields = form.querySelectorAll('.password-fields');
  function toggle() {
    const method = form.querySelector('input[name="login_method"]:checked')?.value;
    const show = method === 'password';
    for (const el of pwFields) el.style.display = show ? '' : 'none';
  }
  for (const r of radios) r.addEventListener('change', toggle);
  toggle();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildCollapsible(title) {
  const header = document.createElement('div');
  header.className = 'collapsible-header';
  const titleSpan = document.createElement('span');
  titleSpan.textContent = title;
  const arrow = document.createElement('span');
  arrow.className = 'arrow';
  header.appendChild(titleSpan);
  header.appendChild(arrow);

  const body = document.createElement('div');
  body.className = 'collapsible-body';

  header.addEventListener('click', () => {
    header.classList.toggle('open');
    body.classList.toggle('open');
  });

  return { header, body };
}

function hasMinRole(role, minRole) {
  if (minRole === 'admin') return role === 'admin';
  if (minRole === 'writer') return role === 'admin' || role === 'writer';
  return true;
}

/** Create a form-group div with a label. */
function formGroup(labelText) {
  const g = document.createElement('div');
  g.className = 'form-group';
  const lbl = document.createElement('label');
  lbl.textContent = labelText;
  g.appendChild(lbl);
  return g;
}
