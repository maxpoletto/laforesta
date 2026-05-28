/**
 * Shared DOM builders for form widgets — used by all domain pages.
 *
 * These are the canonical implementations; never duplicate them locally.
 */

import * as S from './strings.js';
import { show as showModal, dismiss as dismissModal } from './modals.js';
import { cloneTemplate } from './templates.js';

// ---------------------------------------------------------------------------
// Form structure
// ---------------------------------------------------------------------------

export function mkRow(host, modifier) {
  const row = document.createElement('div');
  row.className = 'form-row' + (modifier ? ' ' + modifier : '');
  host.appendChild(row);
  return row;
}

export function mkInput(host, opts) {
  const group = document.createElement('div');
  group.className = 'form-group';
  host.appendChild(group);
  const lbl = document.createElement('label');
  lbl.textContent = opts.label;
  lbl.htmlFor = opts.id;
  group.appendChild(lbl);
  const inp = document.createElement('input');
  inp.type = opts.type || 'text';
  inp.name = opts.name;
  inp.id = opts.id;
  if (opts.required) inp.required = true;
  if (opts.maxLength != null) inp.maxLength = opts.maxLength;
  if (opts.min != null) inp.min = opts.min;
  if (opts.max != null) inp.max = opts.max;
  if (opts.value != null) inp.value = opts.value;
  group.appendChild(inp);
  return inp;
}

export function mkTextarea(host, opts) {
  const group = document.createElement('div');
  group.className = 'form-group';
  host.appendChild(group);
  const lbl = document.createElement('label');
  lbl.textContent = opts.label;
  lbl.htmlFor = opts.id;
  group.appendChild(lbl);
  const ta = document.createElement('textarea');
  ta.name = opts.name;
  ta.id = opts.id;
  ta.rows = opts.rows || 3;
  if (opts.value != null) ta.value = opts.value;
  group.appendChild(ta);
  return ta;
}

export function mkFileInput(host, opts) {
  const row = mkRow(host);
  const group = document.createElement('div');
  group.className = 'form-group';
  row.appendChild(group);
  const lbl = document.createElement('label');
  lbl.textContent = opts.label;
  group.appendChild(lbl);
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = '.csv,text/csv';
  inp.required = true;
  lbl.htmlFor = inp.id = opts.id;
  group.appendChild(inp);
  return inp;
}

export function mkFormActions(host, { onCancel, submitLabel, secondaryLabel }) {
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'btn';
  cancel.dataset.action = 'cancel';
  cancel.textContent = S.CANCEL;
  cancel.addEventListener('click', onCancel);
  const submit = document.createElement('button');
  submit.type = 'submit';
  submit.className = 'btn btn-primary';
  submit.textContent = submitLabel;
  actions.append(cancel, submit);
  if (secondaryLabel) {
    const secondary = document.createElement('button');
    secondary.type = 'submit';
    secondary.className = 'btn btn-primary';
    secondary.dataset.action = 'save-and-add';
    secondary.textContent = secondaryLabel;
    actions.appendChild(secondary);
  }
  host.appendChild(actions);
  return actions;
}

// ---------------------------------------------------------------------------
// CSV import feedback
// ---------------------------------------------------------------------------

export function mkStatusBox(host) {
  const box = document.createElement('div');
  box.className = 'csv-import-status';
  box.hidden = true;
  host.appendChild(box);
  return box;
}

export function mkErrorsBox(host) {
  const box = document.createElement('div');
  box.className = 'csv-import-errors';
  box.hidden = true;
  host.appendChild(box);
  return box;
}

export function renderCsvErrors(box, errors) {
  box.replaceChildren();
  const ul = document.createElement('ul');
  for (const e of errors.slice(0, 50)) {
    const li = document.createElement('li');
    li.textContent = e;
    ul.appendChild(li);
  }
  if (errors.length > 50) {
    const more = document.createElement('li');
    more.textContent = `… +${errors.length - 50} altri errori`;
    ul.appendChild(more);
  }
  box.appendChild(ul);
  box.hidden = false;
}

// ---------------------------------------------------------------------------
// Collapsible sections
// ---------------------------------------------------------------------------

export function mkCollapsible(title, open) {
  const header = document.createElement('div');
  header.className = 'collapsible-header' + (open ? ' open' : '');
  const span = document.createElement('span');
  span.textContent = title;
  const arrow = document.createElement('span');
  arrow.className = 'arrow';
  header.append(span, arrow);

  const body = document.createElement('div');
  body.className = 'collapsible-body' + (open ? ' open' : '');
  return [header, body];
}

// ---------------------------------------------------------------------------
// Edit/delete icon pair (pencil + garbage)
// ---------------------------------------------------------------------------

export function mkEditDeleteIcons(host, { onEdit, onDelete, iconClass }) {
  const cls = iconClass || '';
  const edit = document.createElement('span');
  edit.className = 'action-icon action-edit' + (cls ? ' ' + cls : '');
  edit.title = S.ACTION_EDIT;
  edit.textContent = '✎';
  edit.setAttribute('role', 'button');
  edit.addEventListener('click', onEdit);
  host.appendChild(edit);

  const del = document.createElement('span');
  del.className = 'action-icon action-delete' + (cls ? ' ' + cls : '');
  del.title = S.ACTION_DELETE;
  del.textContent = '\u{1F5D1}\u{FE0E}';
  del.setAttribute('role', 'button');
  del.addEventListener('click', onDelete);
  host.appendChild(del);
}

// ---------------------------------------------------------------------------
// Tabbed modal
// ---------------------------------------------------------------------------

/**
 * Build a standard tabbed modal (title + tab bar + tab bodies).
 *
 * @param {{
 *   title: string,
 *   tabs: Array<{ id: string, label: string, build: (host: HTMLElement) => void }>,
 *   initialTab?: string,
 *   onSwitch?: (id: string) => void,
 * }} opts
 * @returns {{ fragment: DocumentFragment, switchTab: (id: string) => void }}
 */
export function mkTabbedModal({ title, tabs, initialTab, onSwitch }) {
  const frag = document.createDocumentFragment();
  const card = document.createElement('div');
  card.className = 'form-card';
  frag.appendChild(card);

  const h = document.createElement('h2');
  h.textContent = title;
  card.appendChild(h);

  const tabBar = document.createElement('div');
  tabBar.className = 'modal-tabs';
  card.appendChild(tabBar);

  const bodies = document.createElement('div');
  bodies.className = 'modal-tab-bodies';
  card.appendChild(bodies);

  const bodyEls = {};
  for (const t of tabs) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'modal-tab';
    btn.dataset.path = t.id;
    btn.textContent = t.label;
    btn.addEventListener('click', () => switchTab(t.id));
    tabBar.appendChild(btn);

    const body = document.createElement('div');
    body.className = 'modal-tab-body';
    body.dataset.path = t.id;
    t.build(body);
    bodies.appendChild(body);
    bodyEls[t.id] = body;
  }

  function switchTab(id) {
    for (const btn of tabBar.querySelectorAll('.modal-tab')) {
      btn.classList.toggle('active', btn.dataset.path === id);
    }
    for (const [k, b] of Object.entries(bodyEls)) {
      b.classList.toggle('active', k === id);
    }
    onSwitch?.(id);
  }

  /** Call after the fragment is in the DOM to lock min-height to the tallest tab. */
  function lockHeight() {
    const allBodies = Object.values(bodyEls);
    for (const b of allBodies) b.style.display = 'block';
    let max = 0;
    for (const b of allBodies) max = Math.max(max, b.offsetHeight);
    for (const b of allBodies) b.style.display = '';
    if (max > 0) bodies.style.minHeight = `${max}px`;
  }

  switchTab(initialTab || tabs[0]?.id);
  return { fragment: frag, switchTab, lockHeight };
}

// ---------------------------------------------------------------------------
// data-action click delegation
// ---------------------------------------------------------------------------

/**
 * Wire a single delegated click handler that dispatches by `data-action`.
 *
 * Returns a disposer.  Callers attaching to a long-lived element (e.g.
 * `#content`, the persistent SPA shell container) MUST call the disposer
 * before rebuilding, otherwise listeners accumulate on every rebuild.
 * For ephemeral elements that get removed by `replaceChildren()` the
 * disposer can be ignored — the listener dies with the node.
 */
export function wireActions(root, handlers) {
  const handler = (e) => {
    const btn = e.target.closest('[data-action]');
    if (btn) handlers[btn.dataset.action]?.(btn, e);
  };
  root.addEventListener('click', handler);
  return () => root.removeEventListener('click', handler);
}

// ---------------------------------------------------------------------------
// Confirm / cascade-delete modals (backed by <template> in shell)
// ---------------------------------------------------------------------------

export function showConfirmModal(message, onConfirm, { confirmLabel } = {}) {
  const frag = cloneTemplate('tmpl-confirm-modal');
  frag.querySelector('[data-field="message"]').textContent = message;
  const okBtn = frag.querySelector('[data-action="confirm"]');
  okBtn.textContent = confirmLabel || S.ACTION_DELETE;
  frag.querySelector('[data-action="cancel"]')
    .addEventListener('click', () => dismissModal());
  okBtn.addEventListener('click', async () => {
    dismissModal();
    await onConfirm();
  });
  showModal(frag);
}

export function showCascadeDeleteModal({ title, warning, exportRequired, onExportCSV, onDelete }) {
  const frag = cloneTemplate('tmpl-cascade-delete-modal');
  frag.querySelector('[data-field="title"]').textContent = title;
  frag.querySelector('[data-field="warning"]').textContent = warning;

  const exportReqEl = frag.querySelector('[data-field="export-required"]');
  const exportBtn = frag.querySelector('[data-action="export"]');
  const delBtn = frag.querySelector('[data-action="delete"]');

  if (onExportCSV) {
    exportReqEl.textContent = exportRequired || S.CASCADE_EXPORT_REQUIRED;
    exportBtn.addEventListener('click', () => {
      onExportCSV();
      delBtn.disabled = false;
    });
  } else {
    exportReqEl.remove();
    exportBtn.remove();
    delBtn.disabled = false;
  }

  frag.querySelector('[data-action="cancel"]')
    .addEventListener('click', () => dismissModal());
  delBtn.addEventListener('click', () => {
    dismissModal();
    onDelete();
  });
  showModal(frag);
}
