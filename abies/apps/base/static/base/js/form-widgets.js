/**
 * Shared UI widgets and wiring helpers used by all domain pages.
 *
 * These are the canonical implementations; never duplicate them locally.
 */

import * as S from './strings.js';
import { show as showModal, dismiss as dismissModal } from './modals.js';
import { cloneTemplate } from './templates.js';
import { showFormError } from './forms.js';

// ---------------------------------------------------------------------------
// Collapsible sections
// ---------------------------------------------------------------------------

/**
 * Wire click-to-toggle on a collapsible header/body pair: clicks on the
 * header toggle the `.open` class on both elements.  Optional `onToggle`
 * runs after each click with the new open state, so callers can add side
 * effects (URL sync, lazy-render, etc.) without re-implementing the
 * toggle itself.
 */
export function wireCollapsibleToggle(header, body, onToggle) {
  header.addEventListener('click', () => {
    const open = header.classList.toggle('open');
    body.classList.toggle('open', open);
    onToggle?.(open);
  });
}

// ---------------------------------------------------------------------------
// Tabbed modal
// ---------------------------------------------------------------------------

/**
 * Wire tab-switching + height-locking on a tabbed-modal DOM structure
 * (cloned from a server-rendered `<template>`).  Expects `root` to
 * contain `.modal-tab-bodies`, which in turn contains one
 * `.modal-tab-body[data-path="X"]` per tab, and `.modal-tabs`
 * containing matching `.modal-tab[data-path="X"]`.
 *
 * Returns `{ switchTab, lockHeight }`.  Call `lockHeight()` AFTER the
 * fragment is in the DOM — `offsetHeight` is 0 on a detached node.
 */
export function wireTabbedModal(root, { initialTab, onSwitch } = {}) {
  const bodies = root.querySelector('.modal-tab-bodies');
  const tabBtns = [...root.querySelectorAll('.modal-tabs .modal-tab')];
  const bodyEls = [...bodies.querySelectorAll('.modal-tab-body')];

  function switchTab(id) {
    for (const btn of tabBtns) btn.classList.toggle('active', btn.dataset.path === id);
    for (const b of bodyEls) b.classList.toggle('active', b.dataset.path === id);
    onSwitch?.(id);
  }

  for (const btn of tabBtns) {
    btn.addEventListener('click', () => switchTab(btn.dataset.path));
  }

  function lockHeight() {
    for (const b of bodyEls) b.style.display = 'block';
    let max = 0;
    for (const b of bodyEls) max = Math.max(max, b.offsetHeight);
    for (const b of bodyEls) b.style.display = '';
    if (max > 0) bodies.style.minHeight = `${max}px`;
  }

  switchTab(initialTab || tabBtns[0]?.dataset.path);
  return { switchTab, lockHeight };
}

// ---------------------------------------------------------------------------
// Loading overlay
// ---------------------------------------------------------------------------

/**
 * Clear `el` and show a single full-width loading overlay with the
 * standard "caricamento..." message.  Used while a page mount is
 * awaiting its initial data.
 */
export function showLoadingIn(el) {
  el.replaceChildren();
  const loading = document.createElement('div');
  loading.className = 'loading-overlay';
  loading.textContent = S.LOADING;
  el.appendChild(loading);
}

// ---------------------------------------------------------------------------
// CSV import submission lifecycle
// ---------------------------------------------------------------------------

/**
 * Wrap an async CSV import submit with the standard form lifecycle:
 * disable submit button, show "in progress" status, dispatch the
 * caller's network call, render server-reported per-row errors or a
 * form-level error message, re-enable the submit and clear the status
 * in `finally`.
 *
 * `attempt(form)` must return one of:
 *   - `{ ok: true }`       → modal is dismissed
 *   - `{ errors: [...] }`  → error list rendered into `errorsBox`,
 *                            modal stays open
 *   - `{ error: 'msg' }`   → form-level error shown, modal stays open
 *   - anything else        → generic error shown, modal stays open
 *
 * Exceptions from `attempt` are caught and surfaced as a network error.
 *
 * Returns the resolved attempt result (or `{ error: 'network' }` on
 * exception) so the caller can chain follow-up work conditional on
 * success.
 */
export async function submitCsvImport({ form, statusBox, errorsBox, attempt }) {
  const btn = form.querySelector('button[type="submit"]');
  if (btn) btn.disabled = true;
  errorsBox.hidden = true;
  errorsBox.replaceChildren();
  statusBox.textContent = S.CSV_IMPORT_IN_PROGRESS;
  statusBox.hidden = false;
  let result;
  try {
    result = await attempt(form);
    if (result?.ok) dismissModal();
    else if (result?.errors?.length) renderCsvErrors(errorsBox, result.errors);
    else showFormError(form, result?.error || S.ERROR_GENERIC);
  } catch {
    showFormError(form, S.ERROR_NETWORK);
    result = { error: 'network' };
  } finally {
    if (btn) btn.disabled = false;
    statusBox.hidden = true;
  }
  return result;
}

function renderCsvErrors(box, errors) {
  box.replaceChildren();
  const ul = document.createElement('ul');
  for (const e of errors.slice(0, 50)) {
    const li = document.createElement('li');
    li.textContent = e;
    ul.appendChild(li);
  }
  if (errors.length > 50) {
    const more = document.createElement('li');
    more.textContent = S.CSV_EXTRA_ERRORS(errors.length - 50);
    ul.appendChild(more);
  }
  box.appendChild(ul);
  box.hidden = false;
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
