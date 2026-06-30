/**
 * Shared UI widgets and wiring helpers used by all domain pages.
 *
 * These are the canonical implementations; never duplicate them locally.
 */

import * as S from './strings.js';
import { show as showModal, dismiss as dismissModal } from './modals.js';
import { cloneTemplate } from './templates.js';

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
// Cancel buttons
// ---------------------------------------------------------------------------

/**
 * Wire every `[data-action="cancel"]` button inside `container` to call
 * `callback`.  Attaches one listener per button (not delegated), so it
 * works even when `container` is a DocumentFragment whose children get
 * moved out by `appendChild` before any click happens.
 */
export function wireCancelButtons(container, callback) {
  container.querySelectorAll('[data-action="cancel"]')
    .forEach(b => b.addEventListener('click', callback));
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

export function showCascadeDeleteModal({ title, warning, exportRequired, onExport, onDelete }) {
  const frag = cloneTemplate('tmpl-cascade-delete-modal');
  frag.querySelector('[data-field="title"]').textContent = title;
  frag.querySelector('[data-field="warning"]').textContent = warning;

  const exportReqEl = frag.querySelector('[data-field="export-required"]');
  const exportBtn = frag.querySelector('[data-action="export"]');
  const delBtn = frag.querySelector('[data-action="delete"]');

  if (onExport) {
    exportReqEl.textContent = exportRequired || S.CASCADE_EXPORT_REQUIRED;
    exportBtn.addEventListener('click', () => {
      onExport();
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
