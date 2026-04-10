/**
 * Modal display: errors, help, and arbitrary content.
 */

import * as S from './strings.js';

let container = null;

function getContainer() {
  if (!container) container = document.getElementById('modal-container');
  return container;
}

function onKeyDown(e) {
  if (e.key === 'Escape') dismiss();
}

/**
 * Show a modal containing a DOM element.
 *
 * @param {HTMLElement} content
 */
export function show(content) {
  const el = getContainer();
  el.replaceChildren();
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.appendChild(content);
  el.appendChild(modal);
  el.classList.add('open');
  document.addEventListener('keydown', onKeyDown);
}

/**
 * Show an error modal (red text + dismiss button).
 *
 * @param {string} message — plain text, safely inserted via textContent.
 */
export function showError(message) {
  const frag = document.createDocumentFragment();

  const p = document.createElement('p');
  p.className = 'modal-error';
  p.textContent = message;
  frag.appendChild(p);

  const btn = document.createElement('button');
  btn.className = 'btn btn-primary';
  btn.textContent = S.DISMISS;
  btn.addEventListener('click', dismiss);
  frag.appendChild(btn);

  show(frag);
}

/**
 * Dismiss the currently open modal.
 */
export function dismiss() {
  const el = getContainer();
  el.classList.remove('open');
  el.replaceChildren();
  document.removeEventListener('keydown', onKeyDown);
}
