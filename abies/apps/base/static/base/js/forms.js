/**
 * Form infrastructure: fetch, inject, intercept, and submit.
 *
 * Forms are Django-rendered HTML fragments fetched into #content.
 * Each form gets a client-generated idempotency nonce so the server
 * can deduplicate retries.
 */

import { postJSON } from './api.js';
import { showError } from './modals.js';
import * as S from './strings.js';

/**
 * Fetch a form fragment from the server and display it in #content.
 * Injects a fresh idempotency nonce as a hidden field.
 *
 * The endpoint returns JSON: { html: "..." }.
 *
 * @param {string} url — form endpoint (GET returns JSON with html field)
 * @returns {Promise<HTMLFormElement|null>}
 */
export async function fetchForm(url) {
  let data;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    showError(S.ERROR_NETWORK);
    return null;
  }
  return renderFormHTML(data.html);
}

/**
 * Render a form from an HTML string into #content.
 * Used both for initial load and for re-rendering after validation errors.
 * Injects a fresh idempotency nonce.
 *
 * @param {string} html
 * @returns {HTMLFormElement|null}
 */
export function renderFormHTML(html) {
  const content = document.getElementById('content');
  const doc = new DOMParser().parseFromString(html, 'text/html');
  content.replaceChildren(...doc.body.childNodes);
  const form = content.querySelector('form');
  if (form) injectNonce(form);
  return form;
}

/**
 * Attach a submit handler that POSTs form data as JSON and dispatches
 * the response to callbacks.
 *
 * Response dispatch:
 *   200          → onSuccess(data, isSaveAndAdd)
 *   400 conflict → error modal + onConflict(data)
 *   400 other    → error modal + onValidationError(data)
 *   network err  → error modal
 *
 * @param {HTMLFormElement} form
 * @param {string} postUrl
 * @param {object} callbacks
 * @param {function(data: object, isSaveAndAdd: boolean): void} callbacks.onSuccess
 * @param {function(data: object): void} [callbacks.onConflict]
 * @param {function(data: object): void} [callbacks.onValidationError]
 * @param {function(body: object): string|null} [callbacks.validate]
 *   — client-side pre-submit check; return error string to block, null to proceed.
 */
export function interceptSubmit(form, postUrl, callbacks) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const isSaveAndAdd = e.submitter?.dataset.action === 'save-and-add';
    const body = Object.fromEntries(new FormData(form));

    if (callbacks.validate) {
      const err = callbacks.validate(body);
      if (err) { showError(err); return; }
    }

    let data, status;
    try {
      ({ data, status } = await postJSON(postUrl, body));
    } catch {
      showError(S.ERROR_NETWORK);
      return;
    }

    if (status === 200) {
      callbacks.onSuccess(data, isSaveAndAdd);
      return;
    }

    if (data.status === 'conflict') {
      showError(data.message || S.ERROR_CONFLICT);
      callbacks.onConflict?.(data);
    } else {
      showError(data.message || S.ERROR_GENERIC);
      callbacks.onValidationError?.(data);
    }
  });
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function injectNonce(form) {
  let input = form.querySelector('input[name="nonce"]');
  if (!input) {
    input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'nonce';
    form.appendChild(input);
  }
  input.value = crypto.randomUUID();
}
