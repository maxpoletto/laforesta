/**
 * Form infrastructure: fetch, inject, intercept, and submit.
 *
 * Forms are Django-rendered HTML fragments fetched into #content.
 * Each form gets a client-generated idempotency nonce so the server
 * can deduplicate retries.
 */

import { postJSON } from './api.js';
import * as cache from './cache.js';
import { show as showModal, dismiss as dismissModal, showError } from './modals.js';
import * as S from './strings.js';
import { FIELD_NONCE, HTML, ROW_ID, STATUS_CONFLICT, VERSION } from './constants.js';

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
  const frag = parseHTMLFragment(html);
  content.replaceChildren(frag);
  const form = content.querySelector('form');
  if (form) injectNonce(form);
  return form;
}

/** DOMParser -> DocumentFragment for server-rendered HTML fragments. */
export function parseHTMLFragment(html) {
  const doc = new DOMParser().parseFromString(html || '', 'text/html');
  const frag = document.createDocumentFragment();
  for (const node of [...doc.body.childNodes]) frag.appendChild(node);
  return frag;
}

/**
 * Attach a submit handler that POSTs form data as JSON and dispatches
 * the response to callbacks.
 *
 * Response dispatch:
 *   200          → onSuccess(data, isSaveAndAdd)
 *   400 conflict → inline form error + onConflict(data)
 *   400 other    → inline form error + onValidationError(data)
 *   network err  → inline form error
 *
 * @param {HTMLFormElement} form
 * @param {string} postUrl
 * @param {object} callbacks
 * @param {function(data: object, isSaveAndAdd: boolean): void} callbacks.onSuccess
 * @param {function(data: object): void} [callbacks.onConflict]
 * @param {function(data: object): void} [callbacks.onValidationError]
 * @param {function(html: string, data: object): void} [callbacks.onHtml]
 *   — server returned replacement form HTML; caller re-renders and re-wires it.
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
      if (err) { showFormError(form, err); return; }
    }

    let data, status;
    try {
      ({ data, status } = await postJSON(postUrl, body));
    } catch {
      showFormError(form, S.ERROR_NETWORK);
      return;
    }

    if (status === 200) {
      await callbacks.onSuccess?.(data, isSaveAndAdd);
      return;
    }

    if (data?.status === STATUS_CONFLICT) {
      showFormError(form, data.message || S.ERROR_CONFLICT);
      callbacks.onConflict?.(data);
      return;
    }

    const html = data?.[HTML] || data?.html;
    if (html && callbacks.onHtml) {
      callbacks.onHtml(html, data);
      return;
    }

    showFormError(form, data?.message || S.ERROR_GENERIC);
    callbacks.onValidationError?.(data);
  });
}

/**
 * Delete a row guarded by optimistic-lock version.
 *
 * Reads the row's current `version` from the cached `dataId` digest — the
 * single source of truth, never a value passed in by the caller — POSTs
 * `{row_id, version, nonce}` to `url`, and reconciles the cache: on
 * success removes the row; on a version conflict refreshes the row from
 * the server's returned record so a retry carries the current version.
 * View re-rendering and any cross-digest updates are left to the
 * `onSuccess` / `onConflict` callbacks (cf. `interceptSubmit`).
 *
 * @param {string} dataId — cache id of the digest holding the row
 * @param {number} rowId
 * @param {string} url — delete endpoint
 * @param {object} [callbacks]
 * @param {function(data: object): void|Promise<void>} [callbacks.onSuccess]
 * @param {function(data: object): void|Promise<void>} [callbacks.onConflict]
 * @param {string|null} [callbacks.confirmMessage] — null when the caller already confirmed
 * @returns {Promise<boolean>} true iff the row was deleted
 */
export async function deleteRowWithVersion(
  dataId, rowId, url,
  { onSuccess, onConflict, confirmMessage = S.DELETE_CONFIRM } = {},
) {
  if (confirmMessage != null && !confirm(confirmMessage)) return false;
  const digest = cache.get(dataId);
  const row = digest?.rows.find(r => r[0] === rowId);
  if (!row) return false;
  const vcol = digest.columns.indexOf(VERSION);
  const version = vcol >= 0 ? row[vcol] : 0;

  let data, status;
  try {
    ({ data, status } = await postJSON(url, {
      [ROW_ID]: String(rowId),
      [VERSION]: String(version),
      [FIELD_NONCE]: crypto.randomUUID(),
    }));
  } catch {
    showError(S.ERROR_NETWORK);
    return false;
  }

  if (status === 200) {
    cache.removeRow(dataId, rowId);
    await onSuccess?.(data);
    return true;
  }

  if (data?.status === STATUS_CONFLICT) {
    const touched = cache.applyResponseChanges(data);
    if (!touched.size && data.record) {
      cache.updateRow(dataId, data.row_id, data.record);
    }
    await onConflict?.(data);
  }
  showError(data?.message || S.ERROR_GENERIC);
  return false;
}

/**
 * Fetch a form fragment from the server and display it in the overlay modal.
 * Returns the form element inside #modal-container, or null on error.
 */
export async function fetchModalForm(url) {
  let data;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status}`);
    data = await resp.json();
  } catch {
    showError(S.ERROR_NETWORK);
    return null;
  }
  const html = data?.[HTML] || data?.html;
  if (!html) { showError(S.ERROR_GENERIC); return null; }
  return renderModalForm(html);
}

/**
 * Parse HTML into a fragment, inject a nonce, and show in the overlay modal.
 * Used for initial render and for re-render after validation errors.
 * Returns the form element inside the modal.
 */
export function renderModalForm(html) {
  const frag = parseHTMLFragment(html);
  const form = frag.querySelector('form');
  if (form) injectNonce(form);
  showModal(frag);
  return document.querySelector('#modal-container form');
}

/** Show an inline error message inside a form (above .form-actions). */
export function showFormError(form, message) {
  let el = form.querySelector('.form-error');
  if (!el) {
    el = document.createElement('p');
    el.className = 'form-error';
    form.querySelector('.form-actions')?.before(el) || form.appendChild(el);
  }
  el.textContent = message;
}

// ---------------------------------------------------------------------------
// CSV import submission lifecycle
// ---------------------------------------------------------------------------

/**
 * CSV-import-specific form submit lifecycle.  For non-CSV submits use
 * `interceptSubmit` instead — this helper expects a per-row errors list
 * (truncated at 50 via `S.CSV_EXTRA_ERRORS`) and a caller-supplied
 * status/errors element pair, both shaped like a CSV import modal.
 *
 * Wraps an async submit: disable submit button, show "in progress"
 * status, dispatch the caller's network call, render server-reported
 * per-row errors or a form-level error message, re-enable the submit
 * and clear the status in `finally`.
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

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

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

export function injectNonce(form) {
  let input = form.querySelector(`input[name="${FIELD_NONCE}"]`);
  if (!input) {
    input = document.createElement('input');
    input.type = 'hidden';
    input.name = FIELD_NONCE;
    form.appendChild(input);
  }
  input.value = crypto.randomUUID();
}
