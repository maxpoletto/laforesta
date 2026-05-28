// Tests for showConfirmModal / showCascadeDeleteModal in form-widgets.js.
// Run with: node test/test_form_widgets_modals.mjs (also part of `make test-js`).
//
// These functions depend on the DOM (cloneTemplate, showModal, dismissModal),
// so we provide a minimal mock that tracks calls and state.

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

// ---------------------------------------------------------------------------
// DOM mock infrastructure
// ---------------------------------------------------------------------------

class MockElement {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.textContent = '';
    this.className = '';
    this.hidden = false;
    this.disabled = false;
    this.title = '';
    this.dataset = {};
    this._listeners = {};
    this._removed = false;
  }
  setAttribute(k, v) { this[k] = v; }
  getAttribute(k) { return this[k]; }
  addEventListener(type, fn) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }
  click() { (this._listeners.click || []).forEach(fn => fn()); }
  appendChild(child) { this.children.push(child); return child; }
  append(...args) { this.children.push(...args); }
  replaceChildren() { this.children = []; }
  remove() { this._removed = true; }

  querySelector(sel) {
    return this._find(sel);
  }
  querySelectorAll(sel) {
    return this._findAll(sel);
  }

  _find(sel) {
    // Minimal attribute-value selector: [attr="val"]
    const attrMatch = sel.match(/^\[([^=]+)="([^"]+)"\]$/);
    if (attrMatch) {
      const [, attr, val] = attrMatch;
      const key = attr.startsWith('data-') ? attr.slice(5) : null;
      if (key) {
        if (this.dataset[key] === val) return this;
        for (const ch of this.children) {
          if (ch instanceof MockElement) {
            const found = ch._find(sel);
            if (found) return found;
          }
        }
      }
      return null;
    }
    // Class selector: .class
    if (sel.startsWith('.')) {
      const cls = sel.slice(1);
      if (this.className.includes(cls)) return this;
      for (const ch of this.children) {
        if (ch instanceof MockElement) {
          const found = ch._find(sel);
          if (found) return found;
        }
      }
      return null;
    }
    return null;
  }
  _findAll(sel) {
    const results = [];
    if (sel.startsWith('.')) {
      const cls = sel.slice(1);
      if (this.className.includes(cls)) results.push(this);
    }
    for (const ch of this.children) {
      if (ch instanceof MockElement) results.push(...ch._findAll(sel));
    }
    return results;
  }

  cloneNode(deep) {
    const c = new MockElement(this.tagName);
    c.textContent = this.textContent;
    c.className = this.className;
    c.hidden = this.hidden;
    c.disabled = this.disabled;
    c.dataset = { ...this.dataset };
    if (deep) {
      c.children = this.children.map(ch =>
        ch instanceof MockElement ? ch.cloneNode(true) : ch);
    }
    return c;
  }
}

// Build mock templates matching _shell_templates_it.html structure.
function buildConfirmTemplate() {
  const frag = new MockElement('fragment');
  const p = new MockElement('p');
  p.dataset.field = 'message';
  const actions = new MockElement('div');
  actions.className = 'form-actions';
  const cancel = new MockElement('button');
  cancel.className = 'btn';
  cancel.dataset.action = 'cancel';
  cancel.textContent = 'Annulla';
  const confirm = new MockElement('button');
  confirm.className = 'btn btn-primary';
  confirm.dataset.action = 'confirm';
  actions.append(cancel, confirm);
  frag.append(p, actions);
  return frag;
}

function buildCascadeTemplate() {
  const frag = new MockElement('fragment');
  const h2 = new MockElement('h2');
  h2.className = 'cascade-confirm-title';
  h2.dataset.field = 'title';
  const warn = new MockElement('p');
  warn.className = 'cascade-confirm-warning';
  warn.dataset.field = 'warning';
  const expReq = new MockElement('p');
  expReq.dataset.field = 'export-required';
  const actions = new MockElement('div');
  actions.className = 'form-actions';
  const cancel = new MockElement('button');
  cancel.className = 'btn';
  cancel.dataset.action = 'cancel';
  const exportBtn = new MockElement('button');
  exportBtn.className = 'btn btn-primary';
  exportBtn.dataset.action = 'export';
  exportBtn.textContent = 'Esporta CSV';
  const delBtn = new MockElement('button');
  delBtn.className = 'btn btn-primary cascade-delete-btn';
  delBtn.dataset.action = 'delete';
  delBtn.disabled = true;
  actions.append(cancel, exportBtn, delBtn);
  frag.append(h2, warn, expReq, actions);
  return frag;
}

const templateStore = {
  'tmpl-confirm-modal': { content: buildConfirmTemplate() },
  'tmpl-cascade-delete-modal': { content: buildCascadeTemplate() },
};

// Track modal show/dismiss calls.
let modalShown = null;
let modalDismissed = false;
let dismissCallbacks = [];

globalThis.document = {
  getElementById: (id) => {
    if (id === 'modal-container') {
      return {
        replaceChildren() {},
        appendChild(child) { modalShown = child; },
        classList: { add() {}, remove() {} },
      };
    }
    return templateStore[id] || null;
  },
  addEventListener: () => {},
  removeEventListener: () => {},
  createElement: (tag) => new MockElement(tag),
};

// Import after mocking document — ES module evaluation happens at import time
// for modals.js which caches document.getElementById('modal-container').
import { showConfirmModal, showCascadeDeleteModal } from '../apps/base/static/base/js/form-widgets.js';
// Access dismiss directly to reset state between tests.
import { dismiss as dismissModal } from '../apps/base/static/base/js/modals.js';

function resetModal() {
  modalShown = null;
  modalDismissed = false;
}

function findByDataset(el, key, value) {
  if (!el || !(el instanceof MockElement)) return null;
  if (el.dataset[key] === value) return el;
  for (const ch of el.children) {
    const found = findByDataset(ch, key, value);
    if (found) return found;
  }
  return null;
}

// ---------------------------------------------------------------------------
// showConfirmModal
// ---------------------------------------------------------------------------

console.log('showConfirmModal');

resetModal();
let confirmCalled = false;
showConfirmModal('Are you sure?', () => { confirmCalled = true; });

assertEqual(modalShown !== null, true, 'modal was shown');
// Find the message element in the shown content.
const msgEl = findByDataset(modalShown, 'field', 'message');
assertEqual(msgEl?.textContent, 'Are you sure?', 'message text set');

// Confirm button gets default label (ACTION_DELETE = "Elimina").
const okBtn = findByDataset(modalShown, 'action', 'confirm');
assertEqual(okBtn?.textContent, 'Elimina', 'default confirm label');

// Click confirm triggers callback.
okBtn.click();
assertEqual(confirmCalled, true, 'onConfirm called on click');

// Custom confirm label.
resetModal();
showConfirmModal('Delete?', () => {}, { confirmLabel: 'Rimuovi' });
const okBtn2 = findByDataset(modalShown, 'action', 'confirm');
assertEqual(okBtn2?.textContent, 'Rimuovi', 'custom confirm label');

// Cancel dismisses without calling onConfirm.
resetModal();
let confirmCalled2 = false;
showConfirmModal('test', () => { confirmCalled2 = true; });
const cancelBtn = findByDataset(modalShown, 'action', 'cancel');
cancelBtn.click();
assertEqual(confirmCalled2, false, 'cancel does not call onConfirm');

// Async onConfirm: dismiss happens before await.
resetModal();
let asyncOrder = [];
showConfirmModal('async test', async () => {
  asyncOrder.push('callback');
  await Promise.resolve();
  asyncOrder.push('after-await');
});
const okBtn3 = findByDataset(modalShown, 'action', 'confirm');
okBtn3.click();
assertEqual(asyncOrder[0], 'callback', 'async callback starts synchronously');
await Promise.resolve();
assertEqual(asyncOrder.length, 2, 'async callback completes');

// ---------------------------------------------------------------------------
// showCascadeDeleteModal — with export
// ---------------------------------------------------------------------------

console.log('showCascadeDeleteModal — with export');

resetModal();
let exportCalled = false;
let deleteCalled = false;
showCascadeDeleteModal({
  title: 'Confirm delete',
  warning: 'This is dangerous',
  exportRequired: 'Export first!',
  onExportCSV: () => { exportCalled = true; },
  onDelete: () => { deleteCalled = true; },
});

assertEqual(modalShown !== null, true, 'cascade modal shown');
const titleEl = findByDataset(modalShown, 'field', 'title');
assertEqual(titleEl?.textContent, 'Confirm delete', 'title set');
const warnEl = findByDataset(modalShown, 'field', 'warning');
assertEqual(warnEl?.textContent, 'This is dangerous', 'warning set');
const expEl = findByDataset(modalShown, 'field', 'export-required');
assertEqual(expEl?.textContent, 'Export first!', 'export-required set');
assertEqual(expEl?._removed, false, 'export-required not removed');

const delBtn = findByDataset(modalShown, 'action', 'delete');
assertEqual(delBtn?.disabled, true, 'delete starts disabled');

const expBtn = findByDataset(modalShown, 'action', 'export');
assertEqual(expBtn?._removed, false, 'export button present');
expBtn.click();
assertEqual(exportCalled, true, 'onExportCSV called');
assertEqual(delBtn.disabled, false, 'delete enabled after export');

delBtn.click();
assertEqual(deleteCalled, true, 'onDelete called');

// ---------------------------------------------------------------------------
// showCascadeDeleteModal — without export (simple warning)
// ---------------------------------------------------------------------------

console.log('showCascadeDeleteModal — without export');

resetModal();
let deleteCalled2 = false;
showCascadeDeleteModal({
  title: 'Delete item',
  warning: 'Gone forever',
  onDelete: () => { deleteCalled2 = true; },
});

const expEl2 = findByDataset(modalShown, 'field', 'export-required');
assertEqual(expEl2?._removed, true, 'export-required paragraph removed');
const expBtn2 = findByDataset(modalShown, 'action', 'export');
assertEqual(expBtn2?._removed, true, 'export button removed');

const delBtn2 = findByDataset(modalShown, 'action', 'delete');
assertEqual(delBtn2?.disabled, false, 'delete starts enabled (no export)');
delBtn2.click();
assertEqual(deleteCalled2, true, 'onDelete called without export');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
