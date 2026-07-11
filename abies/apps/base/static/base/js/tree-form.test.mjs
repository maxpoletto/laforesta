// Regression tests for shared tree-form live volume/mass preview.
// Run with: node apps/base/static/base/js/tree-form.test.mjs

import { wireVMPreview, ID_D_CM, ID_H_M, ID_SPECIES, ID_PREVIEW, ID_VOLUME, ID_MASS } from './tree-form.js';

let passed = 0;
let failed = 0;

function check(ok, message) {
  if (ok) passed++;
  else { failed++; console.error(`FAIL ${message}`); }
}

function eq(actual, expected, message) {
  check(actual === expected, `${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

class MockElement {
  constructor(tagName, { id = '', value = '', dataset = {} } = {}) {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.value = value;
    this.dataset = { ...dataset };
    this.hidden = false;
    this.textContent = '';
    this.options = [];
    this.selectedIndex = 0;
    this.listeners = {};
  }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatchEvent(type) { for (const fn of this.listeners[type] || []) fn({ target: this }); }
}

class MockForm {
  constructor(elements) { this.elements = elements; }
  querySelector(selector) {
    if (!selector.startsWith('#')) return null;
    return this.elements[selector.slice(1)] || null;
  }
}

const d = new MockElement('input', { id: ID_D_CM, value: '30' });
const h = new MockElement('input', { id: ID_H_M, value: '20' });
const species = new MockElement('select', { id: ID_SPECIES });
species.options = [
  new MockElement('option', { dataset: { name: 'Abete', density: '', presslerDefault: '0.45' } }),
];
const preview = new MockElement('div', { id: ID_PREVIEW });
const volume = new MockElement('input', { id: ID_VOLUME });
const mass = new MockElement('input', { id: ID_MASS });
const form = new MockForm({
  [ID_D_CM]: d,
  [ID_H_M]: h,
  [ID_SPECIES]: species,
  [ID_PREVIEW]: preview,
  [ID_VOLUME]: volume,
  [ID_MASS]: mass,
});

wireVMPreview(form);

check(!preview.hidden, 'preview remains visible when volume is computable');
check(volume.value !== '', 'volume hidden field is still populated without density');
eq(mass.value, '', 'mass hidden field stays blank without density');
check(preview.textContent.includes('m = — q'), 'preview renders missing mass as a dash');

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
