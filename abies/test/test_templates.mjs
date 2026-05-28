// Tests for apps/base/static/base/js/templates.js — cloneTemplate utility.
// Run with: node test/test_templates.mjs (also part of `make test-js`).
//
// Uses a minimal DOM mock — no jsdom dependency.

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

function assertThrows(fn, expectedSubstring, msg) {
  try {
    fn();
    failed++;
    console.error(`FAIL ${msg}: expected throw, none raised`);
  } catch (e) {
    if (e.message.includes(expectedSubstring)) {
      passed++;
    } else {
      failed++;
      console.error(`FAIL ${msg}: wrong error message`);
      console.error(`  expected to include: ${expectedSubstring}`);
      console.error(`                  got: ${e.message}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Minimal DOM mock
// ---------------------------------------------------------------------------

class MockNode {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.textContent = '';
    this.className = '';
    this.dataset = {};
    this._listeners = {};
  }
  cloneNode(deep) {
    const c = new MockNode(this.tagName);
    c.textContent = this.textContent;
    c.className = this.className;
    c.attributes = { ...this.attributes };
    c.dataset = { ...this.dataset };
    if (deep) c.children = this.children.map(ch => ch.cloneNode(true));
    return c;
  }
  querySelector(sel) { return null; }
  querySelectorAll(sel) { return []; }
  appendChild(child) { this.children.push(child); }
  append(...args) { this.children.push(...args); }
}

class MockTemplate {
  constructor(content) {
    this.content = content;
  }
}

const templates = {};

globalThis.document = {
  getElementById: (id) => templates[id] || null,
};

// ---------------------------------------------------------------------------
// cloneTemplate
// ---------------------------------------------------------------------------

import { cloneTemplate } from '../apps/base/static/base/js/templates.js';

console.log('cloneTemplate');

// Set up a mock template.
const mockContent = new MockNode('fragment');
mockContent.children = [new MockNode('div'), new MockNode('p')];
templates['tmpl-test'] = new MockTemplate(mockContent);

const clone = cloneTemplate('tmpl-test');
assertEqual(clone.tagName, 'fragment', 'cloned root is the fragment');
assertEqual(clone.children.length, 2, 'deep clone preserves children');
assertEqual(clone !== mockContent, true, 'clone is a new object');
assertEqual(clone.children[0] !== mockContent.children[0], true, 'children are cloned');

// Missing template throws.
assertThrows(
  () => cloneTemplate('nonexistent'),
  'Template #nonexistent not found',
  'missing template throws with id',
);

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
