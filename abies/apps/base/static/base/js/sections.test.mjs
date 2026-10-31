import assert from 'node:assert/strict';
import { wirePageSections } from './ui-widgets.js';

function node() {
  const classes = new Set();
  return {
    classes, handler: null,
    classList: { toggle(name, force) {
      const open = force ?? !classes.has(name);
      if (open) classes.add(name); else classes.delete(name);
      return open;
    } },
    addEventListener(_event, callback) { this.handler = callback; },
    click() { this.handler(); },
  };
}
const pairs = { a: { header: node(), body: node() }, b: { header: node(), body: node() } };
const root = { querySelector(selector) {
  const key = selector.match(/data-section="([^"]+)"/)[1];
  return pairs[key]?.[selector.includes('header') ? 'header' : 'body'] || null;
} };
const sections = { a: {}, b: {}, missing: {} };
const calls = [];
wirePageSections(root, sections, ['a', 'b', 'missing'], ['b'], (state, open) => {
  assert.equal(state.open, open);
  assert.equal(state.header.classes.has('open'), open);
  assert.equal(state.body.classes.has('open'), open);
  calls.push([state, open]);
});
assert.equal(sections.a.open, false);
assert.equal(sections.b.open, true);
assert.equal(calls.length, 0, 'initial URL restoration does not trigger click side effects');
pairs.a.header.click();
pairs.a.header.click();
assert.deepEqual(calls.map(([, open]) => open), [true, false]);
assert.equal(calls[0][0], sections.a);
console.log('Section binding: URL restoration, missing DOM and one callback per click passed');
