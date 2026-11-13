import assert from 'node:assert/strict';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';
import { confirmDeleteSelection } from './table-delete.js';
import * as cache from './cache.js';
import * as modals from './modals.js';

const body = installDOM();
const modal = new Element();
modal.id = 'modal-container';
body.appendChild(modal);
const template = new Element('template');
template.id = 'tmpl-delete-selected-modal';
template.content = new Element('#fragment');
for (const [tag, field, value] of [['p', 'field', 'message'], ['ul', 'target', 'records'],
                                  ['button', 'action', 'cancel'], ['button', 'action', 'confirm']]) {
  const el = new Element(tag);
  el.dataset[field] = value;
  template.content.appendChild(el);
}
body.appendChild(template);
const dataId = 'selected-trees';
const digest = { columns: ['row_id', 'Name', 'version'], rows: [[1, '<unsafe>', 2], [2, 'Tree 2', 3], [3, 'Keep', 1]] };
let posts = [];
let appliedIds;
globalThis.fetch = async (url, options) => {
  posts.push({ url, body: JSON.parse(options.body) });
  return { status: 200, json: async () => ({ deletes: [1, 2].map(id => ({ data_id: dataId, row_id: id })) }) };
};
cache.set(dataId, structuredClone(digest));
const options = { dataId, rows: digest.rows.slice(0, 2), columns: digest.columns,
  descriptionColumns: ['Name'], url: '/delete/', body: { survey_id: 7 },
  onSuccess: (_data, touched) => { appliedIds = touched; } };
let pending = confirmDeleteSelection(options);
assert.equal(posts.length, 0, 'Entering confirmation never deletes data');
assert.equal(modal.querySelector('[data-target="records"]').children.length, 2);
assert.ok(modal.textContent.includes('<unsafe>'), 'Descriptions are plain text');
assert.equal(modal.querySelector('unsafe'), null);
await modal.querySelector('[data-action="cancel"]').dispatch();
assert.equal(await pending, false, 'Cancel resolves without posting');
assert.equal(posts.length, 0);
pending = confirmDeleteSelection(options);
// Confirmation holds the reviewed versions even if the cache changes meanwhile.
cache.updateRow(dataId, 1, [1, 'Changed elsewhere', 9]);
await modal.querySelector('[data-action="confirm"]').dispatch();
assert.equal(await pending, true);
assert.ok(appliedIds?.has(dataId), 'Callbacks can refresh dependents without applying cache changes twice');
assert.equal(posts.length, 1, 'One request contains the entire selected batch');
assert.deepEqual(posts[0].body.rows, [{ row_id: 1, version: 2 }, { row_id: 2, version: 3 }]);
assert.equal(posts[0].body.survey_id, 7);
assert.ok(posts[0].body.nonce);
assert.deepEqual(cache.get(dataId).rows, [[3, 'Keep', 1]]);

cache.set(dataId, structuredClone(digest));
globalThis.fetch = async () => ({ status: 400, json: async () => ({
  status: 'validation_error', message: 'Riga protetta',
}) });
pending = confirmDeleteSelection(options);
await modal.querySelector('[data-action="confirm"]').dispatch();
assert.equal(await pending, false);
assert.equal(cache.get(dataId).rows.length, 3, 'A rejected batch does not remove cached rows');
assert.ok(modal.textContent.includes('Riga protetta'));
modals.dismiss();

pending = confirmDeleteSelection(options);
modals.dismiss();
assert.equal(await pending, false, 'Escape/dismiss behaves like cancel');
console.log('Selected deletion confirmation tests passed');
