import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';
import { TableWrapper } from './table.js';
import { TABLE_LABELS } from './strings.js';

installDOM();
const require = createRequire(import.meta.url);
globalThis.window = { SortableTable: require('../../vendor/sortable-table/sortable-table.js') };
const digest = { columns: ['row_id', 'Name', 'version'],
  rows: Array.from({ length: 32 }, (_, i) => [i + 1, `Tree ${i + 1}`, 1]) };
let edited = [], deleted = [], batches = [], outcome = false;
const container = new Element();
const wrapper = new TableWrapper({
  container, digest, canModify: true,
  columnDefs: { Name: { label: 'Name' }, version: { hidden: true } },
  labels: TABLE_LABELS,
  actions: { onEdit: id => edited.push(id), onDelete: id => deleted.push(id),
    onDeleteSelected: async rows => { batches.push(rows.map(row => row[0])); return outcome; } },
});
const rows = () => container.querySelectorAll('.sortable-table-row');
const toggle = () => container.querySelector('.table-delete-toggle');
const checkboxes = () => container.querySelectorAll('input[type="checkbox"]');
const submit = () => container.querySelector('.table-delete-selected');

assert.ok(toggle(), 'Header provides the selection control');
assert.equal(toggle().getAttribute('aria-label'), TABLE_LABELS.selectForDelete);
assert.equal(checkboxes().length, 0, 'Normal mode has no selection boxes');
await rows()[0].dispatch();
assert.deepEqual(edited, [1], 'Normal row click still edits');
await toggle().dispatch();
assert.equal(checkboxes().length, 25, 'Only individual row checkboxes; no select-all box');
assert.equal(container.querySelectorAll('.action-edit').length, 0);
assert.equal(container.querySelectorAll('.action-delete').length, 0);
assert.equal(submit().disabled, true, 'An empty selection cannot be submitted');
await rows()[0].dispatch();
assert.equal(checkboxes()[0].checked, true, 'Row click selects instead of editing');
assert.deepEqual(edited, [1]);
await checkboxes()[1].dispatch();
assert.equal(checkboxes()[1].checked, true, 'Checkbox click toggles exactly once');
assert.equal(submit().textContent, 'Elimina selezionati (2)');
wrapper.setSort({ column: 'Name', ascending: false });
await container.querySelector('[data-action="next"]').dispatch();
assert.equal(checkboxes().length, 7);
await rows()[0].dispatch();
await submit().dispatch();
assert.equal(batches.length, 1);
assert.equal(batches[0].length, 3, 'Selection spans pages and follows IDs through sorting');
assert.ok(batches[0].includes(1) && batches[0].includes(2));
assert.equal(wrapper.deleteSelectionActive, true, 'Rejected/canceled deletion retains selection');
wrapper.setSearchText('Tree 1');
assert.equal(submit().disabled, true, 'Filtering clears the complete selection');
assert.ok(container.textContent.includes(TABLE_LABELS.selectionCleared));
assert.equal(wrapper.deleteSelectionActive, true);
wrapper.setSearchText('');
await rows()[0].dispatch();
const updated = { ...digest, rows: digest.rows.map(row => [...row]) };
wrapper.setData(updated);
assert.equal(submit().disabled, false, 'Unchanged refresh preserves selection');
const selectedId = rows()[0] && wrapper.rowForElement(rows()[0])[0];
updated.rows.find(row => row[0] === selectedId)[2] += 1;
wrapper.setData(updated);
assert.equal(submit().disabled, true, 'Changed versions cannot silently remain selected');
await rows()[0].dispatch();
wrapper.setExternalFilter(row => row[0] > 5);
assert.equal(submit().disabled, true, 'External context filters also clear selection');
await toggle().dispatch();
assert.equal(wrapper.deleteSelectionActive, false, 'Header cancel exits mode');
assert.equal(checkboxes().length, 0);
assert.ok(container.querySelector('.action-edit'));
await toggle().dispatch();
await rows()[0].dispatch();
outcome = true;
await submit().dispatch();
assert.equal(wrapper.deleteSelectionActive, false, 'Successful deletion exits selection mode');
assert.deepEqual(deleted, [], 'The selection control never triggers a single-row delete');
wrapper.destroy();
assert.equal(container.children.length, 0);

const readOnly = new Element();
new TableWrapper({ container: readOnly, digest, canModify: false,
  actions: { onDeleteSelected() {} } });
assert.equal(readOnly.querySelector('.table-delete-toggle'), null);
const ordinary = new Element();
new TableWrapper({ container: ordinary, digest, canModify: true,
  actions: { onDelete() {} } });
assert.equal(ordinary.querySelector('.table-delete-toggle'), null, 'Batch deletion is opt-in');
// Pending confirmation/submission freezes the controls and cannot double-submit.
const pendingHost = new Element();
let completeDelete, pendingCalls = 0;
const pendingTable = new TableWrapper({
  container: pendingHost, digest, canModify: true,
  actions: { onDelete() {}, onEdit() {}, onDeleteSelected: () => {
    pendingCalls++;
    return new Promise(resolve => { completeDelete = resolve; });
  } },
});
const pendingToggle = () => pendingHost.querySelector('.table-delete-toggle');
const pendingSubmit = () => pendingHost.querySelector('.table-delete-selected');
const pendingRows = () => pendingHost.querySelectorAll('.sortable-table-row');
await pendingToggle().dispatch();
pendingTable.setExternalFilter(() => true, 'all');
await pendingRows()[0].dispatch();
pendingTable.setExternalFilter(() => true, 'all');
assert.equal(pendingSubmit().disabled, false, 'Reapplying the same filter configuration preserves selection');
const submitted = pendingSubmit().dispatch();
assert.equal(pendingSubmit().disabled, true);
assert.equal(pendingToggle().disabled, true);
assert.ok(pendingHost.querySelectorAll('input[type="checkbox"]').every(input => input.disabled));
await pendingSubmit().dispatch();
await pendingRows()[1].dispatch();
assert.equal(pendingCalls, 1);
completeDelete(false);
await submitted;
assert.equal(pendingSubmit().disabled, false);
assert.equal(pendingHost.querySelectorAll('input[type="checkbox"]').filter(input => input.checked).length, 1);
pendingTable.setExternalFilter(() => true, 'other');
assert.equal(pendingSubmit().disabled, true, 'An actual filter configuration change clears selection');
await pendingRows()[0].dispatch();
const searchInput = new Element('input');
pendingTable.wireSearchInput(searchInput);
searchInput.value = 'Tree';
await searchInput.dispatch('input');
assert.equal(pendingSubmit().disabled, true, 'Typing clears selection before the search debounce fires');
await pendingRows()[0].dispatch();
pendingTable.setData({ columns: ['row_id', 'version', 'Name'],
  rows: digest.rows.map(row => [row[0], row[2], row[1]]) });
assert.equal(pendingSubmit().disabled, true, 'A schema change cannot reinterpret selected row snapshots');
await pendingRows()[0].dispatch();
const outstanding = pendingSubmit().dispatch();
pendingTable.destroy();
completeDelete(true);
await outstanding;
assert.equal(pendingHost.children.length, 0, 'A late response never rebuilds a destroyed table');

const protectedHost = new Element();
const protectedTable = new TableWrapper({
  container: protectedHost, digest: { ...digest, rows: digest.rows.slice(0, 2) }, canModify: true,
  actions: { onDelete() {}, onDeleteSelected() {}, deleteVisible: row => row[0] !== 2 },
});
await protectedHost.querySelector('.table-delete-toggle').dispatch();
assert.equal(protectedHost.querySelectorAll('input[type="checkbox"]').length, 1);
await protectedHost.querySelectorAll('.sortable-table-row')[1].dispatch();
assert.equal(protectedHost.querySelector('.table-delete-selected').disabled, true,
  'Rows excluded from deletion cannot be selected through row clicks');
protectedTable.destroy();
// External search/export controls still use the table's existing action row.
const externalHost = new Element();
const externalTable = new TableWrapper({
  container: externalHost, digest, inlineToolbar: false, canModify: true,
  actions: { onDelete() {}, onAdd() {}, onDeleteSelected: async () => false },
});
const selectionControls = externalHost.querySelector('.table-delete-bar');
assert.ok(selectionControls.closest('.sortable-table-controls'),
  'External filters do not create a separate selection row above the table');
assert.ok(externalHost.querySelector('.sortable-table-controls-start'),
  'A balanced empty start slot keeps pagination centered');
assert.equal(externalHost.querySelector('.table-search'), null,
  'External search is not duplicated inside the table');
assert.equal(externalHost.querySelector('.btn-export'), null,
  'External export is not duplicated inside the table');
externalTable.destroy();
console.log('Table selection integration tests passed');
