import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';
import { TableWrapper } from './table.js';
import * as cache from './cache.js';
import { bindTableData } from './page-sync.js';

installDOM();
const require = createRequire(import.meta.url);
globalThis.window = { SortableTable: require('../../vendor/sortable-table/sortable-table.js') };
const digest = { columns: ['row_id', 'version', 'Name'],
  rows: Array.from({ length: 32 }, (_, i) => [i + 1, 1, `Tree ${i + 1}`]) };
cache.set('table-binding', structuredClone(digest));
const table = new TableWrapper({ container: new Element(), digest,
  columnDefs: { version: { hidden: true } }, inlineToolbar: false });
table.setSearchText('Tree');
table.setSort({ column: 'Name', ascending: false });
table._table.goToPage(2);
const view = table._tableEl;
let updates = [];
const dispose = bindTableData({ dataId: 'table-binding', table,
  onUpdate: (source, filtered) => updates.push([source.rows.length, filtered.length]) });
cache.applyResponseChanges({ deletes: [1, 2].map(row_id => ({ data_id: 'table-binding', row_id })) });
assert.equal(table._tableEl, view);
assert.equal(table.getFilteredRows().length, 30);
assert.equal(table._table.currentPage, 2);
assert.equal(table.getSearchText(), 'Tree');
assert.deepEqual(table.getSort(), { column: 'Name', ascending: false });
assert.deepEqual(updates, [[30, 30]], 'One response produces one complete table/dependent update');
cache.applyResponseChanges({ invalidates: { data_ids: ['table-binding'] } });
assert.deepEqual(updates, [[30, 30]], 'An evicted digest is not rendered as an empty table');
dispose();
cache.set('table-binding', structuredClone(digest));
cache.applyResponseChanges({ deletes: [{ data_id: 'table-binding', row_id: 3 }] });
assert.equal(updates.length, 1, 'Disposal stops stale-view callbacks');
table.destroy();
console.log('Cache-to-table binding preserves view state and has scoped disposal');
