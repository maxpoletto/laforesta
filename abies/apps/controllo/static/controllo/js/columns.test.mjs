import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { matchesSearch, TableWrapper } from '../../../../base/static/base/js/table.js';
import * as S from '../../../../base/static/base/js/strings.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-audit-columns-'));
process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }));
fs.mkdirSync(path.join(root, 'controllo/js'), { recursive: true });
fs.mkdirSync(path.join(root, 'base/js'), { recursive: true });
fs.copyFileSync(path.join(here, 'columns.js'), path.join(root, 'controllo/js/columns.js'));
for (const name of ['constants.js', 'strings.js', 'strings_it.js']) {
  fs.copyFileSync(path.join(here, '../../../../base/static/base/js', name), path.join(root, 'base/js', name));
}
const { auditColumnDefs } = await import(pathToFileURL(path.join(root, 'controllo/js/columns.js')));
const require = createRequire(import.meta.url);
const SortableTable = require('../../../../base/static/vendor/sortable-table/sortable-table.js');
const columns = ['row_id', S.COL_TIMESTAMP, S.COL_USER, S.COL_TABLE,
  S.COL_ACTION, S.COL_OLD_VALUE, S.COL_NEW_VALUE, 'timestamp_label'];
const defs = auditColumnDefs(columns);
const column = defs[S.COL_TIMESTAMP];
const rows = [
  ['base.crew:1', '2026-10-25T00:50:00.000000+00:00', '', 'A', '', '', '', '2026-10-25 02:50:00'],
  ['base.crew:2', '2026-10-25T01:10:00.000100+00:00', '', 'B', '', '', '', '2026-10-25 02:10:00'],
  ['base.crew:3', '2026-10-25T01:10:00.000200+00:00', '', 'C', '', '', '', '2026-10-25 02:10:00'],
];
const table = { originalData: [...rows] };
SortableTable.prototype._sortData.call(table, 1, column.type, false);
assert.deepEqual(table.originalData.map(row => row[0]), ['base.crew:3', 'base.crew:2', 'base.crew:1']);
SortableTable.prototype._sortData.call(table, 1, column.type, true);
assert.deepEqual(table.originalData.map(row => row[0]), ['base.crew:1', 'base.crew:2', 'base.crew:3']);
assert.equal(column.formatter(rows[0][1], rows[0]), '2026-10-25 02:50:00');
assert.equal(defs.timestamp_label.hidden, true);
const searchColumns = columns.map(key => ({ key, label: key, hidden: key === 'row_id', ...defs[key] }));
assert.equal(matchesSearch(rows[0], [`${S.COL_TIMESTAMP.toLowerCase()}:02:50:00`], searchColumns), true);
const wrapper = Object.create(TableWrapper.prototype);
wrapper._table = { data: [...table.originalData].reverse() };
wrapper._stColumns = searchColumns;
wrapper.csvFormat = S.TABLE_CSV_FORMAT;
wrapper.labels = S.TABLE_LABELS;
const csv = wrapper.getCSV();
assert.ok(csv.includes('2026-10-25T01:10:00.000200+00:00'));
assert.ok(csv.indexOf('.000200') < csv.indexOf('.000100'));
assert.equal(csv.includes('timestamp_label'), false);
console.log('Audit timestamps: sort precision, DST, display/search and CSV checks passed');
