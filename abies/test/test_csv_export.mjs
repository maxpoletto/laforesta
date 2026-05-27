// Tests for apps/base/static/base/js/csv-export.js — CSV export utilities.
// Run with: node test/test_csv_export.mjs (also part of `make test-js`).

import { csvField, downloadCSV, exportDigest } from '../apps/base/static/base/js/csv-export.js';

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

const FMT_IT = { separator: ';', decimal: ',' };

// ---------------------------------------------------------------------------
// csvField
// ---------------------------------------------------------------------------

console.log('csvField');

assertEqual(csvField(null, FMT_IT), '', 'null → empty');
assertEqual(csvField(undefined, FMT_IT), '', 'undefined → empty');
assertEqual(csvField(true, FMT_IT), 'true', 'boolean true');
assertEqual(csvField(false, FMT_IT), 'false', 'boolean false');
assertEqual(csvField(3.14, FMT_IT), '3,14', 'number: decimal replaced');
assertEqual(csvField(42, FMT_IT), '42', 'integer: no decimal');
assertEqual(csvField(0, FMT_IT), '0', 'zero');
assertEqual(csvField('hello', FMT_IT), 'hello', 'plain string');
assertEqual(csvField('a;b;c', FMT_IT), 'a b c', 'string: separator replaced');
assertEqual(csvField('no-sep', FMT_IT), 'no-sep', 'string without separator');

const FMT_EN = { separator: ',', decimal: '.' };
assertEqual(csvField(3.14, FMT_EN), '3.14', 'EN: decimal stays as-is');
assertEqual(csvField('a,b', FMT_EN), 'a b', 'EN: comma separator replaced');

// ---------------------------------------------------------------------------
// exportDigest — test the line-building logic
// ---------------------------------------------------------------------------

console.log('exportDigest');

// Mock downloadCSV to capture output instead of creating a Blob.
let capturedLines = null;
let capturedFilename = null;

const originalURL = globalThis.URL;
const originalBlob = globalThis.Blob;
const originalDoc = globalThis.document;

globalThis.URL = { createObjectURL: () => 'blob:mock', revokeObjectURL: () => {} };
globalThis.Blob = class { constructor(parts) { this.data = parts[0]; } };
globalThis.document = {
  createElement: () => {
    const el = {};
    el.click = () => {
      capturedLines = el._lines;
      capturedFilename = el.download;
    };
    return new Proxy(el, {
      set(obj, prop, val) {
        if (prop === 'href') {
          // Parse the blob data (Blob constructor receives [lines.join('\n')])
        }
        obj[prop] = val;
        return true;
      },
    });
  },
};

// Override Blob to capture data.
globalThis.Blob = class {
  constructor(parts) { this.text = parts[0]; }
};
globalThis.URL = {
  createObjectURL: (blob) => { capturedLines = blob.text.split('\n'); return 'blob:x'; },
  revokeObjectURL: () => {},
};
globalThis.document = {
  createElement: () => {
    const el = {};
    el.click = () => { capturedFilename = el.download; };
    return el;
  },
};

const digest = {
  columns: ['id', 'Nome', 'Valore', 'Tipo'],
  rows: [
    [1, 'Alfa', 10.5, 'A'],
    [2, 'Beta', 20.0, 'B'],
    [3, 'Gamma', 30.3, 'A'],
  ],
};

// Basic export — all columns mapped 1:1.
exportDigest(digest, ['Nome', 'Valore'], ['Nome', 'Valore'], 'test.csv');
assertEqual(capturedFilename, 'test.csv', 'filename');
assertEqual(capturedLines[0], 'Nome;Valore', 'header row');
assertEqual(capturedLines[1], 'Alfa;10,5', 'row 1');
assertEqual(capturedLines[2], 'Beta;20', 'row 2');
assertEqual(capturedLines[3], 'Gamma;30,3', 'row 3');
assertEqual(capturedLines.length, 4, '3 data rows + header');

// Export with renamed columns.
exportDigest(digest, ['Name', 'Val'], ['Nome', 'Valore'], 'renamed.csv');
assertEqual(capturedLines[0], 'Name;Val', 'renamed header');
assertEqual(capturedLines[1], 'Alfa;10,5', 'renamed data unchanged');

// Export with filter.
exportDigest(digest, ['Nome', 'Valore'], ['Nome', 'Valore'], 'filtered.csv', {
  filter: row => row[digest.columns.indexOf('Tipo')] === 'A',
});
assertEqual(capturedLines.length, 3, 'filter: 2 rows + header');
assertEqual(capturedLines[1], 'Alfa;10,5', 'filter: first match');
assertEqual(capturedLines[2], 'Gamma;30,3', 'filter: second match');

// Export with transform.
exportDigest(digest, ['Nome', 'Derived'], ['Nome', 'Valore'], 'transform.csv', {
  transform: (row, _i, colName) =>
    colName === 'Derived' ? row[digest.columns.indexOf('Tipo')] === 'A' : undefined,
});
assertEqual(capturedLines[1], 'Alfa;true', 'transform: boolean derived');
assertEqual(capturedLines[2], 'Beta;false', 'transform: boolean derived false');

// Export with filter + transform combined.
exportDigest(digest, ['Nome', 'Derived'], ['Nome', 'Valore'], 'both.csv', {
  filter: row => row[digest.columns.indexOf('Valore')] > 15,
  transform: (row, _i, colName) =>
    colName === 'Derived' ? row[digest.columns.indexOf('Tipo')] : undefined,
});
assertEqual(capturedLines.length, 3, 'filter+transform: 2 rows + header');
assertEqual(capturedLines[1], 'Beta;B', 'filter+transform: row 1');
assertEqual(capturedLines[2], 'Gamma;A', 'filter+transform: row 2');

// Empty digest.
exportDigest({ columns: ['X'], rows: [] }, ['X'], ['X'], 'empty.csv');
assertEqual(capturedLines.length, 1, 'empty: header only');

// Null values in data.
const nullDigest = {
  columns: ['A', 'B'],
  rows: [[null, 'ok'], ['val', null]],
};
exportDigest(nullDigest, ['A', 'B'], ['A', 'B'], 'nulls.csv');
assertEqual(capturedLines[1], ';ok', 'null in first column');
assertEqual(capturedLines[2], 'val;', 'null in second column');

// Restore globals.
globalThis.URL = originalURL;
globalThis.Blob = originalBlob;
globalThis.document = originalDoc;

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
