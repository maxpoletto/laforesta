// Tests for apps/base/static/base/js/csv-export.js — CSV export utilities.
// Run with: node apps/base/static/base/js/csv-export.test.mjs (also part of `make test-js`).

import { csvEscape, csvField, downloadCSV, exportDigest, hardenCSVFormula } from './csv-export.js';

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
assertEqual(csvField('a;b;c', FMT_IT), '"a;b;c"', 'string: separator quoted');
assertEqual(csvField('a"b', FMT_IT), '"a""b"', 'string: quote escaped');
assertEqual(csvField('a\nb', FMT_IT), '"a\nb"', 'string: newline quoted');
assertEqual(csvField('no-sep', FMT_IT), 'no-sep', 'string without separator');
assertEqual(csvField('=1+1', FMT_IT), "'=1+1", 'formula string hardened');
assertEqual(csvField('+cmd', FMT_IT), "'+cmd", 'plus formula string hardened');
assertEqual(csvField('-cmd', FMT_IT), "'-cmd", 'minus formula string hardened');
assertEqual(csvField('@cmd', FMT_IT), "'@cmd", 'at formula string hardened');
assertEqual(csvField('-4', FMT_IT), '-4', 'numeric-looking negative string preserved');
assertEqual(csvField('+3,14', FMT_IT), '+3,14', 'numeric-looking positive decimal preserved');
assertEqual(hardenCSVFormula('\tcmd'), "'\tcmd", 'tab-prefixed formula hardened');

const FMT_EN = { separator: ',', decimal: '.' };
assertEqual(csvField(3.14, FMT_EN), '3.14', 'EN: decimal stays as-is');
assertEqual(csvField('a,b', FMT_EN), '"a,b"', 'EN: comma separator quoted');
assertEqual(csvEscape('A;B', ';'), '"A;B"', 'csvEscape quotes separators');

// ---------------------------------------------------------------------------
// exportDigest — test the line-building logic
// ---------------------------------------------------------------------------

console.log('exportDigest');

// Mock downloadCSV to capture output instead of creating a Blob.
let capturedText = null;
let capturedLines = null;
let capturedFilename = null;

const originalURL = globalThis.URL;
const originalBlob = globalThis.Blob;
const originalDoc = globalThis.document;

globalThis.Blob = class {
  constructor(parts) { this.text = parts[0]; }
};
globalThis.URL = {
  createObjectURL: (blob) => {
    capturedText = blob.text;
    capturedLines = blob.text.replace(/^\ufeff/, '').split('\n');
    return 'blob:x';
  },
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

// Basic export — string descriptors (src == dst).
exportDigest(digest, ['Nome', 'Valore'], 'test.csv');
assertEqual(capturedFilename, 'test.csv', 'filename');
assertEqual(capturedText.startsWith('\ufeff'), true, 'downloadCSV prepends UTF-8 BOM');
assertEqual(capturedLines[0], 'Nome;Valore', 'header row');
assertEqual(capturedLines[1], 'Alfa;10,5', 'row 1');
assertEqual(capturedLines[2], 'Beta;20', 'row 2');
assertEqual(capturedLines[3], 'Gamma;30,3', 'row 3');
assertEqual(capturedLines.length, 4, '3 data rows + header');

// Export with renamed columns via {src, dst}.
exportDigest(digest, [
  { src: 'Nome', dst: 'Name' },
  { src: 'Valore', dst: 'Val' },
], 'renamed.csv');
assertEqual(capturedLines[0], 'Name;Val', 'renamed header');
assertEqual(capturedLines[1], 'Alfa;10,5', 'renamed data unchanged');

// Mixed: string + {src, dst}.
exportDigest(digest, ['Nome', { src: 'Valore', dst: 'Val' }], 'mixed.csv');
assertEqual(capturedLines[0], 'Nome;Val', 'mixed header');
assertEqual(capturedLines[1], 'Alfa;10,5', 'mixed data');

// Export with filter.
exportDigest(digest, ['Nome', 'Valore'], 'filtered.csv', {
  filter: row => row[digest.columns.indexOf('Tipo')] === 'A',
});
assertEqual(capturedLines.length, 3, 'filter: 2 rows + header');
assertEqual(capturedLines[1], 'Alfa;10,5', 'filter: first match');
assertEqual(capturedLines[2], 'Gamma;30,3', 'filter: second match');

// Transform-only column (no src) via {dst, transform}.
exportDigest(digest, [
  'Nome',
  { dst: 'IsA', transform: row => row[digest.columns.indexOf('Tipo')] === 'A' },
], 'transform.csv');
assertEqual(capturedLines[0], 'Nome;IsA', 'transform header');
assertEqual(capturedLines[1], 'Alfa;true', 'transform: boolean derived');
assertEqual(capturedLines[2], 'Beta;false', 'transform: boolean derived false');
assertEqual(capturedLines[3], 'Gamma;true', 'transform: third row');

// Filter + transform combined.
exportDigest(digest, [
  'Nome',
  { dst: 'Tipo', transform: row => row[digest.columns.indexOf('Tipo')] },
], 'both.csv', {
  filter: row => row[digest.columns.indexOf('Valore')] > 15,
});
assertEqual(capturedLines.length, 3, 'filter+transform: 2 rows + header');
assertEqual(capturedLines[1], 'Beta;B', 'filter+transform: row 1');
assertEqual(capturedLines[2], 'Gamma;A', 'filter+transform: row 2');

// Empty digest.
exportDigest({ columns: ['X'], rows: [] }, ['X'], 'empty.csv');
assertEqual(capturedLines.length, 1, 'empty: header only');

// Null values in data.
const nullDigest = {
  columns: ['A', 'B'],
  rows: [[null, 'ok'], ['val', null]],
};
exportDigest(nullDigest, ['A', 'B'], 'nulls.csv');
assertEqual(capturedLines[1], ';ok', 'null in first column');
assertEqual(capturedLines[2], 'val;', 'null in second column');

const formulaDigest = { columns: ['Nome'], rows: [['=cmd'], ['-4']] };
exportDigest(formulaDigest, ['Nome'], 'formula.csv');
assertEqual(capturedLines[1], "'=cmd", 'exportDigest hardens formula-looking text');
assertEqual(capturedLines[2], '-4', 'exportDigest preserves numeric-looking text');

const quotedDigest = { columns: ['A;B', 'Text'], rows: [['x;y', 'line\nnext'], ['quote', 'a"b']] };
exportDigest(quotedDigest, ['A;B', 'Text'], 'quoted.csv');
assertEqual(
  capturedText.replace(/^\ufeff/, ''),
  '"A;B";Text\n"x;y";"line\nnext"\nquote;"a""b"',
  'exportDigest escapes headers, separators, newlines, and quotes',
);

// Restore globals.
globalThis.URL = originalURL;
globalThis.Blob = originalBlob;
globalThis.document = originalDoc;

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
