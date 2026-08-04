// Tests for Bosco observation PDF helpers.
// Run with: node apps/bosco/static/bosco/js/observation-pdf.test.mjs.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-observation-pdf-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const O = await import(staticModule('bosco/js/observation-pdf.js'));
const { PDFDocument } = await import(staticModule('base/js/pdf.js'));
const {
  FIELD_CATEGORIES, FIELD_DATE, FIELD_LAT, FIELD_LON, FIELD_NAME,
} = await import(staticModule('base/js/constants.js'));

let failed = 0;
let passed = 0;
function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

const observation = {
  [FIELD_DATE]: '2026-08-03',
  [FIELD_LAT]: 38.123456,
  [FIELD_LON]: 16.654321,
  [FIELD_CATEGORIES]: [{ [FIELD_NAME]: 'Viabilità interna' }],
};
assertEqual(O.observationCategories(observation), ['Viabilità interna'],
            'observationCategories: category names');
assertEqual(O.primaryCategory(O.observationCategories(observation)), 'Viabilità interna',
            'primaryCategory: first category');
assertEqual(
  O.observationPDFFilename(observation, {
    regionName: 'Serra San Bruno', category: 'Viabilità interna',
  }),
  'osservazione_20260803_serrasanbruno_viabilitainterna.pdf',
  'observationPDFFilename: date, region, category slug',
);
assertEqual(O.observationPositionText(observation), '(38,12346, 16,65432)',
            'observationPositionText: localized coordinates');

const doc = new PDFDocument();
const lines = O.wrapText(doc, 'abc def ghi', 24, { size: 10 });
assertEqual(lines, ['abc', 'def', 'ghi'], 'wrapText: wraps by PDF text width');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
