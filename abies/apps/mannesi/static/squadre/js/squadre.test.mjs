// Tests for Squadre report calculations.
// Run with: node apps/mannesi/static/squadre/js/squadre.test.mjs (also part of `make test-js`).

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-squadre-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'squadre'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'squadre', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const B = await import(staticModule('squadre/js/squadre.js'));
const { PDFDocument, buildPDF, decimalRight } = await import(staticModule('squadre/js/pdf.js'));
const S = await import(staticModule('base/js/strings.js'));
const { ROW_ID, VERSION } = await import(staticModule('base/js/constants.js'));

let passed = 0;
let failed = 0;
function eq(actual, expected, msg) {
  if (Object.is(actual, expected)) {
    passed += 1;
  } else {
    failed += 1;
    console.error(`FAIL ${msg}: expected ${expected}, got ${actual}`);
  }
}

const prelievi = {
  columns: [ROW_ID, S.COL_DATE, S.COL_CREW, S.COL_TYPE, S.COL_QUINTALS],
  rows: [
    [1, '2026-01-20', 'Alfa', 'Legna', 100],
    [2, '2026-02-20', 'Alfa', 'Legna', 80],
    [3, '2026-03-20', 'Alfa', 'Legna', 70],
  ],
};
const hours = {
  columns: [ROW_ID, VERSION, S.COL_DATE, S.COL_CREW, S.COL_HOURS, S.COL_NOTE],
  rows: [
    [1, 1, '2026-01-20', 'Alfa', 6, ''],
    [2, 1, '2026-02-20', 'Alfa', 7, ''],
    [3, 1, '2026-03-20', 'Alfa', 8, ''],
  ],
};
const credits = {
  columns: [ROW_ID, VERSION, S.COL_DATE, S.COL_CREW, S.COL_CREDITS_Q, S.COL_NOTE],
  rows: [
    [1, 1, '2025-12-28', 'Alfa', 2, 'year boundary'],
    [2, 1, '2026-01-10', 'Alfa', 10, 'current month'],
    [3, 1, '2026-02-10', 'Alfa', 5, 'following month'],
    [4, 1, '2026-01-10', 'Beta', 99, 'other crew'],
  ],
};
const meta = { products: ['Legna'] };

function report(month) {
  const reports = B.buildReportsFromDigests(month, meta, prelievi, hours, credits);
  eq(reports.length, 1, `${month} report count`);
  return reports[0];
}

let r = report('2026-01');
eq(r.credits, 8, 'January acconti add January and subtract December');
eq(r.totalProduction + r.credits, 108, 'January adjusted quintali total');

r = report('2026-02');
eq(r.credits, -5, 'February acconti add February and subtract January');
eq(r.totalProduction + r.credits, 75, 'February adjusted quintali total');

r = report('2026-03');
eq(r.credits, -5, 'March acconti subtract February when March has none');
eq(r.totalProduction + r.credits, 65, 'March adjusted quintali total');

eq(B.buildReportsFromDigests('not-a-month', meta, prelievi, hours, credits).length, 0,
   'invalid month returns no reports');

const doc = new PDFDocument({ landscape: true });
B.drawReport(doc, '2026-01', {
  crew: 'Alfa',
  hours: 7.5,
  productTotals: [],
  totalProduction: 0,
  credits: 0,
  harvests: [],
  columns: {},
});
const pdf = buildPDF(doc.width, doc.height, doc.pages);
const valueComma = 34 + 150 + 44;
const valueRight = decimalRight(doc, valueComma);
const hoursX = valueRight - doc.textWidth('7,50', { size: 10 });
const hoursY = doc.height - 88;
eq(
  pdf.includes(`${hoursX.toFixed(2)} ${hoursY.toFixed(2)} Td (7,50)`),
  true,
  'Ore lavorate value is right-aligned to the report numeric column',
);

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
