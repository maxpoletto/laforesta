// Tests for apps/mannesi/static/mannesi/js/pdf.js.
// Run with: node apps/mannesi/static/mannesi/js/pdf.test.mjs (also part of `make test-js`).

import { PDFDocument, buildPDF } from './pdf.js';

let failed = 0;
let passed = 0;
function check(condition, message) {
  if (!condition) {
    failed += 1;
    console.error(`FAIL ${message}`);
    return;
  }
  passed += 1;
}

const doc = new PDFDocument();
doc.text(10, 20, 'A(B) \\ \u00e0', { size: 10, bold: true });
const pdf = buildPDF(doc.width, doc.height, doc.pages);
check(pdf.startsWith('%PDF-1.4'), 'PDF header');
check(pdf.includes('/Count 1'), 'page count');
check(pdf.includes('/BaseFont /Helvetica'), 'regular font resource');
check(pdf.includes('/Encoding /WinAnsiEncoding'), 'font encoding');
check(pdf.includes('(A\\(B\\) \\\\ a)'), 'escaped single-byte text');
check(!pdf.includes('FEFF'), 'no UTF-16 marker in text');
check(pdf.includes('xref'), 'xref table present');
check(pdf.endsWith('%%EOF\n'), 'EOF marker');

const rightDoc = new PDFDocument();
const width = rightDoc.textWidth('12,3', { size: 10 });
check(Math.abs(width - 19.46) < 0.001, 'textWidth uses Helvetica numeric metrics');
const regularWidth = rightDoc.textWidth('Ai', { size: 10 });
const boldWidth = rightDoc.textWidth('Ai', { size: 10, bold: true });
check(Math.abs(regularWidth - 8.89) < 0.001, 'textWidth uses regular Helvetica metrics');
check(Math.abs(boldWidth - 10) < 0.001, 'textWidth uses bold Helvetica metrics');
rightDoc.textRight(100, 30, '12,3', { size: 10 });
const rightPdf = buildPDF(rightDoc.width, rightDoc.height, rightDoc.pages);
check(rightPdf.includes('80.54 811.89 Td (12,3)'), 'textRight uses measured x');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
