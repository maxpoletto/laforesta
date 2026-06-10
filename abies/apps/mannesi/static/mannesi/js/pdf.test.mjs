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

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
