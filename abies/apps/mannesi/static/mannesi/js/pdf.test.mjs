// Tests for apps/mannesi/static/mannesi/js/pdf.js.
// Run with: node apps/mannesi/static/mannesi/js/pdf.test.mjs (also part of `make test-js`).

import { buildPDF } from './pdf.js';

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

const pdf = buildPDF(200, 300, [['BT /F1 10 Tf 10 10 Td <FEFF0041> Tj ET']]);
check(pdf.startsWith('%PDF-1.4'), 'PDF header');
check(pdf.includes('/Count 1'), 'page count');
check(pdf.includes('/BaseFont /Helvetica'), 'regular font resource');
check(pdf.includes('<FEFF0041>'), 'content stream preserved');
check(pdf.includes('xref'), 'xref table present');
check(pdf.endsWith('%%EOF\n'), 'EOF marker');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
