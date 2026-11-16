// Tests for apps/base/static/base/js/pdf.js.
// Run with: node apps/mannesi/static/squadre/js/pdf.test.mjs (also part of `make test-js`).

import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pdfModule = path.resolve(here, '../../../../base/static/base/js/pdf.js');
const { PDFDocument, buildPDF, decimalRight } = await import(pathToFileURL(pdfModule));

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
check(pdf.includes('<41284229205c20e0>'),
      'WinAnsi hex text preserves diacritics');
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

const reportDoc = new PDFDocument({ landscape: true });
const commaX = 180;
const numericRight = decimalRight(reportDoc, commaX);
const expectedRight = commaX + reportDoc.textWidth(',0', { size: 10 });
check(Math.abs(numericRight - expectedRight) < 0.001, 'decimalRight follows one-decimal values');
reportDoc.textRight(numericRight, 42, 'Quintali', { size: 10, bold: true });
const reportPdf = buildPDF(reportDoc.width, reportDoc.height, reportDoc.pages);
const headerX = numericRight - reportDoc.textWidth('Quintali', { size: 10, bold: true });
check(reportPdf.includes(`${headerX.toFixed(2)} 553.28 Td (Quintali)`),
      'Squadre report Quintali header is right-aligned to the numeric column');

const imageDoc = new PDFDocument();
const image = imageDoc.addJPEGImage({ dataBase64: '/9j/2Q==', width: 1, height: 1 });
imageDoc.image(10, 20, 30, 40, image);
const imagePdf = buildPDF(imageDoc.width, imageDoc.height, imageDoc.pages, imageDoc.images);
check(imagePdf.includes('/Subtype /Image'), 'image XObject emitted');
check(imagePdf.includes('/Filter [/ASCIIHexDecode /DCTDecode]'),
      'JPEG image uses ASCIIHex and DCT filters');
check(imagePdf.includes('/Im1 Do'), 'page content draws image XObject');
check(imagePdf.includes('ffd8ffd9>'), 'image data is emitted as ASCII hex');

const vectorDoc = new PDFDocument();
vectorDoc.clipRect(10, 20, 100, 200);
vectorDoc.shape([[[10, 20], [30, 40]]]);
vectorDoc.shape([
  [[10, 20], [60, 20], [60, 70], [10, 70]],
  [[20, 30], [30, 30], [30, 40], [20, 40]],
], { close: true, stroke: [.4, .5, .6], fill: [.1, .2, .3], width: 2, dash: [3, 2] });
vectorDoc.endClip();
vectorDoc.text(150, 250, 'Outside the context clip');
const vectorPdf = buildPDF(vectorDoc.width, vectorDoc.height, vectorDoc.pages);
check(vectorPdf.includes('q 10 621.89 100 200 re W n'), 'clip rectangle uses PDF bottom-left coordinates');
check(vectorPdf.includes('q 0 0 0 RG 0.60 w [] 0 d 10 821.89 m 30 801.89 l S Q'),
      'default open vectors use black strokes and preserve their point coordinates');
check(vectorPdf.includes('0.40 0.50 0.60 RG 0.10 0.20 0.30 rg 2 w [3 2] 0 d'),
      'filled vectors preserve their stroke, fill and dash styles');
check(vectorPdf.includes(' h B* Q\nQ\nBT'),
      'polygon holes use even/odd fill and clipping ends before subsequent text');

console.log(`${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
