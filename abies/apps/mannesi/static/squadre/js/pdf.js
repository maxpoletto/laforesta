/** Minimal PDF writer for Squadre-generated operational PDFs. */

const A4 = { width: 595.28, height: 841.89 };

const HELVETICA_WIDTHS = {
  ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667, "'": 191,
  '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333, '.': 278, '/': 278,
  '0': 556, '1': 556, '2': 556, '3': 556, '4': 556, '5': 556, '6': 556, '7': 556,
  '8': 556, '9': 556, ':': 278, ';': 278, '<': 584, '=': 584, '>': 584, '?': 556,
  '@': 1015, A: 667, B: 667, C: 722, D: 722, E: 667, F: 611, G: 778, H: 722,
  I: 278, J: 500, K: 667, L: 556, M: 833, N: 722, O: 778, P: 667, Q: 778,
  R: 722, S: 667, T: 611, U: 722, V: 667, W: 944, X: 667, Y: 667, Z: 611,
  '[': 278, '\\': 278, ']': 278, '^': 469, _: 556, '`': 222,
  a: 556, b: 556, c: 500, d: 556, e: 556, f: 278, g: 556, h: 556, i: 222,
  j: 222, k: 500, l: 222, m: 833, n: 556, o: 556, p: 556, q: 556, r: 333,
  s: 500, t: 278, u: 556, v: 500, w: 722, x: 500, y: 500, z: 500,
  '{': 334, '|': 260, '}': 334, '~': 584,
};

const HELVETICA_BOLD_WIDTHS = {
  ...HELVETICA_WIDTHS,
  '!': 333, '"': 474, '&': 722, "'": 238, ':': 333, ';': 333, '?': 611, '@': 975,
  A: 722, B: 722, D: 722, E: 667, J: 556, K: 722, L: 611, P: 667, R: 722,
  '[': 333, ']': 333, '^': 584, '`': 333,
  b: 611, c: 556, d: 611, f: 333, g: 611, h: 611, i: 278, j: 278, k: 556,
  l: 278, m: 889, n: 611, o: 611, p: 611, q: 611, r: 389, s: 556, t: 333,
  u: 611, v: 556, w: 778, x: 556, y: 556,
};
const FALLBACK_GLYPH_WIDTH = 556;

export class PDFDocument {
  constructor({ landscape = false } = {}) {
    this.width = landscape ? A4.height : A4.width;
    this.height = landscape ? A4.width : A4.height;
    this.pages = [];
    this.current = null;
    this.addPage();
  }

  addPage() {
    this.current = [];
    this.pages.push(this.current);
  }

  text(x, y, value, { size = 10, bold = false } = {}) {
    const font = bold ? 'F2' : 'F1';
    this.current.push(
      `BT /${font} ${num(size)} Tf ${num(x)} ${num(this.height - y)} Td ${pdfString(value)} Tj ET`,
    );
  }

  textRight(xRight, y, value, opts = {}) {
    this.text(xRight - this.textWidth(value, opts), y, value, opts);
  }

  textWidth(value, { size = 10, bold = false } = {}) {
    const widths = bold ? HELVETICA_BOLD_WIDTHS : HELVETICA_WIDTHS;
    let width = 0;
    for (const ch of safeText(value)) {
      width += widths[ch] || FALLBACK_GLYPH_WIDTH;
    }
    return width * size / 1000;
  }

  line(x1, y1, x2, y2) {
    this.current.push(`${num(x1)} ${num(this.height - y1)} m ${num(x2)} ${num(this.height - y2)} l S`);
  }

  rect(x, y, w, h) {
    this.current.push(`${num(x)} ${num(this.height - y - h)} ${num(w)} ${num(h)} re S`);
  }

  save(filename) {
    const bytes = buildPDF(this.width, this.height, this.pages);
    const blob = new Blob([bytes], { type: 'application/pdf' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
}

export function decimalRight(doc, commaX, { size = 10 } = {}) {
  return commaX + doc.textWidth(',0', { size });
}

export function buildPDF(width, height, pages) {
  const objects = [];
  objects.push('<< /Type /Catalog /Pages 2 0 R >>');
  const kids = pages.map((_, i) => `${5 + i * 2} 0 R`).join(' ');
  objects.push(`<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`);
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>');
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>');

  for (let i = 0; i < pages.length; i++) {
    const pageObj = 5 + i * 2;
    const contentObj = pageObj + 1;
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${num(width)} ${num(height)}] ` +
      `/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents ${contentObj} 0 R >>`,
    );
    const stream = pages[i].join('\n');
    objects.push(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);
  }

  let pdf = '%PDF-1.4\n';
  const offsets = [0];
  for (let i = 0; i < objects.length; i++) {
    offsets.push(pdf.length);
    pdf += `${i + 1} 0 obj\n${objects[i]}\nendobj\n`;
  }
  const xref = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i < offsets.length; i++) {
    pdf += `${String(offsets[i]).padStart(10, '0')} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return pdf;
}

function pdfString(value) {
  let out = '(';
  for (const ch of safeText(value)) {
    if (ch === '\\' || ch === '(' || ch === ')') out += `\\${ch}`;
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else out += ch;
  }
  return `${out})`;
}

function safeText(value) {
  return String(value ?? '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[\u2013\u2014]/g, '-')
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\u20AC/g, 'EUR')
    .replace(/[^\x09\x0A\x0D\x20-\x7E]/g, '?');
}

function num(value) {
  return Number(value).toFixed(2).replace(/\.00$/, '');
}
