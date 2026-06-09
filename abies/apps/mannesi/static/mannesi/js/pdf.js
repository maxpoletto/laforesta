/** Minimal PDF writer for Mannesi-generated operational PDFs. */

const A4 = { width: 595.28, height: 841.89 };

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
    this.current.push(`BT /${font} ${num(size)} Tf ${num(x)} ${num(this.height - y)} Td <${utf16Hex(value)}> Tj ET`);
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

export function buildPDF(width, height, pages) {
  const objects = [];
  objects.push('<< /Type /Catalog /Pages 2 0 R >>');
  const kids = pages.map((_, i) => `${5 + i * 2} 0 R`).join(' ');
  objects.push(`<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`);
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>');

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

function utf16Hex(value) {
  let out = 'FEFF';
  for (const ch of String(value ?? '')) {
    const code = ch.codePointAt(0);
    const safe = code <= 0xffff ? code : 0xfffd;
    out += safe.toString(16).padStart(4, '0').toUpperCase();
  }
  return out;
}

function num(value) {
  return Number(value).toFixed(2).replace(/\.00$/, '');
}
