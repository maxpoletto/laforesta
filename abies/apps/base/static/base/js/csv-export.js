import * as S from './strings.js';

export function csvField(v, fmt) {
  if (v == null) return '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return String(v).replace('.', fmt.decimal);
  return String(v).replace(new RegExp(fmt.separator, 'g'), ' ');
}

export function downloadCSV(lines, filename) {
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function exportDigest(digest, exportCols, srcCols, filename, opts = {}) {
  const c = digest.columns;
  const idx = srcCols.map(s => c.indexOf(s));
  const fmt = S.TABLE_CSV_FORMAT;
  const lines = [exportCols.join(fmt.separator)];
  for (const row of digest.rows) {
    if (opts.filter && !opts.filter(row)) continue;
    const parts = idx.map((i, k) => {
      if (opts.transform) {
        const v = opts.transform(row, i, exportCols[k]);
        if (v !== undefined) return csvField(v, fmt);
      }
      return csvField(row[i], fmt);
    });
    lines.push(parts.join(fmt.separator));
  }
  downloadCSV(lines, filename);
}
