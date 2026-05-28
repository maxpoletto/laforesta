import * as S from './strings.js';

export function csvField(v, fmt) {
  if (v == null) return '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return String(v).replace('.', fmt.decimal);
  return String(v).replaceAll(fmt.separator, ' ');
}

export function downloadCSV(lines, filename) {
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/**
 * Trigger a download from a URL whose response sets Content-Disposition.
 * The browser handles the download without leaving the SPA.  `filename`
 * defaults to empty, deferring to the server's Content-Disposition.
 */
export function downloadFromURL(url, filename = '') {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/**
 * Export a digest to CSV.  `columns` is an array of column descriptors:
 *   - `'X'`                       — use digest column X, render as X
 *   - `{ src, dst }`              — use digest column src, render as dst
 *   - `{ dst, transform: row => v }` — value from transform; no src lookup
 *
 * `opts.filter(row) => bool` skips non-matching rows.
 */
export function exportDigest(digest, columns, filename, opts = {}) {
  const fmt = S.TABLE_CSV_FORMAT;
  const resolved = columns.map(c => {
    if (typeof c === 'string') {
      return { dst: c, idx: digest.columns.indexOf(c), transform: null };
    }
    const dst = c.dst ?? c.src;
    const idx = c.src != null ? digest.columns.indexOf(c.src) : -1;
    return { dst, idx, transform: c.transform ?? null };
  });
  const lines = [resolved.map(r => r.dst).join(fmt.separator)];
  for (const row of digest.rows) {
    if (opts.filter && !opts.filter(row)) continue;
    const parts = resolved.map(r =>
      csvField(r.transform ? r.transform(row) : row[r.idx], fmt),
    );
    lines.push(parts.join(fmt.separator));
  }
  downloadCSV(lines, filename);
}
