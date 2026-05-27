/**
 * Shared number formatters for Italian locale display.
 *
 * Used as sortable-table column formatters (signature: value → string)
 * and in summary/label rendering.  All accept null/empty gracefully.
 */

export function fmtDecimal(v, n) {
  if (v == null || v === '') return '';
  return typeof v === 'number' ? v.toFixed(n).replace('.', ',') : v;
}

export const fmtDecimal1 = v => fmtDecimal(v, 1);
export const fmtDecimal2 = v => fmtDecimal(v, 2);
export const fmtDecimal3 = v => fmtDecimal(v, 3);

export function fmtDecimalBlankZero(v, n) {
  if (!v) return '';
  return fmtDecimal(v, n);
}

export const fmtDecimal1BlankZero = v => fmtDecimalBlankZero(v, 1);

export function fmtCoord(v) {
  if (v == null || v === '') return '';
  return typeof v === 'number' ? v.toFixed(5) : v;
}

export function fmtInt(v) {
  return v == null || v === '' ? '' : String(v);
}

export function fmtBool(v) {
  return v ? '✓' : '';
}

export function fmtWithUnit(v, n, unit) {
  const s = fmtDecimal(v, n);
  return s ? `${s} ${unit}` : '';
}

export const fmtVolume = v => fmtWithUnit(v, 2, 'm³');
export const fmtArea   = v => fmtWithUnit(v, 2, 'ha');
export const fmtMass   = v => fmtWithUnit(v, 2, 'q');
