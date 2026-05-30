/**
 * Shared locale-aware number formatters.
 *
 * Used as sortable-table column formatters (signature: value → string)
 * and in summary/label rendering.  All accept null/empty gracefully.
 * The decimal separator follows the active locale (read once from
 * `<html lang>`); see CLAUDE.md §"Number formatting".
 */

const LOCALE = (typeof document !== 'undefined' && document.documentElement.lang)
  || 'it';

// Intl.NumberFormat instances aren't free to build; cache one per decimal
// count.  No digit grouping, to match the server's `floatformat` rendering.
const _nf = {};
function numberFormat(n) {
  return (_nf[n] ||= new Intl.NumberFormat(LOCALE, {
    minimumFractionDigits: n,
    maximumFractionDigits: n,
    useGrouping: false,
  }));
}

export function fmtDecimal(v, n) {
  if (v == null || v === '') return '';
  return typeof v === 'number' ? numberFormat(n).format(v) : v;
}

export const fmtDecimal1 = v => fmtDecimal(v, 1);
export const fmtDecimal2 = v => fmtDecimal(v, 2);
export const fmtDecimal3 = v => fmtDecimal(v, 3);

export function fmtDecimalBlankZero(v, n) {
  if (!v) return '';
  return fmtDecimal(v, n);
}

export const fmtDecimal1BlankZero = v => fmtDecimalBlankZero(v, 1);

export const fmtCoord = v => fmtDecimal(v, 5);

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
