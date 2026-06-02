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

/**
 * Build a locale-aware number parser — the inverse of `fmtDecimal`.  Maps the
 * locale's decimal separator to '.', so a literal '.' is always accepted (an
 * Italian user may type "3,14" or "3.14"); in a dot-decimal locale a comma is
 * not a decimal separator and yields NaN.  Thousands separators are out of
 * scope, so a value containing one is rejected (NaN) rather than regrouped.
 *
 * Returns `(value) => number`, yielding NaN for blank/garbage input.
 */
export function makeNumberParser(locale) {
  const sep = new Intl.NumberFormat(locale).formatToParts(1.1)
    .find(d => d.type === 'decimal').value;
  return (value) => {
    if (value == null || value === '') return NaN;
    let s = String(value).trim();
    if (sep !== '.') s = s.split(sep).join('.');
    return s === '' ? NaN : Number(s);
  };
}

const _parse = makeNumberParser(LOCALE);

/** Parse a locale-formatted number string to a Number (NaN if blank/invalid). */
export const parseDecimal = v => _parse(v);

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
