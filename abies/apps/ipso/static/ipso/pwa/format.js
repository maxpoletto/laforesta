// Locale-aware formatting helpers for the offline Ipso PWA.
//
// Mirrors apps/base/static/base/js/format.js, but exposed as a classic-script
// global because the PWA shell is intentionally not an ES module graph.
'use strict';

const IpsoFormat = (function() {
  const locale = (typeof document !== 'undefined' && document.documentElement.lang) || 'it';
  const formats = new Map();

  function numberFormat(decimals) {
    if (!formats.has(decimals)) {
      formats.set(decimals, new Intl.NumberFormat(locale, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
        useGrouping: false,
      }));
    }
    return formats.get(decimals);
  }

  function fmtDecimal(value, decimals) {
    if (value == null || value === '') return '';
    const number = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(number)
      ? numberFormat(decimals).format(number)
      : String(value);
  }

  function fmtInt(value) {
    return fmtDecimal(value, 0);
  }

  function fmtDecimal2(value) {
    return fmtDecimal(value, 2);
  }

  function fmtCoord(value) {
    return fmtDecimal(value, 5);
  }

  return { fmtDecimal, fmtInt, fmtDecimal2, fmtCoord };
})();

if (typeof module !== 'undefined') module.exports = { IpsoFormat };
