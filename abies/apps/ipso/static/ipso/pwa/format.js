// Locale-aware formatting helpers for the offline Ipso PWA.
//
// Mirrors apps/base/static/base/js/format.js, but exposed as a classic-script
// global because the PWA shell is intentionally not an ES module graph.
'use strict';

const IpsoFormat = (function() {
  const locale = (typeof document !== 'undefined' && document.documentElement.lang) || 'it';
  const coordFormat = new Intl.NumberFormat(locale, {
    minimumFractionDigits: 5,
    maximumFractionDigits: 5,
    useGrouping: false,
  });

  function fmtCoord(value) {
    if (value == null || value === '') return '';
    return typeof value === 'number' ? coordFormat.format(value) : value;
  }

  return { fmtCoord };
})();

if (typeof module !== 'undefined') module.exports = { IpsoFormat };
