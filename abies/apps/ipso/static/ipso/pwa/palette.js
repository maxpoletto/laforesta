// Offline Ipso palette helpers.
//
// Uses the shared pure palette core while retaining mobile marker styling.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined') {
  require('../../../../base/static/base/js/palette-core.js');
}

const IpsoPalette = (function() {
  const PAI_MARKER_BORDER_COLOR = '#222';
  const PAI_MARKER_FALLBACK_COLOR = '#777';

  const chartSeriesColor = index => globalThis.AbiesPalette.chartSeriesColor(index);
  function speciesColorMap(names, allNames) {
    return globalThis.AbiesPalette.speciesColorMap(names, allNames, locale());
  }

  function speciesColorById(speciesRows) {
    const items = (speciesRows || [])
      .map(row => ({ id: row && row.id, name: row && row.common }))
      .filter(item => item.id !== undefined && String(item.name || '').trim());
    const colorByName = speciesColorMap(
      items.map(item => item.name),
      items.map(item => item.name)
    );
    return new Map(items.map((item, idx) => [
      item.id,
      colorByName.get(String(item.name || '').trim()) || chartSeriesColor(idx),
    ]));
  }

  function paiMarkerStyle(speciesId, speciesColors) {
    const colors = speciesColors || new Map();
    return {
      radius: 6,
      color: PAI_MARKER_BORDER_COLOR,
      weight: 1,
      opacity: 0.9,
      fillColor: colors.get(speciesId) || PAI_MARKER_FALLBACK_COLOR,
      fillOpacity: 0.88,
      bubblingMouseEvents: false,
    };
  }

  function locale() {
    return (globalThis.document && globalThis.document.documentElement.lang) || 'it';
  }

  return { chartSeriesColor, speciesColorMap, speciesColorById, paiMarkerStyle };
})();

if (typeof module !== 'undefined') module.exports = { IpsoPalette };
