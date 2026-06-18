// Offline Ipso palette helpers.
//
// Mirrors apps/base/static/base/js/charts.js for the cached classic-script PWA.
'use strict';

const IpsoPalette = (function() {
  const CHART_SERIES_COLOR_VARS = [
    '--chart-series-1', '--chart-series-2', '--chart-series-3',
    '--chart-series-4', '--chart-series-5', '--chart-series-6',
    '--chart-series-7', '--chart-series-8', '--chart-series-9',
    '--chart-series-10', '--chart-series-11', '--chart-series-12',
  ];
  const CATEGORICAL_COLORS = [
    '#2e7d32', '#11bb1d', '#144b99', '#f899cd', '#d55e00', '#d508f0',
    '#6f8acf', '#e69f00', '#17becf', '#dddd1c', '#fd0303', '#7f7f7f',
  ];
  const PAI_MARKER_BORDER_COLOR = '#222';
  const PAI_MARKER_FALLBACK_COLOR = '#777';

  function chartSeriesColor(index) {
    const i = mod(index, CHART_SERIES_COLOR_VARS.length);
    return cssCustomProperty(CHART_SERIES_COLOR_VARS[i]) || CATEGORICAL_COLORS[i];
  }

  function speciesColorMap(speciesNames, allSpeciesNames) {
    const requested = alphaUnique(speciesNames);
    const universe = alphaUnique(allSpeciesNames || speciesNames);
    const universeSet = new Set(universe);
    const ordered = [...universe, ...requested.filter(name => !universeSet.has(name))];
    const colorByName = new Map(ordered.map((name, idx) => [name, chartSeriesColor(idx)]));
    return new Map(requested.map(name => [name, colorByName.get(name)]));
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

  function alphaUnique(names) {
    return [...new Set((names || [])
      .map(name => String(name || '').trim())
      .filter(Boolean))]
      .sort((a, b) => a.localeCompare(b, locale()));
  }

  function cssCustomProperty(name) {
    const root = globalThis.document && globalThis.document.documentElement;
    const getter = globalThis.getComputedStyle;
    if (!root || typeof getter !== 'function') return '';
    return getter(root).getPropertyValue(name).trim();
  }

  function locale() {
    return (globalThis.document && globalThis.document.documentElement.lang) || 'it';
  }

  function mod(n, base) {
    return ((n % base) + base) % base;
  }

  return { chartSeriesColor, speciesColorMap, speciesColorById, paiMarkerStyle };
})();

if (typeof module !== 'undefined') module.exports = { IpsoPalette };
