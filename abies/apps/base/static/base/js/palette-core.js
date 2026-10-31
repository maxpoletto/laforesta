/* Pure categorical palette shared with the offline classic-script shell. */
(function (root) {
  'use strict';
  const CATEGORICAL_COLORS = Object.freeze([
    '#2e7d32', '#11bb1d', '#144b99', '#f899cd', '#d55e00', '#d508f0',
    '#6f8acf', '#e69f00', '#17becf', '#dddd1c', '#fd0303', '#7f7f7f',
  ]);
  const CHART_SERIES_COLOR_VARS = Object.freeze(CATEGORICAL_COLORS.map((_, i) => `--chart-series-${i + 1}`));
  function chartSeriesColor(index) {
    const i = ((index % CATEGORICAL_COLORS.length) + CATEGORICAL_COLORS.length) % CATEGORICAL_COLORS.length;
    const element = root.document?.documentElement;
    const css = element && typeof root.getComputedStyle === 'function'
      ? root.getComputedStyle(element).getPropertyValue(CHART_SERIES_COLOR_VARS[i]).trim() : '';
    return css || CATEGORICAL_COLORS[i];
  }
  function alphaUnique(names, locale) {
    return [...new Set((names || []).map(name => String(name || '').trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b, locale));
  }
  function speciesColorMap(names, allNames = names, locale = 'it') {
    const requested = alphaUnique(names, locale);
    const universe = alphaUnique(allNames, locale);
    const known = new Set(universe);
    const ordered = [...universe, ...requested.filter(name => !known.has(name))];
    const colors = new Map(ordered.map((name, i) => [name, chartSeriesColor(i)]));
    return new Map(requested.map(name => [name, colors.get(name)]));
  }
  root.AbiesPalette = Object.freeze({ CATEGORICAL_COLORS, CHART_SERIES_COLOR_VARS,
    chartSeriesColor, speciesColorMap, alphaUnique });
  if (typeof module !== 'undefined') module.exports = root.AbiesPalette;
})(globalThis);
