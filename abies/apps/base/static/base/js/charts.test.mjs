import * as S from './strings.js';
import {
  CATEGORICAL_COLORS, CHART_SERIES_COLOR_VARS, chartSeriesColor,
  renderLineChart, renderScatterChart, renderStackedBar, speciesColorMap,
  speciesNamesFromDigest,
} from './charts.js';

let failed = 0;
let passed = 0;

class FakeChart {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.type = config.type;
    this.data = config.data;
    this.options = config.options;
    this.updateMode = null;
    FakeChart.created.push(this);
  }

  update(mode) {
    this.updateMode = mode;
  }
}
FakeChart.created = [];
global.window = { Chart: FakeChart };

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

console.log('charts.js');

assertEqual(CATEGORICAL_COLORS.slice(0, 2), ['#2e7d32', '#11bb1d'], 'shared palette fallback');
assertEqual(CHART_SERIES_COLOR_VARS.slice(0, 2), ['--chart-series-1', '--chart-series-2'],
            'shared palette CSS vars');
assertEqual(chartSeriesColor(0), '#2e7d32', 'chartSeriesColor: fallback color');
const speciesDigest = {
  columns: ['row_id', S.COL_NAME],
  rows: [[1, 'Faggio'], [2, 'Abete'], [3, 'Castagno'], [4, 'Abete']],
};
assertEqual(speciesNamesFromDigest(speciesDigest), ['Abete', 'Castagno', 'Faggio'],
            'speciesNamesFromDigest: alpha unique species names');
const speciesColors = speciesColorMap(['Faggio', 'Abete'], ['Abete', 'Castagno', 'Faggio']);
assertEqual([...speciesColors.entries()], [['Abete', chartSeriesColor(0)], ['Faggio', chartSeriesColor(2)]],
            'speciesColorMap: colors from full alpha species universe');
const unknownSpeciesColors = speciesColorMap(['Robinia'], ['Abete']);
assertEqual(unknownSpeciesColors.get('Robinia'), chartSeriesColor(1),
            'speciesColorMap: unknown species sort after known universe');
global.document = { documentElement: {} };
global.getComputedStyle = () => ({
  getPropertyValue: name => name === '--chart-series-1' ? ' #123456 ' : '',
});
assertEqual(chartSeriesColor(0), '#123456', 'chartSeriesColor: CSS custom property');
assertEqual(chartSeriesColor(1), '#11bb1d', 'chartSeriesColor: missing CSS var falls back');

const canvas = { id: 'chart' };
let chart = renderStackedBar(canvas, {
  labels: ['2026'],
  datasets: [{ label: 'Abete', data: [12.34] }],
}, null);
assertEqual(chart.type, 'bar', 'renderStackedBar: type');
assertEqual(chart.data.labels, ['2026'], 'renderStackedBar: labels');
assertEqual(chart.options.responsive, true, 'renderStackedBar: responsive');
assertEqual(chart.options.maintainAspectRatio, false, 'renderStackedBar: no aspect ratio');
assertEqual(chart.options.animation, false, 'renderStackedBar: no initial animation');
assertEqual(chart.options.plugins.legend.position, 'bottom', 'renderStackedBar: legend position');
assertEqual(chart.options.plugins.legend.display, true, 'renderStackedBar: legend visible by default');
assertEqual(chart.options.scales.x.stacked, true, 'renderStackedBar: stacked x');
assertEqual(chart.options.scales.y.title.text, S.COL_QUINTALS, 'renderStackedBar: default y title');
assertEqual(chart.options.plugins.tooltip.callbacks.label({
  dataset: { label: 'Abete' }, raw: 12.34,
}), 'Abete: 12,3', 'renderStackedBar: tooltip formatting');

const updated = renderStackedBar(canvas, {
  labels: ['2027'],
  yTitle: 'Volume',
  datasets: [{ label: 'Faggio', data: [7] }],
}, chart);
assertEqual(updated === chart, true, 'renderStackedBar: updates existing instance');
assertEqual(chart.updateMode, 'none', 'renderStackedBar: no-animation update');
assertEqual(chart.data.labels, ['2027'], 'renderStackedBar: updated labels');
assertEqual(chart.options.scales.y.title.text, 'Volume', 'renderStackedBar: updated y title');

renderStackedBar(canvas, {
  labels: ['2028'],
  legend: false,
  datasets: [{ label: 'Abete', data: [3] }],
}, chart);
assertEqual(chart.options.plugins.legend.display, false, 'renderStackedBar: can hide legend');

chart = renderScatterChart(canvas, {
  xTitle: S.COL_D_CM,
  yTitle: S.COL_H_M,
  datasets: [{ label: 'Abete', data: [{ x: 10, y: 12 }] }],
}, null);
assertEqual(chart.type, 'scatter', 'renderScatterChart: type');
assertEqual(Object.hasOwn(chart.data, 'labels'), false, 'renderScatterChart: no labels');
assertEqual(chart.options.scales.x.title.text, S.COL_D_CM, 'renderScatterChart: x title');
assertEqual(chart.options.scales.y.title.text, S.COL_H_M, 'renderScatterChart: y title');

chart = renderLineChart(canvas, {
  labels: ['20'],
  yTitle: S.COL_INCREMENT_PCT,
  datasets: [{ label: 'Abete', data: [1.2] }],
}, null);
assertEqual(chart.type, 'line', 'renderLineChart: type');
assertEqual(chart.options.scales.y.title.text, S.COL_INCREMENT_PCT, 'renderLineChart: y title');
assertEqual(renderLineChart(null, { datasets: [] }, null), null, 'renderLineChart: missing canvas');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
