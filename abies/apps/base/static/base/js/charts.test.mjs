import * as S from './strings.js';
import {
  CATEGORICAL_COLORS, renderLineChart, renderScatterChart, renderStackedBar,
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

assertEqual(CATEGORICAL_COLORS.slice(0, 2), ['#2e7d32', '#1565c0'], 'shared palette');

const canvas = { id: 'chart' };
let chart = renderStackedBar(canvas, {
  labels: ['2026'],
  datasets: [{ label: 'Abete', data: [12.34] }],
}, null);
assertEqual(chart.type, 'bar', 'renderStackedBar: type');
assertEqual(chart.data.labels, ['2026'], 'renderStackedBar: labels');
assertEqual(chart.options.responsive, true, 'renderStackedBar: responsive');
assertEqual(chart.options.maintainAspectRatio, false, 'renderStackedBar: no aspect ratio');
assertEqual(chart.options.animation.duration, 300, 'renderStackedBar: animation duration');
assertEqual(chart.options.plugins.legend.position, 'bottom', 'renderStackedBar: legend position');
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
