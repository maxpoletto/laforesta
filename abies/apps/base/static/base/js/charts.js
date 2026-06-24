import * as S from './strings.js';
import { COLUMNS, ROWS } from './constants.js';
import { fmtDecimal1 } from './format.js';

export const CHART_SERIES_COLOR_VARS = [
  '--chart-series-1', '--chart-series-2', '--chart-series-3',
  '--chart-series-4', '--chart-series-5', '--chart-series-6',
  '--chart-series-7', '--chart-series-8', '--chart-series-9',
  '--chart-series-10', '--chart-series-11', '--chart-series-12',
];

export const CATEGORICAL_COLORS = [
  '#2e7d32', '#11bb1d', '#144b99', '#f899cd', '#d55e00', '#d508f0',
  '#6f8acf', '#e69f00', '#17becf', '#dddd1c', '#fd0303', '#7f7f7f',
];

export function chartSeriesColor(index) {
  const i = mod(index, CHART_SERIES_COLOR_VARS.length);
  return cssCustomProperty(CHART_SERIES_COLOR_VARS[i]) || CATEGORICAL_COLORS[i];
}

export function speciesNamesFromDigest(digest) {
  const columns = digest?.[COLUMNS];
  const rows = digest?.[ROWS];
  if (!Array.isArray(columns) || !Array.isArray(rows)) return [];
  const nameIdx = columns.indexOf(S.COL_NAME);
  if (nameIdx < 0) return [];
  return alphaUnique(rows.map(row => row?.[nameIdx]));
}

export function speciesColorMap(speciesNames, allSpeciesNames = speciesNames) {
  const requested = alphaUnique(speciesNames);
  const universe = alphaUnique(allSpeciesNames);
  const universeSet = new Set(universe);
  const ordered = [...universe, ...requested.filter(name => !universeSet.has(name))];
  const colorByName = new Map(ordered.map((name, idx) => [name, chartSeriesColor(idx)]));
  return new Map(requested.map(name => [name, colorByName.get(name)]));
}

export function yearBucket(value) {
  const match = String(value || '').trim().match(/^(\d{4})/);
  return match ? match[1] : '';
}

export function monthBucket(value) {
  const match = String(value || '').trim().match(/^(\d{4})-(\d{2})/);
  if (!match) return '';
  const month = Number(match[2]);
  return month >= 1 && month <= 12 ? `${match[1]}-${match[2]}` : '';
}

export function continuousTimeBuckets(values, byMonth = false) {
  return byMonth ? continuousMonthBuckets(values) : continuousYearBuckets(values);
}

export function continuousYearBuckets(values) {
  const years = [...new Set((values || []).map(yearBucket).filter(Boolean))].sort();
  if (!years.length) return [];
  const start = Number(years[0]);
  const end = Number(years[years.length - 1]);
  const out = [];
  for (let year = start; year <= end; year++) out.push(String(year));
  return out;
}

export function continuousMonthBuckets(values) {
  const months = [...new Set((values || []).map(monthBucket).filter(Boolean))].sort();
  if (!months.length) return [];
  const start = monthIndex(months[0]);
  const end = monthIndex(months[months.length - 1]);
  const out = [];
  for (let idx = start; idx <= end; idx++) out.push(monthLabel(idx));
  return out;
}

export function renderChart(canvas, chartData, existing, config) {
  if (!canvas) return existing || null;
  const includeLabels = config.labels !== false;
  if (existing) {
    if (includeLabels) existing.data.labels = chartData.labels;
    existing.data.datasets = chartData.datasets;
    updateScaleTitles(existing, config.scaleTitles?.(chartData) || {});
    updateLegend(existing, chartData);
    existing.update('none');
    return existing;
  }

  const data = includeLabels
    ? { labels: chartData.labels, datasets: chartData.datasets }
    : { datasets: chartData.datasets };
  return new window.Chart(canvas, {
    type: config.type,
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: config.scales?.(chartData) || {},
      plugins: {
        legend: { position: 'bottom', display: chartData.legend !== false },
        ...(config.tooltipCallbacks ? { tooltip: { animation: false, callbacks: config.tooltipCallbacks } } : {}),
        ...(config.plugins || {}),
      },
    },
  });
}

export function renderStackedBar(canvas, chartData, existing) {
  const data = { ...chartData, yTitle: chartData.yTitle || S.COL_QUINTALS };
  return renderChart(canvas, data, existing, {
    type: 'bar',
    scales: d => ({
      x: { stacked: true },
      y: {
        stacked: true,
        beginAtZero: true,
        min: 0,
        title: axisTitle(d.yTitle),
      },
    }),
    scaleTitles: d => ({ y: d.yTitle }),
    tooltipCallbacks: {
      label: ctx => `${ctx.dataset.label}: ${fmtDecimal1(ctx.raw)}`,
    },
  });
}

export function renderScatterChart(canvas, chartData, existing) {
  return renderChart(canvas, chartData, existing, {
    type: 'scatter',
    labels: false,
    scales: d => ({
      x: {
        beginAtZero: true,
        title: axisTitle(d.xTitle),
      },
      y: {
        beginAtZero: true,
        min: 0,
        title: axisTitle(d.yTitle),
      },
    }),
    scaleTitles: d => ({ x: d.xTitle, y: d.yTitle }),
  });
}

export function renderLineChart(canvas, chartData, existing) {
  return renderChart(canvas, chartData, existing, {
    type: 'line',
    scales: d => ({
      y: {
        beginAtZero: true,
        min: 0,
        title: axisTitle(d.yTitle),
      },
    }),
    scaleTitles: d => ({ y: d.yTitle }),
  });
}

function mod(n, base) {
  return ((n % base) + base) % base;
}

function monthIndex(month) {
  const [year, value] = month.split('-').map(Number);
  return year * 12 + value - 1;
}

function monthLabel(idx) {
  const year = Math.floor(idx / 12);
  const month = String((idx % 12) + 1).padStart(2, '0');
  return `${year}-${month}`;
}

function alphaUnique(names) {
  return [...new Set((names || [])
    .map(name => String(name || '').trim())
    .filter(Boolean))]
    .sort((a, b) => a.localeCompare(b, S.LOCALE));
}

function cssCustomProperty(name) {
  const root = globalThis.document?.documentElement;
  const getter = globalThis.getComputedStyle;
  if (!root || typeof getter !== 'function') return '';
  return getter(root).getPropertyValue(name).trim();
}

function axisTitle(text) {
  return { display: Boolean(text), text: text || '' };
}

function updateLegend(chart, chartData) {
  const legend = chart.options?.plugins?.legend;
  if (legend) legend.display = chartData.legend !== false;
}

function updateScaleTitles(chart, titles) {
  for (const [scaleId, text] of Object.entries(titles)) {
    const title = chart.options?.scales?.[scaleId]?.title;
    if (title) {
      title.display = Boolean(text);
      title.text = text || '';
    }
  }
}
