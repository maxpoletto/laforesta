import * as S from './strings.js';
import './palette-core.js';
import { COLUMNS, ROWS } from './constants.js';
import { fmtDecimal1 } from './format.js';

export const CHART_SERIES_COLOR_VARS = globalThis.AbiesPalette.CHART_SERIES_COLOR_VARS;
export const CATEGORICAL_COLORS = globalThis.AbiesPalette.CATEGORICAL_COLORS;
export const chartSeriesColor = globalThis.AbiesPalette.chartSeriesColor;

export function speciesNamesFromDigest(digest) {
  const columns = digest?.[COLUMNS];
  const rows = digest?.[ROWS];
  if (!Array.isArray(columns) || !Array.isArray(rows)) return [];
  const nameIdx = columns.indexOf(S.COL_NAME);
  if (nameIdx < 0) return [];
  return alphaUnique(rows.map(row => row?.[nameIdx]));
}

export function speciesColorMap(speciesNames, allSpeciesNames = speciesNames) {
  return globalThis.AbiesPalette.speciesColorMap(speciesNames, allSpeciesNames, S.LOCALE);
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
  return globalThis.AbiesPalette.alphaUnique(names, S.LOCALE);
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
