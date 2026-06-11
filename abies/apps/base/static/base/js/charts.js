import * as S from './strings.js';
import { fmtDecimal1 } from './format.js';

export const CATEGORICAL_COLORS = [
  '#2e7d32', '#1565c0', '#e65100', '#6a1b9a', '#c62828',
  '#00838f', '#827717', '#4e342e', '#37474f', '#ad1457',
  '#558b2f', '#0277bd',
];

export const CHART_COLORS = CATEGORICAL_COLORS;

export function renderChart(canvas, chartData, existing, config) {
  if (!canvas) return existing || null;
  const includeLabels = config.labels !== false;
  if (existing) {
    if (includeLabels) existing.data.labels = chartData.labels;
    existing.data.datasets = chartData.datasets;
    updateScaleTitles(existing, config.scaleTitles?.(chartData) || {});
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
      animation: { duration: 300 },
      scales: config.scales?.(chartData) || {},
      plugins: {
        legend: { position: 'bottom' },
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
        title: axisTitle(d.yTitle),
      },
    }),
    scaleTitles: d => ({ y: d.yTitle }),
  });
}

function axisTitle(text) {
  return { display: Boolean(text), text: text || '' };
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
