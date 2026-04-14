/**
 * Prelievi chart aggregation and rendering.
 *
 * All functions operate on digest rows filtered by the caller.
 * Charts use Chart.js stacked bar configuration.
 */

import * as S from '../../base/js/strings.js';

const MAX_SERIES = 12;

const COLORS = [
  '#2e7d32', '#1565c0', '#e65100', '#6a1b9a', '#c62828',
  '#00838f', '#827717', '#4e342e', '#37474f', '#ad1457',
  '#558b2f', '#0277bd',
];

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/**
 * Aggregate production over time, optionally broken down by a dimension.
 *
 * @param {any[][]} rows — filtered digest rows
 * @param {Object} colMap — { columnName: columnIndex }
 * @param {string} breakdown — 'total'|'compresa'|'particella'|'squadra'|'tipo'|'specie'|'trattore'
 * @param {boolean} byMonth — year or month granularity
 * @param {string[]} speciesCols — species quintal column names
 * @param {string[]} tractorCols — tractor quintal column names
 * @returns {{ labels: string[], datasets: Array<{label, data, backgroundColor}> }}
 */
export function aggregateTimeSeries(rows, colMap, breakdown, byMonth, speciesCols, tractorCols) {
  const dateIdx = colMap['Data'];
  const qIdx = colMap['Q.li'];
  const bucket = byMonth ? d => d.substring(0, 7) : d => d.substring(0, 4);

  // Breakdowns that pivot on multiple numeric columns (species/tractors).
  const byColumns = { specie: speciesCols, trattore: tractorCols }[breakdown];
  if (byColumns) {
    return _aggregateColumnsByBucket(rows, dateIdx, colMap, bucket, byColumns);
  }

  const dimIdx = breakdown === 'total' ? null : colMap[{
    compresa: 'Compresa', particella: 'Particella',
    squadra: 'Squadra', tipo: 'Tipo',
  }[breakdown]];

  return _aggregateByBucket(rows, dateIdx, qIdx, bucket, dimIdx);
}

/**
 * Aggregate species by parcel (stacked bar: parcels on x, species as stacks).
 *
 * @param {any[][]} rows
 * @param {Object} colMap
 * @param {string[]} speciesCols
 * @returns {{ labels: string[], datasets: ... }}
 */
export function aggregateSpeciesByParcel(rows, colMap, speciesCols) {
  const parcelIdx = colMap['Particella'];
  const regionIdx = colMap['Compresa'];
  const parcels = {};

  for (const row of rows) {
    const key = row[regionIdx] + ' ' + row[parcelIdx];
    if (!parcels[key]) parcels[key] = {};
    for (const sp of speciesCols) {
      parcels[key][sp] = (parcels[key][sp] || 0) + (row[colMap[sp]] || 0);
    }
  }

  const entries = Object.entries(parcels)
    .map(([name, sps]) => ({ name, total: _sum(Object.values(sps)), sps }))
    .filter(p => p.total > 0)
    .sort((a, b) => b.total - a.total);

  const active = speciesCols.filter(sp =>
    entries.some(p => (p.sps[sp] || 0) > 0),
  );

  return {
    labels: entries.map(p => p.name),
    datasets: active.map((sp, i) => ({
      label: sp,
      data: entries.map(p => _r1(p.sps[sp] || 0)),
      backgroundColor: COLORS[i % COLORS.length],
    })),
  };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Create a new Chart.js stacked bar, or update an existing one in place.
 * Returns the Chart instance.
 */
export function renderStackedBar(canvas, chartData, existing) {
  if (existing) {
    existing.data.labels = chartData.labels;
    existing.data.datasets = chartData.datasets;
    existing.update('none');
    return existing;
  }

  return new window.Chart(canvas, {
    type: 'bar',
    data: { labels: chartData.labels, datasets: chartData.datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: { stacked: true },
        y: {
          stacked: true, beginAtZero: true,
          title: { display: true, text: S.COL_QUINTALS },
        },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          animation: false,
          callbacks: {
            label: ctx =>
              `${ctx.dataset.label}: ${ctx.raw.toFixed(1).replace('.', ',')}`,
          },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Aggregate rows by time bucket, with optional categorical dimension. */
function _aggregateByBucket(rows, dateIdx, qIdx, bucket, dimIdx) {
  const buckets = {};
  for (const row of rows) {
    const key = bucket(row[dateIdx]);
    const dim = dimIdx != null ? (row[dimIdx] || '?') : S.CHART_TOTAL;
    if (!buckets[key]) buckets[key] = {};
    buckets[key][dim] = (buckets[key][dim] || 0) + (row[qIdx] || 0);
  }

  const labels = Object.keys(buckets).sort();
  let dims = _topDims(buckets);

  if (dims.length > MAX_SERIES) {
    dims = _collapseOthers(buckets, dims);
  }

  return {
    labels,
    datasets: dims.map((dim, i) => ({
      label: dim,
      data: labels.map(k => _r1(buckets[k]?.[dim] || 0)),
      backgroundColor: COLORS[i % COLORS.length],
    })),
  };
}

/** Aggregate named numeric columns (e.g. species) by time bucket. */
function _aggregateColumnsByBucket(rows, dateIdx, colMap, bucket, colNames) {
  const buckets = {};
  for (const row of rows) {
    const key = bucket(row[dateIdx]);
    if (!buckets[key]) buckets[key] = {};
    for (const name of colNames) {
      buckets[key][name] = (buckets[key][name] || 0) + (row[colMap[name]] || 0);
    }
  }
  const labels = Object.keys(buckets).sort();
  const active = colNames.filter(n =>
    labels.some(k => (buckets[k]?.[n] || 0) > 0),
  );
  return {
    labels,
    datasets: active.map((name, i) => ({
      label: name,
      data: labels.map(k => _r1(buckets[k]?.[name] || 0)),
      backgroundColor: COLORS[i % COLORS.length],
    })),
  };
}

/** Rank dimension values by total volume descending. */
function _topDims(buckets) {
  const totals = {};
  for (const b of Object.values(buckets)) {
    for (const [dim, val] of Object.entries(b)) {
      totals[dim] = (totals[dim] || 0) + val;
    }
  }
  return Object.entries(totals).sort((a, b) => b[1] - a[1]).map(d => d[0]);
}

/** Collapse tail dimensions into "Altro". Mutates buckets in place. */
function _collapseOthers(buckets, dims) {
  const keep = new Set(dims.slice(0, MAX_SERIES - 1));
  for (const b of Object.values(buckets)) {
    let other = 0;
    for (const [dim, val] of Object.entries(b)) {
      if (!keep.has(dim)) { other += val; delete b[dim]; }
    }
    if (other > 0) b[S.CHART_OTHER] = other;
  }
  return [...dims.slice(0, MAX_SERIES - 1), S.CHART_OTHER];
}

function _r1(n) { return Math.round(n * 10) / 10; }
function _sum(arr) { return arr.reduce((a, b) => a + b, 0); }
