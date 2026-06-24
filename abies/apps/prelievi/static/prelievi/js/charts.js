/**
 * Prelievi chart aggregation.
 *
 * All functions operate on digest rows filtered by the caller.
 * Rendering lives in base/js/charts.js.
 */

import * as S from '../../base/js/strings.js';
import {
  chartSeriesColor, continuousTimeBuckets, monthBucket,
  speciesColorMap as chartSpeciesColorMap, yearBucket,
} from '../../base/js/charts.js';

const MAX_SERIES = 12;

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
 * @param {string[]} allSpeciesNames — full species universe for stable colors
 * @returns {{ labels: string[], datasets: Array<{label, data, backgroundColor}> }}
 */
export function aggregateTimeSeries(
  rows, colMap, breakdown, byMonth, speciesCols, tractorCols, allSpeciesNames = speciesCols,
) {
  const dateIdx = colMap[S.COL_DATE];
  const qIdx = colMap[S.COL_QUINTALS];
  const bucket = byMonth ? monthBucket : yearBucket;

  // Breakdowns that pivot on multiple numeric columns (species/tractors).
  const byColumns = { specie: speciesCols, trattore: tractorCols }[breakdown];
  if (byColumns) {
    const speciesUniverse = breakdown === 'specie' ? allSpeciesNames : null;
    return _aggregateColumnsByBucket(
      rows, dateIdx, colMap, bucket, byMonth, byColumns, speciesUniverse,
    );
  }

  return _aggregateByBucket(rows, dateIdx, qIdx, bucket, byMonth, _dimFn(breakdown, colMap));
}

/** Build a row → category-name function for the single-column breakdowns. */
function _dimFn(breakdown, colMap) {
  if (breakdown === 'total') return () => S.CHART_TOTAL;
  if (breakdown === 'particella') {
    const ri = colMap[S.COL_REGION];
    const pi = colMap[S.COL_PARCEL];
    return row => `${row[ri]}/${row[pi]}`;  // disambiguate parcels across regions
  }
  const idx = colMap[{
    compresa: S.COL_REGION, squadra: S.COL_CREW, tipo: S.COL_TYPE,
  }[breakdown]];
  return row => row[idx] || '?';
}

/**
 * Aggregate species by parcel (stacked bar: parcels on x, species as stacks).
 *
 * @param {any[][]} rows
 * @param {Object} colMap
 * @param {string[]} speciesCols
 * @returns {{ labels: string[], datasets: ... }}
 */
export function aggregateSpeciesByParcel(rows, colMap, speciesCols, allSpeciesNames = speciesCols) {
  const parcelIdx = colMap[S.COL_PARCEL];
  const regionIdx = colMap[S.COL_REGION];
  const parcels = {};

  for (const row of rows) {
    const key = `${row[regionIdx]}/${row[parcelIdx]}`;
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

  const colors = chartSpeciesColorMap(active, allSpeciesNames);
  return {
    labels: entries.map(p => p.name),
    datasets: [...colors.keys()].map((sp, i) => ({
      label: sp,
      data: entries.map(p => _r1(p.sps[sp] || 0)),
      backgroundColor: colors.get(sp) || chartSeriesColor(i),
    })),
  };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Aggregate rows by time bucket, with a row → category function. */
function _aggregateByBucket(rows, dateIdx, qIdx, bucket, byMonth, dimFn) {
  const buckets = {};
  for (const row of rows) {
    const key = bucket(row[dateIdx]);
    if (!key) continue;
    const dim = dimFn(row);
    if (!buckets[key]) buckets[key] = {};
    buckets[key][dim] = (buckets[key][dim] || 0) + (row[qIdx] || 0);
  }

  const labels = continuousTimeBuckets(Object.keys(buckets), byMonth);
  let dims = _topDims(buckets);

  if (dims.length > MAX_SERIES) {
    dims = _collapseOthers(buckets, dims);
  }

  return {
    labels,
    datasets: dims.map((dim, i) => ({
      label: dim,
      data: labels.map(k => _r1(buckets[k]?.[dim] || 0)),
      backgroundColor: chartSeriesColor(i),
    })),
  };
}

/** Aggregate named numeric columns (e.g. species) by time bucket. */
function _aggregateColumnsByBucket(
  rows, dateIdx, colMap, bucket, byMonth, colNames, speciesUniverse = null,
) {
  const buckets = {};
  for (const row of rows) {
    const key = bucket(row[dateIdx]);
    if (!key) continue;
    if (!buckets[key]) buckets[key] = {};
    for (const name of colNames) {
      buckets[key][name] = (buckets[key][name] || 0) + (row[colMap[name]] || 0);
    }
  }
  const labels = continuousTimeBuckets(Object.keys(buckets), byMonth);
  const active = colNames.filter(n =>
    labels.some(k => (buckets[k]?.[n] || 0) > 0),
  );
  const colors = speciesUniverse ? chartSpeciesColorMap(active, speciesUniverse) : null;
  const datasetNames = colors ? [...colors.keys()] : active;
  return {
    labels,
    datasets: datasetNames.map((name, i) => ({
      label: name,
      data: labels.map(k => _r1(buckets[k]?.[name] || 0)),
      backgroundColor: colors?.get(name) || chartSeriesColor(i),
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
