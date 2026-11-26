// Pure helpers for Bosco historical-production summaries.

import * as S from '../../base/js/strings.js';
import { ROWS } from '../../base/js/constants.js';
import { continuousTimeBuckets, monthBucket, yearBucket } from '../../base/js/charts.js';
import { columnMap, toNumber } from '../../base/js/digests.js';
import { compareParcelNames } from '../../base/js/natural-sort.js';
import { harvestScopeSearchTerms } from '../../base/js/prelievi-columns.js';

const TOTAL_COLOR = '#2f8f58';
const PRELIEVI_PATH = '/prelievi';

export function prelieviUrlForScope(scope = {}) {
  const params = new URLSearchParams();
  const search = harvestScopeSearchTerms({region: scope.region, parcel: scope.entry?.parcel}).join(' ');
  if (search) params.set('f', search);
  const query = params.toString();
  return query ? `${PRELIEVI_PATH}?${query}` : PRELIEVI_PATH;
}

export function productionRows(digest, scope = {}) {
  if (!digest) return [];
  const c = columnMap(digest);
  const regionIdx = c[S.COL_REGION];
  const parcelIdx = c[S.COL_PARCEL];
  if (regionIdx == null || parcelIdx == null) return [];
  return digest[ROWS].filter(row => {
    if (scope.region && row[regionIdx] !== scope.region) return false;
    return !scope.parcel || row[parcelIdx] === scope.parcel;
  });
}

export function productionYears(digest, scope = {}) {
  const c = columnMap(digest);
  const dateIdx = c[S.COL_DATE];
  if (dateIdx == null) return [];
  const years = new Set();
  for (const row of productionRows(digest, scope)) {
    const year = harvestYear(row[dateIdx]);
    if (year) years.add(year);
  }
  return [...years].sort();
}

export function productionYearRange(digest, scope = {}) {
  return continuousTimeBuckets(productionYears(digest, scope), false);
}

export function productionMonths(digest, scope = {}) {
  const c = columnMap(digest);
  const dateIdx = c[S.COL_DATE];
  if (dateIdx == null) return [];
  const months = new Set();
  for (const row of productionRows(digest, scope)) {
    const month = harvestMonth(row[dateIdx]);
    if (month) months.add(month);
  }
  return [...months].sort();
}

export function productionMonthRange(digest, scope = {}) {
  return continuousTimeBuckets(productionMonths(digest, scope), true);
}

export function pickProductionYear(years, requested, fallback = 'latest') {
  const clean = [...new Set((years || []).map(harvestYear).filter(Boolean))].sort();
  if (!clean.length) return '';
  const requestedYear = harvestYear(requested);
  if (clean.includes(requestedYear)) return requestedYear;
  return fallback === 'earliest' ? clean[0] : clean[clean.length - 1];
}

export function productionDeltaByParcel(digest, scope, fromYear, toYear) {
  const from = productionByParcelYear(digest, scope, fromYear);
  const to = productionByParcelYear(digest, scope, toYear);
  const keys = new Set([...from.keys(), ...to.keys()]);
  const out = new Map();
  for (const key of keys) out.set(key, (to.get(key) || 0) - (from.get(key) || 0));
  return out;
}

function productionByParcelYear(digest, scope, year) {
  const c = columnMap(digest);
  const dateIdx = c[S.COL_DATE];
  const regionIdx = c[S.COL_REGION];
  const parcelIdx = c[S.COL_PARCEL];
  const qIdx = c[S.COL_QUINTALS];
  const targetYear = harvestYear(year);
  const out = new Map();
  if (!targetYear || dateIdx == null || regionIdx == null
      || parcelIdx == null || qIdx == null) return out;
  for (const row of productionRows(digest, scope)) {
    if (harvestYear(row[dateIdx]) !== targetYear) continue;
    const key = `${row[regionIdx]}-${row[parcelIdx]}`;
    out.set(key, (out.get(key) || 0) + toNumber(row[qIdx], 0));
  }
  return out;
}

export function harvestYear(value) {
  return yearBucket(value);
}

function harvestMonth(value) {
  return monthBucket(value);
}

export function aggregateProduction(digest, scope, opts = {}) {
  const c = columnMap(digest);
  return productionSeries(productionRows(digest, scope).map(row => ({
    date: row[c[S.COL_DATE]], mass: row[c[S.COL_QUINTALS]],
  })), opts);
}

function productionSeries(rows, opts) {
  const areaHa = toNumber(opts.areaHa, 0);
  const divisor = opts.perHa && areaHa > 0 ? areaHa : 1;
  const bucket = opts.byMonth ? harvestMonth : harvestYear;
  const byBucket = new Map();
  let total = 0;

  for (const { date, mass } of rows) {
    const key = bucket(date);
    if (!key) continue;
    const q = toNumber(mass, 0);
    total += q;
    byBucket.set(key, (byBucket.get(key) || 0) + q / divisor);
  }

  const labels = byBucket.size
    ? continuousTimeBuckets(opts.bounds || [...byBucket.keys()], opts.byMonth)
    : [];
  return {
    rowCount: rows.length,
    totalQuintals: total,
    labels,
    chartData: {
      labels,
      yTitle: opts.perHa && areaHa > 0 ? S.BOSCO_QUINTALS_PER_HA : S.COL_QUINTALS,
      datasets: [{
        label: opts.perHa && areaHa > 0 ? S.BOSCO_QUINTALS_PER_HA : S.COL_QUINTALS,
        data: labels.map(label => round1(byBucket.get(label) || 0)),
        backgroundColor: TOTAL_COLOR,
      }],
    },
  };
}

export function parcelMatchesGovernance(governance, coppice) {
  if (governance === 'coppice') return coppice;
  if (governance === 'highforest') return !coppice;
  return true;
}

export function regionalProductionYears(rows) {
  return continuousTimeBuckets(rows.map(row => row.month)).map(Number);
}

function inYearRange(year, { from, to }) {
  return (from == null || year >= from) && (to == null || year <= to);
}

export function regionalProductionChart(rows, opts) {
  const { governance, from, to, byMonth } = opts;
  const selection = rows.filter(row => parcelMatchesGovernance(governance, row.coppice)
    && inYearRange(Number(yearBucket(row.month)), opts));
  const bounds = from == null || to == null ? null : byMonth
    ? [`${from}-01`, `${to}-12`]
    : [from, to];
  return productionSeries(selection.map(row => ({ date: row.month, mass: row.mass_q })), {
    ...opts, bounds,
  });
}

export function regionalYield(rows, opts = {}) {
  const cuts = new Map();
  for (const row of rows) {
    if (!parcelMatchesGovernance(opts.governance, row.coppice)) continue;
    const year = Number(yearBucket(row.month));
    const key = row.worksite_id == null ? `${row.parcel_id}:year:${year}`
      : `${row.parcel_id}:worksite:${row.worksite_id}`;
    if (!cuts.has(key)) cuts.set(key, { row, year, lastYear: year, volumeM3: 0 });
    const cut = cuts.get(key);
    cut.year = Math.min(cut.year, year);
    cut.lastYear = Math.max(cut.lastYear, year);
    cut.volumeM3 += row.volume_m3;
  }

  // A linked cut belongs to its first harvest year, including the volume from
  // later years. Apply the year filter after all its monthly rows are grouped.
  const parcelYears = new Map();
  for (const { row, year, lastYear, volumeM3 } of cuts.values()) {
    if (!inYearRange(year, opts)) continue;
    const key = `${row.parcel_id}:${year}`;
    if (!parcelYears.has(key)) parcelYears.set(key, {
      parcelId: row.parcel_id, parcel: row.parcel, coppice: row.coppice, year, lastYear,
      parcelAreaHa: row.parcel_area_ha, interventionAreaHa: 0, fullParcel: false, volumeM3: 0,
    });
    const entry = parcelYears.get(key);
    const area = row.coppice ? row.intervention_area_ha : null;
    if (area > 0) entry.interventionAreaHa += area;
    else entry.fullParcel = true;
    entry.lastYear = Math.max(entry.lastYear, lastYear);
    entry.volumeM3 += volumeM3;
  }

  const detail = [...parcelYears.values()].map(entry => {
    const area = entry.fullParcel ? entry.parcelAreaHa : entry.interventionAreaHa;
    const { parcelId, parcel, coppice, year, lastYear, volumeM3 } = entry;
    return {
      parcelId, parcel, coppice, year, lastYear, volumeM3: Number(volumeM3.toFixed(3)),
      areaHa: area > 0 ? Number(area.toFixed(2)) : null,
      areaSource: area > 0 ? (entry.fullParcel ? 'parcel' : 'intervention') : null,
    };
  }).sort((a, b) => a.year - b.year || compareParcelNames(a.parcel, b.parcel));
  const measured = detail.filter(entry => entry.areaHa != null);
  return {
    rows: detail,
    volumeM3: measured.reduce((sum, row) => sum + row.volumeM3, 0),
    areaHa: measured.reduce((sum, row) => sum + row.areaHa, 0),
    parcelCount: new Set(measured.map(row => row.parcelId)).size,
    yearCount: new Set(measured.map(row => row.year)).size,
    unknownAreaCount: detail.length - measured.length,
  };
}

function round1(value) {
  return Math.round(value * 10) / 10;
}
