// Pure helpers for Bosco historical-production summaries.

import * as S from '../../base/js/strings.js';
import { ROWS } from '../../base/js/constants.js';
import { columnMap, toNumber } from '../../base/js/digests.js';

const TOTAL_COLOR = '#2f8f58';
const PRELIEVI_PATH = '/prelievi';

export function prelieviUrlForScope(scope = {}) {
  const params = new URLSearchParams();
  if (Number.isInteger(scope.regionId) && scope.regionId > 0) {
    params.set('c', String(scope.regionId));
  }
  if (Number.isInteger(scope.parcelId) && scope.parcelId > 0) {
    params.set('pa', String(scope.parcelId));
  }
  const query = params.toString();
  return query ? `${PRELIEVI_PATH}?${query}` : PRELIEVI_PATH;
}

export function productionRows(digest, scope) {
  if (!digest) return [];
  const c = columnMap(digest);
  const regionIdx = c[S.COL_REGION];
  const parcelIdx = c[S.COL_PARCEL];
  if (regionIdx == null || parcelIdx == null) return [];
  return digest[ROWS].filter(row => {
    if (row[regionIdx] !== scope.region) return false;
    return !scope.parcel || row[parcelIdx] === scope.parcel;
  });
}

export function productionYears(digest, scope) {
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
  const match = String(value || '').trim().match(/^(\d{4})/);
  return match ? match[1] : '';
}

export function aggregateProduction(digest, scope, opts = {}) {
  const c = columnMap(digest);
  const dateIdx = c[S.COL_DATE];
  const qIdx = c[S.COL_QUINTALS];
  const rows = productionRows(digest, scope);
  const areaHa = toNumber(opts.areaHa, 0);
  const divisor = opts.perHa && areaHa > 0 ? areaHa : 1;
  const bucket = opts.byMonth
    ? date => String(date || '').slice(0, 7)
    : date => String(date || '').slice(0, 4);
  const byBucket = new Map();
  let total = 0;

  for (const row of rows) {
    const date = row[dateIdx];
    const key = bucket(date);
    if (!key) continue;
    const q = toNumber(row[qIdx], 0);
    total += q;
    byBucket.set(key, (byBucket.get(key) || 0) + q / divisor);
  }

  const labels = [...byBucket.keys()].sort();
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

function round1(value) {
  return Math.round(value * 10) / 10;
}
