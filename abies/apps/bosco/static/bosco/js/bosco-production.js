// Pure helpers for Bosco historical-production summaries.

import * as S from '../../base/js/strings.js';
import { COLUMNS, ROWS } from '../../base/js/constants.js';

const TOTAL_COLOR = '#2f8f58';

function colMap(digest) {
  const out = {};
  digest[COLUMNS].forEach((name, idx) => { out[name] = idx; });
  return out;
}

function num(value) {
  if (value == null || value === '') return 0;
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

export function productionRows(digest, scope) {
  if (!digest) return [];
  const c = colMap(digest);
  const regionIdx = c[S.COL_REGION];
  const parcelIdx = c[S.COL_PARCEL];
  if (regionIdx == null || parcelIdx == null) return [];
  return digest[ROWS].filter(row => {
    if (row[regionIdx] !== scope.region) return false;
    return !scope.parcel || row[parcelIdx] === scope.parcel;
  });
}

export function aggregateProduction(digest, scope, opts = {}) {
  const c = colMap(digest);
  const dateIdx = c[S.COL_DATE];
  const qIdx = c[S.COL_QUINTALS];
  const rows = productionRows(digest, scope);
  const areaHa = num(opts.areaHa);
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
    const q = num(row[qIdx]);
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
      yTitle: opts.perHa && areaHa > 0 ? 'Q.li/ha' : S.COL_QUINTALS,
      datasets: [{
        label: opts.perHa && areaHa > 0 ? 'Q.li/ha' : S.COL_QUINTALS,
        data: labels.map(label => round1(byBucket.get(label) || 0)),
        backgroundColor: TOTAL_COLOR,
      }],
    },
  };
}

function round1(value) {
  return Math.round(value * 10) / 10;
}
