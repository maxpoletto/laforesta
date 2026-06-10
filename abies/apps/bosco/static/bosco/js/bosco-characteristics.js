import * as S from '../../base/js/strings.js';
import { COLUMNS, ROWS, ROW_ID } from '../../base/js/constants.js';

export const Q_AGE = '1';
export const Q_TYPE = '2';
export const Q_ALTITUDE = '3';
export const Q_HISTORICAL_HARVEST = '4';
export const Q_FUTURE_HARVEST = '5';

export const CHARACTERISTIC_METRICS = {
  [Q_AGE]: { kind: 'continuous', unit: 'a' },
  [Q_TYPE]: { kind: 'type' },
  [Q_ALTITUDE]: { kind: 'continuous', unit: 'm' },
  [Q_HISTORICAL_HARVEST]: { kind: 'continuous', unit: 'q', harvest: true },
  [Q_FUTURE_HARVEST]: { kind: 'continuous', unit: 'm³', harvest: true },
};

function colMap(digest) {
  const out = {};
  digest[COLUMNS].forEach((name, idx) => { out[name] = idx; });
  return out;
}

function num(v) {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function parcelKey(region, parcel) {
  return `${region}-${parcel}`;
}

export function buildParcelEntries(digest) {
  const c = colMap(digest);
  return digest[ROWS].map(row => {
    const altMin = num(row[c[S.COL_ALT_MIN]]);
    const altMax = num(row[c[S.COL_ALT_MAX]]);
    return {
      id: row[c[ROW_ID]],
      regionId: row[c[S.COL_REGION_ID]],
      region: row[c[S.COL_REGION]],
      parcel: row[c[S.COL_PARCEL]],
      key: parcelKey(row[c[S.COL_REGION]], row[c[S.COL_PARCEL]]),
      type: row[c[S.COL_TYPE]],
      className: row[c[S.COL_CLASS]],
      areaHa: num(row[c[S.COL_AREA_HA]]),
      cadastralAreaHa: num(row[c[S.COL_AREA_CAD_HA]]),
      aveAge: num(row[c[S.COL_AVE_AGE]]),
      altitudeMean: altMin !== null && altMax !== null ? (altMin + altMax) / 2 : null,
    };
  });
}

export function historicalHarvestByParcel(digest) {
  if (!digest) return new Map();
  const c = colMap(digest);
  const out = new Map();
  for (const row of digest[ROWS]) {
    const key = parcelKey(row[c[S.COL_REGION]], row[c[S.COL_PARCEL]]);
    out.set(key, (out.get(key) || 0) + (num(row[c[S.COL_QUINTALS]]) || 0));
  }
  return out;
}

export function futureHarvestByParcel(digest) {
  if (!digest) return new Map();
  const c = colMap(digest);
  const out = new Map();
  for (const row of digest[ROWS]) {
    const parcelId = row[c[S.COL_PARCEL_ID]];
    out.set(parcelId, (out.get(parcelId) || 0) + (num(row[c[S.COL_VOLUME_PLANNED]]) || 0));
  }
  return out;
}

export function metricValue(entry, metricId, { historical = new Map(), future = new Map(), perHa = false } = {}) {
  let value = null;
  if (metricId === Q_AGE) value = entry.aveAge;
  else if (metricId === Q_TYPE) return entry.type;
  else if (metricId === Q_ALTITUDE) value = entry.altitudeMean;
  else if (metricId === Q_HISTORICAL_HARVEST) value = historical.get(entry.key) || 0;
  else if (metricId === Q_FUTURE_HARVEST) value = future.get(entry.id) || 0;
  if (value == null) return null;
  if (perHa && CHARACTERISTIC_METRICS[metricId]?.harvest) {
    const area = entry.displayAreaHa || entry.areaHa || entry.cadastralAreaHa;
    if (!area) return null;
    return value / area;
  }
  return value;
}

export function continuousDomain(values) {
  const clean = values.filter(v => v != null && Number.isFinite(v));
  if (!clean.length) return null;
  return { min: Math.min(...clean), max: Math.max(...clean) };
}

export function normalized(value, domain) {
  if (!domain || value == null || !Number.isFinite(value)) return null;
  const range = domain.max - domain.min;
  return range === 0 ? 0.5 : (value - domain.min) / range;
}
