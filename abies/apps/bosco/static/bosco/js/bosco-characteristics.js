import * as S from '../../base/js/strings.js';
import { compareParcelNames } from '../../base/js/natural-sort.js';
import {
  COL_COPPICE, COL_PARCEL_ID, COL_REGION_ID, ROWS, ROW_ID,
} from '../../base/js/constants.js';
import { columnMap, toNumber } from '../../base/js/digests.js';
import {
  CHARACTERISTIC_METRICS,
  CHARACTERISTIC_METRIC_IDS,
  Q_AGE,
  Q_ALTITUDE,
  Q_AVG_GRADE,
  Q_FUTURE_HARVEST,
  Q_HISTORICAL_HARVEST,
  Q_TYPE,
  isHarvestMetric,
} from './bosco-metrics.js';

export {
  CHARACTERISTIC_METRICS,
  CHARACTERISTIC_METRIC_IDS,
  Q_AGE,
  Q_ALTITUDE,
  Q_AVG_GRADE,
  Q_FUTURE_HARVEST,
  Q_HISTORICAL_HARVEST,
  Q_TYPE,
  isHarvestMetric,
} from './bosco-metrics.js';

export function parcelKey(region, parcel) {
  return `${region}-${parcel}`;
}

export { compareParcelNames };

export function compareParcelEntries(a, b) {
  return String(a.region || '').localeCompare(String(b.region || ''), S.LOCALE)
    || compareParcelNames(a.parcel, b.parcel)
    || Number(a.id || 0) - Number(b.id || 0);
}

export function buildParcelEntries(digest) {
  const c = columnMap(digest);
  return digest[ROWS].map(row => {
    const altMin = toNumber(row[c[S.COL_ALT_MIN]]);
    const altMax = toNumber(row[c[S.COL_ALT_MAX]]);
    const type = row[c[S.COL_TYPE]];
    const coppiceIdx = c[COL_COPPICE];
    const coppice = coppiceIdx == null ? null : row[coppiceIdx] === true;
    return {
      id: row[c[ROW_ID]],
      regionId: row[c[COL_REGION_ID]],
      region: row[c[S.COL_REGION]],
      parcel: row[c[S.COL_PARCEL]],
      key: parcelKey(row[c[S.COL_REGION]], row[c[S.COL_PARCEL]]),
      type,
      coppice,
      className: row[c[S.COL_CLASS]],
      areaHa: toNumber(row[c[S.COL_AREA_HA]]),
      cadastralAreaHa: toNumber(row[c[S.COL_AREA_CAD_HA]]),
      aveAge: toNumber(row[c[S.COL_AVE_AGE]]),
      location: row[c[S.COL_LOCATION]] || '',
      altMin,
      altMax,
      aspect: row[c[S.COL_ASPECT]] || '',
      gradePct: toNumber(row[c[S.COL_GRADE_PCT]]),
      descVeg: row[c[S.COL_DESC_VEG]] || '',
      descGeo: row[c[S.COL_DESC_GEO]] || '',
      cuttingPlan: row[c[S.COL_CUTTING_PLAN]] || '',
      harvestMechanism: row[c[S.COL_HARVEST_MECHANISM]] || '',
      stumpsPerHa: toNumber(row[c[S.COL_STUMPS_PER_HA]]),
      interventionInterval: toNumber(row[c[S.COL_INTERVENTION_INTERVAL]]),
      standardsPerHa: toNumber(row[c[S.COL_STANDARDS_PER_HA]]),
      altitudeMean: altMin !== null && altMax !== null ? (altMin + altMax) / 2 : null,
    };
  }).sort(compareParcelEntries);
}

export function historicalHarvestByParcel(digest) {
  if (!digest) return new Map();
  const c = columnMap(digest);
  const out = new Map();
  for (const row of digest[ROWS]) {
    const key = parcelKey(row[c[S.COL_REGION]], row[c[S.COL_PARCEL]]);
    out.set(key, (out.get(key) || 0) + toNumber(row[c[S.COL_QUINTALS]], 0));
  }
  return out;
}

export function futureHarvestByParcel(digest) {
  if (!digest) return new Map();
  const c = columnMap(digest);
  const out = new Map();
  for (const row of digest[ROWS]) {
    const parcelId = row[c[COL_PARCEL_ID]];
    out.set(parcelId, (out.get(parcelId) || 0) + toNumber(row[c[S.COL_VOLUME_PLANNED]], 0));
  }
  return out;
}

export function metricValue(entry, metricId, { historical = new Map(), future = new Map(), perHa = false } = {}) {
  let value = null;
  if (metricId === Q_AGE) value = entry.aveAge;
  else if (metricId === Q_TYPE) return entry.type;
  else if (metricId === Q_ALTITUDE) value = entry.altitudeMean;
  else if (metricId === Q_AVG_GRADE) value = entry.gradePct;
  else if (metricId === Q_HISTORICAL_HARVEST) value = historical.get(entry.key) || 0;
  else if (metricId === Q_FUTURE_HARVEST) value = future.get(entry.id) || 0;
  if (value == null) return null;
  if (perHa && isHarvestMetric(metricId)) {
    const area = perHaArea(entry);
    if (!area) return null;
    return value / area;
  }
  return value;
}

/** Hectare denominator for a parcel entry's per-hectare metrics. */
export function perHaArea(entry) {
  return entry.displayAreaHa || entry.areaHa || entry.cadastralAreaHa;
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
