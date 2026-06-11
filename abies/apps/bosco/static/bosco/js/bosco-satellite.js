// Pure helpers for Bosco satellite-timeseries UI.

import { toNumber } from '../../base/js/digests.js';
import {
  CHARACTERISTIC_METRICS, E_EVI, E_HARVEST, E_NDMI, E_NDVI,
} from './bosco-metrics.js';

export const DEFAULT_EVOLUTION_METRIC = E_NDVI;
// Mirror apps/bosco/views.py SATELLITE_DIFF_VALUE_HEADER / SATELLITE_BYTE_MIDPOINT.
export const SATELLITE_DIFF_VALUE_HEADER = 'X-Bosco-Max-Abs';
export const BYTE_MIDPOINT = 127.5;

export const CHARACTERISTIC_SATELLITE_LAYERS = Object.fromEntries(
  Object.entries(CHARACTERISTIC_METRICS)
    .filter(([, metric]) => metric.kind === 'satellite')
    .map(([id, metric]) => [id, metric.layer]),
);

export const SATELLITE_LAYERS = {
  ndvi: { label: 'NDVI' },
  ndmi: { label: 'NDMI' },
  evi: { label: 'EVI' },
};

export const EVOLUTION_METRICS = {
  [E_NDVI]: { label: 'NDVI', layer: 'ndvi', satellite: true },
  [E_NDMI]: { label: 'NDMI', layer: 'ndmi', satellite: true },
  [E_EVI]: { label: 'EVI', layer: 'evi', satellite: true },
  [E_HARVEST]: { layer: 'prelievo', satellite: false },
};

export const EVOLUTION_METRIC_IDS = Object.keys(EVOLUTION_METRICS);

const INDEX_RAMP = [
  [0,   [139, 90, 43]],
  [128, [245, 235, 200]],
  [255, [0, 100, 0]],
];

// Mirror apps/bosco/views.py SATELLITE_DIFF_RAMP.
const DIFF_RAMP = [
  [0,   [180, 30, 30]],
  [128, [255, 255, 255]],
  [255, [30, 130, 30]],
];

export function evolutionMetricId(raw) {
  const value = String(raw || '');
  return EVOLUTION_METRICS[value] ? value : DEFAULT_EVOLUTION_METRIC;
}

export function characteristicSatelliteLayer(metricId) {
  return CHARACTERISTIC_SATELLITE_LAYERS[String(metricId)] || null;
}

export function normalizeDateParam(raw) {
  if (raw == null) return '';
  const value = String(raw).trim();
  let y;
  let m;
  let d;
  if (/^\d{8}$/.test(value)) {
    y = value.slice(0, 4);
    m = value.slice(4, 6);
    d = value.slice(6, 8);
  } else if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    [y, m, d] = value.split('-');
  } else if (/^\d{4}-\d{2}$/.test(value)) {
    [y, m] = value.split('-');
    d = '01';
  } else {
    return '';
  }
  const month = Number(m);
  const day = Number(d);
  if (month < 1 || month > 12 || day < 1 || day > 31) return '';
  return `${y}-${m}-${d}`;
}

export function dateParam(date) {
  const normalized = normalizeDateParam(date);
  return normalized ? normalized.replaceAll('-', '') : '';
}

export function monthValue(date) {
  const normalized = normalizeDateParam(date);
  return normalized ? normalized.slice(0, 7) : '';
}

export function dateFromMonthValue(month) {
  return normalizeDateParam(month);
}

export function pickDate(dates, requested, fallback = 'latest') {
  const clean = cleanDates(dates);
  if (!clean.length) return '';

  const normalized = normalizeDateParam(requested);
  if (!normalized) return fallback === 'earliest' ? clean[0] : clean[clean.length - 1];
  if (clean.includes(normalized)) return normalized;

  const sameMonth = clean.find(date => date.slice(0, 7) === normalized.slice(0, 7));
  if (sameMonth) return sameMonth;

  const target = Date.parse(`${normalized}T00:00:00Z`);
  if (!Number.isFinite(target)) return fallback === 'earliest' ? clean[0] : clean[clean.length - 1];
  return clean.reduce((best, date) => {
    const bestDiff = Math.abs(Date.parse(`${best}T00:00:00Z`) - target);
    const diff = Math.abs(Date.parse(`${date}T00:00:00Z`) - target);
    return diff < bestDiff ? date : best;
  }, clean[0]);
}

export function availableMonths(dates) {
  const seen = new Set();
  const out = [];
  for (const date of cleanDates(dates)) {
    const month = date.slice(0, 7);
    if (seen.has(month)) continue;
    seen.add(month);
    out.push(month);
  }
  return out;
}

export function satelliteValue(timeseries, parcelKey, layer, date) {
  const dates = cleanDates(timeseries?.dates);
  const requested = normalizeDateParam(date);
  if (!requested) return null;
  const dateIdx = dates.indexOf(pickDate(dates, requested));
  if (dateIdx < 0) return null;
  const value = timeseries?.means?.parcels?.[parcelKey]?.[layer]?.[dateIdx];
  return finite(value);
}

export function satelliteDiffValue(timeseries, parcelKey, layer, date1, date2) {
  const v1 = satelliteValue(timeseries, parcelKey, layer, date1);
  const v2 = satelliteValue(timeseries, parcelKey, layer, date2);
  return v1 == null || v2 == null ? null : v2 - v1;
}

export function satelliteDiffPngUrl(regionId, layer, date1, date2) {
  const d1 = normalizeDateParam(date1);
  const d2 = normalizeDateParam(date2);
  if (!regionId || !layer || !d1 || !d2) return '';
  return `/api/bosco/satellite/${encodeURIComponent(regionId)}/diff/`
    + `${encodeURIComponent(layer)}/${encodeURIComponent(d1)}/${encodeURIComponent(d2)}.png`;
}

export function divergingDomain(values) {
  const clean = values.filter(v => v != null && Number.isFinite(v));
  if (!clean.length) return null;
  return {
    min: Math.min(...clean),
    max: Math.max(...clean),
    maxAbs: Math.max(1e-9, ...clean.map(v => Math.abs(v))),
  };
}

export function satelliteColor(value) {
  const v = Math.max(-1, Math.min(1, finite(value) ?? 0));
  return rgbString(colormapLookup(INDEX_RAMP, Math.round((v + 1) * BYTE_MIDPOINT)));
}

export function diffColor(value, maxAbs) {
  const max = Number.isFinite(maxAbs) && maxAbs > 0 ? maxAbs : 1;
  const clamped = Math.max(-max, Math.min(max, finite(value) ?? 0));
  return rgbString(colormapLookup(DIFF_RAMP, Math.round(((clamped / max) + 1) * BYTE_MIDPOINT)));
}

export function colormapLookup(ramp, value) {
  const val = Math.max(0, Math.min(255, Number(value)));
  if (val <= ramp[0][0]) return ramp[0][1];
  for (let i = 1; i < ramp.length; i++) {
    if (val <= ramp[i][0]) {
      const t = (val - ramp[i - 1][0]) / (ramp[i][0] - ramp[i - 1][0]);
      const low = ramp[i - 1][1];
      const high = ramp[i][1];
      return interpolateRgb(low, high, t);
    }
  }
  return ramp[ramp.length - 1][1];
}

export function interpolateRgb(start, end, t) {
  return start.map((v, i) => Math.round(v + (end[i] - v) * t));
}

export function rgbString(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function cleanDates(dates) {
  const seen = new Set();
  const out = [];
  for (const raw of dates || []) {
    const date = normalizeDateParam(raw);
    if (!date || seen.has(date)) continue;
    seen.add(date);
    out.push(date);
  }
  return out;
}

function finite(value) {
  return toNumber(value);
}
