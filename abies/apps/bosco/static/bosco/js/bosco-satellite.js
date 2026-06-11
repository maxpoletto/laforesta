// Pure helpers for Bosco satellite-timeseries UI.

export const DEFAULT_EVOLUTION_METRIC = '1';

export const CHARACTERISTIC_SATELLITE_LAYERS = {
  '6': 'ndvi',
  '7': 'ndmi',
  '8': 'evi',
};

export const SATELLITE_LAYERS = {
  ndvi: { label: 'NDVI' },
  ndmi: { label: 'NDMI' },
  evi: { label: 'EVI' },
};

export const EVOLUTION_METRICS = {
  '1': { label: 'NDVI', layer: 'ndvi', satellite: true },
  '2': { label: 'NDMI', layer: 'ndmi', satellite: true },
  '3': { label: 'EVI', layer: 'evi', satellite: true },
  '4': { layer: 'prelievo', satellite: false },
};

const INDEX_RAMP = [
  [0,   [139, 90, 43]],
  [128, [245, 235, 200]],
  [255, [0, 100, 0]],
];

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
  return rgbString(colormapLookup(INDEX_RAMP, Math.round((v + 1) * 127.5)));
}

export function diffColor(value, maxAbs) {
  const max = Number.isFinite(maxAbs) && maxAbs > 0 ? maxAbs : 1;
  const clamped = Math.max(-max, Math.min(max, finite(value) ?? 0));
  return rgbString(colormapLookup(DIFF_RAMP, Math.round(((clamped / max) + 1) * 127.5)));
}

export function colormapLookup(ramp, value) {
  const val = Math.max(0, Math.min(255, Number(value)));
  if (val <= ramp[0][0]) return ramp[0][1];
  for (let i = 1; i < ramp.length; i++) {
    if (val <= ramp[i][0]) {
      const t = (val - ramp[i - 1][0]) / (ramp[i][0] - ramp[i - 1][0]);
      const low = ramp[i - 1][1];
      const high = ramp[i][1];
      return [
        Math.round(low[0] + t * (high[0] - low[0])),
        Math.round(low[1] + t * (high[1] - low[1])),
        Math.round(low[2] + t * (high[2] - low[2])),
      ];
    }
  }
  return ramp[ramp.length - 1][1];
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
  if (value == null || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
