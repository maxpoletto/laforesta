// Shared Bosco metric ids and flags.  Keep dependency-free: URL-state parsing
// imports this before the full characteristic digest helpers are needed.

export const Q_AGE = '1';
export const Q_TYPE = '2';
export const Q_ALTITUDE = '3';
export const Q_HISTORICAL_HARVEST = '4';
export const Q_FUTURE_HARVEST = '5';
export const Q_NDVI = '6';
export const Q_NDMI = '7';
export const Q_EVI = '8';

export const E_NDVI = '1';
export const E_NDMI = '2';
export const E_EVI = '3';
export const E_HARVEST = '4';

export const CHARACTERISTIC_METRICS = {
  [Q_AGE]: { kind: 'continuous', unit: 'a' },
  [Q_TYPE]: { kind: 'type' },
  [Q_ALTITUDE]: { kind: 'continuous', unit: 'm' },
  [Q_HISTORICAL_HARVEST]: { kind: 'continuous', unit: 'q', harvest: true },
  [Q_FUTURE_HARVEST]: { kind: 'continuous', unit: 'm³', harvest: true },
  [Q_NDVI]: { kind: 'satellite', layer: 'ndvi' },
  [Q_NDMI]: { kind: 'satellite', layer: 'ndmi' },
  [Q_EVI]: { kind: 'satellite', layer: 'evi' },
};

export const CHARACTERISTIC_METRIC_IDS = Object.keys(CHARACTERISTIC_METRICS);

const HARVEST_METRICS = new Set([Q_HISTORICAL_HARVEST, Q_FUTURE_HARVEST]);

export function isHarvestMetric(metricId) {
  return HARVEST_METRICS.has(String(metricId));
}
