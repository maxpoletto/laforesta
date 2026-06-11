// Shared Bosco metric ids and flags.  Keep dependency-free: URL-state parsing
// imports this before the full characteristic digest helpers are needed.

export const Q_AGE = '1';
export const Q_TYPE = '2';
export const Q_ALTITUDE = '3';
export const Q_HISTORICAL_HARVEST = '4';
export const Q_FUTURE_HARVEST = '5';

const HARVEST_METRICS = new Set([Q_HISTORICAL_HARVEST, Q_FUTURE_HARVEST]);

export function isHarvestMetric(metricId) {
  return HARVEST_METRICS.has(String(metricId));
}
