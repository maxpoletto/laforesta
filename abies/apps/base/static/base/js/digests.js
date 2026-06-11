// Shared helpers for JSON digest consumers.

import { COLUMNS } from './constants.js';

export function columnMap(digestOrColumns) {
  const columns = Array.isArray(digestOrColumns)
    ? digestOrColumns
    : digestOrColumns?.[COLUMNS];
  const out = {};
  for (const [idx, name] of (columns || []).entries()) out[name] = idx;
  return out;
}

export function toNumber(value, fallback = null) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}
