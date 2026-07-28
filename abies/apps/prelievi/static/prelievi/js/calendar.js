/**
 * Prelievi calendar aggregation and search-filter helpers.
 */

import * as S from '../../base/js/strings.js';
import { monthBucket } from '../../base/js/charts.js';

export function buildHarvestCalendar(rows, colMap) {
  const dateIdx = colMap[S.COL_DATE];
  const regionIdx = colMap[S.COL_REGION];
  const parcelIdx = colMap[S.COL_PARCEL];
  const periods = new Set();
  const byRegion = new Map();

  for (const row of rows || []) {
    const period = monthBucket(row?.[dateIdx]);
    const region = cleanLabel(row?.[regionIdx]);
    const parcel = cleanLabel(row?.[parcelIdx]);
    if (!period || !region || !parcel) continue;

    periods.add(period);
    if (!byRegion.has(region)) byRegion.set(region, new Map());
    const parcels = byRegion.get(region);
    if (!parcels.has(parcel)) parcels.set(parcel, new Map());
    const cells = parcels.get(parcel);
    cells.set(period, (cells.get(period) || 0) + 1);
  }

  const sortedPeriods = [...periods].sort();
  const regions = [...byRegion.entries()]
    .sort(([a], [b]) => naturalSort(a, b))
    .map(([name, parcels]) => ({
      name,
      parcels: [...parcels.entries()]
        .sort(([a], [b]) => naturalSort(a, b))
        .map(([parcel, cells]) => ({ parcel, cells })),
    }));

  return { periods: sortedPeriods, regions };
}

export function calendarSearchText(current, { period, region, parcel }) {
  const base = displaySearchTerms(current).filter(term => !isCalendarFilterTerm(term));
  return [
    ...base,
    period,
    columnTerm(S.COL_REGION, region),
    columnTerm(S.COL_PARCEL, parcel),
  ].filter(Boolean).join(' ');
}

function displaySearchTerms(text) {
  const terms = [];
  let current = '';
  let inQuote = false;
  for (const char of String(text ?? '')) {
    if (char === '"') {
      inQuote = !inQuote;
      current += char;
    } else if (/\s/.test(char) && !inQuote) {
      if (current) { terms.push(current); current = ''; }
    } else {
      current += char;
    }
  }
  if (current) terms.push(current);
  return terms;
}

function isCalendarFilterTerm(term) {
  const clean = String(term || '').trim().toLowerCase();
  return /^\d{4}(?:-\d{2})?$/.test(clean)
    || clean.startsWith(`${S.COL_REGION.toLowerCase()}:`)
    || clean.startsWith(`${S.COL_PARCEL.toLowerCase()}:`);
}

function columnTerm(column, value) {
  return `${column}:${quoteCriterion(value)}`;
}

function quoteCriterion(value) {
  const clean = cleanLabel(value).replaceAll('"', '');
  return /\s/.test(clean) ? `"${clean}"` : clean;
}

function cleanLabel(value) {
  return String(value ?? '').trim();
}

function naturalSort(a, b) {
  return String(a).localeCompare(String(b), S.LOCALE, { numeric: true });
}
