/**
 * Prelievi calendar aggregation and search-filter helpers.
 */

import * as S from '../../base/js/strings.js';
import { monthBucket, yearBucket } from '../../base/js/charts.js';
import { compareNaturalLabels, compareParcelNames } from '../../base/js/natural-sort.js';
import { resolveColumnSearchTerm, searchTerms } from '../../base/js/table.js';
import { HARVEST_SCOPE_COLUMNS, harvestScopeSearchTerms } from '../../base/js/prelievi-columns.js';

export function buildHarvestCalendar(rows, colMap, byMonth = false) {
  const dateIdx = colMap[S.COL_DATE];
  const regionIdx = colMap[S.COL_REGION];
  const parcelIdx = colMap[S.COL_PARCEL];
  const bucket = byMonth ? monthBucket : yearBucket;
  const periods = new Set();
  const byRegion = new Map();

  for (const row of rows || []) {
    const period = bucket(row?.[dateIdx]);
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
    .sort(([a], [b]) => compareNaturalLabels(a, b))
    .map(([name, parcels]) => ({
      name,
      parcels: [...parcels.entries()]
        .sort(([a], [b]) => compareParcelNames(a, b))
        .map(([parcel, cells]) => ({ parcel, cells })),
    }));

  return { periods: sortedPeriods, regions };
}

export function calendarSearchText(current, scope, columns) {
  const scopeKeys = Object.values(HARVEST_SCOPE_COLUMNS);
  const base = searchTerms(current, { keepDisplay: true }).filter(term => {
    const clause = resolveColumnSearchTerm(searchTerms(term)[0] ?? '', columns);
    return !scopeKeys.includes(clause?.match?.col.key);
  });
  return [...base, ...harvestScopeSearchTerms(scope)].join(' ');
}

function cleanLabel(value) {
  return String(value ?? '').trim();
}
