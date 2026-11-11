import { FIELD_CALENDAR, FIELD_PARCEL_ID, FIELD_YEARS } from '../../base/js/constants.js';
import { fmtInt } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

export function calendarYearsByParcel(data) {
  return new Map((data?.[FIELD_CALENDAR] || []).map(row => [
    row[FIELD_PARCEL_ID], [...new Set(row[FIELD_YEARS] || [])]
      .filter(year => Number.isInteger(year) && year > 0).sort((a, b) => a - b),
  ]));
}

export function calendarDomain(plan) {
  if (!plan || !Number.isInteger(plan.year_start) || !Number.isInteger(plan.year_end)
      || plan.year_end < plan.year_start) return null;
  return { min: plan.year_start, max: plan.year_end };
}

export function calendarColorPosition(year, plan) {
  const domain = calendarDomain(plan);
  if (year == null || !Number.isFinite(year) || !domain) return null;
  if (domain.max === domain.min) return 0;
  return Math.max(0, Math.min(1, (year - domain.min) / (domain.max - domain.min)));
}

export function calendarYearsText(years = []) {
  return years.length ? years.map(year => fmtInt(year)).join(', ') : S.BOSCO_NO_PLANNED_CUTS;
}
