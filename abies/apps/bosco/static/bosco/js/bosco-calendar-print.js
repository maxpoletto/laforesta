/** Calendar-specific labels; geometry/layout/PDF stay independent of the metric. */
import { FIELD_SELECTED_PLAN } from '../../base/js/constants.js';
import { fmtInt } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';
import { calendarYearsByParcel } from './bosco-calendar.js';
import { parcelIsCoppice, parcelPrintItems } from './bosco-parcel-print.js';

export function calendarPlanSubtitle(entries, data) {
  const plan = data?.[FIELD_SELECTED_PLAN];
  if (!plan) throw new Error(S.BOSCO_NO_HARVEST_PLANS);
  const byId = new Map(entries.map(entry => [entry.id, entry]));
  const ranges = new Map([false, true].map(coppice => [coppice, { min: Infinity, max: -Infinity }]));
  for (const [parcelId, years] of calendarYearsByParcel(data)) {
    if (!years.length) continue;
    const range = ranges.get(parcelIsCoppice(byId.get(parcelId)));
    range.min = Math.min(range.min, years[0]);
    range.max = Math.max(range.max, years.at(-1));
  }
  const labels = [];
  for (const [coppice, range] of ranges) {
    if (!Number.isFinite(range.min)) continue;
    const years = range.min === range.max ? fmtInt(range.min)
      : `${fmtInt(range.min)}-${fmtInt(range.max)}`;
    labels.push(coppice ? S.BOSCO_CALENDAR_COPPICE_YEARS(years)
      : S.BOSCO_CALENDAR_HIGH_FOREST_YEARS(years));
  }
  return S.BOSCO_CALENDAR_MAP_PLAN(plan, labels.join(', '));
}

export function calendarPrintItems(geojson, regionName, entries, data) {
  if (!data?.[FIELD_SELECTED_PLAN]) throw new Error(S.BOSCO_NO_HARVEST_PLANS);
  const years = calendarYearsByParcel(data);
  return parcelPrintItems(geojson, regionName, entries, entry => {
    const planned = years.get(entry.id) || [];
    return [planned.length ? planned.map(year => fmtInt(year)).join(', ') : '-'];
  });
}
