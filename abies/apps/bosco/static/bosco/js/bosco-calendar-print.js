/** Calendar-specific labels; geometry/layout/PDF stay independent of the metric. */
import { parcelIdentity } from '../../base/js/geo.js';
import { compareParcelNames } from '../../base/js/natural-sort.js';
import { FIELD_SELECTED_PLAN } from '../../base/js/constants.js';
import { fmtInt } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';
import { calendarYearsByParcel } from './bosco-calendar.js';

function parcelIsCoppice(entry) {
  if (typeof entry?.coppice !== 'boolean') throw new Error(S.BOSCO_PRINT_PARCEL_MISMATCH);
  return entry.coppice;
}

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
  const byName = new Map(entries.filter(entry => entry.region === regionName)
    .map(entry => [entry.parcel, entry]));
  const years = calendarYearsByParcel(data);
  const byParcel = new Map();
  for (const feature of geojson?.features || []) {
    const identity = parcelIdentity(feature);
    if (identity.regionName !== regionName) continue;
    const entry = byName.get(identity.parcelName);
    if (!entry) throw new Error(S.BOSCO_PRINT_PARCEL_MISMATCH);
    const geometry = feature.geometry;
    if (!['Polygon', 'MultiPolygon'].includes(geometry?.type)) {
      throw new Error(S.BOSCO_PRINT_GEOMETRY_ERROR);
    }
    if (!byParcel.has(entry.id)) {
      const planned = years.get(entry.id) || [];
      byParcel.set(entry.id, { parcel: entry.parcel,
        name: S.BOSCO_CALENDAR_PARCEL_NAME(entry.parcel, parcelIsCoppice(entry)), lines: [
        planned.length ? planned.map(year => fmtInt(year)).join(', ') : '-',
      ], geometry: { type: 'MultiPolygon', coordinates: [] } });
    }
    byParcel.get(entry.id).geometry.coordinates.push(...(
      geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates
    ));
  }
  if (!byParcel.size || [...byName.values()].some(entry => !byParcel.has(entry.id))) {
    throw new Error(S.BOSCO_PRINT_PARCEL_MISMATCH);
  }
  return [...byParcel.values()].sort((a, b) => compareParcelNames(a.parcel, b.parcel));
}
