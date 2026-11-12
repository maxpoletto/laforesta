/** Join printable parcel labels to complete, possibly split GeoJSON geometry. */
import { parcelIdentity } from '../../base/js/geo.js';
import { compareParcelNames } from '../../base/js/natural-sort.js';
import * as S from '../../base/js/strings.js';

export function parcelIsCoppice(entry) {
  if (typeof entry?.coppice !== 'boolean') throw new Error(S.BOSCO_PRINT_PARCEL_MISMATCH);
  return entry.coppice;
}

export function parcelPrintItems(geojson, regionName, entries, linesForEntry) {
  const byName = new Map(entries.filter(entry => entry.region === regionName)
    .map(entry => [entry.parcel, entry]));
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
      byParcel.set(entry.id, { parcel: entry.parcel,
        name: S.BOSCO_PRINT_PARCEL_NAME(entry.parcel, parcelIsCoppice(entry)),
        lines: linesForEntry(entry), geometry: { type: 'MultiPolygon', coordinates: [] } });
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
