import * as S from '../../base/js/strings.js';
import {
  COL_PARCEL_ID, COL_SPECIES_ID, COLUMNS, ROWS, ROW_ID, VERSION,
} from '../../base/js/constants.js';

const COLORS = [
  '#2e7d32', '#1565c0', '#e65100', '#6a1b9a', '#c62828',
  '#00838f', '#827717', '#4e342e', '#37474f', '#ad1457',
];

function coord(v) {
  if (v == null || v === '') return NaN;
  return Number(v);
}

function colMap(digest) {
  const out = {};
  digest[COLUMNS].forEach((name, idx) => { out[name] = idx; });
  return out;
}

export function buildPreservedTrees(digest) {
  if (!digest) return [];
  const c = colMap(digest);
  return digest[ROWS].map(row => ({
    id: row[c[ROW_ID]],
    version: row[c[VERSION]],
    parcelId: row[c[COL_PARCEL_ID]],
    speciesId: row[c[COL_SPECIES_ID]],
    region: row[c[S.COL_REGION]],
    parcel: row[c[S.COL_PARCEL]],
    species: row[c[S.COL_SPECIES]],
    year: row[c[S.COL_YEAR]],
    lat: coord(row[c[S.COL_LAT]]),
    lon: coord(row[c[S.COL_LON]]),
    note: row[c[S.COL_NOTE]] || '',
  })).filter(t => Number.isFinite(t.lat) && Number.isFinite(t.lon));
}

export function filterPaiTrees(trees, { region, parcelIds = null, speciesIds = null } = {}) {
  const parcels = parcelIds == null ? null : new Set(parcelIds);
  const species = speciesIds == null ? null : new Set(speciesIds);
  return trees.filter(t => (!region || t.region === region)
    && (!parcels || parcels.has(t.parcelId))
    && (!species || species.has(t.speciesId)));
}

export function paiParcelItems(entries, trees) {
  const counts = countBy(trees, t => t.parcelId);
  return entries.map(entry => ({
    id: entry.id,
    name: entry.parcel,
    count: counts.get(entry.id) || 0,
  })).sort((a, b) => a.name.localeCompare(b.name, 'it', { numeric: true }));
}

export function paiSpeciesItems(trees) {
  const byId = new Map();
  for (const tree of trees) {
    const item = byId.get(tree.speciesId) || { id: tree.speciesId, name: tree.species, count: 0 };
    item.count += 1;
    byId.set(tree.speciesId, item);
  }
  return [...byId.values()].sort((a, b) => a.name.localeCompare(b.name, 'it'));
}

export function speciesColorMap(speciesItems) {
  const out = new Map();
  speciesItems.forEach((item, idx) => out.set(item.id, COLORS[idx % COLORS.length]));
  return out;
}

function countBy(items, keyFn) {
  const out = new Map();
  for (const item of items) {
    const key = keyFn(item);
    out.set(key, (out.get(key) || 0) + 1);
  }
  return out;
}
