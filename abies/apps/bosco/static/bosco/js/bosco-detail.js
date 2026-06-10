import * as S from '../../base/js/strings.js';
import { COLUMNS, ROWS } from '../../base/js/constants.js';

function colMap(digest) {
  const out = {};
  digest[COLUMNS].forEach((name, idx) => { out[name] = idx; });
  return out;
}

function num(v) {
  if (v == null || v === '') return 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function maybeNum(v) {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function regionMetadata(entries) {
  const count = entries.length;
  const areaHa = sum(entries.map(e => e.displayAreaHa));
  const cadastralAreaHa = sum(entries.map(e => e.cadastralAreaHa));
  const ageWeighted = weightedAverage(entries.map(e => [e.aveAge, e.displayAreaHa || 1]));
  const altMin = min(entries.map(e => e.altMin));
  const altMax = max(entries.map(e => e.altMax));
  const typeCounts = new Map();
  for (const entry of entries) {
    if (!entry.type) continue;
    typeCounts.set(entry.type, (typeCounts.get(entry.type) || 0) + 1);
  }
  return { count, areaHa, cadastralAreaHa, aveAge: ageWeighted, altMin, altMax, typeCounts };
}

export function aggregateDendrometry(digest, scope, { areaHa = null, perHa = true, speciesIds = [] } = {}) {
  if (!digest) return [];
  const c = colMap(digest);
  const allowedSpecies = new Set(speciesIds || []);
  const groups = new Map();

  for (const row of digest[ROWS]) {
    if (scope.parcelId != null && row[c[S.COL_PARCEL_ID]] !== scope.parcelId) continue;
    if (scope.parcelId == null && scope.region && row[c[S.COL_REGION]] !== scope.region) continue;
    const speciesId = row[c[S.COL_SPECIES_ID]];
    if (allowedSpecies.size && !allowedSpecies.has(speciesId)) continue;

    const key = `${speciesId}|${row[c[S.COL_DIAM_CLASS_CM]]}`;
    const g = groups.get(key) || {
      speciesId,
      species: row[c[S.COL_SPECIES]],
      diameterClassCm: row[c[S.COL_DIAM_CLASS_CM]],
      treeCount: 0,
      volumeM3: 0,
      basalAreaM2: 0,
      heightSum: 0,
      heightWeight: 0,
      incrementSum: 0,
      incrementWeight: 0,
    };
    const nTrees = num(row[c[S.COL_N_TREES]]);
    g.treeCount += nTrees;
    g.volumeM3 += num(row[c[S.COL_VOLUME_M3]]);
    g.basalAreaM2 += num(row[c[S.COL_BASAL_AREA_M2]]);
    const h = maybeNum(row[c[S.COL_AVG_H_M]]);
    if (h !== null && nTrees > 0) {
      g.heightSum += h * nTrees;
      g.heightWeight += nTrees;
    }
    const inc = maybeNum(row[c[S.COL_INCREMENT_PCT]]);
    if (inc !== null && nTrees > 0) {
      g.incrementSum += inc * nTrees;
      g.incrementWeight += nTrees;
    }
    groups.set(key, g);
  }

  const scale = perHa && areaHa ? 1 / areaHa : 1;
  return [...groups.values()]
    .sort((a, b) => a.species.localeCompare(b.species, 'it')
      || a.diameterClassCm - b.diameterClassCm)
    .map(g => ({
      speciesId: g.speciesId,
      species: g.species,
      diameterClassCm: g.diameterClassCm,
      treeCount: round(g.treeCount * scale, 4),
      volumeM3: round(g.volumeM3 * scale, 4),
      basalAreaM2: round(g.basalAreaM2 * scale, 6),
      avgHeightM: g.heightWeight ? round(g.heightSum / g.heightWeight, 4) : null,
      incrementPct: g.incrementWeight ? round(g.incrementSum / g.incrementWeight, 4) : null,
    }));
}

export function dendrometrySpecies(digest, scope) {
  const rows = aggregateDendrometry(digest, scope, { perHa: false });
  const out = new Map();
  for (const row of rows) {
    const item = out.get(row.speciesId) || { id: row.speciesId, name: row.species, count: 0 };
    item.count += row.treeCount;
    out.set(row.speciesId, item);
  }
  return [...out.values()].sort((a, b) => a.name.localeCompare(b.name, 'it'));
}

function weightedAverage(pairs) {
  let total = 0;
  let weight = 0;
  for (const [value, w] of pairs) {
    if (value == null || !Number.isFinite(value) || !w) continue;
    total += value * w;
    weight += w;
  }
  return weight ? total / weight : null;
}

function sum(values) {
  let total = 0;
  for (const value of values) if (value != null && Number.isFinite(value)) total += value;
  return total;
}

function min(values) {
  const clean = values.filter(v => v != null && Number.isFinite(v));
  return clean.length ? Math.min(...clean) : null;
}

function max(values) {
  const clean = values.filter(v => v != null && Number.isFinite(v));
  return clean.length ? Math.max(...clean) : null;
}

function round(v, places) {
  const k = 10 ** places;
  return Math.round(v * k) / k;
}
