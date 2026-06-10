import * as S from '../../base/js/strings.js';
import { COLUMNS, ROWS } from '../../base/js/constants.js';

const CHART_COLORS = [
  '#2f8f58', '#1565c0', '#d7aa27', '#8d3f86', '#c94f4f',
  '#00838f', '#6b7f2a', '#6d4c41', '#546e7a', '#ad5a7a',
];

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

export function aggregateDendrometry(digest, scope, { areaHa = null, perHa = true, speciesIds = null } = {}) {
  if (!digest) return [];
  const c = colMap(digest);
  const hasSpeciesFilter = Array.isArray(speciesIds);
  if (hasSpeciesFilter && !speciesIds.length) return [];
  const allowedSpecies = new Set(speciesIds || []);
  const groups = new Map();

  for (const row of digest[ROWS]) {
    if (scope.parcelId != null && row[c[S.COL_PARCEL_ID]] !== scope.parcelId) continue;
    if (scope.parcelId == null && scope.region && row[c[S.COL_REGION]] !== scope.region) continue;
    const speciesId = row[c[S.COL_SPECIES_ID]];
    if (hasSpeciesFilter && !allowedSpecies.has(speciesId)) continue;

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


export function dendrometryBarChartData(rows, metric, yTitle) {
  const { labels, species } = dendrometryChartAxes(rows);
  const values = new Map(rows.map(row => [
    dendrometryChartKey(row.speciesId, row.diameterClassCm), num(row[metric]),
  ]));
  return {
    labels,
    yTitle,
    datasets: species.map((item, idx) => ({
      label: item.name,
      data: labels.map(label => round(values.get(dendrometryChartKey(item.id, Number(label))) || 0, 4)),
      backgroundColor: CHART_COLORS[idx % CHART_COLORS.length],
    })),
  };
}

export function dendrometryLineChartData(rows, metric, yTitle) {
  const { labels, species } = dendrometryChartAxes(rows);
  const values = new Map(rows.map(row => [
    dendrometryChartKey(row.speciesId, row.diameterClassCm),
    maybeNum(row[metric]),
  ]));
  return {
    labels,
    yTitle,
    datasets: species.map((item, idx) => ({
      label: item.name,
      data: labels.map(label => values.get(dendrometryChartKey(item.id, Number(label))) ?? null),
      borderColor: CHART_COLORS[idx % CHART_COLORS.length],
      backgroundColor: CHART_COLORS[idx % CHART_COLORS.length],
      tension: 0.25,
      spanGaps: true,
    })),
  };
}

export function dendrometryTreeTotal(rows) {
  return Math.round(sum(rows.map(row => row.treeCount)));
}

function dendrometryChartAxes(rows) {
  const labels = [...new Set(rows.map(row => row.diameterClassCm))]
    .sort((a, b) => a - b)
    .map(String);
  const bySpecies = new Map();
  for (const row of rows) {
    if (!bySpecies.has(row.speciesId)) bySpecies.set(row.speciesId, row.species);
  }
  const species = [...bySpecies.entries()]
    .map(([id, name]) => ({ id, name }))
    .sort((a, b) => a.name.localeCompare(b.name, 'it'));
  return { labels, species };
}

function dendrometryChartKey(speciesId, diameterClassCm) {
  return `${speciesId}|${diameterClassCm}`;
}


export function dendrometryHeightPoints(digest, scope, { speciesIds = null } = {}) {
  if (!digest) return [];
  const c = colMap(digest);
  const hasSpeciesFilter = Array.isArray(speciesIds);
  if (hasSpeciesFilter && !speciesIds.length) return [];
  const allowedSpecies = new Set(speciesIds || []);
  const rows = [];

  for (const row of digest[ROWS]) {
    if (scope.parcelId != null && row[c[S.COL_PARCEL_ID]] !== scope.parcelId) continue;
    if (scope.parcelId == null && scope.region && row[c[S.COL_REGION]] !== scope.region) continue;
    const speciesId = row[c[S.COL_SPECIES_ID]];
    if (hasSpeciesFilter && !allowedSpecies.has(speciesId)) continue;
    const dCm = maybeNum(row[c[S.COL_D_CM]]);
    const hM = maybeNum(row[c[S.COL_H_M]]);
    if (dCm == null || hM == null) continue;
    rows.push({
      speciesId,
      species: row[c[S.COL_SPECIES]],
      dCm,
      hM,
    });
  }

  return rows.sort((a, b) => a.species.localeCompare(b.species, 'it') || a.dCm - b.dCm);
}

export function dendrometryScatterChartData(points, yTitle) {
  const bySpecies = new Map();
  for (const point of points) {
    const series = bySpecies.get(point.speciesId) || {
      id: point.speciesId,
      name: point.species,
      points: [],
    };
    series.points.push({ x: point.dCm, y: point.hM });
    bySpecies.set(point.speciesId, series);
  }
  const species = [...bySpecies.values()]
    .sort((a, b) => a.name.localeCompare(b.name, 'it'));
  return {
    yTitle,
    datasets: species.map((item, idx) => ({
      type: 'scatter',
      label: item.name,
      data: item.points,
      backgroundColor: CHART_COLORS[idx % CHART_COLORS.length],
      borderColor: CHART_COLORS[idx % CHART_COLORS.length],
      pointRadius: 3,
    })),
  };
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
