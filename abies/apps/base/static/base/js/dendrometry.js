/**
 * Shared dendrometry aggregation, legend, and chart builders.
 *
 * Bosco supplies already-aggregated rows; tree-detail consumers supply
 * individual trees. Both use the same five-centimetre diameter classes,
 * species palette, sparse-series filling, and metric matrix.
 */

import * as S from './strings.js';
import {
  chartSeriesColor, renderStackedBar, speciesColorMap as chartSpeciesColorMap,
} from './charts.js';
import { columnMap, toNumber } from './digests.js';
import { fmtDecimal1, fmtDecimal2, fmtInt, fmtVolume } from './format.js';

export function dendrometrySpeciesColor(idx) {
  return chartSeriesColor(idx);
}

export function dendrometrySpeciesColorMap(speciesNames, allSpeciesNames = []) {
  const colorByName = chartSpeciesColorMap([...speciesNames.values()], allSpeciesNames);
  return new Map([...speciesNames.entries()].map(([id, name], idx) => [
    id, colorByName.get(name) || dendrometrySpeciesColor(idx),
  ]));
}

export function dendrometryChartKey(speciesId, diameterClassCm) {
  return `${speciesId}|${diameterClassCm}`;
}

export function diameterClassCm(dCm) {
  return Math.floor((Number(dCm) + 2) / 5) * 5;
}

export function basalAreaM2(dCm) {
  const radiusM = Number(dCm) / 200;
  return Math.PI * radiusM * radiusM;
}

/** Aggregate individual tree digest rows for dendrometry displays. */
export function aggregateTreeDendrometry(rows, columns, { allSpeciesNames = [] } = {}) {
  const c = columnMap(columns);
  const groups = new Map();
  const speciesNames = new Map();

  for (const row of rows || []) {
    const species = String(row[c[S.COL_SPECIES]] || '').trim();
    const diameter = toNumber(row[c[S.COL_D_CM]]);
    if (!species || diameter == null || diameter <= 0) continue;
    const diameterClass = diameterClassCm(diameter);
    const speciesId = species;
    const key = dendrometryChartKey(speciesId, diameterClass);
    const group = groups.get(key) || {
      speciesId,
      species,
      diameterClassCm: diameterClass,
      treeCount: 0,
      volumeM3: 0,
      basalAreaM2: 0,
    };
    group.treeCount += 1;
    group.volumeM3 += toNumber(row[c[S.COL_V_M3]], 0);
    group.basalAreaM2 += basalAreaM2(diameter);
    groups.set(key, group);
    speciesNames.set(speciesId, species);
  }

  const colors = dendrometrySpeciesColorMap(speciesNames, allSpeciesNames);
  return [...groups.values()]
    .sort((a, b) => a.species.localeCompare(b.species, S.LOCALE)
      || a.diameterClassCm - b.diameterClassCm)
    .map(group => ({
      ...group,
      color: colors.get(group.speciesId),
      volumeM3: round(group.volumeM3, 4),
      basalAreaM2: round(group.basalAreaM2, 6),
    }));
}

// Compatibility for callers outside the page bundle. New code should use the
// generic name: nothing in the aggregation is specific to marked trees.
export const aggregateMarkedTreeDendrometry = aggregateTreeDendrometry;

/** Build species rows × continuous five-centimetre diameter-class columns. */
export function dendrometryMetricMatrix(rows, metric) {
  const { labels, species } = dendrometryChartAxes(rows);
  const values = dendrometryMetricValues(rows, metric);
  return {
    diameterClasses: labels.map(Number),
    species: species.map(item => ({
      ...item,
      values: labels.map(label => round(
        values.get(dendrometryChartKey(item.id, Number(label))) || 0, 4,
      )),
    })),
  };
}

export function dendrometryBarChartData(rows, metric, yTitle) {
  const matrix = dendrometryMetricMatrix(rows, metric);
  return {
    labels: matrix.diameterClasses.map(String),
    yTitle,
    legend: false,
    datasets: matrix.species.map((item, idx) => ({
      label: item.name,
      data: item.values,
      backgroundColor: item.color || dendrometrySpeciesColor(idx),
    })),
  };
}

/** Render the three bar charts shared by Bosco and marked-tree summaries. */
export function renderDendrometryBarCharts({
  rows, canvases, yTitles, existing = {},
}) {
  const definitions = [
    ['treeCount', 'treeCount'],
    ['volume', 'volumeM3'],
    ['basalArea', 'basalAreaM2'],
  ];
  return Object.fromEntries(definitions.map(([key, metric]) => [
    key,
    renderStackedBar(
      canvases?.[key],
      dendrometryBarChartData(rows, metric, yTitles?.[key]),
      existing[key],
    ),
  ]));
}

/** Render the shared, non-interactive species color legend. */
export function renderDendrometryLegend(host, rows) {
  if (!host) return;
  const doc = host.ownerDocument || globalThis.document;
  const species = dendrometryLegendItems(rows);
  host.replaceChildren(...species.map(item => {
    const label = doc.createElement('span');
    label.className = 'dendrometry-species-item';
    const dot = doc.createElement('span');
    dot.className = 'dendrometry-species-dot';
    dot.style.backgroundColor = item.color;
    label.append(dot, doc.createTextNode(item.name));
    return label;
  }));
}

/** Species with a positive summed tree count, in chart display order. */
export function dendrometryLegendItems(rows) {
  const counts = new Map();
  for (const row of rows) {
    counts.set(
      row.speciesId,
      (counts.get(row.speciesId) || 0) + toNumber(row.treeCount, 0),
    );
  }
  const { species } = dendrometryChartAxes(rows);
  return species.filter(item => (counts.get(item.id) || 0) > 0);
}

export function dendrometryLineChartData(rows, metric, yTitle) {
  const { labels, species } = dendrometryChartAxes(rows);
  const values = dendrometryMetricValues(rows, metric);
  return {
    labels,
    yTitle,
    legend: false,
    datasets: species.map((item, idx) => ({
      label: item.name,
      data: labels.map(label => (
        values.get(dendrometryChartKey(item.id, Number(label))) ?? null
      )),
      borderColor: item.color || dendrometrySpeciesColor(idx),
      backgroundColor: item.color || dendrometrySpeciesColor(idx),
      tension: 0.25,
      spanGaps: true,
    })),
  };
}

export function dendrometryTreeSum(rows) {
  return sum(rows.map(row => row.treeCount));
}

export function dendrometryTreeTotal(rows) {
  return Math.round(dendrometryTreeSum(rows));
}

export function dendrometryVolumeSum(rows) {
  return sum(rows.map(row => row.volumeM3));
}

export function dendrometryBasalAreaSum(rows) {
  return sum(rows.map(row => row.basalAreaM2));
}

export function dendrometryDiameterStats(rows) {
  let weight = 0;
  let total = 0;
  for (const row of rows) {
    const diameter = toNumber(row.diameterClassCm);
    const rowWeight = toNumber(row.treeCount, 0);
    if (diameter == null || rowWeight <= 0) continue;
    weight += rowWeight;
    total += diameter * rowWeight;
  }
  if (weight <= 0) return null;

  const mean = total / weight;
  let varianceTotal = 0;
  for (const row of rows) {
    const diameter = toNumber(row.diameterClassCm);
    const rowWeight = toNumber(row.treeCount, 0);
    if (diameter == null || rowWeight <= 0) continue;
    varianceTotal += ((diameter - mean) ** 2) * rowWeight;
  }

  return {
    meanCm: round(mean, 4),
    sigmaCm: round(Math.sqrt(varianceTotal / weight), 4),
  };
}

/** Return the localized lines shown below the three core charts. */
export function dendrometrySummaryLines(rows, { perHa = false } = {}) {
  const basalArea = dendrometryBasalAreaSum(rows);
  const diameterStats = dendrometryDiameterStats(rows);
  const basalLines = [
    perHa
      ? S.BOSCO_BASAL_AREA_PER_HA_SUMMARY(S.BOSCO_BASAL_AREA_PER_HA_VALUE(
        fmtDecimal2(basalArea),
      ))
      : S.BOSCO_TOTAL_BASAL_AREA(S.BOSCO_BASAL_AREA_VALUE(fmtDecimal2(basalArea))),
  ];
  if (diameterStats) {
    basalLines.push(S.BOSCO_AVG_DIAMETER(
      fmtDecimal1(diameterStats.meanCm), fmtDecimal1(diameterStats.sigmaCm),
    ));
  }
  return {
    treeCount: [
      perHa
        ? S.BOSCO_TREES_PER_HA(fmtDecimal1(dendrometryTreeSum(rows)))
        : S.BOSCO_TOTAL_TREES(fmtInt(dendrometryTreeTotal(rows))),
    ],
    volume: [
      perHa
        ? S.BOSCO_VOLUME_PER_HA_SUMMARY(S.BOSCO_VOLUME_PER_HA_VALUE(
          fmtDecimal2(dendrometryVolumeSum(rows)),
        ))
        : S.BOSCO_TOTAL_VOLUME(fmtVolume(dendrometryVolumeSum(rows))),
    ],
    basalArea: basalLines,
  };
}

/** Render Bosco's localized summary lines into the three chart info hosts. */
export function renderDendrometrySummaryInfo(hosts, rows, options = {}) {
  const summaries = dendrometrySummaryLines(rows, options);
  for (const [key, lines] of Object.entries(summaries)) {
    const host = hosts?.[key];
    if (!host) continue;
    const doc = host.ownerDocument || globalThis.document;
    host.replaceChildren(...lines.map(line => {
      const div = doc.createElement('div');
      div.textContent = line;
      return div;
    }));
  }
}

export function clearDendrometrySummaryInfo(hosts) {
  for (const host of Object.values(hosts || {})) host?.replaceChildren();
}

function dendrometryMetricValues(rows, metric) {
  return new Map(rows.map(row => [
    dendrometryChartKey(row.speciesId, row.diameterClassCm),
    toNumber(row[metric], 0),
  ]));
}

function dendrometryChartAxes(rows) {
  const labels = dendrometryClassRange(rows).map(String);
  const bySpecies = new Map();
  for (const row of rows) {
    if (!bySpecies.has(row.speciesId)) {
      bySpecies.set(row.speciesId, {
        id: row.speciesId,
        name: row.species,
        color: row.color,
      });
    }
  }
  const species = [...bySpecies.values()]
    .sort((a, b) => a.name.localeCompare(b.name, S.LOCALE));
  return { labels, species };
}

function dendrometryClassRange(rows) {
  const classes = rows
    .map(row => toNumber(row.diameterClassCm))
    .filter(value => value != null);
  if (!classes.length) return [];
  const start = Math.min(...classes);
  const end = Math.max(...classes);
  const out = [];
  for (let cm = start; cm <= end; cm += 5) out.push(cm);
  return out;
}

function sum(values) {
  let total = 0;
  for (const value of values) if (value != null && Number.isFinite(value)) total += value;
  return total;
}

function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}
