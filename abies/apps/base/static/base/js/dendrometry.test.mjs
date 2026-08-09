// Tests for shared dendrometry aggregation and chart/table matrices.

import * as S from './strings.js';
import {
  DIAMETER_CLASS_SHIFTED_DOWN, DIAMETER_CLASS_SHIFTED_UP,
} from './constants.js';
import {
  aggregateTreeDendrometry, basalAreaM2, dendrometryBarChartData,
  dendrometryLegendItems, dendrometryMetricMatrix, dendrometrySpeciesColor,
  dendrometrySummaryLines, diameterClassCm,
} from './dendrometry.js';

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, message) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${message}: expected ${e}, got ${a}`);
  }
}

function assertClose(actual, expected, tolerance, message) {
  if (Math.abs(actual - expected) <= tolerance) passed++;
  else {
    failed++;
    console.error(`FAIL ${message}: expected ${expected}, got ${actual}`);
  }
}

console.log('dendrometry.js');

assertEqual([diameterClassCm(18), diameterClassCm(22), diameterClassCm(23)],
            [20, 20, 25], 'diameter classes match Bosco');
assertEqual([18, 19, 20, 24, 25].map(d => diameterClassCm(d, DIAMETER_CLASS_SHIFTED_UP)),
            [15, 15, 20, 20, 25],
            'shifted-up class uses its name as the lower integer bound');
assertEqual([15, 16, 20, 21].map(d => diameterClassCm(d, DIAMETER_CLASS_SHIFTED_DOWN)),
            [15, 20, 20, 25],
            'shifted-down class uses its name as the upper integer bound');
assertEqual(diameterClassCm(18, 'legacy-or-invalid'), 20,
            'unknown digest metadata falls back to centered classes');
assertClose(basalAreaM2(20), Math.PI * 0.01, 1e-12, 'basal area uses tree diameter');

const columns = [
  'row_id', S.COL_SPECIES, S.COL_D_CM, S.COL_H_M, S.COL_V_M3,
];
const marked = [
  [1, 'Abete', 18, 12, 0.1],
  [2, 'Abete', 22, 15, 0.2],
  [3, 'Faggio', 30, null, null],
];
const rows = aggregateTreeDendrometry(marked, columns, {
  allSpeciesNames: ['Abete', 'Castagno', 'Faggio'],
});
assertEqual(rows.map(row => [row.species, row.diameterClassCm, row.treeCount]),
            [['Abete', 20, 2], ['Faggio', 30, 1]],
            'unknown-height trees remain in diameter-class counts');
const shiftedRows = aggregateTreeDendrometry(marked, columns, {
  diameterClassMode: DIAMETER_CLASS_SHIFTED_UP,
});
assertEqual(shiftedRows.map(row => [row.species, row.diameterClassCm, row.treeCount]),
            [['Abete', 15, 1], ['Abete', 20, 1], ['Faggio', 30, 1]],
            'tree aggregation honors digest mode metadata');
assertEqual(rows.map(row => row.volumeM3), [0.3, 0], 'null mark volume contributes zero');
assertClose(rows[0].basalAreaM2, basalAreaM2(18) + basalAreaM2(22), 1e-6,
            'basal area sums individual diameters');
assertClose(rows[1].basalAreaM2, basalAreaM2(30), 1e-6,
            'unknown-height trees remain in basal-area totals');
assertEqual(rows.map(row => row.color),
            [dendrometrySpeciesColor(0), dendrometrySpeciesColor(2)],
            'colors stay stable in the full species universe');

const matrix = dendrometryMetricMatrix(rows, 'treeCount');
assertEqual(matrix.diameterClasses, [20, 25, 30],
            'matrix fills intermediate five-centimetre classes');
assertEqual(matrix.species.map(row => row.values), [[2, 0, 0], [0, 0, 1]],
            'matrix has species rows and class columns');

const chart = dendrometryBarChartData(rows, 'volumeM3', S.COL_VOLUME_M3);
assertEqual(chart.labels, ['20', '25', '30'], 'chart reuses matrix classes');
assertEqual(chart.datasets.map(dataset => dataset.data), [[0.3, 0, 0], [0, 0, 0]],
            'chart reuses matrix values');
assertEqual(chart.legend, false, 'chart appearance matches Bosco');
assertEqual(dendrometrySummaryLines(rows), {
  treeCount: ['Alberi totali: 3'],
  volume: ['Volume totale: 0,30 m³'],
  basalArea: ['Area totale: 0,13 m²', 'Diametro medio: 23,3, σ=4,7'],
}, 'chart summaries match Bosco totals and stacked basal-area lines');
const zeroCountRow = {
  ...rows[0], speciesId: 'zero', species: 'Specie zero', treeCount: 0,
};
assertEqual(dendrometryLegendItems([...rows, zeroCountRow]).map(item => item.name),
            ['Abete', 'Faggio'], 'legend omits species with zero total tree count');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
