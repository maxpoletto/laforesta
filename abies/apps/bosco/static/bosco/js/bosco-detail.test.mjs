// Tests for Bosco detail overlay digest helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-detail-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const D = await import(staticModule('bosco/js/bosco-detail.js'));
const S = await import(staticModule('base/js/strings.js'));
const {
  COL_PARCEL_ID, COL_SPECIES_ID, COL_SURVEY_ID, COL_TREE_ID, COLUMNS, ROWS, ROW_ID,
} = await import(staticModule('base/js/constants.js'));

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

function assertClose(actual, expected, tolerance, msg) {
  const diff = Math.abs(actual - expected);
  if (diff <= tolerance) passed++;
  else { failed++; console.error(`FAIL ${msg}: expected ${expected}, got ${actual}`); }
}

console.log('bosco-detail.js');

const dendro = {
  [COLUMNS]: [ROW_ID, COL_PARCEL_ID, COL_SURVEY_ID, COL_SPECIES_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_SURVEY, S.COL_SAMPLE_AREA_HA, S.COL_SPECIES,
    S.COL_DIAM_CLASS_CM, S.COL_N_TREES, S.COL_VOLUME_M3,
    S.COL_BASAL_AREA_M2, S.COL_AVG_H_M, S.COL_INCREMENT_PCT],
  [ROWS]: [
    [1, 10, 1, 5, 'Capistrano', '1', 'R1', 0.5, 'Abete', 20, 2, 0.3, 0.06, 21, 1.0],
    [2, 10, 1, 5, 'Capistrano', '1', 'R1', 0.5, 'Abete', 20, 3, 0.6, 0.09, 24, 2.0],
    [3, 11, 1, 6, 'Capistrano', '2', 'R1', 0.25, 'Faggio', 30, 4, 1.0, 0.2, 18, null],
    [4, 12, 1, 5, 'Serra', '1', 'R1', 0.75, 'Abete', 20, 7, 9.0, 9.0, 30, 9.0],
  ],
};

const allSpeciesNames = ['Abete', 'Castagno', 'Faggio'];

const navEntries = [{ id: 1 }, { id: 2 }, { id: 3 }];
let nav = D.parcelNavigation(navEntries, 1);
assertEqual([nav.previous?.id || null, nav.next?.id || null], [null, 2],
            'parcelNavigation: first parcel');
nav = D.parcelNavigation(navEntries, 2);
assertEqual([nav.previous?.id || null, nav.next?.id || null], [1, 3],
            'parcelNavigation: middle parcel');
nav = D.parcelNavigation(navEntries, 3);
assertEqual([nav.previous?.id || null, nav.next?.id || null], [2, null],
            'parcelNavigation: last parcel');
nav = D.parcelNavigation(navEntries, 99);
assertEqual([nav.previous?.id || null, nav.next?.id || null], [null, null],
            'parcelNavigation: missing parcel');

const points = {
  [COLUMNS]: [ROW_ID, COL_PARCEL_ID, COL_SURVEY_ID, COL_TREE_ID,
    COL_SPECIES_ID, S.COL_REGION, S.COL_PARCEL, S.COL_SURVEY,
    S.COL_SPECIES, S.COL_D_CM, S.COL_H_M],
  [ROWS]: [
    [1, 10, 1, 100, 5, 'Capistrano', '1', 'R1', 'Abete', 18, 20.5],
    [2, 10, 1, 101, 5, 'Capistrano', '1', 'R1', 'Abete', 22, 22.5],
    [3, 11, 1, 102, 6, 'Capistrano', '2', 'R1', 'Faggio', 25, 18],
    [4, 12, 1, 103, 5, 'Serra', '1', 'R1', 'Abete', 35, 30],
  ],
};

let rows = D.aggregateDendrometry(dendro, { parcelId: 10 }, { perHa: false });
assertEqual(rows.length, 1, 'aggregateDendrometry: parcel groups matching rows');
assertEqual(rows[0].treeCount, 5, 'aggregateDendrometry: sums tree count');
assertEqual(rows[0].volumeM3, 0.9, 'aggregateDendrometry: sums volume');
assertEqual(rows[0].basalAreaM2, 0.15, 'aggregateDendrometry: sums basal area');
assertClose(rows[0].avgHeightM, 22.8, 0.0001, 'aggregateDendrometry: weighted height');
assertClose(rows[0].incrementPct, 1.6, 0.0001, 'aggregateDendrometry: weighted increment');

rows = D.aggregateDendrometry(dendro, { region: 'Capistrano' }, { areaHa: 10, perHa: true });
assertEqual(rows.map(r => `${r.species}:${r.diameterClassCm}`), ['Abete:20', 'Faggio:30'],
            'aggregateDendrometry: region sorted species/classes');
assertEqual(rows[0].treeCount, 6.6667, 'aggregateDendrometry: per-ha tree count');
assertEqual(rows[1].volumeM3, 1.3333, 'aggregateDendrometry: per-ha volume');

rows = D.aggregateDendrometry(dendro, { region: 'Capistrano' }, { areaHa: 10, perHa: false });
assertEqual(rows[0].treeCount, 66.6667, 'aggregateDendrometry: expands tree count to scope area');
assertEqual(rows[1].basalAreaM2, 2.666667, 'aggregateDendrometry: expands basal area to scope area');

rows = D.aggregateDendrometry(
  dendro, { region: 'Capistrano' }, { perHa: false, speciesIds: [6], allSpeciesNames },
);
assertEqual(rows.map(r => r.species), ['Faggio'], 'aggregateDendrometry: species filter');
assertEqual(rows[0].color, D.dendrometrySpeciesColor(2),
            'aggregateDendrometry: filtered species keeps full-universe color');

rows = D.aggregateDendrometry(dendro, { region: 'Capistrano' }, { perHa: false, speciesIds: [] });
assertEqual(rows, [], 'aggregateDendrometry: explicit empty species filter');

const species = D.dendrometrySpecies(dendro, { region: 'Capistrano' }, { allSpeciesNames });
assertEqual(species.map(({ id, name, count }) => ({ id, name, count })),
            [{ id: 5, name: 'Abete', count: 5 }, { id: 6, name: 'Faggio', count: 4 }],
            'dendrometrySpecies: counts by species');
assertEqual(species.map(item => item.color),
            [D.dendrometrySpeciesColor(0), D.dendrometrySpeciesColor(2)],
            'dendrometrySpecies: stable full-universe colors');

rows = D.aggregateDendrometry(dendro, { region: 'Capistrano' }, { perHa: false, allSpeciesNames });
let chart = D.dendrometryBarChartData(rows, 'treeCount', 'Tree count');
assertEqual(chart.labels, ['20', '25', '30'], 'dendrometryBarChartData: zero-filled diameter labels');
assertEqual(chart.datasets.map(d => d.label), ['Abete', 'Faggio'],
            'dendrometryBarChartData: species datasets');
assertEqual(chart.datasets[0].data, [5, 0, 0], 'dendrometryBarChartData: sparse series');
assertEqual(chart.datasets[1].data, [0, 0, 4], 'dendrometryBarChartData: second sparse series');
assertEqual(chart.datasets.map(d => d.backgroundColor),
            [D.dendrometrySpeciesColor(0), D.dendrometrySpeciesColor(2)],
            'dendrometryBarChartData: full-universe species colors');
assertEqual(chart.legend, false, 'dendrometryBarChartData: hides repeated legend');
assertEqual(chart.yTitle, 'Tree count', 'dendrometryBarChartData: y title');

chart = D.dendrometryLineChartData(rows, 'incrementPct', S.COL_INCREMENT_PCT);
assertEqual(chart.datasets[0].data, [1.6, null, null], 'dendrometryLineChartData: line values with gaps');
assertEqual(chart.datasets[0].spanGaps, true, 'dendrometryLineChartData: spans gaps');
assertEqual(chart.legend, false, 'dendrometryLineChartData: hides repeated legend');
assertEqual(D.dendrometryTreeTotal(rows), 9, 'dendrometryTreeTotal: raw tree count');
assertEqual(
  D.dendrometryTreeStatusLabel(rows, rows, { perHa: false }),
  'Alberi totali: 9',
  'dendrometryTreeStatusLabel: total trees label',
);
const perHaRows = D.aggregateDendrometry(dendro, { region: 'Capistrano' }, { areaHa: 10, perHa: true });
assertClose(D.dendrometryTreeSum(perHaRows), 12, 0.0001, 'dendrometryTreeSum: per-ha tree sum');
assertEqual(
  D.dendrometryTreeStatusLabel(perHaRows, rows, { perHa: true, formatPerHa: n => n.toFixed(1) }),
  'Alberi per ettaro: 12.0',
  'dendrometryTreeStatusLabel: per-ha trees label',
);

let heightPoints = D.dendrometryHeightPoints(points, { parcelId: 10 }, { speciesIds: [5] });
assertEqual(heightPoints.map(p => [p.species, p.dCm, p.hM]),
            [['Abete', 18, 20.5], ['Abete', 22, 22.5]],
            'dendrometryHeightPoints: parcel and species filter');
heightPoints = D.dendrometryHeightPoints(points, { region: 'Capistrano' }, { speciesIds: [] });
assertEqual(heightPoints, [], 'dendrometryHeightPoints: explicit empty species filter');
heightPoints = D.dendrometryHeightPoints(points, { region: 'Capistrano' }, { allSpeciesNames });
chart = D.dendrometryScatterChartData(heightPoints, S.COL_H_M);
assertEqual(chart.datasets.map(d => d.label), ['Abete', 'Faggio'],
            'dendrometryScatterChartData: species datasets');
assertEqual(chart.datasets[0].data, [{ x: 18, y: 20.5 }, { x: 22, y: 22.5 }],
            'dendrometryScatterChartData: scatter points');
assertEqual(chart.datasets[1].backgroundColor, D.dendrometrySpeciesColor(2),
            'dendrometryScatterChartData: full-universe species colors');
assertEqual(chart.legend, false, 'dendrometryScatterChartData: hides repeated legend');
assertEqual(chart.yTitle, S.COL_H_M, 'dendrometryScatterChartData: y title');

const fitPoints = [10, 20, 30].map(d => ({ speciesId: 7, species: 'Fit', dCm: d, hM: 2 * Math.log(d) + 1 }));
chart = D.dendrometryScatterChartData(fitPoints, S.COL_H_M, { minFitN: 3 });
assertEqual(chart.datasets.length, 2, 'dendrometryScatterChartData: adds fit dataset');
const fitDataset = chart.datasets[1];
assertEqual(fitDataset.type, 'line', 'dendrometryScatterChartData: fit is a line');
assertClose(fitDataset.fit.a, 2, 0.000001, 'dendrometryScatterChartData: fit a');
assertClose(fitDataset.fit.b, 1, 0.000001, 'dendrometryScatterChartData: fit b');
assertClose(fitDataset.fit.r2, 1, 0.000001, 'dendrometryScatterChartData: fit r2');
assertEqual(fitDataset.fit.n, 3, 'dendrometryScatterChartData: fit n');
assertEqual(fitDataset.data[0].x, 10, 'dendrometryScatterChartData: fit starts at min d');
assertEqual(fitDataset.data.at(-1).x, 30, 'dendrometryScatterChartData: fit ends at max d');

const meta = D.regionMetadata([
  { displayAreaHa: 10, cadastralAreaHa: 11, aveAge: 40, altMin: 700, altMax: 900, type: S.TYPE_HIGHFOREST },
  { displayAreaHa: 5, cadastralAreaHa: 6, aveAge: 70, altMin: 600, altMax: 800, type: S.TYPE_COPPICE },
]);
assertEqual(meta.count, 2, 'regionMetadata: count');
assertEqual(meta.areaHa, 15, 'regionMetadata: area sum');
assertEqual(meta.cadastralAreaHa, 17, 'regionMetadata: cadastral area sum');
assertEqual('aveAge' in meta, false, 'regionMetadata: omits region age');
assertEqual(meta.altMin, 600, 'regionMetadata: min altitude');
assertEqual(meta.altMax, 900, 'regionMetadata: max altitude');
assertEqual([...meta.typeCounts.entries()], [[S.TYPE_HIGHFOREST, 1], [S.TYPE_COPPICE, 1]], 'regionMetadata: types');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
