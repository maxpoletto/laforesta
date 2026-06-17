// Tests for Bosco PAI helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-pai-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const P = await import(staticModule('bosco/js/bosco-pai.js'));
const S = await import(staticModule('base/js/strings.js'));
const {
  COL_PARCEL_ID, COL_SPECIES_ID, COL_TREE_ID, COLUMNS, ROWS, ROW_ID, VERSION,
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

console.log('bosco-pai.js');

const digest = {
  [COLUMNS]: [ROW_ID, VERSION, COL_TREE_ID, COL_PARCEL_ID, COL_SPECIES_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_SPECIES, S.COL_NUMBER, S.COL_DATE,
    S.COL_ESTIMATED_BIRTH_YEAR, S.COL_D_CM, S.COL_H_M, S.COL_H_MEASURED,
    S.COL_LAT, S.COL_LON, S.COL_NOTE],
  [ROWS]: [
    [1, 1, 101, 10, 5, 'Capistrano', '2', 'Abete', 7, '2024-09-15', 1920, 42, 18.5, true, 38.1, 16.1, ''],
    [2, 1, 102, 10, 6, 'Capistrano', '2', 'Faggio', 8, '', 1930, '', '', false, 38.2, 16.2, ''],
    [3, 1, 103, 11, 5, 'Capistrano', '10', 'Abete', 1, '', 1940, '', '', false, 38.3, 16.3, ''],
    [4, 1, 104, 12, 7, 'Serra', '1', 'Cerro', 2, '', 1950, '', '', false, 38.4, 16.4, ''],
    [5, 1, 105, 12, 7, 'Serra', '1', 'Cerro', 3, '', 1950, '', '', false, '', 16.4, 'ignored'],
  ],
};

const trees = P.buildPreservedTrees(digest);
assertEqual(trees.length, 4, 'buildPreservedTrees: ignores invalid coordinates');
assertEqual(trees[0], {
  id: 1, version: 1, treeId: 101, parcelId: 10, speciesId: 5,
  region: 'Capistrano', parcel: '2', species: 'Abete', number: 7,
  date: '2024-09-15', estimatedBirthYear: 1920, dCm: 42, hM: 18.5,
  hMeasured: true, lat: 38.1, lon: 16.1, note: '',
}, 'buildPreservedTrees: row object');

assertEqual(P.filterPaiTrees(trees, { region: 'Capistrano' }).map(t => t.id), [1, 2, 3],
            'filterPaiTrees: region');
assertEqual(P.filterPaiTrees(trees, { region: 'Capistrano', parcelIds: [10], speciesIds: [6] }).map(t => t.id), [2],
            'filterPaiTrees: parcel and species');
assertEqual(P.filterPaiTrees(trees, { region: 'Capistrano', parcelIds: [] }), [],
            'filterPaiTrees: explicit empty parcel list means none');

const parcels = P.paiParcelItems([
  { id: 11, parcel: '10' },
  { id: 13, parcel: '10a' },
  { id: 9, parcel: '1' },
  { id: 10, parcel: '2' },
], trees);
assertEqual(parcels, [
  { id: 9, name: '1', count: 0 },
  { id: 10, name: '2', count: 2 },
  { id: 11, name: '10', count: 1 },
  { id: 13, name: '10a', count: 0 },
], 'paiParcelItems: natural sort and counts');

const species = P.paiSpeciesItems(P.filterPaiTrees(trees, { region: 'Capistrano' }));
assertEqual(species, [{ id: 5, name: 'Abete', count: 2 }, { id: 6, name: 'Faggio', count: 1 }],
            'paiSpeciesItems: sorted counts');
const speciesWithOther = P.paiSpeciesItems([
  { speciesId: 999, species: S.CHART_OTHER },
  { speciesId: 8, species: 'Acero' },
  { speciesId: 5, species: 'Abete' },
]);
assertEqual(speciesWithOther, [
  { id: 5, name: 'Abete', count: 1 },
  { id: 8, name: 'Acero', count: 1 },
  { id: 999, name: S.CHART_OTHER, count: 1 },
], 'paiSpeciesItems: literal Altro sorted last');
const colors = P.speciesColorMap(species, ['Abete', 'Castagno', 'Faggio']);
assertEqual([colors.get(5), colors.get(6)], ['#2e7d32', '#144b99'],
            'speciesColorMap: stable full-universe palette');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
