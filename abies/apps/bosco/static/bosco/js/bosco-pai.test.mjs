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
const { COLUMNS, ROWS, ROW_ID, VERSION } = await import(staticModule('base/js/constants.js'));

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
  [COLUMNS]: [ROW_ID, VERSION, S.COL_PARCEL_ID, S.COL_SPECIES_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_SPECIES, S.COL_YEAR,
    S.COL_LAT, S.COL_LON, S.COL_NOTE],
  [ROWS]: [
    [1, 1, 10, 5, 'Capistrano', '2', 'Abete', 1920, 38.1, 16.1, ''],
    [2, 1, 10, 6, 'Capistrano', '2', 'Faggio', 1930, 38.2, 16.2, ''],
    [3, 1, 11, 5, 'Capistrano', '10', 'Abete', 1940, 38.3, 16.3, ''],
    [4, 1, 12, 7, 'Serra', '1', 'Cerro', 1950, 38.4, 16.4, ''],
    [5, 1, 12, 7, 'Serra', '1', 'Cerro', 1950, '', 16.4, 'ignored'],
  ],
};

const trees = P.buildPreservedTrees(digest);
assertEqual(trees.length, 4, 'buildPreservedTrees: ignores invalid coordinates');
assertEqual(trees[0], {
  id: 1, version: 1, parcelId: 10, speciesId: 5, region: 'Capistrano',
  parcel: '2', species: 'Abete', year: 1920, lat: 38.1, lon: 16.1, note: '',
}, 'buildPreservedTrees: row object');

assertEqual(P.filterPaiTrees(trees, { region: 'Capistrano' }).map(t => t.id), [1, 2, 3],
            'filterPaiTrees: region');
assertEqual(P.filterPaiTrees(trees, { region: 'Capistrano', parcelIds: [10], speciesIds: [6] }).map(t => t.id), [2],
            'filterPaiTrees: parcel and species');
assertEqual(P.filterPaiTrees(trees, { region: 'Capistrano', parcelIds: [] }), [],
            'filterPaiTrees: explicit empty parcel list means none');

const parcels = P.paiParcelItems([{ id: 11, parcel: '10' }, { id: 10, parcel: '2' }], trees);
assertEqual(parcels, [{ id: 10, name: '2', count: 2 }, { id: 11, name: '10', count: 1 }],
            'paiParcelItems: numeric sort and counts');

const species = P.paiSpeciesItems(P.filterPaiTrees(trees, { region: 'Capistrano' }));
assertEqual(species, [{ id: 5, name: 'Abete', count: 2 }, { id: 6, name: 'Faggio', count: 1 }],
            'paiSpeciesItems: sorted counts');
const colors = P.speciesColorMap(species);
assertEqual([colors.get(5), colors.get(6)], ['#2e7d32', '#1565c0'],
            'speciesColorMap: stable palette');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
