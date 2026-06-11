// Tests for Bosco historical-production helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-production-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const B = await import(staticModule('bosco/js/bosco-production.js'));
const S = await import(staticModule('base/js/strings.js'));
const { COLUMNS, ROWS, ROW_ID } = await import(staticModule('base/js/constants.js'));

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

console.log('bosco-production.js');

const digest = {
  [COLUMNS]: [ROW_ID, S.COL_DATE, S.COL_REGION, S.COL_PARCEL, S.COL_QUINTALS],
  [ROWS]: [
    [1, '2024-01-02', 'Capistrano', '1', 100],
    [2, '2024-01-15', 'Capistrano', '1', 50],
    [3, '2024-02-01', 'Capistrano', '2', 25],
    [4, '2025-03-01', 'Serra', '1', 999],
  ],
};

assertEqual(B.prelieviUrlForScope({ regionId: 10, parcelId: 101 }), '/prelievi?c=10&pa=101',
            'prelieviUrlForScope: parcel scope');
assertEqual(B.prelieviUrlForScope({ regionId: 10 }), '/prelievi?c=10',
            'prelieviUrlForScope: region scope');
assertEqual(B.prelieviUrlForScope({}), '/prelievi',
            'prelieviUrlForScope: empty scope');

let rows = B.productionRows(digest, { region: 'Capistrano', parcel: '1' });
assertEqual(rows.map(r => r[0]), [1, 2], 'productionRows: parcel scope');
rows = B.productionRows(digest, { region: 'Capistrano' });
assertEqual(rows.map(r => r[0]), [1, 2, 3], 'productionRows: region scope');

let agg = B.aggregateProduction(digest, { region: 'Capistrano' });
assertEqual(agg.rowCount, 3, 'aggregateProduction: row count');
assertEqual(agg.totalQuintals, 175, 'aggregateProduction: raw total');
assertEqual(agg.chartData.labels, ['2024'], 'aggregateProduction: yearly labels');
assertEqual(agg.chartData.datasets[0].data, [175], 'aggregateProduction: yearly total');

agg = B.aggregateProduction(digest, { region: 'Capistrano', parcel: '1' }, {
  byMonth: true, perHa: true, areaHa: 10,
});
assertEqual(agg.chartData.labels, ['2024-01'], 'aggregateProduction: monthly labels');
assertEqual(agg.chartData.yTitle, 'Q.li/ha', 'aggregateProduction: per-ha y title');
assertEqual(agg.chartData.datasets[0].data, [15], 'aggregateProduction: per-ha values');

console.log(`
${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
