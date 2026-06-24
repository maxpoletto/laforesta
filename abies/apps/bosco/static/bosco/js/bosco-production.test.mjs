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
    [4, '2025-03-01', 'Capistrano', '1', 70],
    [5, '2027-04-01', 'Capistrano', '2', 40],
    [6, '2025-03-01', 'Serra', '1', 999],
    [7, '2027-06-01', 'Serra', '1', 1],
  ],
};

assertEqual(B.prelieviUrlForScope({ regionId: 10, parcelId: 101 }), '/prelievi?c=10&pa=101',
            'prelieviUrlForScope: parcel scope');
assertEqual(B.prelieviUrlForScope({ regionId: 10 }), '/prelievi?c=10',
            'prelieviUrlForScope: region scope');
assertEqual(B.prelieviUrlForScope({}), '/prelievi',
            'prelieviUrlForScope: empty scope');

let rows = B.productionRows(digest, { region: 'Capistrano', parcel: '1' });
assertEqual(rows.map(r => r[0]), [1, 2, 4], 'productionRows: parcel scope');
rows = B.productionRows(digest, { region: 'Capistrano' });
assertEqual(rows.map(r => r[0]), [1, 2, 3, 4, 5], 'productionRows: region scope');

assertEqual(B.productionYears(digest, { region: 'Capistrano' }), ['2024', '2025', '2027'],
            'productionYears: region years');
assertEqual(B.productionYears(digest), ['2024', '2025', '2027'],
            'productionYears: global years');
assertEqual(B.productionYearRange(digest), ['2024', '2025', '2026', '2027'],
            'productionYearRange: global continuous range');
assertEqual(B.productionYearRange(digest, { region: 'Capistrano' }),
            ['2024', '2025', '2026', '2027'],
            'productionYearRange: scoped continuous range');
assertEqual(B.productionMonths(digest, { region: 'Capistrano' }),
            ['2024-01', '2024-02', '2025-03', '2027-04'],
            'productionMonths: region months');
const globalMonths = B.productionMonthRange(digest);
assertEqual(globalMonths.length, 42, 'productionMonthRange: global continuous length');
assertEqual([globalMonths[0], globalMonths.at(-1)], ['2024-01', '2027-06'],
            'productionMonthRange: global endpoints');
assertEqual(B.pickProductionYear(['2024', '2025'], '20240101'), '2024',
            'pickProductionYear: compact date converted to year');
const delta = B.productionDeltaByParcel(digest, { region: 'Capistrano' }, '2024', '2027');
assertEqual(delta.get('Capistrano-1'), -150, 'productionDeltaByParcel: parcel 1 delta');
assertEqual(delta.get('Capistrano-2'), 15, 'productionDeltaByParcel: parcel 2 delta');

let agg = B.aggregateProduction(digest, { region: 'Capistrano' });
assertEqual(agg.rowCount, 5, 'aggregateProduction: row count');
assertEqual(agg.totalQuintals, 285, 'aggregateProduction: raw total');
assertEqual(agg.chartData.labels, ['2024', '2025', '2026', '2027'],
            'aggregateProduction: global yearly labels');
assertEqual(agg.chartData.datasets[0].data, [175, 70, 0, 40],
            'aggregateProduction: zero-filled yearly total');

agg = B.aggregateProduction(digest, { region: 'Capistrano', parcel: '1' }, {
  byMonth: true, perHa: true, areaHa: 10,
});
assertEqual(agg.chartData.labels.length, 15, 'aggregateProduction: monthly scoped label count');
assertEqual([agg.chartData.labels[0], agg.chartData.labels.at(-1)], ['2024-01', '2025-03'],
            'aggregateProduction: monthly scoped endpoints');
assertEqual(agg.chartData.yTitle, S.BOSCO_QUINTALS_PER_HA, 'aggregateProduction: per-ha y title');
assertEqual(agg.chartData.datasets[0].data[agg.chartData.labels.indexOf('2024-01')], 15,
            'aggregateProduction: per-ha first month value');
assertEqual(agg.chartData.datasets[0].data[agg.chartData.labels.indexOf('2024-02')], 0,
            'aggregateProduction: per-ha zero-filled month');
assertEqual(agg.chartData.datasets[0].data[agg.chartData.labels.indexOf('2025-03')], 7,
            'aggregateProduction: per-ha second month value');

console.log(`
${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
