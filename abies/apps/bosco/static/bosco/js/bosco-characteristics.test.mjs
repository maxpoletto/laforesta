// Tests for Bosco characteristic digest helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const B = await import(staticModule('bosco/js/bosco-characteristics.js'));
const S = await import(staticModule('base/js/strings.js'));
const {
  COL_COPPICE, COL_PARCEL_ID, COL_REGION_ID, COLUMNS, ROWS, ROW_ID, VERSION,
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

console.log('bosco-characteristics.js');

const parcels = {
  [COLUMNS]: [ROW_ID, VERSION, COL_REGION_ID, S.COL_REGION, S.COL_PARCEL,
    S.COL_CLASS, COL_COPPICE, S.COL_AREA_HA, S.COL_AREA_CAD_HA,
    S.COL_AVE_AGE, S.COL_LOCATION, S.COL_ALT_MIN, S.COL_ALT_MAX,
    S.COL_ASPECT, S.COL_GRADE_PCT, S.COL_TYPE,
    S.COL_DESC_VEG, S.COL_DESC_GEO],
  [ROWS]: [
    [10, 0, 1, 'Capistrano', '1', 'A', false, 11, 10, 70, '', 800, 900, '', 0, S.TYPE_HIGHFOREST, '', ''],
    [11, 0, 1, 'Capistrano', '2', 'F', true, 6, 5, null, '', null, null, '', 0, S.TYPE_COPPICE, '', ''],
  ],
};
const entries = B.buildParcelEntries(parcels);
assertEqual(entries[0].key, 'Capistrano-1', 'buildParcelEntries: key');
assertEqual(entries[0].altitudeMean, 850, 'buildParcelEntries: mean altitude');
assertEqual(entries[0].coppice, false, 'buildParcelEntries: highforest flag');
assertEqual(entries[1].coppice, true, 'buildParcelEntries: coppice flag');
assertEqual(entries[0].altMin, 800, 'buildParcelEntries: altitude min');
assertEqual(entries[0].altMax, 900, 'buildParcelEntries: altitude max');
assertEqual(entries[1].altitudeMean, null, 'buildParcelEntries: missing altitude');

const prelievi = {
  [COLUMNS]: [ROW_ID, S.COL_DATE, S.COL_REGION, S.COL_PARCEL, S.COL_QUINTALS],
  [ROWS]: [
    [1, '2024-01-01', 'Capistrano', '1', 100],
    [2, '2024-01-02', 'Capistrano', '1', 50],
    [3, '2024-01-03', 'Capistrano', '2', 20],
  ],
};
const historical = B.historicalHarvestByParcel(prelievi);
assertEqual(historical.get('Capistrano-1'), 150, 'historicalHarvestByParcel: sum');

const future = {
  [COLUMNS]: [ROW_ID, VERSION, S.COL_HARVEST_PLAN, COL_PARCEL_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_YEAR_PLANNED, S.COL_VOLUME_PLANNED],
  [ROWS]: [
    [1, 1, 99, 10, 'Capistrano', '1', 2026, 30],
    [2, 1, 99, 10, 'Capistrano', '1', 2027, 45],
  ],
};
const planned = B.futureHarvestByParcel(future);
assertEqual(planned.get(10), 75, 'futureHarvestByParcel: sum');

assertEqual(B.metricValue(entries[0], B.Q_AGE), 70, 'metricValue: age');
assertEqual(B.metricValue(entries[0], B.Q_TYPE), S.TYPE_HIGHFOREST, 'metricValue: type');
assertEqual(B.metricValue(entries[0], B.Q_ALTITUDE), 850, 'metricValue: altitude');
assertEqual(B.metricValue(entries[0], B.Q_HISTORICAL_HARVEST, { historical }), 150,
            'metricValue: historical harvest');
assertClose(B.metricValue({ ...entries[0], displayAreaHa: 15 }, B.Q_FUTURE_HARVEST,
                          { future: planned, perHa: true }), 5, 0.0001,
            'metricValue: future harvest per ha');
assertEqual(B.continuousDomain([null, 1, 4, undefined]), { min: 1, max: 4 },
            'continuousDomain: clean range');
assertEqual(B.normalized(2.5, { min: 1, max: 4 }), 0.5, 'normalized: range');
assertEqual(B.normalized(4, { min: 4, max: 4 }), 0.5, 'normalized: flat domain');
assertEqual(B.isHarvestMetric('4'), true, 'isHarvestMetric: historical harvest');
assertEqual(B.isHarvestMetric(5), true, 'isHarvestMetric: numeric future harvest');
assertEqual(B.isHarvestMetric('1'), false, 'isHarvestMetric: non-harvest');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
