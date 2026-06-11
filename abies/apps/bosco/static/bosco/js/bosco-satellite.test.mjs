// Tests for Bosco satellite-timeseries helpers.

import {
  availableMonths, characteristicSatelliteLayer, dateFromMonthValue, dateParam,
  diffColor, divergingDomain, evolutionMetricId, monthValue, normalizeDateParam,
  pickDate, satelliteColor, satelliteDiffPngUrl, satelliteDiffValue, satelliteValue,
} from './bosco-satellite.js';

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

function assertMatch(actual, pattern, msg) {
  if (pattern.test(actual)) passed++;
  else { failed++; console.error(`FAIL ${msg}: ${actual}`); }
}

console.log('bosco-satellite.js');

assertEqual(evolutionMetricId('2'), '2', 'evolutionMetricId: valid');
assertEqual(evolutionMetricId('8'), '1', 'evolutionMetricId: fallback');
assertEqual(characteristicSatelliteLayer('6'), 'ndvi', 'characteristicSatelliteLayer: NDVI');
assertEqual(characteristicSatelliteLayer('5'), null, 'characteristicSatelliteLayer: non-satellite');

assertEqual(normalizeDateParam('20240102'), '2024-01-02', 'normalizeDateParam: compact');
assertEqual(normalizeDateParam('2024-05'), '2024-05-01', 'normalizeDateParam: month');
assertEqual(normalizeDateParam('2024-13-01'), '', 'normalizeDateParam: invalid month');
assertEqual(dateParam('2024-05'), '20240501', 'dateParam: URL encoding');
assertEqual(monthValue('20240502'), '2024-05', 'monthValue: compact input');
assertEqual(dateFromMonthValue('2024-06'), '2024-06-01', 'dateFromMonthValue: first day');

const dates = ['2024-01-15', '2024-07-03', '2025-01-20'];
assertEqual(pickDate(dates, null, 'earliest'), '2024-01-15', 'pickDate: earliest fallback');
assertEqual(pickDate(dates, null, 'latest'), '2025-01-20', 'pickDate: latest fallback');
assertEqual(pickDate(dates, '20240701'), '2024-07-03', 'pickDate: same month');
assertEqual(pickDate(dates, '2024-10-01'), '2024-07-03', 'pickDate: nearest date');
assertEqual(availableMonths(['2024-01-15', '2024-01-20', '2024-07-03']),
            ['2024-01', '2024-07'], 'availableMonths: unique months');

const timeseries = {
  dates,
  means: {
    parcels: {
      'Capistrano-1': { ndvi: [0.2, 0.5, 0.1], ndmi: [0.1, null, 0.4] },
      'Capistrano-2': { ndvi: [null, 0.7, 0.9] },
    },
  },
};
assertEqual(satelliteValue(timeseries, 'Capistrano-1', 'ndvi', '2024-07'), 0.5,
            'satelliteValue: month-matched value');
assertEqual(satelliteValue(timeseries, 'Capistrano-2', 'ndvi', '2024-01-15'), null,
            'satelliteValue: null stays null');
assertEqual(satelliteDiffValue(timeseries, 'Capistrano-1', 'ndvi', '2024-01-15', '2024-07-03'), 0.3,
            'satelliteDiffValue: v2 minus v1');
assertEqual(satelliteDiffPngUrl(7, 'ndvi', '2024-01-15', '2024-07'),
            '/api/bosco/satellite/7/diff/ndvi/2024-01-15/2024-07-01.png',
            'satelliteDiffPngUrl: normalized endpoint');
assertEqual(satelliteDiffValue(timeseries, 'Capistrano-1', 'ndmi', '2024-01-15', '2024-07-03'), null,
            'satelliteDiffValue: missing endpoint');
assertEqual(divergingDomain([null, -0.2, 0.1]), { min: -0.2, max: 0.1, maxAbs: 0.2 },
            'divergingDomain: finite max abs');
assertEqual(divergingDomain([null, undefined]), null, 'divergingDomain: no data');

assertMatch(satelliteColor(0), /^rgb\(/, 'satelliteColor: css rgb');
assertMatch(diffColor(-0.2, 0.4), /^rgb\(/, 'diffColor: css rgb');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
