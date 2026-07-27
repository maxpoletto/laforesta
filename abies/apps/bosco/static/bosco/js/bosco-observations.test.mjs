// Tests for Bosco observation helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-observations-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const O = await import(staticModule('bosco/js/bosco-observations.js'));
const S = await import(staticModule('base/js/strings.js'));
const {
  COLUMNS, FIELD_CATEGORIES, FIELD_CATEGORY_IDS, FIELD_ID, FIELD_NAME,
  FIELD_PHOTO_COUNT, FIELD_REGION_ID, ROW_ID, ROWS, VERSION,
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

console.log('bosco-observations.js');

const digest = {
  [COLUMNS]: [ROW_ID, VERSION, FIELD_REGION_ID, FIELD_CATEGORY_IDS,
    S.COL_DATE, S.COL_TEXT, S.COL_LAT, S.COL_LON, S.CSV_COL_ACC_M,
    S.COL_OPERATOR, S.COL_OBSERVATION_CATEGORIES, FIELD_PHOTO_COUNT],
  [FIELD_CATEGORIES]: [
    { [FIELD_ID]: 13, [FIELD_NAME]: 'incendio' },
    { [FIELD_ID]: 12, [FIELD_NAME]: 'fitosanitario' },
    { [FIELD_ID]: 11, [FIELD_NAME]: 'rifiuti' },
    { [FIELD_ID]: 10, [FIELD_NAME]: 'viabilità' },
  ],
  [ROWS]: [
    [1, 1, 1, [10, 11], '2026-07-25', 'Frana sul sentiero', 38.5, 16.3, 4,
      'Mario', 'viabilità, rifiuti', 2],
    [2, 1, 2, [12], '2025-05-10', 'Chioma secca', 38.7, 16.6, '', '',
      'fitosanitario', 0],
    [3, 1, 1, [10], '2026-01-03', 'Fuori particella', 39.0, 17.0, 7,
      'Mario', 'viabilità', 0],
    [4, 1, null, [11], '2026-01-04', 'Storica in particella', 38.4, 16.2,
      '', '', 'rifiuti', 0],
    [5, 1, 1, [], 'bad', 'Ignorata', '', 16.9, '', '', '', 0],
  ],
};

const observations = O.buildObservations(digest);
assertEqual(observations.length, 4, 'buildObservations: ignores invalid coords');
assertEqual({
  id: observations[0].id,
  regionId: observations[0].regionId,
  categoryIds: observations[0].categoryIds,
  categoryNames: observations[0].categoryNames,
  year: observations[0].year,
  text: observations[0].text,
  accM: observations[0].accM,
  operator: observations[0].operator,
  photoCount: observations[0].photoCount,
}, {
  id: 1,
  regionId: 1,
  categoryIds: [10, 11],
  categoryNames: ['viabilità', 'rifiuti'],
  year: 2026,
  text: 'Frana sul sentiero',
  accM: 4,
  operator: 'Mario',
  photoCount: 2,
}, 'buildObservations: row object');

const features = [
  {
    type: 'Feature',
    properties: { layer: 'Capistrano', name: 'Capistrano-1' },
    geometry: { type: 'Polygon', coordinates: [[[16.0, 38.0], [16.4, 38.0], [16.4, 38.6], [16.0, 38.6], [16.0, 38.0]]] },
  },
  {
    type: 'Feature',
    properties: { layer: 'Serra', name: 'Serra-2' },
    geometry: { type: 'Polygon', coordinates: [[[16.5, 38.6], [16.8, 38.6], [16.8, 38.8], [16.5, 38.8], [16.5, 38.6]]] },
  },
];
const attributed = O.attributeObservationParcels(observations, features);
assertEqual(attributed.map(o => [o.region, o.parcel]), [
  ['Capistrano', '1'], ['Serra', '2'], ['', ''], ['Capistrano', '1'],
], 'attributeObservationParcels: region/parcel from geometry');

const categories = O.buildObservationCategories(digest);
assertEqual(categories, [
  { id: 13, name: 'incendio' },
  { id: 12, name: 'fitosanitario' },
  { id: 11, name: 'rifiuti' },
  { id: 10, name: 'viabilità' },
], 'buildObservationCategories: active categories from digest metadata');
assertEqual(O.observationCategoryItems(attributed, categories), [
  { id: 12, name: 'fitosanitario', count: 1 },
  { id: 13, name: 'incendio', count: 0 },
  { id: 11, name: 'rifiuti', count: 2 },
  { id: 10, name: 'viabilità', count: 2 },
], 'observationCategoryItems: sorted counts including zero categories');
assertEqual(O.observationYears(attributed), [2025, 2026], 'observationYears: sorted years');
assertEqual(O.observationYears([{ year: 2024 }, { year: 2026 }]),
            [2024, 2025, 2026],
            'observationYears: includes years between first and last observation');
assertEqual(O.normalizeObservationYearRange(null, null, [2025, 2026]), { from: 2025, to: 2026 },
            'normalizeObservationYearRange: defaults');
assertEqual(O.normalizeObservationYearRange(2026, 2025, [2025, 2026]), { from: 2025, to: 2026 },
            'normalizeObservationYearRange: swaps reversed range');
assertEqual(O.filterObservations(attributed, {
  regionId: 1, region: 'Capistrano', categoryIds: [10], yearFrom: 2026, yearTo: 2026,
}).map(o => o.id), [1, 3],
            'filterObservations: region ID includes outside-parcel observations');
assertEqual(O.filterObservations(attributed, {
  regionId: 1, region: 'Capistrano', categoryIds: [11], yearFrom: 2026, yearTo: 2026,
}).map(o => o.id), [1, 4],
            'filterObservations: legacy rows can fall back to geometry region');
assertEqual(O.filterObservations(attributed, {
  regionId: 2, region: 'Serra', categoryIds: [10], yearFrom: 2026, yearTo: 2026,
}).map(o => o.id), [],
            'filterObservations: explicit region ID overrides geometry region');
assertEqual(O.filterObservations(attributed, { categoryIds: [] }).map(o => o.id), [],
            'filterObservations: explicit empty categories means none');

console.log(`
${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
