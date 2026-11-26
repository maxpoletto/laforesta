import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';
// Tests for Bosco historical-production helpers.

import fs from 'node:fs';
import { createRequire } from 'node:module';
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
copyGeodesicVendor(staticRoot);
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const B = await import(staticModule('bosco/js/bosco-production.js'));
const { wireProductionReport } = await import(staticModule('bosco/js/bosco-production-report.js'));
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

const parcelLink = new URL(B.prelieviUrlForScope({
  regionId: 10, region: 'Capistrano', parcelId: 101, entry: { parcel: '1' },
}), 'http://localhost');
assertEqual(parcelLink.searchParams.get('f'), 'compresa:Capistrano particella:1',
            'prelieviUrlForScope: parcel scope uses the visible search filter');
assertEqual([...parcelLink.searchParams.keys()], ['f'],
            'prelieviUrlForScope: parcel link has no hidden ID filters');
const regionLink = new URL(B.prelieviUrlForScope({
  regionId: 10, region: 'Serra San Bruno',
}), 'http://localhost');
assertEqual(regionLink.searchParams.get('f'), 'compresa:"Serra San Bruno"',
            'prelieviUrlForScope: region scope quotes multi-word names');
const specialLink = new URL(B.prelieviUrlForScope({
  region: 'Nord & Sud', entry: { parcel: '1 b' },
}), 'http://localhost');
assertEqual(specialLink.searchParams.get('f'),
            'compresa:"Nord & Sud" particella:"1 b"',
            'prelieviUrlForScope: search terms survive URL encoding');
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

const prelievo = (parcelId, month, volume, attrs = {}) => ({
  parcel_id: parcelId, parcel: String(parcelId), coppice: true, parcel_area_ha: 10,
  worksite_id: null, intervention_area_ha: null, month, volume_m3: volume, mass_q: volume * 8,
  ...attrs,
});
const winterWorksite = { worksite_id: 3, intervention_area_ha: 4 };
const regionalRows = [
  prelievo(1, '2025-01', 30, winterWorksite),
  prelievo(1, '2024-12', 50, winterWorksite),
  prelievo(2, '2024-11', 150, { parcel_area_ha: 20 }),
  prelievo(2, '2024-12', 250, { parcel_area_ha: 20 }),
  prelievo(3, '2024-10', 20, { parcel_area_ha: null }),
  prelievo(4, '2026-01', 100, { coppice: false }),
];
const report = B.regionalYield(regionalRows, { governance: 'coppice', from: 2024, to: 2024 });
assertEqual(report.rows.map(row => [row.parcel, row.year, row.areaHa, row.volumeM3, row.areaSource]), [
  ['1', 2024, 4, 80, 'intervention'], ['2', 2024, 20, 400, 'parcel'],
  ['3', 2024, null, 20, null],
], 'yield counts the full winter intervention once and deduplicates historical parcel area');
assertEqual([report.volumeM3, report.areaHa, report.parcelCount, report.yearCount, report.unknownAreaCount],
  [480, 24, 2, 1, 1], 'yield uses the same measurable rows in its numerator and denominator');
assertEqual(B.regionalYield(regionalRows, { from: 2025, to: 2025 }).rows, [],
  'the later half of a cross-year intervention does not make a second yield entry');
const dated = B.regionalProductionChart(regionalRows, { governance: 'coppice', from: 2024, to: 2026 });
assertEqual(dated.chartData.labels, ['2024', '2025', '2026'], 'regional history includes inactive years');
assertEqual(dated.chartData.datasets[0].data, [3760, 240, 0],
  'the chart retains actual harvest dates even when the yield uses whole interventions');
const mixed = B.regionalYield([
  prelievo(1, '2024-09', 70, winterWorksite), prelievo(1, '2024-10', 30),
]);
assertEqual([mixed.areaHa, mixed.volumeM3], [10, 100],
  'an unknown cut in a parcel-year uses the full parcel only once');

const reportHost = installDOM();
const require = createRequire(import.meta.url);
globalThis.window = {
  SortableTable: require('../../../../base/static/vendor/sortable-table/sortable-table.js'),
};
const measure = (tag, key, value) => {
  const node = new Element(tag);
  node.dataset[key] = value;
  reportHost.appendChild(node);
  return node;
};
const governance = measure('select', 'role', 'production-governance');
const yearFrom = measure('select', 'role', 'production-year-from');
measure('select', 'role', 'production-year-to');
const summary = measure('p', 'target', 'production-yield');
measure('p', 'target', 'production-yield-context');
const missing = measure('p', 'target', 'production-yield-missing');
const breakdown = measure('details', 'target', 'production-breakdown');
const tableHost = measure('div', 'target', 'production-breakdown-table');
const tableRows = () => tableHost.querySelectorAll('.sortable-table-row');
let selected;
const controller = wireProductionReport(reportHost, filters => { selected = filters; });
controller.render({ rows: regionalRows }, { governance: 'coppice', from: 2024, to: 2024 });
assertEqual(summary.textContent, 'Prelievo medio negli anni di taglio: 20,00 m³/ha',
  'report shows the area-weighted yield with the local decimal separator');
assertEqual(missing.hidden, false, 'report flags rows whose areas are unavailable');
breakdown.open = true;
await breakdown.dispatch('toggle');
assertEqual(tableRows()[0]?.children.map(cell => cell.textContent),
  ['2024–2025', '1', 'Ceduo', '80,00', '4,00', 'Superficie intervento'],
  'breakdown displays the parcel governo for a complete cross-year cut');
assertEqual(tableRows()[0]?.children[3].classList.contains('numeric'), true,
  'volume uses the shared numeric alignment');
assertEqual(tableRows()[2]?.children[4].textContent, '-', 'unknown area uses the standard placeholder');
let csv;
URL.createObjectURL = blob => { csv = blob; return 'blob:report'; };
URL.revokeObjectURL = () => {};
document.createElement = tag => Object.assign(new Element(tag), { click() {} });
await tableHost.querySelector('.btn-export')?.dispatch();
const [csvHeader, firstCsvRow] = (csv ? await csv.text() : '').split('\n');
assertEqual(csvHeader.split(';')[2], 'Governo', 'export labels the parcel governo column');
assertEqual(firstCsvRow, '2024–2025;1;Ceduo;80;4;Superficie intervento',
  'export writes the same parcel governo and filtered figures shown in the breakdown');
governance.value = 'highforest';
yearFrom.value = '2026';
await yearFrom.dispatch('change');
assertEqual(selected, { governance: 'highforest', from: 2026, to: null },
  'a new first year can move the end of the selected period forward');
controller.render({ rows: regionalRows }, selected);
assertEqual(tableRows()[0]?.children[2].textContent, 'Fustaia',
  'breakdown displays fustaia for high-forest parcels');

const manyRows = Array.from({ length: 18 }, (_, i) => prelievo(i + 1, '2024-05', 18 - i));
controller.render({ rows: manyRows }, { governance: '', from: null, to: null });
assertEqual(tableRows().length, 15, 'breakdown shows fifteen rows on its first page');
await tableHost.querySelectorAll('.sortable-table-header')[3]?.dispatch('click');
assertEqual(tableRows()[0]?.children[3].textContent, '1,00', 'volume sorts numerically');
await tableHost.querySelector('.btn-export')?.dispatch();
assertEqual(csv && (await csv.text()).split('\n').length, 19, 'export includes rows on later pages');

console.log(`
${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
