import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-calendar-'));
for (const app of ['base', 'bosco']) {
  fs.cpSync(path.resolve(here, `../../../../${app}/static/${app}/js`),
    path.join(tmp, app, 'js'), { recursive: true });
}
copyGeodesicVendor(tmp);
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const moduleURL = rel => pathToFileURL(path.join(tmp, rel));
const C = await import(moduleURL('bosco/js/bosco-calendar.js'));
const M = await import(moduleURL('bosco/js/bosco-metrics.js'));
const { readBoscoParams } = await import(moduleURL('bosco/js/bosco-state.js'));
const data = {
  selected_plan: { id: 1, name: 'Plan', year_start: 2026, year_end: 2040 },
  calendar: [{ parcel_id: 10, years: [2030, 2028, 2030] },
    { parcel_id: 11, years: [2040] }],
};
const years = C.calendarYearsByParcel(data);
assert.deepEqual(years.get(10), [2028, 2030]);
assert.equal(years.has(999), false, 'unscheduled parcels have no fictitious year zero');
assert.equal(C.calendarColorPosition(2026, data.selected_plan), 0);
assert.equal(C.calendarColorPosition(2040, data.selected_plan), 1);
assert.equal(C.calendarColorPosition(2033, data.selected_plan), 0.5);
assert.equal(C.calendarColorPosition(2028, data.selected_plan), 2 / 14,
  'colors use the full plan range, not the first harvest in this compresa');
assert.equal(C.calendarColorPosition(null, data.selected_plan), null);
assert.equal(C.calendarColorPosition(2026, { year_start: 2026, year_end: 2026 }), 0);
assert.equal(C.calendarColorPosition(2026, null), null);
assert.equal(C.calendarYearsText(years.get(10)), '2028, 2030');
assert.ok(C.calendarYearsText([]));
assert.equal(M.usesPlanSource(M.Q_FUTURE_HARVEST), true);
assert.equal(M.usesPlanSource(M.Q_CUTTING_CALENDAR), true);
assert.equal(M.usesPlanSource(M.Q_AGE), false);
assert.equal(M.isHarvestMetric(M.Q_CUTTING_CALENDAR), false);
const state = readBoscoParams({ q: M.Q_CUTTING_CALENDAR, fh: '1' }, [1]);
assert.equal(state.q, M.Q_CUTTING_CALENDAR);
assert.equal(state.harvestPerHa, false, 'calendar colors never divide years by area');
console.log('Cutting calendar: years, fixed color range, tooltips and URL state passed.');
