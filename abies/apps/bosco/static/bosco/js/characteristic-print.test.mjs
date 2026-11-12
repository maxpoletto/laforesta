import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-print-map-'));
for (const app of ['base', 'bosco']) {
  fs.cpSync(path.resolve(here, `../../../../${app}/static/${app}/js`),
    path.join(tmp, app, 'js'), { recursive: true });
}
copyGeodesicVendor(tmp);
process.on('exit', () => fs.rmSync(tmp, { recursive: true, force: true }));
const M = await import(pathToFileURL(path.join(tmp, 'bosco/js/characteristic-print.js')));
const C = await import(pathToFileURL(path.join(tmp, 'bosco/js/bosco-calendar-print.js')));
const { pointInPolygon } = await import(pathToFileURL(path.join(tmp, 'base/js/geo.js')));
const polygon = (x, y, w, h) => ({ type: 'Polygon', coordinates: [
  [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]],
] });
const items = [
  { name: '1', geometry: polygon(16, 38, .01, .01), lines: ['2026, 2030'] },
  { name: '2b', geometry: polygon(16.011, 38, .00005, .01), lines: ['2031'] },
  { name: '3', geometry: polygon(16, 38.012, .01, .01), lines: ['-'] },
];
const layout = M.buildPrintMapLayout({ items, title: 'Compresa',
  subtitle: 'Piano 2026-2040', footer: '2026-11-11' });
assert.equal(layout.labels.length, 3, 'every parcel is labelled once');
assert.ok(layout.labels.some(label => label.leader), 'thin parcel gets a callout');
for (const label of layout.labels) {
  assert.ok(label.x >= 20 && label.y >= 60);
  assert.ok(label.x + label.width <= layout.width - 20);
  assert.ok(label.y + label.height <= layout.height - 25);
  assert.ok(label.lines.every(line => line.size >= 8), 'never shrink labels to fit');
}
for (let i = 0; i < layout.labels.length; i++) {
  for (const other of layout.labels.slice(i + 1)) {
    const a = layout.labels[i], b = other;
    assert.ok(a.x + a.width <= b.x || b.x + b.width <= a.x
      || a.y + a.height <= b.y || b.y + b.height <= a.y, 'labels cannot overlap');
  }
}
const bytes = M.printMapPDF(layout);
assert.ok(bytes.startsWith('%PDF-1.4'));
assert.ok(!bytes.includes('/Subtype /Image'), 'map stays vector for LaTeX');
assert.ok(bytes.includes('(2b)') && bytes.includes('(2031)'));
assert.ok(bytes.includes(`/MediaBox [0 0 ${layout.width} ${layout.height}]`));
const hole = polygon(16, 38, .02, .02);
hole.coordinates.push(polygon(16.003, 38.003, .014, .014).coordinates[0]);
const holeLayout = M.buildPrintMapLayout({ items: [{ name: 'hole', geometry: hole,
  lines: ['2029'] }], title: '', subtitle: '', footer: '' });
const anchor = holeLayout.labels[0].anchor;
assert.ok(pointInPolygon(anchor[0], anchor[1], holeLayout.parcels[0].geometry),
  'callout anchor must lie inside the parcel, never inside a hole');
const multipart = { type: 'MultiPolygon', coordinates: [
  polygon(16, 38, .01, .01).coordinates, polygon(16.02, 38, .01, .01).coordinates,
] };
const multi = M.buildPrintMapLayout({ items: [{ name: 'multi', geometry: multipart,
  lines: ['2030'] }], title: '', subtitle: '', footer: '' });
assert.equal(multi.parcels[0].geometry.coordinates.length, 2, 'retain all polygon parts');
assert.equal(multi.labels.length, 1, 'one label per parcel, including multipart parcels');
assert.throws(() => M.buildPrintMapLayout({ items: [], title: '' }));
assert.throws(() => M.buildPrintMapLayout({ items: [{ name: 'bad',
  geometry: { type: 'Point', coordinates: [16, 38] }, lines: [] }] }));
const features = items.map((item, index) => ({ type: 'Feature', geometry: item.geometry,
  properties: { layer: 'Forest', name: `Forest-${item.name}`, id: 900 + index } }));
features.push({ ...features[0], geometry: polygon(16.03, 38, .001, .001) });
features.push({ ...features[0], properties: { layer: 'Elsewhere', name: 'Elsewhere-1' } });
const entries = items.map((item, index) => ({ id: index + 1, region: 'Forest', parcel: item.name,
  coppice: index === 1 }));
const calendar = { selected_plan: { id: 1, name: 'Plan', year_start: 2026, year_end: 2040 },
  calendar: [{ parcel_id: 1, years: [2030, 2026] }, { parcel_id: 2, years: [2031] }] };
const printItems = C.calendarPrintItems({ features }, 'Forest', entries, calendar);
assert.deepEqual(printItems.map(item => item.name), ['1 (F)', '2b (C)', '3 (F)'],
  'each parcel label identifies the forest type, including unscheduled parcels');
assert.equal(printItems.length, 3, 'merge split GeoJSON features, exclude other regions');
assert.equal(printItems[0].geometry.coordinates.length, 2, 'retain both pieces of split parcel');
assert.deepEqual(printItems[0].lines, ['2026, 2030'], 'join by DB identity, not GeoJSON feature ID');
assert.deepEqual(printItems[2].lines, ['-'], 'dash means no scheduled harvest');
assert.throws(() => C.calendarPrintItems({ features }, 'Forest', entries.slice(1), calendar),
  'never misrepresent an unmatched geometry as an unscheduled parcel');
assert.throws(() => C.calendarPrintItems({ features: features.filter(f => f.properties.name !== 'Forest-2b') },
  'Forest', entries, calendar));
assert.throws(() => C.calendarPrintItems({ features }, 'Forest', entries, null));
assert.equal(C.calendarPlanSubtitle(entries, calendar), 'Plan (fustaia 2026-2030, ceduo 2031)');
const allEntries = [...entries, { id: 4, region: 'Elsewhere', parcel: '8', coppice: true }];
const allCalendar = { ...calendar, selected_plan: { ...calendar.selected_plan, year_end: 2050 },
  calendar: [...calendar.calendar, { parcel_id: 4, years: [2050] }] };
assert.equal(C.calendarPlanSubtitle(allEntries, allCalendar),
  'Plan (fustaia 2026-2030, ceduo 2031-2050)', 'heading ranges span all comprese in the plan');
assert.equal(C.calendarPlanSubtitle(entries, { ...calendar,
  calendar: [calendar.calendar[0]] }), 'Plan (fustaia 2026-2030)', 'omit forest types without planned cuts');
assert.equal(C.calendarPlanSubtitle(entries, { ...calendar,
  calendar: [calendar.calendar[1]] }), 'Plan (ceduo 2031)');
assert.equal(C.calendarPlanSubtitle(entries, { ...calendar, calendar: [] }), 'Plan');
assert.throws(() => C.calendarPlanSubtitle(entries.slice(1), calendar));
assert.throws(() => C.calendarPrintItems({ features }, 'Forest',
  entries.map(entry => ({ ...entry, coppice: null })), calendar));
const longHeader = M.buildPrintMapLayout({ items, title: 'Compresa '.repeat(12),
  subtitle: 'Piano di taglio '.repeat(15), footer: 'Data: 2026-11-11' });
assert.ok(longHeader.headerLines.length > 2, 'long names wrap instead of running off the A4 page');
for (const line of longHeader.headerLines) {
  assert.ok(line.y < Math.min(...longHeader.labels.map(label => label.y)));
}
globalThis.document = { createElementNS(namespace, tag) {
  return { namespace, tag, attrs: {}, children: [], textContent: '',
    setAttribute(key, value) { this.attrs[key] = value; },
    appendChild(child) { this.children.push(child); return child; } };
} };
const preview = M.buildPrintMapPreview({ ...layout, title: '<img onerror=alert(1)>' });
assert.equal(preview.attrs['aria-label'], '<img onerror=alert(1)>');
assert.equal(preview.attrs.viewBox, `0 0 ${layout.width} ${layout.height}`);
assert.equal(preview.children.filter(el => el.tag === 'text').length,
  layout.headerLines.length + 1 + layout.labels.reduce((n, label) => n + label.lines.length, 0));
assert.ok(preview.children.every(el => el.innerHTML === undefined), 'preview never uses HTML interpolation');
console.log('Print maps: complete labels, callouts, holes, multipart and vector PDF passed.');
