// Regression tests for shared formatting helpers.
// Run with: node apps/base/static/base/js/format.test.mjs

process.env.TZ = 'Europe/Rome';

const { localISODate } = await import('./format.js');

let passed = 0;
let failed = 0;

function eq(actual, expected, msg) {
  if (actual === expected) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}: expected ${expected}, got ${actual}`);
  }
}

const cestAfterMidnight = new Date('2026-07-09T22:30:00Z');
eq(cestAfterMidnight.toISOString().slice(0, 10), '2026-07-09',
   'UTC ISO date exposes the previous day around local midnight');
eq(localISODate(cestAfterMidnight), '2026-07-10',
   'localISODate returns the local calendar day');

const winterMorning = new Date('2026-01-10T07:05:00Z');
eq(localISODate(winterMorning), '2026-01-10',
   'localISODate pads month and day');

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
