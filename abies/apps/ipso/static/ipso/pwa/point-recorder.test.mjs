import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const { GPS_STALE_MS } = require('./constants.js');
const { createPointRecorder } = require('./point-recorder.js');
globalThis.GPS_STALE_MS = GPS_STALE_MS;

test('notes pause the walk, freeze a fix, and resume in a new segment on save or cancel', async t => {
  let time = 20000;
  t.mock.method(Date, 'now', () => time);
  const rows = [];
  let nextId = 0;
  let fail = false;
  const recorder = createPointRecorder({
    newId: () => ++nextId,
    append: async row => {
      if (fail) throw new Error('quota');
      rows.push(row);
    },
    onError: assert.fail,
  });
  function fix(lat) { return { lat, lon: 16, acc: 3.125, t: ++time }; }
  recorder.toggle();
  const first = fix(38);
  recorder.onFix(first);
  recorder.onFix(first); // heartbeat
  recorder.begin(fix(38.1));
  recorder.onFix(fix(39));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(recorder.savedFixes, 1);
  fail = true;
  await assert.rejects(recorder.save('gate'), /quota/);
  assert.ok(recorder.editing);
  fail = false;
  await recorder.save('gate');
  assert.equal(recorder.savedFixes, 1);
  recorder.onFix(fix(38.2));
  recorder.begin(fix(38.3));
  const stationary = fix(39);
  recorder.onFix(stationary);
  recorder.cancel();
  recorder.onFix(stationary);
  recorder.onFix(fix(38.4));
  time += GPS_STALE_MS;
  recorder.onFix(fix(38.5));
  recorder.interrupt();
  recorder.onFix(fix(38.6));
  await recorder.stop();
  assert.deepEqual(rows.map(r => [r.segment, r.lat, r.note]), [
    [1, 38, ''], [null, 38.1, 'gate'], [2, 38.2, ''], [3, 38.4, ''],
    [4, 38.5, ''], [5, 38.6, ''],
  ]);
  assert.ok(rows.every(r => r.acc_m === 3.125));
});

test('trace feedback counts committed fixes from the current run', async t => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'], now: 20000 });
  const writes = [];
  const counts = [];
  const errors = [];
  const recorder = createPointRecorder({
    newId: () => 'segment',
    append: () => new Promise((resolve, reject) => writes.push({ resolve, reject })),
    onTraceSaved: () => counts.push(recorder.savedFixes),
    onError: error => errors.push(error.message),
  });
  const fix = () => recorder.onFix({ lat: 38, lon: 16, acc: 4, t: Date.now() });
  const tick = () => new Promise(resolve => setImmediate(resolve));
  recorder.toggle();
  fix();
  t.mock.timers.tick(1000);
  await tick();
  assert.equal(recorder.savedFixes, 0);
  recorder.toggle();
  recorder.toggle();
  fix();
  t.mock.timers.tick(1000);
  writes.shift().resolve();
  await tick();
  assert.deepEqual(counts, []);
  writes.shift().resolve();
  await tick();
  assert.deepEqual(counts, [1]);
  fix();
  t.mock.timers.tick(1000);
  await tick();
  const closing = recorder.stop();
  writes.shift().reject(new Error('quota'));
  await assert.rejects(closing, /quota/);
  await recorder.stop();
  assert.deepEqual(errors, ['quota']);
  assert.deepEqual(counts, [1]);
});

test('the three field bursts save three original winners, including a sub-metre tie-break', async t => {
  const start = Date.parse('2026-11-18T10:40:51.809Z');
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'], now: start });
  const rows = [];
  const recorder = createPointRecorder({ newId: () => 'walk', append: row => rows.push(row), onError: assert.fail });
  recorder.toggle();
  const fixes = [
    [0, 8], [10, 5.49], [61, 5.41], [63, 5.41], [64, 6],
    [3998, 2], [4015, 8], [4056, 4], [4058, 9], [4059, 3],
    [7997, 8], [8006, 7], [8049, 6], [8052, 3.25],
  ];
  for (const [elapsed, acc] of fixes) {
    t.mock.timers.tick(start + elapsed - Date.now());
    const fix = { lat: 38 + elapsed / 100000, lon: 16, acc, t: Date.now() };
    recorder.onFix(fix);
    recorder.onFix(fix); // GPS heartbeat replays the same measurement.
  }
  t.mock.timers.tick(1000);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(recorder.savedFixes, 3);
  assert.deepEqual(rows, [fixes[2], fixes[5], fixes[13]].map(([elapsed, acc]) => ({
    lat: 38 + elapsed / 100000, lon: 16, acc_m: acc,
    taken_at: new Date(start + elapsed).toISOString(), segment: 'walk', note: '',
  })));
  await recorder.stop();
  assert.equal(rows.length, 3);
});

test('sustained fixes cross second boundaries and the final pending winner precedes export', async t => {
  const start = 20750; // the first burst straddles a wall-clock second
  t.mock.timers.enable({ apis: ['Date', 'setTimeout'], now: start });
  const rows = [];
  const recorder = createPointRecorder({ newId: () => 'walk', append: row => rows.push(row), onError: assert.fail });
  recorder.toggle();
  // Deliver a batch synchronously as after a delayed timer on a busy device.
  for (const [offset, acc] of [[0, 8], [300, 2], [900, 7], [1000, 4], [1800, 3], [2100, 6]]) {
    recorder.onFix({ lat: 38, lon: 16, acc, t: start - 3000 + offset });
  }
  await recorder.stop();
  t.mock.timers.tick(2000);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(rows.map(row => row.acc_m), [2, 3, 6]);
});
