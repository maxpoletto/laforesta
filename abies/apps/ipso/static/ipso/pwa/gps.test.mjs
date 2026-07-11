// Tests for apps/ipso/static/ipso/pwa/gps.js.
// Run with: node apps/ipso/static/ipso/pwa/gps.test.mjs

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, 'constants.js'), 'utf8') + '\n' +
  fs.readFileSync(path.join(here, 'gps.js'), 'utf8') +
  '\nglobalThis.__gpsTest = { createGps };\n';

let passed = 0;
let failed = 0;

function check(ok, msg) {
  if (ok) passed += 1;
  else {
    failed += 1;
    console.error(`FAIL ${msg}`);
  }
}

function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
}

{
  let heartbeat = null;
  let visibilityHandler = null;
  let geolocationError = null;
  let watchCalls = 0;
  let clearCalls = 0;
  const states = [];

  const context = {
    console,
    Date,
    setInterval(fn) { heartbeat = fn; return 1; },
    clearInterval() {},
    setTimeout(fn) { fn(); return 1; },
    clearTimeout() {},
    navigator: {
      geolocation: {
        watchPosition(_success, error) {
          watchCalls += 1;
          geolocationError = error;
          return watchCalls;
        },
        clearWatch() { clearCalls += 1; },
      },
    },
    document: {
      visibilityState: 'visible',
      addEventListener(type, fn) {
        if (type === 'visibilitychange') visibilityHandler = fn;
      },
      removeEventListener() {},
    },
  };
  context.globalThis = context;
  vm.runInNewContext(source, context);

  const gps = context.__gpsTest.createGps(state => states.push(state));
  gps.start();
  eq(watchCalls, 1, 'gps.start registers the watcher once');
  geolocationError({ code: 1, PERMISSION_DENIED: 1 });
  eq(gps.state().error, 'denied', 'permission denial is visible in state');

  heartbeat();
  heartbeat();
  eq(watchCalls, 1, 'heartbeat does not restart after permission denial');

  visibilityHandler();
  eq(watchCalls, 2, 'visibilitychange may retry after a permission/settings change');
  check(clearCalls >= 1, 'permission denial clears the active watcher');
  check(states.some(state => state.error === 'denied'), 'denied state is emitted to the UI');
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
