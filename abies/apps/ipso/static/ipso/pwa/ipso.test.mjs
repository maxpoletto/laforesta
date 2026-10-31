import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const ipso = require('./ipso.js');
const equation = { a: 7, b: -4, hypso_param_set_id: 5 };
const reference = { hypsometry_by_id: { 2: { 17: equation } },
  ipsometrica: { OldRegion: { Abete: { a: 99, b: 99 } } } };
assert.deepEqual(ipso.lookupReference(reference, 2, 17, 'Renamed region', 'Abete bianco'), equation);
assert.equal(ipso.lookupReference(reference, 2, 18, 'OldRegion', 'Abete'), null);
assert.deepEqual(ipso.lookupReference({ ipsometrica: reference.ipsometrica }, 2, 17,
  'OldRegion', 'Abete'), { a: 99, b: 99 });
assert.equal(ipso.computeH(equation, 30), 20);
console.log('Ipso height equations: stable IDs and old offline reference compatibility passed');
