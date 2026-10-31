import assert from 'node:assert/strict';
import { volumeFromCoefficientsM3 } from './volume.js';
import { heightEquationFor } from './equations.js';

assert.equal(volumeFromCoefficientsM3(30, 20, [.81151, .038965]), .70218151);
assert.equal(volumeFromCoefficientsM3(1, 1, [-2, .036524, .74466]), 0);
assert.equal(volumeFromCoefficientsM3(30, 20, null), null);
assert.equal(volumeFromCoefficientsM3(30, 20, [1000, 0]), 1);
assert.equal(volumeFromCoefficientsM3(0, 20, [1000, 0]), null);
const digest = {
  columns: ['row_id', 'Compresa', 'Specie', 'funzione', 'a', 'b', 'n', 'r²', 'region_id', 'species_id'],
  rows: [[1, 'Old region label', 'Abete', 'ln', 7, -4, 20, .6, 2, 17]],
};
assert.deepEqual(heightEquationFor(digest, 2, 17), { a: 7, b: -4, func: 'ln' });
assert.equal(heightEquationFor(digest, 2, 18), null);
assert.equal(heightEquationFor(digest, 3, 17), null);
console.log('Equation previews: stable identity, custom coefficients and unavailable values passed');
