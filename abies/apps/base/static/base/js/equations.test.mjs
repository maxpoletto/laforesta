import assert from 'node:assert/strict';
import { volumeFromCoefficientsM3 } from './volume.js';
import { heightEquationFor, wireHeightAutofill } from './equations.js';

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
class Field {
  constructor(value) { this.value = value; this.listeners = {}; }
  addEventListener(kind, fn) { (this.listeners[kind] ??= []).push(fn); }
  dispatchEvent(event) { for (const fn of this.listeners[event.type] || []) fn(); }
}
const fields = { diameter: new Field('30'), height: new Field(''), species: new Field('17'),
  measured: new Field('0'), notice: { hidden: true, textContent: '' } };
wireHeightAutofill(fields, digest, 2);
assert.ok(fields.height.value);
fields.species.value = '18';
fields.species.dispatchEvent(new Event('change'));
assert.equal(fields.height.value, '');
assert.equal(fields.measured.value, '0');
assert.equal(fields.notice.hidden, false);
fields.height.value = '23';
fields.height.dispatchEvent(new Event('input'));
fields.diameter.value = '40';
fields.diameter.dispatchEvent(new Event('input'));
assert.equal(fields.height.value, '23'); // Preserve a measured height while editing D.
console.log('Equation previews: stable identity, custom coefficients and unavailable values passed');
