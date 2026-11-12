import { copyGeodesicVendor } from '../../../../../test/js-static.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-species-editor-'));
copyGeodesicVendor(root);
process.on('exit', () => fs.rmSync(root, { recursive: true, force: true }));
fs.mkdirSync(path.join(root, 'impostazioni/js'), { recursive: true });
fs.cpSync(path.join(here, '../../../../base/static/base/js'), path.join(root, 'base/js'), { recursive: true });
fs.copyFileSync(path.join(here, 'species-equations.js'), path.join(root, 'impostazioni/js/species-equations.js'));
const { collectSpeciesEquations } = await import(pathToFileURL(path.join(root, 'impostazioni/js/species-equations.js')));
const values = { volume_mode: 'own', volume_b0: '-1,8381', volume_b1: '0,037836',
  volume_b2: '0,39934', volume_source: 'Tabacchi', volume_proxy_species_id: '17' };
const region = { dataset: { regionId: '2' }, querySelector(selector) {
  const key = selector.match(/data-field="([^"]+)"/)[1];
  return key === 'enabled' ? { checked: true } : { value: { a: '7,1234', b: '-4', n: '', r2: '' }[key] };
} };
const form = {
  querySelector(selector) { return { value: values[selector.match(/name="([^"]+)"/)[1]] }; },
  querySelectorAll() { return [region]; },
};
let data = collectSpeciesEquations(form);
assert.equal(data.volume_equation.b1, '0,037836'); // Preserve locale and precision.
assert.equal(data.height_equations[0].region_id, 2);
assert.equal(data.height_equations[0].a, '7,1234');
assert.equal(data.height_equations[0].enabled, true);
values.volume_mode = 'proxy';
data = collectSpeciesEquations(form);
assert.equal(data.volume_equation.proxy_species_id, '17');
assert.equal(data.volume_equation.mode, 'proxy');
console.log('Species editor: regional identity, proxy selection and coefficient precision passed');
