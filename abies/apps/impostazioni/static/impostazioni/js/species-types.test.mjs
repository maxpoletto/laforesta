import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Mirror Django's namespaced static layout, as the other domain tests do.
const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-species-js-'));
fs.mkdirSync(path.join(tmpRoot, 'impostazioni'), { recursive: true });
fs.mkdirSync(path.join(tmpRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(tmpRoot, 'impostazioni/js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(tmpRoot, 'base/js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const { speciesTypeColumns } = await import(pathToFileURL(path.join(tmpRoot, 'impostazioni/js/species-types.js')));
const S = await import(pathToFileURL(path.join(tmpRoot, 'base/js/strings.js')));

globalThis.document = {
  createElement(tag) {
    return {
      tagName: tag,
      children: [],
      appendChild(child) { this.children.push(child); },
      get value() {
        return this._value ?? this.children.find(option => option.selected)?.value
          ?? this.children[0]?.value ?? '';
      },
      set value(value) { this._value = value; },
      setAttribute(name, value) { this[name] = value; },
      addEventListener(name, handler) { this[name] = handler; },
    };
  },
};
const columns = ['row_id', S.COL_NAME, S.COL_TYPE];
const calls = [];
const pending = new Set();
const defs = speciesTypeColumns(columns, (...args) => calls.push(args), pending);
const cell = () => ({ replaceChildren(...children) { this.children = children; } });
const row = [1, 'Abete', S.COL_CONIFER];
const typeCell = cell();
const type = defs[S.COL_TYPE];
assert.deepEqual(Object.keys(defs), [S.COL_TYPE]);
type.renderCell(typeCell, row[2], row);
const select = typeCell.children[0];
assert.equal(select.tagName, 'select');
assert.equal(select.value, 'conifer');
assert.equal(select.disabled, false);
assert.equal(select['aria-label'], 'Abete: Tipo');
assert.deepEqual(select.children.map(option => [option.value, option.textContent]),
                 [['conifer', 'Conifera'], ['broadleaf', 'Latifoglia']]);
select.value = 'broadleaf';
select.change();
assert.deepEqual(calls, [[row, 'broadleaf']]);
assert.equal(type.formatter(S.COL_BROADLEAF), 'Latifoglia');
assert.equal(type.formatter(''), '—');
for (const [name, value] of [
  [S.SPECIES_OTHER, ''],
  [S.SPECIES_OTHER_CONIFER, S.COL_CONIFER],
  [S.SPECIES_OTHER_BROADLEAF, S.COL_BROADLEAF],
]) {
  const protectedCell = cell();
  type.renderCell(protectedCell, value, [2, name, value]);
  assert.equal(protectedCell.children.length, 0);
  assert.equal(protectedCell.textContent, value || '—');
}
const unknownCell = cell();
type.renderCell(unknownCell, '', [3, 'Unknown custom', '']);
const unknownSelect = unknownCell.children[0];
assert.equal(unknownSelect.value, '');
assert.equal(unknownSelect.children[0].textContent, '—');
assert.equal(unknownSelect.children[0].disabled, true);
assert.equal(unknownSelect.disabled, false);
const broadleafCell = cell();
type.renderCell(broadleafCell, S.COL_BROADLEAF, [4, 'Faggio', S.COL_BROADLEAF]);
assert.equal(broadleafCell.children[0].value, 'broadleaf');
pending.add(1);
type.renderCell(typeCell, row[2], row);
assert.equal(typeCell.children[0].disabled, true);
console.log('21 species dropdown assertions passed');
