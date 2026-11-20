import assert from 'node:assert/strict';
import { addSketchPoint, cancelSketch, changeGeometry, undoEdit } from './draft-edits.js';

const outline = {type: 'Polygon', coordinates: [[[16, 38], [17, 38], [17, 39], [16, 38]]]};
const draft = {parts: [outline], active: 0, edits: [], neighbors: [], added: [],
  features: [{parts: [outline]}]};
changeGeometry(draft, d => {
  d.parts[0].coordinates[0].splice(1, 0, [16.5, 38]);
  d.neighbors.push('Neighbor');
});
changeGeometry(draft, d => { d.parts[0].coordinates[0][1] = [16.6, 38]; });
Object.assign(draft, {tool: 'Polygon', vertices: []});
for (const point of [[18, 38], [19, 38], [19, 39]]) addSketchPoint(draft, point);
assert.equal(addSketchPoint(draft, [19, 39]), false);
changeGeometry(draft, d => {
  const piece = {type: 'Polygon', coordinates: [[...d.vertices, d.vertices[0]]]};
  d.parts.push(piece);
  d.added.push(piece);
  delete d.tool;
  delete d.vertices;
});
undoEdit(draft);
assert.equal(draft.tool, 'Polygon');
assert.deepEqual(draft.vertices, [[18, 38], [19, 38], [19, 39]]);
assert.equal(draft.parts.length, 1);
assert.deepEqual(draft.added, []);
undoEdit(draft);
assert.equal(draft.vertices.length, 2);
cancelSketch(draft);
assert.equal(draft.tool, undefined);
undoEdit(draft);
assert.deepEqual(draft.parts[0].coordinates[0][1], [16.5, 38]);
assert.equal(draft.parts[0], draft.features[0].parts[0]);
assert.deepEqual(draft.neighbors, ['Neighbor']);
undoEdit(draft);
assert.equal(draft.parts[0].coordinates[0].length, 4);
assert.deepEqual(draft.neighbors, []);
assert.equal(draft.dirty, undefined);

const sketch = {tool: 'LineString', vertices: []};
for (let i = 0; i < 12; i++) addSketchPoint(sketch, [16 + i / 10, 38]);
assert.equal(sketch.history.length, 10);
assert.equal(changeGeometry(sketch, () => false), false);
for (let i = 0; i < 10; i++) undoEdit(sketch);
assert.deepEqual(sketch.vertices, [[16, 38], [16.1, 38]]);
assert.equal(undoEdit(sketch), false);
