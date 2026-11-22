import assert from 'node:assert/strict';
import { addSketchPoint, addSketchRun, cancelSketch, changeGeometry, reduceActivePart, undoEdit } from './draft-edits.js';
import { referenceVertices, traceBetween, traceInsertion, vertexKey } from './boundary-targets.js';

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

const a = [16, 38], b = [16.01, 38.01], c = [16, 38.02];
const west = [15.9, 38.01], east = [16.1, 38.01], south = [16, 37.9];
const targets = referenceVertices([
  {name: 'West', ring: [a, b, c, west, a]},
  {name: 'East', ring: [a, c, east, a]},
  {name: 'South', ring: [a, south, [15.9, 37.9], a]},
]);
const target = point => targets.get(vertexKey(point));
assert.deepEqual(target(a).positions.map(p => p.name), ['West', 'East', 'South']);
assert.deepEqual(traceBetween(target(a), target(b)), [a, b]);
assert.deepEqual(traceBetween(target(c), target(a), path => path.length > 2), [c, b, a]);
assert.equal(traceBetween(target(west), target(south)), null);
assert.deepEqual(traceInsertion([a, c, east, a], 0, [a, b, c]), {index: 1, points: [b]});
assert.deepEqual(traceInsertion([a, east, c, a], 0, [a, b, c]), {index: 3, points: [b]});
assert.deepEqual(traceInsertion([a, [16, 38.03], east, a], 0, [a, b, c]), {index: 1, points: [b, c]});
assert.deepEqual(traceInsertion([a, c, east, a], 0, [a, c]), {index: 1, points: []});
assert.deepEqual(traceBetween(target(a), target(c), path => !!traceInsertion([a, c, east, a], 0, path)), [a, c]);

const traced = {tool: 'Polygon', vertices: []};
addSketchPoint(traced, a);
assert.ok(addSketchRun(traced, [b, c]));
assert.deepEqual(traced.vertices, [a, b, c]);
undoEdit(traced);
assert.deepEqual(traced.vertices, [a]);
addSketchRun(traced, [b, c]);
cancelSketch(traced);
assert.equal(traced.vertices, undefined);
assert.equal(undoEdit(traced), false);

const metres = (lat1, lon1, lat2, lon2) => Math.hypot(lat2 - lat1, lon2 - lon1);
const source = {type: 'LineString', coordinates: [[0, 0], [6, 0], [10, 0], [16, 0], [25, 0]],
  fixes: [[1, 'start'], [2, ''], [3, '10 m'], null, [1, 'end']]};
const reduction = {parts: [source], active: 0};
assert.ok(reduceActivePart(reduction, metres));
assert.equal(reduction.reduction.spacing, 5);
assert.deepEqual(reduction.parts[0].coordinates, [[0, 0], [6, 0], [16, 0], [25, 0]]);
assert.ok(reduceActivePart(reduction, metres));
assert.equal(reduction.reduction.spacing, 10);
assert.deepEqual(reduction.parts[0].coordinates, [[0, 0], [10, 0], [25, 0]]);
assert.deepEqual(reduction.parts[0].fixes, [[1, 'start'], [3, '10 m'], [1, 'end']]);
undoEdit(reduction);
assert.equal(reduction.reduction.spacing, 5);
assert.deepEqual(reduction.parts[0].coordinates, [[0, 0], [6, 0], [16, 0], [25, 0]]);
changeGeometry(reduction, draft => { draft.parts[0].coordinates[1] = [8, 0]; });
assert.equal(reduction.reduction, undefined);
assert.equal(reduceActivePart({...reduction, kind: 'road'}, metres), false);
assert.equal(reduceActivePart({...reduction, kind: 'parcel'}, metres), false);
assert.equal(reduceActivePart({parts: [{type: 'Point', coordinates: [0, 0]}], active: 0}, metres), false);
