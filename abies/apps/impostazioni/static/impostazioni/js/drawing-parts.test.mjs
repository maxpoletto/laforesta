import assert from 'node:assert/strict';
import {
  breakPart, canJoinEndpoints, canRemoveVertex, closeLine, insertVertex, joinEndpoints,
  moveVertex, removeVertex,
} from './drawing-parts.js';
import { boundaryVertexRemoval, removeBoundaryVertex } from './parcel-boundaries.js';

const track = {type: 'LineString', coordinates: [[16, 38], [16.1, 38], [16, 38.2]],
  fixes: [[2, ''], [3, 'Bivio'], [2, '']]};
const pieces = breakPart(track, 0, 1);
assert.deepEqual(pieces.map(p => p.coordinates.length), [2, 2]);
assert.deepEqual(pieces[1].fixes[0], [3, 'Bivio']);
joinEndpoints(pieces, {part: 1, index: 0}, {part: 0, index: 1});
assert.deepEqual(pieces, [track]);
assert.equal(breakPart(track, 0, 0), null);
assert.equal(breakPart(track, 0, 2), null);

const line = (coordinates, fixes) => ({type: 'LineString', coordinates, fixes});
const forward = line([[16, 38], [16.1, 38]], [[2, 'a'], [3, 'b']]);
const reverse = line([[16.3, 38], [16.2, 38]], [[2, 'd'], [4, 'c']]);
const point = {type: 'Point', coordinates: [17, 38]};
const nonconsecutive = [forward, point, reverse];
joinEndpoints(nonconsecutive, {part: 2, index: 1}, {part: 0, index: 1});
assert.deepEqual(nonconsecutive, [line([[16, 38], [16.1, 38], [16.2, 38], [16.3, 38]],
  [[2, 'a'], [3, 'b'], [4, 'c'], [2, 'd']]), point]);
assert.equal(joinEndpoints([forward], {part: 0, index: 0}, {part: 0, index: 1}), null);
assert.equal(joinEndpoints([track], {part: 0, index: 0}, {part: 0, index: 0}), null);
const closed = [track];
joinEndpoints(closed, {part: 0, index: 2}, {part: 0, index: 0});
assert.deepEqual(closed, [closeLine(track)]);
assert.equal(canJoinEndpoints([]), false);
assert.equal(canJoinEndpoints([...closed, point]), false);
assert.equal(canJoinEndpoints([forward, point]), true);
assert.equal(canJoinEndpoints([line([[16, 38], [17, 38], [16, 38]])]), false);
assert.equal(canJoinEndpoints([track]), true);
assert.equal(canJoinEndpoints([track], {close: false}), false);
assert.equal(canJoinEndpoints([forward, point, reverse], {close: false}), true);

const gate = {type: 'Point', coordinates: [16.2, 38], fixes: [[1, 'Cancello']]};
const singles = [point, gate];
joinEndpoints(singles, {part: 1, index: 0}, {part: 0, index: 0});
assert.deepEqual(singles, [line([point.coordinates, gate.coordinates], [null, gate.fixes[0]])]);
const prepend = [reverse, gate];
joinEndpoints(prepend, {part: 0, index: 0}, {part: 1, index: 0});
assert.deepEqual(prepend, [line([gate.coordinates, ...reverse.coordinates], [gate.fixes[0], ...reverse.fixes])]);
const append = [gate, forward];
joinEndpoints(append, {part: 0, index: 0}, {part: 1, index: 1});
assert.deepEqual(append, [line([...forward.coordinates, gate.coordinates], [...forward.fixes, gate.fixes[0]])]);
const coincident = [forward, {...gate, coordinates: forward.coordinates[1]}];
joinEndpoints(coincident, {part: 0, index: 1}, {part: 1, index: 0});
assert.deepEqual(coincident, [line(forward.coordinates, [forward.fixes[0], gate.fixes[0]])]);

const polygon = closeLine(track);
polygon.coordinates.push([[16.01, 38.02], [16.02, 38.02], [16.01, 38.03], [16.01, 38.02]]);
polygon.fixes.push(null, null, [2, 'Interno'], null);
moveVertex(polygon, 1, 0, [16.01, 38.015]);
assert.deepEqual(polygon.coordinates[1][0], polygon.coordinates[1].at(-1));
insertVertex(polygon, 1, 1, [16.015, 38.02]);
const [outer, inner] = breakPart(polygon, 1, 1);
assert.deepEqual(outer.coordinates, polygon.coordinates[0]);
assert.deepEqual(inner.coordinates, polygon.coordinates[1].slice(2, -1).concat([polygon.coordinates[1][0]]));
assert.deepEqual(inner.fixes, [null, [2, 'Interno'], null]);
assert.deepEqual(breakPart(closeLine(track), 0, 0), [
  line([[16.1, 38], [16, 38.2]], [[3, 'Bivio'], [2, '']]),
]);

assert.equal(canRemoveVertex(polygon, 0, 0), false);
assert.equal(canRemoveVertex(polygon, 1, 0), true);
const oldOuterFixes = polygon.fixes.slice(0, polygon.coordinates[0].length);
removeVertex(polygon, 1, 0);
assert.deepEqual(polygon.coordinates[1][0], polygon.coordinates[1].at(-1));
assert.deepEqual(polygon.fixes, [...oldOuterFixes, null, null, [2, 'Interno'], null]);

const shortened = structuredClone(track);
assert.equal(canRemoveVertex(shortened, 0, 0), true);
assert.equal(canRemoveVertex(shortened, 0, 2), true);
assert.equal(canRemoveVertex(shortened, 0, 1), true);
removeVertex(shortened, 0, 1);
assert.deepEqual(shortened, line([[16, 38], [16, 38.2]], [[2, ''], [2, '']]));
assert.equal(canRemoveVertex(shortened, 0, 1), true);

const collapsed = structuredClone(forward);
removeVertex(collapsed, 0, 0);
assert.deepEqual(collapsed, {type: 'Point', coordinates: [16.1, 38], fixes: [[3, 'b']]});
assert.equal(canRemoveVertex(collapsed, 0, 1), false);
assert.equal(canRemoveVertex(collapsed, 0, 0), true);
assert.equal(removeVertex(collapsed, 0, 0), null);

const west = {type: 'Polygon', coordinates: [[[16, 38], [17, 38], [17.1, 38.5], [17, 39], [16, 39], [16, 38]]]};
const east = {type: 'Polygon', coordinates: [[[17, 38], [18, 38], [18, 39], [17, 39], [17.1, 38.5], [17, 38]]]};
const draft = {
  parcelId: 1, active: 0, parts: [west], edits: [], neighbors: [],
  features: [{id: 1, parcel_id: 1, name: 'Ovest', parts: [west]},
    {id: 2, parcel_id: 2, name: 'Est', parts: [east]}],
};
assert.equal(boundaryVertexRemoval(draft, 0, 1), null);
const removal = boundaryVertexRemoval(draft, 0, 2);
removeBoundaryVertex(draft, 0, 2, removal);
assert.deepEqual(draft.neighbors, ['Est']);
assert.deepEqual(draft.edits, [{id: 1, part: 0, ring: 0, index: 2, operation: 'remove'}]);
assert.deepEqual(west.coordinates[0].slice(1, 3), [[17, 38], [17, 39]]);
assert.deepEqual(east.coordinates[0].slice(-2), [[17, 39], [17, 38]]);
