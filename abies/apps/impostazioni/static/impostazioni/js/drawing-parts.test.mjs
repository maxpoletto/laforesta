import assert from 'node:assert/strict';
import { canRemoveVertex, closeLine, insertVertex, joinParts, moveVertex, openPolygon, removeVertex, splitLine } from './drawing-parts.js';
import { boundaryVertexRemoval, removeBoundaryVertex } from './parcel-boundaries.js';

const track = {type: 'LineString', coordinates: [[16, 38], [16.1, 38], [16, 38.2]],
  fixes: [[2, ''], [3, 'Bivio'], [2, '']]};
const pieces = splitLine(track, 1);
assert.deepEqual(pieces.map(p => p.coordinates.length), [2, 2]);
assert.deepEqual(pieces[1].fixes[0], [3, 'Bivio']);
joinParts(pieces, 0);
assert.deepEqual(pieces, [track]);

const polygon = closeLine(track);
polygon.coordinates.push([[16.01, 38.02], [16.02, 38.02], [16.01, 38.03], [16.01, 38.02]]);
polygon.fixes.push(null, null, [2, 'Interno'], null);
moveVertex(polygon, 1, 0, [16.01, 38.015]);
assert.deepEqual(polygon.coordinates[1][0], polygon.coordinates[1].at(-1));
insertVertex(polygon, 1, 1, [16.015, 38.02]);
const [outer, inner] = openPolygon(polygon);
assert.deepEqual(outer, track);
assert.deepEqual(inner.fixes, [null, null, null, [2, 'Interno']]);
assert.equal(inner.coordinates.length, 4);

assert.equal(canRemoveVertex(polygon, 0, 0), false);
assert.equal(canRemoveVertex(polygon, 1, 0), true);
const oldOuterFixes = polygon.fixes.slice(0, polygon.coordinates[0].length);
removeVertex(polygon, 1, 0);
assert.deepEqual(polygon.coordinates[1][0], polygon.coordinates[1].at(-1));
assert.deepEqual(polygon.fixes, [...oldOuterFixes, null, null, [2, 'Interno'], null]);

assert.equal(canRemoveVertex(track, 0, 1), true);
removeVertex(track, 0, 1);
assert.deepEqual(track.coordinates, [[16, 38], [16, 38.2]]);
assert.deepEqual(track.fixes, [[2, ''], [2, '']]);
assert.equal(canRemoveVertex(track, 0, 0), false);

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
