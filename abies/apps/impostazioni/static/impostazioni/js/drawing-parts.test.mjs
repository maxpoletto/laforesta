import assert from 'node:assert/strict';
import { closeLine, insertVertex, joinParts, moveVertex, openPolygon, splitLine } from './drawing-parts.js';

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
