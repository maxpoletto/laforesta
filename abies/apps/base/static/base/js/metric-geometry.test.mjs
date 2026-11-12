import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as geo from './geo.js';
const reference = JSON.parse(fs.readFileSync(new URL('../../../../../test/fixtures/wgs84-geodesy.json', import.meta.url)));
for (const { polygon, query, meters } of reference.boundaries) {
  const feature = { geometry: reference.boundary_polygons[polygon] };
  const actual = geo.distanceToBoundaryMeters(...query, feature);
  assert.ok(Math.abs(actual - meters) < .01,
    `boundary distance ${query}: expected ${meters}, got ${actual}`);
}
for (const { start, azimuth, meters, point } of reference.destinations) {
  const actual = geo.destination(...start, azimuth, meters);
  assert.ok(geo.distanceMeters(actual.lat, actual.lng, ...point) < 1e-6);
  const plane = geo.localProjection(...start);
  const xy = plane.project(...point);
  assert.ok(Math.abs(Math.hypot(xy.x, xy.y) - meters) < 1e-6);
  const back = plane.unproject(xy.x, xy.y);
  assert.ok(geo.distanceMeters(back.lat, back.lng, ...point) < 1e-6);
}
const plane = geo.localProjection(38.5, 16.3);
const ring = [[-2500,-2500],[2500,-2500],[2500,2500],[-2500,2500],[-2500,-2500]]
  .map(([x,y]) => { const p = plane.unproject(x,y); return [p.lng,p.lat]; });
const feature = { type: 'Feature', properties: { layer: 'Forest', name: 'Forest-1' },
  geometry: { type: 'Polygon', coordinates: [ring] } };
const points = geo.generateGrid([feature], 500);
assert.ok(points.length >= 81 && points.length <= 121, 'generate a metric grid across a 5 km region');
for (const point of points) {
  assert.ok(geo.pointInPolygon(point.lng, point.lat, feature.geometry));
  assert.equal(point.compresa, 'Forest');
  assert.equal(point.particella, '1');
  const neighbors = points.filter(other => other !== point).map(other =>
    geo.distanceMeters(point.lat, point.lng, other.lat, other.lng));
  assert.ok(neighbors.some(distance => Math.abs(distance - 500) < .001),
    'ground spacing remains within 1 mm of 500 m across the whole 5 km patch');
}
const planned = geo.planGridForTarget([feature], 50);
assert.ok(planned.length >= 45 && planned.length <= 55, 'target-count planner retains its count contract');
assert.deepEqual(geo.generateGrid([], 500), []);
assert.deepEqual(geo.planGridForTarget([], 50), []);
assert.throws(() => geo.generateGrid([feature], 0));
console.log('Metric geometry: boundary proximity, local planes and 5 km grid spacing passed.');
