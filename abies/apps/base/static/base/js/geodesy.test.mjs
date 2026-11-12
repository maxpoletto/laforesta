import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import * as geo from './geo.js';

const reference = JSON.parse(fs.readFileSync(new URL('../../../../../test/fixtures/wgs84-geodesy.json', import.meta.url)));
const upload = createRequire(import.meta.url)('../../../../ipso/static/ipso/pwa/upload.js');
const browser = vm.createContext({});
browser.window = browser;
vm.runInContext(fs.readFileSync(new URL('../../vendor/geographiclib/geographiclib-geodesic.min.js', import.meta.url), 'utf8'), browser);
vm.runInContext(fs.readFileSync(new URL('./geodesy.js', import.meta.url), 'utf8'), browser);
assert.equal(browser.AbiesGeodesy.WGS84.a, 6378137);
assert.equal(browser.AbiesGeodesy.WGS84.f, 1 / 298.257223563);
for (const { points, metres } of reference.distances) {
  assert.ok(Math.abs(geo.distanceMeters(...points) - metres) < 1e-6,
    `WGS84 inverse distance ${points}: expected ${metres}, got ${geo.distanceMeters(...points)}`);
  assert.equal(upload.distanceMeters(...points), geo.distanceMeters(...points), 'Ipso and Abies use the same numerical core');
  assert.equal(browser.AbiesGeodesy.distanceMeters(...points), geo.distanceMeters(...points), 'offline classic-script loading matches ES modules');
}
for (const { ring, area } of reference.areas) {
  const vertices = ring.map(([lng, lat]) => ({ lng, lat }));
  assert.ok(Math.abs(geo.geodesicArea(vertices) - area) < .01, 'ellipsoidal area matches independent Python reference');
  assert.ok(Math.abs(geo.geodesicArea([...vertices].reverse()) - area) < .01, 'ring winding does not change parcel area');
  assert.ok(Math.abs(browser.AbiesGeodesy.polygonAreaMeters2(vertices) - area) < .01);
}
assert.equal(geo.geodesicArea([]), 0);
assert.equal(geo.geodesicArea([{ lat: 38.5, lng: 16.3 }]), 0);
for (const { geometry, area } of reference.polygons) {
  assert.ok(Math.abs(geo.geoJSONFeatureArea({ geometry }) - area) < .01,
    'holes and multipart polygons match independent area references');
}
const sw = fs.readFileSync(new URL('../../../../ipso/static/ipso/pwa/sw.js', import.meta.url), 'utf8');
assert.ok(sw.includes("'/static/base/js/geodesy.js'"));
assert.ok(sw.includes("'/static/vendor/geographiclib/geographiclib-geodesic.min.js'"));
console.log('WGS84 inverse distances, antipodes, poles, dateline, areas and Ipso parity passed.');
