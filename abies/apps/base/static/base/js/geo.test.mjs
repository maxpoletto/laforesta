// Tests for apps/base/static/base/js/geo.js — pure geometry helpers.
// Run with: node apps/base/static/base/js/geo.test.mjs (also part of `make test-js`).
//
// Hand-rolled assertions in the same style as ipso/tests.js so this
// pure-JS suite stays free of npm test-framework dependencies.

import * as geo from './geo.js';

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

function assertClose(actual, expected, tolerance, msg) {
  const diff = Math.abs(actual - expected);
  if (diff <= tolerance) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL ${msg}: expected ~${expected}, got ${actual} (diff ${diff})`);
  }
}

console.log('geo.js');

const unitSquare = {
  type: 'Feature',
  properties: { layer: 'X', name: 'X-1' },
  geometry: { type: 'Polygon', coordinates: [[[0,0],[1,0],[1,1],[0,1],[0,0]]] },
};

assertEqual(geo.pointInPolygon(0.5, 0.5, unitSquare.geometry), true,
            'pointInPolygon: centre inside');
assertEqual(geo.pointInPolygon(2, 2, unitSquare.geometry), false,
            'pointInPolygon: clearly outside');
assertEqual(geo.pointInPolygon(-0.1, 0.5, unitSquare.geometry), false,
            'pointInPolygon: just outside left edge');

assertEqual(geo.featureBbox(unitSquare), [0, 0, 1, 1],
            'featureBbox: unit square');

const twoSquares = [
  { geometry: { type: 'Polygon',
                coordinates: [[[0,0],[2,0],[2,3],[0,3],[0,0]]] } },
  { geometry: { type: 'Polygon',
                coordinates: [[[5,5],[10,5],[10,10],[5,10],[5,5]]] } },
];
geo.buildBboxIndex(twoSquares);
assertEqual(twoSquares[0].bbox, [0, 0, 2, 3], 'buildBboxIndex: feature 0');
assertEqual(twoSquares[1].bbox, [5, 5, 10, 10], 'buildBboxIndex: feature 1');

assertEqual(geo.findContainingParcel(1, 1, twoSquares) === twoSquares[0], true,
            'findContainingParcel: hits feature 0 (bbox-prefiltered)');
assertEqual(geo.findContainingParcel(7, 7, twoSquares) === twoSquares[1], true,
            'findContainingParcel: hits feature 1 (bbox-prefiltered)');
assertEqual(geo.findContainingParcel(20, 20, twoSquares), null,
            'findContainingParcel: outside all features');

// 1° × 1° square at lat 45. Centroid (0.5, 45.5): nearest edges are
// left/right walls at 0.5° × 111 km/° × cos(lat). Expected ≈ 38.94 km.
const square45 = {
  geometry: { type: 'Polygon',
              coordinates: [[[0,45],[1,45],[1,46],[0,46],[0,45]]] },
};
const expectedCentroidDist = 0.5 * 111132.92 * Math.cos(45.5 * Math.PI / 180);
assertClose(geo.distanceToBoundaryMeters(0.5, 45.5, square45),
            expectedCentroidDist, 1,
            'distanceToBoundaryMeters: 1° square at lat 45, centroid');
assertClose(geo.distanceToBoundaryMeters(0, 45, square45), 0, 0.001,
            'distanceToBoundaryMeters: on a vertex');

assertEqual(geo.parcelLabel({ properties: { layer: 'Capistrano',
                                            name: 'Capistrano-3a' } }),
            'Capistrano 3a', 'parcelLabel: standard Compresa-Particella');
assertEqual(geo.parcelTypeLabel({ properties: { coppice: false } }), 'Fustaia',
            'parcelTypeLabel: high forest');
assertEqual(geo.parcelTypeLabel({ properties: { coppice: true } }), 'Ceduo',
            'parcelTypeLabel: coppice');
assertEqual(geo.parcelLabel({ properties: { layer: 'Capistrano',
                                            name: 'Capistrano-3a',
                                            coppice: false } }),
            'Capistrano 3a\nFustaia', 'parcelLabel: enriched high-forest parcel');
assertEqual(geo.parcelLabel({ properties: { layer: '', name: '' } }), '',
            'parcelLabel: empty properties');
assertEqual(geo.parcelLabel({}), '', 'parcelLabel: no properties');

// generateGrid emits Leaflet-convention points {lat, lng, compresa,
// particella} — NOT `lon`.  Consumers that persist to the `lon` schema
// (the campionamenti grid-planner) must translate lng→lon at the boundary.
const gridPts = geo.generateGrid([unitSquare], 0.5, 0.5);
assertEqual(gridPts.length > 0, true,
            'generateGrid: yields at least one interior point');
const gp = gridPts[0];
assertEqual([typeof gp.lat, typeof gp.lng], ['number', 'number'],
            'generateGrid: point carries numeric lat + lng (Leaflet convention)');
assertEqual(gp.lon, undefined,
            'generateGrid: point has no `lon` key (consumers map lng→lon)');
assertEqual([gp.compresa, gp.particella], ['X', '1'],
            'generateGrid: compresa=layer, particella=name after the dash');

// --- geodesicArea / geoJSONFeatureArea / sortFeaturesByArea -----------------
// Moved here from map-common.js so the geometry layer is self-contained.

// ringToLatLngs: GeoJSON [lng,lat] → Leaflet {lat,lng}.
assertEqual(geo.ringToLatLngs([[0, 1], [2, 3]]),
            [{ lat: 1, lng: 0 }, { lat: 3, lng: 2 }],
            'ringToLatLngs: [lng,lat] → {lat,lng}');

// 1° × 1° square centred on the equator. A degree of latitude is ~111 km, so
// the area is ~111 km × 111 km ≈ 1.23e10 m². The loose tolerance is deliberate:
// it still catches a degrees-vs-radians (~3283×) or wrong-radius bug, while the
// hole/sort tests below pin down the compositional logic exactly.
const eqSquare = [
  { lat: -0.5, lng: 0 }, { lat: -0.5, lng: 1 },
  { lat:  0.5, lng: 1 }, { lat:  0.5, lng: 0 },
];
assertClose(geo.geodesicArea(eqSquare), 1.23e10, 2e8,
            'geodesicArea: 1° square on the equator ≈ 1.23e10 m²');

// geoJSONFeatureArea subtracts holes (relative check — exact by construction).
const outerRing = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]];
const holeRing  = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]];
const outerA = geo.geodesicArea(geo.ringToLatLngs(outerRing));
const holeA  = geo.geodesicArea(geo.ringToLatLngs(holeRing));
assertClose(geo.geoJSONFeatureArea({ geometry: { type: 'Polygon',
              coordinates: [outerRing, holeRing] } }),
            outerA - holeA, 1, 'geoJSONFeatureArea: exterior minus hole');
assertClose(geo.geoJSONFeatureArea({ geometry: { type: 'MultiPolygon',
              coordinates: [[outerRing], [holeRing]] } }),
            outerA + holeA, 1, 'geoJSONFeatureArea: MultiPolygon sums polygons');
assertEqual(geo.geoJSONFeatureArea({ geometry: null }), 0,
            'geoJSONFeatureArea: no geometry → 0');

// sortFeaturesByArea: largest-first, annotating each feature with _areaM2.
const big   = { properties: {}, geometry: { type: 'Polygon', coordinates: [outerRing] } };
const small = { properties: {}, geometry: { type: 'Polygon', coordinates: [holeRing] } };
const sorted = geo.sortFeaturesByArea({ type: 'FeatureCollection', features: [small, big] });
assertEqual(sorted.features[0] === big, true,
            'sortFeaturesByArea: largest feature first');
assertEqual(typeof big.properties._areaM2 === 'number'
            && big.properties._areaM2 > small.properties._areaM2, true,
            'sortFeaturesByArea: annotates _areaM2 (descending)');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
