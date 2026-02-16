// Tests for p/compute.js — pure computational functions.
// Run with: node bosco/p/tests.js

'use strict';

const {
    colormapLookup, uint8ToIndex, interpolateColor,
    computeDiff, normalizeDiff, geoToPixel,
} = require('./compute.js');

function assertClose(actual, expected, tolerance, msg) {
    const diff = Math.abs(actual - expected);
    console.assert(diff <= tolerance,
        `${msg}: expected ~${expected}, got ${actual} (diff ${diff})`);
}

function assertArrayClose(actual, expected, tolerance, msg) {
    for (let i = 0; i < expected.length; i++) {
        assertClose(actual[i], expected[i], tolerance, `${msg}[${i}]`);
    }
}

// ---------------------------------------------------------------------------
// colormapLookup
// ---------------------------------------------------------------------------

function testColormapLookup() {
    console.log('Testing colormapLookup...');

    const twoStop = [
        [0,   [0, 0, 0]],
        [255, [255, 255, 255]],
    ];

    // Exact stops
    assertArrayClose(colormapLookup(twoStop, 0), [0, 0, 0], 0, 'first stop');
    assertArrayClose(colormapLookup(twoStop, 255), [255, 255, 255], 0, 'last stop');

    // Midpoint
    assertArrayClose(colormapLookup(twoStop, 127.5), [128, 128, 128], 1, 'midpoint');

    // Below first stop -> clamps to first
    assertArrayClose(colormapLookup(twoStop, -10), [0, 0, 0], 0, 'below first');

    // Above last stop -> clamps to last
    assertArrayClose(colormapLookup(twoStop, 300), [255, 255, 255], 0, 'above last');

    // Three-stop ramp (matches INDEX_RAMP in app.js)
    const threeStop = [
        [0,   [139, 90, 43]],
        [128, [245, 235, 200]],
        [255, [0, 100, 0]],
    ];

    assertArrayClose(colormapLookup(threeStop, 0), [139, 90, 43], 0, 'three-stop first');
    assertArrayClose(colormapLookup(threeStop, 128), [245, 235, 200], 0, 'three-stop mid');
    assertArrayClose(colormapLookup(threeStop, 255), [0, 100, 0], 0, 'three-stop last');

    // Between first and second stop (val=64, t=0.5 of first segment)
    const mid01 = colormapLookup(threeStop, 64);
    assertClose(mid01[0], (139 + 245) / 2, 1, 'three-stop interp r');
    assertClose(mid01[1], (90 + 235) / 2, 1, 'three-stop interp g');
    assertClose(mid01[2], (43 + 200) / 2, 1, 'three-stop interp b');

    console.log('  colormapLookup: PASS');
}

// ---------------------------------------------------------------------------
// uint8ToIndex
// ---------------------------------------------------------------------------

function testUint8ToIndex() {
    console.log('Testing uint8ToIndex...');

    assertClose(uint8ToIndex(0), -1.0, 1e-6, 'val 0');
    assertClose(uint8ToIndex(255), 1.0, 1e-6, 'val 255');
    // 128 -> 128/127.5 - 1 = 0.003921...
    assertClose(uint8ToIndex(128), 128 / 127.5 - 1, 1e-6, 'val 128');
    // 127.5 would map to exactly 0, but uint8 can't represent it; verify neighbors
    assertClose(uint8ToIndex(127), 127 / 127.5 - 1, 1e-6, 'val 127');

    console.log('  uint8ToIndex: PASS');
}

// ---------------------------------------------------------------------------
// interpolateColor
// ---------------------------------------------------------------------------

function testInterpolateColor() {
    console.log('Testing interpolateColor...');

    const low = [0, 100, 0];
    const high = [154, 205, 50];

    assertArrayClose(interpolateColor(0, low, high), low, 0, 't=0');
    assertArrayClose(interpolateColor(1, low, high), high, 0, 't=1');

    const mid = interpolateColor(0.5, low, high);
    assertClose(mid[0], 77, 1, 't=0.5 r');
    assertClose(mid[1], 153, 1, 't=0.5 g');  // (100+205)/2 = 152.5
    assertClose(mid[2], 25, 1, 't=0.5 b');

    console.log('  interpolateColor: PASS');
}

// ---------------------------------------------------------------------------
// computeDiff
// ---------------------------------------------------------------------------

function testComputeDiff() {
    console.log('Testing computeDiff...');

    // Identical rasters -> all zeros, maxAbs = 0.01 (floor)
    const r1 = new Uint8Array([100, 100, 100]);
    const r2 = new Uint8Array([100, 100, 100]);
    let result = computeDiff(r1, r2, null);
    assertClose(result.diff[0], 0, 1e-6, 'identical diff[0]');
    assertClose(result.diff[1], 0, 1e-6, 'identical diff[1]');
    assertClose(result.maxAbs, 0.01, 1e-6, 'identical maxAbs floor');

    // One pixel brighter in raster2
    const r3 = new Uint8Array([100, 100, 100]);
    const r4 = new Uint8Array([100, 200, 100]);
    result = computeDiff(r3, r4, null);
    console.assert(result.diff[1] > 0, 'brighter pixel should have positive diff');
    assertClose(result.maxDiff, result.diff[1], 1e-6, 'maxDiff matches bright pixel');

    // With mask: only masked pixels affect min/max
    const r5 = new Uint8Array([50, 100, 200]);
    const r6 = new Uint8Array([200, 100, 50]);
    const mask = new Uint8Array([0, 1, 0]); // only middle pixel is masked
    result = computeDiff(r5, r6, mask);
    // Middle pixel: uint8ToIndex(100) - uint8ToIndex(100) = 0
    assertClose(result.minDiff, 0, 1e-6, 'masked minDiff');
    assertClose(result.maxDiff, 0, 1e-6, 'masked maxDiff');
    assertClose(result.maxAbs, 0.01, 1e-6, 'masked maxAbs floor');
    // But unmasked pixels still have their diff values computed
    console.assert(result.diff[0] > 0, 'unmasked pixel 0 has diff');
    console.assert(result.diff[2] < 0, 'unmasked pixel 2 has diff');

    // Symmetric: maxAbs = max(|min|, |max|)
    const r7 = new Uint8Array([0,   255]);
    const r8 = new Uint8Array([255, 0]);
    result = computeDiff(r7, r8, null);
    assertClose(result.maxAbs, Math.max(Math.abs(result.minDiff), Math.abs(result.maxDiff)),
        1e-6, 'maxAbs is symmetric max');

    console.log('  computeDiff: PASS');
}

// ---------------------------------------------------------------------------
// normalizeDiff
// ---------------------------------------------------------------------------

function testNormalizeDiff() {
    console.log('Testing normalizeDiff...');

    const maxAbs = 0.5;

    // diff=0 -> middle (128)
    assertClose(normalizeDiff(0, maxAbs), 128, 1, 'zero diff');

    // diff=+maxAbs -> 255
    assertClose(normalizeDiff(maxAbs, maxAbs), 255, 0, 'positive max');

    // diff=-maxAbs -> 0
    assertClose(normalizeDiff(-maxAbs, maxAbs), 0, 0, 'negative max');

    // diff=+maxAbs/2 -> ~191
    assertClose(normalizeDiff(maxAbs / 2, maxAbs), 191, 1, 'half positive');

    console.log('  normalizeDiff: PASS');
}

// ---------------------------------------------------------------------------
// geoToPixel
// ---------------------------------------------------------------------------

function testGeoToPixel() {
    console.log('Testing geoToPixel...');

    const bbox = { south: 39.0, west: 16.0, north: 40.0, east: 17.0 };
    const width = 100, height = 200;

    // Southwest corner -> (0, height)
    let p = geoToPixel(16.0, 39.0, bbox, width, height);
    assertClose(p.x, 0, 1e-6, 'SW x');
    assertClose(p.y, height, 1e-6, 'SW y');

    // Northeast corner -> (width, 0)
    p = geoToPixel(17.0, 40.0, bbox, width, height);
    assertClose(p.x, width, 1e-6, 'NE x');
    assertClose(p.y, 0, 1e-6, 'NE y');

    // Center -> (width/2, height/2)
    p = geoToPixel(16.5, 39.5, bbox, width, height);
    assertClose(p.x, width / 2, 1e-6, 'center x');
    assertClose(p.y, height / 2, 1e-6, 'center y');

    console.log('  geoToPixel: PASS');
}

// ---------------------------------------------------------------------------
// Run all tests
// ---------------------------------------------------------------------------

function runTests() {
    console.log('=== p/compute.js Tests ===\n');

    try {
        testColormapLookup();
        testUint8ToIndex();
        testInterpolateColor();
        testComputeDiff();
        testNormalizeDiff();
        testGeoToPixel();
        console.log('\n=== All tests passed ===');
    } catch (err) {
        console.error('\nTest failed:', err.message);
        console.error(err.stack);
        process.exit(1);
    }
}

runTests();
