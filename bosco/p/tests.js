// Tests for p/compute.js — pure computational functions.
// Run with: node bosco/p/tests.js

'use strict';

const {
    colormapLookup, interpolateColor,
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

    // Identical rasters -> all zeros, maxAbs = 1 (floor)
    const r1 = new Uint8Array([100, 100, 100]);
    const r2 = new Uint8Array([100, 100, 100]);
    let result = computeDiff(r1, r2, null);
    console.assert(result.diff[0] === 0, 'identical diff[0]');
    console.assert(result.diff[1] === 0, 'identical diff[1]');
    console.assert(result.maxAbs === 1, 'identical maxAbs floor');

    // One pixel brighter in raster2
    const r3 = new Uint8Array([100, 100, 100]);
    const r4 = new Uint8Array([100, 200, 100]);
    result = computeDiff(r3, r4, null);
    console.assert(result.diff[1] === 100, 'brighter pixel diff = 100');
    console.assert(result.maxDiff === 100, 'maxDiff matches bright pixel');
    console.assert(result.minDiff === 0, 'minDiff is 0 for unchanged pixels');

    // With mask: only masked pixels affect min/max
    const r5 = new Uint8Array([50, 100, 200]);
    const r6 = new Uint8Array([200, 100, 50]);
    const mask = new Uint8Array([0, 1, 0]); // only middle pixel is masked
    result = computeDiff(r5, r6, mask);
    // Middle pixel: 100 - 100 = 0
    console.assert(result.minDiff === 0, 'masked minDiff');
    console.assert(result.maxDiff === 0, 'masked maxDiff');
    console.assert(result.maxAbs === 1, 'masked maxAbs floor');
    // Unmasked pixels still have their diff values computed
    console.assert(result.diff[0] === 150, 'unmasked pixel 0: 200-50=150');
    console.assert(result.diff[2] === -150, 'unmasked pixel 2: 50-200=-150');

    // Symmetric: maxAbs = max(|min|, |max|)
    const r7 = new Uint8Array([0,   255]);
    const r8 = new Uint8Array([255, 0]);
    result = computeDiff(r7, r8, null);
    console.assert(result.maxAbs === 255, 'full range maxAbs = 255');
    console.assert(result.minDiff === -255, 'full range minDiff = -255');
    console.assert(result.maxDiff === 255, 'full range maxDiff = 255');

    console.log('  computeDiff: PASS');
}

// ---------------------------------------------------------------------------
// normalizeDiff
// ---------------------------------------------------------------------------

function testNormalizeDiff() {
    console.log('Testing normalizeDiff...');

    const maxAbs = 100;

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
