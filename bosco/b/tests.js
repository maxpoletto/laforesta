// Tests for p/compute.js — pure computational functions.
// Run with: node bosco/p/tests.js

'use strict';

const {
    colormapLookup, interpolateColor,
    computeDiff, diffToRgba,
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
// diffToRgba
// ---------------------------------------------------------------------------

function testDiffToRgba() {
    console.log('Testing diffToRgba...');

    // Simple black-to-white ramp: normalized 0 -> black, 255 -> white
    const bwRamp = [
        [0,   [0, 0, 0]],
        [255, [255, 255, 255]],
    ];

    // Zero diff -> normalized to 128 -> mid-gray
    const zeroDiff = new Int16Array([0]);
    let rgba = diffToRgba(zeroDiff, 100, bwRamp, null, 210, 60);
    assertClose(rgba[0], 128, 1, 'zero diff -> mid-gray R');
    assertClose(rgba[1], 128, 1, 'zero diff -> mid-gray G');
    assertClose(rgba[2], 128, 1, 'zero diff -> mid-gray B');
    console.assert(rgba[3] === 210, 'no mask -> insideAlpha');

    // +maxAbs -> normalized to 255 -> white
    const posDiff = new Int16Array([100]);
    rgba = diffToRgba(posDiff, 100, bwRamp, null, 210, 60);
    console.assert(rgba[0] === 255, '+maxAbs -> white R');
    console.assert(rgba[3] === 210, '+maxAbs alpha');

    // -maxAbs -> normalized to 0 -> black
    const negDiff = new Int16Array([-100]);
    rgba = diffToRgba(negDiff, 100, bwRamp, null, 210, 60);
    console.assert(rgba[0] === 0, '-maxAbs -> black R');

    // Mask: inside pixel gets insideAlpha, outside gets outsideAlpha
    const twoDiff = new Int16Array([0, 0]);
    const mask = new Uint8Array([1, 0]);
    rgba = diffToRgba(twoDiff, 100, bwRamp, mask, 210, 60);
    console.assert(rgba[3] === 210, 'masked inside alpha');
    console.assert(rgba[7] === 60, 'masked outside alpha');

    // Multiple pixels with diverging ramp (matches DIFF_RAMP structure)
    const diffRamp = [
        [0,   [180, 30, 30]],
        [128, [255, 255, 255]],
        [255, [30, 130, 30]],
    ];
    const multiDiff = new Int16Array([-50, 0, 50]);
    rgba = diffToRgba(multiDiff, 50, diffRamp, null, 255, 0);
    // -maxAbs -> ramp position 0 -> red
    assertClose(rgba[0], 180, 1, 'negative -> red R');
    // 0 -> ramp position 128 -> white
    assertClose(rgba[4], 255, 1, 'zero -> white R');
    assertClose(rgba[5], 255, 1, 'zero -> white G');
    // +maxAbs -> ramp position 255 -> green
    assertClose(rgba[8], 30, 1, 'positive -> green R');
    assertClose(rgba[9], 130, 1, 'positive -> green G');

    console.log('  diffToRgba: PASS');
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
        testDiffToRgba();
        console.log('\n=== All tests passed ===');
    } catch (err) {
        console.error('\nTest failed:', err.message);
        console.error(err.stack);
        process.exit(1);
    }
}

runTests();
