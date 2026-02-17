// Test algorithms for polygon/line manipulation tools
// Run with: node tests.js

'use strict';

// Minimal mocks so map-common.js can be loaded in Node
// (the IIFE only references browser globals when create() is called)
global.document = { getElementById: () => null };
const MapCommon = require('../a/map-common.js');
const { geodesicArea } = MapCommon;

// Mock the path index function
function getPathIndices(from, to, n, direction) {
    const indices = [from];
    let current = from;
    while (current !== to) {
        current = (current + direction + n) % n;
        indices.push(current);
    }
    return indices;
}

// Test getPathIndices
function testGetPathIndices() {
    console.log('Testing getPathIndices...');

    // Square: 0-1-2-3 (clockwise: 0->1->2->3->0)
    const n = 4;

    // From 0 to 2 clockwise should be [0, 1, 2]
    let path = getPathIndices(0, 2, n, 1);
    console.assert(JSON.stringify(path) === '[0,1,2]', `Expected [0,1,2], got ${JSON.stringify(path)}`);

    // From 0 to 2 counter-clockwise should be [0, 3, 2]
    path = getPathIndices(0, 2, n, -1);
    console.assert(JSON.stringify(path) === '[0,3,2]', `Expected [0,3,2], got ${JSON.stringify(path)}`);

    // From 1 to 3 clockwise should be [1, 2, 3]
    path = getPathIndices(1, 3, n, 1);
    console.assert(JSON.stringify(path) === '[1,2,3]', `Expected [1,2,3], got ${JSON.stringify(path)}`);

    // From 3 to 1 clockwise should be [3, 0, 1]
    path = getPathIndices(3, 1, n, 1);
    console.assert(JSON.stringify(path) === '[3,0,1]', `Expected [3,0,1], got ${JSON.stringify(path)}`);

    // Pentagon: 0-1-2-3-4
    const n5 = 5;
    path = getPathIndices(0, 3, n5, 1);
    console.assert(JSON.stringify(path) === '[0,1,2,3]', `Expected [0,1,2,3], got ${JSON.stringify(path)}`);

    path = getPathIndices(0, 3, n5, -1);
    console.assert(JSON.stringify(path) === '[0,4,3]', `Expected [0,4,3], got ${JSON.stringify(path)}`);

    console.log('  getPathIndices: PASS');
}

// Simulate executeSnip algorithm (simplified: delete one vertex, open polygon to line)
function simulateSnip(coords, deleteIdx) {
    const n = coords.length;

    if (n < 4) {
        throw new Error('Polygon too small to open');
    }

    // Build line coordinates starting from vertex after deleted, going all the way around
    // Example: A-B-C-D-E-F-A, delete C -> line D-E-F-A-B
    const lineCoords = [];
    for (let i = 1; i < n; i++) {
        const idx = (deleteIdx + i) % n;
        lineCoords.push(coords[idx]);
    }

    return { type: 'line', coords: lineCoords };
}

function testSnip() {
    console.log('Testing snip algorithm...');

    // Square: A(0,0) - B(1,0) - C(1,1) - D(0,1) (indices 0-1-2-3)
    const square = [[0, 0], [1, 0], [1, 1], [0, 1]];

    // Delete vertex C (index 2) -> line D-A-B
    let result = simulateSnip(square, 2);
    console.assert(result.type === 'line', 'Should produce line');
    console.assert(result.coords.length === 3, `Expected 3 coords, got ${result.coords.length}`);
    console.assert(JSON.stringify(result.coords) === '[[0,1],[0,0],[1,0]]',
        `Expected [[0,1],[0,0],[1,0]], got ${JSON.stringify(result.coords)}`);

    // Delete vertex A (index 0) -> line B-C-D
    result = simulateSnip(square, 0);
    console.assert(result.coords.length === 3, `Expected 3 coords, got ${result.coords.length}`);
    console.assert(JSON.stringify(result.coords) === '[[1,0],[1,1],[0,1]]',
        `Expected [[1,0],[1,1],[0,1]], got ${JSON.stringify(result.coords)}`);

    // Pentagon: A(0,0) - B(1,0) - C(2,1) - D(1,2) - E(0,1)
    const pentagon = [[0, 0], [1, 0], [2, 1], [1, 2], [0, 1]];

    // Delete vertex C (index 2) -> line D-E-A-B
    result = simulateSnip(pentagon, 2);
    console.assert(result.coords.length === 4, `Expected 4 coords, got ${result.coords.length}`);
    console.assert(JSON.stringify(result.coords) === '[[1,2],[0,1],[0,0],[1,0]]',
        `Expected [[1,2],[0,1],[0,0],[1,0]], got ${JSON.stringify(result.coords)}`);

    console.log('  snip: PASS');
}

// Simulate executeClose algorithm
function simulateClose(coords, lv1Idx, lv2Idx) {
    // Ensure lv1 comes before lv2
    let idx1 = lv1Idx;
    let idx2 = lv2Idx;
    if (idx1 > idx2) {
        [idx1, idx2] = [idx2, idx1];
    }

    const polygonCoords = coords.slice(idx1, idx2 + 1);
    if (polygonCoords.length < 3) {
        throw new Error('Too few vertices for polygon');
    }

    const result = {
        polygon: polygonCoords,
        leftovers: []
    };

    // Create leftover lines if any
    if (idx1 > 0) {
        const leftCoords = coords.slice(0, idx1 + 1);
        if (leftCoords.length >= 2) {
            result.leftovers.push(leftCoords);
        }
    }

    if (idx2 < coords.length - 1) {
        const rightCoords = coords.slice(idx2);
        if (rightCoords.length >= 2) {
            result.leftovers.push(rightCoords);
        }
    }

    return result;
}

function testClose() {
    console.log('Testing close algorithm...');

    // Line: A(0,0) - B(1,0) - C(2,0) - D(3,0) - E(4,0)
    const line = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];

    // Close from B(1) to D(3)
    // Should produce polygon B-C-D and leftovers A-B, D-E
    let result = simulateClose(line, 1, 3);
    console.assert(result.polygon.length === 3, `Expected 3-point polygon, got ${result.polygon.length}`);
    console.assert(JSON.stringify(result.polygon) === '[[1,0],[2,0],[3,0]]',
        `Expected [[1,0],[2,0],[3,0]], got ${JSON.stringify(result.polygon)}`);
    console.assert(result.leftovers.length === 2, `Expected 2 leftovers, got ${result.leftovers.length}`);
    console.assert(JSON.stringify(result.leftovers[0]) === '[[0,0],[1,0]]',
        `Expected [[0,0],[1,0]], got ${JSON.stringify(result.leftovers[0])}`);
    console.assert(JSON.stringify(result.leftovers[1]) === '[[3,0],[4,0]]',
        `Expected [[3,0],[4,0]], got ${JSON.stringify(result.leftovers[1])}`);

    // Close entire line from A(0) to E(4) - no leftovers
    result = simulateClose(line, 0, 4);
    console.assert(result.polygon.length === 5, `Expected 5-point polygon, got ${result.polygon.length}`);
    console.assert(result.leftovers.length === 0, `Expected 0 leftovers, got ${result.leftovers.length}`);

    // Close with reversed indices (should still work)
    result = simulateClose(line, 3, 1);
    console.assert(result.polygon.length === 3, `Expected 3-point polygon with reversed indices`);

    console.log('  close: PASS');
}

// Simulate executeSplit algorithm
function simulateSplit(coords, splitIdx) {
    if (splitIdx === 0 || splitIdx === coords.length - 1) {
        throw new Error('Cannot split at endpoints');
    }

    const leftCoords = coords.slice(0, splitIdx);
    const rightCoords = coords.slice(splitIdx + 1);

    if (leftCoords.length < 2 || rightCoords.length < 2) {
        throw new Error('Split would create lines too short');
    }

    return { left: leftCoords, right: rightCoords };
}

function testSplit() {
    console.log('Testing split algorithm...');

    // Line: A(0,0) - B(1,0) - C(2,0) - D(3,0) - E(4,0)
    const line = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];

    // Split at C(2) -> left: A-B, right: D-E
    let result = simulateSplit(line, 2);
    console.assert(result.left.length === 2, `Expected left with 2 points, got ${result.left.length}`);
    console.assert(result.right.length === 2, `Expected right with 2 points, got ${result.right.length}`);
    console.assert(JSON.stringify(result.left) === '[[0,0],[1,0]]',
        `Expected [[0,0],[1,0]], got ${JSON.stringify(result.left)}`);
    console.assert(JSON.stringify(result.right) === '[[3,0],[4,0]]',
        `Expected [[3,0],[4,0]], got ${JSON.stringify(result.right)}`);

    // Split at B(1) -> left: A, right: C-D-E (left too short)
    try {
        simulateSplit(line, 1);
        console.assert(false, 'Should have thrown error for short left segment');
    } catch (e) {
        // Expected
    }

    // Split at endpoint should fail
    try {
        simulateSplit(line, 0);
        console.assert(false, 'Should have thrown error for endpoint split');
    } catch (e) {
        // Expected
    }

    console.log('  split: PASS');
}

// Simulate executeJoin algorithm (simplified: connect at endpoints only)
function simulateJoin(coords1, coords2, e1Idx, e2Idx) {
    // e1Idx is endpoint from line 1 (must be 0 or last)
    // e2Idx is endpoint from line 2 (must be 0 or last)

    let segment1, segment2;

    // If e1 is at the end of line1 (last index), use line1 as-is
    // If e1 is at the start of line1 (index 0), reverse line1
    if (e1Idx === 0) {
        segment1 = coords1.slice().reverse();
    } else {
        segment1 = coords1.slice();
    }

    // If e2 is at the start of line2 (index 0), use line2 as-is
    // If e2 is at the end of line2 (last index), reverse line2
    if (e2Idx === 0) {
        segment2 = coords2.slice();
    } else {
        segment2 = coords2.slice().reverse();
    }

    return [...segment1, ...segment2];
}

function testJoin() {
    console.log('Testing join algorithm...');

    // Line 1: A(0,0) - B(1,0) - C(2,0)
    const line1 = [[0, 0], [1, 0], [2, 0]];
    // Line 2: D(3,0) - E(4,0) - F(5,0)
    const line2 = [[3, 0], [4, 0], [5, 0]];

    // Connect end of line1 (C) to start of line2 (D)
    // Result: A-B-C-D-E-F
    let result = simulateJoin(line1, line2, 2, 0);
    console.assert(result.length === 6, `Expected 6 points, got ${result.length}`);
    console.assert(JSON.stringify(result) === '[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]]',
        `Expected A-B-C-D-E-F, got ${JSON.stringify(result)}`);

    // Connect start of line1 (A) to end of line2 (F)
    // Result: C-B-A-F-E-D (both reversed)
    result = simulateJoin(line1, line2, 0, 2);
    console.assert(result.length === 6, `Expected 6 points, got ${result.length}`);
    console.assert(JSON.stringify(result) === '[[2,0],[1,0],[0,0],[5,0],[4,0],[3,0]]',
        `Expected C-B-A-F-E-D, got ${JSON.stringify(result)}`);

    // Connect end of line1 (C) to end of line2 (F)
    // Result: A-B-C-F-E-D (line2 reversed)
    result = simulateJoin(line1, line2, 2, 2);
    console.assert(result.length === 6, `Expected 6 points, got ${result.length}`);
    console.assert(JSON.stringify(result) === '[[0,0],[1,0],[2,0],[5,0],[4,0],[3,0]]',
        `Expected A-B-C-F-E-D, got ${JSON.stringify(result)}`);

    // Connect start of line1 (A) to start of line2 (D)
    // Result: C-B-A-D-E-F (line1 reversed)
    result = simulateJoin(line1, line2, 0, 0);
    console.assert(result.length === 6, `Expected 6 points, got ${result.length}`);
    console.assert(JSON.stringify(result) === '[[2,0],[1,0],[0,0],[3,0],[4,0],[5,0]]',
        `Expected C-B-A-D-E-F, got ${JSON.stringify(result)}`);

    console.log('  join: PASS');
}

// --- Area calculation tests ---

const DEG_TO_RAD = Math.PI / 180;

// Local-projection Shoelace: project to meters from bounding-box lower-left, then Shoelace
function shoelaceAreaM2(latlngs) {
    // Find lower-left corner of bounding box
    let minLat = Infinity, minLng = Infinity;
    for (const p of latlngs) {
        if (p.lat < minLat) minLat = p.lat;
        if (p.lng < minLng) minLng = p.lng;
    }

    // Convert to local meters (offset from lower-left)
    const cosLat = Math.cos(minLat * DEG_TO_RAD);
    const mPerDegLat = 111132.92;
    const mPerDegLng = 111132.92 * cosLat;

    const pts = latlngs.map(p => ({
        x: (p.lng - minLng) * mPerDegLng,
        y: (p.lat - minLat) * mPerDegLat,
    }));

    // Shoelace formula
    let area = 0;
    const n = pts.length;
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        area += pts[i].x * pts[j].y;
        area -= pts[j].x * pts[i].y;
    }
    return Math.abs(area / 2);
}

function testArea() {
    console.log('Testing area calculations...');

    // Pentagon in Calabria (~39°N, 16.5°E), roughly 500m across
    const center = { lat: 39.0, lng: 16.5 };
    const r = 0.003; // ~300m in degrees
    const pentagon = [];
    for (let i = 0; i < 5; i++) {
        const angle = (2 * Math.PI * i / 5) - Math.PI / 2;
        pentagon.push({
            lat: center.lat + r * Math.cos(angle),
            lng: center.lng + r * Math.sin(angle)
        });
    }

    const geodesic = geodesicArea(pentagon);
    const shoelace = shoelaceAreaM2(pentagon);
    const geodesicHa = geodesic / 10000;
    const shoelaceHa = shoelace / 10000;
    const relError = Math.abs(geodesic - shoelace) / shoelace;

    console.log(`  Pentagon (~300m radius at 39°N):`);
    console.log(`    geodesicArea: ${geodesicHa.toFixed(4)} ha (${geodesic.toFixed(1)} m²)`);
    console.log(`    shoelace:     ${shoelaceHa.toFixed(4)} ha (${shoelace.toFixed(1)} m²)`);
    console.log(`    relative error: ${(relError * 100).toFixed(4)}%`);
    console.assert(relError < 0.01, `Error too large: ${(relError * 100).toFixed(4)}%`);

    // Large polygon: ~5km across (agricultural scale)
    const bigR = 0.03;
    const bigHex = [];
    for (let i = 0; i < 6; i++) {
        const angle = (2 * Math.PI * i / 6);
        bigHex.push({
            lat: center.lat + bigR * Math.cos(angle),
            lng: center.lng + bigR * Math.sin(angle)
        });
    }

    const bigGeodesic = geodesicArea(bigHex);
    const bigShoelace = shoelaceAreaM2(bigHex);
    const bigRelError = Math.abs(bigGeodesic - bigShoelace) / bigShoelace;

    console.log(`  Hexagon (~3km radius at 39°N):`);
    console.log(`    geodesicArea: ${(bigGeodesic / 10000).toFixed(4)} ha`);
    console.log(`    shoelace:     ${(bigShoelace / 10000).toFixed(4)} ha`);
    console.log(`    relative error: ${(bigRelError * 100).toFixed(4)}%`);
    console.assert(bigRelError < 0.01, `Error too large: ${(bigRelError * 100).toFixed(4)}%`);

    // Test at equator (different latitude)
    const eqPentagon = [];
    for (let i = 0; i < 5; i++) {
        const angle = (2 * Math.PI * i / 5) - Math.PI / 2;
        eqPentagon.push({
            lat: 0.0 + r * Math.cos(angle),
            lng: 30.0 + r * Math.sin(angle)
        });
    }

    const eqGeodesic = geodesicArea(eqPentagon);
    const eqShoelace = shoelaceAreaM2(eqPentagon);
    const eqRelError = Math.abs(eqGeodesic - eqShoelace) / eqShoelace;

    console.log(`  Pentagon (~300m radius at equator):`);
    console.log(`    geodesicArea: ${(eqGeodesic / 10000).toFixed(4)} ha`);
    console.log(`    shoelace:     ${(eqShoelace / 10000).toFixed(4)} ha`);
    console.log(`    relative error: ${(eqRelError * 100).toFixed(4)}%`);
    console.assert(eqRelError < 0.01, `Error too large: ${(eqRelError * 100).toFixed(4)}%`);

    // Test at high latitude (60°N, Norway)
    const hiPentagon = [];
    for (let i = 0; i < 5; i++) {
        const angle = (2 * Math.PI * i / 5) - Math.PI / 2;
        hiPentagon.push({
            lat: 60.0 + r * Math.cos(angle),
            lng: 10.0 + r * Math.sin(angle)
        });
    }

    const hiGeodesic = geodesicArea(hiPentagon);
    const hiShoelace = shoelaceAreaM2(hiPentagon);
    const hiRelError = Math.abs(hiGeodesic - hiShoelace) / hiShoelace;

    console.log(`  Pentagon (~300m radius at 60°N):`);
    console.log(`    geodesicArea: ${(hiGeodesic / 10000).toFixed(4)} ha`);
    console.log(`    shoelace:     ${(hiShoelace / 10000).toFixed(4)} ha`);
    console.log(`    relative error: ${(hiRelError * 100).toFixed(4)}%`);
    console.assert(hiRelError < 0.01, `Error too large: ${(hiRelError * 100).toFixed(4)}%`);

    console.log('  area: PASS');
}

// Run all tests
function runTests() {
    console.log('=== Algorithm Tests ===\n');

    try {
        testGetPathIndices();
        testSnip();
        testClose();
        testSplit();
        testJoin();
        testArea();
        console.log('\n=== All tests passed ===');
    } catch (err) {
        console.error('\nTest failed:', err.message);
        console.error(err.stack);
        process.exit(1);
    }
}

runTests();
