// Pure computational functions. No DOM or Leaflet dependencies, testable with Node.js.

'use strict';

const MIN_DIFF_RANGE = 0.01; // floor to prevent division by zero in normalization

// Interpolate through a ramp of [position, [r, g, b]] stops.
// Returns [r, g, b] array.
function colormapLookup(ramp, val) {
    if (val <= ramp[0][0]) return ramp[0][1];
    for (let i = 1; i < ramp.length; i++) {
        if (val <= ramp[i][0]) {
            const t = (val - ramp[i - 1][0]) / (ramp[i][0] - ramp[i - 1][0]);
            const a = ramp[i - 1][1], b = ramp[i][1];
            return [
                Math.round(a[0] + t * (b[0] - a[0])),
                Math.round(a[1] + t * (b[1] - a[1])),
                Math.round(a[2] + t * (b[2] - a[2])),
            ];
        }
    }
    return ramp[ramp.length - 1][1];
}

// Decode uint8 [0,255] to real index [-1, +1].
function uint8ToIndex(v) {
    return v / 127.5 - 1;
}

// Linear interpolation between two [r, g, b] colors.
// t in [0, 1]: 0 = low, 1 = high.
function interpolateColor(t, low, high) {
    return [
        Math.round(low[0] + t * (high[0] - low[0])),
        Math.round(low[1] + t * (high[1] - low[1])),
        Math.round(low[2] + t * (high[2] - low[2])),
    ];
}

// Compute pixel-wise difference between two uint8 rasters.
// If mask is provided (Uint8Array, >0 inside), only masked pixels affect the range.
// Returns { diff: Float32Array, minDiff, maxDiff, maxAbs }.
function computeDiff(raster1, raster2, mask) {
    const n = raster1.length;
    const diff = new Float32Array(n);
    let minDiff = Infinity, maxDiff = -Infinity;

    for (let i = 0; i < n; i++) {
        const d = uint8ToIndex(raster2[i]) - uint8ToIndex(raster1[i]);
        diff[i] = d;
        if (!mask || mask[i]) {
            if (d < minDiff) minDiff = d;
            if (d > maxDiff) maxDiff = d;
        }
    }

    const maxAbs = Math.max(Math.abs(minDiff), Math.abs(maxDiff)) || MIN_DIFF_RANGE;
    return { diff, minDiff, maxDiff, maxAbs };
}

// Normalize a diff value to uint8 [0, 255] for colormap lookup.
// diff=0 -> 128, diff=+maxAbs -> 255, diff=-maxAbs -> 0.
function normalizeDiff(diffVal, maxAbs) {
    const normalized = Math.round(((diffVal / maxAbs) + 1) * 127.5);
    return Math.max(0, Math.min(255, normalized));
}

// Convert geographic coordinates to pixel coordinates.
// bbox = { south, west, north, east }.
function geoToPixel(lon, lat, bbox, width, height) {
    const x = (lon - bbox.west) / (bbox.east - bbox.west) * width;
    const y = (bbox.north - lat) / (bbox.north - bbox.south) * height;
    return { x, y };
}

if (typeof module !== 'undefined') {
    module.exports = {
        colormapLookup, uint8ToIndex, interpolateColor,
        computeDiff, normalizeDiff, geoToPixel,
    };
}
