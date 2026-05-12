/**
 * Geometry helpers — vendored from `bosco/pac/app.js`.
 *
 * Used by the Campionamenti grid planner ("Genera automaticamente")
 * to generate a regular lattice of points inside polygon parcels.
 * Kept in `apps/base/` because future Bosco map work will reuse
 * point-in-polygon and area math.
 */

import MapCommon from './map-common.js';

const DEG_TO_RAD = Math.PI / 180;

export function metersToDegLat(m) { return m / 111132.92; }

export function metersToDegLng(m, lat) {
  return m / (111132.92 * Math.cos(lat * DEG_TO_RAD));
}

/**
 * Ray-casting point-in-ring test for a single GeoJSON linear ring
 * (array of [lng, lat] pairs).
 */
export function pointInRing(lng, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    if (((yi > lat) !== (yj > lat)) &&
        (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Point-in-polygon for a GeoJSON Polygon geometry (exterior + holes).
 */
export function pointInPolygon(lng, lat, geometry) {
  const coords = geometry.coordinates;
  if (!pointInRing(lng, lat, coords[0])) return false;
  for (let i = 1; i < coords.length; i++) {
    if (pointInRing(lng, lat, coords[i])) return false;
  }
  return true;
}

/**
 * Geodesic area of a GeoJSON Polygon feature, in m².
 * Uses MapCommon.geodesicArea (Leaflet.draw's algorithm).
 */
export function featureArea(feature) {
  const ring = feature.geometry.coordinates[0];
  const latlngs = ring.map(c => ({ lat: c[1], lng: c[0] }));
  return MapCommon.geodesicArea(latlngs);
}

/** Bounding box of an array of GeoJSON Polygon features. */
export function boundingBox(features) {
  let minLng = Infinity, minLat = Infinity;
  let maxLng = -Infinity, maxLat = -Infinity;
  for (const f of features) {
    for (const c of f.geometry.coordinates[0]) {
      if (c[0] < minLng) minLng = c[0];
      if (c[0] > maxLng) maxLng = c[0];
      if (c[1] < minLat) minLat = c[1];
      if (c[1] > maxLat) maxLat = c[1];
    }
  }
  return { minLng, minLat, maxLng, maxLat };
}

/** Return the first feature whose polygon contains (lng, lat), or null. */
export function findContainingParcel(lng, lat, features) {
  for (const f of features) {
    if (pointInPolygon(lng, lat, f.geometry)) return f;
  }
  return null;
}

/**
 * Generate a regular grid of interior points across `features`.
 * @returns {Array<{lat, lng, compresa, particella}>} where
 *   `compresa` = feature.properties.layer, `particella` =
 *   short name after the first `-` in feature.properties.name
 *   (the convention used by `bosco/data/terreni.geojson`).
 */
export function generateGrid(features, spacingLng, spacingLat) {
  const bb = boundingBox(features);
  const points = [];
  for (let lat = bb.minLat; lat <= bb.maxLat; lat += spacingLat) {
    for (let lng = bb.minLng; lng <= bb.maxLng; lng += spacingLng) {
      const f = findContainingParcel(lng, lat, features);
      if (f) {
        const name = f.properties.name || '';
        const dash = name.indexOf('-');
        const particella = dash >= 0 ? name.slice(dash + 1) : name;
        points.push({
          lat, lng,
          compresa: f.properties.layer,
          particella,
        });
      }
    }
  }
  return points;
}

/**
 * Binary-search the lattice spacing (in meters) until the resulting
 * point count is within `tolerance` of `targetN`, returning the
 * best-fit point set.  Mirrors `bosco/pac/app.js:202-221`.
 *
 * @param {Array} features — array of GeoJSON Polygon features.
 * @param {number} targetN — desired point count.
 * @param {object} opts — { maxIter, tolerance, hiMeters }.
 *   hiMeters defaults to sqrt(totalAreaM2) — a coarse upper bound.
 */
export function planGridForTarget(features, targetN, opts = {}) {
  const { maxIter = 40, tolerance = 0.05 } = opts;
  const totalAreaM2 = features.reduce((s, f) => s + featureArea(f), 0);
  const bb = boundingBox(features);
  const midLat = (bb.minLat + bb.maxLat) / 2;
  let lo = 1;
  let hi = opts.hiMeters || Math.sqrt(totalAreaM2);
  let bestPoints = [];

  for (let i = 0; i < maxIter; i++) {
    const mid = (lo + hi) / 2;
    const sLat = metersToDegLat(mid);
    const sLng = metersToDegLng(mid, midLat);
    const pts = generateGrid(features, sLng, sLat);
    if (Math.abs(pts.length - targetN) < Math.abs(bestPoints.length - targetN)) {
      bestPoints = pts;
    }
    if (pts.length === targetN) break;
    if (Math.abs(pts.length - targetN) / targetN < tolerance) break;
    if (pts.length > targetN) lo = mid; else hi = mid;
  }
  return bestPoints;
}
