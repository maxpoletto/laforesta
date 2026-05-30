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

/** Bounding box of a single Polygon feature, [minLng, minLat, maxLng, maxLat]. */
export function featureBbox(feature) {
  let minLng = Infinity, minLat = Infinity;
  let maxLng = -Infinity, maxLat = -Infinity;
  for (const c of feature.geometry.coordinates[0]) {
    if (c[0] < minLng) minLng = c[0];
    if (c[0] > maxLng) maxLng = c[0];
    if (c[1] < minLat) minLat = c[1];
    if (c[1] > maxLat) maxLat = c[1];
  }
  return [minLng, minLat, maxLng, maxLat];
}

/**
 * Annotate every feature in `features` with a `.bbox` (see `featureBbox`)
 * and return the same array, so callers can chain. `findContainingParcel`
 * uses this as a cheap prefilter.
 */
export function buildBboxIndex(features) {
  for (const f of features) f.bbox = featureBbox(f);
  return features;
}

/**
 * Minimum distance, in meters, from (lng, lat) to the boundary of the
 * given Polygon feature (exterior ring + any holes). Uses an
 * equirectangular projection anchored at the query point — accurate to
 * well under 1 m at the few-km scale of a parcel.
 */
export function distanceToBoundaryMeters(lng, lat, feature) {
  const mLat = 111132.92;
  const mLng = mLat * Math.cos(lat * DEG_TO_RAD);
  let best = Infinity;
  for (const ring of feature.geometry.coordinates) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const ax = (ring[j][0] - lng) * mLng;
      const ay = (ring[j][1] - lat) * mLat;
      const bx = (ring[i][0] - lng) * mLng;
      const by = (ring[i][1] - lat) * mLat;
      const dx = bx - ax;
      const dy = by - ay;
      const len2 = dx * dx + dy * dy;
      let t = len2 > 0 ? -(ax * dx + ay * dy) / len2 : 0;
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      const d = Math.hypot(ax + t * dx, ay + t * dy);
      if (d < best) best = d;
    }
  }
  return best;
}

/**
 * Split a parcel feature into its `{ compresa, particella }` names, following
 * the `bosco/data/terreni.geojson` convention: compresa lives in
 * `properties.layer`; particella is the slice after the first `-` of
 * `properties.name`.
 */
export function parcelNames(feature) {
  const p = (feature && feature.properties) || {};
  const compresa = p.layer || '';
  const fullName = p.name || '';
  const dash = fullName.indexOf('-');
  const particella = dash >= 0 ? fullName.slice(dash + 1) : fullName;
  return { compresa, particella };
}

/**
 * Format a parcel feature as "<compresa> <particella>" for tooltips.
 * Returns '' for non-parcel features.
 */
export function parcelLabel(feature) {
  const { compresa, particella } = parcelNames(feature);
  if (!compresa && !particella) return '';
  return `${compresa} ${particella}`.trim();
}

/**
 * Return the first feature whose polygon contains (lng, lat), or null.
 * If features have been annotated with `.bbox` (via `buildBboxIndex`),
 * the bbox is used as a cheap prefilter; otherwise every feature is
 * tested with `pointInPolygon`.
 */
export function findContainingParcel(lng, lat, features) {
  for (const f of features) {
    const bb = f.bbox;
    if (bb && (lng < bb[0] || lng > bb[2] || lat < bb[1] || lat > bb[3])) continue;
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
        const { compresa, particella } = parcelNames(f);
        points.push({ lat, lng, compresa, particella });
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
