/** Shared parcel geometry and WGS84 measurements for Abies and offline Ipso. */

import * as S from './strings.js';
import '../../vendor/geographiclib/geographiclib-geodesic.min.js';
import './geodesy.js';

export const { distanceMeters, destination, localProjection } = globalThis.AbiesGeodesy;

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

function geometryPolygons(geometry) {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates;
  return [];
}

function pointInPolygonRings(lng, lat, rings) {
  if (!pointInRing(lng, lat, rings[0])) return false;
  for (let i = 1; i < rings.length; i++) {
    if (pointInRing(lng, lat, rings[i])) return false;
  }
  return true;
}

function forEachGeometryCoordinate(geometry, fn) {
  for (const rings of geometryPolygons(geometry)) {
    for (const ring of rings) {
      for (const coord of ring) fn(coord);
    }
  }
}

/**
 * Point-in-polygon for a GeoJSON Polygon or MultiPolygon geometry
 * (exterior + holes).
 */
export function pointInPolygon(lng, lat, geometry) {
  return geometryPolygons(geometry).some(rings => pointInPolygonRings(lng, lat, rings));
}

/**
 * WGS84 ellipsoidal area of a polygon ring expressed as [{lat, lng}, …], in m².
 * Edges follow shortest ellipsoidal geodesics; winding does not change area.
 */
export function geodesicArea(latlngs) {
  return globalThis.AbiesGeodesy.polygonAreaMeters2(latlngs);
}

/** Convert a GeoJSON coordinate ring [[lng, lat], …] to [{lat, lng}, …]. */
export function ringToLatLngs(ring) {
  return ring.map(c => ({ lat: c[1], lng: c[0] }));
}

/**
 * Geodesic area of a GeoJSON Polygon/MultiPolygon feature, in m².
 */
export function featureArea(feature) {
  return geoJSONFeatureArea(feature);
}

/** Geodesic area of a GeoJSON feature in m² (exterior minus holes). */
export function geoJSONFeatureArea(feature) {
  const geom = feature.geometry;
  if (!geom) return 0;
  const polygons = geometryPolygons(geom);
  let total = 0;
  for (const rings of polygons) {
    total += geodesicArea(ringToLatLngs(rings[0]));
    for (let i = 1; i < rings.length; i++) {
      total -= geodesicArea(ringToLatLngs(rings[i]));
    }
  }
  return total;
}

/**
 * Precompute geodesic area (m²) on each feature and sort largest-first, so
 * smaller polygons render — and tooltip — on top of the larger ones that
 * contain them.  Mutates and returns `geojson`.
 */
export function sortFeaturesByArea(geojson) {
  geojson.features.forEach(f => {
    f.properties._areaM2 = geoJSONFeatureArea(f);
  });
  geojson.features.sort((a, b) => b.properties._areaM2 - a.properties._areaM2);
  return geojson;
}

/** Bounding box of an array of GeoJSON Polygon/MultiPolygon features. */
export function boundingBox(features) {
  let minLng = Infinity, minLat = Infinity;
  let maxLng = -Infinity, maxLat = -Infinity;
  for (const f of features) {
    forEachGeometryCoordinate(f.geometry, (c) => {
      if (c[0] < minLng) minLng = c[0];
      if (c[0] > maxLng) maxLng = c[0];
      if (c[1] < minLat) minLat = c[1];
      if (c[1] > maxLat) maxLat = c[1];
    });
  }
  return { minLng, minLat, maxLng, maxLat };
}

/** Bounding box of a single Polygon/MultiPolygon feature. */
export function featureBbox(feature) {
  let minLng = Infinity, minLat = Infinity;
  let maxLng = -Infinity, maxLat = -Infinity;
  forEachGeometryCoordinate(feature.geometry, (c) => {
    if (c[0] < minLng) minLng = c[0];
    if (c[0] > maxLng) maxLng = c[0];
    if (c[1] < minLat) minLat = c[1];
    if (c[1] > maxLat) maxLat = c[1];
  });
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
 * given Polygon/MultiPolygon feature, including holes. Uses an ellipsoidal
 * azimuthal-equidistant plane centred at the query point. Segment projection
 * is a local approximation, tested within 1 cm of exact geodesic-segment
 * minima over 5 km forestry extents; this is not a global segment solver.
 */
export function distanceToBoundaryMeters(lng, lat, feature) {
  const plane = localProjection(lat, lng);
  let best = Infinity;
  for (const rings of geometryPolygons(feature.geometry)) {
    for (const ring of rings) {
      const projected = ring.map(([lon, latitude]) => plane.project(latitude, lon));
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const { x: ax, y: ay } = projected[j];
        const { x: bx, y: by } = projected[i];
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
  }
  return best;
}

/** Normalize supported source aliases without changing their string labels. */
export function parcelIdentity(feature) {
  const p = (feature && feature.properties) || {};
  const regionName = p.layer || p.compresa || p.Compresa || '';
  const fullName = p.name || p.particella || p.Particella || '';
  const dash = fullName.indexOf('-');
  const parcelName = dash >= 0 ? fullName.slice(dash + 1) : fullName;
  return { regionName, parcelName };
}

/** Legacy result shape retained for existing desktop/offline consumers. */
export function parcelNames(feature) {
  const { regionName, parcelName } = parcelIdentity(feature);
  return { compresa: regionName, particella: parcelName };
}

export function parcelTypeLabel(feature) {
  const p = (feature && feature.properties) || {};
  if (p.coppice === true) return S.TYPE_COPPICE;
  if (p.coppice === false) return S.TYPE_HIGHFOREST;
  return '';
}

/**
 * Return display data for a parcel feature. Enriched terreni.geojson features
 * include forest type; raw geometries only carry the parcel title. Returns
 * null for non-parcel features.
 */
export function parcelLabel(feature) {
  const { regionName, parcelName } = parcelIdentity(feature);
  if (!regionName && !parcelName) return null;
  return {
    title: `${regionName} ${parcelName}`.trim(),
    type: parcelTypeLabel(feature),
  };
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

/** Project parcel extents once for all target-spacing trials. */
function gridContext(features) {
  const bb = boundingBox(features);
  const plane = localProjection((bb.minLat + bb.maxLat) / 2, (bb.minLng + bb.maxLng) / 2);
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (const feature of features) {
    forEachGeometryCoordinate(feature.geometry, ([lng, lat]) => {
      const { x, y } = plane.project(lat, lng);
      bounds.minX = Math.min(bounds.minX, x); bounds.maxX = Math.max(bounds.maxX, x);
      bounds.minY = Math.min(bounds.minY, y); bounds.maxY = Math.max(bounds.maxY, y);
    });
  }
  return { plane, bounds,
    features: features.map(feature => ({ ...feature, bbox: featureBbox(feature) })) };
}

function generateGridInPlane(context, spacingMeters) {
  if (!Number.isFinite(spacingMeters) || spacingMeters <= 0) {
    throw new RangeError('Grid spacing must be a positive finite number');
  }
  const { plane, bounds, features } = context;
  const points = [];
  for (let y = bounds.minY + spacingMeters / 2; y < bounds.maxY; y += spacingMeters) {
    for (let x = bounds.minX + spacingMeters / 2; x < bounds.maxX; x += spacingMeters) {
      const { lat, lng } = plane.unproject(x, y);
      const feature = findContainingParcel(lng, lat, features);
      if (feature) {
        const { regionName, parcelName } = parcelIdentity(feature);
        // Preserve the existing grid endpoint shape at this boundary.
        points.push({ lat, lng, compresa: regionName, particella: parcelName });
      }
    }
  }
  return points;
}

/**
 * Generate a cell-centred lattice in a local WGS84 metre plane, then clip
 * point centres against the supplied parcels. The plane's axes are true
 * north/east at the region centre; stored grids are never regenerated here.
 * @returns {Array<{lat, lng, compresa, particella}>} where
 *   `compresa` = feature.properties.layer, `particella` =
 *   short name after the first `-` in feature.properties.name.
 */
export function generateGrid(features, spacingMeters) {
  return features.length ? generateGridInPlane(gridContext(features), spacingMeters) : [];
}

/**
 * Binary-search the lattice spacing (in meters) until the resulting
 * point count is within `tolerance` of `targetN`, returning the
 * best-fit point set. All trials reuse one ellipsoidal local plane.
 *
 * @param {Array} features — array of GeoJSON Polygon/MultiPolygon features.
 * @param {number} targetN — desired point count.
 * @param {object} opts — { maxIter, tolerance, hiMeters }.
 *   hiMeters defaults to sqrt(totalAreaM2) — a coarse upper bound.
 */
export function planGridForTarget(features, targetN, opts = {}) {
  if (!features.length || !Number.isFinite(targetN) || targetN <= 0) return [];
  const { maxIter = 40, tolerance = 0.05 } = opts;
  const totalAreaM2 = features.reduce((s, f) => s + featureArea(f), 0);
  const context = gridContext(features);
  let lo = 1;
  let hi = opts.hiMeters || Math.sqrt(totalAreaM2);
  let bestPoints = [];

  for (let i = 0; i < maxIter; i++) {
    const mid = (lo + hi) / 2;
    const pts = generateGridInPlane(context, mid);
    if (Math.abs(pts.length - targetN) < Math.abs(bestPoints.length - targetN)) {
      bestPoints = pts;
    }
    if (pts.length === targetN) break;
    if (Math.abs(pts.length - targetN) / targetN < tolerance) break;
    if (pts.length > targetN) lo = mid; else hi = mid;
  }
  return bestPoints;
}
