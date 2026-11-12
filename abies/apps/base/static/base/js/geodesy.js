/* WGS84 ellipsoidal geometry shared by Abies modules and Ipso classic scripts. */
(function (root) {
  'use strict';
  const library = typeof module !== 'undefined' && module.exports
    ? require('../../vendor/geographiclib/geographiclib-geodesic.min.js') : root.geodesic;
  const geod = library.Geodesic.WGS84;
  const WGS84 = Object.freeze({ a: geod.a, f: geod.f });

  function distanceMeters(lat1, lon1, lat2, lon2) {
    return geod.Inverse(lat1, lon1, lat2, lon2).s12;
  }

  function destination(lat, lng, azimuth, metres) {
    const point = geod.Direct(lat, lng, azimuth, metres);
    return { lat: point.lat2, lng: point.lon2 };
  }

  function polygonAreaMeters2(vertices) {
    if (vertices.length < 3) return 0;
    const polygon = geod.Polygon(false);
    for (const { lat, lng } of vertices) polygon.AddPoint(lat, lng);
    return Math.abs(polygon.Compute(false, true).area);
  }

  /** Ellipsoidal azimuthal-equidistant plane, in metres, around a local origin. */
  function localProjection(lat, lng) {
    return {
      project(pointLat, pointLng) {
        const result = geod.Inverse(lat, lng, pointLat, pointLng);
        const angle = result.azi1 * Math.PI / 180;
        return { x: result.s12 * Math.sin(angle), y: result.s12 * Math.cos(angle) };
      },
      unproject(x, y) {
        return destination(lat, lng, Math.atan2(x, y) * 180 / Math.PI, Math.hypot(x, y));
      },
    };
  }

  root.AbiesGeodesy = Object.freeze({
    WGS84, distanceMeters, destination, polygonAreaMeters2, localProjection,
  });
  if (typeof module !== 'undefined') module.exports = root.AbiesGeodesy;
})(globalThis);
