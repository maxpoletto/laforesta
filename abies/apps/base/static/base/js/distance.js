/* Pure distance core shared by ES modules and Ipso's offline classic scripts. */
(function (root) {
  'use strict';
  const MEAN_EARTH_RADIUS_M = 6371009;
  const radians = degrees => degrees * Math.PI / 180;
  function distanceMeters(lat1, lon1, lat2, lon2) {
    const phi1 = radians(lat1), phi2 = radians(lat2);
    const dPhi = radians(lat2 - lat1), dLambda = radians(lon2 - lon1);
    const a = Math.min(1, Math.max(0, Math.sin(dPhi / 2) ** 2 +
      Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2));
    return MEAN_EARTH_RADIUS_M * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }
  root.AbiesDistance = Object.freeze({ MEAN_EARTH_RADIUS_M, distanceMeters });
  if (typeof module !== 'undefined') module.exports = root.AbiesDistance;
})(globalThis);
