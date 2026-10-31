/**
 * Tabacchi volume evaluation using configured, species-linked coefficients.
 *
 * Used by Campionamenti and Piano di taglio entry forms for the
 * live V/m preview line. Forms supply the database coefficients by species ID;
 * the browser carries no independent species/default-coefficient catalogue.
 *
 *     if len(b) == 2:  V_dm3 = b[0] + b[1] * D²h
 *     if len(b) == 3:  V_dm3 = b[0] + b[1] * D²h + b[2] * D
 *     V_m3 = V_dm3 / 1000
 *
 * D in cm, h in m.
 */

/** Current settings supply coefficients by species ID; labels never select them. */
export function volumeFromCoefficientsM3(dCm, hM, b) {
  if (!Array.isArray(b) || ![2, 3].includes(b.length) || !b.every(Number.isFinite)) return null;
  const d = Number(dCm);
  const h = Number(hM);
  if (!Number.isFinite(d) || !Number.isFinite(h) || d <= 0 || h <= 0) return null;
  const d2h = d * d * h;
  const vDm3 = b.length === 2
    ? b[0] + b[1] * d2h
    : b[0] + b[1] * d2h + b[2] * d;
  // Small trees can fall below the fitted intercept: their harvestable
  // volume is zero, not negative. Do not floor small positive estimates.
  return Math.max(0, vDm3) / 1000;
}

/**
 * mass m (quintals) = volume m³ × species.density (q/m³).
 *
 * @param {number} vM3
 * @param {number} densityQPerM3
 * @returns {number|null}
 */
export function massQ(vM3, densityQPerM3) {
  if (vM3 == null || densityQPerM3 == null) return null;
  if (!Number.isFinite(vM3) || !Number.isFinite(densityQPerM3)) return null;
  return vM3 * densityQPerM3;
}
