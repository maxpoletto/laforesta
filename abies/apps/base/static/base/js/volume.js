/**
 * Tabacchi volume equations (JS mirror of apps/base/tabacchi.py).
 *
 * Used by Campionamenti and Piano di taglio entry forms for the
 * live V/m preview line.  Parity with the Python copy is enforced
 * by test/test_tabacchi.py.
 *
 *     if len(b) == 2:  V_dm3 = b[0] + b[1] * D²h
 *     if len(b) == 3:  V_dm3 = b[0] + b[1] * D²h + b[2] * D
 *     V_m3 = V_dm3 / 1000
 *
 * D in cm, h in m.
 */

export const SP_ABETE = 'Abete';
export const SP_ACERO = 'Acero';
export const SP_CASTAGNO = 'Castagno';
export const SP_CERRO = 'Cerro';
export const SP_CILIEGIO = 'Ciliegio';
export const SP_DOUGLAS = 'Douglas';
export const SP_FAGGIO = 'Faggio';
export const SP_LECCIO = 'Leccio';
export const SP_ONTANO = 'Ontano';
export const SP_PINO = 'Pino';
export const SP_PINO_LARICIO = 'Pino Laricio';
export const SP_PINO_MARITTIMO = 'Pino Marittimo';
export const SP_PINO_NERO = 'Pino Nero';
export const SP_SORBO = 'Sorbo';

/** Per-species `b` coefficient vectors.  Length 2 or 3. */
export const TABACCHI_B = {
  [SP_ABETE]:          [-1.8381,     3.7836e-2, 3.9934e-1],
  [SP_ACERO]:          [ 1.6905,     3.7082e-2],
  [SP_CASTAGNO]:       [-2.0,        3.6524e-2, 7.4466e-1],
  [SP_CERRO]:          [-4.3221e-2,  3.8079e-2],
  [SP_CILIEGIO]:       [ 2.3118,     3.1278e-2, 3.7159e-1],
  [SP_DOUGLAS]:        [-7.9946,     3.3343e-2, 1.2186],
  [SP_FAGGIO]:         [ 8.1151e-1,  3.8965e-2],
  [SP_LECCIO]:         [-2.2219,     3.9685e-2, 6.2762e-1],
  [SP_ONTANO]:         [-2.2932e1,   3.2641e-2, 2.991],
  [SP_PINO]:           [ 6.4383,     3.8594e-2],
  [SP_PINO_LARICIO]:   [ 6.4383,     3.8594e-2],
  [SP_PINO_MARITTIMO]: [ 2.9963,     3.8302e-2],
  [SP_PINO_NERO]:      [-2.1480e1,   3.3448e-2, 2.9088],
  [SP_SORBO]:          [ 2.3118,     3.1278e-2, 3.7159e-1],
};

/**
 * Compute tree volume in m³ via the species-specific Tabacchi equation.
 * Returns null if the species isn't in the Tabacchi table (e.g.,
 * 'Altro') so callers can hide the preview gracefully.
 *
 * @param {number} dCm
 * @param {number} hM
 * @param {string} speciesName
 * @returns {number|null}
 */
export function tabacchiVolumeM3(dCm, hM, speciesName) {
  const b = TABACCHI_B[speciesName];
  if (!b) return null;
  const d = Number(dCm);
  const h = Number(hM);
  if (!Number.isFinite(d) || !Number.isFinite(h) || d <= 0 || h <= 0) return null;
  const d2h = d * d * h;
  const vDm3 = b.length === 2
    ? b[0] + b[1] * d2h
    : b[0] + b[1] * d2h + b[2] * d;
  return vDm3 / 1000;
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

export function hasSpecies(name) {
  return name in TABACCHI_B;
}
