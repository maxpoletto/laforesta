// Ipsometric regression: h = a * ln(D) + b, per (compresa, specie).
// Pure functions; the reference table is whatever shape `reference.json`
// produces — {compresa: {specie: {a, b}}}.
'use strict';

// Returns {a, b} or null if no entry for this (compresa, specie).
function lookup(ipsometrica, compresa, specie) {
  if (!ipsometrica) return null;
  const r = ipsometrica[compresa];
  if (!r) return null;
  const eq = r[specie];
  return eq || null;
}

// Returns rounded integer meters, or null if the inputs are unusable.
function computeH(eq, d_cm) {
  if (!eq) return null;
  if (d_cm == null || !(d_cm > 0)) return null;  // ln(0) and negatives
  const h = eq.a * Math.log(d_cm) + eq.b;
  if (!Number.isFinite(h) || h <= 0) return null;
  return Math.round(h);
}

const ipso = { lookup, computeH };
if (typeof module !== 'undefined') module.exports = ipso;
