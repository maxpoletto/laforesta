// Ipsometric regression: h = a * ln(D) + b, per (compresa, specie).
// Pure functions; the reference table is whatever shape `reference.json`
// produces, preferring region/species IDs with legacy name-map support.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof IPSO_REF_HYPSOMETRY === 'undefined') {
  Object.assign(globalThis, require('./constants.js'));
}

// Returns {a, b} or null if no entry for this (compresa, specie).
function lookup(ipsometrica, compresa, specie) {
  if (!ipsometrica) return null;
  const r = ipsometrica[compresa];
  if (!r) return null;
  const eq = r[specie];
  return eq || null;
}

function lookupReference(reference, regionId, speciesId, regionName, speciesName) {
  const byId = reference && reference[IPSO_REF_HYPSOMETRY_IDS];
  if (byId && Number.isInteger(regionId) && Number.isInteger(speciesId)) {
    return byId[regionId] && byId[regionId][speciesId] || null;
  }
  return lookup(reference && reference[IPSO_REF_HYPSOMETRY], regionName, speciesName);
}

// Returns rounded integer meters, or null if the inputs are unusable.
function computeH(eq, d_cm) {
  if (!eq) return null;
  if (d_cm == null || !(d_cm > 0)) return null;  // ln(0) and negatives
  const h = eq.a * Math.log(d_cm) + eq.b;
  if (!Number.isFinite(h) || h <= 0) return null;
  return Math.round(h);
}

const ipso = { lookup, lookupReference, computeH };
if (typeof module !== 'undefined') module.exports = ipso;
