// Session-level pure helpers.
//
// The full session lifecycle (DB writes, GPS snapshotting, auto-backup
// triggering) lives in app.js where it can await the store. This file holds
// the side-effect-free helpers so they can be unit-tested in node.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof UPLOAD_SCHEMA_VERSION === 'undefined') {
  Object.assign(globalThis, require('./constants.js'));
}

// Constraints: D in [1, 999] cm, h in [1, 99] m. The numpad enforces digit
// count; this is the final gate before write.
const D_MIN = 1, D_MAX = 999;
const H_MIN = 1, H_MAX = 99;

// Trees with D at or below this threshold (cm) aren't physically numbered
// in the field, so their stored numero is forced to blank regardless of
// what the operator typed. The counter ignores blank-number trees, so the
// next visible-numbered tree continues the sequence.
const NUMBER_BLANK_D_THRESHOLD = 17;

// Returns [] when the record is valid, otherwise an array of error keys.
// Error keys are stable strings the UI can map to user-facing messages.
function validateTree(rec, options) {
  const opts = options || {};
  const errors = [];
  if (!rec || typeof rec !== 'object') return ['empty'];
  if (!rec.specie || typeof rec.specie !== 'string') errors.push('specie');
  const dOptional = rec.d_cm == null && opts.dRequired === false;
  if (!dOptional &&
      (!Number.isInteger(rec.d_cm) || rec.d_cm < D_MIN || rec.d_cm > D_MAX)) {
    errors.push('d_cm');
  }
  const hOptional = rec.h_m == null && opts.hRequired === false;
  if (!hOptional &&
      (!Number.isInteger(rec.h_m) || rec.h_m < H_MIN || rec.h_m > H_MAX)) {
    errors.push('h_m');
  }
  // Whether a number may be blank is per-mode, but a recorded number must
  // be a positive integer in every mode — the server rejects zero/negative
  // numbers on upload, where the offending row can no longer be edited.
  const numberMissing = rec.numero == null;
  if ((opts.numberRequired && numberMissing) ||
      (!numberMissing && (!Number.isInteger(rec.numero) || rec.numero <= 0))) {
    errors.push('numero');
  }
  if (opts.parcelRequired &&
      (typeof rec.particella !== 'string' || rec.particella.trim() === '')) {
    errors.push('particella');
  }
  if (opts.sampleAreaRequired && !Number.isInteger(rec[FIELD_SAMPLE_AREA_ID])) {
    errors.push(FIELD_SAMPLE_AREA_ID);
  }
  if (opts.gpsRequired &&
      (!Number.isFinite(rec[FIELD_LAT]) || !Number.isFinite(rec[FIELD_LON]))) {
    errors.push('gps');
  }
  return errors;
}

// Returns true when seq should trigger a backup CSV download.
// Plan: every 20 trees. Defensive: only at exact multiples of 20.
const BACKUP_EVERY = 20;
function shouldBackup(seq) {
  return typeof seq === 'number' && seq > 0 && (seq % BACKUP_EVERY) === 0;
}

// Default for the next entry's number field: one above the largest non-null
// number across the supplied trees. Returns null if no tree carries a
// number (fresh session, or every recorded tree was below the size
// threshold). Monotonic and derived from the current tree list, so it
// survives resume and self-corrects after deletion.
function nextNumberDefault(trees) {
  if (!trees || !trees.length) return null;
  let max = null;
  for (const t of trees) {
    const n = t && t.numero;
    if (Number.isInteger(n) && (max === null || n > max)) max = n;
  }
  return max === null ? null : max + 1;
}

const session = {
  D_MIN, D_MAX, H_MIN, H_MAX, BACKUP_EVERY, NUMBER_BLANK_D_THRESHOLD,
  validateTree, shouldBackup, nextNumberDefault,
};
if (typeof module !== 'undefined') module.exports = session;
