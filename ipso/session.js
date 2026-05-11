// Session-level pure helpers.
//
// The full session lifecycle (DB writes, GPS snapshotting, auto-backup
// triggering) lives in app.js where it can await the store. This file holds
// the side-effect-free helpers so they can be unit-tested in node.
'use strict';

// Constraints: D in [1, 999] cm, h in [1, 99] m. The numpad enforces digit
// count; this is the final gate before write.
const D_MIN = 1, D_MAX = 999;
const H_MIN = 1, H_MAX = 99;

// Returns next per-session sequence number given an iterable of existing
// seq values. seq starts at 1.
function nextSeq(existingSeqs) {
  let max = 0;
  for (const s of existingSeqs) {
    if (typeof s === 'number' && s > max) max = s;
  }
  return max + 1;
}

// Returns [] when the record is valid, otherwise an array of error keys.
// Error keys are stable strings the UI can map to user-facing messages.
function validateTree(rec) {
  const errors = [];
  if (!rec || typeof rec !== 'object') return ['empty'];
  if (!rec.specie || typeof rec.specie !== 'string') errors.push('specie');
  if (!Number.isInteger(rec.d_cm) || rec.d_cm < D_MIN || rec.d_cm > D_MAX) {
    errors.push('d_cm');
  }
  if (!Number.isInteger(rec.h_m) || rec.h_m < H_MIN || rec.h_m > H_MAX) {
    errors.push('h_m');
  }
  return errors;
}

// Italian summary line for the "ultimo" pill.
function summarizePill(rec) {
  if (!rec) return 'nessun albero';
  const d = rec.d_cm != null ? `D=${rec.d_cm}` : 'D=—';
  const h = rec.h_m != null ? `h=${rec.h_m}` : 'h=—';
  return `${rec.specie}, ${d}, ${h}`;
}

// Returns true when seq should trigger a backup CSV download.
// Plan: every 20 trees. Defensive: only at exact multiples of 20.
const BACKUP_EVERY = 20;
function shouldBackup(seq) {
  return typeof seq === 'number' && seq > 0 && (seq % BACKUP_EVERY) === 0;
}

const session = {
  D_MIN, D_MAX, H_MIN, H_MAX, BACKUP_EVERY,
  nextSeq, validateTree, summarizePill, shouldBackup,
};
if (typeof module !== 'undefined') module.exports = session;
