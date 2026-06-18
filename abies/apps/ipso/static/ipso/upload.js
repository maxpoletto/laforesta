// Upload an ipso session to the Abies staged-upload endpoint.
//
// Pure-logic helpers (backoff, classify, payload construction) are testable in
// node; the network-touching uploadSession() is exercised in browser via the
// screen-upload state machine.
'use strict';

const UPLOAD_SCHEMA_VERSION = 1;
const UPLOAD_MODE_MARTELLATE = typeof IPSO_MODE_MARTELLATE !== 'undefined'
  ? IPSO_MODE_MARTELLATE
  : 'martellate';
const UPLOAD_MODE_SAMPLES = typeof IPSO_MODE_SAMPLES !== 'undefined'
  ? IPSO_MODE_SAMPLES
  : 'samples';
const UPLOAD_MODE_PAI = typeof IPSO_MODE_PAI !== 'undefined'
  ? IPSO_MODE_PAI
  : 'pai';
const BACKOFF_SCHEDULE_MS = [2000, 4000, 8000, 16000];
const BACKOFF_CAP_MS = 30000;

// 1-based attempt number -> wait BEFORE that attempt. Attempt 0 is "no
// wait yet", attempt N>0 picks index N-1 from the schedule (or the cap).
function backoffMs(attempt) {
  if (!Number.isFinite(attempt) || attempt <= 0) return 0;
  const i = Math.floor(attempt) - 1;
  if (i < BACKOFF_SCHEDULE_MS.length) return BACKOFF_SCHEDULE_MS[i];
  return BACKOFF_CAP_MS;
}

// HTTP status -> outcome class. The state machine drives bail/retry off the
// "hard:" / "soft:" prefix; the suffix lets the UI pick an error string.
function classifyHttp(status) {
  if (status === 200) return 'ok';
  if (status === 401) return 'hard:auth';
  if (status === 409) return 'hard:conflict';
  if (status === 413) return 'hard:too_large';
  if (status === 422) return 'hard:invalid_csv';
  if (status === 429) return 'soft:rate_limited';
  if (status >= 500 && status < 600) return 'soft:server';
  // Anything else 4xx-shaped: treat as a bug. Hard error stops retries.
  return 'hard:invalid_csv';
}

// Network / aborted-fetch failures classify the same way as 5xx (soft).
function classifyNetwork() { return 'soft:network'; }

class UploadError extends Error {
  constructor(klass, detail) {
    super(klass + (detail ? ': ' + detail : ''));
    this.klass = klass;          // 'hard:auth' | 'soft:server' | ...
    this.detail = detail || '';
  }
}

function buildUploadPayload(sess, trees, reference, csvText) {
  if (!sess || !reference) throw new Error('missing upload context');
  const regionId = regionIdForCompresa(reference, sess.compresa);
  const records = trees.map((t) => canonicalRecord(sess, t, reference));
  return {
    session: {
      session_id: sess.id,
      mode: sess.mode || UPLOAD_MODE_MARTELLATE,
      schema_version: UPLOAD_SCHEMA_VERSION,
      reference_version:
        sess.reference_version || reference.reference_version || '',
      work_package_id: sess.work_package_id || '',
      operator: sess.operatore || '',
      created_at: sess.started_at || '',
      completed_at: sess.completed_at || sess.exported_at || '',
      catastrofata: !!sess.catastrofata,
      region_id: regionId,
    },
    records,
    csv_text: csvText || '',
  };
}

function canonicalRecord(sess, t, reference) {
  const speciesId = speciesIdForName(reference, t.specie);
  const parcel = parcelForName(reference, sess.compresa, t.particella);
  return {
    client_record_id: String(t.seq || t.id),
    date: sess.data,
    region_id: parcel.region_id,
    parcel_id: parcel.parcel_id,
    species_id: speciesId,
    number: Number.isInteger(t.numero) ? t.numero : null,
    d_cm: t.d_cm == null ? null : t.d_cm,
    h_m: t.h_m == null ? null : String(t.h_m),
    h_measured: !!t.h_measured,
    hypso_param_set_id: Number.isInteger(t.hypso_param_set_id)
      ? t.hypso_param_set_id
      : null,
    lat: t.lat == null ? null : t.lat,
    lon: t.lon == null ? null : t.lon,
    acc_m: t.acc_m == null ? null : t.acc_m,
  };
}

function speciesIdForName(reference, name) {
  const row = (reference.species || []).find((s) => s.common === name);
  if (!row || !Number.isInteger(row.id)) {
    throw new Error('specie senza ID Abies: ' + (name || ''));
  }
  return row.id;
}

function regionIdForCompresa(reference, compresa) {
  const row = (reference.parcels || []).find((p) => p.compresa === compresa);
  if (!row || !Number.isInteger(row.region_id)) {
    throw new Error('compresa senza ID Abies: ' + (compresa || ''));
  }
  return row.region_id;
}

function parcelForName(reference, compresa, particella) {
  const row = (reference.parcels || []).find(
    (p) => p.compresa === compresa && p.particella === particella
  );
  if (!row || !Number.isInteger(row.parcel_id) || !Number.isInteger(row.region_id)) {
    throw new Error('particella senza ID Abies: ' + (compresa || '') + '/' + (particella || ''));
  }
  return row;
}

// Posts the canonical staged JSON payload. Resolves with { duplicate: bool } on
// 200, throws UploadError otherwise. Caller passes signal for cancellation.
async function uploadSession(args) {
  const { token, sessionId, payload, signal } = args;
  const headers = {
    'Content-Type': 'application/json',
    'X-Ipso-Session-Id': sessionId,
  };
  if (token) headers.Authorization = 'Bearer ' + token;

  let resp;
  try {
    resp = await fetch('/api/ipso/uploads/', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal,
    });
  } catch (e) {
    if (e && e.name === 'AbortError') throw new UploadError('aborted');
    throw new UploadError(classifyNetwork(), e && e.message);
  }
  const klass = classifyHttp(resp.status);
  if (klass === 'ok') {
    let responsePayload = {};
    try { responsePayload = await resp.json(); } catch (_) {}
    return {
      duplicate: !!responsePayload.duplicate,
      storedAs: responsePayload.stored_as,
    };
  }
  let detail = '';
  try {
    const responsePayload = await resp.json();
    detail = responsePayload && responsePayload.error ? responsePayload.error : '';
  } catch (_) {}
  throw new UploadError(klass, detail);
}

const upload = {
  UPLOAD_SCHEMA_VERSION, UPLOAD_MODE_MARTELLATE, UPLOAD_MODE_SAMPLES,
  UPLOAD_MODE_PAI,
  BACKOFF_SCHEDULE_MS, BACKOFF_CAP_MS,
  backoffMs, classifyHttp, classifyNetwork,
  UploadError, buildUploadPayload, uploadSession,
};

if (typeof module !== 'undefined') module.exports = upload;
