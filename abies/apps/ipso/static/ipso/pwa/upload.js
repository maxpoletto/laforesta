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
const DEFAULT_SAMPLE_RADIUS_M = 12;
const DEFAULT_PRESSLER_COEFF = '2.00';
const UPLOAD_FIELD_SAMPLE_AREA_ID = 'sample_area_id';
const UPLOAD_FIELD_COPPICE = 'coppice';
const UPLOAD_FIELD_SHOOT = 'shoot';
const UPLOAD_FIELD_STANDARD = 'standard';
const UPLOAD_FIELD_L10_MM = 'l10_mm';
const UPLOAD_FIELD_PRESSLER_COEFF = 'pressler_coeff';
const UPLOAD_FIELD_PRESERVED = 'preserved';
const UPLOAD_FIELD_ESTIMATED_BIRTH_YEAR = 'estimated_birth_year';
const UPLOAD_FIELD_OPERATOR = 'operator';
const UPLOAD_FIELD_NOTE = 'note';

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
      damaged: !!sess.catastrofata,
      region_id: regionId,
    },
    records,
    csv_text: csvText || '',
  };
}

function canonicalRecord(sess, t, reference) {
  const speciesId = speciesIdForName(reference, t.specie);
  const parcel = parcelForName(reference, sess.compresa, t.particella);
  const record = {
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
  if ((sess.mode || UPLOAD_MODE_MARTELLATE) === UPLOAD_MODE_SAMPLES) {
    Object.assign(record, sampleRecordContext(reference, sess, t, parcel));
  } else if ((sess.mode || UPLOAD_MODE_MARTELLATE) === UPLOAD_MODE_PAI) {
    Object.assign(record, paiRecordContext(sess, t));
  }
  return record;
}


function sampleRecordContext(reference, sess, tree, parcel) {
  const area = sampleAreaForTree(reference, sess, tree, parcel);
  return {
    [UPLOAD_FIELD_SAMPLE_AREA_ID]: area ? area.sample_area_id : null,
    [UPLOAD_FIELD_COPPICE]: area && typeof area.coppice === 'boolean'
      ? area.coppice
      : null,
    [UPLOAD_FIELD_SHOOT]: Number.isInteger(tree.shoot) ? tree.shoot : 0,
    [UPLOAD_FIELD_STANDARD]: !!tree.standard,
    [UPLOAD_FIELD_L10_MM]: Number.isInteger(tree.l10_mm) ? tree.l10_mm : 0,
    [UPLOAD_FIELD_PRESSLER_COEFF]: tree.pressler_coeff || DEFAULT_PRESSLER_COEFF,
    [UPLOAD_FIELD_PRESERVED]: !!tree.preserved,
  };
}

function paiRecordContext(sess, tree) {
  return {
    [UPLOAD_FIELD_ESTIMATED_BIRTH_YEAR]: Number.isInteger(tree.estimated_birth_year)
      ? tree.estimated_birth_year
      : null,
    [UPLOAD_FIELD_OPERATOR]: tree.operator || sess.operatore || '',
    [UPLOAD_FIELD_NOTE]: tree.note || '',
  };
}

function sampleAreaForTree(reference, sess, tree, parcel) {
  if (!reference || !reference.sampling) return null;
  const areas = reference.sampling.sample_areas || [];
  if (Number.isInteger(tree.sample_area_id)) {
    const stored = areas.find((area) =>
      area && area.sample_area_id === tree.sample_area_id
    );
    if (stored) return stored;
  }
  if (tree.lat == null || tree.lon == null) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const area of areas) {
    if (!area || area.compresa !== sess.compresa || area.parcel_id !== parcel.parcel_id) {
      continue;
    }
    if (area.lat == null || area.lon == null) continue;
    const distance = distanceMeters(tree.lat, tree.lon, area.lat, area.lon);
    const radius = Number.isFinite(area.r_m) ? area.r_m : DEFAULT_SAMPLE_RADIUS_M;
    if (distance <= radius && distance < bestDistance) {
      best = area;
      bestDistance = distance;
    }
  }
  return best;
}

function distanceMeters(lat1, lon1, lat2, lon2) {
  const toRad = (deg) => deg * Math.PI / 180;
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const dPhi = toRad(lat2 - lat1);
  const dLambda = toRad(lon2 - lon1);
  const a = Math.sin(dPhi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2;
  return 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
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
  UPLOAD_MODE_PAI, DEFAULT_SAMPLE_RADIUS_M,
  BACKOFF_SCHEDULE_MS, BACKOFF_CAP_MS,
  backoffMs, classifyHttp, classifyNetwork, distanceMeters,
  UploadError, buildUploadPayload, uploadSession,
};

if (typeof module !== 'undefined') module.exports = upload;
