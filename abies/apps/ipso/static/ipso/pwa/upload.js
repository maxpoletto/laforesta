// Upload an ipso session to the Abies staged-upload endpoint.
//
// Pure-logic helpers (backoff, classify, payload construction) are testable in
// node; the network-touching uploadSession() is exercised in browser via the
// screen-upload state machine.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof UPLOAD_SCHEMA_VERSION === 'undefined') {
  Object.assign(globalThis, require('./constants.js'));
}

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof S === 'undefined') {
  Object.assign(globalThis, require('./strings.js'));
}

const BACKOFF_SCHEDULE_MS = [2000, 4000, 8000, 16000];
const BACKOFF_CAP_MS = 30000;
const DEFAULT_UPLOAD_MAX_BYTES = 30 * 1024 * 1024;
const MULTIPART_ESTIMATE_OVERHEAD_BYTES = 4096;
const MULTIPART_ESTIMATE_PART_OVERHEAD_BYTES = 1024;

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
  if (!sess || !reference) throw new Error(S.UPLOAD_ERROR_CONTEXT_MISSING);
  const mode = sess.mode || IPSO_MODE_MARTELLATE;
  if (!isSupportedUploadMode(mode)) throw new Error(S.UPLOAD_ERROR_MODE_UNSUPPORTED);
  const regionId = Number.isInteger(sess[FIELD_REGION_ID])
    ? sess[FIELD_REGION_ID]
    : regionIdForCompresa(reference, sess.compresa);
  const records = mode === IPSO_MODE_OBSERVATIONS
    ? trees.map((t) => canonicalObservationRecord(sess, t))
    : trees.map((t) => canonicalRecord(sess, t, reference));
  validateRecordNumbers(sess, records, reference);
  return {
    [SESSION]: {
      [FIELD_SESSION_ID]: sess.id,
      [FIELD_MODE]: mode,
      [FIELD_SCHEMA_VERSION]: UPLOAD_SCHEMA_VERSION,
      [FIELD_REFERENCE_VERSION]:
        sess.reference_version || reference.reference_version || '',
      [FIELD_WORK_PACKAGE_ID]: sess.work_package_id || '',
      [FIELD_OPERATOR]: sess.operatore || '',
      [FIELD_CREATED_AT]: sess.started_at || '',
      [FIELD_COMPLETED_AT]: completedAt(sess),
      [FIELD_DAMAGED]: !!sess.catastrofata,
      [FIELD_REGION_ID]: regionId,
    },
    [RECORDS]: records,
    [FIELD_CSV_TEXT]: csvText || '',
  };
}

function isSupportedUploadMode(mode) {
  return mode === IPSO_MODE_MARTELLATE ||
    mode === IPSO_MODE_SAMPLES ||
    mode === IPSO_MODE_FREE_SURVEY ||
    mode === IPSO_MODE_OBSERVATIONS;
}

function completedAt(sess) {
  return sess.completed_at || sess.exported_at || new Date().toISOString();
}

function canonicalObservationRecord(sess, rec) {
  return {
    [FIELD_CLIENT_RECORD_ID]: String(rec.seq || rec.id),
    [FIELD_DATE]: sess.data,
    [FIELD_REGION_ID]: Number.isInteger(rec[FIELD_REGION_ID])
      ? rec[FIELD_REGION_ID]
      : Number.isInteger(sess[FIELD_REGION_ID]) ? sess[FIELD_REGION_ID] : null,
    [FIELD_TEXT]: rec[FIELD_TEXT] || '',
    [FIELD_CATEGORY_IDS]: Array.isArray(rec[FIELD_CATEGORY_IDS])
      ? rec[FIELD_CATEGORY_IDS].slice() : [],
    [FIELD_LAT]: rec.lat == null ? null : rec.lat,
    [FIELD_LON]: rec.lon == null ? null : rec.lon,
    [FIELD_ACC_M]: rec.acc_m == null ? null : rec.acc_m,
    [FIELD_PHOTOS]: observationPhotoMetadata(rec[FIELD_PHOTOS]),
  };
}

function observationPhotoMetadata(photos) {
  if (!Array.isArray(photos)) return [];
  return photos.map((photo, index) => {
    const blob = photo && photo.blob;
    const clientId = photo && photo[FIELD_CLIENT_PHOTO_ID]
      ? String(photo[FIELD_CLIENT_PHOTO_ID])
      : String(index + 1);
    return {
      [FIELD_CLIENT_PHOTO_ID]: clientId,
      [FIELD_CONTENT_TYPE]: photo && photo[FIELD_CONTENT_TYPE] || blob && blob.type || '',
      [FIELD_SIZE_BYTES]: Number.isInteger(photo && photo[FIELD_SIZE_BYTES])
        ? photo[FIELD_SIZE_BYTES]
        : blob && Number.isInteger(blob.size) ? blob.size : 0,
      [FIELD_WIDTH_PX]: ipsoPositiveInt(photo && photo[FIELD_WIDTH_PX]),
      [FIELD_HEIGHT_PX]: ipsoPositiveInt(photo && photo[FIELD_HEIGHT_PX]),
      [FIELD_ORIGINAL_FILENAME]: photo && photo[FIELD_ORIGINAL_FILENAME] || '',
      blob: blob || null,
    };
  });
}

function canonicalRecord(sess, t, reference) {
  const speciesId = Number.isInteger(t[FIELD_SPECIES_ID])
    ? t[FIELD_SPECIES_ID]
    : speciesIdForName(reference, t.specie);
  let parcel = null;
  if (!Number.isInteger(t[FIELD_PARCEL_ID]) ||
      (!Number.isInteger(t[FIELD_REGION_ID]) && !Number.isInteger(sess[FIELD_REGION_ID]))) {
    parcel = parcelForName(reference, sess.compresa, t.particella);
  }
  const parcelId = Number.isInteger(t[FIELD_PARCEL_ID])
    ? t[FIELD_PARCEL_ID] : parcel[FIELD_PARCEL_ID];
  const regionId = Number.isInteger(t[FIELD_REGION_ID])
    ? t[FIELD_REGION_ID]
    : Number.isInteger(sess[FIELD_REGION_ID])
      ? sess[FIELD_REGION_ID] : parcel[FIELD_REGION_ID];
  const record = {
    [FIELD_CLIENT_RECORD_ID]: String(t.seq || t.id),
    [FIELD_DATE]: sess.data,
    [FIELD_REGION_ID]: regionId,
    [FIELD_PARCEL_ID]: parcelId,
    [FIELD_SPECIES_ID]: speciesId,
    [FIELD_NUMBER]: Number.isInteger(t.numero) ? t.numero : null,
    [FIELD_D_CM]: t.d_cm == null ? null : t.d_cm,
    [FIELD_H_M]: t.h_m == null ? null : String(t.h_m),
    [FIELD_H_MEASURED]: !!t.h_measured,
    [FIELD_HYPSO_PARAM_SET_ID]: Number.isInteger(t.hypso_param_set_id)
      ? t.hypso_param_set_id
      : null,
    [FIELD_LAT]: t.lat == null ? null : t.lat,
    [FIELD_LON]: t.lon == null ? null : t.lon,
    [FIELD_ACC_M]: t.acc_m == null ? null : t.acc_m,
  };
  if ((sess.mode || IPSO_MODE_MARTELLATE) === IPSO_MODE_SAMPLES) {
    Object.assign(record, sampleRecordContext(reference, sess, t, parcelId));
  } else if ((sess.mode || IPSO_MODE_MARTELLATE) === IPSO_MODE_FREE_SURVEY) {
    Object.assign(record, freeSurveyRecordContext(sess, t));
  }
  return record;
}

function validateRecordNumbers(sess, records, reference) {
  const mode = sess.mode || IPSO_MODE_MARTELLATE;
  if (mode === IPSO_MODE_MARTELLATE || mode === IPSO_MODE_OBSERVATIONS) return;
  if (mode === IPSO_MODE_FREE_SURVEY) {
    validateFreeSurveyNumbers(records, reference);
    return;
  }

  const seen = new Set();
  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const number = record[FIELD_NUMBER];
    if (!Number.isInteger(number) || number <= 0) {
      throw new Error(S.UPLOAD_ERROR_NUMBER_REQUIRED(i + 1));
    }
    const scope = mode === IPSO_MODE_SAMPLES
      ? record[FIELD_SAMPLE_AREA_ID]
      : record[FIELD_PARCEL_ID];
    if (!Number.isInteger(scope)) continue;
    const key = scope + ':' + number;
    if (seen.has(key)) {
      throw new Error(S.UPLOAD_ERROR_NUMBER_DUPLICATE(i + 1));
    }
    seen.add(key);
    if (mode === IPSO_MODE_SAMPLES) {
      const maxNumber = sampleMaxNumberForArea(reference, sess, scope);
      if (Number.isInteger(maxNumber) && number <= maxNumber) {
        throw new Error(S.UPLOAD_ERROR_NUMBER_ALREADY_USED(i + 1));
      }
    }
  }
}

function validateFreeSurveyNumbers(records, reference) {
  const sampleNumbers = new Set();
  const preservedNumbers = new Set();
  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const number = record[FIELD_NUMBER];
    const preserved = !!record[FIELD_PRESERVED];
    if (number == null) {
      if (preserved) throw new Error(S.UPLOAD_ERROR_NUMBER_REQUIRED(i + 1));
      continue;
    }
    if (!Number.isInteger(number) || number <= 0) {
      throw new Error(S.UPLOAD_ERROR_NUMBER_REQUIRED(i + 1));
    }
    if (!preserved) {
      const sampleKey = String(number);
      if (sampleNumbers.has(sampleKey)) {
        throw new Error(S.UPLOAD_ERROR_NUMBER_DUPLICATE(i + 1));
      }
      sampleNumbers.add(sampleKey);
      continue;
    }
    const parcelId = record[FIELD_PARCEL_ID];
    const preservedKey = parcelId + ':' + number;
    if (preservedNumbers.has(preservedKey)) {
      throw new Error(S.UPLOAD_ERROR_NUMBER_DUPLICATE(i + 1));
    }
    preservedNumbers.add(preservedKey);
    if (paiNumberExists(reference, parcelId, number)) {
      throw new Error(S.UPLOAD_ERROR_NUMBER_ALREADY_USED(i + 1));
    }
  }
}

function sampleSurveyIdFromWorkPackage(workPackageId) {
  const raw = String(workPackageId || '');
  if (!raw.startsWith(IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX)) return null;
  const id = parseInt(raw.slice(IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX.length), 10);
  return Number.isInteger(id) ? id : null;
}

function sampleMaxNumberForArea(reference, sess, sampleAreaId) {
  const sampling = reference && reference[IPSO_REF_SAMPLING];
  const surveyId = sess && Number.isInteger(sess[FIELD_SURVEY_ID])
    ? sess[FIELD_SURVEY_ID]
    : sampleSurveyIdFromWorkPackage(sess && sess.work_package_id);
  const surveys = sampling ? sampling[IPSO_REF_SURVEYS] || [] : [];
  const survey = surveys.find((row) => row && row[FIELD_SURVEY_ID] === surveyId);
  const maxNumbers = survey ? survey[IPSO_REF_SAMPLE_AREA_MAX_NUMBERS] || {} : {};
  const value = maxNumbers[String(sampleAreaId)];
  return Number.isInteger(value) ? value : null;
}

function paiPreservedRows(reference) {
  const pai = reference && reference[IPSO_REF_PAI];
  return pai ? pai[IPSO_REF_PRESERVED_TREES] || [] : [];
}

function paiNumberExists(reference, parcelId, number) {
  return paiPreservedRows(reference).some((row) =>
    row && row[FIELD_PARCEL_ID] === parcelId && row[FIELD_NUMBER] === number
  );
}

function paiMaxNumberForParcel(reference, parcelId) {
  const rows = paiPreservedRows(reference);
  let maxNumber = null;
  for (const row of rows) {
    if (!row || row[FIELD_PARCEL_ID] !== parcelId) continue;
    const number = row[FIELD_NUMBER];
    if (Number.isInteger(number) && (maxNumber == null || number > maxNumber)) {
      maxNumber = number;
    }
  }
  return maxNumber;
}

function sampleRecordContext(reference, sess, tree, parcelId) {
  const area = sampleAreaForTree(reference, sess, tree, parcelId);
  const sampleAreaId = Number.isInteger(tree[FIELD_SAMPLE_AREA_ID])
    ? tree[FIELD_SAMPLE_AREA_ID]
    : area ? area[FIELD_SAMPLE_AREA_ID] : null;
  const coppice = typeof tree[FIELD_COPPICE] === 'boolean'
    ? tree[FIELD_COPPICE]
    : area && typeof area[FIELD_COPPICE] === 'boolean' ? area[FIELD_COPPICE] : null;
  return {
    [FIELD_SAMPLE_AREA_ID]: sampleAreaId,
    [FIELD_COPPICE]: coppice,
    [FIELD_SHOOT]: Number.isInteger(tree[FIELD_SHOOT]) ? tree[FIELD_SHOOT] : 0,
    [FIELD_STANDARD]: !!tree[FIELD_STANDARD],
    [FIELD_L10_MM]: Number.isInteger(tree[FIELD_L10_MM]) ? tree[FIELD_L10_MM] : 0,
    [FIELD_PRESSLER_COEFF]: tree[FIELD_PRESSLER_COEFF] || PRESSLER_DEFAULT,
    [FIELD_PRESERVED]: !!tree[FIELD_PRESERVED],
  };
}

function freeSurveyRecordContext(sess, tree) {
  return {
    [FIELD_PRESERVED]: !!tree[FIELD_PRESERVED],
    [FIELD_OPERATOR]: tree[FIELD_OPERATOR] || sess.operatore || '',
    [FIELD_NOTE]: tree[FIELD_NOTE] || '',
  };
}

function sampleAreaForTree(reference, sess, tree, parcelId) {
  if (!reference || !reference[IPSO_REF_SAMPLING]) return null;
  const areas = reference[IPSO_REF_SAMPLING][IPSO_REF_SAMPLE_AREAS] || [];
  if (Number.isInteger(tree[FIELD_SAMPLE_AREA_ID])) {
    const stored = areas.find((area) =>
      area && area[FIELD_SAMPLE_AREA_ID] === tree[FIELD_SAMPLE_AREA_ID]
    );
    if (stored) return stored;
  }
  if (tree[FIELD_LAT] == null || tree[FIELD_LON] == null) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const area of areas) {
    if (!area || area.compresa !== sess.compresa || area[FIELD_PARCEL_ID] !== parcelId) {
      continue;
    }
    if (area[FIELD_LAT] == null || area[FIELD_LON] == null) continue;
    const distance = distanceMeters(
      tree[FIELD_LAT], tree[FIELD_LON], area[FIELD_LAT], area[FIELD_LON]
    );
    const radius = Number.isFinite(area[FIELD_R_M]) ? area[FIELD_R_M] : DEFAULT_SAMPLE_RADIUS_M;
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
  const row = (reference[IPSO_REF_SPECIES] || []).find((s) => s.common === name);
  if (!row || !Number.isInteger(row.id)) {
    throw new Error(S.UPLOAD_ERROR_SPECIES_ID_MISSING(name));
  }
  return row.id;
}

function regionIdForCompresa(reference, compresa) {
  const row = (reference[IPSO_REF_PARCELS] || []).find((p) => p.compresa === compresa);
  if (!row || !Number.isInteger(row[FIELD_REGION_ID])) {
    throw new Error(S.UPLOAD_ERROR_REGION_ID_MISSING(compresa));
  }
  return row[FIELD_REGION_ID];
}

function parcelForName(reference, compresa, particella) {
  const row = (reference[IPSO_REF_PARCELS] || []).find(
    (p) => p.compresa === compresa && p.particella === particella
  );
  if (!row || !Number.isInteger(row[FIELD_PARCEL_ID]) || !Number.isInteger(row[FIELD_REGION_ID])) {
    throw new Error(S.UPLOAD_ERROR_PARCEL_ID_MISSING(compresa, particella));
  }
  return row;
}


function uploadPayloadHasFiles(payload) {
  const records = Array.isArray(payload && payload[RECORDS])
    ? payload[RECORDS] : [];
  return records.some((record) =>
    Array.isArray(record && record[FIELD_PHOTOS]) &&
    record[FIELD_PHOTOS].some((photo) => photo && photo.blob)
  );
}

function stripUploadPayloadFiles(payload) {
  return {
    ...payload,
    [RECORDS]: (payload[RECORDS] || []).map((record) => ({
      ...record,
      [FIELD_PHOTOS]: (record[FIELD_PHOTOS] || []).map((photo) => {
        const { blob, ...metadata } = photo || {};
        return metadata;
      }),
    })),
  };
}

function uploadMaxBytes(reference) {
  const raw = reference && reference[IPSO_REF_UPLOAD] &&
    reference[IPSO_REF_UPLOAD][FIELD_MAX_BYTES];
  return Number.isInteger(raw) && raw > 0 ? raw : DEFAULT_UPLOAD_MAX_BYTES;
}

function estimatedUploadBytes(payload) {
  const wirePayload = stripUploadPayloadFiles(payload);
  let total = utf8Bytes(JSON.stringify(wirePayload));
  if (!uploadPayloadHasFiles(payload)) return total;
  total += MULTIPART_ESTIMATE_OVERHEAD_BYTES;
  total += utf8Bytes(IPSO_UPLOAD_MULTIPART_PAYLOAD_FIELD);
  for (const record of payload[RECORDS] || []) {
    for (const photo of record[FIELD_PHOTOS] || []) {
      if (!photo || !photo.blob) continue;
      total += MULTIPART_ESTIMATE_PART_OVERHEAD_BYTES;
      total += utf8Bytes(IPSO_UPLOAD_MULTIPART_PHOTO_PREFIX + photo[FIELD_CLIENT_PHOTO_ID]);
      total += utf8Bytes(photo[FIELD_ORIGINAL_FILENAME] || 'photo');
      total += Number.isInteger(photo[FIELD_SIZE_BYTES])
        ? photo[FIELD_SIZE_BYTES]
        : Number.isInteger(photo.blob.size) ? photo.blob.size : 0;
    }
  }
  return total;
}

function utf8Bytes(value) {
  if (typeof TextEncoder !== 'undefined') {
    return new TextEncoder().encode(String(value || '')).length;
  }
  return unescape(encodeURIComponent(String(value || ''))).length;
}

function multipartBody(payload) {
  const form = new FormData();
  const wirePayload = stripUploadPayloadFiles(payload);
  form.append(IPSO_UPLOAD_MULTIPART_PAYLOAD_FIELD, JSON.stringify(wirePayload));
  for (const record of payload[RECORDS] || []) {
    for (const photo of record[FIELD_PHOTOS] || []) {
      if (!photo || !photo.blob) continue;
      const clientId = photo[FIELD_CLIENT_PHOTO_ID];
      const filename = photo[FIELD_ORIGINAL_FILENAME] || clientId || 'photo';
      form.append(IPSO_UPLOAD_MULTIPART_PHOTO_PREFIX + clientId, photo.blob, filename);
    }
  }
  return form;
}

// Posts the canonical staged JSON payload. Resolves with { duplicate: bool } on
// 200, throws UploadError otherwise. Caller passes signal for cancellation.
async function uploadSession(args) {
  const { token, sessionId, payload, signal, maxBytes } = args;
  const hasFiles = uploadPayloadHasFiles(payload);
  const estimatedBytes = estimatedUploadBytes(payload);
  if (Number.isInteger(maxBytes) && maxBytes > 0 && estimatedBytes > maxBytes) {
    throw new UploadError('hard:too_large', String(estimatedBytes));
  }
  const headers = {
    'X-Ipso-Session-Id': sessionId,
  };
  if (!hasFiles) headers['Content-Type'] = 'application/json';
  if (token) headers.Authorization = 'Bearer ' + token;
  const body = hasFiles ? multipartBody(payload) : JSON.stringify(payload);

  let resp;
  try {
    resp = await fetch('/api/ipso/uploads/', {
      method: 'POST',
      headers,
      body,
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
  UPLOAD_SCHEMA_VERSION,
  UPLOAD_MODE_MARTELLATE: IPSO_MODE_MARTELLATE,
  UPLOAD_MODE_SAMPLES: IPSO_MODE_SAMPLES,
  UPLOAD_MODE_FREE_SURVEY: IPSO_MODE_FREE_SURVEY,
  UPLOAD_MODE_OBSERVATIONS: IPSO_MODE_OBSERVATIONS,
  DEFAULT_SAMPLE_RADIUS_M,
  BACKOFF_SCHEDULE_MS, BACKOFF_CAP_MS,
  backoffMs, classifyHttp, classifyNetwork, distanceMeters,
  estimatedUploadBytes, uploadMaxBytes,
  isSupportedUploadMode, validateRecordNumbers, sampleSurveyIdFromWorkPackage,
  sampleMaxNumberForArea,
  paiNumberExists, paiMaxNumberForParcel,
  UploadError, buildUploadPayload, uploadPayloadHasFiles,
  stripUploadPayloadFiles, multipartBody, uploadSession,
};

if (typeof module !== 'undefined') module.exports = upload;
