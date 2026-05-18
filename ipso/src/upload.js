// Upload an ipso session CSV to the ipso-upload endpoint.
//
// Pure-logic helpers (backoff, classify) are testable in node; the
// network-touching uploadSession() is exercised in browser via the
// screen-upload state machine, and in node tests via a mocked
// globalThis.fetch.
//
// See docs/superpowers/specs/2026-05-17-ipso-upload-design.md for the
// wire format and retry contract.
'use strict';

const BACKOFF_SCHEDULE_MS = [2000, 4000, 8000, 16000];
const BACKOFF_CAP_MS = 30000;

// 1-based attempt number → wait BEFORE that attempt. Attempt 0 is "no
// wait yet", attempt N>0 picks index N-1 from the schedule (or the cap).
function backoffMs(attempt) {
  if (!Number.isFinite(attempt) || attempt <= 0) return 0;
  const i = Math.floor(attempt) - 1;
  if (i < BACKOFF_SCHEDULE_MS.length) return BACKOFF_SCHEDULE_MS[i];
  return BACKOFF_CAP_MS;
}

// HTTP status → outcome class. The state machine drives bail/retry off the
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

// Posts the CSV. Resolves with { duplicate: bool } on 200, throws
// UploadError otherwise. Caller passes signal for cancellation.
//
// base + token + schemaVersion come from upload-config.js (browser
// globals); the function accepts them as args so tests can inject a
// fake config without mucking with globals.
async function uploadSession(args) {
  const { base, token, schemaVersion, sessionId, csvText, signal } = args;
  let resp;
  try {
    resp = await fetch(base + '/upload', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'text/csv; charset=utf-8',
        'X-Ipso-Session-Id': sessionId,
        'X-Ipso-Schema-Version': '' + schemaVersion,
      },
      body: csvText,
      signal,
    });
  } catch (e) {
    if (e && e.name === 'AbortError') throw new UploadError('aborted');
    throw new UploadError(classifyNetwork(), e && e.message);
  }
  const klass = classifyHttp(resp.status);
  if (klass === 'ok') {
    let payload = {};
    try { payload = await resp.json(); } catch (_) {}
    return { duplicate: !!payload.duplicate, storedAs: payload.stored_as };
  }
  let detail = '';
  try {
    const payload = await resp.json();
    detail = payload && payload.error ? payload.error : '';
  } catch (_) {}
  throw new UploadError(klass, detail);
}

const upload = {
  BACKOFF_SCHEDULE_MS, BACKOFF_CAP_MS,
  backoffMs, classifyHttp, classifyNetwork,
  UploadError, uploadSession,
};

if (typeof module !== 'undefined') module.exports = upload;
