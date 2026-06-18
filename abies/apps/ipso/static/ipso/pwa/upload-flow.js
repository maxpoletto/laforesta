// Upload screen state machine for Ipso.
//
// Owns retry/backoff, upload status persistence, and the screen-upload UI.
// App-level code passes the few lifecycle hooks this module needs.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof S === 'undefined') {
  Object.assign(globalThis, require('./strings.js'));
}

function createUploadFlow(opts) {
  let current = null;

  function wire() {
    document.getElementById('upload-title').textContent = S.UPLOAD_TITLE;
    document.getElementById('btn-upload-bail').textContent = S.UPLOAD_BAIL;
    document.getElementById('btn-upload-bail').addEventListener('click', bail);
  }

  function enter(sessionId, uploadPayload, treeCount) {
    if (current && current.retryTimer) clearTimeout(current.retryTimer);
    if (current && current.abortController) {
      try { current.abortController.abort(); } catch (_) {}
    }
    opts.stopRecording();
    current = {
      sessionId,
      payload: uploadPayload,
      treeCount,
      attempt: 0,
      abortController: null,
      retryTimer: null,
    };
    document.getElementById('upload-detail').textContent = '';
    document.getElementById('upload-detail').classList.remove('error');
    opts.showScreen('screen-upload');
    opts.acquireWakeLock();
    scheduleAttempt(0);
  }

  function scheduleAttempt(waitMs) {
    if (!current) return;
    if (waitMs > 0) {
      const secs = Math.ceil(waitMs / 1000);
      document.getElementById('upload-detail').textContent =
        S.UPLOAD_NEXT_RETRY_IN(secs);
      current.retryTimer = setTimeout(runAttempt, waitMs);
    } else {
      runAttempt();
    }
  }

  async function runAttempt() {
    if (!current) return;
    current.retryTimer = null;
    current.attempt += 1;
    document.getElementById('upload-attempt').textContent =
      S.UPLOAD_ATTEMPT(current.attempt);

    const ac = new AbortController();
    current.abortController = ac;
    try {
      await upload.uploadSession({
        token: opts.uploadToken(),
        sessionId: current.sessionId,
        payload: current.payload,
        signal: ac.signal,
      });
      await completeUpload();
    } catch (err) {
      if (err && err.klass === 'aborted') return;
      failAttempt(err);
    } finally {
      if (current) current.abortController = null;
    }
  }

  function failAttempt(err) {
    if (!current) return;
    const klass = (err && err.klass) || 'soft:network';
    const isHard = klass.startsWith('hard:');
    const detailEl = document.getElementById('upload-detail');
    detailEl.classList.toggle('error', isHard);
    detailEl.textContent = uploadErrorMessage(klass);
    if (isHard) {
      document.getElementById('upload-attempt').textContent = '';
      return;
    }
    scheduleAttempt(upload.backoffMs(current.attempt));
  }

  function uploadErrorMessage(klass) {
    switch (klass) {
      case 'hard:auth':         return S.UPLOAD_ERROR_AUTH;
      case 'hard:conflict':     return S.UPLOAD_ERROR_CONFLICT;
      case 'hard:invalid_csv':  return S.UPLOAD_ERROR_INVALID;
      case 'hard:too_large':    return S.UPLOAD_ERROR_TOO_LARGE;
      case 'soft:rate_limited': return S.UPLOAD_ERROR_RATE_LIMITED;
      case 'soft:server':       return S.UPLOAD_ERROR_SERVER;
      case 'soft:network':      return S.UPLOAD_ERROR_NETWORK;
      default:                  return klass;
    }
  }

  async function completeUpload() {
    if (!current) return;
    const sessionId = current.sessionId;
    const treeCount = current.treeCount;
    current = null;
    try {
      await Store.setSessionUploadStatus(
        opts.db(), sessionId, Store.UPLOAD_STATUS_UPLOADED
      );
      await Store.setSessionStatus(opts.db(), sessionId, Store.STATUS_EXPORTED);
    } catch (e) {
      opts.showToast(S.TOAST_UPLOAD_STATE_ERROR(e.message));
    }
    opts.showToast(S.UPLOAD_SUCCESS_TOAST);
    end(treeCount, true);
  }

  async function bail() {
    if (!current) return;
    if (current.retryTimer) {
      clearTimeout(current.retryTimer);
      current.retryTimer = null;
    }
    if (current.abortController) {
      try { current.abortController.abort(); } catch (_) {}
    }
    const sessionId = current.sessionId;
    const treeCount = current.treeCount;
    current = null;
    try {
      await Store.setSessionUploadStatus(
        opts.db(), sessionId, Store.UPLOAD_STATUS_LOCAL_ONLY
      );
      await Store.setSessionStatus(opts.db(), sessionId, Store.STATUS_EXPORTED);
    } catch (e) {
      opts.showToast(S.TOAST_STATE_SAVE_ERROR(e.message));
    }
    opts.showToast(S.UPLOAD_LOCAL_ONLY_TOAST);
    end(treeCount, false);
  }

  function end(treeCount, uploaded) {
    current = null;
    opts.releaseWakeLock();
    opts.showDone(treeCount, uploaded);
  }

  return { wire, enter };
}

if (typeof module !== 'undefined') module.exports = { createUploadFlow };
