// ipso — app shell and state machine.
//
// The state machine lives in this file, but the heavy lifting is delegated
// to the pure modules (csv, ipso, session) and the side-effectful modules
// (store, gps, numpad). This file is mostly UI wiring.
'use strict';

const APP_VERSION = '0.2.3';

const State = {
  reference: null,    // parsed reference.json
  db: null,           // open IDBDatabase
  session: null,      // current session row (null until pre_session submits)
  specie: '',
  hMeasured: false,   // flips true once the user touches the h field
  inAutoFill: false,  // true while recomputeAutoH() is writing to h
  saveLockUntil: 0,   // ms timestamp; Salva is disabled while now < this
  gps: null,          // GPS controller
  numpad: null,       // numpad controller
  lastTreeRow: null,  // most-recent tree in the current session
  wakeLock: null,     // active WakeLockSentinel during recording (or null)
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  document.getElementById('footer-version').textContent =
    'v' + APP_VERSION;
  document.getElementById('pre-title').textContent = S.PRE_NEW_SESSION;
  document.getElementById('lbl-operatore').textContent = S.PRE_OPERATORE;
  document.getElementById('lbl-data').textContent = S.PRE_DATA;
  document.getElementById('lbl-compresa').textContent = S.PRE_COMPRESA;
  document.getElementById('lbl-particella').textContent = S.PRE_PARTICELLA;
  document.getElementById('lbl-catastrofata').textContent = S.PRE_CATASTROFATA;
  document.getElementById('btn-start').textContent = S.PRE_START;
  document.getElementById('lbl-specie').textContent = S.REC_SPECIE;
  document.getElementById('lbl-d').textContent = S.REC_D;
  document.getElementById('lbl-h').textContent = S.REC_H;
  document.getElementById('btn-save').textContent = S.REC_SAVE;
  document.getElementById('btn-end').textContent = S.REC_END;
  document.getElementById('resume-title').textContent = S.RESUME_TITLE;
  document.getElementById('resume-body').textContent = S.RESUME_BODY;
  document.getElementById('end-title').textContent = S.END_TITLE;
  document.getElementById('end-cancel').textContent = S.REC_CANCEL;
  document.getElementById('end-confirm').textContent = S.END_CONFIRM;
  document.getElementById('edit-title').textContent = S.REC_EDIT_LAST;
  document.getElementById('edit-cancel').textContent = S.REC_CANCEL;
  document.getElementById('edit-delete').textContent = S.REC_DELETE_LAST;
  document.getElementById('done-title').textContent = S.DONE_TITLE;
  document.getElementById('btn-new-session').textContent = S.DONE_NEW;

  // Register service worker but never block the UI on it.
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {
      // Quietly ignore; the app works without one (just no offline).
    });
  }

  try {
    State.reference = await fetchReference();
  } catch (e) {
    showToast('Errore caricamento reference.json: ' + e.message);
    return;
  }

  try {
    State.db = await Store.openDb();
  } catch (e) {
    showToast('Errore apertura database: ' + e.message);
    return;
  }

  populateOperatore();
  populateComprese();
  wirePreSession();
  wireRecording();
  wireDone();

  // Re-acquire the screen wake lock when the page returns to the
  // foreground (the Wake Lock API auto-releases on visibility loss).
  setupWakeLockVisibility();

  // Resume-on-open: check for any open sessions before showing pre-session.
  const open = await Store.listOpenSessions(State.db);
  if (open && open.length > 0) {
    showResumeModal(open);
  } else {
    showScreen('screen-pre');
  }

  // Request persistent storage (R2). Doesn't require a user gesture.
  requestPersist();
}

async function fetchReference() {
  const r = await fetch('reference.json', { cache: 'reload' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return await r.json();
}

async function requestPersist() {
  if (!navigator.storage || !navigator.storage.persist) return;
  const granted = await navigator.storage.persist();
  if (!granted) showBanner(S.STORAGE_WARNING);
}

// ---------------------------------------------------------------------------
// Pre-session
// ---------------------------------------------------------------------------

function populateOperatore() {
  const stored = localStorage.getItem('ipso.operatore') || '';
  document.getElementById('in-operatore').value = stored;
  const today = new Date();
  document.getElementById('in-data').value = formatYMD(today);
}

function formatYMD(d) {
  return d.getFullYear() + '-' +
    String(d.getMonth() + 1).padStart(2, '0') + '-' +
    String(d.getDate()).padStart(2, '0');
}

function populateComprese() {
  const sel = document.getElementById('in-compresa');
  sel.replaceChildren();
  const comprese = [...new Set(State.reference.parcels.map((p) => p.compresa))].sort();
  appendOption(sel, '', S.PRE_PICK_COMPRESA, true);
  for (const c of comprese) appendOption(sel, c, c);

  sel.addEventListener('change', () => populateParticelle(sel.value));
}

function populateParticelle(compresa) {
  const sel = document.getElementById('in-particella');
  sel.replaceChildren();
  if (!compresa) {
    sel.disabled = true;
    appendOption(sel, '', S.PRE_PICK_PARTICELLA, true);
    return;
  }
  sel.disabled = false;
  appendOption(sel, '', S.PRE_PICK_PARTICELLA, true);
  for (const p of State.reference.parcels) {
    if (p.compresa === compresa) appendOption(sel, p.particella, p.particella);
  }
}

function appendOption(sel, value, label, selected) {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = label;
  if (selected) o.selected = true;
  sel.appendChild(o);
}

function wirePreSession() {
  const cb = document.getElementById('in-catastrofata');
  const particellaField = document.getElementById('field-particella');
  const particellaSel = document.getElementById('in-particella');

  cb.addEventListener('change', () => {
    const on = cb.checked;
    particellaField.hidden = on;
    // Required attribute drives the implicit HTML5 form validation. Pull
    // it off when the field is hidden so submit doesn't refuse.
    if (on) particellaSel.removeAttribute('required');
    else particellaSel.setAttribute('required', '');
  });

  document.getElementById('pre-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const operatore = document.getElementById('in-operatore').value.trim();
    const data = document.getElementById('in-data').value;
    const compresa = document.getElementById('in-compresa').value;
    const catastrofata = cb.checked;
    const particella = catastrofata
      ? ''
      : document.getElementById('in-particella').value;
    if (!operatore || !data || !compresa) return;
    if (!catastrofata && !particella) return;

    localStorage.setItem('ipso.operatore', operatore);

    try {
      const sess = await Store.startSession(State.db, {
        data, compresa, particella, operatore, catastrofata,
      });
      State.session = sess;
      State.lastTreeRow = null;
      enterRecording();
    } catch (err) {
      showToast('Errore avvio sessione: ' + err.message);
    }
  });
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

function wireRecording() {
  const inD = document.getElementById('in-d');
  const inH = document.getElementById('in-h');
  State.numpad = createNumpad({
    container: document.getElementById('numpad'),
    inputs: { d: inD, h: inH },
    onChange: (field) => {
      if (field === 'h' && !State.inAutoFill) State.hMeasured = true;
      if (field === 'd' && !State.hMeasured) recomputeAutoH();
      updateSaveEnabled();
    },
  });
  State.numpad.mount();

  document.getElementById('in-specie').addEventListener('change', (e) => {
    State.specie = e.target.value;
    if (!State.hMeasured) recomputeAutoH();
    updateSaveEnabled();
  });

  document.getElementById('btn-save').addEventListener('click', onSave);
  document.getElementById('btn-end').addEventListener('click', () => {
    document.getElementById('end-body').textContent =
      S.END_BODY(State.session.tree_count);
    showModal('modal-confirm-end');
  });

  document.getElementById('end-confirm').addEventListener('click', onEnd);
  document.getElementById('end-cancel').addEventListener('click', () => {
    hideModal('modal-confirm-end');
  });

  document.getElementById('pill-action').addEventListener('click', () => {
    if (!State.lastTreeRow) return;
    document.getElementById('edit-body').textContent =
      S.pill(State.lastTreeRow);
    showModal('modal-edit-last');
  });
  document.getElementById('edit-delete').addEventListener('click', onDeleteLast);
  document.getElementById('edit-cancel').addEventListener('click', () => {
    hideModal('modal-edit-last');
  });
}

function enterRecording() {
  populateSpecie();
  const where = S.where(State.session);
  document.getElementById('rec-where').textContent = where;
  document.getElementById('sub-status').textContent = where;
  resetEntryFields();
  refreshPill();
  startGps();
  acquireWakeLock();
  showScreen('screen-rec');
}

function populateSpecie() {
  const sel = document.getElementById('in-specie');
  sel.replaceChildren();
  appendOption(sel, '', S.REC_PICK_SPECIE, true);
  const ordered = State.reference.species.slice().sort(
    (a, b) => a.sort_order - b.sort_order
  );
  for (const sp of ordered) appendOption(sel, sp.common, sp.common);
  // Sticky: if a prior session set a default in localStorage, prefer that.
  const sticky = localStorage.getItem('ipso.specie') || '';
  if (sticky && ordered.some((s) => s.common === sticky)) {
    sel.value = sticky;
    State.specie = sticky;
  } else {
    State.specie = '';
  }
}

function resetEntryFields() {
  State.numpad.clear();
  State.numpad.setFocus('d');
  State.hMeasured = false;
  document.getElementById('hint-autoh').hidden = true;
  updateSaveEnabled();
}

function refreshPill() {
  const pill = document.getElementById('pill-text');
  const btn = document.getElementById('pill-action');
  if (!State.lastTreeRow) {
    pill.textContent = S.REC_NO_LAST;
    btn.hidden = true;
  } else {
    pill.textContent = S.REC_LAST_PREFIX + ' n.' +
      State.lastTreeRow.seq + ' · ' + session.summarizePill(State.lastTreeRow);
    btn.textContent = S.REC_EDIT_LAST + '/' + S.REC_DELETE_LAST;
    btn.hidden = false;
  }
}

function startGps() {
  if (State.gps) State.gps.stop();
  const dot = document.getElementById('gps-dot');
  const text = document.getElementById('gps-text');
  State.gps = createGps((st) => {
    dot.className = 'gps-dot ' + st.tier;
    if (st.fix && (st.age == null || st.age < 10000)) {
      let line =
        st.fix.lat.toFixed(5) + ' ' + st.fix.lng.toFixed(5) +
        ' ±' + Math.round(st.fix.acc) + ' m';
      // Show fix age when noticeable, so the operator can see at a glance
      // when the indicator is reporting an older reading.
      const ageSec = st.age != null ? Math.round(st.age / 1000) : 0;
      if (ageSec >= 2) line += ' · ' + ageSec + 's';
      text.textContent = line;
    } else if (st.error === 'denied') {
      text.textContent = S.GPS_DENIED;
    } else {
      text.textContent = S.REC_GPS_WAITING;
    }
  });
  State.gps.start();
}

// ---------------------------------------------------------------------------
// Wake lock
// ---------------------------------------------------------------------------

async function acquireWakeLock() {
  if (!('wakeLock' in navigator)) return;
  if (State.wakeLock) return;
  try {
    const lock = await navigator.wakeLock.request('screen');
    State.wakeLock = lock;
    lock.addEventListener('release', () => {
      // Auto-released on visibility loss; clear our reference and let the
      // visibilitychange handler re-acquire when we come back.
      if (State.wakeLock === lock) State.wakeLock = null;
    });
  } catch (_) {
    // Lock failures are non-fatal — recording still works without the
    // lock, the OS just gets to throttle us more aggressively.
  }
}

function releaseWakeLock() {
  if (!State.wakeLock) return;
  const lock = State.wakeLock;
  State.wakeLock = null;
  try { lock.release(); } catch (_) {}
}

function setupWakeLockVisibility() {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState !== 'visible') return;
    // Only re-acquire while a recording session is active.
    if (State.session && !State.wakeLock) acquireWakeLock();
  });
}

function recomputeAutoH() {
  const compresa = State.session ? State.session.compresa : '';
  const ipsTable = State.reference.ipsometrica;
  const eq = ipso.lookup(ipsTable, compresa, State.specie);
  const d = parseInt(State.numpad.value('d'), 10);
  const hint = document.getElementById('hint-autoh');

  State.inAutoFill = true;
  try {
    if (!State.specie) {
      State.numpad.setValue('h', '');
      hint.hidden = true;
      return;
    }
    if (!eq) {
      State.numpad.setValue('h', '');
      hint.textContent = S.REC_AUTO_H_MISSING;
      hint.hidden = false;
      return;
    }
    hint.hidden = true;
    if (Number.isFinite(d) && d > 0) {
      const h = ipso.computeH(eq, d);
      State.numpad.setValue('h', h == null ? '' : '' + h);
    } else {
      State.numpad.setValue('h', '');
    }
  } finally {
    State.inAutoFill = false;
  }
}

function currentRecord() {
  const d = parseInt(State.numpad.value('d'), 10);
  const h = parseInt(State.numpad.value('h'), 10);
  return {
    specie: State.specie || '',
    d_cm: Number.isFinite(d) ? d : null,
    h_m: Number.isFinite(h) ? h : null,
    h_measured: State.hMeasured ? 1 : 0,
  };
}

function updateSaveEnabled() {
  const rec = currentRecord();
  const errs = session.validateTree(rec);
  const btn = document.getElementById('btn-save');
  btn.disabled = errs.length > 0 || Date.now() < State.saveLockUntil;
}

async function onSave() {
  const rec = currentRecord();
  if (session.validateTree(rec).length > 0) return;

  // Snapshot GPS atomically into the record (R9).
  const gps = State.gps ? State.gps.snapshot() : null;
  const full = Object.assign({}, rec, {
    lat: gps ? gps.lat : null,
    lng: gps ? gps.lng : null,
    acc_m: gps ? gps.acc_m : null,
  });

  // R12: 300 ms cooldown after save completes.
  State.saveLockUntil = Date.now() + 300;
  document.getElementById('btn-save').disabled = true;

  try {
    const row = await Store.addTree(State.db, State.session.id, full);
    // Mirror tree_count from the freshly read row.
    State.session.tree_count = row.seq;
    State.lastTreeRow = row;
    localStorage.setItem('ipso.specie', State.specie);
    refreshPill();
    resetEntryFields();

    if (session.shouldBackup(row.seq)) {
      await downloadBackup(row.seq);
    }
  } catch (e) {
    showToast('Errore salvataggio: ' + e.message);
  } finally {
    setTimeout(updateSaveEnabled, 320);
  }
}

async function onDeleteLast() {
  hideModal('modal-edit-last');
  if (!State.lastTreeRow) return;
  try {
    await Store.deleteTree(State.db, State.session.id, State.lastTreeRow.id);
    // Refresh from DB so we get the new "last" and the corrected tree_count.
    State.session = await Store.getSession(State.db, State.session.id);
    State.lastTreeRow = await Store.lastTree(State.db, State.session.id);
    refreshPill();
    updateSaveEnabled();
  } catch (e) {
    showToast('Errore eliminazione: ' + e.message);
  }
}

async function onEnd() {
  hideModal('modal-confirm-end');
  try {
    const trees = await Store.listTrees(State.db, State.session.id);
    trees.sort((a, b) => a.seq - b.seq);
    await Store.setSessionStatus(State.db, State.session.id, Store.STATUS_EXPORTED);
    downloadFinal(State.session, trees);
    enterDone(trees.length);
  } catch (e) {
    showToast('Errore esportazione: ' + e.message);
  }
}

function downloadFinal(sess, trees) {
  const text = csv.formatFile(sess, trees);
  const name = csv.filename(sess, new Date(), 'final');
  downloadText(text, name);
}

async function downloadBackup(seq) {
  const trees = await Store.listTrees(State.db, State.session.id);
  trees.sort((a, b) => a.seq - b.seq);
  const text = csv.formatFile(State.session, trees);
  const name = csv.filename(State.session, new Date(), 'backup', seq);
  downloadText(text, name);
  showToast(S.BACKUP_SAVED(seq));
}

// ---------------------------------------------------------------------------
// Done screen
// ---------------------------------------------------------------------------

function wireDone() {
  document.getElementById('btn-new-session').addEventListener('click', () => {
    State.session = null;
    State.lastTreeRow = null;
    if (State.gps) { State.gps.stop(); State.gps = null; }
    document.getElementById('sub-status').textContent = '';
    document.getElementById('pre-form').reset();
    // Clear catastrofata UI state — the form.reset() above unchecks the
    // box, but we also need to un-hide the particella row that the
    // change handler had toggled.
    document.getElementById('field-particella').hidden = false;
    document.getElementById('in-particella').setAttribute('required', '');
    populateOperatore();
    populateComprese();
    showScreen('screen-pre');
  });
}

function enterDone(n) {
  if (State.gps) { State.gps.stop(); State.gps = null; }
  releaseWakeLock();
  document.getElementById('done-body').textContent = S.DONE_BODY(n);
  showScreen('screen-done');
}

// ---------------------------------------------------------------------------
// Resume modal
// ---------------------------------------------------------------------------

function showResumeModal(sessions) {
  const list = document.getElementById('resume-list');
  list.replaceChildren();
  for (const s of sessions) {
    const li = document.createElement('li');
    li.className = 'resume-item';
    const meta = document.createElement('div');
    meta.className = 'resume-meta';
    meta.textContent =
      formatItalianDate(s.data) + ' · ' + S.where(s) +
      ' · ' + (s.operatore || '—') + ' · ' + (s.tree_count || 0) + ' alberi';
    li.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'resume-actions';
    const resume = mkBtn(S.RESUME_RESUME, 'btn-primary', async () => {
      State.session = s;
      State.lastTreeRow = await Store.lastTree(State.db, s.id);
      hideModal('modal-resume');
      enterRecording();
    });
    const exp = mkBtn(S.RESUME_EXPORT, 'btn-secondary', async () => {
      const trees = await Store.listTrees(State.db, s.id);
      trees.sort((a, b) => a.seq - b.seq);
      await Store.setSessionStatus(State.db, s.id, Store.STATUS_EXPORTED);
      downloadFinal(s, trees);
      li.remove();
      if (!list.children.length) {
        hideModal('modal-resume');
        showScreen('screen-pre');
      }
    });
    const discard = mkBtn(S.RESUME_DISCARD, 'btn-danger', async () => {
      await Store.setSessionStatus(State.db, s.id, Store.STATUS_ABANDONED);
      li.remove();
      if (!list.children.length) {
        hideModal('modal-resume');
        showScreen('screen-pre');
      }
    });
    actions.appendChild(resume);
    actions.appendChild(exp);
    actions.appendChild(discard);
    li.appendChild(actions);
    list.appendChild(li);
  }
  showModal('modal-resume');
}

function mkBtn(label, klass, handler) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = klass;
  b.textContent = label;
  b.addEventListener('click', handler);
  return b;
}

function formatItalianDate(ymd) {
  try { return csv.formatDate(ymd); } catch (_) { return ymd; }
}

// ---------------------------------------------------------------------------
// UI primitives
// ---------------------------------------------------------------------------

function showScreen(id) {
  for (const el of document.querySelectorAll('main > section.screen')) {
    el.hidden = el.id !== id;
  }
}

function showModal(id) {
  document.getElementById(id).classList.remove('hidden');
}
function hideModal(id) {
  document.getElementById(id).classList.add('hidden');
}

let toastTimer = null;
function showToast(msg, ms) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), ms || 2500);
}

function showBanner(msg) {
  const b = document.getElementById('banner-storage');
  b.textContent = msg;
  b.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Kickoff
// ---------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  boot().catch((e) => showToast('Errore avvio: ' + e.message));
});
