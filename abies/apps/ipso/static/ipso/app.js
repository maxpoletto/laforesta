// ipso — app shell and state machine.
//
// The state machine lives in this file, but the heavy lifting is delegated
// to the pure modules (csv, ipso, session) and the side-effectful modules
// (store, gps, numpad). This file is mostly UI wiring.
'use strict';

// APP_VERSION is defined in version.js (loaded before this script).

const State = {
  mode: null,         // current IpsoModes entry
  reference: null,    // parsed reference.json
  terreni: null,      // parsed terreni.geojson features array (compresa-wide)
  db: null,           // open IDBDatabase
  session: null,      // current session row (null until pre_session submits)
  specie: '',
  hMeasured: false,   // flips true once the user touches the h field
  inAutoFill: false,  // true while recomputeAutoH() is writing to h
  saveLockUntil: 0,   // ms timestamp; Salva is disabled while now < this
  gps: null,          // GPS controller
  locator: null,      // parcel-locator instance (one per recording session)
  numpad: null,       // numpad controller
  lastTreeRow: null,  // most-recent tree in the current session
  wakeLock: null,     // active WakeLockSentinel during recording (or null)
  upload: null,       // { sessionId, attempt, abortController, retryTimer, ... } | null
  currentScreen: null, // id of the visible screen
  lastFix: null,       // most recent fresh GPS fix, { lat, lon, acc, t }
  map: null,           // lazy orientation map controller
  mapReturnScreen: 'screen-rec',
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  setMode(IpsoModes.MARTELLATE);
  document.getElementById('footer-version').textContent =
    'v' + APP_VERSION;
  document.getElementById('lbl-operatore').textContent = S.PRE_OPERATOR;
  document.getElementById('lbl-data').textContent = S.PRE_DATA;
  document.getElementById('lbl-compresa').textContent = S.PRE_COMPRESA;
  document.getElementById('lbl-catastrofata').textContent = S.PRE_CATASTROFATA;
  document.getElementById('btn-start').textContent = S.PRE_START;
  document.getElementById('lbl-specie').textContent = S.REC_SPECIE;
  document.getElementById('lbl-numero').textContent = S.REC_NUMBER;
  document.getElementById('lbl-particella-rec').textContent = S.PRE_PARTICELLA;
  document.getElementById('lbl-gruppo').textContent = S.REC_GRUPPO;
  document.getElementById('lbl-d').textContent = S.REC_D;
  document.getElementById('lbl-h').textContent = S.REC_H;
  document.getElementById('btn-save').textContent = S.REC_SAVE;
  document.getElementById('btn-rec-map').textContent = S.REC_MAP;
  document.getElementById('btn-view-data').textContent = S.REC_VIEW_DATA;
  document.getElementById('btn-end').textContent = S.REC_END;
  document.getElementById('data-title').textContent = S.DATA_TITLE;
  document.getElementById('data-groups-h').textContent = S.DATA_GROUPS;
  document.getElementById('data-trees-h').textContent = S.DATA_TREES;
  document.getElementById('btn-data-map').textContent = S.REC_MAP;
  document.getElementById('btn-data-close').textContent = S.DATA_CLOSE;
  document.getElementById('map-title').textContent = S.MAP_TITLE;
  document.getElementById('btn-map-back').textContent = S.MAP_BACK;
  document.getElementById('btn-map-center').textContent = S.MAP_CENTER;
  document.getElementById('map').setAttribute('aria-label', S.MAP_TITLE);
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

  // terreni.geojson powers GPS-driven parcel detection. A failure here
  // is non-fatal: recording still works, the parcel pulldown just
  // doesn't auto-track.
  try {
    State.terreni = await fetchTerreni();
  } catch (e) {
    State.terreni = null;
    showToast('Errore caricamento terreni.geojson: ' + e.message);
  }

  try {
    State.db = await Store.openDb();
  } catch (e) {
    showToast('Errore apertura database: ' + e.message);
    return;
  }

  populateOperator();
  populateComprese();
  wirePreSession();
  wireRecording();
  wireMap();
  wireUpload();
  wireDone();

  // Re-acquire the screen wake lock when the page returns to the
  // foreground (the Wake Lock API auto-releases on visibility loss).
  setupWakeLockVisibility();

  // Resume-on-open: check for any sessions awaiting follow-up
  // (STATUS_OPEN or STATUS_PENDING_UPLOAD) before showing pre-session.
  const resumable = await Store.listResumableSessions(State.db);
  if (resumable && resumable.length > 0) {
    showResumeModal(resumable);
  } else {
    showScreen('screen-pre');
  }

  // Request persistent storage (R2). Doesn't require a user gesture.
  requestPersist();

  // Check whether GPS permission was previously denied; banner if so.
  // The actual prompt is triggered from the Inizia click handler.
  checkGpsPermission();
}

async function fetchReference() {
  const r = await fetch('reference.json', { cache: 'reload' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return await r.json();
}

async function fetchTerreni() {
  const r = await fetch('terreni.geojson', { cache: 'reload' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const gj = await r.json();
  if (!gj || !Array.isArray(gj.features)) throw new Error('not a FeatureCollection');
  return gj.features;
}

async function requestPersist() {
  if (!navigator.storage || !navigator.storage.persist) return;
  const granted = await navigator.storage.persist();
  if (!granted) showBanner(S.STORAGE_WARNING);
}

// Check geolocation permission state at boot; surface a banner if it
// was previously denied. The actual prompt is fired from the pre-session
// submit handler (see promptGps()) so the operator deals with it during
// setup rather than mid-mark.
async function checkGpsPermission() {
  if (!navigator.geolocation) return;
  if (!navigator.permissions || !navigator.permissions.query) return;
  try {
    const status = await navigator.permissions.query({ name: 'geolocation' });
    if (status.state === 'denied') showBanner(S.GPS_PERMISSION_BANNER);
  } catch (_) { /* unsupported -> no banner */ }
}

// Fires a one-shot geolocation request from the pre-session submit
// handler. Two effects, depending on the permission state:
//  - 'granted': retrieves a fix, warming the GPS chip so the real watcher
//    in the recording screen lands its first fix faster;
//  - 'prompt':  surfaces the OS permission dialog;
//  - 'denied':  immediately errors and surfaces the banner.
function promptGps() {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    () => {},
    (err) => {
      if (err && err.code === err.PERMISSION_DENIED) {
        showBanner(S.GPS_PERMISSION_BANNER);
      }
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
  );
}

// ---------------------------------------------------------------------------
// Pre-session
// ---------------------------------------------------------------------------

function populateOperator() {
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
}

function appendOption(sel, value, label, selected) {
  const o = document.createElement('option');
  o.value = value;
  o.textContent = label;
  if (selected) o.selected = true;
  sel.appendChild(o);
}

function currentMode() {
  return State.mode || IpsoModes.defaultMode();
}

function setMode(modeId) {
  State.mode = IpsoModes.get(modeId);
  const title = document.getElementById('pre-title');
  if (title) title.textContent = modeString('preTitleKey', S.PRE_NEW_SESSION);
}

function modeString(field, fallback) {
  const key = currentMode()[field];
  if (!key || !Object.prototype.hasOwnProperty.call(S, key)) return fallback;
  return S[key];
}

function wirePreSession() {
  document.getElementById('pre-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const operator = document.getElementById('in-operatore').value.trim();
    const data = document.getElementById('in-data').value;
    const compresa = document.getElementById('in-compresa').value;
    const catastrofata = document.getElementById('in-catastrofata').checked;
    if (!operator || !data || !compresa) return;

    // Fire the GPS permission prompt now (during setup, not mid-mark).
    // Runs in parallel with the rest of submit — we don't await it
    // because the operator can still record without GPS if they deny.
    promptGps();

    localStorage.setItem('ipso.operatore', operator);

    try {
      const sess = await Store.startSession(State.db, {
        mode: currentMode().id,
        reference_version: State.reference.reference_version || '',
        work_package_id: '',
        data, compresa, operatore: operator, catastrofata,
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
  const inNumber = document.getElementById('in-numero');
  State.numpad = createNumpad({
    container: document.getElementById('numpad'),
    inputs: { d: inD, h: inH, numero: inNumber },
    maxLen: { d: 3, h: 2, numero: 4 },
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

  // Particella select: sticky-override transitions. The sentinel option
  // (value AUTO_SENTINEL) toggles auto mode; any other value enters
  // manual mode. Focus restores the "(automatica)" label on the open
  // picker; blur restores the closed-state display via refresh.
  const partSel = document.getElementById('in-particella-rec');
  partSel.addEventListener('focus', () => {
    const sentinel = partSel.querySelector('option[value="' + AUTO_SENTINEL + '"]');
    if (sentinel) sentinel.textContent = S.REC_PARTICELLA_AUTO;
  });
  partSel.addEventListener('blur', refreshParticellaSelect);
  partSel.addEventListener('change', () => {
    if (!State.override) return;
    if (partSel.value === AUTO_SENTINEL) State.override.setAuto();
    else State.override.setManual(partSel.value);
    refreshParticellaSelect();
  });

  document.getElementById('btn-save').addEventListener('click', onSave);
  document.getElementById('btn-view-data').addEventListener('click', enterDataScreen);
  document.getElementById('btn-data-close').addEventListener('click', exitDataScreen);
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

function wireMap() {
  document.getElementById('btn-rec-map').addEventListener('click', () => {
    enterMapScreen('screen-rec');
  });
  document.getElementById('btn-data-map').addEventListener('click', () => {
    enterMapScreen('screen-data');
  });
  document.getElementById('btn-map-back').addEventListener('click', exitMapScreen);
  document.getElementById('btn-map-center').addEventListener('click', centerMapOnContext);
}

function enterRecording() {
  populateSpecie();
  populateGruppo();
  // Sticky within a session. On a fresh start lastTreeRow is null and the
  // dropdown starts blank; on resume after force-quit we recover the last
  // tree's gruppo so the operator picks up where they left off.
  const lastGruppo = (State.lastTreeRow && State.lastTreeRow.gruppo) || '';
  document.getElementById('in-gruppo').value = lastGruppo;
  const where = S.where(State.session);
  State.lastFix = null;
  document.getElementById('rec-where').textContent = where;
  document.getElementById('sub-status').textContent = where;
  setupLocator();
  resetEntryFields();
  refreshPill();
  startGps();
  acquireWakeLock();
  showScreen('screen-rec');
}

// Auto-mode sentinel value for the recording-screen particella select.
// Real particella names are short alphanumeric strings (e.g. "7a"); the
// "__auto__" form is unambiguous as a non-particella token.
const AUTO_SENTINEL = '__auto__';

// Build the per-session parcel locator and sticky-override controllers:
// filter terreni to the session's compresa, bbox-index, populate the
// particella select, and subscribe a callback that keeps both
// #rec-where and the select in sync with each GPS commit.
function setupLocator() {
  State.locator = null;
  State.override = createOverride();
  if (!State.terreni || !State.session) {
    populateParticellaOptions(null);
    refreshParticellaSelect();
    return;
  }
  const compresa = State.session.compresa;
  const filtered = State.terreni.filter(
    (f) => f.properties && f.properties.layer === compresa
  );
  populateParticellaOptions(filtered);
  if (filtered.length === 0) {
    refreshParticellaSelect();
    return;
  }
  buildBboxIndex(filtered);
  State.locator = createLocator(filtered);
  State.locator.subscribe(onLocatorCommit);
  refreshParticellaSelect();
}

function onLocatorCommit(feature) {
  updateRecWhere(feature);
  refreshParticellaSelect();
  refreshMapParcels();
  updateMapHeader();
}

function updateRecWhere(feature) {
  document.getElementById('rec-where').textContent =
    feature ? formatParcelText(feature) : S.REC_OUT_OF_BOUNDS;
}

function formatParcelText(feature) {
  const label = parcelLabel(feature);
  if (!label) return '';
  return [label.title, label.type].filter((v) => v).join(' · ');
}

// Extract the bare particella portion of `feature.properties.name`
// (which has the form "Compresa-Particella" by terreni.geojson
// convention). Returns '' for malformed features.
function particellaName(feature) {
  const name = (feature && feature.properties && feature.properties.name) || '';
  const dash = name.indexOf('-');
  return dash >= 0 ? name.slice(dash + 1) : name;
}

function currentAutoName() {
  if (!State.locator) return '';
  return particellaName(State.locator.getCommitted());
}

function populateParticellaOptions(features) {
  const sel = document.getElementById('in-particella-rec');
  sel.replaceChildren();
  const sentinel = document.createElement('option');
  sentinel.value = AUTO_SENTINEL;
  sentinel.textContent = S.REC_PARTICELLA_AUTO;
  sel.appendChild(sentinel);
  if (!features) return;
  // Use reference.json order for the option list — it's the curated
  // ordering the rest of the UI also uses. Only include parcels that
  // exist as polygons (so manual picks can be cross-checked against
  // GPS).
  const known = new Set(features.map(particellaName).filter((n) => n));
  for (const p of State.reference.parcels) {
    if (p.compresa !== State.session.compresa) continue;
    if (!known.has(p.particella)) continue;
    appendOption(sel, p.particella, p.particella);
  }
}

function refreshParticellaSelect() {
  const sel = document.getElementById('in-particella-rec');
  const ov = State.override;
  if (!ov) return;
  const autoName = currentAutoName();
  const sentinel = sel.querySelector('option[value="' + AUTO_SENTINEL + '"]');
  if (ov.getMode() === 'manual') {
    sel.value = ov.getManual();
    if (sentinel) sentinel.textContent = S.REC_PARTICELLA_AUTO;
  } else {
    sel.value = AUTO_SENTINEL;
    if (sentinel) {
      sentinel.textContent = autoName || S.REC_PARTICELLA_PLACEHOLDER;
    }
  }
  sel.classList.toggle('error', ov.isMismatch(autoName));
}

function populateGruppo() {
  const sel = document.getElementById('in-gruppo');
  if (sel.options.length > 0) return;   // populate once per page life
  appendOption(sel, '', '—', true);
  for (let i = 0; i < 26; i++) {
    const letter = String.fromCharCode(65 + i);
    appendOption(sel, letter, letter);
  }
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
  // Prefill the number field asynchronously from the tree list. The numpad
  // starts empty (above); we only fill if the operator hasn't started
  // typing yet.
  prefillNumber();
  updateSaveEnabled();
}

async function prefillNumber() {
  if (!State.session) return;
  try {
    const trees = await Store.listTrees(State.db, State.session.id);
    const next = await computeNextNumberDefault(trees);
    if (next == null) return;
    if (State.numpad.value('numero') === '') {
      State.numpad.setValue('numero', '' + next);
    }
  } catch (_) { /* leave blank on error */ }
}

// In-session max+1 takes precedence; on a fresh session (no numbered trees
// yet) we fall back to the per-operator counter persisted across sessions.
async function computeNextNumberDefault(trees) {
  const inSession = session.nextNumberDefault(trees);
  if (inSession != null) return inSession;
  if (!State.session) return null;
  return Store.getNextNumberForOperator(State.db, State.session.operatore);
}

function refreshPill() {
  const pill = document.getElementById('pill-text');
  const btn = document.getElementById('pill-action');
  if (!State.lastTreeRow) {
    pill.textContent = S.REC_NO_LAST;
    btn.hidden = true;
  } else {
    pill.textContent = S.pill(State.lastTreeRow);
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
      // Keep the line a fixed shape so it doesn't jitter the rec-header on
      // narrow screens. The dot already encodes accuracy tier; the
      // GPS-stale fallback below kicks in when age > 10 s.
      text.textContent =
        st.fix.lat.toFixed(5) + ' ' + st.fix.lon.toFixed(5) +
        ' ±' + Math.round(st.fix.acc) + ' m';
      State.lastFix = {
        lat: st.fix.lat,
        lon: st.fix.lon,
        acc: st.fix.acc,
        t: st.fix.t,
      };
      if (State.locator) State.locator.onFix(st.fix);
      updateMapPosition();
      updateMapHeader();
    } else if (st.error === 'denied') {
      text.textContent = S.GPS_DENIED;
      updateMapHeader();
    } else {
      text.textContent = S.REC_GPS_WAITING;
      updateMapHeader();
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
  const nStr = State.numpad.value('numero');
  const n = nStr === '' ? null : parseInt(nStr, 10);
  const gruppo = document.getElementById('in-gruppo').value || '';
  const particella = State.override
    ? State.override.resolve(currentAutoName())
    : '';
  return {
    specie: State.specie || '',
    d_cm: Number.isFinite(d) ? d : null,
    h_m: Number.isFinite(h) ? h : null,
    h_measured: State.hMeasured ? 1 : 0,
    hypso_param_set_id: State.hMeasured ? null : currentHypsoParamSetId(),
    numero: Number.isInteger(n) ? n : null,
    gruppo,
    particella,
  };
}


function currentHypsoParamSetId() {
  if (!State.session || !State.reference) return null;
  const eq = ipso.lookup(State.reference.ipsometrica, State.session.compresa, State.specie);
  return eq && Number.isInteger(eq.hypso_param_set_id)
    ? eq.hypso_param_set_id
    : null;
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

  // Small trees are not physically numbered in the field, so we force the
  // stored number blank regardless of what the operator typed. The counter
  // ignores blank-number trees, so the next visible tree continues the
  // sequence.
  if (rec.d_cm != null && rec.d_cm <= session.NUMBER_BLANK_D_THRESHOLD) {
    rec.numero = null;
  }

  // Snapshot GPS atomically into the record (R9).
  const gps = State.gps ? State.gps.snapshot() : null;
  const full = Object.assign({}, rec, {
    lat: gps ? gps.lat : null,
    lon: gps ? gps.lon : null,
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
    // Reset the number field to the new default — the operator just freed
    // up a slot and almost always wants to redo with the same number. D /
    // h / specie are intentionally left alone (they reflect the redo state).
    const trees = await Store.listTrees(State.db, State.session.id);
    const next = await computeNextNumberDefault(trees);
    State.numpad.setValue('numero', next == null ? '' : '' + next);
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
    const csvText = csv.formatFile(State.session, trees);
    const uploadPayload = upload.buildUploadPayload(
      State.session, trees, State.reference, csvText
    );
    await Store.setSessionStatus(State.db, State.session.id, Store.STATUS_PENDING_UPLOAD);
    State.session.status = Store.STATUS_PENDING_UPLOAD;
    // Always download the local CSV first — it is the trust anchor and
    // the operator must not lose data if the upload never succeeds.
    downloadFinal(State.session, trees);
    enterUploadScreen(State.session.id, uploadPayload, trees.length);
  } catch (e) {
    showToast('Errore esportazione: ' + e.message);
  }
}

function enterUploadScreen(sessionId, uploadPayload, treeCount) {
  // Reset any prior state.
  if (State.upload && State.upload.retryTimer) {
    clearTimeout(State.upload.retryTimer);
  }
  if (State.upload && State.upload.abortController) {
    try { State.upload.abortController.abort(); } catch (_) {}
  }
  // Recording is over — release the GPS watcher so it doesn't drain the
  // battery while the operator waits for the upload to settle.
  if (State.gps) { State.gps.stop(); State.gps = null; }
  State.upload = {
    sessionId,
    payload: uploadPayload,
    treeCount,
    attempt: 0,
    abortController: null,
    retryTimer: null,
  };
  document.getElementById('upload-detail').textContent = '';
  document.getElementById('upload-detail').classList.remove('error');
  showScreen('screen-upload');
  acquireWakeLock();
  scheduleUploadAttempt(0);
}

function scheduleUploadAttempt(waitMs) {
  if (!State.upload) return;
  if (waitMs > 0) {
    const secs = Math.ceil(waitMs / 1000);
    document.getElementById('upload-detail').textContent =
      S.UPLOAD_NEXT_RETRY_IN(secs);
    State.upload.retryTimer = setTimeout(runUploadAttempt, waitMs);
  } else {
    runUploadAttempt();
  }
}

async function runUploadAttempt() {
  if (!State.upload) return;
  State.upload.retryTimer = null;
  State.upload.attempt += 1;
  document.getElementById('upload-attempt').textContent =
    S.UPLOAD_ATTEMPT(State.upload.attempt);

  const ac = new AbortController();
  State.upload.abortController = ac;
  try {
    await upload.uploadSession({
      token: UPLOAD_TOKEN,
      sessionId: State.upload.sessionId,
      payload: State.upload.payload,
      signal: ac.signal,
    });
    await onUploadSuccess();
  } catch (err) {
    if (err && err.klass === 'aborted') return;  // bail handled elsewhere
    onUploadAttemptFailed(err);
  } finally {
    if (State.upload) State.upload.abortController = null;
  }
}

function onUploadAttemptFailed(err) {
  if (!State.upload) return;
  const klass = (err && err.klass) || 'soft:network';
  const isHard = klass.startsWith('hard:');
  const detailEl = document.getElementById('upload-detail');
  detailEl.classList.toggle('error', isHard);
  detailEl.textContent = uploadErrorMessage(klass);
  if (isHard) {
    // Stop retrying; operator must bail.
    document.getElementById('upload-attempt').textContent = '';
    return;
  }
  scheduleUploadAttempt(upload.backoffMs(State.upload.attempt));
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

async function onUploadSuccess() {
  if (!State.upload) return;
  // Capture + clear State.upload synchronously so a late bail-tap during the
  // following awaits can't reach the bail branch and clobber upload_status.
  const sessionId = State.upload.sessionId;
  const treeCount = State.upload.treeCount;
  State.upload = null;
  try {
    await Store.setSessionUploadStatus(
      State.db, sessionId, Store.UPLOAD_STATUS_UPLOADED
    );
    await Store.setSessionStatus(
      State.db, sessionId, Store.STATUS_EXPORTED
    );
  } catch (e) {
    showToast('Errore salvataggio stato upload: ' + e.message);
  }
  showToast(S.UPLOAD_SUCCESS_TOAST);
  endUploadScreen(treeCount, true);
}

async function onUploadBail() {
  if (!State.upload) return;
  if (State.upload.retryTimer) {
    clearTimeout(State.upload.retryTimer);
    State.upload.retryTimer = null;
  }
  if (State.upload.abortController) {
    try { State.upload.abortController.abort(); } catch (_) {}
  }
  // Same race-prevention move as onUploadSuccess.
  const sessionId = State.upload.sessionId;
  const treeCount = State.upload.treeCount;
  State.upload = null;
  try {
    await Store.setSessionUploadStatus(
      State.db, sessionId, Store.UPLOAD_STATUS_LOCAL_ONLY
    );
    await Store.setSessionStatus(
      State.db, sessionId, Store.STATUS_EXPORTED
    );
  } catch (e) {
    showToast('Errore salvataggio stato: ' + e.message);
  }
  showToast(S.UPLOAD_LOCAL_ONLY_TOAST);
  endUploadScreen(treeCount, false);
}

function endUploadScreen(treeCount, uploaded) {
  State.upload = null;
  // Repurpose the existing done screen, but tailor the body text.
  document.getElementById('done-title').textContent = S.DONE_TITLE;
  document.getElementById('done-body').textContent = uploaded
    ? S.UPLOAD_DONE_BODY(treeCount)
    : S.DONE_BODY(treeCount);
  releaseWakeLock();
  showScreen('screen-done');
}

function downloadFinal(sess, trees) {
  const text = csv.formatFile(sess, trees);
  const name = csv.filename(sess, new Date(), 'final');
  downloadText(text, name);
}

// ---------------------------------------------------------------------------
// Map screen
// ---------------------------------------------------------------------------

function enterMapScreen(returnScreen) {
  if (!State.session) return;
  if (typeof L === 'undefined' || typeof createOrientationMap === 'undefined') {
    showToast(S.MAP_UNAVAILABLE);
    return;
  }
  State.mapReturnScreen = returnScreen || 'screen-rec';
  showScreen('screen-map');
  ensureMap();
  renderMapParcels();
  updateMapPosition();
  updateMapHeader();
  setTimeout(() => {
    if (!State.map) return;
    State.map.invalidate();
    centerMapOnContext();
  }, 0);
}

function exitMapScreen() {
  showScreen(State.mapReturnScreen || 'screen-rec');
}

function ensureMap() {
  if (State.map) {
    State.map.ensure();
    return;
  }
  State.map = createOrientationMap({
    elementId: 'map',
    formatFeatureLabel: formatParcelText,
    featureName: particellaName,
    getActiveName: currentAutoName,
    getManualName() {
      return State.override && State.override.getMode() === 'manual'
        ? State.override.getManual()
        : '';
    },
    onFeatureClick(label) {
      document.getElementById('map-sub').textContent = label;
    },
  });
  State.map.ensure();
}

function currentMapFeatures() {
  if (!State.terreni || !State.session) return [];
  const compresa = State.session.compresa;
  return State.terreni.filter(
    (f) => f.properties && f.properties.layer === compresa
  );
}

function renderMapParcels() {
  if (!State.map) return;
  State.map.renderParcels(currentMapFeatures());
}

function refreshMapParcels() {
  if (State.currentScreen !== 'screen-map') return;
  renderMapParcels();
}

function updateMapPosition() {
  if (State.map) State.map.updatePosition(State.lastFix);
}

function centerMapOnContext() {
  if (!State.map) return;
  const ok = State.map.center({
    fix: State.lastFix,
    committedFeature: State.locator ? State.locator.getCommitted() : null,
  });
  if (!ok) showToast(S.MAP_NO_PARCELS);
}

function updateMapHeader() {
  const title = document.getElementById('map-title');
  const sub = document.getElementById('map-sub');
  if (!title || !sub) return;
  title.textContent = State.session ? S.where(State.session) : S.MAP_TITLE;
  const committed = State.locator ? State.locator.getCommitted() : null;
  if (committed) {
    sub.textContent = formatParcelText(committed);
  } else if (State.lastFix) {
    sub.textContent =
      State.lastFix.lat.toFixed(5) + ' ' + State.lastFix.lon.toFixed(5) +
      ' ±' + Math.round(State.lastFix.acc) + ' m';
  } else {
    sub.textContent = S.MAP_WAITING;
  }
}

// ---------------------------------------------------------------------------
// Visualizza dati raccolti screen
// ---------------------------------------------------------------------------

async function enterDataScreen() {
  if (!State.session) return;
  let trees;
  try {
    trees = await Store.listTrees(State.db, State.session.id);
  } catch (e) {
    showToast('Errore caricamento dati: ' + e.message);
    return;
  }
  trees.sort((a, b) => a.seq - b.seq);
  renderGroupsTable(trees);
  renderTreesTable(trees);
  showScreen('screen-data');
}

function exitDataScreen() {
  showScreen('screen-rec');
}

function renderGroupsTable(trees) {
  const tbl = document.getElementById('data-groups-table');
  tbl.replaceChildren();
  const counts = new Map();
  for (const t of trees) {
    const g = (t && t.gruppo) || '';
    if (g) counts.set(g, (counts.get(g) || 0) + 1);
  }
  const keys = Array.from(counts.keys()).sort();
  if (!keys.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 2;
    td.className = 'empty';
    td.textContent = S.DATA_NO_GROUPS;
    tr.appendChild(td);
    tbl.appendChild(tr);
    return;
  }
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  const cols = [
    { label: S.DATA_COL_GRUPPO, num: false },
    { label: S.DATA_COUNT, num: true },
  ];
  for (const c of cols) {
    const th = document.createElement('th');
    th.textContent = c.label;
    if (c.num) th.className = 'num';
    trh.appendChild(th);
  }
  thead.appendChild(trh);
  tbl.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const k of keys) {
    const tr = document.createElement('tr');
    const tdG = document.createElement('td');
    tdG.textContent = k;
    const tdC = document.createElement('td');
    tdC.className = 'num';
    tdC.textContent = '' + counts.get(k);
    tr.appendChild(tdG);
    tr.appendChild(tdC);
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
}

function renderTreesTable(trees) {
  const tbl = document.getElementById('data-trees-table');
  tbl.replaceChildren();
  if (!trees.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 6;
    td.className = 'empty';
    td.textContent = S.DATA_EMPTY;
    tr.appendChild(td);
    tbl.appendChild(tr);
    return;
  }
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  const headers = [
    { label: S.DATA_COL_NUMBER, num: true },
    { label: S.DATA_COL_SPECIE, num: false },
    { label: S.DATA_COL_PARTICELLA, num: false },
    { label: S.DATA_COL_GRUPPO, num: false },
    { label: S.DATA_COL_D, num: true },
    { label: S.DATA_COL_H, num: true },
  ];
  for (const h of headers) {
    const th = document.createElement('th');
    th.textContent = h.label;
    if (h.num) th.className = 'num';
    trh.appendChild(th);
  }
  thead.appendChild(trh);
  tbl.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (const t of trees) {
    const tr = document.createElement('tr');
    const cells = [
      { v: t.numero == null ? '' : '' + t.numero, num: true },
      { v: t.specie || '', num: false },
      { v: t.particella || '', num: false },
      { v: t.gruppo || '', num: false },
      { v: t.d_cm == null ? '' : '' + t.d_cm, num: true },
      { v: t.h_m == null ? '' : '' + t.h_m, num: true },
    ];
    for (const c of cells) {
      const td = document.createElement('td');
      td.textContent = c.v;
      if (c.num) td.className = 'num';
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
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
// Upload screen
// ---------------------------------------------------------------------------

function wireUpload() {
  document.getElementById('upload-title').textContent = S.UPLOAD_TITLE;
  document.getElementById('btn-upload-bail').textContent = S.UPLOAD_BAIL;
  document.getElementById('btn-upload-bail').addEventListener('click', onUploadBail);
}

// ---------------------------------------------------------------------------
// Done screen
// ---------------------------------------------------------------------------

function wireDone() {
  document.getElementById('btn-new-session').addEventListener('click', () => {
    State.session = null;
    setMode(IpsoModes.MARTELLATE);
    State.lastTreeRow = null;
    State.locator = null;
    State.override = null;
    if (State.gps) { State.gps.stop(); State.gps = null; }
    document.getElementById('sub-status').textContent = '';
    document.getElementById('pre-form').reset();
    populateOperator();
    populateComprese();
    showScreen('screen-pre');
  });
}

// ---------------------------------------------------------------------------
// Resume modal
// ---------------------------------------------------------------------------

function showResumeModal(sessions) {
  // Mixed list: STATUS_OPEN sessions need resume/export/discard; new
  // STATUS_PENDING_UPLOAD sessions need carica-ora / mantieni-solo-locale.
  const hasUpload = sessions.some(
    (s) => s.status === Store.STATUS_PENDING_UPLOAD
  );
  const hasOpen = sessions.some(
    (s) => s.status === Store.STATUS_OPEN
  );
  document.getElementById('resume-title').textContent = hasUpload && !hasOpen
    ? S.UPLOAD_RESUME_TITLE
    : S.RESUME_TITLE;
  // Body line is generic enough for either case; leave RESUME_BODY in place.

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
    if (s.status === Store.STATUS_PENDING_UPLOAD) {
      const carica = mkBtn(S.UPLOAD_RESUME_DO_NOW, 'btn-primary', async () => {
        hideModal('modal-resume');
        const trees = await Store.listTrees(State.db, s.id);
        trees.sort((a, b) => a.seq - b.seq);
        const csvText = csv.formatFile(s, trees);
        const uploadPayload = upload.buildUploadPayload(
          s, trees, State.reference, csvText
        );
        // Re-download the local CSV on every entry to screen-upload —
        // the browser auto-renames duplicates so this can never lose
        // the original copy. See spec.
        downloadFinal(s, trees);
        State.session = s;
        setMode(s.mode);
        enterUploadScreen(s.id, uploadPayload, trees.length);
      });
      const local = mkBtn(S.UPLOAD_RESUME_KEEP_LOCAL, 'btn-secondary', async () => {
        await Store.setSessionUploadStatus(State.db, s.id, Store.UPLOAD_STATUS_LOCAL_ONLY);
        await Store.setSessionStatus(State.db, s.id, Store.STATUS_EXPORTED);
        li.remove();
        if (!list.children.length) {
          hideModal('modal-resume');
          showScreen('screen-pre');
        }
      });
      actions.appendChild(carica);
      actions.appendChild(local);
    } else {
      const resume = mkBtn(S.RESUME_RESUME, 'btn-primary', async () => {
        State.session = s;
        setMode(s.mode);
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
    }
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
  State.currentScreen = id;
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
