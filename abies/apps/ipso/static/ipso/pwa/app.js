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
  bearerToken: '',    // non-public bearer from authenticated Abies bootstrap
  db: null,           // open IDBDatabase
  session: null,      // current session row (null until pre_session submits)
  specie: '',
  hMeasured: false,   // flips true once the user touches the h field
  inAutoFill: false,  // true while recomputeAutoH() is writing to h
  saveLockUntil: 0,   // ms timestamp; Salva is disabled while now < this
  gps: null,          // GPS controller
  locator: null,      // parcel-locator instance (one per recording session)
  numpad: null,       // numpad controller
  autoNumberValue: null, // current number value filled by prefillNumber()
  inAutoNumberFill: false,
  lastTreeRow: null,  // most-recent tree in the current session
  wakeLock: null,     // active WakeLockSentinel during recording (or null)
  currentScreen: null, // id of the visible screen
  lastFix: null,       // most recent fresh GPS fix, { lat, lon, acc, t }
  sampleAreaId: null,  // selected/inferred sample-area id in sample mode
  map: null,           // lazy orientation map controller
  mapReturnScreen: 'screen-rec',
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  if (!window.AbiesGeoReady) throw new Error(S.ERROR_GEO_UNAVAILABLE);
  await window.AbiesGeoReady;

  setMode(IpsoModes.MARTELLATE);
  document.getElementById('footer-version').textContent =
    'v' + APP_VERSION;
  document.getElementById('mode-title').textContent = S.MODE_TITLE;
  document.getElementById('lbl-operatore').textContent = S.PRE_OPERATOR;
  document.getElementById('lbl-data').textContent = S.PRE_DATA;
  document.getElementById('lbl-compresa').textContent = S.PRE_COMPRESA;
  document.getElementById('lbl-sample-survey').textContent = S.PRE_SURVEY;
  document.getElementById('lbl-catastrofata').textContent = S.PRE_CATASTROFATA;
  document.getElementById('btn-start').textContent = S.PRE_START;
  document.getElementById('btn-pre-mode').textContent = S.MODE_BACK;
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
    State.bearerToken = await fetchBootstrap();
  } catch (e) {
    showToast(S.TOAST_REFERENCE_LOAD_ERROR(e.message));
    return;
  }

  try {
    State.reference = await fetchReference();
  } catch (e) {
    showToast(S.TOAST_REFERENCE_LOAD_ERROR(e.message));
    return;
  }

  // terreni.geojson powers GPS-driven parcel detection. A failure here
  // is non-fatal: recording still works, the parcel pulldown just
  // doesn't auto-track.
  try {
    State.terreni = await fetchTerreni();
  } catch (e) {
    State.terreni = null;
    showToast(S.TOAST_TERRENI_LOAD_ERROR(e.message));
  }

  try {
    State.db = await Store.openDb();
  } catch (e) {
    showToast(S.TOAST_DB_OPEN_ERROR(e.message));
    return;
  }

  populateOperator();
  populateComprese();
  wireModeSelection();
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
    showModeScreen();
  }

  // Request persistent storage (R2). Doesn't require a user gesture.
  requestPersist();

  // Check whether GPS permission was previously denied; banner if so.
  // The actual prompt is triggered from the Inizia click handler.
  checkGpsPermission();
}

async function fetchBootstrap() {
  const bootstrapToken = bootstrapTokenFromHash();
  if (bootstrapToken) return await exchangeBootstrapToken(bootstrapToken);
  const stored = storedBearerToken();
  if (stored) return stored;
  throw new Error(S.ERROR_TOKEN_MISSING);
}

async function exchangeBootstrapToken(bootstrapToken) {
  const r = await fetch('/api/ipso/bootstrap/', {
    method: 'POST',
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {
      Accept: 'application/json',
      Authorization: 'Bearer ' + bootstrapToken,
    },
  });
  if (!r.ok) throw new Error(S.ERROR_HTTP_STATUS(r.status));
  const payload = await r.json();
  const token = payload && payload[IPSO_BOOTSTRAP_BEARER_TOKEN];
  if (!payload || payload.ok !== true || !token) throw new Error(S.ERROR_BOOTSTRAP_INVALID);
  storeBearerToken(token);
  clearBootstrapHash();
  return token;
}

function storedBearerToken() {
  try { return localStorage.getItem(IPSO_BEARER_STORAGE_KEY) || ''; }
  catch (_) { return ''; }
}

function storeBearerToken(token) {
  try { localStorage.setItem(IPSO_BEARER_STORAGE_KEY, token); } catch (_) {}
}

function bootstrapTokenFromHash() {
  const params = new URLSearchParams((window.location.hash || '').replace(/^#/, ''));
  return params.get(IPSO_BOOTSTRAP_HASH_PARAM) || '';
}

function clearBootstrapHash() {
  if (!window.location.hash) return;
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ''));
  if (!params.has(IPSO_BOOTSTRAP_HASH_PARAM)) return;
  params.delete(IPSO_BOOTSTRAP_HASH_PARAM);
  const nextHash = params.toString();
  const nextUrl = window.location.pathname + window.location.search +
    (nextHash ? '#' + nextHash : '');
  window.history.replaceState(null, '', nextUrl);
}

function bearerHeaders() {
  if (!State.bearerToken) throw new Error(S.ERROR_TOKEN_MISSING);
  return { Authorization: 'Bearer ' + State.bearerToken };
}

async function fetchReference() {
  const r = await fetch('reference.json', {
    cache: 'reload',
    headers: bearerHeaders(),
  });
  if (!r.ok) throw new Error(S.ERROR_HTTP_STATUS(r.status));
  return await r.json();
}

async function fetchTerreni() {
  const r = await fetch('terreni.geojson', {
    cache: 'reload',
    headers: bearerHeaders(),
  });
  if (!r.ok) throw new Error(S.ERROR_HTTP_STATUS(r.status));
  const gj = await r.json();
  if (!gj || !Array.isArray(gj.features)) throw new Error(S.ERROR_GEOJSON_INVALID);
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
  const comprese = [...new Set(State.reference[IPSO_REF_PARCELS].map((p) => p.compresa))].sort();
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

function isSamplesMode() {
  const mode = State.session ? State.session.mode : currentMode().id;
  return mode === IpsoModes.SAMPLES;
}

function selectedSampleSurveyId() {
  const raw = document.getElementById('in-sample-survey').value;
  const id = parseInt(raw, 10);
  return Number.isInteger(id) ? id : null;
}

function sampleSurveyIdFromWorkPackage(workPackageId) {
  const raw = String(workPackageId || '');
  if (!raw.startsWith(IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX)) return null;
  const id = parseInt(raw.slice(IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX.length), 10);
  return Number.isInteger(id) ? id : null;
}

function currentSampleSurveyId() {
  if (State.session) return sampleSurveyIdFromWorkPackage(State.session.work_package_id);
  return selectedSampleSurveyId();
}

function samplingRows(kind) {
  return State.reference && State.reference[IPSO_REF_SAMPLING] &&
    Array.isArray(State.reference[IPSO_REF_SAMPLING][kind])
    ? State.reference[IPSO_REF_SAMPLING][kind]
    : [];
}

function sampleAreasForSurvey(surveyId, compresa) {
  const survey = samplingRows(IPSO_REF_SURVEYS).find((s) =>
    s && s[FIELD_SURVEY_ID] === surveyId
  );
  if (!survey) return [];
  const maxNumbers = survey[IPSO_REF_SAMPLE_AREA_MAX_NUMBERS] || {};
  return samplingRows(IPSO_REF_SAMPLE_AREAS)
    .filter((area) =>
      area && area[FIELD_SAMPLE_GRID_ID] === survey[FIELD_SAMPLE_GRID_ID] &&
      (!compresa || area.compresa === compresa)
    )
    .map((area) => {
      const maxNumber = maxNumbers[String(area[FIELD_SAMPLE_AREA_ID])];
      return Object.assign({}, area, {
        [FIELD_MAX_TREE_NUMBER]: Number.isInteger(maxNumber) ? maxNumber : null,
      });
    });
}

function sampleSurveys() {
  return samplingRows(IPSO_REF_SURVEYS).filter((survey) => survey);
}

function populateSampleSurveyOptions() {
  const sel = document.getElementById('in-sample-survey');
  if (!sel) return;
  const keep = sel.value;
  sel.replaceChildren();
  appendOption(sel, '', S.PRE_PICK_SURVEY, true);
  for (const survey of sampleSurveys()) {
    appendOption(sel, '' + survey[FIELD_SURVEY_ID], sampleSurveyLabel(survey));
  }
  if (keep && Array.from(sel.options).some((opt) => opt.value === keep)) {
    sel.value = keep;
  }
}

function sampleSurveyLabel(survey) {
  return [survey.name, survey.sample_grid_name].filter((v) => v).join(' - ');
}

function sampleAreasForCurrentSurvey() {
  if (!State.session) return [];
  return sampleAreasForSurvey(currentSampleSurveyId(), State.session.compresa);
}

function sampleAreaById(id) {
  return sampleAreasForCurrentSurvey().find((area) =>
    area && area[FIELD_SAMPLE_AREA_ID] === id
  ) || null;
}

function sampleAreaLabel(area) {
  if (!area) return '';
  return [area.particella, area.number]
    .filter((v) => v !== null && v !== undefined && v !== '')
    .join('/');
}

function currentAutoSampleArea() {
  if (!State.lastFix) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const area of sampleAreasForCurrentSurvey()) {
    if (!area || area.lat == null || area.lon == null) continue;
    const distance = upload.distanceMeters(
      State.lastFix.lat, State.lastFix.lon, area.lat, area.lon
    );
    const radius = Number.isFinite(area[FIELD_R_M]) ? area[FIELD_R_M] : upload.DEFAULT_SAMPLE_RADIUS_M;
    if (distance <= radius && distance < bestDistance) {
      best = area;
      bestDistance = distance;
    }
  }
  return best;
}

function currentSampleArea() {
  if (!isSamplesMode() || !State.override) return null;
  if (State.override.getMode() === 'manual') {
    const id = parseInt(State.override.getManual(), 10);
    return Number.isInteger(id) ? sampleAreaById(id) : null;
  }
  return currentAutoSampleArea();
}

function currentMode() {
  return State.mode || IpsoModes.defaultMode();
}

function setMode(modeId) {
  State.mode = IpsoModes.get(modeId);
  const title = document.getElementById('pre-title');
  if (title) title.textContent = modeString('preTitleKey', S.PRE_NEW_SESSION);
  applyModeUi();
}

function applyModeUi() {
  const catastrofata = document.getElementById('in-catastrofata');
  const catastrofataField = catastrofata ? catastrofata.closest('label') : null;
  const surveyField = document.getElementById('field-sample-survey');
  const surveySelect = document.getElementById('in-sample-survey');
  const martellate = currentMode().id === IpsoModes.MARTELLATE;
  const samples = currentMode().id === IpsoModes.SAMPLES;
  if (catastrofataField) catastrofataField.hidden = !martellate;
  if (catastrofata && !martellate) catastrofata.checked = false;
  if (surveyField) surveyField.hidden = !samples;
  if (surveySelect && !samples) surveySelect.value = '';
  if (samples) populateSampleSurveyOptions();
}

function modeString(field, fallback) {
  return modeStringFor(currentMode(), field, fallback);
}

function modeStringFor(mode, field, fallback) {
  const key = mode && mode[field];
  if (!key || !Object.prototype.hasOwnProperty.call(S, key)) return fallback;
  return S[key];
}

function showModeScreen() {
  State.session = null;
  State.lastTreeRow = null;
  State.locator = null;
  State.override = null;
  State.sampleAreaId = null;
  setMode(IpsoModes.MARTELLATE);
  document.getElementById('sub-status').textContent = '';
  showScreen('screen-mode');
}

function enterPreSession(modeId) {
  const mode = IpsoModes.get(modeId);
  if (!mode.enabled) return;
  setMode(mode.id);
  populateSampleSurveyOptions();
  showScreen('screen-pre');
}

function wireModeSelection() {
  for (const mode of IpsoModes.all()) {
    const button = document.getElementById(mode.buttonId);
    if (!button) continue;
    button.textContent = modeStringFor(mode, 'labelKey', mode.id);
    button.disabled = !mode.enabled;
    button.addEventListener('click', () => enterPreSession(mode.id));
  }
}

function wirePreSession() {
  document.getElementById('btn-pre-mode').addEventListener('click', showModeScreen);
  document.getElementById('in-compresa').addEventListener('change', populateSampleSurveyOptions);
  document.getElementById('pre-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const operator = document.getElementById('in-operatore').value.trim();
    const data = document.getElementById('in-data').value;
    const compresa = document.getElementById('in-compresa').value;
    const catastrofata = document.getElementById('in-catastrofata').checked;
    const modeId = currentMode().id;
    const sampleSurveyId = selectedSampleSurveyId();
    if (!operator || !data || !compresa) return;
    if (modeId === IpsoModes.SAMPLES && sampleSurveyId == null) return;

    // Fire the GPS permission prompt now (during setup, not mid-mark).
    // Runs in parallel with the rest of submit — we don't await it
    // because the operator can still record without GPS if they deny.
    promptGps();

    localStorage.setItem('ipso.operatore', operator);

    try {
      const sess = await Store.startSession(State.db, {
        mode: modeId,
        reference_version: State.reference.reference_version || '',
        work_package_id: modeId === IpsoModes.SAMPLES
          ? IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX + sampleSurveyId
          : '',
        data, compresa, operatore: operator, catastrofata,
      });
      State.session = sess;
      State.lastTreeRow = null;
      State.sampleAreaId = null;
      enterRecording();
    } catch (err) {
      showToast(S.TOAST_SESSION_START_ERROR(err.message));
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
      if (field === 'numero' && !State.inAutoNumberFill) State.autoNumberValue = null;
      if (field === 'h' && !State.inAutoFill) State.hMeasured = true;
      if (field === 'd' && shouldAutoHeight() && !State.hMeasured) recomputeAutoH();
      updateSaveEnabled();
    },
  });
  State.numpad.mount();

  document.getElementById('in-specie').addEventListener('change', (e) => {
    State.specie = e.target.value;
    if (shouldAutoHeight() && !State.hMeasured) recomputeAutoH();
    updateSaveEnabled();
  });

  // Particella select: sticky-override transitions. The sentinel option
  // (value AUTO_SENTINEL) toggles auto mode; any other value enters
  // manual mode. Focus restores the "(automatica)" label on the open
  // picker; blur restores the closed-state display via refresh.
  const partSel = document.getElementById('in-particella-rec');
  partSel.addEventListener('focus', () => {
    const sentinel = partSel.querySelector('option[value="' + AUTO_SENTINEL + '"]');
    if (sentinel) {
      sentinel.textContent = isSamplesMode()
        ? S.REC_SAMPLE_AREA_AUTO
        : S.REC_PARTICELLA_AUTO;
    }
  });
  partSel.addEventListener('blur', refreshParticellaSelect);
  partSel.addEventListener('change', () => {
    if (!State.override) return;
    if (partSel.value === AUTO_SENTINEL) State.override.setAuto();
    else State.override.setManual(partSel.value);
    if (isSamplesMode()) State.numpad.setValue('numero', '');
    refreshParticellaSelect();
    if (isSamplesMode()) prefillNumber();
    updateSaveEnabled();
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
  document.getElementById('lbl-particella-rec').textContent = isSamplesMode()
    ? S.REC_SAMPLE_AREA
    : S.PRE_PARTICELLA;
  State.sampleAreaId = null;
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
  if (isSamplesMode()) {
    populateParticellaOptions(null);
    refreshParticellaSelect();
    return;
  }
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
  if (isSamplesMode()) {
    const area = currentAutoSampleArea();
    return area ? area.particella : '';
  }
  if (!State.locator) return '';
  return particellaName(State.locator.getCommitted());
}

function populateParticellaOptions(features) {
  const sel = document.getElementById('in-particella-rec');
  sel.replaceChildren();
  const sentinel = document.createElement('option');
  sentinel.value = AUTO_SENTINEL;
  sentinel.textContent = isSamplesMode()
    ? S.REC_SAMPLE_AREA_AUTO
    : S.REC_PARTICELLA_AUTO;
  sel.appendChild(sentinel);
  if (isSamplesMode()) {
    for (const area of sampleAreasForCurrentSurvey()) {
      appendOption(sel, '' + area[FIELD_SAMPLE_AREA_ID], sampleAreaLabel(area));
    }
    return;
  }
  if (!features) return;
  // Use reference.json order for the option list — it's the curated
  // ordering the rest of the UI also uses. Only include parcels that
  // exist as polygons (so manual picks can be cross-checked against
  // GPS).
  const known = new Set(features.map(particellaName).filter((n) => n));
  for (const p of State.reference[IPSO_REF_PARCELS]) {
    if (p.compresa !== State.session.compresa) continue;
    if (!known.has(p.particella)) continue;
    appendOption(sel, p.particella, p.particella);
  }
}

function refreshParticellaSelect() {
  if (isSamplesMode()) {
    refreshSampleAreaSelect();
    return;
  }
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

function refreshSampleAreaSelect() {
  const sel = document.getElementById('in-particella-rec');
  const ov = State.override;
  if (!sel || !ov) return;
  const autoArea = currentAutoSampleArea();
  const selectedArea = currentSampleArea();
  const sentinel = sel.querySelector('option[value="' + AUTO_SENTINEL + '"]');
  if (ov.getMode() === 'manual') {
    sel.value = ov.getManual();
    if (sentinel) sentinel.textContent = S.REC_SAMPLE_AREA_AUTO;
  } else {
    sel.value = AUTO_SENTINEL;
    if (sentinel) {
      sentinel.textContent = autoArea
        ? sampleAreaLabel(autoArea)
        : S.REC_SAMPLE_AREA_PLACEHOLDER;
    }
  }
  const nextId = selectedArea ? selectedArea[FIELD_SAMPLE_AREA_ID] : null;
  const autoId = autoArea ? autoArea[FIELD_SAMPLE_AREA_ID] : null;
  const manualMismatch = ov.getMode() === 'manual' && nextId !== autoId;
  const changed = State.sampleAreaId !== nextId;
  State.sampleAreaId = nextId;
  sel.classList.toggle('error', !selectedArea || manualMismatch);
  document.getElementById('rec-where').textContent = selectedArea
    ? sampleAreaLabel(selectedArea)
    : S.REC_SAMPLE_AREA_OUT_OF_BOUNDS;
  if (changed && State.numpad && shouldReplaceNumberDefault()) {
    prefillNumber();
  }
  refreshMapParcels();
  refreshMapSampleAreas();
  updateMapHeader();
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
  const ordered = State.reference[IPSO_REF_SPECIES].slice().sort(
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
    if (next == null || !shouldReplaceNumberDefault()) return;
    setAutoNumberDefault(next);
  } catch (_) { /* leave blank on error */ }
}

function shouldReplaceNumberDefault() {
  const current = State.numpad ? State.numpad.value('numero') : '';
  return current === '' || current === State.autoNumberValue;
}

function setAutoNumberDefault(number) {
  const value = number == null ? '' : '' + number;
  State.inAutoNumberFill = true;
  try {
    State.numpad.setValue('numero', value);
  } finally {
    State.inAutoNumberFill = false;
  }
  State.autoNumberValue = value;
}

// In-session max+1 takes precedence; on a fresh sample session, fall
// back to Abies' max tree number for the selected survey/area. Other modes
// then use their configured first number or per-operator counter.
async function computeNextNumberDefault(trees) {
  let rows = trees;
  if (isSamplesMode()) {
    rows = trees.filter((t) => t && t[FIELD_SAMPLE_AREA_ID] === State.sampleAreaId);
  }
  const inSession = session.nextNumberDefault(rows);
  if (inSession != null) return inSession;
  if (isSamplesMode()) {
    const sampleArea = currentSampleArea();
    const maxNumber = sampleArea ? sampleArea[FIELD_MAX_TREE_NUMBER] : null;
    if (Number.isInteger(maxNumber)) return maxNumber + 1;
  }
  if (Number.isInteger(currentMode().firstNumber)) return currentMode().firstNumber;
  if (!State.session || !currentMode().persistNumber) return null;
  return Store.getNextNumberForOperator(
    State.db, State.session.operatore, State.session.mode
  );
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
        IpsoFormat.fmtCoord(st.fix.lat) + ' ' + IpsoFormat.fmtCoord(st.fix.lon) +
        ' ±' + Math.round(st.fix.acc) + ' m';
      State.lastFix = {
        lat: st.fix.lat,
        lon: st.fix.lon,
        acc: st.fix.acc,
        t: st.fix.t,
      };
      if (State.locator) State.locator.onFix(st.fix);
      if (isSamplesMode()) refreshSampleAreaSelect();
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

function shouldAutoHeight() {
  return !!currentMode().autoHeight;
}

function treeValidationOptions() {
  return {
    dRequired: currentMode().dRequired !== false,
    hRequired: currentMode().hRequired !== false,
    numberRequired: !!currentMode().numberRequired,
    sampleAreaRequired: !!currentMode().sampleAreaRequired,
  };
}

function recomputeAutoH() {
  if (!shouldAutoHeight()) {
    const hint = document.getElementById('hint-autoh');
    if (hint) hint.hidden = true;
    return;
  }
  const compresa = State.session ? State.session.compresa : '';
  const ipsTable = State.reference[IPSO_REF_HYPSOMETRY];
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
  const sampleArea = currentSampleArea();
  const particella = sampleArea
    ? sampleArea.particella
    : State.override
      ? State.override.resolve(currentAutoName())
      : '';
  const autoHeight = shouldAutoHeight();
  const hMeasured = autoHeight ? State.hMeasured : Number.isFinite(h);
  return {
    specie: State.specie || '',
    d_cm: Number.isFinite(d) ? d : null,
    h_m: Number.isFinite(h) ? h : null,
    h_measured: hMeasured ? 1 : 0,
    hypso_param_set_id: autoHeight && !hMeasured ? currentHypsoParamSetId() : null,
    numero: Number.isInteger(n) ? n : null,
    gruppo,
    particella,
    [FIELD_SAMPLE_AREA_ID]: sampleArea ? sampleArea[FIELD_SAMPLE_AREA_ID] : null,
  };
}


function currentHypsoParamSetId() {
  if (!State.session || !State.reference) return null;
  const eq = ipso.lookup(State.reference[IPSO_REF_HYPSOMETRY], State.session.compresa, State.specie);
  return eq && Number.isInteger(eq.hypso_param_set_id)
    ? eq.hypso_param_set_id
    : null;
}

function updateSaveEnabled() {
  const rec = currentRecord();
  const errs = session.validateTree(rec, treeValidationOptions());
  const btn = document.getElementById('btn-save');
  btn.disabled = errs.length > 0 || Date.now() < State.saveLockUntil;
}

async function onSave() {
  const rec = currentRecord();
  if (session.validateTree(rec, treeValidationOptions()).length > 0) return;

  // Small trees are not physically numbered in the field, so we force the
  // stored number blank regardless of what the operator typed. The counter
  // ignores blank-number trees, so the next visible tree continues the
  // sequence.
  if (currentMode().blankSmallNumber &&
      rec.d_cm != null && rec.d_cm <= session.NUMBER_BLANK_D_THRESHOLD) {
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
    refreshMapRecords();

    if (session.shouldBackup(row.seq)) {
      await downloadBackup(row.seq);
    }
  } catch (e) {
    showToast(S.TOAST_SAVE_ERROR(e.message));
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
    setAutoNumberDefault(next);
    updateSaveEnabled();
    refreshMapRecords();
  } catch (e) {
    showToast(S.TOAST_DELETE_ERROR(e.message));
  }
}

async function onEnd() {
  hideModal('modal-confirm-end');
  try {
    const trees = await Store.listTrees(State.db, State.session.id);
    trees.sort((a, b) => a.seq - b.seq);
    if (trees.length === 0) {
      await closeEmptySession(State.session);
      return;
    }
    const csvText = csv.formatFile(State.session, trees);
    const uploadPayload = upload.buildUploadPayload(
      State.session, trees, State.reference, csvText
    );
    await Store.setSessionStatus(State.db, State.session.id, Store.STATUS_PENDING_UPLOAD);
    State.session.status = Store.STATUS_PENDING_UPLOAD;
    // Always download the local CSV first — it is the trust anchor and
    // the operator must not lose data if the upload never succeeds.
    downloadFinal(State.session, trees);
    uploadFlow().enter(State.session.id, uploadPayload, trees.length);
  } catch (e) {
    showToast(S.TOAST_EXPORT_ERROR(e.message));
  }
}

function downloadFinal(sess, trees) {
  const text = csv.formatFile(sess, trees);
  const name = csv.filename(sess, new Date(), 'final');
  downloadText(text, name);
}

// ---------------------------------------------------------------------------
// Map screen
// ---------------------------------------------------------------------------

async function enterMapScreen(returnScreen) {
  if (!State.session) return;
  if (typeof createOrientationMap === 'undefined') {
    showToast(S.MAP_UNAVAILABLE);
    return;
  }
  State.mapReturnScreen = returnScreen || 'screen-rec';
  showScreen('screen-map');
  ensureMap();
  try {
    await State.map.ensure();
  } catch (_) {
    showToast(S.MAP_UNAVAILABLE);
    return;
  }
  renderMapParcels();
  renderMapSampleAreas();
  await renderMapRecords();
  renderMapPai();
  updateMapPosition();
  updateMapHeader();
  setTimeout(() => {
    if (!State.map || !State.map.ready()) return;
    State.map.invalidate();
    centerMapOnContext();
  }, 0);
}

function exitMapScreen() {
  showScreen(State.mapReturnScreen || 'screen-rec');
}

function ensureMap() {
  if (State.map) return;
  State.map = createOrientationMap({
    elementId: 'map',
    formatFeatureLabel: formatParcelText,
    featureName: particellaName,
    formatRecordLabel: formatMapRecordText,
    formatPaiLabel: formatMapPaiText,
    formatSampleAreaLabel: sampleAreaLabel,
    sampleAreaDefaultRadius: upload.DEFAULT_SAMPLE_RADIUS_M,
    paiControlTitle: S.MAP_PAI_TOGGLE,
    getActiveName: currentAutoName,
    getManualName() {
      if (isSamplesMode() && State.override && State.override.getMode() === 'manual') {
        const area = currentSampleArea();
        return area ? area.particella : '';
      }
      return State.override && State.override.getMode() === 'manual'
        ? State.override.getManual()
        : '';
    },
    onFeatureClick(label) {
      document.getElementById('map-sub').textContent = label;
    },
  });
}

function currentMapFeatures() {
  if (!State.terreni || !State.session) return [];
  const compresa = State.session.compresa;
  return State.terreni.filter(
    (f) => f.properties && f.properties.layer === compresa
  );
}

function renderMapParcels() {
  if (!State.map || !State.map.ready()) return;
  State.map.renderParcels(currentMapFeatures());
}

function refreshMapParcels() {
  if (State.currentScreen !== 'screen-map') return;
  renderMapParcels();
}

function renderMapSampleAreas() {
  if (!State.map || !State.map.ready()) return;
  const enabled = State.session && State.session.mode === IpsoModes.SAMPLES;
  State.map.renderSampleAreas(enabled ? sampleAreasForCurrentSurvey() : [], enabled);
}

function refreshMapSampleAreas() {
  if (State.currentScreen !== 'screen-map') return;
  renderMapSampleAreas();
}

async function renderMapRecords() {
  if (!State.map || !State.map.ready() || !State.session) return;
  try {
    const trees = await Store.listTrees(State.db, State.session.id);
    trees.sort((a, b) => a.seq - b.seq);
    State.map.renderRecords(trees);
  } catch (e) {
    showToast(S.TOAST_MAP_POINTS_LOAD_ERROR(e.message));
  }
}

function refreshMapRecords() {
  if (State.currentScreen !== 'screen-map') return;
  renderMapRecords();
}

function renderMapPai() {
  if (!State.map || !State.map.ready()) return;
  const enabled = State.session && State.session.mode === IpsoModes.PAI;
  const records = enabled ? currentMapPaiRecords() : [];
  State.map.renderPai(records, enabled, currentPaiSpeciesColors());
}

function currentMapPaiRecords() {
  if (!State.reference || !State.reference[IPSO_REF_PAI] || !State.session) return [];
  const rows = State.reference[IPSO_REF_PAI][IPSO_REF_PRESERVED_TREES] || [];
  return rows.filter((r) => r && r.compresa === State.session.compresa);
}

function currentPaiSpeciesColors() {
  const rows = State.reference && State.reference[IPSO_REF_SPECIES] || [];
  return IpsoPalette.speciesColorById(rows);
}

function formatMapRecordText(rec) {
  if (!rec) return '';
  const bits = [];
  if (Number.isInteger(rec.numero)) bits.push('n. ' + rec.numero);
  if (rec.specie) bits.push(rec.specie);
  if (rec.d_cm != null) bits.push('D=' + rec.d_cm);
  if (rec.h_m != null) bits.push('h=' + rec.h_m);
  return bits.join(' · ');
}

function formatMapPaiText(rec) {
  if (!rec) return '';
  const bits = [S.MODE_PAI];
  if (Number.isInteger(rec.number)) bits.push('n. ' + rec.number);
  const species = speciesNameById(rec.species_id);
  if (species) bits.push(species);
  if (rec.particella) bits.push(rec.particella);
  if (rec.d_cm != null) bits.push('D=' + rec.d_cm);
  if (rec.h_m != null) bits.push('h=' + rec.h_m);
  return bits.join(' · ');
}

function speciesNameById(id) {
  const rows = State.reference && State.reference[IPSO_REF_SPECIES] || [];
  const found = rows.find((sp) => sp && sp.id === id);
  return found ? found.common : '';
}

function updateMapPosition() {
  if (State.map && State.map.ready()) State.map.updatePosition(State.lastFix);
}

function centerMapOnContext() {
  if (!State.map || !State.map.ready()) return;
  const ok = State.map.center({
    fix: State.lastFix,
    sampleArea: currentSampleArea(),
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
  const sampleArea = currentSampleArea();
  if (sampleArea) {
    sub.textContent = sampleAreaLabel(sampleArea);
  } else if (committed) {
    sub.textContent = formatParcelText(committed);
  } else if (State.lastFix) {
    sub.textContent =
      IpsoFormat.fmtCoord(State.lastFix.lat) + ' ' +
      IpsoFormat.fmtCoord(State.lastFix.lon) +
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
    showToast(S.TOAST_DATA_LOAD_ERROR(e.message));
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

function sampleAreaTextForTree(tree) {
  if (!tree || !Number.isInteger(tree[FIELD_SAMPLE_AREA_ID])) return tree && tree.particella || '';
  const area = sampleAreaById(tree[FIELD_SAMPLE_AREA_ID]);
  return area ? sampleAreaLabel(area) : (tree.particella || '');
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
      { v: isSamplesMode() ? sampleAreaTextForTree(t) : (t.particella || ''), num: false },
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
  uploadFlow().wire();
}

let uploadFlowInstance = null;
function uploadFlow() {
  if (!uploadFlowInstance) {
    uploadFlowInstance = createUploadFlow({
      db: () => State.db,
      uploadToken: () => State.bearerToken,
      stopRecording: stopRecordingSensors,
      acquireWakeLock,
      releaseWakeLock,
      showScreen,
      showToast,
      showDone: showUploadDone,
    });
  }
  return uploadFlowInstance;
}

function stopRecordingSensors() {
  if (State.gps) {
    State.gps.stop();
    State.gps = null;
  }
}


async function closeEmptySession(sess) {
  await Store.setSessionStatus(State.db, sess.id, Store.STATUS_ABANDONED);
  if (State.session && State.session.id === sess.id) {
    State.session.status = Store.STATUS_ABANDONED;
  }
  stopRecordingSensors();
  releaseWakeLock();
  document.getElementById('done-title').textContent = S.DONE_EMPTY_TITLE;
  document.getElementById('done-body').textContent = S.DONE_EMPTY_BODY;
  showScreen('screen-done');
}

function showUploadDone(treeCount, uploaded) {
  document.getElementById('done-title').textContent = S.DONE_TITLE;
  document.getElementById('done-body').textContent = uploaded
    ? S.UPLOAD_DONE_BODY(treeCount)
    : S.DONE_BODY(treeCount);
  showScreen('screen-done');
}

// ---------------------------------------------------------------------------
// Done screen
// ---------------------------------------------------------------------------

function wireDone() {
  document.getElementById('btn-new-session').addEventListener('click', () => {
    if (State.gps) { State.gps.stop(); State.gps = null; }
    document.getElementById('pre-form').reset();
    populateOperator();
    populateComprese();
    populateSampleSurveyOptions();
    showModeScreen();
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
        if (trees.length === 0) {
          State.session = s;
          setMode(s.mode);
          await closeEmptySession(s);
          return;
        }
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
        uploadFlow().enter(s.id, uploadPayload, trees.length);
      });
      const local = mkBtn(S.UPLOAD_RESUME_KEEP_LOCAL, 'btn-secondary', async () => {
        await Store.setSessionUploadStatus(State.db, s.id, Store.UPLOAD_STATUS_LOCAL_ONLY);
        await Store.setSessionStatus(State.db, s.id, Store.STATUS_EXPORTED);
        li.remove();
        if (!list.children.length) {
          hideModal('modal-resume');
          showModeScreen();
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
          showModeScreen();
        }
      });
      const discard = mkBtn(S.RESUME_DISCARD, 'btn-danger', async () => {
        await Store.setSessionStatus(State.db, s.id, Store.STATUS_ABANDONED);
        li.remove();
        if (!list.children.length) {
          hideModal('modal-resume');
          showModeScreen();
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
  boot().catch((e) => showToast(S.TOAST_BOOT_ERROR(e.message)));
});
