// Regression tests for Ipso app-shell upload safety ordering.
// Run with: node apps/ipso/static/ipso/pwa/app.test.mjs

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const appSource = fs.readFileSync(path.join(here, 'app.js'), 'utf8') + `\n` +
  `globalThis.__ipsoAppTest = { State, boot, onSave, onEnd, onDeleteTree, ` +
  `showResumeModal, prefillNumber, currentRecord, currentObservationRecord, renderTreesTable, ` +
  `renderObservationsTable, onObservationPhotosPicked, ` +
  `wireModeSelection, onHeightMeasuredToggle, recomputeAutoH, ` +
  `shouldAutoHeight, validateReference, validateTerreniFeatures, ` +
  `restoreCachedBootResources, refreshBootResources, wireAppUpdateButton, ` +
  `registerServiceWorker, watchServiceWorkerUpdates, activatePendingAppUpdate, ` +
  `loadBearerToken, bearerHeaders };\n`;

let pass = 0;
const failures = [];
function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
}

class MockElement {
  constructor(tag, id = '') {
    this.tagName = tag.toLowerCase();
    this.id = id;
    this.children = [];
    this.parentNode = null;
    this.className = '';
    this.textContent = '';
    this.type = '';
    this.value = '';
    this.hidden = false;
    this.checked = false;
    this.disabled = false;
    this._listeners = {};
    this.classList = {
      add: (...names) => this._setClasses(new Set([...this._classes(), ...names])),
      remove: (...names) => {
        const next = this._classes();
        for (const name of names) next.delete(name);
        this._setClasses(next);
      },
      contains: name => this._classes().has(name),
      toggle: (name, force) => {
        const next = this._classes();
        const add = force === undefined ? !next.has(name) : force;
        if (add) next.add(name); else next.delete(name);
        this._setClasses(next);
      },
    };
  }
  get options() { return this.children; }
  _classes() { return new Set(this.className.split(/\s+/).filter(Boolean)); }
  _setClasses(classes) { this.className = [...classes].join(' '); }
  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  remove() {
    if (this.parentNode) {
      this.parentNode.children = this.parentNode.children.filter(c => c !== this);
      this.parentNode = null;
    }
  }
  replaceChildren(...children) {
    this.children = [];
    for (const child of children) this.appendChild(child);
  }
  setAttribute(name, value) { this[name] = String(value); }
  closest() { return { hidden: false }; }
  querySelector() { return null; }
  reset() {}
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  async click() {
    for (const fn of this._listeners.click || []) {
      await fn({ target: this, preventDefault() {} });
    }
  }
}

function makeHarness({ storedToken = 'test-token', hash = '', serviceWorker = null } = {}) {
  const events = [];
  const elements = new Map();
  const buttons = [];
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new MockElement('div', id));
    return elements.get(id);
  };
  element('modal-confirm-end');
  element('modal-resume').className = 'hidden';
  element('banner-reference').className = 'banner hidden';
  element('banner-storage').className = 'banner hidden';
  element('btn-app-update').className = 'hidden';
  element('resume-title');
  element('resume-body');
  element('resume-list');
  element('resume-footer');
  element('toast');

  const localValues = new Map(storedToken ? [['ipso.bearer_token', storedToken]] : []);
  const modes = {
    martellate: {
      id: 'martellate', labelKey: 'MODE_MARTELLATE',
      buttonId: 'btn-mode-martellate', enabled: true,
    },
    samples: {
      id: 'samples', labelKey: 'MODE_SAMPLES',
      buttonId: 'btn-mode-samples', enabled: true,
    },
    free_survey: {
      id: 'free_survey', labelKey: 'MODE_FREE_SURVEYS',
      preTitleKey: 'PRE_NEW_FREE_SURVEY', buttonId: 'btn-mode-free-survey',
      autoHeight: true, freeSurvey: true, enabled: true,
    },
    map: {
      id: 'map', labelKey: 'MODE_MAP', buttonId: 'btn-mode-map',
      mapOnly: true, enabled: true,
    },
  };
  const strings = new Proxy({
    APP_UPDATE: 'Aggiorna app',
    APP_UPDATING: 'Aggiornamento…',
    MODE_MARTELLATE: 'Martellate',
    MODE_SAMPLES: 'Rilevamenti predefiniti',
    MODE_FREE_SURVEYS: 'Rilevamenti liberi',
    MODE_MAP: 'Mappa',
    ERROR_GEO_UNAVAILABLE: 'geo unavailable',
    ERROR_HTTP_STATUS: (status) => `HTTP ${status}`,
    ERROR_TOKEN_MISSING: 'token missing',
    ERROR_REFERENCE_INVALID: 'invalid reference',
    ERROR_GEOJSON_INVALID: 'invalid geojson',
    TOAST_REFERENCE_LOAD_ERROR: (msg) => `reference error: ${msg}`,
    TOAST_TERRENI_LOAD_ERROR: (msg) => `terreni error: ${msg}`,
    TOAST_DB_OPEN_ERROR: (msg) => `db error: ${msg}`,
    TOAST_BOOT_CACHE_ERROR: (msg) => `cache error: ${msg}`,
    TOAST_REFERENCE_REQUIRED: 'reference required',
    REC_OBSERVATION_PHOTO_OK: 'ok',
    REC_OBSERVATION_PHOTO_ERROR: 'errore',
    TOAST_PHOTO_PROCESS_ERROR: (msg) => `photo error: ${msg}`,
    TOAST_UPLOAD_SIZE_WARNING: (size, limit) => `upload size ${size}/${limit}`,
    REC_OBSERVATION_PHOTOS_PROCESSING: (count) => `processing ${count}`,
    REFERENCE_OFFLINE_WARNING: 'offline reference',
    STORAGE_WARNING: 'storage warning',
    TOAST_DUPLICATE_NUMBER: (number) => `duplicate ${number}`,
    GPS_PERMISSION_BANNER: 'gps denied',
    PRE_PICK_COMPRESA: 'pick region',
    PRE_PICK_SURVEY: 'pick survey',
    PRE_NEW_FREE_SURVEY: 'Nuovo rilevamento libero',
    REC_H_MEASURED: 'h misurata',
    REC_PRESERVED: 'Albero da preservare',
    DONE_TITLE: 'Sessione esportata',
    DONE_BODY: (n) => `${n} alberi salvati su CSV.`,
    TOAST_EXPORT_ERROR: (msg) => `export error: ${msg}`,
    UPLOAD_RESUME_TITLE: 'Upload sospeso',
    RESUME_TITLE: 'Riprendi',
    RESUME_BODY: 'body',
    UPLOAD_RESUME_DO_NOW: 'Carica',
    UPLOAD_RESUME_KEEP_LOCAL: 'Mantieni locale',
    RESUME_RESUME: 'Riprendi',
    RESUME_EXPORT: 'Esporta',
    RESUME_DISCARD: 'Scarta',
    RESUME_ARCHIVE_TITLE: 'Archivio locale',
    RESUME_ARCHIVE_BODY: 'archive body',
    RESUME_STATUS_EXPORTED: 'esportata',
    RESUME_STATUS_ABANDONED: 'scartata',
    RESUME_CLOSE: 'Continua',
    RESUME_DISCARD_CONFIRM: 'confirm discard',
    where: (sess) => sess.compresa || '',
  }, {
    get(target, property) {
      return Object.prototype.hasOwnProperty.call(target, property)
        ? target[property]
        : String(property);
    },
  });
  const context = {
    console,
    setTimeout: () => 0,
    clearTimeout: () => {},
    URLSearchParams,
    APP_VERSION: 'test',
    IPSO_BEARER_STORAGE_KEY: 'ipso.bearer_token',
    IPSO_OPERATOR_STORAGE_KEY: 'ipso.operatore',
    IPSO_SPECIES_STORAGE_KEY: 'ipso.specie',
    GPS_STALE_MS: 10000,
    SAVE_COOLDOWN_MS: 300,
    SAVE_COOLDOWN_RECHECK_MS: 320,
    IPSO_SECRET_HASH_PARAM: 'secret',
    IPSO_REF_SPECIES: 'species',
    IPSO_REF_PARCELS: 'parcels',
    IPSO_REF_HYPSOMETRY: 'ipsometrica',
    IPSO_REF_SAMPLING: 'sampling',
    IPSO_REF_SURVEYS: 'surveys',
    IPSO_REF_SAMPLE_AREAS: 'sample_areas',
    IPSO_REF_PAI: 'pai',
    IPSO_REF_PRESERVED_TREES: 'preserved_trees',
    IPSO_REF_OBSERVATION_CATEGORIES: 'observation_categories',
    IPSO_REF_UPLOAD: 'upload',
    RECORDS: 'records',
    FIELD_SURVEY_ID: 'survey_id',
    FIELD_SAMPLE_GRID_ID: 'sample_grid_id',
    FIELD_SAMPLE_AREA_ID: 'sample_area_id',
    FIELD_MAX_TREE_NUMBER: 'max_tree_number',
    FIELD_REGION_ID: 'region_id',
    FIELD_PARCEL_ID: 'parcel_id',
    FIELD_SPECIES_ID: 'species_id',
    FIELD_CSV_TEXT: 'csv_text',
    FIELD_CLIENT_PHOTO_ID: 'client_photo_id',
    FIELD_CONTENT_TYPE: 'content_type',
    FIELD_SIZE_BYTES: 'size_bytes',
    FIELD_MAX_BYTES: 'max_bytes',
    FIELD_WIDTH_PX: 'width_px',
    FIELD_HEIGHT_PX: 'height_px',
    FIELD_ORIGINAL_SIZE_BYTES: 'original_size_bytes',
    FIELD_ORIGINAL_WIDTH_PX: 'original_width_px',
    FIELD_ORIGINAL_HEIGHT_PX: 'original_height_px',
    FIELD_CONVERSION_STATUS: 'conversion_status',
    FIELD_CONVERSION_REASON: 'conversion_reason',
    FIELD_ORIGINAL_FILENAME: 'original_filename',
    PHOTO_CONVERSION_CONVERTED: 'converted',
    PHOTO_CONVERSION_ORIGINAL: 'original',
    PHOTO_CONVERSION_UNAVAILABLE: 'unavailable',
    PHOTO_CONVERSION_FAILED: 'failed',
    FIELD_LAT: 'lat',
    FIELD_LON: 'lon',
    FIELD_ACC_M: 'acc_m',
    FIELD_TEXT: 'text',
    FIELD_CATEGORIES: 'categories',
    FIELD_CATEGORY_IDS: 'category_ids',
    FIELD_PHOTOS: 'photos',
    FIELD_DATE: 'date',
    ipsoPositiveInt(value) {
      return Number.isInteger(value) && value > 0 ? value : null;
    },
    FIELD_COPPICE: 'coppice',
    FIELD_PRESERVED: 'preserved',
    IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX: 'sampling_survey:',
    window: {
      AbiesGeoReady: Promise.resolve(),
      location: {
        hash, pathname: '/ipso/', search: '',
        reload() { events.push('reload'); },
      },
      history: { replaceState(_state, _title, url) { events.push(['replaceState', url]); } },
      addEventListener() {},
      confirm() { events.push('confirm'); return true; },
    },
    navigator: serviceWorker ? { serviceWorker } : {},
    localStorage: {
      getItem(key) { return localValues.get(key) || ''; },
      setItem(key, value) { localValues.set(key, value); },
    },
    document: {
      createElement(tag) {
        const el = new MockElement(tag);
        if (tag === 'button') buttons.push(el);
        return el;
      },
      createElementNS(_ns, tag) { return new MockElement(tag); },
      getElementById: element,
      querySelectorAll: () => [],
      addEventListener() {},
      visibilityState: 'visible',
    },
    S: strings,
    IpsoModes: {
      MARTELLATE: 'martellate',
      SAMPLES: 'samples',
      FREE_SURVEY: 'free_survey',
      OBSERVATIONS: 'observations',
      get(id) { events.push(['modeGet', id]); return modes[id] || modes.martellate; },
      defaultMode() { return modes.martellate; },
      all() {
        return [
          modes.samples, modes.free_survey, modes.martellate,
          modes.map,
        ];
      },
    },
    session: {
      nextNumberDefault(trees) {
        const numbers = trees.map(tree => tree.numero)
          .filter(number => Number.isInteger(number));
        return numbers.length ? Math.max(...numbers) + 1 : null;
      },
    },
    Store: {
      STATUS_PENDING_UPLOAD: 'pending_upload',
      STATUS_OPEN: 'open',
      STATUS_EXPORTED: 'exported',
      STATUS_ABANDONED: 'abandoned',
      UPLOAD_STATUS_LOCAL_ONLY: 'local_only',
      uuid() { return `uuid-${events.length}`; },
      async openDb() { events.push('openDb'); return {}; },
      async getCachedBootResources() {
        events.push('getCachedBootResources');
        return { reference: null, terreni: null };
      },
      async cacheReference(_db, value) { events.push(['cacheReference', value]); },
      async cacheTerreni(_db, value) { events.push(['cacheTerreni', value]); },
      async listResumableSessions() { events.push('listResumableSessions'); return []; },
      async listRecoverableSessions() { events.push('listRecoverableSessions'); return []; },
      async listTrees() {
        events.push('listTrees');
        return [{ id: 1, seq: 1, specie: 'Abete' }];
      },
      async addTree() { events.push('addTree'); return { seq: 1, gruppo: '' }; },
      async deleteTree(_db, sessionId, treeId) { events.push(['deleteTree', sessionId, treeId]); },
      async getSession(_db, sessionId) { events.push(['getSession', sessionId]); return { ...session, id: sessionId }; },
      async setSessionStatus() { events.push('setSessionStatus'); },
      async setSessionPendingUpload(_db, sessionId, payload, count) {
        events.push(['setSessionPendingUpload', sessionId, payload, count]);
      },
      async setSessionUploadStatus() { events.push('setSessionUploadStatus'); },
    },
    csv: {
      formatFile() { events.push('csv'); return 'csv-text'; },
      filename() { return 'final.csv'; },
      formatDate: (ymd) => ymd,
    },
    upload: {
      uploadMaxBytes() { return 30 * 1024 * 1024; },
      distanceMeters() { return 0; },
      buildUploadPayload() {
        events.push('buildPayload');
        throw new Error('validation failed');
      },
    },
    IpsoPhotos: {
      async prepareObservationPhoto(file) {
        return {
          blob: file,
          contentType: file.type || '',
          sizeBytes: file.convertedSize || file.size || 0,
          originalFilename: file.name || '',
          width_px: file.width_px || null,
          height_px: file.height_px || null,
          original_size_bytes: file.size || 0,
          original_width_px: file.original_width_px || null,
          original_height_px: file.original_height_px || null,
          conversion_status: file.conversion_status || 'converted',
          conversion_reason: file.conversion_reason || '',
        };
      },
      formatBytes(bytes) { return `${bytes} B`; },
    },
    IpsoFormat: {
      fmtCoord(value) { return Number(value).toFixed(6); },
    },
    ipso: {
      lookup(ipsometrica, compresa, specie) {
        return ipsometrica && ipsometrica[compresa]
          ? ipsometrica[compresa][specie] || null : null;
      },
      computeH(eq, d) { return eq ? Math.round(eq.a * Math.log(d) + eq.b) : null; },
    },
    createUploadFlow() {
      return {
        enter(sessionId, payload, count) {
          events.push(['uploadEnter', sessionId, payload, count]);
        },
        wire() {},
      };
    },
    createNumpad() {
      return { mount() {}, value() { return ''; }, setValue() {} };
    },
    downloadText() { events.push('download'); },
    fetch: async () => { throw new Error('offline'); },
  };
  vm.createContext(context);
  vm.runInContext(appSource, context, { filename: 'app.js' });
  return { context, events, elements, buttons, localValues };
}

function referenceFixture(version = 'cached') {
  return {
    reference_version: version,
    species: [{ id: 10, common: 'Abete' }],
    parcels: [{ region_id: 1, parcel_id: 100, compresa: 'Serra', particella: '1' }],
    ipsometrica: {},
    sampling: { surveys: [], sample_areas: [] },
    pai: { preserved_trees: [] },
    observation_categories: [],
  };
}

const session = {
  id: 's1',
  status: 'pending_upload',
  mode: 'martellate',
  data: '2026-06-17',
  compresa: 'Serra',
  operatore: 'Mario',
  tree_count: 1,
};

// The free-survey landing entry starts the standard pre-session flow but is
// marked local-only until Abies import support lands in the next CL.
{
  const { context, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.reference = referenceFixture();
  app.wireModeSelection();

  const samplesButton = elements.get('btn-mode-samples');
  const freeButton = elements.get('btn-mode-free-survey');
  check(samplesButton.textContent === 'Rilevamenti predefiniti',
        'sample mode is labelled as predefined surveys');
  check(!samplesButton.disabled, 'predefined survey mode remains enabled');
  check(freeButton.textContent === 'Rilevamenti liberi',
        'free-survey mode is labelled on the landing page');
  check(!freeButton.disabled, 'free-survey mode is enabled');

  await freeButton.click();
  check(app.State.mode.id === 'free_survey',
        'free-survey button selects the free-survey mode');
  check(app.State.currentScreen === 'screen-pre',
        'free-survey button opens the pre-session screen');
  check(elements.get('pre-title').textContent === 'Nuovo rilevamento libero',
        'free-survey pre-session title is localized');
}

// A previously provisioned device keeps using the bearer secret stored before
// the upgrade unless a fresh provisioning fragment explicitly replaces it.
{
  const { context, localValues } = makeHarness({ storedToken: 'old-shared-secret' });
  const app = context.__ipsoAppTest;
  app.State.bearerToken = app.loadBearerToken();
  check(app.State.bearerToken === 'old-shared-secret',
        'upgrade reuses the stored shared Ipso secret');
  eq(app.bearerHeaders(), { Authorization: 'Bearer old-shared-secret' },
     'stored shared secret is sent as the bearer token');
  check(localValues.get('ipso.bearer_token') === 'old-shared-secret',
        'reading the stored secret does not rewrite it');
}

{
  const { context, localValues, events } = makeHarness({
    storedToken: 'old-shared-secret',
    hash: '#secret=new-shared-secret',
  });
  const app = context.__ipsoAppTest;
  check(app.loadBearerToken() === 'new-shared-secret',
        'provisioning fragment intentionally replaces the stored shared secret');
  check(localValues.get('ipso.bearer_token') === 'new-shared-secret',
        'new provisioning secret is persisted for later launches');
  check(events.some(event => Array.isArray(event) && event[0] === 'replaceState' &&
        !event[1].includes('secret=')),
        'provisioning fragment is removed from the address bar');
}

// In free surveys an unchecked h_measured box means h is derived from the
// region/species regression, not manually entered.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  context.session.validateTree = () => [];
  app.State.reference = {
    ...referenceFixture(),
    ipsometrica: { Serra: { Abete: { a: 0, b: 18, hypso_param_set_id: 44 } } },
  };
  app.State.session = { ...session, status: 'open', mode: 'free_survey', region_id: 1 };
  app.State.specie = 'Abete';
  app.State.override = { resolve: () => '1' };
  let hValue = '';
  app.State.numpad = {
    value(field) { return { d: '42', h: hValue, numero: '' }[field] || ''; },
    setValue(field, value) { if (field === 'h') hValue = value; },
  };

  const measured = context.document.getElementById('in-h-measured');
  measured.checked = false;
  app.onHeightMeasuredToggle();
  check(app.shouldAutoHeight(), 'unchecked free-survey h_measured enables auto height');
  check(hValue === '18', 'unchecked free-survey h_measured computes h');
  let record = app.currentRecord();
  check(record.h_measured === 0, 'derived free-survey h is recorded as unmeasured');
  check(record.hypso_param_set_id === 44,
        'derived free-survey h records the regression parameter set');

  measured.checked = true;
  app.onHeightMeasuredToggle();
  check(hValue === '', 'checking h_measured clears the derived height');
  record = app.currentRecord();
  check(record.h_measured === 1, 'checked free-survey h is recorded as measured');
}

// Free-survey rows preserve the explicit h_measured and preserved flags.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.reference = referenceFixture();
  app.State.session = { ...session, status: 'open', mode: 'free_survey', region_id: 1 };
  app.State.specie = 'Abete';
  app.State.override = { resolve: () => '1' };
  app.State.numpad = {
    value(field) { return { d: '42', h: '22', numero: '' }[field] || ''; },
  };
  context.document.getElementById('in-h-measured').checked = false;
  context.document.getElementById('in-preserved').checked = true;

  const record = app.currentRecord();
  check(record.numero === null, 'free-survey tree number is optional');
  check(record.h_measured === 0, 'free-survey row preserves h_measured false');
  check(record.preserved === true, 'free-survey row preserves the PAI flag');
}

// Canonical IDs are captured with each observation; names remain display/CSV
// data and are not deferred until upload time.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.reference = referenceFixture();
  app.State.session = { ...session, status: 'open', region_id: 1 };
  app.State.specie = 'Abete';
  app.State.override = { resolve: () => '1' };
  app.State.numpad = {
    value(field) { return { d: '42', h: '22', numero: '7' }[field] || ''; },
  };
  context.document.getElementById('in-gruppo').value = 'A';
  const record = app.currentRecord();
  check(record.region_id === 1, 'record captures canonical region ID');
  check(record.parcel_id === 100, 'record captures canonical parcel ID');
  check(record.species_id === 10, 'record captures canonical species ID');
  check(record.specie === 'Abete' && record.particella === '1',
        'record retains names for display and CSV export');
}

// A waiting service worker is exposed as an explicit footer update action.
{
  const workerMessages = [];
  const waiting = {
    postMessage(message) { workerMessages.push(message); },
  };
  const registration = {
    waiting,
    async update() {},
    addEventListener() {},
  };
  const swListeners = {};
  const serviceWorker = {
    controller: {},
    async register(url, options) {
      eq([url, options.updateViaCache], ['./sw.js', 'none'],
         'service worker registration bypasses the browser cache');
      return registration;
    },
    addEventListener(type, handler) { swListeners[type] = handler; },
  };
  const { context, elements, events } = makeHarness({ serviceWorker });
  const app = context.__ipsoAppTest;
  app.wireAppUpdateButton();
  app.registerServiceWorker();
  await Promise.resolve();

  const button = elements.get('btn-app-update');
  check(!button.classList.contains('hidden'),
        'waiting service worker shows the update button');
  check(button.textContent === 'Aggiorna app',
        'update button uses the localized label');
  await button.click();
  eq(workerMessages, [{ type: 'SKIP_WAITING' }],
     'update button asks the waiting worker to activate');
  check(button.disabled, 'update button is disabled while activation is pending');
  check(button.textContent === 'Aggiornamento…',
        'update button switches to the localized pending label');
  swListeners.controllerchange();
  check(events.includes('reload'),
        'controllerchange reloads the page into the new shell');
}

// Installing workers only expose the update action once they reach installed.
{
  const workerListeners = {};
  const installing = {
    state: 'installing',
    addEventListener(type, handler) { workerListeners[type] = handler; },
  };
  const registrationListeners = {};
  const registration = {
    waiting: null,
    installing: null,
    async update() {},
    addEventListener(type, handler) { registrationListeners[type] = handler; },
  };
  const serviceWorker = {
    controller: {},
    async register() { return registration; },
    addEventListener() {},
  };
  const { context, elements } = makeHarness({ serviceWorker });
  const app = context.__ipsoAppTest;
  app.wireAppUpdateButton();
  app.registerServiceWorker();
  await Promise.resolve();

  const button = elements.get('btn-app-update');
  check(button.classList.contains('hidden'),
        'installing service worker does not show the update button');
  registration.installing = installing;
  registrationListeners.updatefound();
  installing.state = 'installed';
  registration.waiting = installing;
  workerListeners.statechange();
  check(!button.classList.contains('hidden'),
        'installed service worker shows the update button');
}

// Cached protected resources are validated and restored before network work.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture();
  const terreni = [{ type: 'Feature', properties: { particella: '1' } }];
  app.State.db = {};
  context.Store.getCachedBootResources = async () => ({ reference, terreni });
  await app.restoreCachedBootResources();
  eq(app.State.reference, reference, 'boot restores the last-good reference snapshot');
  eq(app.State.terreni, terreni, 'boot restores the last-good parcel geometry');

  let invalidRejected = false;
  try { app.validateReference({ parcels: [] }); } catch (_) { invalidRejected = true; }
  check(invalidRejected, 'invalid reference snapshots are rejected');
}

// A successful online refresh validates, persists, and adopts both resources.
{
  const { context, events } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture('fresh');
  const terreni = [{ type: 'Feature', properties: { particella: '2' } }];
  app.State.db = {};
  app.State.bearerToken = 'test-token';
  context.fetch = async (url) => ({
    ok: true,
    json: async () => url === 'reference.json'
      ? reference
      : { type: 'FeatureCollection', features: terreni },
  });
  const result = await app.refreshBootResources();
  eq(result, { reference: true, terreni: true }, 'online refresh reports both resources fresh');
  eq(app.State.reference, reference, 'online refresh adopts the fresh reference');
  eq(app.State.terreni, terreni, 'online refresh adopts fresh parcel geometry');
  check(events.some((event) => Array.isArray(event) && event[0] === 'cacheReference'),
        'online refresh persists the validated reference');
  check(events.some((event) => Array.isArray(event) && event[0] === 'cacheTerreni'),
        'online refresh persists validated parcel geometry');
}

// Failed refresh leaves last-good data in place and makes staleness visible.
{
  const { context, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture();
  const terreni = [{ type: 'Feature', properties: { particella: '1' } }];
  app.State.db = {};
  app.State.bearerToken = 'test-token';
  app.State.reference = reference;
  app.State.terreni = terreni;
  context.fetch = async () => ({ ok: false, status: 503 });
  const result = await app.refreshBootResources();
  eq(result, { reference: false, terreni: false }, 'offline refresh reports cached resources');
  eq(app.State.reference, reference, 'offline refresh preserves the cached reference');
  eq(app.State.terreni, terreni, 'offline refresh preserves cached parcel geometry');
  check(elements.get('banner-reference').textContent === 'offline reference',
        'offline refresh shows a persistent stale-data warning');
  check(!elements.get('banner-reference').classList.contains('hidden'),
        'offline stale-data warning is visible');
}

// Cold-start with cached resources must not await the opportunistic network
// refresh; IndexedDB and resumable sessions are read before fetch begins.
{
  const { context, events, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  const reference = referenceFixture();
  const terreni = [{ type: 'Feature', properties: { particella: '1' } }];
  context.Store.openDb = async () => { events.push('openDb'); return {}; };
  context.Store.getCachedBootResources = async () => {
    events.push('getCachedBootResources');
    return { reference, terreni };
  };
  context.Store.listResumableSessions = async () => {
    events.push('listResumableSessions');
    return [{ ...session, status: 'open' }];
  };
  context.fetch = (url) => {
    events.push('fetch:' + url);
    return new Promise(() => {});
  };
  await app.boot();
  const firstFetch = events.findIndex((event) =>
    typeof event === 'string' && event.startsWith('fetch:')
  );
  check(events.indexOf('openDb') < firstFetch, 'cold boot opens IndexedDB before network fetch');
  check(events.indexOf('getCachedBootResources') < firstFetch,
        'cold boot restores protected resources before network fetch');
  check(events.indexOf('listResumableSessions') < firstFetch,
        'cold boot lists resumable sessions before network fetch');
  check(!elements.get('modal-resume').classList.contains('hidden'),
        'cold boot exposes locally stored resumable sessions immediately');
}

// Devices upgraded with an already-open session may not have a resource
// snapshot yet. Even then, failed network bootstrap must expose export/discard
// actions instead of returning before the resume modal is built.
{
  const { context, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  context.Store.listResumableSessions = async () => [
    { ...session, status: 'open' },
  ];
  context.fetch = async () => ({ ok: false, status: 503 });
  await app.boot();
  check(!elements.get('modal-resume').classList.contains('hidden'),
        'upgraded offline device still exposes its pre-cache session');
  const resume = buttons.find((button) => button.textContent === 'Riprendi');
  await resume.click();
  check(app.State.session === null, 'recording cannot resume without valid reference data');
  check(elements.get('toast').textContent === 'reference required',
        'missing reference explains why recording cannot resume');
}

// Ending a live session must download the local CSV before payload validation.
{
  const { context, events } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  app.State.session = { ...session, status: 'open' };
  await app.onEnd();
  check(events.indexOf('download') >= 0, 'onEnd downloads CSV even when payload validation throws');
  check(events.indexOf('download') < events.indexOf('buildPayload'),
        'onEnd downloads CSV before building the upload payload');
  check(!events.includes('setSessionStatus'), 'onEnd does not mark pending after payload validation fails');
  check(!events.some(e => Array.isArray(e) && e[0] === 'setSessionPendingUpload'),
        'onEnd does not persist a pending upload payload after payload validation fails');
}

// Ending a free-survey session now follows the normal staged-upload path.
{
  const { context, events } = makeHarness();
  const app = context.__ipsoAppTest;
  const payload = { records: [{ client_record_id: 'free-1' }], csv_text: 'csv-text' };
  app.State.db = {};
  app.State.reference = referenceFixture();
  app.State.session = { ...session, status: 'open', mode: 'free_survey' };
  context.Store.listTrees = async () => [{
    id: 1, seq: 1, particella: '1', specie: 'Abete', d_cm: 42, h_m: 22,
    h_measured: 1, preserved: true, lat: 38.5, lon: 16.3, acc_m: 5,
  }];
  context.upload.buildUploadPayload = () => {
    events.push('buildPayload');
    return payload;
  };

  await app.onEnd();

  check(events.includes('download'), 'free-survey end downloads the local CSV');
  check(events.includes('buildPayload'),
        'free-survey end builds an upload payload');
  check(events.some(e => Array.isArray(e) && e[0] === 'setSessionPendingUpload'),
        'free-survey end persists the pending upload');
  check(events.some(e => Array.isArray(e) && e[0] === 'uploadEnter'),
        'free-survey end enters the upload screen');
}


// Free-survey preserved rows use the same per-parcel next-number source as PAI.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {
    ...referenceFixture(),
    pai: { preserved_trees: [{ parcel_id: 100, number: 8 }] },
  };
  app.State.session = { ...session, status: 'open', mode: 'free_survey' };
  app.State.override = { resolve: () => '1' };
  context.document.getElementById('in-preserved').checked = true;
  let numberValue = '';
  app.State.numpad = {
    value(field) { return field === 'numero' ? numberValue : ''; },
    setValue(field, value) { if (field === 'numero') numberValue = value; },
  };
  context.Store.listTrees = async () => [
    { numero: 9, particella: '1', parcel_id: 100, preserved: true },
    { numero: 14, particella: '2', parcel_id: 101, preserved: true },
  ];

  await app.prefillNumber();

  check(numberValue === '10',
        'free-survey preserved numbering advances within the selected parcel');
}

// A successful session end persists the exact upload payload before entering upload.
{
  const { context, events } = makeHarness();
  const app = context.__ipsoAppTest;
  const payload = { records: [{ client_record_id: 'r1' }], csv_text: 'csv-text' };
  app.State.db = {};
  app.State.reference = {};
  app.State.session = { ...session, status: 'open' };
  context.upload.buildUploadPayload = () => {
    events.push('buildPayload');
    return payload;
  };
  await app.onEnd();

  const persisted = events.find(e => Array.isArray(e) && e[0] === 'setSessionPendingUpload');
  const entered = events.find(e => Array.isArray(e) && e[0] === 'uploadEnter');
  eq(persisted, ['setSessionPendingUpload', 's1', payload, 1],
     'onEnd stores the exact payload and tree count on the pending session');
  eq(entered, ['uploadEnter', 's1', payload, 1],
     'onEnd enters upload with the exact payload it persisted');
}

// Retrying a legacy pending upload from the resume modal has the same safety ordering.
{
  const { context, events, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  app.showResumeModal([{ ...session }]);
  const carica = buttons.find(b => b.textContent === 'Carica');
  check(Boolean(carica), 'resume modal renders the pending-upload Carica button');
  await carica.click();
  check(events.indexOf('download') >= 0, 'resume upload downloads CSV even when payload validation throws');
  check(events.indexOf('download') < events.indexOf('buildPayload'),
        'resume upload downloads CSV before building the upload payload');
}

// Retrying a new pending upload reuses the previously persisted payload verbatim.
{
  const { context, events, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  const payload = { records: [{ client_record_id: 'r1' }], csv_text: 'persisted csv' };
  app.State.db = {};
  app.State.reference = { changed: true };
  app.showResumeModal([{ ...session, upload_payload: payload, upload_tree_count: 1 }]);
  const carica = buttons.find(b => b.textContent === 'Carica');
  await carica.click();

  const entered = events.find(e => Array.isArray(e) && e[0] === 'uploadEnter');
  check(events.includes('download'), 'stored pending upload still downloads a local CSV copy');
  check(!events.includes('listTrees'), 'stored pending upload does not reread local tree rows');
  check(!events.includes('buildPayload'), 'stored pending upload does not rebuild against current reference data');
  eq(entered, ['uploadEnter', 's1', payload, 1],
     'stored pending upload enters upload with the persisted payload');
}

// Terminal sessions must not interrupt startup; only open or pending-upload
// rows require an operator decision on launch.
{
  const { context, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  context.Store.listResumableSessions = async () => [
    { ...session, status: 'exported' },
    { ...session, id: 's2', status: 'abandoned' },
  ];
  context.fetch = async () => ({ ok: false, status: 503 });
  await app.boot();
  check(elements.get('modal-resume').classList.contains('hidden'),
        'cold boot ignores terminal local sessions');
  check(!buttons.some(b => b.textContent === 'Esporta'),
        'terminal local sessions do not render archive export buttons on startup');
}

// Resume-time export marks exported only after CSV generation/download starts.
{
  const { context, events, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  context.Store.setSessionStatus = async (_db, _id, status) => {
    events.push(['setSessionStatus', status]);
  };
  app.showResumeModal([{ ...session, status: 'open' }]);

  const exp = buttons.find(b => b.textContent === 'Esporta');
  await exp.click();

  const downloadIndex = events.indexOf('download');
  const statusIndex = events.findIndex(
    e => Array.isArray(e) && e[0] === 'setSessionStatus',
  );
  check(downloadIndex >= 0, 'resume export starts a download');
  check(statusIndex > downloadIndex, 'resume export marks status after download starts');
  eq(events[statusIndex], ['setSessionStatus', 'exported'],
     'resume export marks the session exported');
  check(elements.get('modal-resume').classList.contains('hidden'),
        'exporting the last active session closes the startup modal');
}

// If CSV generation fails, export must not hide the session by changing status.
{
  const { context, events, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  context.csv.formatFile = () => { throw new Error('format failed'); };
  context.Store.setSessionStatus = async (_db, _id, status) => {
    events.push(['setSessionStatus', status]);
  };
  app.showResumeModal([{ ...session, status: 'open' }]);

  const exp = buttons.find(b => b.textContent === 'Esporta');
  await exp.click();

  check(!events.some(e => Array.isArray(e) && e[0] === 'setSessionStatus'),
        'failed resume export does not mark the session exported');
  check(elements.get('toast').textContent === 'export error: format failed',
        'failed resume export reports the formatting error');
}

// Discard requires confirmation before marking the session abandoned.
{
  const { context, events, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  context.window.confirm = () => { events.push('confirm'); return false; };
  context.Store.setSessionStatus = async (_db, _id, status) => {
    events.push(['setSessionStatus', status]);
  };
  app.showResumeModal([{ ...session, status: 'open' }]);

  const discard = buttons.find(b => b.textContent === 'Scarta');
  await discard.click();

  check(events.includes('confirm'), 'resume discard asks for confirmation');
  check(!events.some(e => Array.isArray(e) && e[0] === 'setSessionStatus'),
        'cancelled resume discard does not abandon the session');
}

// Confirmed discard resolves the active session and closes the modal.
{
  const { context, events, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = {};
  context.Store.setSessionStatus = async (_db, _id, status) => {
    events.push(['setSessionStatus', status]);
  };
  app.showResumeModal([{ ...session, status: 'open' }]);

  const discard = buttons.find(b => b.textContent === 'Scarta');
  await discard.click();

  check(events.some(e => Array.isArray(e) && e[1] === 'abandoned'),
        'confirmed resume discard marks the session abandoned');
  check(elements.get('modal-resume').classList.contains('hidden'),
        'discarding the last active session closes the startup modal');
}

// Marking a pending upload as local-only also resolves the last active session.
{
  const { context, events, elements, buttons } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  context.Store.setSessionUploadStatus = async (_db, _id, status) => {
    events.push(['setSessionUploadStatus', status]);
  };
  context.Store.setSessionStatus = async (_db, _id, status) => {
    events.push(['setSessionStatus', status]);
  };
  app.showResumeModal([{ ...session, status: 'pending_upload' }]);

  const local = buttons.find(b => b.textContent === 'Mantieni locale');
  await local.click();

  check(events.some(e => Array.isArray(e) && e[1] === 'local_only'),
        'local-only pending upload records the upload status');
  check(events.some(e => Array.isArray(e) && e[1] === 'exported'),
        'local-only pending upload marks the session exported');
  check(elements.get('modal-resume').classList.contains('hidden'),
        'local-only pending upload closes the startup modal');
}

// A prefill based on an older tree list must not replace a newer proposal when
// overlapping IndexedDB reads resolve out of order.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  const pending = [];
  let numberValue = '';

  app.State.db = {};
  app.State.session = { id: 's1', mode: 'martellate' };
  app.State.numpad = {
    value: () => numberValue,
    setValue: (_field, value) => { numberValue = value; },
  };
  context.Store.listTrees = () => new Promise(resolve => pending.push(resolve));

  const stalePrefill = app.prefillNumber();
  const freshPrefill = app.prefillNumber();
  pending[1]([{ numero: 1 }, { numero: 2 }]);
  await freshPrefill;
  eq(numberValue, '3', 'newer prefill proposes the post-save number');

  pending[0]([{ numero: 1 }]);
  await stalePrefill;
  eq(numberValue, '3', 'older prefill cannot replace the newer proposal');
}

// Save returns before writing when validation reports a missing required GPS fix.
{
  const { context, events, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = referenceFixture();
  app.State.session = { ...session, id: 's1', status: 'open', region_id: 1 };
  app.State.specie = 'Abete';
  app.State.override = { resolve: () => '1' };
  app.State.numpad = {
    value(field) { return { d: '42', h: '22', numero: '7' }[field] || ''; },
    setValue() {},
  };
  context.session.validateTree = () => ['gps'];

  await app.onSave();

  check(!events.includes('addTree'), 'onSave does not persist when GPS validation fails');
  check(elements.get('gps-text').textContent === 'REC_GPS_WAITING',
        'onSave refreshes GPS status after a GPS validation failure');
}

// Save rejects duplicate numbers in the current local parcel before persisting.
{
  const { context, events, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = referenceFixture();
  app.State.session = { ...session, id: 's1', status: 'open', region_id: 1 };
  app.State.specie = 'Abete';
  app.State.override = { resolve: () => '1' };
  app.State.numpad = {
    value(field) { return { d: '42', h: '22', numero: '7' }[field] || ''; },
    setValue() {},
  };
  context.session.validateTree = () => [];
  context.Store.listTrees = async () => [
    { numero: 7, particella: '1', region_id: 1, parcel_id: 100 },
  ];

  await app.onSave();

  check(!events.includes('addTree'), 'onSave does not persist a duplicate number');
  check(elements.get('toast').textContent === 'duplicate 7',
        'onSave reports the duplicate number immediately');
}




// Observation GPS captured before mobile photo processing remains available
// when save-time GPS has gone stale. Photo metadata is ignored for the
// observation position.
{
  const { context } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.session = { ...session, mode: 'observations', region_id: 1 };
  context.document.getElementById('in-observation-text').value = 'foto vecchia';
  app.State.gps = {
    snapshot() { return { lat: 38.0, lon: 16.0, acc_m: 4 }; },
  };
  let rec = app.currentObservationRecord();
  eq(
    { lat: rec.lat, lon: rec.lon, acc_m: rec.acc_m },
    { lat: 38.0, lon: 16.0, acc_m: 4 },
    'currentObservationRecord captures fresh observation GPS',
  );

  app.State.observationPhotos = [{ lat: 38.5, lon: 16.25 }];
  app.State.gps = { snapshot() { return null; } };
  rec = app.currentObservationRecord();
  eq(
    { lat: rec.lat, lon: rec.lon, acc_m: rec.acc_m },
    { lat: 38.0, lon: 16.0, acc_m: 4 },
    'currentObservationRecord ignores photo coordinates and keeps device GPS',
  );
}

// Observation photo inputs share the same ingestion path. Gallery selection
// may include multiple files; camera capture usually contributes one at a time.
{
  const { context, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.session = { ...session, mode: 'observations' };
  app.State.observationPhotos = [];
  const input = {
    files: [
      {
        name: 'galleria.jpg', type: 'image/jpeg', size: 7000000,
        convertedSize: 500000, original_width_px: 4000,
        original_height_px: 3000, width_px: 2000, height_px: 1500,
      },
      { name: 'camera.png', type: 'image/png', size: 9 },
    ],
    value: 'selected',
  };

  await app.onObservationPhotosPicked({ target: input });

  check(app.State.observationPhotos.length === 2,
        'observation photo picker stores all selected files');
  check(app.State.observationPhotos[0].original_filename === 'galleria.jpg',
        'observation photo picker records original filenames');
  const photoRow = elements.get('observation-photo-list')
    .children[0].textContent;
  check(photoRow.includes('7000000 B -> 500000 B'),
        'observation photo picker displays before/after byte size');
  check(photoRow.includes('ok'),
        'observation photo picker displays simplified upload status');
  check(input.value === '', 'observation photo picker resets the file input');
}

// Observation data rows render as wrapped cards instead of narrow table columns.
{
  const { context, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.session = { ...session, mode: 'observations' };

  app.renderObservationsTable([{
    id: 77,
    seq: 1,
    text: 'testo molto lungo dell\'osservazione',
    categories: ['viabilità', 'rifiuti'],
    photos: [{ original_filename: 'foto.jpg' }],
    lat: 38.123456,
    lon: 16.654321,
  }]);

  const table = elements.get('data-trees-table');
  const tbody = table.children[0];
  const row = tbody.children[0];
  const card = row.children[0].children[0];
  check(table.className === 'data-table observation-table',
        'observation data view switches to card table styling');
  check(card.className === 'observation-card',
        'observation data view renders a card per row');
  check(card.children[0].textContent.includes('testo molto lungo'),
        'observation data card shows wrapped text as the primary content');
}

// The data-table delete button is wired to onDeleteTree and refreshes the view.
{
  const { context, events, buttons, elements } = makeHarness();
  const app = context.__ipsoAppTest;
  app.State.db = {};
  app.State.reference = referenceFixture();
  app.State.session = { ...session, id: 's1', status: 'open', region_id: 1 };
  app.State.specie = 'Abete';
  app.State.override = { resolve: () => '1' };
  app.State.numpad = {
    value(field) { return { d: '42', h: '22', numero: '' }[field] || ''; },
    setValue(field, value) { events.push(['setValue', field, value]); },
  };
  context.session.validateTree = () => [];
  context.Store.getSession = async (_db, sessionId) => {
    events.push(['getSession', sessionId]);
    return { ...app.State.session, tree_count: 0 };
  };
  context.Store.listTrees = async () => [];

  app.renderTreesTable([
    { id: 77, seq: 1, numero: 7, specie: 'Abete', particella: '1', gruppo: '', d_cm: 42, h_m: 22 },
  ]);
  const deleteButton = buttons.find(button => button.className === 'tree-delete-btn');
  await deleteButton.click();

  check(events.some(event => JSON.stringify(event) === JSON.stringify(['deleteTree', 's1', 77])),
        'data-table delete button calls Store.deleteTree with the row id');
  check(elements.get('data-trees-table').children.length === 1,
        'onDeleteTree re-renders the tree table after deletion');
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} app-shell tests passed`);
