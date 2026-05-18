// Tests for ipso pure-logic modules.
// Run with: make test (which depends on `make build`).
// Tests load modules from build/ rather than src/ because the staged
// build is the actual deploy artefact.
'use strict';

const csv = require('../build/csv.js');
const ipso = require('../build/ipso.js');
const session = require('../build/session.js');
const geo = require('../build/geo.js');
// parcel-locator.js calls `findContainingParcel` and
// `distanceToBoundaryMeters` as free identifiers. In the browser those
// resolve to globals declared by geo.js's classic <script>; in node we
// install them on globalThis here so the same lookup works.
Object.assign(globalThis, geo);
const locator = require('../build/parcel-locator.js');

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

function assertClose(actual, expected, tolerance, msg) {
  const diff = Math.abs(actual - expected);
  if (diff <= tolerance) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL ${msg}: expected ~${expected}, got ${actual} (diff ${diff})`);
  }
}

function assertThrows(fn, msg) {
  try {
    fn();
    failed++;
    console.error(`FAIL ${msg}: expected throw, none raised`);
  } catch (e) {
    passed++;
  }
}

// ---------------------------------------------------------------------------
// csv.js
// ---------------------------------------------------------------------------

console.log('csv.js');

assertEqual(csv.formatDate('2026-05-11'), '11/05/2026', 'formatDate basic');
assertEqual(csv.formatDate('2024-01-01'), '01/01/2024', 'formatDate jan');
assertThrows(() => csv.formatDate('11/05/2026'), 'formatDate rejects DD/MM/YYYY');
assertThrows(() => csv.formatDate(''), 'formatDate rejects empty');

assertEqual(csv.fmtFloat(38.4253101, 6), '38,425310', 'fmtFloat 6dp');
assertEqual(csv.fmtFloat(-16.123, 6), '-16,123000', 'fmtFloat negative');
assertEqual(csv.fmtFloat(null, 6), '', 'fmtFloat null -> empty');
assertEqual(csv.fmtFloat(undefined, 6), '', 'fmtFloat undefined -> empty');
assertEqual(csv.fmtFloat(NaN, 6), '', 'fmtFloat NaN -> empty');

assertEqual(csv.fmtInt(42), '42', 'fmtInt');
assertEqual(csv.fmtInt(7.6), '8', 'fmtInt rounds');
assertEqual(csv.fmtInt(null), '', 'fmtInt null');
assertEqual(csv.fmtInt(0), '0', 'fmtInt zero');

assertEqual(csv.escapeField('Pino Nero'), 'Pino Nero', 'escape: no special');
assertEqual(csv.escapeField('a;b'), '"a;b"', 'escape: contains sep');
assertEqual(csv.escapeField('he said "hi"'), '"he said ""hi"""', 'escape: quotes');
assertEqual(csv.escapeField('a\nb'), '"a\nb"', 'escape: newline');
assertEqual(csv.escapeField(null), '', 'escape: null');

assertEqual(
  csv.formatHeader(),
  'Data;Compresa;Particella;Catastrofata;Numero;Specie;D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore',
  'header literal'
);

const sess = {
  data: '2026-05-11', compresa: 'Serra', operatore: 'Mario Rossi',
};
const r1 = {
  specie: 'Abete', d_cm: 42, h_m: 24, h_measured: 0,
  lat: 38.4253101, lng: 16.1204400, acc_m: 7,
  particella: '1',
};
assertEqual(
  csv.formatRow(r1, sess),
  '11/05/2026;Serra;1;0;;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi',
  'formatRow happy path, numero blank'
);

// Same record with a numero set.
assertEqual(
  csv.formatRow(Object.assign({}, r1, { numero: 42 }), sess),
  '11/05/2026;Serra;1;0;42;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi',
  'formatRow with numero'
);

// 4-digit numero.
assertEqual(
  csv.formatRow(Object.assign({}, r1, { numero: 1234 }), sess),
  '11/05/2026;Serra;1;0;1234;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi',
  'formatRow 4-digit numero'
);

const r2 = {
  specie: 'Faggio', d_cm: 30, h_m: 18, h_measured: 1,
  lat: null, lng: null, acc_m: null,
  particella: '1',
};
assertEqual(
  csv.formatRow(r2, sess),
  '11/05/2026;Serra;1;0;;Faggio;30;18;1;;;;Mario Rossi',
  'formatRow no GPS'
);

// Missing operatore on the session row -> empty string in the column.
const sessNoOp = { data: '2026-05-11', compresa: 'Serra' };
assertEqual(
  csv.formatRow(r1, sessNoOp),
  '11/05/2026;Serra;1;0;;Abete;42;24;0;38,425310;16,120440;7;',
  'formatRow no operatore'
);

// Particella source is rec, regardless of catastrofata flag.
const sessCat = {
  data: '2026-05-12', compresa: 'Capistrano',
  catastrofata: true, operatore: 'Anna Bianchi',
};
assertEqual(
  csv.formatRow(Object.assign({}, r1, { numero: 7, particella: '3b' }), sessCat),
  '12/05/2026;Capistrano;3b;1;7;Abete;42;24;0;38,425310;16,120440;7;Anna Bianchi',
  'formatRow catastrofata: Particella from rec, Catastrofata=1'
);

// Empty rec.particella (auto + outside boundaries) -> blank column.
assertEqual(
  csv.formatRow(Object.assign({}, r1, { particella: '' }), sess),
  '11/05/2026;Serra;;0;;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi',
  'formatRow: blank rec.particella -> blank column'
);

const r3 = {
  specie: 'Pino Nero', d_cm: 50, h_m: 26, h_measured: 0,
  lat: 38.4, lng: 16.1, acc_m: 12, particella: '1',
};
const file = csv.formatFile(sess, [r1, r3]);
assertEqual(
  file.charCodeAt(0), 0xFEFF, 'file starts with BOM'
);
assertEqual(
  file.endsWith('\r\n'), true, 'file ends with CRLF'
);
assertEqual(
  file.split('\r\n').length, 4, 'file: header + 2 rows + trailing empty'
);

const t0 = new Date(2026, 4, 11, 9, 7);  // 09:07 local
assertEqual(
  csv.filename({ data: '2026-05-11', compresa: 'Serra' }, t0),
  'ipso_Serra_2026-05-11_0907.csv',
  'filename final'
);
assertEqual(
  csv.filename({ data: '2026-05-11', compresa: 'Serra' }, t0, 'backup', 20),
  'ipso_Serra_2026-05-11_0907_backup_20.csv',
  'filename backup'
);
assertEqual(
  csv.filename({ data: '2026-05-12', compresa: 'Capistrano', catastrofata: true }, t0),
  'ipso_Capistrano_catastrofate_2026-05-12_0907.csv',
  'filename catastrofate final'
);
assertEqual(
  csv.filename({ data: '2026-05-12', compresa: 'Serra', catastrofata: true }, t0, 'backup', 40),
  'ipso_Serra_catastrofate_2026-05-12_0907_backup_40.csv',
  'filename catastrofate backup'
);
assertEqual(
  csv.filename({ data: '2026-05-11', compresa: 'San Giorgio' }, t0),
  'ipso_San_Giorgio_2026-05-11_0907.csv',
  'filename sanitises spaces in compresa'
);

// ---------------------------------------------------------------------------
// ipso.js
// ---------------------------------------------------------------------------

console.log('ipso.js');

const ipsTable = {
  Serra: {
    Abete:       { a: 12.2572, b: -18.2468 },
    'Pino Nero': { a: 11.0961, b: -11.6745 },
  },
  Capistrano: {
    Abete:   { a: 7.0306, b: -4.2563 },
    Douglas: { a: 15.7207, b: -30.6302 },
  },
};

assertEqual(
  ipso.lookup(ipsTable, 'Serra', 'Abete'),
  { a: 12.2572, b: -18.2468 },
  'lookup: hit'
);
assertEqual(ipso.lookup(ipsTable, 'Serra', 'Ontano'), null, 'lookup: no specie');
assertEqual(ipso.lookup(ipsTable, 'Capistrano', 'Faggio'), null, 'lookup: gap');
assertEqual(ipso.lookup(ipsTable, 'Nowhere', 'Abete'), null, 'lookup: no compresa');
assertEqual(ipso.lookup(null, 'Serra', 'Abete'), null, 'lookup: null table');

// Serra/Abete on D=42cm: 12.2572*ln(42) - 18.2468 = 45.81 - 18.25 = 27.56 → 28
assertEqual(
  ipso.computeH(ipsTable.Serra.Abete, 42), 28,
  'computeH: Serra Abete D=42'
);
// Capistrano/Abete on D=42: 7.0306*ln(42) - 4.2563 = 26.28 - 4.26 = 22.01 → 22
assertEqual(
  ipso.computeH(ipsTable.Capistrano.Abete, 42), 22,
  'computeH: Capistrano Abete D=42'
);
assertEqual(ipso.computeH(null, 42), null, 'computeH: null eq');
assertEqual(ipso.computeH(ipsTable.Serra.Abete, 0), null, 'computeH: D=0');
assertEqual(ipso.computeH(ipsTable.Serra.Abete, -5), null, 'computeH: D<0');
assertEqual(ipso.computeH(ipsTable.Serra.Abete, null), null, 'computeH: D null');
// Very small D may produce negative h; should return null.
assertEqual(ipso.computeH({ a: 1, b: -100 }, 1), null, 'computeH: negative h -> null');

// ---------------------------------------------------------------------------
// session.js
// ---------------------------------------------------------------------------

console.log('session.js');

assertEqual(session.nextSeq([]), 1, 'nextSeq: empty');
assertEqual(session.nextSeq([1, 2, 3]), 4, 'nextSeq: dense');
assertEqual(session.nextSeq([3, 1, 2]), 4, 'nextSeq: unsorted');
assertEqual(session.nextSeq([5, null, 'x']), 6, 'nextSeq: ignores non-numbers');

assertEqual(
  session.validateTree({ specie: 'Abete', d_cm: 42, h_m: 24 }),
  [],
  'validate: valid'
);
assertEqual(
  session.validateTree({}),
  ['specie', 'd_cm', 'h_m'],
  'validate: empty'
);
assertEqual(
  session.validateTree({ specie: 'Abete', d_cm: 0, h_m: 24 }),
  ['d_cm'],
  'validate: D=0'
);
assertEqual(
  session.validateTree({ specie: 'Abete', d_cm: 42, h_m: 100 }),
  ['h_m'],
  'validate: h=100'
);
assertEqual(
  session.validateTree({ specie: '', d_cm: 42, h_m: 20 }),
  ['specie'],
  'validate: empty specie'
);
assertEqual(
  session.validateTree({ specie: 'Abete', d_cm: 42.5, h_m: 20 }),
  ['d_cm'],
  'validate: D not integer'
);
assertEqual(session.validateTree(null), ['empty'], 'validate: null');

assertEqual(
  session.summarizePill({ specie: 'Abete', d_cm: 42, h_m: 24 }),
  'Abete, D=42, h=24',
  'pill: no numero -> no n. prefix'
);
assertEqual(
  session.summarizePill({ specie: 'Pino Nero', d_cm: 30, h_m: null }),
  'Pino Nero, D=30, h=—',
  'pill: no h, no numero'
);
assertEqual(
  session.summarizePill({ specie: 'Abete', d_cm: 42, h_m: 24, numero: 7 }),
  'n. 7 · Abete, D=42, h=24',
  'pill: numero prepended'
);
assertEqual(
  session.summarizePill({ specie: 'Abete', d_cm: 42, h_m: 24, numero: null }),
  'Abete, D=42, h=24',
  'pill: explicit null numero -> no n. prefix'
);
assertEqual(session.summarizePill(null), 'nessun albero', 'pill: null');

assertEqual(session.shouldBackup(20), true, 'backup: 20');
assertEqual(session.shouldBackup(40), true, 'backup: 40');
assertEqual(session.shouldBackup(19), false, 'backup: 19');
assertEqual(session.shouldBackup(0), false, 'backup: 0');
assertEqual(session.shouldBackup(null), false, 'backup: null');

assertEqual(session.NUMERO_BLANK_D_THRESHOLD, 17, 'numero blanking threshold');

assertEqual(session.nextNumeroDefault([]), null, 'nextNumero: empty');
assertEqual(session.nextNumeroDefault(null), null, 'nextNumero: null');
assertEqual(
  session.nextNumeroDefault([{ numero: null }, { numero: null }]),
  null,
  'nextNumero: all blanks'
);
assertEqual(
  session.nextNumeroDefault([{ numero: 5 }, { numero: 7 }, { numero: 3 }]),
  8,
  'nextNumero: takes max'
);
assertEqual(
  session.nextNumeroDefault([{ numero: 5 }, { numero: null }, { numero: 8 }]),
  9,
  'nextNumero: ignores blanks'
);
assertEqual(
  session.nextNumeroDefault([{ numero: 0 }]),
  1,
  'nextNumero: zero counts'
);
assertEqual(
  session.nextNumeroDefault([{}, { specie: 'Abete' }]),
  null,
  'nextNumero: rows without numero field'
);

// ---------------------------------------------------------------------------
// geo.js — vendored from abies. The geometry tests proper live in
// abies/test/test_geo.mjs; here we only smoke-test that the vendor
// pipeline produced a loadable CommonJS module with the names the rest
// of the ipso code expects.
// ---------------------------------------------------------------------------

console.log('\ngeo.js (vendor smoke)');

for (const name of ['pointInPolygon', 'findContainingParcel', 'parcelLabel',
                    'featureBbox', 'buildBboxIndex',
                    'distanceToBoundaryMeters']) {
  assertEqual(typeof geo[name], 'function', `geo.${name} is exported`);
}

// ---------------------------------------------------------------------------
// parcel-locator.js — hysteresis + sticky-override state machines
// ---------------------------------------------------------------------------

console.log('\nparcel-locator.js');

// Two unit-square parcels A and B, side by side at the equator.
// A spans lng [0,1], B spans lng [1,2]; both span lat [0,1]. Adjacent
// along lng=1 so a fix near the boundary is a meaningful edge case.
function makeFeatures() {
  const A = { properties: { layer: 'X', name: 'X-A' },
              geometry: { type: 'Polygon',
                          coordinates: [[[0,0],[1,0],[1,1],[0,1],[0,0]]] } };
  const B = { properties: { layer: 'X', name: 'X-B' },
              geometry: { type: 'Polygon',
                          coordinates: [[[1,0],[2,0],[2,1],[1,1],[1,0]]] } };
  geo.buildBboxIndex([A, B]);
  return { A, B, features: [A, B] };
}

// --- Hysteresis -----------------------------------------------------------

{
  const { A, B, features } = makeFeatures();
  const loc = locator.createLocator(features);
  const commits = [];
  loc.subscribe(f => commits.push(f));

  assertEqual(loc.getCommitted(), null, 'locator: initial committed=null');

  // Two fixes inside A — count=2, not yet committed.
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 10 });
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 10 });
  assertEqual(loc.getCommitted(), null, 'locator: 2 fixes is not enough');
  assertEqual(commits.length, 0, 'locator: no callback before commit');

  // 3rd fix inside A: dist to A boundary ~55 km, acc=10, commit.
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 10 });
  assertEqual(loc.getCommitted() === A, true, 'locator: committed=A after 3 fixes');
  assertEqual(commits.length, 1, 'locator: callback fired once');
  assertEqual(commits[0] === A, true, 'locator: callback got feature A');

  // A fix still inside A — no transition.
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 10 });
  assertEqual(commits.length, 1, 'locator: same parcel does not refire');

  // Walk into B; 2 fixes not enough yet.
  loc.onFix({ lat: 0.5, lng: 1.5, acc: 10 });
  loc.onFix({ lat: 0.5, lng: 1.5, acc: 10 });
  assertEqual(loc.getCommitted() === A, true, 'locator: 2 inside B keeps A');

  // 3rd fix inside B commits the transition.
  loc.onFix({ lat: 0.5, lng: 1.5, acc: 10 });
  assertEqual(loc.getCommitted() === B, true, 'locator: committed=B');
  assertEqual(commits.length, 2, 'locator: callback fired a second time');
  assertEqual(commits[1] === B, true, 'locator: second callback got B');
}

// Accuracy gate near a boundary: 3 fixes inside the new candidate but
// with poor accuracy → no commit.
{
  const { A, B, features } = makeFeatures();
  const loc = locator.createLocator(features);

  // Commit A first (using a far-from-boundary point with tight acc).
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 5 });
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 5 });
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 5 });
  assertEqual(loc.getCommitted() === A, true, 'locator (acc gate): pre-committed A');

  // Hover just inside B at lng=1.001 — dist to B boundary ≈ 111 m at
  // the equator. Use acc=500 (worse) → no commit.
  loc.onFix({ lat: 0.5, lng: 1.001, acc: 500 });
  loc.onFix({ lat: 0.5, lng: 1.001, acc: 500 });
  loc.onFix({ lat: 0.5, lng: 1.001, acc: 500 });
  assertEqual(loc.getCommitted() === A, true,
              'locator: bad accuracy near boundary keeps A');

  // Same point with tight accuracy → commit.
  loc.onFix({ lat: 0.5, lng: 1.001, acc: 50 });
  assertEqual(loc.getCommitted() === B, true,
              'locator: 4th fix with tight acc commits to B');
}

// Transition to "outside" (null) when leaving the previous parcel.
{
  const { A, features } = makeFeatures();
  const loc = locator.createLocator(features);
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 5 });
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 5 });
  loc.onFix({ lat: 0.5, lng: 0.5, acc: 5 });
  assertEqual(loc.getCommitted() === A, true, 'locator (exit): pre-committed A');

  // Three fixes outside both polygons. Boundary check uses A's polygon.
  loc.onFix({ lat: 5, lng: 5, acc: 10 });
  loc.onFix({ lat: 5, lng: 5, acc: 10 });
  loc.onFix({ lat: 5, lng: 5, acc: 10 });
  assertEqual(loc.getCommitted(), null, 'locator: committed=null after exit');
}

// --- Sticky override ------------------------------------------------------

{
  const ov = locator.createOverride();
  assertEqual(ov.getMode(), 'auto', 'override: initial mode=auto');
  assertEqual(ov.resolve('A'), 'A', 'override: auto resolves to autoName');
  assertEqual(ov.isMismatch('A'), false, 'override: auto never mismatches');

  // Outside / no fix: autoName is ''; auto mode still no mismatch.
  assertEqual(ov.resolve(''), '', 'override: auto with empty auto -> empty');
  assertEqual(ov.isMismatch(''), false, 'override: auto+empty: not red');

  // Manual selection of B while auto says A → mismatch, sticky.
  ov.setManual('B');
  assertEqual(ov.getMode(), 'manual', 'override: setManual -> manual');
  assertEqual(ov.resolve('A'), 'B', 'override: manual overrides auto');
  assertEqual(ov.isMismatch('A'), true, 'override: manual!=auto -> mismatch');

  // Manual matches auto → no mismatch.
  ov.setManual('A');
  assertEqual(ov.isMismatch('A'), false, 'override: manual==auto -> no mismatch');

  // Manual while auto is '' (outside) → mismatch.
  ov.setManual('A');
  assertEqual(ov.isMismatch(''), true,
              'override: manual while auto empty -> mismatch');

  // Return to auto resets mode but keeps manual value remembered.
  ov.setAuto();
  assertEqual(ov.getMode(), 'auto', 'override: setAuto -> auto');
  assertEqual(ov.resolve('C'), 'C', 'override: auto resumes tracking');
  assertEqual(ov.isMismatch('C'), false, 'override: auto: never red');
  assertEqual(ov.getManual(), 'A', 'override: setAuto preserves manual value');
}

// ---------------------------------------------------------------------------
// store.js — pure helpers (the IndexedDB code is exercised only in the
// browser; here we lock the status-set contract that the resume flow relies
// on).
// ---------------------------------------------------------------------------

console.log('\nstore.js (pure helpers)');

const { Store } = require('../build/store.js');

assertEqual(Store.SCHEMA_VERSION, 5, 'store: SCHEMA_VERSION bumped to 5');

assertEqual(Store.STATUS_OPEN, 'open', 'store: STATUS_OPEN constant');
assertEqual(Store.STATUS_PENDING_UPLOAD, 'pending_upload',
            'store: STATUS_PENDING_UPLOAD constant');
assertEqual(Store.STATUS_EXPORTED, 'exported', 'store: STATUS_EXPORTED constant');
assertEqual(Store.STATUS_ABANDONED, 'abandoned',
            'store: STATUS_ABANDONED constant');

assertEqual(Store.isResumableStatus(Store.STATUS_OPEN), true,
            'store: OPEN is resumable');
assertEqual(Store.isResumableStatus(Store.STATUS_PENDING_UPLOAD), true,
            'store: PENDING_UPLOAD is resumable');
assertEqual(Store.isResumableStatus(Store.STATUS_EXPORTED), false,
            'store: EXPORTED is not resumable');
assertEqual(Store.isResumableStatus(Store.STATUS_ABANDONED), false,
            'store: ABANDONED is not resumable');
assertEqual(Store.isResumableStatus(null), false,
            'store: null is not resumable');
assertEqual(Store.isResumableStatus('nonsense'), false,
            'store: unknown status is not resumable');

// ---------------------------------------------------------------------------
// upload.js — pure helpers (backoff schedule, response classifier). The
// network-touching uploadSession() is exercised via a mocked globalThis.fetch
// further below.
// ---------------------------------------------------------------------------

console.log('\nupload.js (pure helpers)');

const upload = require('../build/upload.js');

// Backoff schedule: 2,4,8,16,30,30,30,...
assertEqual(upload.backoffMs(1), 2000, 'backoff: attempt 1');
assertEqual(upload.backoffMs(2), 4000, 'backoff: attempt 2');
assertEqual(upload.backoffMs(3), 8000, 'backoff: attempt 3');
assertEqual(upload.backoffMs(4), 16000, 'backoff: attempt 4');
assertEqual(upload.backoffMs(5), 30000, 'backoff: attempt 5 (cap)');
assertEqual(upload.backoffMs(6), 30000, 'backoff: attempt 6 (capped)');
assertEqual(upload.backoffMs(99), 30000, 'backoff: attempt 99 (capped)');
assertEqual(upload.backoffMs(0), 0, 'backoff: 0 attempts = 0 ms');

// Error classification
assertEqual(upload.classifyHttp(200), 'ok', 'classify: 200');
assertEqual(upload.classifyHttp(401), 'hard:auth', 'classify: 401');
assertEqual(upload.classifyHttp(409), 'hard:conflict', 'classify: 409');
assertEqual(upload.classifyHttp(413), 'hard:too_large', 'classify: 413');
assertEqual(upload.classifyHttp(422), 'hard:invalid_csv', 'classify: 422');
assertEqual(upload.classifyHttp(429), 'soft:rate_limited', 'classify: 429');
assertEqual(upload.classifyHttp(500), 'soft:server', 'classify: 500');
assertEqual(upload.classifyHttp(502), 'soft:server', 'classify: 502');
assertEqual(upload.classifyHttp(503), 'soft:server', 'classify: 503');
assertEqual(upload.classifyHttp(599), 'soft:server', 'classify: 599');
assertEqual(upload.classifyHttp(418), 'hard:invalid_csv',
            'classify: unknown 4xx -> hard:invalid_csv');

// ---------------------------------------------------------------------------
// upload.js — uploadSession with mocked globalThis.fetch.
// ---------------------------------------------------------------------------

console.log('\nupload.uploadSession (mocked fetch)');

async function withMockFetch(handler, fn) {
  const original = globalThis.fetch;
  globalThis.fetch = handler;
  try { return await fn(); } finally { globalThis.fetch = original; }
}

function mockResponse(status, payload) {
  return Promise.resolve({
    status,
    json: () => Promise.resolve(payload || {}),
  });
}

(async () => {
  // Happy path
  await withMockFetch(
    (url, init) => mockResponse(200, { ok: true, duplicate: false, stored_as: 'X.csv' }),
    async () => {
      const r = await upload.uploadSession({
        base: 'https://h', token: 't', schemaVersion: 5,
        sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        csvText: 'data',
      });
      assertEqual(r, { duplicate: false, storedAs: 'X.csv' },
                  'uploadSession: 200 returns duplicate=false');
    }
  );

  // 200 duplicate
  await withMockFetch(
    () => mockResponse(200, { ok: true, duplicate: true, stored_as: 'X.csv' }),
    async () => {
      const r = await upload.uploadSession({
        base: 'https://h', token: 't', schemaVersion: 5,
        sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        csvText: 'data',
      });
      assertEqual(r, { duplicate: true, storedAs: 'X.csv' },
                  'uploadSession: 200 returns duplicate=true');
    }
  );

  // 401 -> UploadError 'hard:auth'
  await withMockFetch(
    () => mockResponse(401, { ok: false, error: 'auth' }),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession 401: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'hard:auth', 'uploadSession 401 -> hard:auth');
      }
    }
  );

  // 409 -> UploadError 'hard:conflict'
  await withMockFetch(
    () => mockResponse(409, { ok: false, error: 'conflict' }),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession 409: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'hard:conflict', 'uploadSession 409 -> hard:conflict');
      }
    }
  );

  // 503 -> UploadError 'soft:server'
  await withMockFetch(
    () => mockResponse(503, { ok: false, error: 'server' }),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession 503: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'soft:server', 'uploadSession 503 -> soft:server');
      }
    }
  );

  // Network error -> 'soft:network'
  await withMockFetch(
    () => Promise.reject(new TypeError('network down')),
    async () => {
      try {
        await upload.uploadSession({
          base: 'https://h', token: 't', schemaVersion: 5,
          sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', csvText: 'd',
        });
        failed++; console.error('FAIL uploadSession network: expected throw');
      } catch (e) {
        assertEqual(e.klass, 'soft:network',
                    'uploadSession network failure -> soft:network');
      }
    }
  );

  // Headers + body sent correctly
  let captured;
  await withMockFetch(
    (url, init) => {
      captured = { url, init };
      return mockResponse(200, { ok: true, duplicate: false, stored_as: 'X.csv' });
    },
    async () => {
      await upload.uploadSession({
        base: 'https://example.invalid', token: 'tok',
        schemaVersion: 5,
        sessionId: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
        csvText: 'BODY',
      });
    }
  );
  assertEqual(captured.url, 'https://example.invalid/upload',
              'uploadSession: url base + /upload');
  assertEqual(captured.init.method, 'POST', 'uploadSession: method=POST');
  assertEqual(captured.init.headers.Authorization, 'Bearer tok',
              'uploadSession: Authorization header');
  assertEqual(captured.init.headers['Content-Type'], 'text/csv; charset=utf-8',
              'uploadSession: Content-Type header');
  assertEqual(captured.init.headers['X-Ipso-Session-Id'],
              'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
              'uploadSession: X-Ipso-Session-Id header');
  assertEqual(captured.init.headers['X-Ipso-Schema-Version'], '5',
              'uploadSession: X-Ipso-Schema-Version header');
  assertEqual(captured.init.body, 'BODY', 'uploadSession: body=csvText');

  // -------------------------------------------------------------------------
  // Summary (inside the IIFE so async assertions are counted before exit).
  // -------------------------------------------------------------------------

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
})();
