// Tests for ipso pure-logic modules: csv.js, ipso.js, session.js.
// Run with: node tests.js
'use strict';

const csv = require('./csv.js');
const ipso = require('./ipso.js');
const session = require('./session.js');

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
  'Data;Compresa;Particella;Catastrofata;Specie;D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore',
  'header literal'
);

const sess = {
  data: '2026-05-11', compresa: 'Serra', particella: '1', operatore: 'Mario Rossi',
};
const r1 = {
  specie: 'Abete', d_cm: 42, h_m: 24, h_measured: 0,
  lat: 38.4253101, lng: 16.1204400, acc_m: 7,
};
assertEqual(
  csv.formatRow(r1, sess),
  '11/05/2026;Serra;1;0;Abete;42;24;0;38,425310;16,120440;7;Mario Rossi',
  'formatRow happy path'
);

const r2 = {
  specie: 'Faggio', d_cm: 30, h_m: 18, h_measured: 1,
  lat: null, lng: null, acc_m: null,
};
assertEqual(
  csv.formatRow(r2, sess),
  '11/05/2026;Serra;1;0;Faggio;30;18;1;;;;Mario Rossi',
  'formatRow no GPS'
);

// Missing operatore on the session row -> empty string in the column.
const sessNoOp = { data: '2026-05-11', compresa: 'Serra', particella: '1' };
assertEqual(
  csv.formatRow(r1, sessNoOp),
  '11/05/2026;Serra;1;0;Abete;42;24;0;38,425310;16,120440;7;',
  'formatRow no operatore'
);

// Catastrofate session: Particella column empty, Catastrofata = 1.
const sessCat = {
  data: '2026-05-12', compresa: 'Capistrano', particella: '',
  catastrofata: true, operatore: 'Anna Bianchi',
};
assertEqual(
  csv.formatRow(r1, sessCat),
  '12/05/2026;Capistrano;;1;Abete;42;24;0;38,425310;16,120440;7;Anna Bianchi',
  'formatRow catastrofata'
);

const r3 = {
  specie: 'Pino Nero', d_cm: 50, h_m: 26, h_measured: 0,
  lat: 38.4, lng: 16.1, acc_m: 12,
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
  csv.filename({ data: '2026-05-11', compresa: 'Serra', particella: '1' }, t0),
  'ipso_Serra_1_2026-05-11_0907.csv',
  'filename final'
);
assertEqual(
  csv.filename({ data: '2026-05-11', compresa: 'Serra', particella: '2a' }, t0, 'backup', 20),
  'ipso_Serra_2a_2026-05-11_0907_backup_20.csv',
  'filename backup'
);
assertEqual(
  csv.filename({ data: '2026-05-12', compresa: 'Capistrano', particella: '', catastrofata: true }, t0),
  'ipso_Capistrano_catastrofate_2026-05-12_0907.csv',
  'filename catastrofate final'
);
assertEqual(
  csv.filename({ data: '2026-05-12', compresa: 'Serra', particella: '', catastrofata: true }, t0, 'backup', 40),
  'ipso_Serra_catastrofate_2026-05-12_0907_backup_40.csv',
  'filename catastrofate backup'
);
assertEqual(
  csv.filename({ data: '2026-05-11', compresa: 'San Giorgio', particella: '1/b' }, t0),
  'ipso_San_Giorgio_1_b_2026-05-11_0907.csv',
  'filename sanitises spaces and slashes'
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
  'pill: full'
);
assertEqual(
  session.summarizePill({ specie: 'Pino Nero', d_cm: 30, h_m: null }),
  'Pino Nero, D=30, h=—',
  'pill: no h'
);
assertEqual(session.summarizePill(null), 'nessun albero', 'pill: null');

assertEqual(session.shouldBackup(20), true, 'backup: 20');
assertEqual(session.shouldBackup(40), true, 'backup: 40');
assertEqual(session.shouldBackup(19), false, 'backup: 19');
assertEqual(session.shouldBackup(0), false, 'backup: 0');
assertEqual(session.shouldBackup(null), false, 'backup: null');

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
