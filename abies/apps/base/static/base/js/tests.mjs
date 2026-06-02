/**
 * Unit tests for the shared locale-aware number formatters/parsers.
 * Run with: node tests.mjs   (ES modules; node 18+ with full ICU).
 * Also run as part of `make test-js`.
 *
 * The parser maps the active locale's decimal separator to '.' and parses;
 * a literal '.' is always accepted (lenient), so an Italian user may type
 * either "3,14" or "3.14".  In a dot-decimal locale a comma is not a decimal
 * and is rejected.  Thousands separators are out of scope.  format.js reads
 * the active locale from `document` at import; in node there is no document,
 * so it defaults to 'it' — locale-specific parsing is tested via
 * makeNumberParser(locale).
 */

import {
  makeNumberParser, parseDecimal, fmtDecimal, fmtCoord,
} from './format.js';

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass++; else failures.push(msg);
}

function eqNum(actual, expected, msg) {
  check(typeof actual === 'number' && Math.abs(actual - expected) < 1e-9,
        `${msg}: expected ${expected}, got ${actual}`);
}

function eqStr(actual, expected, msg) {
  check(actual === expected, `${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

// --- Italian: comma is the decimal separator, dot accepted leniently --------
const it = makeNumberParser('it');
eqNum(it('3,14'), 3.14, 'it native comma');
eqNum(it('3.14'), 3.14, 'it lenient dot');
eqNum(it('38,12346'), 38.12346, 'it coord');
eqNum(it('-16,98765'), -16.98765, 'it negative coord');
eqNum(it('9,50'), 9.5, 'it trailing zero');
eqNum(it('5'), 5, 'it bare integer');

// --- English: dot is the decimal separator, comma is NOT a decimal ----------
const en = makeNumberParser('en');
eqNum(en('3.14'), 3.14, 'en native dot');
eqNum(en('-16.98765'), -16.98765, 'en negative');
check(Number.isNaN(en('3,14')), 'en rejects comma decimal');

// --- German: comma decimal, like Italian ------------------------------------
const de = makeNumberParser('de');
eqNum(de('3,14'), 3.14, 'de comma');

// --- Blank / invalid → NaN --------------------------------------------------
check(Number.isNaN(parseDecimal('')), 'empty → NaN');
check(Number.isNaN(parseDecimal('   ')), 'blank → NaN');
check(Number.isNaN(parseDecimal(null)), 'null → NaN');
check(Number.isNaN(parseDecimal(undefined)), 'undefined → NaN');
check(Number.isNaN(it('abc')), 'garbage → NaN');
check(Number.isNaN(it('1.2.3')), 'malformed → NaN');

// --- Module-locale parseDecimal (it in node) --------------------------------
eqNum(parseDecimal('38,12346'), 38.12346, 'parseDecimal module-locale');

// --- Round-trip: parse(format(x)) === x (the data-integrity guarantee) ------
for (const [x, n] of [[38.12346, 5], [9.5, 2], [-16.987, 3], [0.1, 1], [0, 5]]) {
  eqNum(parseDecimal(fmtDecimal(x, n)), x, `round-trip ${x}@${n}dp`);
}

// --- Display: locale separator, no thousands grouping -----------------------
eqStr(fmtCoord(1234.5), '1234,50000', 'fmtCoord no grouping');
eqStr(fmtDecimal(1234.5, 2), '1234,50', 'fmtDecimal no grouping');
eqStr(fmtDecimal(3.14, 2), '3,14', 'fmtDecimal it comma');

// --- Report -----------------------------------------------------------------
console.log(`${pass} passed, ${failures.length} failed`);
for (const f of failures) console.error('  FAIL ' + f);
process.exit(failures.length ? 1 : 0);
