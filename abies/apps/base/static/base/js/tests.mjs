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

import { createRequire } from 'module';
import {
  makeNumberParser, parseDecimal, fmtDecimal, fmtCoord, fmtMass,
} from './format.js';
import { matchesSearch } from './table.js';

let pass = 0;
const failures = [];
const require = createRequire(import.meta.url);
const SortableTable = require('../../vendor/sortable-table/sortable-table.js');

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
eqStr(fmtMass(1234.5), '1235 q', 'fmtMass whole quintals');

// --- matchesSearch: haystack is the FORMATTED (displayed) text (§6) ----------
// Columns carry the same formatters the table renders with; in node format.js
// defaults to the it locale, so the number column displays "3,14".
const searchCols = [
  { formatter: v => fmtDecimal(v, 2) },  // 3.14 → "3,14"
  {},                                     // text column, no formatter
  { hidden: true },                       // hidden → excluded from the haystack
];
const searchRow = [3.14, 'Abete', 999];
check(matchesSearch(searchRow, ['3,14'], searchCols), 'search matches formatted comma decimal');
check(!matchesSearch(searchRow, ['3.14'], searchCols), 'raw dot does not match formatted cell');
check(matchesSearch(searchRow, ['abete'], searchCols), 'search matches a text term');
check(!matchesSearch(searchRow, ['999'], searchCols), 'hidden column excluded from search');
check(matchesSearch(searchRow, ['3,14', 'abete'], searchCols), 'ordered multi-term match');
check(matchesSearch(searchRow, ['3.14'], null), 'no columns → raw fallback match');

// --- SortableTable: HTML escaping is default, trustedHTML is opt-in ---------
const fakeContainer = {
  innerHTML: '',
  classList: { contains: () => false, add() {}, remove() {} },
  querySelectorAll: () => [],
  querySelector: () => null,
};
new SortableTable({
  container: fakeContainer,
  data: [['<img src=x onerror=alert(1)>', '<strong>ok</strong>']],
  columns: [
    { key: 'unsafe', label: 'Unsafe <em>label</em>' },
    { key: 'trusted', label: 'Trusted', formatter: v => v, trustedHTML: true },
  ],
  showPagination: false,
  allowSorting: false,
});
check(fakeContainer.innerHTML.includes('&lt;img src=x onerror=alert(1)&gt;'),
      'sortable-table escapes untrusted cell HTML');
check(!fakeContainer.innerHTML.includes('<img'),
      'sortable-table does not emit untrusted HTML elements');
check(fakeContainer.innerHTML.includes('Unsafe &lt;em&gt;label&lt;/em&gt;'),
      'sortable-table escapes header labels');
check(fakeContainer.innerHTML.includes('<strong>ok</strong>'),
      'sortable-table trustedHTML column renders markup');

// --- Report -----------------------------------------------------------------
console.log(`${pass} passed, ${failures.length} failed`);
for (const f of failures) console.error('  FAIL ' + f);
process.exit(failures.length ? 1 : 0);
