/**
 * Tests for the shared Prelievi column definitions (prelievi-columns.js),
 * used by BOTH the Prelievi page and the Piano-di-taglio Prelievi sub-table.
 * The whole point is that they format identically — especially the dynamic
 * species/tractor quintal columns (blank zero, one decimal), which had
 * diverged.  Run: node prelievi-columns.test.mjs  (also via `make test-js`).
 */
import { buildPrelieviColumnDefs } from './prelievi-columns.js';
import * as S from './strings.js';
import { VERSION } from './constants.js';

let pass = 0;
const failures = [];
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) pass++; else failures.push(`${msg}: expected ${e}, got ${a}`);
}

// row_id, version, a static quantity col, the two dynamic kinds (species =
// one word, tractor = has a space), a hidden "%" pre-fill col, the hidden
// cantiere link.
const SPECIES = 'Pino';
const TRACTOR = 'John Deere';
const defs = buildPrelieviColumnDefs([
  'row_id', VERSION, S.COL_QUINTALS, SPECIES, TRACTOR,
  `${SPECIES} %`, S.COL_WORKSITE,
]);

// row_id is the table key, never a rendered column.
eq('row_id' in defs, false, 'row_id is not a column');

// Bookkeeping / pre-fill columns are hidden.
eq(defs[VERSION]?.hidden, true, 'version hidden');
eq(defs[S.COL_WORKSITE]?.hidden, true, 'cantiere hidden');
eq(defs[`${SPECIES} %`]?.hidden, true, 'percent column hidden');

// The bug this guards against: dynamic species/tractor columns must blank
// zeros and show one decimal (Italian comma) — not raw "0" / "1234".
const sp = defs[SPECIES];
eq(sp.formatter(0), '', 'species: zero is blank, not "0"');
eq(sp.formatter(1234), '1234,0', 'species: 1234 -> "1234,0", not "1234"');
eq(sp.width, '90px', 'species column width');

const tr = defs[TRACTOR];
eq(tr.formatter(0), '', 'tractor: zero is blank');
eq(tr.width, '100px', 'tractor column width (wider for manufacturer+model)');

// Static quantity column keeps its own one-decimal format.
eq(defs[S.COL_QUINTALS].formatter(1234), '1234,0', 'quintals one decimal');

console.log(`${pass} passed, ${failures.length} failed`);
for (const f of failures) console.error('  FAIL ' + f);
process.exit(failures.length ? 1 : 0);
