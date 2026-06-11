import { COLUMNS } from './constants.js';
import { columnMap, toNumber } from './digests.js';

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

console.log('digests.js');

assertEqual(columnMap(['row_id', 'Nome']), { row_id: 0, Nome: 1 },
            'columnMap: columns array');
assertEqual(columnMap({ [COLUMNS]: ['a', 'b'] }), { a: 0, b: 1 },
            'columnMap: digest object');
assertEqual(columnMap(null), {}, 'columnMap: null-safe');
assertEqual(toNumber('12.5'), 12.5, 'toNumber: decimal string');
assertEqual(toNumber(''), null, 'toNumber: blank defaults null');
assertEqual(toNumber('', 0), 0, 'toNumber: blank explicit fallback');
assertEqual(toNumber('abc', 7), 7, 'toNumber: invalid explicit fallback');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
