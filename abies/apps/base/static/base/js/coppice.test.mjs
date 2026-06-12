import { COL_COPPICE } from './constants.js';
import { recordIsCoppice } from './coppice.js';
import * as S from './strings.js';

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  if (actual === expected) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${expected}`);
    console.error(`       got: ${actual}`);
  }
}

console.log('coppice.js');

assertEqual(recordIsCoppice([true], [COL_COPPICE]), true,
            'stable Coppice=true');
assertEqual(recordIsCoppice([false], [COL_COPPICE]), false,
            'stable Coppice=false');
assertEqual(recordIsCoppice([S.TYPE_COPPICE], [S.COL_TYPE]), true,
            'display Tipo fallback: current label');
assertEqual(recordIsCoppice(['ceduo'], [S.COL_TYPE]), true,
            'display Tipo fallback: old lowercase label');
assertEqual(recordIsCoppice([S.TYPE_HIGHFOREST], [S.COL_TYPE]), false,
            'display Tipo fallback: high forest');
assertEqual(recordIsCoppice(['fustaia'], [S.COL_TYPE]), false,
            'display Tipo fallback: old lowercase high forest');
assertEqual(recordIsCoppice([], []), false,
            'missing columns default false');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
