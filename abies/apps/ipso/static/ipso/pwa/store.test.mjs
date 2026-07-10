import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { Store } = require('./store.js');

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

check(
  Store.nextSeqAfterRows([]) === 1,
  'nextSeqAfterRows starts fresh sessions at one',
);
check(
  Store.nextSeqAfterRows([{ seq: 1 }, { seq: 3 }]) === 4,
  'nextSeqAfterRows continues from the highest existing sequence',
);
check(
  Store.nextSeqAfterRows([{ seq: 1 }, { seq: 3 }, { seq: 2 }]) === 4,
  'nextSeqAfterRows is independent of row order',
);
check(
  Store.nextSeqAfterRows([{ seq: 1 }, { seq: null }, {}]) === 2,
  'nextSeqAfterRows ignores missing sequence values',
);

check(
  Store.nextNumberAfterSave(null, 5) === 6,
  'nextNumberAfterSave initializes the operator counter',
);
check(
  Store.nextNumberAfterSave(10, 5) === 10,
  'nextNumberAfterSave never moves the operator counter backwards',
);
check(
  Store.nextNumberAfterSave(10, 12) === 13,
  'nextNumberAfterSave advances beyond a new maximum',
);
check(
  Store.nextNumberAfterSave(10, null) === 10,
  'nextNumberAfterSave ignores blank numbers',
);
check(
  Store.nextNumberAfterDelete(111, [
    { numero: 101 }, { numero: 103 }, { numero: 102 },
  ]) === 104,
  'nextNumberAfterDelete follows the highest remaining number',
);
check(
  Store.nextNumberAfterDelete(8, [
    { numero: null }, {}, null, { numero: '9' },
  ]) === 8,
  'nextNumberAfterDelete preserves cross-session state without numbered rows',
);
check(
  Store.nextNumberAfterDelete(null, [
    { numero: 4 }, { numero: null }, { numero: 2 },
  ]) === 5,
  'nextNumberAfterDelete ignores blanks and row order',
);

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} store tests passed`);
