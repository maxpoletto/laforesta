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

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} store tests passed`);
