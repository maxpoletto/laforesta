import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const session = require('./session.js');

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

function validTree(overrides = {}) {
  return Object.assign({
    specie: 'Abete',
    d_cm: 42,
    h_m: 22,
    numero: 1,
    particella: '7',
    sample_area_id: 3,
    lat: 38.5,
    lon: 16.3,
    acc_m: 12,
  }, overrides);
}

check(
  session.validateTree(validTree(), { parcelRequired: true }).length === 0,
  'parcelRequired accepts a selected parcel',
);
check(
  session.validateTree(validTree({ particella: '' }), { parcelRequired: true }).includes('particella'),
  'parcelRequired rejects blank parcel',
);
check(
  session.validateTree(validTree({ particella: '   ' }), { parcelRequired: true }).includes('particella'),
  'parcelRequired rejects whitespace parcel',
);
check(
  !session.validateTree(validTree({ particella: '' }), {}).includes('particella'),
  'blank parcel remains allowed unless the mode requires it',
);
check(
  session.validateTree(validTree({ sample_area_id: null }), { sampleAreaRequired: true }).includes('sample_area_id'),
  'sampleAreaRequired still rejects missing sample area',
);
check(
  session.validateTree(validTree({ numero: 0 }), { numberRequired: true }).includes('numero'),
  'numberRequired rejects zero',
);
check(
  session.validateTree(validTree({ numero: 0 }), {}).includes('numero'),
  'a recorded zero number is rejected even when the mode does not require one',
);
check(
  session.validateTree(validTree({ numero: -3 }), {}).includes('numero'),
  'a recorded negative number is rejected even when the mode does not require one',
);
check(
  !session.validateTree(validTree({ numero: null }), {}).includes('numero'),
  'blank number remains allowed unless the mode requires a number',
);
check(
  session.validateTree(validTree({ lat: null, lon: 16.3 }), { gpsRequired: true }).includes('gps'),
  'gpsRequired rejects a missing latitude',
);
check(
  session.validateTree(validTree({ lat: 38.5, lon: 16.3 }), { gpsRequired: true }).length === 0,
  'gpsRequired accepts fresh coordinates',
);

check(session.shouldBackup(20), 'shouldBackup triggers at the first interval');
check(session.shouldBackup(40), 'shouldBackup triggers at later exact intervals');
check(!session.shouldBackup(19), 'shouldBackup does not trigger before an interval');
check(!session.shouldBackup(21), 'shouldBackup does not trigger after an interval');
check(!session.shouldBackup(0), 'shouldBackup rejects zero');
check(!session.shouldBackup('20'), 'shouldBackup rejects non-numeric sequences');

check(
  session.nextNumberDefault([]) === null,
  'nextNumberDefault leaves a fresh session blank',
);
check(
  session.nextNumberDefault([{ numero: null }, {}, null]) === null,
  'nextNumberDefault leaves an all-blank session blank',
);
check(
  session.nextNumberDefault([
    { numero: 3 }, { numero: 1 }, { numero: null }, { numero: 2 },
  ]) === 4,
  'nextNumberDefault follows the highest number independent of row order',
);
check(
  session.nextNumberDefault([{ numero: 1 }, { numero: 3 }]) === 4,
  'nextNumberDefault does not reuse a deleted middle number',
);
check(
  session.nextNumberDefault([{ numero: '9' }, { numero: 4 }]) === 5,
  'nextNumberDefault ignores non-integer values',
);

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} session tests passed`);
