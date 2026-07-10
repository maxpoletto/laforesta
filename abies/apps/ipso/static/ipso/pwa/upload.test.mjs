import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const upload = require('./upload.js');

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

function checkThrows(fn, needle, msg) {
  try {
    fn();
    failures.push(msg + ' (no error)');
  } catch (e) {
    if (String(e.message || e).includes(needle)) pass += 1;
    else failures.push(msg + ` (got ${e.message})`);
  }
}

function reference() {
  return {
    reference_version: 'ref-1',
    species: [{ id: 10, common: 'Abete' }],
    parcels: [
      { region_id: 1, parcel_id: 100, compresa: 'Serra', particella: '1' },
      { region_id: 1, parcel_id: 101, compresa: 'Serra', particella: '2' },
    ],
    sampling: {
      surveys: [{
        survey_id: 7,
        sample_grid_id: 70,
        sample_area_max_numbers: { '123': 3 },
      }],
      sample_areas: [
        { sample_area_id: 123, sample_grid_id: 70, parcel_id: 100, compresa: 'Serra', particella: '1', number: 'A1' },
        { sample_area_id: 124, sample_grid_id: 70, parcel_id: 101, compresa: 'Serra', particella: '2', number: 'A2' },
      ],
    },
    pai: {
      preserved_trees: [
        { parcel_id: 100, number: 8 },
        { parcel_id: 101, number: 4 },
      ],
    },
  };
}

function sess(mode) {
  return {
    id: '11111111-1111-4111-8111-111111111111',
    mode,
    region_id: 1,
    survey_id: mode === upload.UPLOAD_MODE_SAMPLES ? 7 : null,
    reference_version: 'ref-1',
    work_package_id: mode === upload.UPLOAD_MODE_SAMPLES ? 'sampling_survey:7' : '',
    operatore: 'Mario Rossi',
    data: '2026-06-17',
    compresa: 'Serra',
    catastrofata: false,
    started_at: '2026-06-17T08:00:00Z',
    exported_at: '2026-06-17T09:00:00Z',
  };
}

function tree(overrides = {}) {
  return Object.assign({
    id: 1,
    seq: 1,
    particella: '1',
    region_id: 1,
    parcel_id: 100,
    species_id: 10,
    sample_area_id: 123,
    coppice: false,
    numero: 4,
    specie: 'Abete',
    d_cm: 42,
    h_m: 22,
    h_measured: false,
    lat: 38.5,
    lon: 16.3,
    acc_m: 5,
  }, overrides);
}

const martellatePayload = upload.buildUploadPayload(
  sess(upload.UPLOAD_MODE_MARTELLATE),
  [tree({ numero: null })],
  reference(),
  'csv',
);
check(
  martellatePayload.records[0].number === null,
  'martellate upload keeps blank numbers blank',
);

checkThrows(
  () => upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_SAMPLES), [tree({ numero: null })], reference(), 'csv',
  ),
  'numero obbligatorio',
  'samples upload rejects missing number',
);

checkThrows(
  () => upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_PAI), [tree({ numero: null })], reference(), 'csv',
  ),
  'numero obbligatorio',
  'PAI upload rejects missing number',
);

checkThrows(
  () => upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_PAI), [tree({ numero: 0 })], reference(), 'csv',
  ),
  'numero obbligatorio',
  'PAI upload rejects zero number',
);

checkThrows(
  () => upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_SAMPLES),
    [tree({ id: 1, seq: 1, numero: 4 }), tree({ id: 2, seq: 2, numero: 4 })],
    reference(),
    'csv',
  ),
  'numero già usato',
  'samples upload rejects duplicate number in the same sample area',
);

const sampleDistinctAreas = upload.buildUploadPayload(
  sess(upload.UPLOAD_MODE_SAMPLES),
  [
    tree({ id: 1, seq: 1, sample_area_id: 123, particella: '1', numero: 4 }),
    tree({ id: 2, seq: 2, sample_area_id: 124, particella: '2', parcel_id: 101, numero: 4 }),
  ],
  reference(),
  'csv',
);
check(
  sampleDistinctAreas.records.length === 2,
  'samples upload allows same number in different sample areas',
);

checkThrows(
  () => upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_SAMPLES), [tree({ numero: 3 })], reference(), 'csv',
  ),
  'numero già presente',
  'samples upload rejects numbers already covered by reference max',
);

checkThrows(
  () => upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_PAI), [tree({ numero: 8 })], reference(), 'csv',
  ),
  'numero già presente',
  'PAI upload rejects number already present in the same parcel',
);

const paiDistinctParcels = upload.buildUploadPayload(
  sess(upload.UPLOAD_MODE_PAI),
  [tree({ id: 1, seq: 1, particella: '1', numero: 9 }),
   tree({ id: 2, seq: 2, particella: '2', parcel_id: 101, numero: 9 })],
  reference(),
  'csv',
);
check(
  paiDistinctParcels.records.length === 2,
  'PAI upload allows same number in different parcels',
);

// New rows use IDs captured at record time even if names are later renamed or
// reused for different entities in the current reference bundle.
{
  const changed = reference();
  changed.species = [{ id: 99, common: 'Abete' }];
  changed.parcels = [
    { region_id: 9, parcel_id: 999, compresa: 'Serra', particella: '1' },
  ];
  changed.sampling.sample_areas = [];
  const payload = upload.buildUploadPayload(
    sess(upload.UPLOAD_MODE_SAMPLES), [tree()], changed, 'csv',
  );
  const record = payload.records[0];
  check(payload.session.region_id === 1, 'upload preserves the recorded region ID');
  check(record.region_id === 1, 'record payload preserves its recorded region ID');
  check(record.parcel_id === 100, 'record payload preserves its recorded parcel ID');
  check(record.species_id === 10, 'record payload preserves its recorded species ID');
  check(record.sample_area_id === 123, 'record payload preserves its recorded sample-area ID');
  check(record.coppice === false, 'record payload preserves recorded sample-area context');
}

// Pre-v7 rows remain uploadable through the old name-resolution fallback.
{
  const legacySession = sess(upload.UPLOAD_MODE_MARTELLATE);
  delete legacySession.region_id;
  delete legacySession.survey_id;
  const legacyTree = tree({ numero: null });
  delete legacyTree.region_id;
  delete legacyTree.parcel_id;
  delete legacyTree.species_id;
  const payload = upload.buildUploadPayload(legacySession, [legacyTree], reference(), 'csv');
  check(payload.session.region_id === 1, 'legacy session resolves region by display name');
  check(payload.records[0].parcel_id === 100, 'legacy row resolves parcel by display name');
  check(payload.records[0].species_id === 10, 'legacy row resolves species by display name');
}

check(upload.paiMaxNumberForParcel(reference(), 100) === 8, 'PAI max number helper reads reference');

const backoffCases = [
  [0, 0], [1, 2000], [2, 4000], [3, 8000], [4, 16000], [5, 30000], [99, 30000],
];
for (const [attempt, expected] of backoffCases) {
  check(
    upload.backoffMs(attempt) === expected,
    `backoffMs(${attempt}) returns ${expected}`,
  );
}
check(upload.backoffMs(Number.NaN) === 0, 'backoffMs ignores non-finite attempts');

const httpCases = [
  [200, 'ok'],
  [401, 'hard:auth'],
  [409, 'hard:conflict'],
  [413, 'hard:too_large'],
  [422, 'hard:invalid_csv'],
  [429, 'soft:rate_limited'],
  [500, 'soft:server'],
  [599, 'soft:server'],
  [400, 'hard:invalid_csv'],
  [404, 'hard:invalid_csv'],
  [600, 'hard:invalid_csv'],
];
for (const [status, expected] of httpCases) {
  check(
    upload.classifyHttp(status) === expected,
    `classifyHttp(${status}) returns ${expected}`,
  );
}
check(
  upload.classifyNetwork() === 'soft:network',
  'classifyNetwork keeps network failures retryable',
);

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} upload tests passed`);
