import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const csv = require('./csv.js');

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

const session = {
  mode: 'samples',
  data: '2026-06-17',
  compresa: 'Capistrano',
  catastrofata: false,
  operatore: 'Mario Rossi',
};
const tree = {
  particella: '1',
  sample_area_id: 123,
  numero: 4,
  specie: 'Abete',
  d_cm: 42,
  h_m: 22,
  h_measured: false,
  lat: 38.512345,
  lon: 16.123456,
  acc_m: 5,
};

const reference = {
  sampling: {
    sample_areas: [{ sample_area_id: 123, number: '3 bis' }],
  },
};

const lines = csv.formatFile(session, [tree], reference).slice(csv.CSV_BOM.length).trimEnd().split(csv.CSV_NL);
const header = lines[0].split(csv.CSV_SEP);
const row = lines[1].split(csv.CSV_SEP);
check(header[3] === csv.SAMPLE_AREA_HEADER, 'sample CSV header includes sample area after Particella');
check(row[3] === '3 bis', 'sample CSV row includes human-readable sample area number');
check(row[4] === '0', 'Catastrofata column stays after sample area');

const missingArea = csv.formatRow({ ...tree, sample_area_id: null }, session, reference).split(csv.CSV_SEP);
check(missingArea[3] === '', 'missing sample area is exported blank');

const unknownArea = csv.formatRow({ ...tree, sample_area_id: 999 }, session, reference).split(csv.CSV_SEP);
check(unknownArea[3] === '', 'unknown sample area is exported blank rather than leaking an internal id');

const markLines = csv.formatFile({ ...session, mode: 'martellate' }, [tree], reference).slice(csv.CSV_BOM.length).trimEnd().split(csv.CSV_NL);
const markHeader = markLines[0].split(csv.CSV_SEP);
const markRow = markLines[1].split(csv.CSV_SEP);
check(markHeader[3] === 'Catastrofata', 'martellate CSV header stays unchanged');
check(markRow[3] === '0', 'martellate CSV row does not include sample area');

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} csv tests passed`);
