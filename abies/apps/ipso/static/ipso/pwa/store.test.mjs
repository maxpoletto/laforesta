import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { Store } = require('./store.js');

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

function same(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function makeMetaDb() {
  const rows = new Map();
  return {
    transaction() {
      let pending = 0;
      let completed = false;
      const transaction = {
        error: null,
        objectStore() {
          return {
            get(key) {
              return request(() => rows.get(key));
            },
            put(row) {
              return request(() => {
                rows.set(row.key, row);
                return row.key;
              });
            },
          };
        },
        abort() {},
      };
      function request(action) {
        const result = {};
        pending += 1;
        queueMicrotask(() => {
          try {
            result.result = action();
            if (result.onsuccess) result.onsuccess();
          } catch (error) {
            result.error = error;
            transaction.error = error;
            if (result.onerror) result.onerror();
            if (transaction.onerror) transaction.onerror();
          } finally {
            pending -= 1;
            if (pending === 0) {
              setTimeout(() => {
                if (!completed && transaction.oncomplete) {
                  completed = true;
                  transaction.oncomplete();
                }
              }, 0);
            }
          }
        });
        return result;
      }
      return transaction;
    },
  };
}

check(
  Store.nextSeqAfterRows([]) === 1,
  'nextSeqAfterRows starts fresh sessions at one',
);
check(Store.SCHEMA_VERSION === 7, 'schema v7 identifies canonical-ID session and tree rows');
check(
  Store.isResumableStatus(Store.STATUS_OPEN) &&
    Store.isResumableStatus(Store.STATUS_PENDING_UPLOAD) &&
    !Store.isResumableStatus(Store.STATUS_EXPORTED),
  'resumable sessions are active recording or pending upload rows',
);
check(
  Store.isTerminalStatus(Store.STATUS_EXPORTED) &&
    Store.isTerminalStatus(Store.STATUS_ABANDONED) &&
    !Store.isTerminalStatus(Store.STATUS_OPEN),
  'terminal sessions are exported or abandoned rows',
);
check(
  Store.isRecoverableStatus(Store.STATUS_OPEN) &&
    Store.isRecoverableStatus(Store.STATUS_PENDING_UPLOAD) &&
    Store.isRecoverableStatus(Store.STATUS_EXPORTED) &&
    Store.isRecoverableStatus(Store.STATUS_ABANDONED),
  'recoverable sessions include active rows and the local archive',
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

// Protected reference data is application-cached in the existing meta store;
// writes resolve only after the IndexedDB transaction completes.
{
  const db = makeMetaDb();
  const reference = { reference_version: 'v1', parcels: [{ parcel_id: 7 }] };
  const terreni = [{ type: 'Feature', properties: { parcel_id: 7 } }];
  await Store.cacheReference(db, reference);
  await Store.cacheTerreni(db, terreni);
  const cached = await Store.getCachedBootResources(db);
  check(same(cached.reference, reference), 'reference snapshot round-trips through meta');
  check(same(cached.terreni, terreni), 'parcel geometry snapshot round-trips through meta');

  const replacement = { reference_version: 'v2', parcels: [{ parcel_id: 8 }] };
  await Store.cacheReference(db, replacement);
  const refreshed = await Store.getCachedBootResources(db);
  check(same(refreshed.reference, replacement), 'new reference replaces the last-good snapshot');
  check(same(refreshed.terreni, terreni), 'reference refresh retains last-good geometry');
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`${pass} store tests passed`);
