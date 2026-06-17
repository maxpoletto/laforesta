// IndexedDB wrapper for ipso.
//
// One database (`ipso`), three object stores:
//   sessions   keyPath id (uuid string), index by_status
//   trees      keyPath id (autoIncrement), indexes by_session, by_session_seq
//   meta       keyPath key
//
// All write helpers return a Promise that resolves on `tx.oncomplete`, never
// on `req.onsuccess`. This is the "synchronously before UI advances"
// guarantee in the plan: UI handlers `await` these helpers, and the await
// only returns once the data is durable.
'use strict';

const DB_NAME = 'ipso';
// Bumped each time the on-disk row shape changes. IndexedDB tolerates
// missing/extra fields without a structural change, so the bump is the
// contract for "this code wrote/read vN-shaped rows" — not a migration
// trigger at the moment.
const SCHEMA_VERSION = 5;

const STORE_SESSIONS = 'sessions';
const STORE_TREES = 'trees';
const STORE_META = 'meta';

// Meta-store key namespace for per-operator "next tree number" persistence.
// Full key is `${META_KEY_NEXT_NUMBER_PREFIX}${normalizeOperator(name)}` and
// the value row is `{key, value: <int>}`.  Max-bumped on save; rewritten to
// the new in-session max+1 on delete-last so it tracks the same value
// prefillNumber would compute (see addTree / deleteTree for the gory bits).
const META_KEY_NEXT_NUMBER_PREFIX = 'next_number:';

const STATUS_OPEN = 'open';
const STATUS_PENDING_UPLOAD = 'pending_upload';
const STATUS_EXPORTED = 'exported';
const STATUS_ABANDONED = 'abandoned';

// upload_status enum (orthogonal to session status). Null on OPEN sessions
// and on pre-v5 rows.
const UPLOAD_STATUS_UPLOADED = 'uploaded';
const UPLOAD_STATUS_LOCAL_ONLY = 'local_only';

function isResumableStatus(s) {
  return s === STATUS_OPEN || s === STATUS_PENDING_UPLOAD;
}

// Canonical form for operator names: trim + lowercase. Used as the key the
// per-operator next-number meta entry lives under, so " Mario Rossi " and
// "mario rossi" share a counter.
function normalizeOperator(s) {
  return typeof s === 'string' ? s.trim().toLowerCase() : '';
}

// Pure rules for evolving the per-operator next-number counter. Extracted
// from addTree / deleteTree so node tests can hit them directly without an
// IDB mock. `prior` is the current persisted value (null if no row yet);
// the return value is what the meta entry should hold afterwards (which may
// equal `prior` — call sites skip the write in that case).
function nextNumberAfterSave(prior, number) {
  if (!Number.isInteger(number)) return prior;
  const candidate = number + 1;
  if (prior == null || candidate > prior) return candidate;
  return prior;
}

function nextNumberAfterDelete(prior, remainingTrees) {
  let maxNumber = null;
  if (remainingTrees) {
    for (const r of remainingTrees) {
      if (r && Number.isInteger(r.numero) &&
          (maxNumber == null || r.numero > maxNumber)) {
        maxNumber = r.numero;
      }
    }
  }
  // Empty / all-blank session leaves the counter alone: that meta value
  // may carry cross-session memory we don't want to discard.
  if (maxNumber == null) return prior;
  return maxNumber + 1;
}

function uuid() {
  // Prefer crypto.randomUUID; fall back to a v4-ish manual implementation
  // for ancient Android Chromes (Chrome 92+ has randomUUID — should be fine).
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, '0'));
  return `${h.slice(0, 4).join('')}-${h.slice(4, 6).join('')}-${h.slice(6, 8).join('')}-${h.slice(8, 10).join('')}-${h.slice(10).join('')}`;
}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, SCHEMA_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      const sessions = db.createObjectStore(STORE_SESSIONS, { keyPath: 'id' });
      sessions.createIndex('by_status', 'status');

      const trees = db.createObjectStore(STORE_TREES, {
        keyPath: 'id', autoIncrement: true,
      });
      trees.createIndex('by_session', 'session_id');
      trees.createIndex('by_session_seq', ['session_id', 'seq']);

      db.createObjectStore(STORE_META, { keyPath: 'key' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error('ipso: another tab holds an older DB version open'));
  });
}

// Helper: wrap a transaction in a Promise that resolves on oncomplete with
// the value returned by `body`. body receives the transaction.
function tx(db, stores, mode, body) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(stores, mode);
    let result;
    try {
      result = body(t);
    } catch (e) {
      reject(e);
      try { t.abort(); } catch (_) {}
      return;
    }
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error('ipso: transaction aborted'));
  });
}

// One-shot promise wrapper for an IDBRequest, used inside tx() bodies to
// read intermediate results before the transaction completes.
function req(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

async function startSession(db, fields) {
  // fields: {mode, reference_version, work_package_id, data, compresa,
  //          operatore, catastrofata}
  const id = uuid();
  const catastrofata = !!fields.catastrofata;
  const row = {
    id,
    schema_version: SCHEMA_VERSION,
    mode: fields.mode || 'martellate',
    status: STATUS_OPEN,
    started_at: new Date().toISOString(),
    exported_at: null,
    data: fields.data,
    compresa: fields.compresa,
    catastrofata,
    operatore: fields.operatore || '',
    reference_version: fields.reference_version || '',
    work_package_id: fields.work_package_id || '',
    tree_count: 0,
    upload_status: null,
    uploaded_at: null,
  };
  await tx(db, [STORE_SESSIONS], 'readwrite', (t) => {
    t.objectStore(STORE_SESSIONS).add(row);
  });
  return row;
}

async function getSession(db, id) {
  return tx(db, [STORE_SESSIONS], 'readonly', (t) =>
    req(t.objectStore(STORE_SESSIONS).get(id))
  );
}

async function listResumableSessions(db) {
  // Returns all sessions in a status that wants operator follow-up:
  // STATUS_OPEN (resume the recording) or STATUS_PENDING_UPLOAD
  // (retry the upload or confirm local-only).
  return tx(db, [STORE_SESSIONS], 'readonly', async (t) => {
    const all = await req(t.objectStore(STORE_SESSIONS).getAll());
    return all.filter((row) => isResumableStatus(row.status));
  });
}

async function setSessionStatus(db, id, status) {
  await tx(db, [STORE_SESSIONS], 'readwrite', async (t) => {
    const store = t.objectStore(STORE_SESSIONS);
    const row = await req(store.get(id));
    if (!row) throw new Error('ipso: session not found: ' + id);
    row.status = status;
    if (status === STATUS_EXPORTED) {
      row.exported_at = new Date().toISOString();
    }
    store.put(row);
  });
}

async function setSessionUploadStatus(db, id, uploadStatus) {
  await tx(db, [STORE_SESSIONS], 'readwrite', async (t) => {
    const store = t.objectStore(STORE_SESSIONS);
    const row = await req(store.get(id));
    if (!row) throw new Error('ipso: session not found: ' + id);
    row.upload_status = uploadStatus;
    if (uploadStatus === UPLOAD_STATUS_UPLOADED) {
      row.uploaded_at = new Date().toISOString();
    }
    store.put(row);
  });
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

// Add a tree to a session. Allocates `seq` inside the same transaction so a
// retry after partial failure cannot duplicate (plan R8).
// rec fields (caller supplies): specie, d_cm, h_m, h_measured,
// hypso_param_set_id, lat, lon, acc_m, numero, gruppo, particella.
async function addTree(db, sessionId, rec) {
  return tx(db, [STORE_SESSIONS, STORE_TREES, STORE_META], 'readwrite', async (t) => {
    const sessStore = t.objectStore(STORE_SESSIONS);
    const sess = await req(sessStore.get(sessionId));
    if (!sess) throw new Error('ipso: session not found: ' + sessionId);

    const seq = (sess.tree_count || 0) + 1;
    const row = {
      session_id: sessionId,
      seq,
      ts: new Date().toISOString(),
      specie: rec.specie,
      d_cm: rec.d_cm,
      h_m: rec.h_m,
      h_measured: rec.h_measured ? 1 : 0,
      hypso_param_set_id: rec.hypso_param_set_id == null ? null : rec.hypso_param_set_id,
      lat: rec.lat == null ? null : rec.lat,
      lon: rec.lon == null ? null : rec.lon,
      acc_m: rec.acc_m == null ? null : rec.acc_m,
      numero: Number.isInteger(rec.numero) ? rec.numero : null,
      gruppo: rec.gruppo || '',
      particella: rec.particella || '',
    };
    const treesStore = t.objectStore(STORE_TREES);
    const id = await req(treesStore.add(row));
    row.id = id;

    sess.tree_count = seq;
    sessStore.put(sess);

    // Persist the operator's next-number counter via the pure rule. On add
    // we max-bump it so a manual lower-number override can't roll it
    // backwards mid-session; delete-last is handled in deleteTree and
    // intentionally rolls back to the new in-session max. This is what a
    // fresh session for the same operator reads back from prefillNumber().
    const opKey = normalizeOperator(sess.operatore);
    if (opKey) {
      const metaStore = t.objectStore(STORE_META);
      const metaKey = META_KEY_NEXT_NUMBER_PREFIX + opKey;
      const existing = await req(metaStore.get(metaKey));
      const prior = existing && Number.isInteger(existing.value)
        ? existing.value : null;
      const next = nextNumberAfterSave(prior, row.numero);
      if (next !== prior) metaStore.put({ key: metaKey, value: next });
    }

    return row;
  });
}

// Returns the persisted "next tree number" for `operator`, or null if
// nothing was ever recorded for that (normalized) name.
async function getNextNumberForOperator(db, operator) {
  const opKey = normalizeOperator(operator);
  if (!opKey) return null;
  return tx(db, [STORE_META], 'readonly', async (t) => {
    const row = await req(
      t.objectStore(STORE_META).get(META_KEY_NEXT_NUMBER_PREFIX + opKey)
    );
    return row && Number.isInteger(row.value) ? row.value : null;
  });
}

async function listTrees(db, sessionId) {
  return tx(db, [STORE_TREES], 'readonly', (t) =>
    req(t.objectStore(STORE_TREES).index('by_session').getAll(sessionId))
  );
}

async function updateTree(db, sessionId, treeId, patch) {
  return tx(db, [STORE_TREES], 'readwrite', async (t) => {
    const store = t.objectStore(STORE_TREES);
    const row = await req(store.get(treeId));
    if (!row) throw new Error('ipso: tree not found: ' + treeId);
    if (row.session_id !== sessionId) {
      throw new Error('ipso: tree does not belong to session');
    }
    // Whitelist fields callers may patch.
    for (const k of ['specie', 'd_cm', 'h_m', 'h_measured', 'numero', 'gruppo']) {
      if (k in patch) row[k] = patch[k];
    }
    if (patch.h_measured != null) row.h_measured = patch.h_measured ? 1 : 0;
    store.put(row);
    return row;
  });
}

async function deleteTree(db, sessionId, treeId) {
  await tx(db, [STORE_SESSIONS, STORE_TREES, STORE_META], 'readwrite', async (t) => {
    const treesStore = t.objectStore(STORE_TREES);
    const row = await req(treesStore.get(treeId));
    if (!row) return;
    if (row.session_id !== sessionId) {
      throw new Error('ipso: tree does not belong to session');
    }
    treesStore.delete(treeId);
    // Decrement session.tree_count so the next addTree allocates the same
    // seq the deleted tree had — clean numbering for the typical
    // delete-last-then-rerecord flow.
    const sessStore = t.objectStore(STORE_SESSIONS);
    const sess = await req(sessStore.get(sessionId));
    if (sess && sess.tree_count > 0) {
      sess.tree_count -= 1;
      sessStore.put(sess);
    }

    // Roll the operator's next-number counter back to match the new
    // in-session max (e.g. trees 101,102,103,110 → delete-last → 104),
    // mirroring the in-session nextNumberDefault. The pure rule
    // (`nextNumberAfterDelete`) leaves `prior` untouched when no numbered
    // trees remain so cross-session counter state isn't erased.
    if (sess) {
      const opKey = normalizeOperator(sess.operatore);
      if (opKey) {
        const metaStore = t.objectStore(STORE_META);
        const metaKey = META_KEY_NEXT_NUMBER_PREFIX + opKey;
        const remaining = await req(
          treesStore.index('by_session').getAll(sessionId)
        );
        const existing = await req(metaStore.get(metaKey));
        const prior = existing && Number.isInteger(existing.value)
          ? existing.value : null;
        const next = nextNumberAfterDelete(prior, remaining);
        if (next !== prior) {
          metaStore.put({ key: metaKey, value: next });
        }
      }
    }
  });
}

async function lastTree(db, sessionId) {
  return tx(db, [STORE_TREES], 'readonly', async (t) => {
    const idx = t.objectStore(STORE_TREES).index('by_session_seq');
    // openCursor with a bound range, descending, on [session_id, seq]:
    // upperBound is [session_id, +Infinity]; we want the largest seq.
    const range = IDBKeyRange.bound(
      [sessionId, -Infinity], [sessionId, Infinity]
    );
    const cur = await req(idx.openCursor(range, 'prev'));
    return cur ? cur.value : null;
  });
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

const Store = {
  DB_NAME, SCHEMA_VERSION,
  STATUS_OPEN, STATUS_PENDING_UPLOAD, STATUS_EXPORTED, STATUS_ABANDONED,
  UPLOAD_STATUS_UPLOADED, UPLOAD_STATUS_LOCAL_ONLY,
  isResumableStatus, normalizeOperator,
  nextNumberAfterSave, nextNumberAfterDelete,
  openDb,
  startSession, getSession, listResumableSessions, setSessionStatus,
  setSessionUploadStatus,
  addTree, listTrees, updateTree, deleteTree, lastTree,
  getNextNumberForOperator,
  uuid,
};

if (typeof module !== 'undefined') module.exports = { Store };
