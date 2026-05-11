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
const SCHEMA_VERSION = 1;

const STORE_SESSIONS = 'sessions';
const STORE_TREES = 'trees';
const STORE_META = 'meta';

const STATUS_OPEN = 'open';
const STATUS_EXPORTED = 'exported';
const STATUS_ABANDONED = 'abandoned';

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
    req.onupgradeneeded = (e) => {
      const db = req.result;
      const oldVersion = e.oldVersion;
      if (oldVersion < 1) {
        const sessions = db.createObjectStore(STORE_SESSIONS, { keyPath: 'id' });
        sessions.createIndex('by_status', 'status');

        const trees = db.createObjectStore(STORE_TREES, {
          keyPath: 'id', autoIncrement: true,
        });
        trees.createIndex('by_session', 'session_id');
        trees.createIndex('by_session_seq', ['session_id', 'seq']);

        db.createObjectStore(STORE_META, { keyPath: 'key' });
      }
      // Future migrations go here.
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
  // fields: {data, compresa, particella, operatore}
  const id = uuid();
  const row = {
    id,
    schema_version: SCHEMA_VERSION,
    status: STATUS_OPEN,
    started_at: new Date().toISOString(),
    exported_at: null,
    data: fields.data,
    compresa: fields.compresa,
    particella: fields.particella,
    operatore: fields.operatore || '',
    tree_count: 0,
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

async function listOpenSessions(db) {
  return tx(db, [STORE_SESSIONS], 'readonly', async (t) => {
    const idx = t.objectStore(STORE_SESSIONS).index('by_status');
    return req(idx.getAll(STATUS_OPEN));
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

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

// Add a tree to a session. Allocates `seq` inside the same transaction so a
// retry after partial failure cannot duplicate (plan R8).
// rec fields (caller supplies): specie, d_cm, h_m, h_measured, lat, lng, acc_m.
async function addTree(db, sessionId, rec) {
  return tx(db, [STORE_SESSIONS, STORE_TREES], 'readwrite', async (t) => {
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
      lat: rec.lat == null ? null : rec.lat,
      lng: rec.lng == null ? null : rec.lng,
      acc_m: rec.acc_m == null ? null : rec.acc_m,
    };
    const treesStore = t.objectStore(STORE_TREES);
    const id = await req(treesStore.add(row));
    row.id = id;

    sess.tree_count = seq;
    sessStore.put(sess);

    return row;
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
    for (const k of ['specie', 'd_cm', 'h_m', 'h_measured']) {
      if (k in patch) row[k] = patch[k];
    }
    if (patch.h_measured != null) row.h_measured = patch.h_measured ? 1 : 0;
    store.put(row);
    return row;
  });
}

async function deleteTree(db, sessionId, treeId) {
  await tx(db, [STORE_SESSIONS, STORE_TREES], 'readwrite', async (t) => {
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
  STATUS_OPEN, STATUS_EXPORTED, STATUS_ABANDONED,
  openDb,
  startSession, getSession, listOpenSessions, setSessionStatus,
  addTree, listTrees, updateTree, deleteTree, lastTree,
  uuid,
};

if (typeof module !== 'undefined') module.exports = { Store };
