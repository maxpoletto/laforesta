/**
 * Tests for the in-memory cache + conditional-GET refresh (cache.js),
 * driven against a faithful model of the server's serve_digest:
 *   - lazy regeneration on read when the digest is stale,
 *   - If-Modified-Since compared at whole-second (HTTP-date) resolution.
 *
 * fetchJSON (api.js) calls global fetch, so we stub fetch with the model
 * and exercise the REAL cache.js through the client's actual
 * write-then-reload sequences.  Run: node cache.test.mjs (also `make test-js`).
 */

// --- Faithful model of apps/base/digests.py:serve_digest --------------------
// Holds the relational "DB" (source of truth), the on-disk digest file
// (data + whole-second mtime), and the stale flag.  Reads regenerate when
// stale or absent; the If-Modified-Since check floors mtime to the second,
// exactly like `int(os.path.getmtime(...))` against an HTTP date.
class DigestServer {
  constructor(data) {
    this.db = clone(data);     // {columns, rows}: the committed DB state
    this.file = null;          // {data, mtime} | null  (null = never built)
    this.stale = false;
    this.clock = 1000;         // whole seconds; advance() to simulate elapsed time
  }
  advance(seconds) { this.clock += seconds; }
  /** A committed write + mark_stale(): the digest now lags the DB. */
  write(data) { this.db = clone(data); this.stale = true; }
  _regenerateIfStale() {
    if (this.stale || this.file === null) {
      this.file = { data: clone(this.db), mtime: this.clock };
      this.stale = false;
    }
  }
  serve(ims) {
    this._regenerateIfStale();
    const lastModified = String(this.file.mtime);
    if (ims !== null && ims !== undefined && Number(ims) >= this.file.mtime) {
      return { status: 304, body: null, lastModified };
    }
    return { status: 200, body: clone(this.file.data), lastModified };
  }
}

function clone(x) { return JSON.parse(JSON.stringify(x)); }

// --- Wire the model in as global fetch (what api.js fetchJSON calls) ---------
const servers = new Map();   // url -> DigestServer
globalThis.fetch = async (url, opts) => {
  const ims = opts?.headers?.['If-Modified-Since'] ?? null;
  const res = servers.get(url).serve(ims);
  return {
    status: res.status,
    ok: res.status >= 200 && res.status < 300,
    headers: { get: (h) => (h === 'Last-Modified' ? res.lastModified : null) },
    json: async () => res.body,
  };
};

const cache = await import('./cache.js');

// --- Tiny assert harness ----------------------------------------------------
let pass = 0;
const failures = [];
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) pass++; else failures.push(`${msg}: expected ${e}, got ${a}`);
}

const COLS = ['row_id', 'version', 'd_cm'];
const digest = (rows) => ({ columns: COLS, rows });
const dcm = (data, rowId) => {
  const r = data.rows.find(row => row[0] === rowId);
  return r ? r[COLS.indexOf('d_cm')] : undefined;
};

let n = 0;
function freshDataId() { return `mark_trees_${n++}`; }   // isolate store entries

/**
 * Replays the client's edit flow: open (load) -> save (server write +
 * optimistic updateRow) -> renderItemView reload (load) -> user reload (load).
 * `gap` = seconds the user spends between steps (0 = same wall-clock second).
 * Returns the d_cm the table would show after the final reload.
 */
async function editThenReload(gap) {
  const id = freshDataId();
  const url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30]]));
  servers.set(url, server);
  cache.register(id, url);

  await cache.load(id);                       // open view
  eq(dcm(cache.get(id), 1), 30, `[gap ${gap}] open shows 30`);

  server.advance(gap);                        // user reads, then edits
  server.write(digest([[1, 2, 40]]));         // POST mark/save commits, marks stale
  cache.updateRow(id, 1, [1, 2, 40]);         // optimistic patch
  await cache.load(id);                        // renderItemView re-fetch
  eq(dcm(cache.get(id), 1), 40, `[gap ${gap}] after save shows 40`);

  server.advance(gap);                        // user reloads
  await cache.load(id);
  return dcm(cache.get(id), 1);
}

/** Open -> delete (server delete + optimistic removeRow) -> reload. */
async function deleteThenReload(gap) {
  const id = freshDataId();
  const url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30], [2, 1, 50]]));
  servers.set(url, server);
  cache.register(id, url);

  await cache.load(id);
  server.advance(gap);
  server.write(digest([[2, 1, 50]]));         // row 1 deleted, marks stale
  cache.removeRow(id, 1);                      // optimistic removal
  await cache.load(id);
  server.advance(gap);
  await cache.load(id);
  return cache.get(id).rows.map(r => r[0]);    // remaining row_ids
}

// A reload after an optimistic write must show the new state, whether the
// user paused (different second) or acted fast (same wall-clock second).
// With cache.js + the conditional GET as the only cache layers (no browser
// HTTP cache), both hold: a 200 brings fresh data, a 304 keeps the patched
// in-memory copy.
eq(await editThenReload(5), 40, 'edit + reload (elapsed) shows new value');
eq(await editThenReload(0), 40, 'edit + reload (same second) shows new value');
eq(await deleteThenReload(5), [2], 'delete + reload (elapsed): row stays gone');
eq(await deleteThenReload(0), [2], 'delete + reload (same second): row stays gone');

console.log(`${pass} passed, ${failures.length} failed`);
for (const f of failures) console.error('  FAIL ' + f);
process.exit(failures.length ? 1 : 0);
