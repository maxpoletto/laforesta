/**
 * Tests for the in-memory cache + conditional-GET refresh (cache.js),
 * driven against a faithful model of the server's serve_digest:
 *   - lazy regeneration on read when the digest is stale,
 *   - content ETags distinguish same-second published files.
 *
 * fetchJSON (api.js) calls global fetch, so we stub fetch with the model
 * and exercise the REAL cache.js through the client's actual
 * write-then-reload sequences.  Run: node cache.test.mjs (also `make test-js`).
 */

// --- Faithful model of apps/base/digests.py:serve_digest --------------------
// Holds the relational "DB" (source of truth), the on-disk digest file
// (data + whole-second mtime + content ETag), and the stale flag.
import { createHash } from 'node:crypto';

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
      const etag = `"${createHash('sha256').update(JSON.stringify(this.db)).digest('hex')}"`;
      this.file = { data: clone(this.db), mtime: this.clock, etag };
      this.stale = false;
    }
  }
  serve(ims, inm = null) {
    this._regenerateIfStale();
    const lastModified = String(this.file.mtime);
    const etag = this.file.etag;
    if (inm === etag) {
      return { status: 304, body: null, lastModified, etag };
    }
    return { status: 200, body: clone(this.file.data), lastModified, etag };
  }
}

function clone(x) { return JSON.parse(JSON.stringify(x)); }

// --- Wire the model in as global fetch (what api.js fetchJSON calls) ---------
const servers = new Map();   // url -> DigestServer
const digestFetch = async (url, opts) => {
  const ims = opts?.headers?.['If-Modified-Since'] ?? null;
  const res = servers.get(url).serve(ims, opts?.headers?.['If-None-Match'] ?? null);
  return {
    status: res.status,
    ok: res.status >= 200 && res.status < 300,
    headers: { get: h => ({ 'Last-Modified': res.lastModified, ETag: res.etag })[h] ?? null },
    json: async () => res.body,
  };
};
globalThis.fetch = digestFetch;

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

// Generic write-response envelope: patches + deletes.
{
  const id = freshDataId();
  const url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30], [2, 1, 50]]));
  servers.set(url, server);
  cache.register(id, url);
  await cache.load(id);

  const updates = [];
  const unsubscribe = cache.onUpdate(id, () => updates.push(clone(cache.get(id).rows)));
  const touched = cache.applyResponseChanges({
    patches: [{ data_id: id, row_id: 2, record: [2, 2, 55] }],
    deletes: [{ data_id: id, row_id: 1 }],
  });
  unsubscribe();

  eq([...touched], [id], 'generic envelope reports touched data id');
  eq(cache.get(id).rows, [[2, 2, 55]], 'generic envelope patches and deletes rows');
  eq(updates, [[[2, 2, 55]]], 'generic envelope notifies subscribers after patch/delete');
}

// A background response that started before an optimistic patch must not
// replace that newer local state when it resolves later.
{
  const id = freshDataId();
  const url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30]]));
  servers.set(url, server);
  cache.register(id, url);
  await cache.load(id);
  cache.setVisible([id]);

  let resolveFetch;
  globalThis.fetch = () => new Promise(resolve => { resolveFetch = resolve; });
  const refresh = cache.refreshVisible();
  await Promise.resolve();

  cache.updateRow(id, 1, [1, 3, 60]);
  resolveFetch({
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? '1001' : null },
    json: async () => digest([[1, 2, 50]]),
  });
  const changed = await refresh;

  eq(dcm(cache.get(id), 1), 60,
     'delayed background response preserves newer optimistic patch');
  eq([...changed], [],
     'discarded background response is not reported as a cache change');
  globalThis.fetch = digestFetch;
}

// Full-digest invalidation evicts exact IDs and dynamic families, not unrelated data.
{
  const exactId = 'diameter_parcel';
  const prefixedA = 'diameter_mark_1';
  const prefixedB = 'diameter_mark_2';
  const keepId = 'diameter_points';
  cache.set(exactId, digest([[1, 1, 10]]));
  cache.set(prefixedA, digest([[1, 1, 20]]));
  cache.set(prefixedB, digest([[1, 1, 30]]));
  cache.set(keepId, digest([[1, 1, 40]]));

  const updates = [];
  const unsubscribe = cache.onUpdate(prefixedA, () => updates.push(cache.get(prefixedA)));
  const touched = cache.applyResponseChanges({
    invalidates: {
      data_ids: [exactId],
      prefixes: ['diameter_mark_'],
    },
  });
  unsubscribe();

  eq([...touched], [exactId, prefixedA, prefixedB],
     'invalidation envelope reports every evicted cache entry');
  eq([cache.get(exactId), cache.get(prefixedA), cache.get(prefixedB)],
     [null, null, null], 'invalidation envelope evicts exact and prefixed entries');
  eq(dcm(cache.get(keepId), 1), 40, 'invalidation envelope preserves unrelated data');
  eq(updates, [null], 'invalidation envelope notifies subscribers after eviction');
}

// An invalidation must also discard a fetch that was already in flight. The
// following unconditional load then obtains the newly regenerated digest.
{
  const id = freshDataId();
  const url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30]]));
  servers.set(url, server);
  cache.register(id, url);
  await cache.load(id);

  server.write(digest([[1, 2, 50]]));
  let resolveFetch;
  globalThis.fetch = () => new Promise(resolve => { resolveFetch = resolve; });
  const pending = cache.load(id);
  await Promise.resolve();

  cache.applyResponseChanges({ invalidates: { data_ids: [id] } });
  globalThis.fetch = digestFetch;
  resolveFetch({
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? '1000' : null },
    json: async () => digest([[1, 1, 30]]),
  });
  await pending;
  eq(cache.get(id), digest([[1, 2, 50]]),
     'invalidated load retries unconditionally instead of returning an empty cache');
}

// No cache entry exists while the first request is in flight. Invalidation
// (or a row patch/delete) must still fence it, and load must obtain usable data.
for (const kind of ['exact', 'prefix', 'patch', 'delete', 'evicted-again']) {
  const prefix = `${freshDataId()}_cold_`, id = `${prefix}1`, url = `/api/${id}/`;
  cache.register(id, url);
  if (kind === 'evicted-again') {
    cache.set(id, digest([[1, 1, 20]]));
    cache.applyResponseChanges({ invalidates: { data_ids: [id] } });
  }
  const fresh = digest(kind === 'delete' ? [] : [[1, 2, 50]]);
  const updates = [];
  const unsubscribe = cache.onUpdate(id, () => updates.push(cache.get(id)));
  const calls = [];
  let resolveOld;
  globalThis.fetch = (url, options) => {
    calls.push(options.headers);
    if (calls.length === 1) return new Promise(resolve => { resolveOld = resolve; });
    return Promise.resolve({
      status: 200, ok: true,
      headers: { get: h => h === 'ETag' ? '"fresh"' : null },
      json: async () => fresh,
    });
  };
  const pending = cache.load(id);
  const envelope = {
    exact: { invalidates: { data_ids: [id] } },
    prefix: { invalidates: { prefixes: [prefix] } },
    patch: { patches: [{ data_id: id, row_id: 1, record: [1, 2, 50] }] },
    delete: { deletes: [{ data_id: id, row_id: 1 }] },
    'evicted-again': { invalidates: { data_ids: [id] } },
  }[kind];
  const touched = cache.applyResponseChanges(envelope);
  eq([...touched], [id], `${kind}: absent data ID is still reported as touched`);
  eq(cache.get(id), null, `${kind}: no partial digest is synthesized from a patch`);
  resolveOld({
    status: 200, ok: true,
    headers: { get: h => h === 'ETag' ? '"old"' : null },
    json: async () => digest([[1, 1, 30]]),
  });
  const loaded = await pending;
  eq(loaded, fresh, `${kind}: invalidated cold load returns fresh data`);
  eq(cache.get(id), fresh, `${kind}: old response cannot populate an absent entry`);
  eq(calls.length, 2, `${kind}: discarded load retries once`);
  eq(calls[1]?.['If-None-Match'] ?? null, null, `${kind}: retry has no evicted validator`);
  eq(updates, [null, fresh], `${kind}: subscribers never see the superseded response`);
  unsubscribe();
  globalThis.fetch = digestFetch;
}

// A newer cold request may finish before an older one, with or without an
// intervening invalidation. The older promise returns the already fresh cache.
for (const invalidate of [false, true]) {
  const id = freshDataId(), url = `/api/${id}/`;
  cache.register(id, url);
  let resolveOld, calls = 0, notifications = 0;
  const fresh = digest([[1, 2, 60]]);
  const unsubscribe = cache.onUpdate(id, () => { notifications++; });
  globalThis.fetch = () => {
    calls++;
    if (calls === 1) return new Promise(resolve => { resolveOld = resolve; });
    return Promise.resolve({
      status: 200, ok: true, headers: { get: () => null }, json: async () => fresh,
    });
  };
  const old = cache.load(id);
  if (invalidate) cache.applyResponseChanges({ invalidates: { data_ids: [id] } });
  await cache.load(id);
  resolveOld({
    status: 200, ok: true, headers: { get: () => null },
    json: async () => digest([[1, 1, 30]]),
  });
  eq(await old, fresh, `overlapping cold loads (${invalidate}): older promise returns current data`);
  eq(cache.get(id), fresh, `overlapping cold loads (${invalidate}): fresh entry survives`);
  eq(calls, 2, `overlapping cold loads (${invalidate}): no retry when fresh data already exists`);
  eq(notifications, invalidate ? 2 : 1,
     `overlapping cold loads (${invalidate}): discarded response sends no update`);
  unsubscribe();
  globalThis.fetch = digestFetch;
}

// A delayed 304 must not restore an evicted body or validator.
{
  const id = freshDataId(), url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30]]));
  servers.set(url, server);
  cache.register(id, url);
  await cache.load(id);
  let resolveOld;
  const calls = [], fresh = digest([[1, 2, 80]]);
  globalThis.fetch = (url, options) => {
    calls.push(options.headers);
    if (calls.length === 1) return new Promise(resolve => { resolveOld = resolve; });
    return Promise.resolve({
      status: 200, ok: true,
      headers: { get: h => h === 'ETag' ? '"new"' : null }, json: async () => fresh,
    });
  };
  const pending = cache.load(id);
  cache.applyResponseChanges({ invalidates: { data_ids: [id] } });
  resolveOld({
    status: 304, ok: false, headers: { get: () => server.file.etag },
  });
  eq(await pending, fresh, 'delayed 304: load retries instead of returning an evicted body');
  eq(calls[1]?.['If-None-Match'] ?? null, null, 'delayed 304: retry is unconditional');
  await cache.load(id);
  eq(calls[2]?.['If-None-Match'], '"new"', 'delayed 304: old ETag is not restored');
  globalThis.fetch = digestFetch;
}

// An unrelated invalidation must not discard a valid pending first load.
{
  const id = freshDataId();
  cache.register(id, `/api/${id}/`);
  let resolveFetch, calls = 0;
  globalThis.fetch = () => {
    calls++;
    return new Promise(resolve => { resolveFetch = resolve; });
  };
  const pending = cache.load(id), expected = digest([[1, 1, 30]]);
  cache.applyResponseChanges({ invalidates: { data_ids: ['unrelated'], prefixes: ['other_'] } });
  resolveFetch({
    status: 200, ok: true, headers: { get: () => null }, json: async () => expected,
  });
  eq(await pending, expected, 'unrelated invalidation preserves a pending cold load');
  eq(calls, 1, 'unrelated invalidation does not cause another request');
  globalThis.fetch = digestFetch;
}

// Reader B can regenerate a changed digest within reader A's cached second.
// A must receive the new content despite an unchanged Last-Modified date.
{
  const id = freshDataId(), url = `/api/${id}/`;
  const server = new DigestServer(digest([[1, 1, 30]]));
  servers.set(url, server);
  cache.register(id, url);
  await cache.load(id);
  server.write(digest([[1, 2, 60]]));
  server.serve(null); // reader B regenerates; no remaining stale flag
  await cache.load(id);
  eq(dcm(cache.get(id), 1), 60, 'reader A sees reader B\'s same-second regeneration');
  cache.setVisible([id]);
  eq([...(await cache.refreshVisible())], [], 'unchanged ETag background refresh returns 304');
  server.write(digest([[1, 3, 70]]));
  server.serve(null);
  eq([...(await cache.refreshVisible())], [id], 'background refresh notices same-second replacement');
  eq(dcm(cache.get(id), 1), 70, 'background refresh stores new content');
}

console.log(`${pass} passed, ${failures.length} failed`);
for (const f of failures) console.error('  FAIL ' + f);
process.exit(failures.length ? 1 : 0);
