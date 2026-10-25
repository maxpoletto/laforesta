/**
 * In-memory data cache with conditional-GET refresh.
 *
 * Each entry is keyed by a data ID (e.g. "prelievi", "crews") and stores
 * the parsed JSON, ETag/Last-Modified validators, and the local timestamp
 * of the last successful fetch.
 */

import { fetchJSON } from './api.js';
import {
  DATA_ID, DELETES, FIELD_DATA_IDS, FIELD_PREFIXES, INVALIDATES, PATCHES,
  RECORD, ROW_ID,
} from './constants.js';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000;  // 5 minutes

/** @type {Map<string, {data: any, etag: string|null, lastModified: string|null, refreshedAt: number}>} */
const store = new Map();

/** Revisions survive eviction and exist independently of cached bodies. */
const revisions = new Map();

function bumpRevision(dataId) {
  revisions.set(dataId, (revisions.get(dataId) ?? 0) + 1);
}

/** URLs registered for each data ID (set via register). */
const urls = new Map();

/** IDs currently visible — only these get background-refreshed. */
const visible = new Set();

/** Listeners notified when a data ID is updated via background refresh. */
const listeners = new Map();

/** Timer handle for background refresh. */
let timer = null;

/**
 * Register a data ID → URL mapping.
 * Call once per data source at module init time.
 */
export function register(dataId, url) {
  urls.set(dataId, url);
}

/**
 * Return cached data (or null if not yet fetched).
 */
export function get(dataId) {
  const entry = store.get(dataId);
  return entry ? entry.data : null;
}

/**
 * Manually replace the entire cached dataset for a data ID.
 */
export function set(dataId, data) {
  bumpRevision(dataId);
  const entry = store.get(dataId);
  if (entry) {
    entry.data = data;
    entry.refreshedAt = Date.now();
  } else {
    store.set(dataId, {
      data,
      lastModified: null,
      etag: null,
      refreshedAt: Date.now(),
    });
  }
}

/**
 * Update or insert a single row in a cached digest.
 * Assumes digest format: { columns: [...], rows: [[row_id, ...], ...] }
 * where the first element of each row is the row_id.
 *
 * @param {string} dataId
 * @param {number} rowId
 * @param {Array} record — full row array matching the digest columns
 */
export function updateRow(dataId, rowId, record) {
  // A patch cannot synthesize a missing digest's columns, but must fence any
  // older fetch that could otherwise populate it with pre-write data.
  bumpRevision(dataId);
  const entry = store.get(dataId);
  if (!entry?.data?.rows) return;
  const idx = entry.data.rows.findIndex(r => r[0] === rowId);
  if (idx >= 0) {
    entry.data.rows[idx] = record;
  } else {
    entry.data.rows.push(record);
  }
}

/**
 * Remove a single row from a cached digest by row_id.
 *
 * @param {string} dataId
 * @param {number} rowId
 */
export function removeRow(dataId, rowId) {
  bumpRevision(dataId);
  const entry = store.get(dataId);
  if (!entry?.data?.rows) return;
  const rows = entry.data.rows.filter(r => r[0] !== rowId);
  if (rows.length === entry.data.rows.length) return;
  entry.data.rows = rows;
}

function takeSnapshot(dataId) {
  const entry = store.get(dataId) || null;
  return { entry, revision: revisions.get(dataId) ?? 0 };
}

function snapshotIsCurrent(dataId, snapshot) {
  const entry = store.get(dataId) || null;
  return entry === snapshot.entry
    && (revisions.get(dataId) ?? 0) === snapshot.revision;
}

/**
 * Apply a fetch result only if the cache entry has not changed since the
 * request started. This prevents a delayed response from replacing a newer
 * optimistic patch or a response from another request.
 */
function applyFetchResult(dataId, snapshot, result) {
  if (!snapshotIsCurrent(dataId, snapshot)) {
    return { data: get(dataId), changed: false, discarded: true };
  }

  if (result.status === 304) {
    if (snapshot.entry) {
      snapshot.entry.refreshedAt = Date.now();
      snapshot.entry.etag = result.etag;
    }
    return { data: snapshot.entry?.data ?? null, changed: false };
  }

  store.set(dataId, {
    data: result.data,
    lastModified: result.lastModified,
    etag: result.etag,
    refreshedAt: Date.now(),
  });
  bumpRevision(dataId);
  return { data: result.data, changed: true };
}

/**
 * Apply the generic write-response cache envelope.
 *
 * Supported keys:
 *   patches: [{data_id, row_id, record}]
 *   deletes: [{data_id, row_id}]
 *   invalidates: {data_ids: [...], prefixes: [...]}
 *
 * Returns a Set of touched data IDs so page modules can update local mirrors
 * and re-render affected views.  Also fires each touched data ID's onUpdate
 * listeners once, matching cache.load() / background refresh semantics.
 *
 * @param {object} data
 * @returns {Set<string>}
 */
export function applyResponseChanges(data) {
  const touched = new Set();
  if (!data) return touched;

  const invalidations = data[INVALIDATES] || {};
  const dataIds = new Set(invalidations[FIELD_DATA_IDS] || []);
  const prefixes = invalidations[FIELD_PREFIXES] || [];
  // Registered IDs include cold/in-flight loads. Explicit IDs also need a
  // generation even if they have never had a cached entry or registered URL.
  const knownIds = new Set([...store.keys(), ...urls.keys(), ...revisions.keys(), ...dataIds]);
  for (const dataId of knownIds) {
    if (!dataIds.has(dataId) && !prefixes.some(prefix => dataId.startsWith(prefix))) {
      continue;
    }
    bumpRevision(dataId);
    // Evict validators as well as data. The revision fences even a first
    // fetch whose snapshot and current cache entry are both null.
    store.delete(dataId);
    touched.add(dataId);
  }

  for (const patch of data[PATCHES] || []) {
    const dataId = patch[DATA_ID];
    if (!dataId) continue;
    if (patch[RECORD] && patch[ROW_ID] != null) {
      updateRow(dataId, patch[ROW_ID], patch[RECORD]);
      touched.add(dataId);
    }
  }

  for (const del of data[DELETES] || []) {
    const dataId = del[DATA_ID];
    if (!dataId || del[ROW_ID] == null) continue;
    removeRow(dataId, del[ROW_ID]);
    touched.add(dataId);
  }

  for (const dataId of touched) notify(dataId);
  return touched;
}

/**
 * Fetch data, using conditional GET if we have a cached copy.
 * Returns the (possibly updated) data.
 *
 * Fires `onUpdate` listeners when the server returns fresh data
 * (status 200), but not on a 304.  This matches `refreshVisible` and
 * lets in-place callers (e.g., a row delete that just wants the table
 * to re-render) rely on the same listener as the periodic background
 * refresh — no need to manually re-render after every write.
 *
 * @param {string} dataId
 * @returns {Promise<any>}
 */
export async function load(dataId) {
  const url = urls.get(dataId);
  if (!url) throw new Error(`Unknown data ID: ${dataId}`);

  while (true) {
    const snapshot = takeSnapshot(dataId);
    const result = await fetchJSON(
      url, snapshot.entry?.lastModified ?? null, snapshot.entry?.etag ?? null,
    );
    const applied = applyFetchResult(dataId, snapshot, result);
    // An invalidated first load must still fulfill its caller with usable
    // data. Reuse a newer cache entry if present, otherwise fetch the current
    // generation unconditionally. Never notify with the discarded body.
    if (applied.discarded && applied.data === null) continue;
    if (applied.changed) notify(dataId);
    return applied.data;
  }
}

/**
 * Register a callback invoked when a data ID is updated by background refresh.
 * Returns an unsubscribe function.
 *
 * @param {string} dataId
 * @param {function} callback
 * @returns {function} unsubscribe
 */
export function onUpdate(dataId, callback) {
  if (!listeners.has(dataId)) listeners.set(dataId, new Set());
  listeners.get(dataId).add(callback);
  return () => listeners.get(dataId).delete(callback);
}

function notify(dataId) {
  const cbs = listeners.get(dataId);
  if (cbs) for (const cb of cbs) cb();
}

/**
 * Mark which data IDs are currently visible.  Only visible IDs participate
 * in background refresh.
 *
 * @param {string[]} dataIds
 */
export function setVisible(dataIds) {
  visible.clear();
  for (const id of dataIds) visible.add(id);
}

/**
 * Start the background refresh timer.
 */
export function startRefresh() {
  if (timer) return;
  timer = setInterval(refreshVisible, REFRESH_INTERVAL_MS);
}

/**
 * Stop the background refresh timer.
 */
export function stopRefresh() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

/**
 * Refresh all visible data IDs via conditional GET.
 * Returns a map of data IDs whose data actually changed.
 *
 * @returns {Promise<Set<string>>}
 */
export async function refreshVisible() {
  const changed = new Set();
  const jobs = [];

  for (const dataId of visible) {
    if (!store.has(dataId)) continue;
    const url = urls.get(dataId);
    if (!url) continue;

    const snapshot = takeSnapshot(dataId);
    jobs.push(
      fetchJSON(url, snapshot.entry.lastModified, snapshot.entry.etag)
        .then(result => {
          const applied = applyFetchResult(dataId, snapshot, result);
          if (applied.changed) changed.add(dataId);
        })
        .catch(() => { /* swallow background refresh errors */ })
    );
  }

  await Promise.all(jobs);
  for (const dataId of changed) notify(dataId);
  return changed;
}
