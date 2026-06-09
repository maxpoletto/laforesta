/**
 * In-memory data cache with conditional-GET refresh.
 *
 * Each entry is keyed by a data ID (e.g. "prelievi", "crews") and stores
 * the parsed JSON, the Last-Modified header value, and the local timestamp
 * of the last successful fetch.
 */

import { fetchJSON } from './api.js';
import { DATA_ID, DELETES, PATCHES, RECORD, RECORDS, ROW_ID } from './constants.js';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000;  // 5 minutes

/** @type {Map<string, {data: any, lastModified: string|null, refreshedAt: number}>} */
const store = new Map();

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
  const entry = store.get(dataId);
  if (entry) {
    entry.data = data;
    entry.refreshedAt = Date.now();
  } else {
    store.set(dataId, { data, lastModified: null, refreshedAt: Date.now() });
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
 * Update or insert N rows.  Each record's row_id is record[0]
 * (digest convention).  Used by multi-row writes (e.g. the coppice
 * tree-save path that creates one TreeSample per shoot in one POST).
 *
 * @param {string} dataId
 * @param {Array<Array>} records
 */
export function updateRows(dataId, records) {
  for (const r of records) {
    if (Array.isArray(r) && r.length > 0) updateRow(dataId, r[0], r);
  }
}

/**
 * Remove a single row from a cached digest by row_id.
 *
 * @param {string} dataId
 * @param {number} rowId
 */
export function removeRow(dataId, rowId) {
  const entry = store.get(dataId);
  if (!entry?.data?.rows) return;
  entry.data.rows = entry.data.rows.filter(r => r[0] !== rowId);
}

/**
 * Apply the generic write-response cache envelope.
 *
 * Supported keys:
 *   patches: [{data_id, row_id, record}]
 *   patches: [{data_id, records: [[row_id, ...], ...]}]
 *   deletes: [{data_id, row_id}]
 *
 * Returns a Set of touched data IDs so page modules can update local mirrors
 * and re-render affected views.
 *
 * @param {object} data
 * @returns {Set<string>}
 */
export function applyResponseChanges(data) {
  const touched = new Set();
  if (!data) return touched;

  for (const patch of data[PATCHES] || []) {
    const dataId = patch[DATA_ID];
    if (!dataId) continue;
    if (Array.isArray(patch[RECORDS])) {
      updateRows(dataId, patch[RECORDS]);
      touched.add(dataId);
    } else if (patch[RECORD] && patch[ROW_ID] != null) {
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

  const entry = store.get(dataId);
  const lastMod = entry ? entry.lastModified : null;

  const result = await fetchJSON(url, lastMod);

  if (result.status === 304) {
    entry.refreshedAt = Date.now();
    return entry.data;
  }

  store.set(dataId, {
    data: result.data,
    lastModified: result.lastModified,
    refreshedAt: Date.now(),
  });
  // Fire onUpdate listeners so in-place callers (e.g., a row delete that
  // just wants the table to refresh without a full page rebuild) don't
  // have to re-implement the re-render themselves.  refreshVisible already
  // notifies on the same condition; mirroring it here keeps the manual
  // and background paths consistent.
  notify(dataId);
  return result.data;
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

    const entry = store.get(dataId);
    jobs.push(
      fetchJSON(url, entry.lastModified)
        .then(result => {
          if (result.status !== 304) {
            store.set(dataId, {
              data: result.data,
              lastModified: result.lastModified,
              refreshedAt: Date.now(),
            });
            changed.add(dataId);
          } else {
            entry.refreshedAt = Date.now();
          }
        })
        .catch(() => { /* swallow background refresh errors */ })
    );
  }

  await Promise.all(jobs);
  for (const dataId of changed) notify(dataId);
  return changed;
}
