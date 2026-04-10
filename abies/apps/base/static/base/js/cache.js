/**
 * In-memory data cache with conditional-GET refresh.
 *
 * Each entry is keyed by a data ID (e.g. "prelievi", "crews") and stores
 * the parsed JSON, the Last-Modified header value, and the local timestamp
 * of the last successful fetch.
 */

import { fetchJSON } from './api.js';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000;  // 5 minutes

/** @type {Map<string, {data: any, lastModified: string|null, refreshedAt: number}>} */
const store = new Map();

/** URLs registered for each data ID (set via register). */
const urls = new Map();

/** IDs currently visible — only these get background-refreshed. */
const visible = new Set();

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
 * Manually update the cache for a data ID (e.g., after a successful POST
 * that returns an updated row).
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
 * Fetch data, using conditional GET if we have a cached copy.
 * Returns the (possibly updated) data.
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
  return result.data;
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
  return changed;
}
