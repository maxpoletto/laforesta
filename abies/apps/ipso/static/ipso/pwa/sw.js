// ipso service worker — offline shell, cache-first.
//
// Update safety: when a new SW is installed, it sits in `waiting` until the
// app is fully closed or the operator explicitly presses the in-app update
// button. An operator who started a session on version N completes it on
// version N unless they deliberately request the new version.
'use strict';

// Django supplies the release number and the fingerprint of the offline shell.
importScripts('./version.js');
const CACHE_PREFIX = 'ipso-';
const CACHE = CACHE_PREFIX + APP_BUILD;

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(
      IPSO_SHELL.map((url) => new Request(url, { cache: 'no-cache' }))
    ))
  );
  // Deliberately do NOT call self.skipWaiting() here — see header note.
});

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(
      names
        .filter((n) => n.startsWith(CACHE_PREFIX) && n !== CACHE)
        .map((n) => caches.delete(n))
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (req.cache === 'no-store') return;
  e.respondWith((async () => {
    // IMPORTANT: scope the lookup to THIS SW's cache. The global
    // `caches.match(req)` searches every cache name in the origin, which
    // means an old (waiting) SW can serve files from a newer SW's
    // pre-populated cache (or vice versa), mixing versions and breaking
    // the page. Bug fixed after v0.2.7→v0.3.0 cross-cache contamination
    // produced blank screens on first reload after deploy.
    const cache = await caches.open(CACHE);
    const cached = await cache.match(req);
    if (cached) return cached;
    try {
      const res = await fetch(req);
      // Don't cache opaque, error, or explicitly non-storeable responses.
      const cacheControl = (res && res.headers.get('Cache-Control') || '').toLowerCase();
      if (res && res.ok && !cacheControl.includes('no-store') &&
          (res.type === 'basic' || res.type === 'default')) {
        cache.put(req, res.clone());
      }
      return res;
    } catch (_) {
      // Offline + not cached → let it bubble up as a network error.
      return new Response('offline', { status: 503 });
    }
  })());
});
