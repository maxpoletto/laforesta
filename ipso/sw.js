// ipso service worker — offline shell, cache-first.
//
// Update safety: when a new SW is installed, it sits in `waiting` until the
// app is fully closed. No skipWaiting, no clients.claim(). An operator who
// started a session on version N completes it on version N; the new version
// only activates after a clean app exit.
'use strict';

// APP_VERSION lives in version.js, shared with the page.
importScripts('./version.js');
const CACHE = 'ipso-v' + APP_VERSION;

const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './style.css',
  './version.js',
  './app.js',
  './csv.js',
  './ipso.js',
  './session.js',
  './strings.js',
  './download.js',
  './gps.js',
  './numpad.js',
  './store.js',
  './reference.json',
  './img/f.gif',
  './img/l.gif',
  './img/icon-192.png',
  './img/icon-512.png',
  './img/icon-512-maskable.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL))
  );
  // Deliberately do NOT call self.skipWaiting() — see header note.
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(
      names.filter((n) => n !== CACHE).map((n) => caches.delete(n))
    );
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
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
      // Don't cache opaque or error responses.
      if (res && res.ok && (res.type === 'basic' || res.type === 'default')) {
        cache.put(req, res.clone());
      }
      return res;
    } catch (_) {
      // Offline + not cached → let it bubble up as a network error.
      return new Response('offline', { status: 503 });
    }
  })());
});
