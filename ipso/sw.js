// ipso service worker — offline shell, cache-first.
//
// Update safety: when a new SW is installed, it sits in `waiting` until the
// app is fully closed. No skipWaiting, no clients.claim(). An operator who
// started a session on version N completes it on version N; the new version
// only activates after a clean app exit.
'use strict';

const APP_VERSION = '0.2.2';
const CACHE = 'ipso-v' + APP_VERSION;

const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './style.css',
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
    const cached = await caches.match(req);
    if (cached) return cached;
    try {
      const res = await fetch(req);
      // Don't cache opaque or error responses.
      if (res && res.ok && (res.type === 'basic' || res.type === 'default')) {
        const cache = await caches.open(CACHE);
        cache.put(req, res.clone());
      }
      return res;
    } catch (_) {
      // Offline + not cached → let it bubble up as a network error.
      return new Response('offline', { status: 503 });
    }
  })());
});
