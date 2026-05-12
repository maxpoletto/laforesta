# Ipso: field-data-entry PWA

Tree-marking ("martellata") app for forest survey work. Offline-first
PWA: vanilla JS + IndexedDB + service worker, no npm, no bundler.
Operator picks a parcel, the app records species/D/h/lat/lng per tree,
emits a CSV that drops into abies.

User-facing docs are in `README.md`; this file is the Claude-targeted
module map and gotcha log.

# Commands

```bash
make reference   # rebuild reference.json from ../bosco/data/*.csv
make icons       # rebuild img/*.png + *.gif from ../logo/logo-grande.png
make test        # node tests.js — ~62 tests over csv/ipso/session
make serve       # python3 -m http.server 8000 (localhost = secure for SW)
make deploy      # reference → icons → test → rsync to ipso.laforesta.it
```

`tools/`, `tests.js`, `Makefile`, `README.md` are NOT shipped (see
`DEPLOY_EXCLUDES` in `Makefile`).

# Module layout

```
index.html          app shell
style.css           all styles
manifest.webmanifest PWA manifest (5 icons under img/)
sw.js               service worker (cache-first, no skipWaiting)
app.js              state machine, screen wiring, GPS UI, wake lock
store.js            IndexedDB wrapper (SCHEMA_VERSION = 2)
session.js          pure session-level helpers (resumability, etc.)
csv.js              CSV serialisation (UTF-8 BOM, ; sep, comma decimal, CRLF)
ipso.js             ipsometric regression lookup (h = a·ln(D) + b)
gps.js              self-healing watchPosition with heartbeat + restart
numpad.js           on-screen numeric keypad
download.js         browser-download helper
strings.js          Italian UI strings + helpers (S.where, S.pill)
reference.json      bundled reference data (generated)
img/                f.gif, l.gif, icon-192.png, icon-512.png, icon-512-maskable.png
tools/              build_reference.py, build_icons.py (not shipped)
tests.js            node tests for pure-logic modules
```

Each `.js` module ends with the CommonJS guard so `tests.js` can
require it:
```js
if (typeof module !== 'undefined') module.exports = { ... };
```

# Versioning

Bump `APP_VERSION` in **both** `app.js` and `sw.js` on every shippable
change (the SW cache name is derived from it). Old caches are deleted
in the SW `activate` handler. The SW deliberately does NOT call
`self.skipWaiting()` — a new SW stays in `waiting` until the app is
fully closed, so an operator who started a session on version N
completes it on version N.

# Storage

IndexedDB, schema v2. Session rows carry a `catastrofata` boolean;
catastrofate rows have `particella = ''` and the CSV emits column
`Catastrofata=1` between `Particella` and `Specie`. Old v1 rows are
treated as `catastrofata: false` via `||` fallback. Writes use
`transaction.oncomplete` (not the request `success` event) so the
operator never advances past a tree whose row hasn't been durably
committed.

# CSV format

```
Data;Compresa;Particella;Catastrofata;Specie;D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore
```

UTF-8 with BOM, `;` separator, `,` decimal, CRLF, `DD/MM/YYYY` dates.
`H_measured = 1` if the operator typed or edited h; `0` if the auto-h
estimate was accepted unchanged. `Lat`/`Lng` are 6-fractional comma
decimal; empty if no fix at save time. `Pino` is split into `Pino Nero`
and `Pino Marittimo` because their ipsometric regressions diverge.

# Gotchas

- **Apache `/icons/` alias collision.** A default `mods-enabled/alias.conf`
  on the host has `Alias /icons/ "/usr/share/apache2/icons/"`, which
  intercepts `/icons/*` before reaching the DocumentRoot. Icons live
  under `img/` for this reason. Do not rename to `icons/`.
- **Permissions-Policy header gates geolocation.** The global apache
  config (`/etc/apache2/apache2.conf` on the VM) sets
  `Permissions-Policy: geolocation=()`, which silently denies the API
  with no UI affordance for the user to override. The ipso vhost
  opts in via `allow_geolocation: true` in
  `../../system/ansible/group_vars/foresta/vars.yml`, which makes the
  vhost template emit a per-Directory override
  `Permissions-Policy: ... geolocation=(self) ...`. If GPS suddenly
  stops working on a new deploy, check the response headers with
  curl before debugging anything client-side.
- **Service-worker caching of `manifest.webmanifest`.** Bumping the
  version updates the cache name but Chrome's "Add to Home Screen"
  still reads the previously installed manifest. After a manifest
  change, the home-screen icon won't refresh until the user does
  Site settings → Clear & reset → revisit URL → re-add. Removing
  and re-adding the home-screen shortcut on its own is NOT enough.
  README covers this for end users.
- **GPS cadence under canopy.** `watchPosition` with `maximumAge: 1000`
  (not `0`) gives sub-second-old fixes a free pass and moves callback
  cadence from ~5–10 s to ~1–2 s. `GPS_RESTART_THRESHOLD_MS = 3000`
  restarts the watcher if no fix arrives in 3 s; tightening below that
  fights the OS's Kalman filter. A `navigator.wakeLock.request('screen')`
  is held while the recording screen is mounted and re-acquired on
  `visibilitychange → visible`.
- **Security-reminder hook false-positives.** Two patterns trip the
  hook spuriously: regex-match calls that look like `child_process`
  invocations, and empty-string assignment to `innerHTML`. Prefer
  `String.prototype.match` over the regex equivalent when a one-shot
  match is enough, and `el.replaceChildren()` over assigning an empty
  string to `innerHTML`.

# Reference data

`tools/build_reference.py` reads `../bosco/data/particelle.csv` (filters
`Governo=Fustaia`, `Comparto≠F`) and `../bosco/data/equazioni_ipsometro.csv`
to produce `reference.json` (62 parcels, 8 species, 13 regression
entries across 3 regions). The species list is hardcoded in the build
script and must stay in sync with
`../abies/apps/base/management/commands/import_reference.py`.

# Deployment

Deploy target is `ipso.laforesta.it:/var/www/ipso/html`. The vhost
(provisioned via `../../system/ansible/foresta.yml`) opts into
geolocation via the per-Directory `Permissions-Policy` override —
see Gotchas above. The vhost otherwise inherits the shared static-SSL
template (`templates/apache2-sites/static-ssl.conf.j2`).
