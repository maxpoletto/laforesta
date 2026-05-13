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
index.html          app shell (4 screens: pre, rec, done, data)
style.css           all styles
manifest.webmanifest PWA manifest (5 icons under img/)
version.js          APP_VERSION constant — single source of truth
sw.js               service worker (cache-first, no skipWaiting)
app.js              state machine, screen wiring, GPS UI, wake lock
store.js            IndexedDB wrapper (SCHEMA_VERSION = 3)
session.js          pure session-level helpers (resumability, etc.)
csv.js              CSV serialisation (UTF-8 BOM, ; sep, comma decimal, CRLF)
ipso.js             ipsometric regression lookup (h = a·ln(D) + b)
gps.js              self-healing watchPosition with heartbeat + restart
numpad.js           on-screen numeric keypad (generic over field set)
download.js         browser-download helper
strings.js          Italian UI strings + helpers (S.where, S.pill)
reference.json      bundled reference data (generated)
img/                f.gif, l.gif, icon-192.png, icon-512.png, icon-512-maskable.png
tools/              build_reference.py, build_icons.py (not shipped)
tests.js            node tests for pure-logic modules
```

The `screen-data` section ("Visualizza dati raccolti") renders two
read-only tables — per-group counts and a per-tree list — from a
snapshot of `Store.listTrees`.  It's a side excursion from the
recording screen and preserves all in-progress field state.

Each `.js` module ends with the CommonJS guard so `tests.js` can
require it:
```js
if (typeof module !== 'undefined') module.exports = { ... };
```

# Versioning

`APP_VERSION` lives in `version.js`.  Bump that single constant on
every shippable change.  Both consumers read it:

- `index.html` loads `version.js` as the first `<script>` tag, so
  `APP_VERSION` is in the page's lexical scope by the time
  `app.js` references it.
- `sw.js` does `importScripts('./version.js')` at the top and
  derives `CACHE = 'ipso-v' + APP_VERSION` from it.

The SW cache name therefore tracks the version automatically.  Old
caches are deleted in the SW `activate` handler.  The SW
deliberately does NOT call `self.skipWaiting()` — a new SW stays
in `waiting` until the app is fully closed, so an operator who
started a session on version N completes it on version N.

# Storage

IndexedDB, schema v3.

- v2 added `catastrofata` (boolean) on session rows.  Catastrofate
  rows have `particella = ''` and the CSV emits column
  `Catastrofata=1`.  Old v1 rows are treated as `catastrofata: false`
  via the `||` fallback.
- v3 added `numero` (int|null) and `gruppo` (string) on tree rows.
  Old v2 rows surface as blank in the visualizza screen and CSV.

No structural migrations across these bumps — IndexedDB tolerates
extra fields, and the version bump is the contract for the new row
shape.  Writes use `transaction.oncomplete` (not the request
`success` event) so the operator never advances past a tree whose
row hasn't been durably committed.

# CSV format

```
Data;Compresa;Particella;Catastrofata;Numero;Specie;D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore
```

UTF-8 with BOM, `;` separator, `,` decimal, CRLF, `DD/MM/YYYY` dates.
`H_measured = 1` if the operator typed or edited h; `0` if the auto-h
estimate was accepted unchanged.  `Numero` is the operator-assigned
tree number (4-digit max); blank for trees with
`D ≤ NUMERO_BLANK_D_THRESHOLD` (17 cm).  The `Gruppo` tag is an
in-app working aid and is intentionally NOT exported.  `Lat`/`Lng`
are 6-fractional comma decimal; empty if no fix at save time.
`Pino` is split into `Pino Nero` and `Pino Marittimo` because their
ipsometric regressions diverge.

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
- **SW fetch handler MUST scope cache lookup to its own cache.**
  Use `caches.open(CACHE).then(c => c.match(req))`, never the
  bare `caches.match(req)`. The bare form searches every cache
  name on the origin, so when a new SW has installed (and
  pre-populated its own `ipso-v<N+1>` cache) while the old SW is
  still active, the old SW's fetch handler can return files from
  the *new* cache — mixing versions and breaking the page with a
  blank screen (an old `index.html` paired with a new `app.js`
  that references DOM IDs the old HTML doesn't have, for
  example).  Surfaced in v0.3.0; fixed in v0.3.1.
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
