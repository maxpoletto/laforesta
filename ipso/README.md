# ipso

Field-data-entry PWA for tree-marking sessions ("martellata"). Records
species, diameter (D, cm), height (h, m), and GPS coordinates per tree.
Outputs a CSV that drops into abies (once the matching import flow is
built on the abies side).

## Quick start (developer)

```bash
make build       # stage src/ + generated artefacts into build/
make test        # build → run node tests
make serve       # build → local preview at http://localhost:8000/
make deploy      # build → test → rsync build/ to ipso.laforesta.it
make deploy-test # deploy with test polygons (test/test.geojson) +
                 # a synthetic reference.json for the test compresa
make icons       # rebuild src/img/ icons from ../logo/logo-grande.png
make clean       # rm -rf build/
```

Everything that ships lives in `build/` — assembled from `src/` plus
three generated artefacts (`reference.json`, `geo.js`,
`terreni.geojson`). `build/` is gitignored; a fresh clone produces it
with `make build`. The `src/`, `test/`, and `tools/` trees are never
mutated by build steps.

`make deploy-test` is for in-the-field testing of GPS auto-detect
against a small set of polygons near the developer
(`test/test.geojson`). It builds normally, then overlays the test
polygons and a synthetic `reference.json` whose `parcels` list comes
from those polygons (so the test compresa appears in the pre-session
pulldown). Deploys to the same host as `make deploy`; bump
`APP_VERSION` or clear the phone's site data first so the SW
refetches the polygons.

The plan that drove the design is at
`~/.claude/plans/proud-humming-kite.md`.

## Installing on an Android phone

1. Open Chrome on the phone.
2. Visit https://ipso.laforesta.it/ over Wi-Fi at least once.
3. Chrome's menu → "Add to Home Screen" → name "Ipso" → confirm. A green
   tree icon appears on the home screen.
4. Open the app from the icon — it runs full-screen as a standalone PWA.

If a previous version of ipso was installed and storage is suspect, in
Chrome go to Site settings → ipso.laforesta.it → Storage → Clear, then
re-add to home screen.

**If the home-screen icon doesn't update after a deploy** (e.g., still
shows a generic letter, or the wrong artwork), it's the service worker
caching the old manifest. Removing-and-readding the shortcut is *not*
enough — the SW survives that. Fix with: Site settings →
ipso.laforesta.it → Clear & reset → revisit the URL → Add to Home
Screen. Same applies to Firefox (Settings → Site permissions →
Clear). Caveat: this also wipes IndexedDB, so export any in-progress
session first.

## Using the app

**Pre-session screen**. Enter your name, the date (defaults to today),
and the compresa. Tick **Piante catastrofate** if this is a storm-
damage roaming session. Tap **Inizia**. The particella is no longer
picked here — the app reads it from GPS during recording (see below).

**Recording screen**.

- Top bar shows your current parcel (auto-detected from GPS) and the
  current GPS reading. The dot is green when accuracy is good
  (≤ 10 m), yellow when usable (≤ 25 m), red otherwise. The parcel
  label updates as you cross polygon boundaries — but only after the
  GPS confidently shows you're inside a new parcel (a few seconds of
  stable readings). If you're outside any parcel of the configured
  compresa, the label reads **Fuori dai confini**. You can save
  without a GPS fix — the row is recorded with empty coordinates and
  an empty particella unless you set one manually.
- Pick a species. The pulldown remembers your last choice between
  trees.
- **Numero** is the operator-assigned tree number (e.g. painted on
  the trunk). Up to 4 digits. The field may be left blank. If you
  type a numero `N`, the next entry's numero pre-fills to `N + 1`.
  Trees with `D ≤ 17 cm` are not physically numbered, so their
  stored numero is auto-blanked on save regardless of what you
  typed — the counter ignores them so the next visible-numbered
  tree continues the sequence. The counter also persists across
  sessions per operator (matching on name, case- and whitespace-
  insensitive): if you finish today's session at tree `N`, the
  first tree of your next session pre-fills to `N + 1` instead of
  blank. Deleting the most recent tree rolls the counter back to
  match the new highest numero in the session — so if the trees
  are `101, 102, 103, 110` and you delete `110`, the next number
  becomes `104`.
- **Particella** defaults to **(automatica)** — the pulldown closed
  shows the parcel the GPS sees you in, or a dash if you're outside
  the compresa boundaries. To override (e.g. when the GPS lags or
  you know you're standing right on a boundary), open the pulldown
  and pick a specific parcel; your choice **sticks** across
  subsequent trees until you re-open the list and pick
  **(automatica)** at the top to resume GPS tracking. A **red
  border** lights up whenever your manual choice doesn't match where
  the GPS thinks you are — that's a hint that either the GPS needs
  to settle, or your manual pick is wrong for the current tree.
- **Gruppo** is an A–Z working tag for grouping trees during a
  session (e.g. a cluster of trees marked for the same buyer).
  Sticky across saves; not exported to the CSV.
- Enter D (diameter in cm) using the on-screen number pad. h (height
  in m) auto-fills from the ipsometric regression for the current
  compresa and species. If a regression is missing for this combination
  (e.g. Capistrano + Faggio, or any region + Ontano) a hint appears
  and you must enter h manually.
- Tap on the h field and type to override the auto-h. A flag is set
  on the saved record so the office knows the operator measured this
  height rather than accepted the regression's estimate.
- Tap **Salva e prossimo**. The record is written to the device's
  local database before the screen advances; D and h reset; species
  and gruppo stay selected; numero pre-fills to the next default.
- **Visualizza dati raccolti** opens a snapshot view of the trees
  recorded so far in the current session. Top table: per-group
  counts (only A–Z groups with at least one tree). Bottom table:
  every tree in entry order with number, species, particella, group,
  D and h — so you can verify that the parcel attribution looks
  right per tree. Tap **Chiudi** to return to data entry; all
  in-progress fields are preserved.
- The "ultimo" pill at the top shows the most recent record. Tap the
  button on the right to delete it (e.g. after a typo) and re-record.
  Only the most recent record can be undone — earlier records are
  immutable from the field side.

Every 20 saved trees, the app automatically downloads a backup CSV
(named `..._backup_20.csv`, `..._backup_40.csv`, ...) to your
Downloads folder. This is a safety net — the authoritative export
still happens at session end.

**Termina e esporta CSV** writes the final CSV (named with the
compresa and a date+time stamp; catastrofate sessions get a
`_catastrofate` suffix) to Downloads and ends the session.

After Termina, the app also uploads the same CSV to the server (the
office sees it in the inbox without manual handoff).  A progress
screen shows the attempt counter; if there's no signal, it retries
every few seconds (capped at 30 s).  You can hit **Annulla
caricamento e salva solo sul telefono** to keep the local copy and
move on — the upload will be offered again at the next app open if
you change your mind.

**Resume**. If the app is force-killed mid-session, the next open will
show a Sessioni non chiuse modal listing the unfinished sessions. Each
offers Riprendi (continue), Esporta CSV (close and download), or
Scarta (discard).

## CSV format

UTF-8 with BOM, semicolon-separated, comma decimal, CRLF line endings,
Italian date format `DD/MM/YYYY`. Column order:

```
Data;Compresa;Particella;Catastrofata;Numero;Specie;D_cm;H_m;H_measured;Lat;Lon;Acc_m;Operatore
```

- `Particella` is the parcel where this particular tree was marked —
  GPS-detected by default, or whatever the operator chose if they
  overrode the automatic value. May be empty when the tree was
  marked outside any parcel of the configured compresa AND the
  operator left auto mode on; the row's `Lat`/`Lon` still pin the
  exact position.
- `Catastrofata` is `1` for storm-damage sessions, `0` otherwise.
  This column is independent of `Particella` — catastrofate rows
  still carry the actual particella where the tree was marked.
- `Numero` is the operator-assigned tree number. Empty when the
  operator left the field blank, and also empty for trees with
  `D ≤ 17 cm` (small trees aren't physically numbered).
- `D_cm`, `H_m`, `Acc_m` are integers (no decimal).
- `H_measured` is `1` if the operator typed or edited the height; `0`
  if the auto-h value was accepted unchanged.
- `Lat`, `Lon` are decimal degrees with 6 fractional digits (comma
  decimal). Empty if there was no GPS fix at save time.
- `Specie` is the verbatim Italian common name from the species
  pulldown, including `Pino Nero` and `Pino Marittimo` (kept distinct
  because their ipsometric regressions diverge significantly).
- The in-app **Gruppo** tag is NOT exported. It is a working aid
  for the operator only.

## Source data

Three artefacts are produced into `build/` during `make build`:

- `build/reference.json` from `tools/build_reference.py`, which reads
  `../bosco/data/particelle.csv` (parcels, filtered to high-forest)
  and `../bosco/data/equazioni_ipsometro.csv` (regression
  coefficients). The species list is hardcoded in the build script,
  mirroring `../abies/apps/base/management/commands/import_reference.py`,
  with `Pino` split into `Pino Nero` and `Pino Marittimo`.
- `build/terreni.geojson` — parcel polygons used for GPS-driven
  auto-detection, copied verbatim from `../bosco/data/terreni.geojson`.
- `build/geo.js` — point-in-polygon and label helpers, vendored from
  `../abies/apps/base/static/base/js/geo.js` by `tools/vendor_geo.py`.
  abies is the authoritative source so the two apps stay in sync.

`make deploy-test` builds the same way, then overlays
`test/test.geojson` (replacing `build/terreni.geojson`) and a
synthetic `build/reference.json` produced by
`tools/build_test_reference.py` from those test polygons.

## Layout

```
src/                  handwritten browser source (all committed)
  index.html          app shell
  manifest.webmanifest  PWA manifest
  sw.js               service worker (offline shell)
  style.css           all styles
  app.js              state machine and UI wiring
  store.js            IndexedDB wrapper
  session.js          session-level pure helpers
  csv.js              CSV serialisation
  ipso.js             ipsometric regression lookup
  gps.js              geolocation manager
  numpad.js           custom on-screen numeric keypad
  download.js         browser-download helper
  strings.js          Italian UI strings
  parcel-locator.js   GPS auto-detect + sticky-override state
  img/                icons (f.gif, l.gif, icon-*.png)

test/                 tests and field fixtures (committed)
  tests.js            node tests for pure-logic modules
  test.geojson        polygons near the developer for in-the-field testing

tools/                build scripts (committed, never shipped)
  build_reference.py        reference.json from bosco/data CSVs
  build_test_reference.py   test reference.json from test.geojson
  vendor_geo.py             geo.js from abies's ES-module copy
  build_icons.py            src/img/ from ../logo/logo-grande.png

build/                deploy artefact — gitignored, regenerable
  ...                 copy of src/ + generated reference.json, geo.js,
                      terreni.geojson

Makefile              build / test / serve / deploy / deploy-test
```
