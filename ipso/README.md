# ipso

Field-data-entry PWA for tree-marking sessions ("martellata"). Records
species, diameter (D, cm), height (h, m), and GPS coordinates per tree.
Outputs a CSV that drops into abies (once the matching import flow is
built on the abies side).

## Quick start (developer)

```bash
make reference   # generate reference.json from bosco/data/*.csv
make test        # run node tests
make serve       # local preview at http://localhost:8000/
make deploy      # rsync to ipso.laforesta.it (requires DNS+TLS pre-provisioned)
```

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
the compresa, and the particella. Tap **Inizia**. Only high-forest
parcels appear in the particella pulldown (coppice parcels are filtered
out).

**Recording screen**.

- Top bar shows the parcel and the current GPS reading. The dot is
  green when accuracy is good (≤ 10 m), yellow when usable (≤ 25 m),
  red otherwise. You can save without a GPS fix — the row is recorded
  with empty coordinates.
- Pick a species. The pulldown remembers your last choice between
  trees.
- **Numero** is the operator-assigned tree number (e.g. painted on
  the trunk). Up to 4 digits. The field may be left blank. If you
  type a numero `N`, the next entry's numero pre-fills to `N + 1`.
  Trees with `D ≤ 17 cm` are not physically numbered, so their
  stored numero is auto-blanked on save regardless of what you
  typed — the counter ignores them so the next visible-numbered
  tree continues the sequence.
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
  every tree in entry order with number, species, group, D and h.
  Tap **Chiudi** to return to data entry; all in-progress fields
  are preserved.
- The "ultimo" pill at the top shows the most recent record. Tap the
  button on the right to delete it (e.g. after a typo) and re-record.
  Only the most recent record can be undone — earlier records are
  immutable from the field side.

Every 20 saved trees, the app automatically downloads a backup CSV
(named `..._backup_20.csv`, `..._backup_40.csv`, ...) to your
Downloads folder. This is a safety net — the authoritative export
still happens at session end.

**Termina e esporta CSV** writes the final CSV (named with compresa,
particella, date, and time) to Downloads and ends the session.

**Resume**. If the app is force-killed mid-session, the next open will
show a Sessioni non chiuse modal listing the unfinished sessions. Each
offers Riprendi (continue), Esporta CSV (close and download), or
Scarta (discard).

## CSV format

UTF-8 with BOM, semicolon-separated, comma decimal, CRLF line endings,
Italian date format `DD/MM/YYYY`. Column order:

```
Data;Compresa;Particella;Catastrofata;Numero;Specie;D_cm;H_m;H_measured;Lat;Lng;Acc_m;Operatore
```

- `Catastrofata` is `1` for storm-damage sessions (Particella is
  empty in that case), `0` otherwise.
- `Numero` is the operator-assigned tree number. Empty when the
  operator left the field blank, and also empty for trees with
  `D ≤ 17 cm` (small trees aren't physically numbered).
- `D_cm`, `H_m`, `Acc_m` are integers (no decimal).
- `H_measured` is `1` if the operator typed or edited the height; `0`
  if the auto-h value was accepted unchanged.
- `Lat`, `Lng` are decimal degrees with 6 fractional digits (comma
  decimal). Empty if there was no GPS fix at save time.
- `Specie` is the verbatim Italian common name from the species
  pulldown, including `Pino Nero` and `Pino Marittimo` (kept distinct
  because their ipsometric regressions diverge significantly).
- The in-app **Gruppo** tag is NOT exported. It is a working aid
  for the operator only.

## Source data

Generated at build time by `tools/build_reference.py` from:

- `../bosco/data/particelle.csv` (parcels, filtered to high-forest)
- `../bosco/data/equazioni_ipsometro.csv` (regression coefficients)

Species list is hardcoded in the build script, mirroring
`../abies/apps/base/management/commands/import_reference.py`, with `Pino`
split into `Pino Nero` and `Pino Marittimo`.

## Layout

```
index.html               app shell
manifest.webmanifest     PWA manifest
sw.js                    service worker (offline shell)
style.css                all styles
app.js                   state machine and UI wiring
store.js                 IndexedDB wrapper
session.js               session-level pure helpers
csv.js                   CSV serialisation
ipso.js                  ipsometric regression lookup
gps.js                   geolocation manager
numpad.js                custom on-screen numeric keypad
download.js              browser-download helper
strings.js               Italian UI strings
reference.json           bundled reference data (generated)
img/                     icons (f.gif, l.gif, icon-192.png, icon-512.png)
tools/build_reference.py reference-data build script
tests.js                 node tests for pure-logic modules
Makefile                 reference / test / serve / deploy targets
```
