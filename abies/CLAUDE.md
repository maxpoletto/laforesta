# Abies: integrated management of forestry operations

Abies is a full-stack web app to manage forestry operations end-to-end, focused
for now on the Italian forestry industry.

Goals: monitor forest health; plan and assess harvests over multi-year periods;
enable timber traceability; support day-to-day forestry operations such as
timber cruising and harvest recording.

The initial customers are staff at a 40-person lumber company in Calabria.
Simplicity and speed are key requirements.

# Priorities

1. Correctness
2. Security
3. Speed (the UI works well over 3G)
4. Low maintenance over time

# Functional overview

The app features the following main functional areas ("domains"):

- "Forestscope" ("Boscoscopio"): geospatial tool to visualize forest
  characteristics (age, type, productivity, etc), including satellite imagery
  and related health metrics.

- Harvest plans ("Piani di taglio"): scheduling of harvests over multi-year
  timeframes, in compliance with regional forest agency guidelines.

- Samples ("Campionamenti"): tools for systematic assessments of forest
  productivity (generation of sampling grids, recording of data from timber
  cruising).

- Harvests ("Prelievi"): support for daily felling operations, including
  detailed activity logs and receipts for lumber crews.

# Architecture overview

The app is a SPA-lite: Django serves a single shell page after authentication,
and vanilla JS handles client-side routing, data caching, and content rendering.
There is no JS framework.

- Vanilla JS on the client, with minimal dependencies:
  - @maxpoletto/sortable-table for tabular data display
  - Chart.js for data visualization
  - Leaflet for mapping
- Django on the server, served via Apache with mod_wsgi:
  - JSON endpoints for data (consumed by sortable-table, Chart.js, and Leaflet)
  - HTML fragment endpoints for forms (injected into the shell)
  - django-allauth for authentication
  - django-simple-history for audits
- SQLite for storage.

# Docs index

Read the doc for the topic you are working on.  Cross-cutting concerns
(project structure, string constants, storage, deployment, testing) stay in
this file.

## Architecture and design

Read **before touching** the relevant area — these are prescriptive, not
just reference.

- [`docs/data-architecture.md`](docs/data-architecture.md) — JSON digest
  serving, staleness/regeneration, client-side caching, data entry
  POST/response protocol, optimistic table updates.  **Read before writing
  or modifying any view that reads or writes data.**
- [`docs/ui-architecture.md`](docs/ui-architecture.md) — SPA shell, header,
  client-side routing, error reporting.  **Read before changing navigation,
  the shell template, or URL handling.**
- [`docs/ui-design-patterns.md`](docs/ui-design-patterns.md) — visual rules
  (fonts, colors, tables, charts, maps, modals, form card, button layout,
  field widths, read-only fields).  **Read before building or modifying any
  UI component.**
- [`docs/security.md`](docs/security.md) — permission model, auth flow,
  OAuth configuration, rate limiting, CSP, auditing.  **Read before touching
  auth, user management, or any endpoint that handles credentials.**

## Reference

- [`docs/glossary.md`](docs/glossary.md) — Italian forestry terms used in
  column names, UI labels, and variable names.
- [`docs/database.md`](docs/database.md) — full DB schema (every table with
  fields, types, and invariants).

## Per-page specs

Each page doc covers its URL, query parameters, visual layout, and JSON
digests.  **Read the page doc for the domain you are working on.**

- [`docs/page-login.md`](docs/page-login.md) — Login page
- [`docs/page-bosco.md`](docs/page-bosco.md) — Bosco (map) page
- [`docs/page-campionamenti.md`](docs/page-campionamenti.md) — Campionamenti (sampling) page
- [`docs/page-piano-di-taglio.md`](docs/page-piano-di-taglio.md) — Piano di taglio (harvest plan) page
- [`docs/page-prelievi.md`](docs/page-prelievi.md) — Prelievi (harvests) page
- [`docs/page-controllo.md`](docs/page-controllo.md) — Controllo (audit) page
- [`docs/page-impostazioni.md`](docs/page-impostazioni.md) — Impostazioni (settings) page

## Plans

- [`docs/implementation-plan.md`](docs/implementation-plan.md) — original
  M0..M4 milestone plan
- [`docs/piano-di-taglio-plan.md`](docs/piano-di-taglio-plan.md) —
  fine-grained Piano-di-taglio plan (supersedes M1/M2 in
  implementation-plan.md)

# CSS

No style is defined in-line in HTML. All style is defined in CSS files. 

Most styles are maintained in a single common.css style file. Styles that are
only required on one page appear in a page-specific style file.

# Javascript dependencies

All JS dependencies (Leaflet, Chart.js, sortable-table, (Google) fonts, etc.)
are minified and vendored, served from Django's static/vendor. There are no
external dependencies at runtime. A Makefile target (`make update-vendor`)
re-copies from source when needed.

# Storage

SQLite (WAL mode). Digests are compressed JSON on the filesystem
(see `docs/data-architecture.md`). GeoTIFF and GeoJSON on disk.

Row-level optimistic locking via a `version` column that increments on
every save. Edit forms include `version` as a hidden field; the server
rejects stale submissions.

## Database model

Full schema in [`docs/database.md`](docs/database.md). Cross-cutting conventions:

- Implicit `version`, `created_at`, `modified_at` on all tables.
- Mutable domain tables tracked by django-simple-history.
- Validation is done client-side (quick feedback) and in Django (authoritative);
  only a small subset of constraints are enforced in the schema (see `docs/database.md`).
- Per-domain JSON digests documented in each `docs/page-*.md`.

# Internationalization

Currently Italian (UI, URL paths, locale for numbers/dates, CSV `;` separator),
but the architecture supports switching to other languages.

No inline strings. All user-facing text is a named constant. The pattern
applies at three levels:

- **Python**: `config/strings_it.py` defines constants; `config/strings.py`
  re-exports via `from config.strings_it import *`. Import as
  `from config import strings as S` (e.g., `S.PARCEL`).
- **JS**: `strings_it.js` defines constants; `strings.js` re-exports via
  `export * from './strings_it.js'`.
- **Templates**: `foo_it.html` is the real file; `foo.html` is a symlink
  to `foo_it.html`.

## Number formatting

Numbers are stored and transmitted **canonical** (dot decimal, no digit
grouping): DB columns, JSON digests, CSV files, and `grid/save-auto` payloads.
Localization happens **only at the edges**, driven by the active locale
(`settings.LANGUAGE_CODE` → `<html lang>` → JS reads
`document.documentElement.lang`):

- **Display is centralized.** JS goes through `format.js`
  (`fmtDecimal`/`fmtCoord`/…, built on `Intl.NumberFormat`); server-rendered
  numbers use Django's locale-aware `{{ value|floatformat:N }}`. The decimal
  *count* is per quantity (coords 5, mass 1, …) and locale-independent — only
  the separator is localized.
- **Form inputs** can't be `type="number"` (that is dot-only and follows the
  OS locale, not the app's), so decimal inputs are
  `type="text" inputmode="decimal"`, rendered with `floatformat:N` and parsed
  back with `apps.base.formats.normalize_decimal()` (wraps Django
  `sanitize_separators`) before `float()`/`Decimal()`. This applies to **form
  strings only** — never to CSV rows or JSON payloads, which are already
  canonical (in the it locale `sanitize_separators("38.6")` reads `.` as a
  thousands separator → `"386"`). Integer inputs stay `type="number"`.

To add a language: create parallel `*_en.*` files, retarget the re-exports /
symlinks, and set `LANGUAGE_CODE`. Numbers localize automatically via
`Intl`/Django — no per-field or per-call-site changes.

Code itself (variables, functions) is in English: 'coppice' not 'ceduo'.

# Mobile

The app is usable on mobile in portrait mode (without needing to switch to
landscape).  Page-specific mobile adaptations are covered in each page's doc
(`docs/page-*.md`).

# Project structure

## Directory layout

    abies/
    ├── manage.py
    ├── config/                         # Django project settings
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── apps/                           # Django apps
    │   ├── base/                       # shared models, shell, common JS/CSS
    │   │   ├── models.py               # region, eclass, parcel, sample_area,
    │   │   │                           # crew, tractor, species, optype, note
    │   │   ├── views.py                # shell view, login view
    │   │   ├── urls.py
    │   │   ├── migrations/
    │   │   ├── digests.py              # all JSON digest generation
    │   │   ├── templates/base/
    │   │   │   ├── shell.html          # the long-lived SPA shell
    │   │   │   └── login.html
    │   │   └── static/base/
    │   │       ├── css/common.css
    │   │       ├── js/
    │   │       │   ├── app.js          # boot: imports domains, inits router
    │   │       │   ├── router.js       # pushState / popstate / route table
    │   │       │   ├── cache.js        # in-memory cache, conditional GETs
    │   │       │   ├── api.js          # fetch helpers
    │   │       │   ├── forms.js        # form fetch, intercept, validate, submit
    │   │       │   ├── modals.js       # error and help modals
    │   │       │   ├── table.js        # sortable-table wrapper + CSV export
    │   │       │   └── strings.js      # Italian UI string constants
    │   │       └── vendor/             # vendored: Leaflet, Chart.js,
    │   │                               # sortable-table, Roboto
    │   ├── prelievi/                   # Prelievi domain
    │   │   ├── models.py               # harvest_op, harvest_species, harvest_tractor
    │   │   ├── views.py                # JSON endpoints, form fragments, POST
    │   │   ├── urls.py
    │   │   ├── templates/prelievi/
    │   │   │   └── _form.html          # add/edit form fragment
    │   │   └── static/prelievi/
    │   │       ├── css/prelievi.css
    │   │       └── js/prelievi.js      # PrelieviPage class
    │   ├── bosco/                      # Bosco domain
    │   ├── controllo/                  # Audit domain
    │   └── impostazioni/               # Settings domain
    ├── ingest/                         # one-time ETL / data import scripts
    ├── data/                           # host-mounted runtime dir (gitignored)
    │   ├── db.sqlite3
    │   ├── digests/                    # pre-computed JSON files
    │   └── geo/                        # GeoJSON, satellite imagery
    ├── docs/
    ├── test/
    ├── Dockerfile
    ├── Makefile
    └── requirements.txt

Apps are organized under `apps/` with dotted paths in `INSTALLED_APPS` (e.g.,
`apps.prelievi`). `config/` holds the Django project settings.

## JS module conventions

ES modules (`<script type="module">`). `base/js/app.js` is the sole entry
point; it imports shared modules and each domain's page module (all loaded
at boot, not lazy). Each domain exports a page class: `mount(params)`,
`unmount()`, `onQueryChange(params)`. `make minify` for release builds.

## Static file conventions

Each Django app owns its static files under `static/<app_name>/`, following
Django's default namespacing convention. Templates follow the same pattern
(`templates/<app_name>/`). Common CSS and JS live in `base`.

# Deployment

Prod: `https://abies.laforesta.it/`; dev: `https://abies-dev.laforesta.it/`
(same VM, separate data, basic-auth gate). Apache reverse-proxies to
gunicorn. Host provisioning via `../../system/ansible/foresta.yml`.

Deploy from laptop via `bin/deploy <dev|prod> [git-ref]` over a docker
context. Sequence: backup → build → stop → migrate → collectstatic → up.
Container env vars from `compose/.env.{prod,dev}` (gitignored).

```sh
bin/deploy dev          # deploy current working tree to abies-dev
bin/deploy prod v0.1.0  # deploy a tagged release to abies-prod
```

Prod releases use git tags; `bin/deploy prod <tag>` requires clean tree.
Dev deploys the working tree (uncommitted changes included).
Local dev uses `manage.py runserver` directly (no Docker).

# Testing

Python: `make test` runs pytest-django with coverage (`--cov-report=term-missing`).
JS: `node tests.js` in individual app directories.
No browser-based E2E; UI verified by manual smoke testing.

# Development environment

Runs directly on the host (no Docker): `manage.py runserver` + Python 3.14
virtualenv. `data/` holds `db.sqlite3`, `digests/`, `geo/`.

- `make dev`: zero-to-working in one command (migrate + import + geo + digest + admin).
- `make migrate`: run Django migrations.
- `make import`: ETL from CSVs in `bosco/data/`.
- `make geo`: import geo data into `data/geo/`.
- `make digest`: precompute all JSON digests via `apps/base/digests.py`.
- `make admin`: create initial admin user.
