# Abies: integrated management of forestry company operations

Abies is a full-stack web app used to manage production operations of a forestry
company, including forest harvests, sawmill production, biomass energy
generation.

It is intended to help with day to day operations, track production, and monitor
forest health and productivity.

# User base

The target audience is office workers and field staff at a small (40-employee)
lumber company in Italy that also produces electricity from biomass and solar.
They currently use Microsoft 365 Business and complain about its slowness. They
are unsophisticated computer users. They employ Word and Excel daily but only
use basic features (e.g., they do not know how to set up Excel pivot tables).
Simplicity and speed are key requirements.

# Priorities

1. Correctness
2. Security
3. Speed (the UI is snappy and works well over 3G)
4. Low maintenance over time

# Functional overview

Version 1 of the app covers the following functional areas, which we call _domains_:

- "Bosco": forest visualization and planning. Geospatial tool that displays
  historical harvest data, supports planning and execution of forest surveys
  (setting up sampling locations; recording geo-referenced annotations; etc.),
  and displays other geo-located information about the forest, including
  satellite imagery and health metrics.

- "Prelievi": forest harvesting. A log of the daily activities of crews of
  lumberjacks, including how much wood of different species was harvested where
  and with what tractors. The tool also supports day-to-day operations by
  generating PDF record slips that the crews fill out and provide to the office
  staff for data entry.

Future versions may also cover the following domains.

- "Segheria": sawmill. A log of daily sawmill operations, including volume and
  type of wood products, maintenance activities, failures or other incidents,
  etc., as well as monthly rollups of income and other parameters.

- "Biomassa": biomass plant. A log of daily biomass operations (amount of
  biomass consumed, energy produced, various operational parameters), as well as
  monthly rollups of energy produced, energy consumed, income broken down by
  various parameters, etc.

- "Fotovoltaico": photovoltaic plant. Daily log of production, monthly log of
  verified production and revenue.

- "Rifornimenti": fuel facility. Log of operations on the company's diesel fuel
  tank: who refueled what vehicle, how much fuel they used, refueling of the
  fuel tank itself, etc.

Each of these domains is handled in a distinct page of the app and is separate
from the others. However, the outputs of some areas are inputs to others (for
example, wood from the forest flows into the sawmill and biomass plant).

Historical data can be displayed in searchable tables and in graphical charts,
as well as in the forest maps.

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

As mentioned previously, core relational data is stored in SQLite on the server.

SQLite is appropriate because:
  - The app has a small user base, and is read-mostly.
  - WAL mode handles concurrent reads and writes fine at this scale.
  - No moving parts: no database server process to maintain.
  - Simple backups (copy one file).
  - Django's ORM abstracts the DB, so migrating to Postgres later is
    straightforward.

In addition, summary statistics are stored as compressed JSON digest files on
the server filesystem for consumption by visualization tools (charts and maps).
These digests are regenerated lazily on read (see `docs/data-architecture.md`).

Satellite imagery (GeoTIFF) and GeoJSON geometries (for geo data) are also
stored as files on disk.

## Concurrency

We handle concurrent modification of database data with row-level optimistic
locking. Every table has an implicit version column (not listed in the schemas
below) that increments on every save. The edit includes the version as a hidden
field; the server checks the version on POST, and if the version has changed
that indicates a write conflict.

## Disconnected operation

The app does not support disconnected operation but works well on relatively
low-bandwidth (e.g., 3G) connections thanks to pervasive caching, compressed
pre-processed data files, and client-side filtering.

## Backups

A cron job on the server runs a nightly backup of the database.

Compressed nightly backups are retained for a month, and one copy a month
(arbitrarily on the first day of the month) is uploaded to OneDrive using the
Microsoft Graph API.

Setting up Microsoft Graph OAuth credentials is part of the initial
configuration flow of the server app (in the shell, not via a web UI).

## Data import

Importing data is done through single-use ETL scripts that parse existing CSV
files and normalize them.

## Data export

Every domain page includes a button (upper right) that allows its data to be
exported as a CSV file.

## Database model

Full schema — every table with fields, types, and notes on invariants — is
in [`docs/database.md`](docs/database.md).

Conventions that apply across the board:

- All tables carry implicit `version` (int), `created_at`, and `modified_at`
  columns.  Concurrent edits are handled via row-level optimistic locking on
  `version`; the edit request includes the version as a hidden field and the
  server rejects stale submissions.
- Mutable domain tables (harvest_op, user, crew, tractor, species) are tracked
  by django-simple-history for audit.
- Crews, tractors, and species are never deleted — they are deactivated via
  an `active` flag.  Harvest-op deletion cascades to its harvest_species and
  harvest_tractor junction rows.
- Species and tractor percentage sums (100 per harvest_op) are enforced by
  client-side JS and server-side Django validation, not by SQL constraints.
- Per-domain JSON digests (prelievi.json, parcels.json, etc.) are documented
  in each page's own doc (`docs/page-*.md`).

# Internationalization

The app initially only supports Italian. The URL paths are in Italian also.

But there are no inline strings in the code. All are named constants (in both
Python on the backend and JS on the front-end) to make a future
internationalization easier. Path names are also named constants. The
assumption is that any future language choice will be at the level of the entire
app, not individual users.

## String constant scheme

On the backend, `config/strings_it.py` defines all user-facing string constants
(model verbose names, form labels, error messages, etc.) as module-level
variables. `config/strings.py` re-exports the active language module:

    from config.strings_it import *

To switch language, change this single import to point at a different language
file (e.g., `config/strings_en`). All code imports strings via
`from config import strings as S` and references constants like `S.PARCEL`.

On the frontend, the same pattern applies: `strings_it.js` defines all
constants, `strings.js` re-exports the active language.

Numbers and dates are represented using Italian locale.

Exported CSV uses semi-colons as separators.

The code itself (variable names, function names, etc.) is all in English. We
use terms like 'coppice' in the code instead of 'ceduo', and so on.

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

Client-side code uses ES modules (`<script type="module">`). The shell page
loads `base/js/app.js` as its sole entry point. `app.js` imports shared modules
(router, cache, api, forms, etc.) and each domain's page module.

All domain modules are loaded at boot, not lazy-loaded per tab click. Total JS
is small (order of tens of KB), and paying the load cost once per session keeps
tab switching instant.

Each domain module exports a page class with a known interface: `mount(params)`,
`unmount()`, `onQueryChange(params)`. `app.js` registers these in a static route
table keyed by URL path.

Release builds use minified JS, produced by a `make minify` Makefile target.

## Static file conventions

Each Django app owns its static files under `static/<app_name>/`, following
Django's default namespacing convention. Templates follow the same pattern
(`templates/<app_name>/`). Common CSS and JS live in `base`.

# Code location, deployment, and releases

The Django project is rooted at `laforesta/abies` in the code repository.

Production deployment lives at the root of its own subdomain
(`https://abies.laforesta.it/`); a parallel dev instance at
`https://abies-dev.laforesta.it/` shares the same VM but with separate data,
env file, and a basic-auth gate. Apache fronts both, reverse-proxying to the
container's gunicorn on 127.0.0.1.

Host-side state on the VM (Apache vhosts, Docker engine, certbot/Let's
Encrypt via DNS-01, the bind-mount data/static/backup directories) is
provisioned by `../../system/ansible/foresta.yml`. Ansible knows nothing
about abies images, builds, env files, or the container lifecycle; its job
ends at "make the VM capable of *hosting* an abies container."

Releases are deployed from the laptop via `bin/deploy <dev|prod> [git-ref]`,
which drives the VM's Docker daemon over a docker context. The script does
backup → build → stop → migrate → collectstatic → up; no source tree, no
git, no env file ever lands on the VM. Container env vars come from
`compose/.env.{prod,dev}` on the laptop (gitignored; copy from the
`.example` templates). One-time setup:

```sh
REMOTE_HOST=<hostname>
cd ~/src/laforesta/abies
docker context create vm-abies --docker host=ssh://${USER}@${REMOTE_HOST}
cp compose/.env.dev.example  compose/.env.dev   # then fill in values
cp compose/.env.prod.example compose/.env.prod  # then fill in values
```

Subsequent deployments:
```sh
cd ~/src/laforesta/abies
bin/deploy dev          # deploy current working tree to abies-dev
bin/deploy prod v0.1.0  # deploy a tagged release to abies-prod
```

Official prod releases are numbered with git tags; `bin/deploy prod <tag>`
refuses to run with a dirty working tree.

Local dev instance runs without containers (`make dev` + `python manage.py
runserver`).

The VM dev instance (`bin/deploy dev`) deploys the current local working tree --
uncommitted changes go into the container, which is the point.

`bin/deploy` runs the full sequence (backup → build → stop → migrate →
collectstatic → up) against the VM via docker context. The compose files, env
files, and source tree all live locally; the VM holds only the running
container, the data volumes, and the apache vhosts in front.

The container bind-mounts host paths for:

- SQLite database file (`/var/lib/abies-{prod,dev}/data`);
- Pre-processed JSON digests (subdir of data);
- Static files (`/var/lib/abies-{prod,dev}/staticfiles`);
- DB backups written by `bin/deploy` (`/var/backups/abies-{prod,dev}`).

# Testing

Python tests use pytest-django with pytest-cov for coverage reporting
(--cov-report=term-missing), matching the existing pdg-2026 test setup. Tests
cover models, views, form validation, and ETL scripts.

Client-side JS tests use the existing `node tests.js` pattern (see bosco/b/).

No browser-based E2E testing framework in v1. UI is verified by manual smoke
testing.

The test instance of Abies is deployed locally and does not use Docker, to speed
up testing and debugging.

# Development environment

The dev instance runs directly on the host (no Docker) using `manage.py
runserver` and a Python virtualenv (latest Python, currently 3.14).

The `data/` directory at the project root holds `db.sqlite3`, `digests/`, and
`geo/`, identically to production but without Docker mounts.

Makefile targets for dev setup:

- `make migrate`: run Django migrations to create/update tables.
- `make import`: run ETL scripts (`ingest/`) to populate `db.sqlite3` from
  existing CSVs in `bosco/data/`.
- `make geo`: import geo data into `data/geo/` using existing bosco scripts.
- `make digest`: precompute all JSON digests. Calls `apps/base/digests.py`
  to avoid divergence with the lazy-on-read production path. Also useful for
  debugging digest generation outside of the request cycle.
- `make admin`: create the initial admin user (prompts for username/password).
- `make dev`: runs `migrate`, `import`, `geo`, `digest`, and `admin` in
  sequence — one command to go from zero to a working dev instance.
