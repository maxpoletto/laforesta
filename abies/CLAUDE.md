# Abies: integrated management of forestry company operations

<!-- TOC (line numbers are approximate — re-run grep '^#{1,2} ' to refresh)
  User base .................. 36
  Priorities ................. 45
  Functional overview ........ 52
  Glossary ................... 93
  Architecture overview ..... 109
  Security .................. 126
  UI architecture ........... 179
  UI design patterns ........ 428
  Storage ................... 547
  Internationalization ...... 623
  Mobile .................... 655
  Project structure ......... 661
  Code location / deployment  743
  Testing ................... 760
  Development environment ... 774
  Relationship to bosco apps  795
  Build order ............... 801
  Detailed description ...... 852
  See also:
    docs/database.md              — full DB schema
    docs/pages/login.md           — login page
    docs/pages/bosco.md           — Bosco (map) page
    docs/pages/prelievi.md        — Prelievi (harvests) page
    docs/pages/controllo.md       — Controllo (audit) page
    docs/pages/impostazioni.md    — Impostazioni (settings) page
-->

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

# Glossary

- Particella: a parcel of forest land. Usually but not always contiguous. Atomic
  unit for harvest planning.
- Compresa: a forest region comprising multiple particelle.
- Mannesi: a lumberjack. Existing / legacy harvest table is 'mannesi.{xlsx,csv}'.
- VDP and Prot: "verbale di pesata" and "protocollo", id numbers of lumberjack
  harvest operations in mannesi file.
- Bosco apps: Current web-apps in laforesta/bosco. Includes bosco/ads (aree di
  saggio, sample areas), bosco/pai (piante ad accrescimento indefinito, trees to
  be preserved), boscoscopio (geo-based stats).
- PSR: Programma di Sviluppo Rurale. Funding plan for some harvest operations.
- Fitosanitario: a harvest operation performed for forest disease containment.
- Catastrofate: a harvest operation performed to remove trees damaged by
  weather events.

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

# Security

Data security (both privacy and integrity) is important because the app stores
the core of the company's operations.

## Permission model

We have a three-role permission model, "admin", "writer", and "reader". Readers
have read-only access. Writers can modify any data not related to access
control, performing insertions/edits/deletes on any tables that allow it (see
details below). "admins" can also create and edit users.

If using username/password pairs, users can change their own password.

## Authorization

The app is access-controlled on the server side.

Authorization supports MS 365 credentials and user/password pairs via
django-allauth,
[integrated](https://django-axes.readthedocs.io/en/latest/6_integration.html#integration-with-django-allauth)
with django-axes for basic brute-force protection.

OAuth must be pre-provisioned (the email address whitelisted) by an admin.

Session expiration is server-configurable and defaults to 12 hours.

In the future we may need to support other OAuth identity providers.

## Password policy

Django's default `AUTH_PASSWORD_VALIDATORS` are enabled: minimum length (8
characters), common password check, numeric-only check, similarity to username.

## Rate limiting

django-axes handles login brute-force protection. Data entry endpoints are
rate-limited per user (e.g., 60 requests/minute) to guard against runaway
scripts or bugs.

## Content security

The shell page sets a Content-Security-Policy header that blocks inline scripts.
All user-provided content (notes, descriptions, etc.) rendered into the DOM must
use `textContent`, never `innerHTML`, to prevent XSS.

## Auditing

The app records all writes using django-simple-history.

The audit log is readable and searchable by all users. (More on this below in
the UI section.)
 
# UI architecture

The app is structured as a SPA-lite. After authentication, Django renders a
single shell page that persists for the duration of the session. All subsequent
navigation happens client-side without full page reloads.

## Shell and header

The shell is a single Django template containing just:
- The header, shared by all domain pages.
- A content area where the domain-specific page content is rendered.

The shell is rendered once and never reloads during normal use.

The header is adaptive for desktop and mobile. On narrow displays it contains only:
- The logo of the company.
- The name of the currently active domain (Bosco, Prelievi, Segheria, Biomassa,
  Fotovoltaico, Rifornimenti, Controllo, Impostazioni)
- A hamburger icon for a menu that allows switching to other domains.

On wider displays it contains:
- The logo and name of the company.
- The names of the domains as tabs, with the currently active domain highlighted.
- No hamburger icon.

The header is fixed in the viewport. Content scrolls beneath it.

## Routing and URLs

URLs are human readable and are the canonical representation of the view state.
They encode the domain (in the path), and any data filters, chart selection,
etc. as query parameters.

Changing domain or page-specific parameters changes the URL via
`history.pushState()` and renders the appropriate content. The back button works
via `popstate`.

All URLs are bookmarkable. A user can send a URL to a colleague and it
reproduces what they were looking at.

Because of this design choice and the relatively flat tabbed navigation
structure, we omit any explicit "breadcrumb" navigation component.

The router also remembers the last-visited URL for each domain within the
session.  Clicking a *different* tab restores that tab's last URL (preserving
its filters / sort / chart state), so jumping Prelievi → Controllo → Prelievi
returns you to where you were.  Clicking the tab you are already on goes to
its bare URL — a deliberate "reset" affordance.  Browser back / forward use
the real history stack and are unaffected.  The stash is in-memory only;
a full page reload clears it.

The paths and query parameters for each domain are documented in the
per-page specs under `docs/pages/`.

## Data storage and serving

All data except some geographic data (e.g., satellite images) lives in
relational form in SQLite on the server.

However, to reduce latency, it is always served as compressed pre-computed JSON.
The mapping of relational tables to JSON files is specified in the detailed
domain descriptions.

For tabular data, the format is a JSON object with two fields:

    { "columns": ["row_id", "Data", "Compresa", ...],
      "rows": [[1, "2024-03-15", "Alpe", ...], ...] }

`columns` lists column names once; `rows` is an array of arrays containing pure
data. Every row's first element is `row_id`, used for updates (see below). The
client maps columns by name, so adding a column does not break cached data.

### JSON digest regeneration

Digests are regenerated lazily on read, not on write. This avoids wasted work
during batch data entry (where the inserting user's view is already kept current
via cache sync from the POST response).

A small table tracks staleness:

    digest_status(name TEXT PRIMARY KEY, stale BOOLEAN DEFAULT FALSE)

Write path: after a successful save, the view marks affected digests as stale:
`UPDATE digest_status SET stale = TRUE WHERE name IN (...)`. This is the only
write-path cost.

Read path (conditional GET for a digest):
1. Check stale flag for the requested digest (one PK lookup).
2. If not stale: normal If-Modified-Since check against file mtime, return 304
   or 200.
3. If stale: regenerate the digest, then serve the new file. Regeneration
   proceeds as follows:
   a. Generate the digest and write it to a temp file (gzip-compressed).
   b. `os.rename()` the temp file to the digest path (POSIX-atomic).
   c. Clear the stale flag: `UPDATE digest_status SET stale = FALSE WHERE
      name = ? AND stale = TRUE`. The `AND stale = TRUE` acts as a
      compare-and-swap so that concurrent readers don't regenerate twice.

   This ordering ensures that on any crash, the stale flag may over-report but
   never under-report: a digest file on disk is always complete and valid.

All digest generation logic lives in `apps/base/digests.py`, since digests
often span multiple domain tables (e.g., a prelievi write also updates
`parcel_year_production.json` used by bosco).

Digest files are gzip-compressed (stored as `.json.gz`) and served with
`Content-Encoding: gzip`.

## Caching

Fetched data is cached client-side in memory. The cache is keyed by a "data_id"
that identifies a particular dataset corresponding to a server-side JSON file,
e.g., "daily harvest operations" or "monthly sawmill production", and it stores
the last refresh time.

The maximum size of the cache is the sum of all tabular and pre-processed data,
on the order of 10-20 MB uncompressed, plus 10-100 MB of satellite imagery in
the worst case.

Page reloads clear all cache state.

When changing domain pages:

1. The app renders from cache immediately if data is available.

1. The app also sends background conditional GETs to the server with
   If-Modified-Since for every dataset displayed in the domain page. If the
   server returns 304 Not Modified, no action is required. If it returns 200 OK,
   the app updates the cache with the newly received data and re-renders. In
   both cases it updates the dataset's last refresh time.

Conditional GETs also run once every 5 minutes (on a timer) for every dataset
visible in a particular domain page (i.e., the page will refresh even if the
user performs no navigation).

Domain switching feels instant for previously-viewed data.

When a domain page is loaded for the first time (empty cache), a modal displays
"Caricamento..." until the initial data fetch completes. Subsequent cache
refreshes happen silently in the background.

## Data entry and cache updates

Data entry forms are Django-rendered HTML fetched as fragments into the shell's
content area.

Each form has custom HTML and validation JS as needed, but common patterns (form
interception, error display) are extracted into shared libraries.

Fields that are enum-like (correspond to finite sets of values defined within
the app itself) are implemented as pull-downs. These include worker names, crew
names, tractor names, and tree species names (see below for details).

The process of data entry runs as follows:

1. The user initiates a data addition or edit by clicking on a UI button (the
   visual details of this are below).
1. JS fetches the form HTML from Django (including the CSRF token and an
   idempotency nonce as a hidden field).
1. The form is rendered in the current page (it replaces the current view). The
   URL *does not change*. This is the one exception to the "canonical
   representation of view state" rule, since we never need to share the input
   form.
1. Client-side JS validation provides immediate feedback: the submit button is
   inactive until JS validation passes.
1. On submit, JS intercepts and forwards the POST request (including the CSRF
   token and idempotency nonce), and waits for a response.
1. The server checks the nonce: if it has already been used (i.e., a previous
   request with the same nonce succeeded), it returns the original success
   response without writing again. This prevents duplicate records when the
   network drops between server-commit and client-receive and the client retries.
   Used nonces are stored in the database with a timestamp. The nightly backup
   cron job also prunes nonces older than 24 hours.
1. The server provides authoritative validation.

The server response has one of three values. The payload is always JSON.

1. Success: Code = 200 OK, payload = { data_id: X, row_id: Y, "record": [
   Y, ... ] }

   The client updates row Y of entry X in the case with the new record and
   refreshes the tabular display.

   Note that a future background conditional GET might refresh other cache
   entries, such as those corresponding to digested data for graphs. Concrete
   example: user enters data corresponding to a new harvest operation. The
   tabular display of harvest operations updates immediately. Digested data for bar chart of
   monthly harvests might be loaded after the next conditional GET.

1. Validation error: Code = 400 Bad request, payload = { status:
   "validation_error", message: "...", html: "..." }

   The page displays the error message in a modal and again displays the blank
   HTML form given by the "html" field.

   This rarely happens, since most server-side validation is consistent with
   client-side validation.

1. Conflict: Code = 400 Bad request, payload = { status: "conflict", message:
   "...", data_id: X, row_id: Y, "record": [ Y, ... ], html: "..."}

   This error happens on attempted edit or delete. Another user edited or
   deleted the entry between the time when the current page's last cache refresh
   and the time of submit. The HTML contains the form populated with the current
   server state. If the record has been deleted on the server, the message
   provides this information and the user can click "submit" to re-add the
   record. If the user escapes out of the edit form, the cached data is updated
   with the returned record, as for a successful update.

## Data deletion and cache updates

If the user deletes a record (more on the UI details of this below), the UI
displays an alert warning that the action cannot be undone. 

If the user confirms, a POST is sent to the server as for data insertion / edit
above.

1. Successful responses contain a row_id but no record field. The client removes
   the given id from the cache.

1. No validation errors are possible.

1. Conflict means that a row was edited since last cache refresh. The response
   contains no HTML but a valid record. The cache is updated as for a successful
   update. The error message is displayed in a modal and the user has the chance
   to try deletion again.

## Error reporting

Errors are reported in modals. Errors include validation errors, conflicts, and
other conditions such as network errors.

An error modal contains the error message and a dismiss button. It can also be
dismissed with the "escape" key.

## CSS

No style is defined in-line in HTML. All style is defined in CSS files. 

Most styles are maintained in a single common.css style file. Styles that are
only required on one page appear in a page-specific style file.

## Javascript dependencies

All JS dependencies (Leaflet, Chart.js, sortable-table, (Google) fonts, etc.)
are minified and vendored, served from Django's static/vendor. There are no
external dependencies at runtime. A Makefile target (`make update-vendor`)
re-copies from source when needed.

# UI design patterns

## Objectives

The objectives of the visual design are:

- Readability: data is presented simply and clearly, with good use of screen
  real estate on both desktop and mobile.
- Predictability: consistent visual guidelines, no unexpected behavior.
- Discoverability: navigation is easy and fast.
- Restfulness: cognitive and visual load are low.

## Fonts and colors

Roboto is used throughout.

The UI is strictly two-dimensional: there are no drop-shadows, text inputs are
flat, scroll bars are flat.

Page margins are moderate (15 px) on desktop and almost disappear (2 px) on mobile.

Text inputs have very slightly rounded corners (2-4 px radius).

Buttons have rounded corners (4-8px radius). They are dark green and turn
lighter when hovered over.

Horizontal rules outline the page header as well as collapsible elements
(each page's collapsibles are documented in `docs/pages/`). They are thin
(4px), dark green, and rectangular.

## Data display overview

Within each domain page, information is displayed consistently in one of three
ways:
- Tabular data is consistently displayed in sortable-table (more on this below).
- Graphical data is displayed using Chart.js.
- Geographic data is displayed using Leaflet.

Some visual elements may be hidden within collapsible sections.

## Tabular data

All tabular data appears in sortable-tables.

- All fields are sortable.
- All tables are searchable via a text input immediately above the table, on the
  left. Search is immediate (no search button) after a 1/2-second sleep to
  debounce rapid keyboard input.
  - Conceptually (not necessarily in actual implementation), the search operates
    as follows:
    1. Split the search text on whitespace.
    1. For every row, join all fields into a single string.
    1. Return all rows that contain all elements of the search text in the given
        order.
  - The search acts purely as a filter. The table does not move, but displays only
    matching rows. Any pre-existing sort order is preserved.
- A table displays rows as far down as the bottom of the viewport. If there are
  more rows, the table has a scrollbar that is separate from the page scrollbar.
  (On mobile, there is enough lateral space to allow the user to also scroll the
  page, not just the table).
- Tables have 1px medium-grey borders and column headers have light grey background.

Additionally, for users with role "writer" or "admin", tables may (depending on their semantics — see the relevant page doc under `docs/pages/`) allow modification:

- Tables that allow row addition have a "+" button below the bottom row, on the
  right.
- Tables that allow row editing have a "pencil" icon on the right of each row.
- Tables that allow row deletion also have a "garbage can" icon on the right.

## Graphs and charts

All graphs and charts are implemented in Chart.js.

All charts have y-axes that begin at 0.

All color maps range from yellow-green (for low values) to dark green (for high values).

Graphs occupy the full screen width and legends appear below the graph (on both
desktop and mobile).

## Maps

Maps have the following visual structure:

- A navigation bar on the right.
- The following Leaflet tools appear in the upper left corner, top to bottom:
  - A hamburger button to hide/display the navigation bar.
  - A location pin.
  - Zoom +/- buttons
  - A ruler

The top of the navigation bar always contains, top to bottom:
- A status panel
- A map type selector. Buttons for OSM, Topo, Satellite. Satellite is the
  default.
- A pull-down region ("compresa") selector. Choosing a region centers it on the
  map and sets the zoom level to the most detailed level that still includes the
  full region in the viewport.

Below these elements are application-specific controls, organized in collapsible
sections.

This structure is very similar to that of the Boscoscopio app (laforesta/bosco/b).

## Modals

Modals have a consistent style, with slightly rounded corners and thin dark
green borders. Their background is white but they cause the rest of the page to
darken by about 50%.

They are used to display error message (with red text) and help information
where available (e.g., "?" links next to map navbar elements).

## Accessibility considerations

In its initial version, given the target staff, Abies has no special
accessibility features (high contrast, etc.), though of course enlarging fonts
in the browser is always an option for users.

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
These digests are regenerated lazily on read (see "JSON digest regeneration"
above).

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
  in each page's own doc under `docs/pages/`.

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
under `docs/pages/`.

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
    │   ├── bosco/                      # Bosco domain (Stage 2)
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

The Django project is rooted at laforesta/abies in the code repository, and is
served from a similar location once deployed (i.e.,
https://laforesta.it/abies/).

Abies is designed to be deployed on a single server (e.g., laforesta.it).

Official releases are numbered using git tag.

Abies is deployed via a Docker image that includes all dependencies, and it
mounts the host filesystem for:

- SQLite database file;
- Pre-processed JSON files;
- Backups.

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

# Relationship to existing "bosco" apps

The Forest Visualization domain of Abies subsumes "boscoscopio" and certain
other "bosco" apps. The "bosco" apps remain unchanged for now but will
eventually be taken offline and replaced entirely by Abies.

# Build order

v1 ships in two stages. Stage 1 is the production MVP; Stage 2 follows on a
separate track.

## Stage 1: Prelievi MVP

This is the first production release. It exercises every major subsystem of the
architecture (auth, shell, router, cache, conditional GETs, form injection,
optimistic locking, audit, CSV export, ETL) against the simplest UI, and it is
the domain where real users can begin entering data earliest.

Stage 1 includes:

- Shell page, header, client-side router, client cache, error modal, CSV export
  button.
- Authentication (both password and MS 365 OAuth) via django-allauth +
  django-axes.
- `base` app: common models (region, eclass, parcel, crew, tractor, species,
  optype, note, harvest_op, harvest_species, harvest_tractor), ETL scripts for
  initial data import, JSON digest generation, and shared templates/static/CSS.
- `prelievi` app: the Prelievi page (sortable-table + date range slider),
  add/edit/delete form, and the prelievi.json digest.
- `controllo` app: the audit page. Nearly free once django-simple-history is
  wired up, so include it in Stage 1 so staff can see their own activity as
  they start using the app.
- `impostazioni` app: all Settings sections (Personal settings,
  Crews/Tractors/Trees, App Users). App Users is required in Stage 1 so that
  early beta users can be onboarded.
- Bosco tab: placeholder content (e.g., a link to the existing Boscoscopio app)
  so the tab is not broken.

Until Stage 2 ships, the landing page after login is Prelievi, not Bosco.

## Stage 2: Bosco

A port of Boscoscopio into the Abies shell, with the additions described in the
Bosco page section below (aree di saggio, piante ad accrescimento indefinito,
bookmarkable URLs). Novelty is low relative to Stage 1 because much of the
code can be lifted from the existing bosco/b app, so Stage 2 can proceed
independently once the Stage 1 architecture is stable.

When Stage 2 ships, the landing page reverts to Bosco.

## Future stages

The remaining domains (Segheria, Biomassa, Fotovoltaico, Rifornimenti) are
separate post-v1 stages and are not scoped here.

# Detailed description

The v1 domain pages are documented per-page under `docs/pages/`.  Each doc
covers that page's URL, query parameters, visual layout, and JSON digests.
Read the page doc for the domain you are working on; cross-cutting concerns
(shell, routing, caching, forms, error handling, etc.) stay in this file.

- [Login](docs/pages/login.md) — centred card with username/password and
  an "Log in with Microsoft" button.  On success, lands on the default
  domain (Prelievi in Stage 1, Bosco from Stage 2).
- [Bosco](docs/pages/bosco.md) — map-centric view ported from Boscoscopio,
  with added overlays for aree di saggio and piante ad accrescimento
  indefinito, and bookmarkable URLs.  Stage 2.
- [Prelievi](docs/pages/prelievi.md) — harvest operations: filter bar,
  three collapsible sections (Produzione chart, Specie-per-particella
  chart, Interventi table), CRUD forms with optimistic locking.  Stage 1.
- [Controllo](docs/pages/controllo.md) — read-only audit trail from
  django-simple-history.  Stage 1.
- [Impostazioni](docs/pages/impostazioni.md) — password change, crew/
  tractor/species tables (writers+), and app-users management (admins
  only).  Stage 1.
