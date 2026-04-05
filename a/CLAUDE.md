# Abies: integrated management of forestry company operations

Abies is a full-stack web app used to manage production operations of a forestry
company, including forest harvests, sawmill production, biomass energy
generation.

It is intended to help with day to day operations, track production, and monitor
forest health and productivity.

# User base

The target audience is office workers and field staff at a small (40-employee)
lumber company that also produces electricity from biomass. They currently use
Microsoft 365 Business and complain about its slowness. They are unsophisticated
computer users. They employ Word and Excel daily, but only use basic features
(e.g., they do not know how to set up Excel pivot tables). Simplicity and speed
are key requirements.

# Priorities

1. Correctness
2. Security
3. Speed (the UI is snappy and works well over 3G)
4. Low maintenance over time

# Functional overview

The app covers the following areas:

- Forest harvesting. A log of the daily activities of crews of lumberjacks,
  including how much wood of different species was harvested where and with what
  tractors. The tool also generates PDF record slips that the crews fill out and
  provide to the office staff for data entry.

- Forest health and planning. Geospatial tool that displays historical harvest
  data, supports planning and execution of forest surveys (by setting up
  sampling locations and allowing recording of geo-referenced annotations), and
  displays other geo-based information about the forest, including satellite
  imagery and health metrics.

- Sawmill operations. A log of daily sawmill operations, including volume and
  type of wood products, maintenance activities, failures or other incidents,
  etc., as well as monthly rollups of income and other parameters.

- Biomass plant operations. A log of daily biomass operations (amount of biomass
  consumed, energy produced, various operational parameters), as well as monthly
  rollups of energy produced, energy consumed, income broken down by various
  parameters, etc.

- Photovoltaic plant operations. Daily log of production, monthly log of
  verified production and revenue.

- Fuel operations. Log of operations on the company's diesel fuel tank (e.g.,
  who refueled what vehicle and how much fuel they used).

Each of these functional areas is handled in a distinct tab of the app and is
separate from the others. However, the outputs of some areas are inputs to
others (for example, wood from the forest flows into the sawmill and biomass
plant).

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

# Security

Data security (both privacy and integrity) is important because the app stored
the core of the company's operations.

## Authorization

The app is access-controlled on the server side. Authorization supports MS 365
credentials and user/password pairs via django-allauth.

In the future we may need to support other OAuth identity providers.

## Permission model

We have a three-role permission model, 'admin', 'writer' and 'reader'. Readers
are read-only, writers can also modify data, 'admins' can also create new users.

If using username/password pairs, users can change their own password.

## Auditing

The app records all writes using django-simple-history.

The audit log is readable and searchable by all users. (More on this below in
the UI section.)
 
# UI

## Principles and design choices

* Clean, minimal design, with an emphasis on readability: no drop shadows, no
  fancy borders.
* High information density without being overwhelming. Overall, efficient use
  of the browser window real estate.
* Muted, pastel color scheme with green and yellow accents, adopted
  from the existing bosco apps.
* Clearly breadcrumbs for navigation ("What part of the app am I in?").
* Simple, site-wide CSS with no redundancy. No inline styles in HTML.

## Architecture

The app is structured as a SPA-lite. After authentication, Django renders a
single shell page that persists for the duration of the session. All subsequent
navigation happens client-side without full page reloads.

### The shell

The shell is a single Django template containing:
- Top navigation with tabs for each functional area (harvest, sawmill, etc.)
- Breadcrumbs
- A date range slider (shared across all tabs, sticky)
- A content area where tab content is rendered

The shell is rendered once and never reloads during normal use.

### Routing

Clicking a tab updates the URL via `history.pushState()` and renders the
appropriate content. The back button works via `popstate`. All URLs are
bookmarkable: loading a bookmarked URL renders the shell and activates the
correct tab.

### Data display

Data for tables and charts is fetched from Django as JSON and rendered
client-side by sortable-table and Chart.js respectively.

Tabular data is fetched in one-year chunks; the date range slider controls which
years are displayed and defaults to the current year.

Most graphical and map data is pre-processed into compact static JSON
server-side, allowing the full time range to be served at once.

### Caching

Fetched data is cached client-side in memory, keyed by (tab, year):

1. On tab switch or date range change, render from cache immediately if
   available.
2. Fire a background conditional GET (using ETags). If the server returns
   304 Not Modified, done. If it returns new data, update the cache and
   re-render.

Tab switching and slider changes feel instant for previously-viewed data.

### Server-side precomputation

XXX

### Data entry

Data entry forms are Django-rendered HTML fetched as fragments into the
shell's content area:

1. User clicks "Add" or "Edit".
2. JS fetches the form HTML from Django (including CSRF token, field values,
   validation state).
3. The form replaces the current tab content within the shell.
4. Client-side JS validates for immediate feedback; server-side validation
   is authoritative.
5. On submit, JS intercepts the form POST via `fetch()`.
6. On success: return to the list view (still cached).
   On validation error: replace the form with Django's re-rendered HTML
   (including error messages).

Each form has custom HTML and custom client-side validation JS as needed, but
common patterns (percentage-group validation, form interception, error display)
are extracted into shared libraries.

### Dependencies

JS dependencies (Leaflet, Chart.js, sortable-table) are minified and vendored,
served from Django's static/vendor. A Makefile target (`make update-vendor`)
re-copies from source when needed.


# Storage

## Database

We use SQLite for storage, not a server DB like Postgres.

Arguments for SQLite:
  - Small user base (a small forestry company, < 10 concurrent users).
  - WAL mode handles concurrent reads + writes fine at this scale.
  - Zero moving parts: no database server process to maintain, crash, or misconfigure.
  - Backup = copy one file.
  - Django's ORM abstracts the DB, so migrating to Postgres later is a DATABASES setting change + migrate.

## Concurrency

We handle concurrent modification with row-level optimistic locking.

To detect stale edits, we store a version counter on harvest that increments on
every save. The edit includes the version as a hidden field; the server checks
the version on POST, and if the version has changed that indicates a write
conflict.

## Disconnected operation

There is no need for disconnected operation. We require that the server be
reachable. (However, we do explicitly consider low-bandwidth operation, so the
website is light and simple.)

## Backups

On the backend, we run a nightly backup of the database. We retain nightly
backups for a month, and once a month copy a backup offsite for permanent
retention.

## Data migration

Importing different kinds of data is done through roughly-single-use ETL scripts
that parse CSV files and normalize them.



# Detailed description

XXX

----- Stopped rewriting here

## Overview

After login the user arrives at a simple landing page that will eventually
display different functional areas, e.g., "Mannesi" (harvests), "Segheria", ecc.
as well as tools like task management ("gestione attività") and the existing
"Bosco" apps (see below), as well as a settings section ("Impostazioni"). The
first things to be built are the harvest and settings sections.

## Harvest section

The Harvest section is titled "Mannesi" and contains links for:
- "Aggiungi dati", which sends us to the data entry page.
- "Modifica dati" displays a list of all the entries and allows the user to
  select one to edit (then goes to the data entry page).
- "Visualizza dati", which for now just lists the raw data.

Both "Modifica" and "Visualizza" show filters according to the following criteria:
- Date (a double-ended slider with year granularity, see bosco/a/range-slider.*).
- Team
- Tractor
- Region
- Parcel within region
- Species (any matching species)
- Note value (see Data model for valid enum values)

The data is displayed in a sortable-table.

In "Modifica" mode, each row has an "edit" link that navigates to the edit form
for that row (prepopulated with existing values). On successful save (or
cancel), the server redirects back to the list with filters preserved via query
parameters. On a write conflict (version mismatch), the form re-renders with
the current data from the database and an error message.

## Other sections

We will add more tools over time.

## Settings section

We need to be able to define parameters such as valid mannesi work teams,
tractors, wood species, etc.

We want this functionality to be available to writers (to create or edit
parameters) and readers (to view their values) independently of creating new
accounts. As a result, this is a custom page, not a Django admin interface.

It is not possible to delete a parameter (tractor, team, species, etc.) if it
exists in the historical data, but it is possible to deactivate it so that it no
longer appears as a choice for new data entry or data modification.

If a parameter does not exist in historical data, it may be deleted.

## Mannesi data entry UX example

For mannesi we have entry fields for date, team name, type of operation, amount
of material harvested, percentage breakdown by species and by tractor, etc.

Some fields (e.g., team names) come from a finite predefined list that can be
configured via the Settings section (see above). These
fields should be implemented as pull-downs, such that tabbing over to them and
starting to type also selects the right row in the pull-down.

Some fields (species and tractors) can have multiple values that must add to
100% (e.g., harvest operation was 80% fir, 20% pine, using 50% tractor A and 50%
tractor B). For these fields, we list all values and provide next to each one
both a numerical entry text input and a "100%" button for the common case.

Fir: [box] [100%] Beech: [box] [100%] ...

Pressing a button sets the corresponding box to 100 and the others to 0.

We should also consider other ways to make the data entry faster, e.g.,
prepopulating the date field with the current date or the most recently entered
date.

Note, because of "modifica dati", data entry forms work both in create and edit
modes. In edit mode, the form is prepopulated with existing values. We detect
stale edits as described in storage below.

# Data model

## Common tables

- region: (id, name)
- parcel: corresponds to all fields in bosco/data/particelle.csv (including long
  textual columns), plus an entry for each region with parcel name 'X' (meaning
  all/any, for certain kind of cleanup work). Each parcel has a unique id.
- tractor: (id, manufacturer, model, year, active)
- species: (id, common name, latin name, active)
- team: (id, name, active)

The `active` flag supports soft-delete: deactivated entries no longer appear as
choices for new data entry but are preserved in historical data.

## Harvest-related

- operation_type: enum (tronchi, cippato, ramaglia, pertiche_puntelli, pertiche_tronchi)
- note: enum (none, PSR, fitosanitario, catastrofato)
- harvest: (id, date, operation_type, parcel_id, team_id, vdp (int, nullable),
  prot (int, nullable), quintals (float), note (enum), extra_note (text),
  version (int), created_at, modified_at)
- harvest_species: (harvest_id, species_id, percent) — PK is (harvest_id, species_id)
- harvest_tractor: (harvest_id, tractor_id, percent) — PK is (harvest_id, tractor_id)

Note that "Coppice" (ceduo) is both a stored property of parcels and a computed
property of individual harvest operations (true if castagno > 50%). It is
possible to extract some coppices from a parcel that is otherwise high forest
(fustaia).

## Other tables

We will enrich the data model over time to incorporate other kinds of production
and consumption data.

## Non-database data

Satellite imagery (GeoTIFF), GeoJSON geometries, and production timeseries JSON
remain as files served from disk. Only tabular operational data that needs
user-editable CRUD moves to the database.

# Relationship to "bosco" apps

The existing bosco/index.html landing page is subsumed into the landing page for
this app. The bosco apps are no longer world-accessible.

Existing bosco/ apps are served as Django static files behind Django's
authentication middleware. We fetch the static data (primary satellite images
and GeoJSON) via a custom view that points to the existing bosco/data directory.

In phase 1, mannesi.csv and particelle.csv move to the database. Bosco apps
that currently load these files will instead fetch the data from Django views.

In later phases, other operational CSVs (alberi, aree-di-saggio, calendario*,
registro*) will also move to the database once their schemas are revised.

Note that the CSV files themselves will not be deleted: they are still consumed
by the harvest plan document pipeline in pdg-2026/.

Note also that the data pipeline in bosco/util may need to change slightly to
publish processed data to an appropriate Docker bind mount or similar.

# Django project structure

The Django project is organized into apps by domain:

- `core` — shared models (region, parcel, species, team, tractor) and the
  settings ("Impostazioni") views that manage them. Also contains base
  templates, common CSS/JS, and the landing page.
- `harvest` — harvest models (harvest, harvest_species, harvest_tractor),
  data entry forms, views, and visualization.

As new datasets move to the database (alberi, aree-di-saggio, etc.), each
becomes its own app with its own models, forms, and views, following the same
patterns established by `harvest`.

Authentication is handled by django-allauth (no custom auth app). Audit history
is handled by django-simple-history (added to models via mixin, no custom app).

# Code location, deployment, and releases

The Django project is rooted at laforesta/a in the code repository, and will be
served from a similar location once deployed (i.e., https://laforesta.it/a/).

The app is designed to be deployed on a single server (e.g., laforesta.it).
The server already serves other sites: we use a Docker container for isolation.

We periodically publish numbered releases using git tag.

# Testing

Python tests use pytest-django with pytest-cov for coverage reporting
(--cov-report=term-missing), matching the existing pdg-2026 test setup. Tests
cover models, views, form validation, and ETL scripts.

Client-side JS tests use the existing `node tests.js` pattern (see bosco/b/).

# Development notes

* UI text is in Italian, but code (variable names, function names, etc) should
  all be **in** English. For examples, columns or variables might be parcel_id
  and region_id, not particella_id and compresa_id.
* Obsessive focus on simplicity and DRY. See also ~maxp/.claude/CLAUDE.md.


