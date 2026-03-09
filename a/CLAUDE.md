# Bosco: Forestry company management application

This is a full-stack app with Python backend used to manage production
operations of a forestry company, including forest harvests, lumber production,
biomass energy production, etc.

It is intended to help with day to day operations and track production.

# Context

The target audience is relatively unsophisticated computer users. They currently
use Microsoft 365 Business and dislike its slowness. They use Word and Excel
daily, but only in a basic way: for example, they do not know how to set up
pivot tables or use Excel lookup functions. They tend to be instinctively
suspicious of and impatient with computers.

# Priorities

1. Correctness
2. Security
3. Speed (the UI should be snappy and not require much bandwidth)
4. Ease of maintenance

# Security

Data security (both privacy and integrity) is important in this use case because
we will be storing the core of the company's operations.

## Authorization

The app must be access-controlled on the server side. Authorization must support
MS 365 credentials and user/password pairs.

In the future we may need to support other OAuth identity providers.

We will use django-allauth to provide this functionality.

## Auditing

The app records all writes using django-simple-history and makes them available
for review through an audit table in the UI. The audit table is searchable and
sortable via the sortable-table Javascript widget (more on this below).

## Permission model

We have a three-role permission model, 'admin', 'writer' and 'reader'. Readers
are read-only, writers can also modify data, 'admins' can also create new users.

The audit log is viewable by everyone and intended to keep things transparent
and above-board.

If using username/password, users can change their password.

# General app structure

For now the app needs to provide two types of functionality, what we will call
data management and task management.

## Data management

Data management is about tracking production and consumption data: for example,
recording individual harvest operations ("on date D, team T harvested X tons of
logs of species S from forest parcel P") or resource usage ("on date D, tractor
M was refueled with Y liters of gasoline").

To see an example of harvest operations data, see bosco/data/mannesi.csv.

For each data set / type of data, we provide mechanisms for data entry and
analysis.

### Data entry

The data entry feature is a web-page equivalent of adding a new row of a
spreadsheet (which is how most data entry happens today, and why, e.g.,
mannesi.csv is so painful to work with).

More on the mannesi-specific UI below.

### Data visualization and analysis

There are three basic types of data visualization:

1. Viewing the data in tabular form, as one would in Excel. This is the
   same/standard for all data, basically displaying the underlying table.
2. Charts / graphs of one data set only (e.g., productivity by team or by parcel
   over time). This is custom code for each data set, but with clean, modular
   interfaces.
3. Joins of different data sets, for example, using part of the outputs of the
   harvest process as one of the inputs of the biomass plant. Again, custom
   code.

## Task management

Task management is a very basic project management tool, basically a task
tracker with a Gantt chart UI, also capable of reminders. We will discuss that
in a later phase of the project.

# UX

## Key principles and design choices

* Clean, minimal design, with an emphasis on readability: no drop shadows, no
  fancy borders, but also not too much whitespace.
* High information density without being overwhelming. Overall, an efficient use
  of the browser window real estate.
* Pastel, not overly saturated colors. Color theme (e.g., green, yellow) adopted
  from the existing bosco apps.
* Clearly visible breadcrumbs for navigation ("What part of the site am I in?").
* Simple, site-wide CSS with no redundancy.

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

# UI architecture

Server-rendered with progressive enhancement, using Django templates and forms,
JS added for real-time validation:

- Django renders the form HTML.
- JS validates client-side for snappy UX but the server never trusts the client.
- Django's form validation runs server-side on submit (the authoritative check).
- Traditional form POST, redirect on success.

Each data entry page (and other similar pages) has its own Django template with
custom form HTML, rather than using a generic auto-generated form. Likewise in
general for client-side JS and server-side validation, though it is a top
priority to not duplicate code and to move common code into libraries.

While the code is generally custom per data set, the overall structure is
modular. I.e., it is possible to add a new HTML edit page, validation
javascript, and server-side data conversion functionality easily, side-by-side
with others, using existing patterns.

All tabular data is displayed uniformly using the sortable-table component
(vendored from ~/src/jsutil/sortable-table).

Graphical displays use Chart.js.

Common CSS and JS from bosco apps (bosco/a, e.g., range-slider.*) are also
incorporated as dependencies.

JS dependencies (Leaflet, ChartJS, Sortable-Table) are minified and vendored,
served from Django's static/vendor. A Makefile target (make update-vendor)
re-copies from source when needed.

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


