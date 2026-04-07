# Abies: integrated management of forestry company operations

Abies is a full-stack web app used to manage production operations of a forestry
company, including forest harvests, sawmill production, biomass energy
generation.

It is intended to help with day to day operations, track production, and monitor
forest health and productivity.

# User base

The target audience is office workers and field staff at a small (40-employee)
lumber company that also produces electricity from biomass and solar. They
currently use Microsoft 365 Business and complain about its slowness. They are
unsophisticated computer users. They employ Word and Excel daily but only use
basic features (e.g., they do not know how to set up Excel pivot tables).
Simplicity and speed are key requirements.

# Priorities

1. Correctness
2. Security
3. Speed (the UI is snappy and works well over 3G)
4. Low maintenance over time

# Functional overview

The app covers the following functional areas, which we call _domains_:

- Forest visualization and planning. Geospatial tool that displays historical
  harvest data, supports planning and execution of forest surveys (setting up
  sampling locations; recording geo-referenced annotations; etc.), and displays
  other geo-located information about the forest, including satellite imagery
  and health metrics.

- Forest harvesting. A log of the daily activities of crews of lumberjacks,
  including how much wood of different species was harvested where and with what
  tractors. The tool also supports day-to-day operations by generating PDF
  record slips that the crews fill out and provide to the office staff for data
  entry.

- Sawmill. A log of daily sawmill operations, including volume and
  type of wood products, maintenance activities, failures or other incidents,
  etc., as well as monthly rollups of income and other parameters.

- Biomass plant. A log of daily biomass operations (amount of biomass
  consumed, energy produced, various operational parameters), as well as monthly
  rollups of energy produced, energy consumed, income broken down by various
  parameters, etc.

- Photovoltaic plant. Daily log of production, monthly log of
  verified production and revenue.

- Fuel facility. Log of operations on the company's diesel fuel tank: who
  refueled what vehicle, how much fuel they used, refueling of the fuel tank
  itself, etc.

Each of these domains is handled in a distinct tab of the app and is separate
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

# Security

Data security (both privacy and integrity) is important because the app stores
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

## Architecture

The app is structured as a SPA-lite. After authentication, Django renders a
single shell page that persists for the duration of the session. All subsequent
navigation happens client-side without full page reloads.

### Shell

The shell is a single Django template containing just:
- The header (more on this in "Detailed description" below)
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

Fetched data is cached client-side in memory, keyed by (data_id, year).

'data_id' identifies a particular dataset and display type, e.g., "table of
harvest operations" or "graph of monthly sawmill production".

1. On tab switch or date range change, the app renders from cache immediately if
   available.
2. The app also fires a background conditional GET (using ETags). If the server
   returns 304 Not Modified, no action is required. If it returns new data,
   the app updates the cache and re-renders.

Tab switching and slider changes feel instant for previously-viewed data.

### Server-side precomputation

A server-side job digests detailed data (always stored in relational form in
SQLite) into compact JSON form for specific types of display (specific
'data_id's).

The job runs from cron every 5 minutes (more frequent updates are unnecessary).
If a particular source table has not changed, it ignores it. Otherwise, it
regenerates the JSON representation.

Details about which data is pre-processed for which view are in the detailed
per-tab descriptions.

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

### CSS

No style is define in-line in HTML. All style is defined in CSS files. 

Most styles are maintained in a single common.css style file. Styles that are
only required on one page appear in a page-specific style file.

### Javascript dependencies

All JS dependencies (Leaflet, Chart.js, sortable-table, (Google) fonts, etc.)
are minified and vendored, served from Django's static/vendor. There are no
external dependencies at runtime. A Makefile target (`make update-vendor`)
re-copies from source when needed.

## Design patterns

## Objectives

The objectives of the visual design are:

- Readability: data is presented simply and clearly, with good use of screen
  real estate on both desktop and mobile.
- Predictability: consistent visual guidelines, no unexpected behavior.
- Discoverability: navigation is easy and fast.
- Restfulness: cognitive and visual load are low.

### Fonts and colors

DM Sans is used throughout.

The UI is strictly two-dimensional: there are no drop-shadows or other 3-D elements.

Page margins are moderate (15 px) on desktop and almost disappear (2 px) on mobile.

Buttons have subtly rounded corners. They are dark green and turn lighter when hovered over.

Horizontal rules outline the page header as well as collapsible elements  (more
on these in "Detailed description" below). They are relatively thin (4px) and
dark green.

### Tabular data

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
  - A table displays rows as far down as the bottom of the viewport. If there
    are more rows, the table has a scrollbar that is separate from the page
    scrollbar. (On mobile, there is enough lateral space to allow the user to
    also scroll the page, not just the table).
- Tables that allow row modification have a "pencil" icon on the right of each
  row.
- Tables that allow row deletion also have a "garbage can" icon on the right.
  (The type of table is specified in "Detailed description" below.)
- Tables that allow row addition have a "+" button below the bottom row, on the right.
- Tables have 1px medium-grey borders and column headers have light grey background.

#### Graphs and charts

All charts have y-axes that begin at 0.

All color maps range from yellow-green (for low values) to dark green (for high values).

Graphs occupy the full screen width and legends appear below the graph (on both
desktop anbd mobile).

# Storage

As mentioned previously, core relational data is stored in SQLite on the server.

SQLite is appropriate because:
  - The app has a small user base, and is read-mostly.
  - WAL mode handles concurrent reads and writes fine at this scale.
  - No moving parts: no database server process to maintain.
  - Simple backups (copy one file).
  - Django's ORM abstracts the DB, so migrating to Postgres later is
    straightforward.

In addition, summary statistics are computed periodically and stored as
compressed JSON files on the server filesystem for consumption by visualization
tools (charts and maps). (The files are compressed at creation time to decrease
load during serving.)

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
low-bandwidth (e.g., 3G) connections thanks to pervasive caching, pagination of
relational data, and compressed, pre-processed data files.

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

Every domanin tab includes a button (upper right) that allows its data to be
exported as a CSV file.

## Database model

Here we describe only common tables used across the app. Per-domain tables
appear in the detailed app description below.

- region: (id:int, name:string)
  - Denotes a forest region or "compresa".

- class: (id:int, name:string, coppice:bool,min_harvest_volume:int)
  - Represents a parcel type or economic class. It may be coppice or high
    forest (coppice=false).
  - Characterized by a minimum volume (m3/ha) before harvesting is permitted.

- harvest_plan: (id:int, description:text, interval:int)
  - Represents a harvest plan for a parcel.
  - 'interval' denotes harvest interval for coppice parcels.

- parcel: (id;int, name:string, region_id:int, class_id:int, area_ha:int,
  year:int, location_name:string, altitude_min_m:int, altitude_max_m:int,
  aspect:str, grade_pct:int, desc_veg:string, desc_geo:string,
  harvest_plan_id:int)
  - Represents a forest parcel. 'name' is typically an alphanumeric string like
    '11' or '2a'.
  - 'area_ha' is surface area in hectares.
  - 'year' is the average birth year of mature trees, used to compute average
    age.
  - altitudes are in meters.
  - 'desc_veg' and 'desc_geo' are string that describe the vegetative and
    geologic state of the parcel, respectively.

- tractor: (id:int, manufacturer:string, model:string, year:int, active:bool)
  - Represents a tractor. 'year' denotes date of manufacture.
  - Retired tractors have active=false.

- worker: (id:int, first_name:string, last_name:string, year:int, notes:string, active:bool)
  - Represents a worker, e.g., a lumberjack.
  - 'year' denotes birth year and is optional.
  - Workers who are no longer employed have active=false.

- crew: (id:int, name:string, active:bool)
  - Represents a team of workers, e.g., a group of lumberjacks.

- staffing: (crew_id:int, worker_id:int)
  - Maps workers to crews. A worker may appear in at most one crew at a time, so
    worker_ids in this table are distinct.
  - A crew need not have staffing for it to be used, e.g., in harvest tables.
    Staffing details are optional.

- species: (id:int, common_name:string, latin_name:string)
  - Represents a tree species.

# Internationalization

By default Abies supports Italian and US English, using Django's
LocaleMiddleware and Javascript's Intl APIs.

Users can set their preferred locale persistently in the settings page. See
below.

In addition to text strings, locale influences the representation of numbers
(thousands and decimal separators) and dates.

The _code_ itself (variable names, function names, etc.) is all in English. We
use terms like 'coppice' in the code instead of 'ceduo', and so on.

# Mobile

The app is usable on mobile in portrait mode (without needing to switch to landscape).

More on this is in the detailed description below.

# Django project structure

The Django project is organized into apps by domain.

- `core` includes common models (see above), the shell temnplate, and common
  CSS/JS.
- Then there are apps for each domain: 'forest', 'harvest', 'biomass', 'pv',
  'fuel', etc., that contain models and templates for each domain.

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

To speed up end-to-end tests, the test instance of Abies does not use Docker.

# Relationship to existing "bosco" apps

The Forest Visualization domain of Abies subsumes "boscoscopio" and certain
other "bosco" apps. The "bosco" apps remain unchanged for now but will
eventually be taken offline and replaces entirely by Abies.

# Detailed description

## Login page

The login page is mostly blank. In the center, inside a black-bordered square,
is the company logo and name, entry fields for username and password, and a "Log
in with Microsoft button".

Upon successful login, users land on the Forest Visualization page.

## Frame

All pages share the same header setup. The top of the page contains, from left to right:

- The logo and name of the company.
- The name of the currently active domain (Forest, Harvesting, Sawmill, Biomass, Photovoltaic,
  Fuel, Audit, Settings).
- A hamburger icon that opens to a menu with the list of the other domains.

This header is fixed. Content scrolls beneath it.

The same frame appears on mobile, though the company name is omitted.

## Settings page

The settings page contains several collapsible sections,  all collapsed by
default.

### Language

Every user (reader, writer, admin) sees a setting to configure app language.
This takes the form of a pull-down menu with available options (initially,
'English' and 'Italiano').

### Workers

Writers can create new workers.

Workers appear in a sortable-table that lists last name, first name, birth year
(optional), team name (optional), and a checkbox for whether active or not.

Above the table is a checkbox for "Show inactive". It is unchecked by default to
avoid clutter.

The worker sortable-table allows 


Readers and writers can create new users, XXX

We need to be able to define parameters such as valid mannesi work teams,
tractors, wood species, etc.

We want this functionality to be available to writers (to create or edit
parameters) and readers (to view their values) independently of creating new
accounts. As a result, this is a custom page, not a Django admin interface.

It is not possible to delete a parameter (tractor, team, species, etc.) if it
exists in the historical data, but it is possible to deactivate it so that it no
longer appears as a choice for new data entry or data modification.

If a parameter does not exist in historical data, it may be deleted.

## Audit page

## Forest visualization

TBD.

## Harvesting

### Data model

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

### UI

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


## Biomass power plant

TBD.

## Photovoltaic plant

TBD.

## Fuel facility

TBD.

